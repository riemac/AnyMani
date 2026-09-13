r"""真实 FlashSACEnvironment 的 CPU 桥接合同：终点属于旧回合，live 观测属于下一次控制。

时序依据为本地 IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py 的 step：
physics → terminated/truncated → reward → _reset_idx → compute(update_history=True)。
manager_based_env.py 的初始 reset 同样先改变场景，再返回新初态；RewardManager 输出 [N]。
observation_manager.py 中 update_history=False 不追加历史，因此此时 current 与 history 末帧可以不同。

假环境保留 N=4、16 JOINT、21 owner、History30；控制步长标记为 0.05 s，但不积分任何真实动力学。
动态数值是可辨认的无量纲测试标签，不能解释为可实现的关节、力或物体姿态。
原始数组原地重用，观测组直接引用这些数组；该别名压力比真实 ObservationManager 的 clone 更强。
只隔离包级注册，environment/observations/replay 及其模型类型执行真实源码；禁止 Isaac/CUDA 初始化。
运行：CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest <本文件>。
"""

from __future__ import annotations

import importlib
import importlib.abc
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

# 独立声明任务动态 ABI；不能从被测 compact 函数生成期望，否则漏字段可能被测试一起漏掉。
_LAYOUT = {
    "actor_jnt_current": ("policy", "jnt_current", (16, 5)),  # q/pi、u/pi、动作、自身/TIP 接触。
    "actor_owner_contact": ("policy", "owner_contact", (21, 1)),  # Actor 二元触觉，无量纲。
    "critic_jnt_state": ("critic", "jnt_state", (16, 4)),  # 实际含 q/pi、qdot(rad/s)、u/pi、动作。
    "critic_owner_contact": ("critic", "owner_contact", (21, 2)),  # F/(0.25 N) 与接触位。
    "critic_obj": ("critic", "obj", (1, 15)),  # 特权物体状态，测试仅使用可辨认标签。
    "critic_task": ("critic", "task", (1, 8)),  # 命令/进展/时域动态状态。
    "critic_reward_release": ("critic", "reward_release", (1,)),  # 无量纲奖励释放系数。
}
_PHYSICS_ROOTS = {"isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_rl", "isaacsim", "omni", "carb", "pxr"}


class _RejectPhysicsImports(importlib.abc.MetaPathFinder):
    r"""在执行模块体之前阻止物理运行时导入，避免包级注册触发模拟器。"""

    def find_spec(self, fullname, path=None, target=None):
        r"""纯 torch 模块交给标准加载器；运行时导入立即形成明确失败。"""
        if fullname.split(".")[0] in _PHYSICS_ROOTS:
            raise AssertionError(f"CPU environment contract attempted physics import: {fullname}")
        # 隐式返回 None，交回标准加载器，不加载任何被禁止的模块。


@pytest.fixture(scope="module")
def cpu_api():
    r"""隔离加载真实桥接/回放源码；原绝对模型导入指向同一份真实模型类型。"""
    root = Path(__file__).resolve().parents[3]  # 当前 checkout 的 distill 目录。
    prefix = "_flash_sac_environment_contract"  # 与其他合同测试隔离的模块身份。
    guard = _RejectPhysicsImports()  # 必须在任何项目源码 import 之前安装。
    previous_modules = set(sys.modules)  # 仅审计本合同新增的依赖。
    previous_threads = torch.get_num_threads()  # 退出时恢复其他 CPU 合同的线程配置。
    sys.meta_path.insert(0, guard)  # 全部测试执行期间维持禁止物理导入的边界。
    try:
        with pytest.MonkeyPatch.context() as devices, torch.device("cpu"):
            devices.setenv("CUDA_VISIBLE_DEVICES", "")  # 本进程不暴露任何 GPU。
            devices.setenv("OMP_NUM_THREADS", "1")  # 与独立测试命令保持相同 CPU 预算。
            torch.set_num_threads(1)  # torch 已导入，显式同步其 CPU 线程数。

            def reject_cuda(*args, **kwargs):
                r"""阻止意外创建 CUDA 上下文，不探测或访问正在训练的 GPU。"""
                raise AssertionError("CPU environment contract attempted CUDA initialization")

            devices.setattr(torch.cuda, "_lazy_init", reject_cuda)  # 设备分配失败于驱动初始化之前。
            # 空包只提供源码路径；数学、dataclass 和桥接函数没有替身。
            for suffix, directory in (
                ("", root),  # 隔离根包。
                (".models", root / "models"),  # 真实 observation 类型。
                (".models.backbones", root / "models" / "backbones"),  # 真实模型的纯 torch 依赖。
                (".flash_sac", root / "rl" / "flash_sac"),  # 被测环境、观测与回放。
            ):
                package = ModuleType(prefix + suffix)  # 不执行生产 __init__.py。
                package.__path__ = [str(directory)]  # 固定到本次 checkout。
                sys.modules[package.__name__] = package  # dataclass 需要可解析的模块身份。
            policy = importlib.import_module(prefix + ".models.palm_rotation_policy")  # 真实模型类型。
            with pytest.MonkeyPatch.context() as imports:
                for name in ("anymani", "anymani.distill", "anymani.distill.models"):
                    package = ModuleType(name)  # 临时解析 observations.py 的绝对类型导入。
                    package.__path__ = []  # 不沿绝对包扫描任务注册。
                    imports.setitem(sys.modules, name, package)  # 退出上下文后恢复已有模块。
                imports.setitem(sys.modules, "anymani.distill.models.palm_rotation_policy", policy)
                environment = importlib.import_module(prefix + ".flash_sac.environment")  # 真实桥接。
                replay = importlib.import_module(prefix + ".flash_sac.replay")  # 真实小容量回放。
            yield SimpleNamespace(environment=environment, replay=replay)  # 无 runtime/几何 provider 替换。
            loaded = set(sys.modules) - previous_modules  # 检查测试执行期的依赖闭包。
            assert not any(name.split(".")[0] in _PHYSICS_ROOTS for name in loaded), "CPU 合同导入了物理运行时"
            assert not torch.cuda.is_initialized(), "CPU 合同不得创建 CUDA 上下文"
    finally:
        sys.meta_path.remove(guard)  # 不影响本文件以外的测试导入。
        torch.set_num_threads(previous_threads)  # 恢复进入合同前的 CPU 配置。
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]  # 只清理本合同拥有的隔离模块。


def _frame(tag: int, dtype=torch.float64) -> dict[str, torch.Tensor]:
    r"""生成完整动态帧；环境/字段/通道均可辨，二元 Actor 接触始终合法。"""
    result = {}  # 每个 tensor 为 [4,*sample_shape]，默认 FP64 用于核对传输转型。
    for field, (name, (_, _, shape)) in enumerate(_LAYOUT.items()):  # 各字段采用不同偏移，能检测路由交换。
        channels = torch.arange(math.prod(shape), dtype=dtype).reshape(shape) / 512  # 二进制精确偏移。
        rows = torch.arange(4, dtype=dtype).reshape(4, *([1] * len(shape)))  # 环境轴广播。
        result[name] = 4 * tag + 2 * rows + field / 8 + channels  # 无量纲状态标签，不是真实物理样本。
    owners = torch.arange(21).reshape(1, 21, 1)  # owner 槽位。
    rows = torch.arange(4).reshape(4, 1, 1)  # 环境身份。
    result["actor_owner_contact"] = ((owners + rows + tag) % 2).to(dtype)  # 精确 0/1，不用连续标签冒充触觉。
    return result  # 全部数组只位于 CPU。


class _NoHistoryRead(Mapping):
    r"""终点组保留 history 键但禁止读取：False 计算时该历史尚未包含新的物理终点。"""

    def __init__(self, values):
        r"""保存完整 policy 组；current 和所有动态字段仍正常可读。"""
        self.values_by_name = values  # history 是已有的 [4,30,16,5]，不伪造终点历史。

    def __getitem__(self, key):
        r"""包括 get/items 在内的 Mapping 读取都须通过此处。"""
        # 禁止从 H_t 的末帧取 x_{t+1}；终点必须直接读取新的单帧项。
        assert key != "jnt_history", "pre-reset compact 必须读 current，不能读尚未追加终点的旧 history"
        return self.values_by_name[key]  # 返回原始动态数组的引用以施加别名压力。

    def __iter__(self):
        r"""仍暴露完整键集合，避免通过缺少 history 键掩盖错误接口。"""
        return iter(self.values_by_name)  # Mapping.items 会调用受保护的 __getitem__。

    def __len__(self):
        r"""返回完整 policy 组的字段数。"""
        return len(self.values_by_name)  # 包含不可用于终点的 history 字段。


class _ObservationManager:
    r"""只复现单帧/History30 的读写时序，不执行真实传感器或观测项。"""

    def __init__(self, raw):
        r"""绑定原始数组；记录 capture 时的真实旧历史，便于独立审计。"""
        self.raw = raw  # 同一环境的可变数据源。
        self.calls = []  # 每次 compute 的 update_history 参数。
        self.history_at_capture = []  # [4,30,16,5] 快照，仅供测试审计。

    def compute(self, *, update_history=False):
        r"""True 左移并追加当前帧；刚 reset 的行用新首帧填满全部 30 个位置。"""
        raw = self.raw  # 当前物理时刻，可能位于 reset 前或后。
        self.calls.append(update_history)  # 检验 capture 只调用 False。
        raw.events.append("observe:live" if update_history else "observe:terminal")  # 标记观测时刻。
        current = raw.dynamic["actor_jnt_current"]  # [4,16,5]，在物理推进后已经改变。
        if update_history:  # 只在 live 时推进历史时钟。
            raw.history.copy_(torch.cat((raw.history[:, 1:], current[:, None]), dim=1))  # oldest-to-latest。
            raw.history[raw.fresh] = current[raw.fresh, None]  # 部分 reset 只重建其自身历史。
            raw.fresh.zero_()  # 首帧补齐只执行一次。
        else:
            self.history_at_capture.append(raw.history.clone())  # False 不改 history，也不复制到 compact。
        groups = {"policy": {}, "critic": {}}  # 两组包含全部七个动态 ABI 字段。
        for name, (group, field, _) in _LAYOUT.items():  # 两组分别服务部署 Actor 与特权 Critic。
            groups[group][field] = raw.dynamic[name]  # 直接引用后续 reset 会原地覆盖的数组。
        groups["policy"]["jnt_history"] = raw.history  # 完整历史只属于 live 控制接口。
        if not update_history:  # False 仍暴露完整键，但不可从历史提取终点。
            groups["policy"] = _NoHistoryRead(groups["policy"])  # 终点读取 history 必须立即失败。
        return groups  # raw policy/critic named groups，不预先调用被测 compact 函数。


class _RawEnv:
    r"""按本地 ManagerBasedRLEnv 顺序推进假数据；reset 确实覆盖原始数组再返回 live 观测。"""

    def __init__(self, dtype):
        r"""N=4、dt=0.05 s；奖励与 flags 均使用会被下一步重用的一维原始 buffer。"""
        self.dtype = dtype  # 原始数据精度与传输的 FP32 精度分开控制。
        self.dynamic = _frame(0, dtype)  # 构造器态还不是 reset 后可采集的初态。
        self.history = torch.full((4, 30, 16, 5), float("nan"), dtype=dtype)  # reset 前历史无效。
        self.fresh = torch.ones(4, dtype=torch.bool)  # 下一次 live compute 应首帧补齐。
        self.reset_terminated = torch.zeros(4, dtype=torch.bool)  # 物理 drop 标志。
        self.reset_time_outs = torch.zeros(4, dtype=torch.bool)  # 有限时域 timeout 标志。
        self.reward = torch.zeros(4, dtype=torch.float32)  # 真实 RewardManager 的 [N] ABI。
        self.terminated_ids, self.truncated_ids = (), ()  # 下一步可同时设置两种结束事件。
        self.skip_capture_ids = ()  # 故障注入：仍执行 reset，只绕开桥接回调。
        self.split_reset = False  # 可将同一批 done 分两次回调，检验 capture 合并。
        self.next_physics = None  # 仅极值测试覆盖下一物理帧的输入数据。
        self.tick, self.reset_count = 0, 0  # vector-step 与 reset 回调计数，均无量纲。
        self.events = []  # 用时序证据区分 pre-reset capture 与 post-reset live。
        self.observation_manager = _ObservationManager(self)  # 完整动态组与 history 规则。

    def _reset_idx(self, env_ids):
        r"""原地改写指定行并使其历史重新播种；不清掉本步 terminated/truncated/reward。"""
        self.events.append("reset")  # 此时此前的物理终点即将被覆盖。
        self.reset_count += 1  # 每次 reset 得到可辨认的新初态。
        reset_frame = _frame(-self.reset_count, self.dtype)  # 新回合标签为负，与正数物理推进明确区分。
        for name, value in self.dynamic.items():  # 所有动态量共享相同的环境子集，不能只重置关节。
            value[env_ids] = reset_frame[name][env_ids]  # 保持底层 storage，只改变被重置的环境行。
        self.fresh[env_ids] = True  # 下一个 live observation 填满这些行的 History30。

    def reset(self):
        r"""先执行原始 reset 的状态变化，再 compute 并交付新初态。"""
        self._reset_idx(torch.arange(4))  # 通过实例回调，初始 reset 也经过真实桥接。
        return self.observation_manager.compute(update_history=True), {}  # 模拟 Gymnasium reset 返回。

    def step(self, actions):
        r"""复现 physics → flags → reward → reset → live；动作仅检查 [4,16] CPU ABI。"""
        assert actions.shape == (4, 16) and actions.device.type == "cpu", "动作必须为 CPU [N=4,16]"
        self.tick += 1  # 一次调用对应一个 0.05 s 控制步，未执行真实积分。
        # 全部动态字段属于同一物理时刻，原地写入后才计算结束条件。
        physics = self.next_physics if self.next_physics is not None else _frame(self.tick, self.dtype)
        for name, value in self.dynamic.items():  # 不能只在返回观测中制造“推进”的表象。
            value.copy_(physics[name])  # 第一步先改变原始物理数据，不能只改变返回值。
        self.events.append("physics")  # 此后才允许读取终止条件与奖励。
        self.reset_terminated.zero_()  # 像真实 TerminationManager.compute 一样原地重用 flags。
        self.reset_time_outs.zero_()  # reset 本身不清 flags，下一次 compute 才清。
        self.reset_terminated[list(self.terminated_ids)] = True  # [N] drop。
        self.reset_time_outs[list(self.truncated_ids)] = True  # [N] timeout，可与 drop 同时为真。
        self.events.append("flags")  # 奖励读取的是同一物理时刻的结束条件。
        self.reward.copy_(torch.tensor([-123.5, 0.0, 2.25, 1111.0]) + self.tick / 4)  # [N] 原始奖励。
        self.events.append("reward")  # 大于 clip100 的奖励仍应原样传递。
        ids = torch.where(self.reset_terminated | self.reset_time_outs)[0]  # 本步的完整 done 集合。
        # 此局部 mask 的长度是 done 数量，不能与全 N 环境轴混淆。
        bypass = torch.tensor([int(row) in self.skip_capture_ids for row in ids], dtype=torch.bool)
        capture_ids = ids[~bypass]  # 故障注入仅影响回调是否可见，物理 reset 仍执行。
        batches = (capture_ids[::2], capture_ids[1::2]) if self.split_reset else (capture_ids,)  # 同步步内的互斥子集。
        for batch in batches:  # 同步步内各批的时刻相同，但先前批次可能已经 reset。
            if batch.numel():  # 空集合不产生额外 capture。
                self._reset_idx(batch)  # SAC 在此读取 pre-reset current，然后调用原始 reset。
        if bool(bypass.any()):  # 模拟 hook 失联，仍完成真实的数据覆盖。
            _RawEnv._reset_idx(self, ids[bypass])  # 模拟上游漏调实例 hook 的错误路径。
        groups = self.observation_manager.compute(update_history=True)  # 返回 reset 后的真实 live 数据。
        extras = {"policy_dt_s": 0.05, "tick": self.tick}  # 仅传输诊断，不驱动采样语义。
        return groups, self.reward, self.reset_terminated, self.reset_time_outs, extras


class _Transport:
    r"""CPU named transport；与生产 _float 一样执行 FP32 转换和对称 clip100。"""

    def __init__(self, raw):
        r"""仅实现真实桥接消费的原有接口，不包含训练循环或 runtime。"""
        self.unwrapped, self.num_envs = raw, 4  # 桥接回调绑定同一个 raw 实例。
        self.done_override = None  # 故障注入：模拟 transport 与 raw flags 不一致。
        self.column_reward = False  # 正常链路为 [N]，另测桥接对 [N,1] 的展平承诺。
        self.close_calls, self.callback_at_close = 0, None  # 记录 close 是否已恢复原始回调。

    def _float(self, value, *, clip=True):
        r"""对应 palm_rotation_vecenv.py 的 _float；只裁剪观测，不裁剪原始奖励。"""
        result = value.to(device="cpu", dtype=torch.float32)  # raw 任意浮点精度 → CPU FP32。
        return result.clamp(-100.0, 100.0) if clip else result  # transport 宽松数值界，不是物理投影。

    def _transport(self, groups):
        r"""按独立 ABI 映射完整动态组；额外历史与几何不能进入 compact。"""
        # 只重命名并转型；保持环境、关节、owner 和特征轴的原顺序。
        result = {name: self._float(groups[group][field]) for name, (group, field, _) in _LAYOUT.items()}
        result["actor_jnt_history"] = self._float(groups["policy"]["jnt_history"])  # live History30。
        result["geometry_tokens"] = torch.zeros(4, 21, 128)  # 静态/几何排除哨兵，不执行 N040。
        return {"obs": result}  # 真实 rl_games transport 的外层封装。

    def reset(self):
        r"""原始 reset 完成后才转换观测，防止假环境掩盖 auto-reset 污染。"""
        groups, _ = self.unwrapped.reset()  # 必须先改变原始数组。
        return self._transport(groups)  # 此时才读取 reset 后 live 值。

    def step(self, actions):
        r"""保留 raw 的奖励和两类结束事实，向桥接提供合并 done。"""
        groups, reward, terminated, truncated, extras = self.unwrapped.step(actions)  # 原始完整时序。
        done = terminated | truncated if self.done_override is None else self.done_override  # [N]。
        reward = reward[:, None] if self.column_reward else reward  # 正常输入严格为一维。
        return self._transport(groups), reward, done, extras  # live 永远在 reset 之后。

    def close(self):
        r"""记录生命周期调用及当时的 raw 回调身份。"""
        self.close_calls += 1  # 每个独立桥接应关闭一次 transport。
        self.callback_at_close = self.unwrapped._reset_idx  # 验证恢复发生在 transport.close 之前。


@pytest.fixture
def bridge(cpu_api, request):
    r"""每个测试拥有独立 raw/transport 和真实桥接；精度可由 indirect 参数指定。"""
    raw = _RawEnv(getattr(request, "param", torch.float64))  # 默认 FP64 raw，所有输出应为 FP32。
    original = raw._reset_idx  # 保存安装 hook 之前的实例/函数绑定关系。
    transport = _Transport(raw)  # CPU fake，只承担环境与张量搬运。
    env = cpu_api.environment.FlashSACEnvironment(transport)  # 真实实现，不覆盖其任何方法。
    yield SimpleNamespace(env=env, raw=raw, transport=transport, original=original, api=cpu_api)
    if not transport.close_calls:
        env.close()  # 即使断言或预期异常发生，也释放此测试拥有的 hook。


def _assert_compact(actual, expected):
    r"""逐字段核对紧凑七字段、[N,*ABI]、CPU FP32 与数值；容差为零。"""
    assert isinstance(actual, dict), "compact 必须为动态字段 dict"  # 可直接交付 CompactReplay.append。
    # 持久状态只由七个动态字段组成，History30 与几何 Z 由下游重建。
    assert set(actual) == set(_LAYOUT), f"compact 字段错误：实际={tuple(actual)}，期望={tuple(_LAYOUT)}"
    for name, (_, _, shape) in _LAYOUT.items():  # 全字段检查避免只验证 Actor 而漏掉特权终点。
        reference = expected[name].detach().float().clamp(-100, 100)  # 独立 transport 数值合同。
        # 禁止环境轴广播；即使数值可广播匹配，也不是合法的同步转移。
        assert actual[name].shape == (4, *shape), f"{name}: 实际 {actual[name].shape}，期望 {(4, *shape)}"
        # current/terminal 两端采用相同精度，防止类型提升掩盖终点漏转型。
        assert actual[name].device.type == "cpu" and actual[name].dtype == torch.float32, f"{name}: 必须 CPU FP32"
        assert not actual[name].requires_grad, f"{name}: compact 不得保留外部计算图"  # 回放只接收采集事实。
        # 同一确定性标签经过相同转型应逐元素精确相等，不需要数值近似容差。
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0, msg=f"{name}: 环境/字段/通道或终点时刻错误")


def test_initial_reset_is_not_a_transition(bridge):
    r"""初始 reset 不捕获构造器态，live 历史从 reset 后首帧开始，compact 不含历史/几何。"""
    env, raw = bridge.env, bridge.raw  # 两个对象共享同一 raw buffer。
    observation = env.reset()  # 此处没有动作、奖励或 replay append。
    # 初始化不消耗动作预算，不能形成可用于 Bellman target 的转移。
    assert raw.tick == 0 and raw.events == ["reset", "observe:live"], "初始 reset 被错误计为物理转移"
    assert raw.observation_manager.calls == [True], "初始 reset 不应读取 pre-reset 观测"  # 只播种新回合历史。
    assert env._captured_mask is None or not env._captured_mask.any(), "构造器态不应生成有效终点"  # 此时没有物理转移。
    _assert_compact(env.compact(observation), _frame(-1))  # 新初态必须来自已经改变的原始数据。
    history = observation["actor_jnt_current"][:, None].repeat(1, 30, 1, 1)  # [4,30,16,5] 首帧填满。
    torch.testing.assert_close(observation["actor_jnt_history"], history, rtol=0, atol=0)  # H_0=[x_0,...,x_0]。


@pytest.mark.parametrize(
    "terminated,truncated",
    [((), ()), ((0,), ()), ((), (1,)), ((2,), (2,)), ((0, 2), (1, 2))],
    ids=["ordinary", "drop", "timeout", "both-flags", "mixed-partial-reset"],
)
def test_step_separates_live_reset_from_physics_endpoint(bridge, terminated, truncated):
    r"""普通 next 是物理推进态；done next 必须仍为该物理终点，不能成为 reset 后的新初态。"""
    env, raw = bridge.env, bridge.raw  # N=4，不同结束组合中均保留至少一个连续副本。
    initial = env.reset()  # current_0 是合法初态，不是构造器态。
    raw.terminated_ids, raw.truncated_ids = terminated, truncated  # 两种结束事件独立配置。
    raw.events.clear()  # 单独审计本步的 post-physics/pre-reset 时序。
    result = env.step(torch.zeros(4, 16))  # 原始奖励在 reset 前确定，观测在 reset 后返回。
    term = torch.tensor([row in terminated for row in range(4)])  # 独立 terminated 真值 [N]。
    trunc = torch.tensor([row in truncated for row in range(4)])  # 独立 truncated 真值 [N]。
    done = term | trunc  # 两种标志同时为真时不能互相覆盖。
    _assert_compact(result.next_compact, _frame(1))  # 全部行都应来自同一个物理推进时刻。
    _assert_compact(env.compact(result.observation), raw.dynamic)  # live 必须实际读取 reset 后的原始数组。
    torch.testing.assert_close(result.terminated, term)  # drop 保留 bool [N]。
    torch.testing.assert_close(result.truncated, trunc)  # timeout 保留 bool [N]。
    # 每只手一个原始标量奖励；它不随观测的 clip100 被截断或归一化。
    assert result.reward.shape == (4,) and result.reward.dtype == torch.float32, "真实 SACStep.reward 必须为 FP32 [N]"
    # 输入同时包含负值、正值和两端超过观测裁剪界的数值。
    torch.testing.assert_close(result.reward, torch.tensor([-123.25, 0.25, 2.5, 1111.25]), rtol=0, atol=0)
    assert result.extras == {"policy_dt_s": 0.05, "tick": 1}, "诊断 extras 不应丢失"  # 保留秒单位的控制步长。
    # 先交付旧回合终点，再执行 reset，最后才生成下一次控制使用的 live。
    expected_events = ["physics", "flags", "reward"] + (["observe:terminal", "reset"] if done.any() else [])
    # 普通步没有 reset，但也必须先完成 physics/flags/reward 再读取 live。
    assert raw.events == expected_events + ["observe:live"], f"桥接时序错误：{raw.events}"
    current, next_frame = initial["actor_jnt_current"], result.next_compact["actor_jnt_current"]  # [4,16,5]。
    # 每个副本必须确实推进；批内只让一个副本变化不足以验证普通 next。
    assert bool((current != next_frame).any(dim=-1).any(dim=-1).all()), "物理 next 不得误用步前 current"
    expected_history = torch.cat((initial["actor_jnt_history"][:, 1:], next_frame[:, None]), dim=1)  # 连续行推进一帧。
    expected_history[done] = result.observation["actor_jnt_current"][done, None]  # 只有 done 行重建初态历史。
    # capture 的 False 计算不能使普通行多左移一次，done 行则只含新回合初态。
    torch.testing.assert_close(result.observation["actor_jnt_history"], expected_history, rtol=0, atol=0)
    if bool(done.any()):  # 有结束事件才存在两种不同物理时刻的“下一状态”。
        live_end = result.observation["actor_jnt_current"][done]  # reset 后下一回合初态。
        assert bool((next_frame[done] != live_end).all()), "终点被 reset 初态污染"  # x_{t+1}^{-} 与新回合 x_0 不同。
        torch.testing.assert_close(
            raw.observation_manager.history_at_capture[0], initial["actor_jnt_history"].to(raw.dtype)
        )


def test_disjoint_reset_callbacks_preserve_previously_captured_rows(bridge):
    r"""第二次 compute 会看见第一批的 reset 初态，合并 capture 时只能更新本批 env_ids。"""
    env, raw = bridge.env, bridge.raw  # 同步 step 内进行两批互不相交的部分 reset。
    env.reset()  # 四行均从正常初态开始。
    raw.terminated_ids, raw.truncated_ids, raw.split_reset = (0, 2), (1, 2), True  # [0,2] 后 [1]。
    result = env.step(torch.zeros(4, 16))  # 第二次 capture 不得覆盖第一批已经保存的终点。
    _assert_compact(result.next_compact, _frame(1))  # 行 0/2 仍是正数物理终点，行 3 连续。
    assert raw.observation_manager.calls == [True, False, False, True], "分批 reset 应各读取一次无历史更新的观测"


@pytest.mark.parametrize("bridge", [torch.float32, torch.float64], indirect=True, ids=["raw-fp32", "raw-fp64"])
def test_terminal_clip_and_dtype_match_ordinary_transport(bridge):
    r"""终点和连续行采用相同 FP32/clip100；原始奖励不使用观测截断。"""
    env, raw, transport = bridge.env, bridge.raw, bridge.transport  # 原始精度由参数确定。
    env.reset()  # 保证已有有效 history，False 计算不能从空历史补齐。
    extreme = _frame(1, raw.dtype)  # 接触仍保持精确二元。
    for name, value in extreme.items():  # Actor/Critic 的每个实数字段都经过边界压力。
        if name != "actor_owner_contact":  # 二元触觉维持合法，避免与回放 bool 压缩混淆。
            # 极值检查正负截断，内部非二进制小数检查 FP64 → FP32 舍入。
            pattern = torch.tensor([-150.0, -99.875, 0.123456789, 99.875, 150.0], dtype=raw.dtype)
            value.copy_(
                pattern.repeat(math.ceil(value.numel() / 5))[: value.numel()].reshape_as(value)
            )  # 正/负界与内部值。
    raw.next_physics, raw.terminated_ids = extreme, (0, 2)  # 同一极值分布同时测试 terminal/ordinary 两条路径。
    result = env.step(torch.zeros(4, 16))  # reset 后 live 的 done 行不再携带这些极值。
    _assert_compact(result.next_compact, extreme)  # 独立参考：先转 FP32，再对称 clip100。
    for name in _LAYOUT:  # 不以只检查 actor_current 代替完整传输合同。
        # 与普通 transport 的数值路径逐位一致，包括 Critic 物体和任务动态。
        torch.testing.assert_close(result.next_compact[name], transport._float(extreme[name]), rtol=0, atol=0)
    assert result.reward.min() < -100 and result.reward.max() > 100, "观测裁剪错误地作用到了 reward"


def test_capture_and_delivered_next_survive_inplace_raw_resets(bridge):
    r"""reset 覆盖同一 storage；capture 及已交付 next 都必须拥有独立数据。"""
    env, raw = bridge.env, bridge.raw  # 默认 FP64 raw 仍保留真实原地覆盖行为。
    env.reset()  # 保存的终点必须属于随后的动作转移。
    pointers = {name: value.data_ptr() for name, value in raw.dynamic.items()}  # 审计假环境确实重用数组。
    raw.terminated_ids = (0, 2)  # 两个不连续环境索引，不能靠连续 slice 偶然通过。
    result = env.step(torch.zeros(4, 16))  # capture 在本次 reset 内已经历一次原始数组覆盖。
    saved = {name: value.clone() for name, value in result.next_compact.items()}  # 交付时刻真值。
    buffers = dict(env._captured)  # 持有实际捕获 tensor；reset 是否清空内部容器不影响其数据所有权合同。
    captured = {name: value[[0, 2]].clone() for name, value in buffers.items()}  # 只读 mask 有效行。
    env.reset()  # 全部原始数组再次原地变成新的初态，不产生新 transition。
    assert pointers == {name: value.data_ptr() for name, value in raw.dynamic.items()}, "fake reset 未施加原地覆盖压力"
    for name in _LAYOUT:
        torch.testing.assert_close(buffers[name][[0, 2]], captured[name], rtol=0, atol=0)  # 只审计有效终点行。
    _assert_compact(result.next_compact, saved)  # 下游持有的 terminal next 不得被 raw reset 改写。
    raw.terminated_ids = (0, 2)  # 再次捕获相同两行，故意重用 capture storage。
    env.step(torch.zeros(4, 16))  # 新终点与上一步 saved 明确不同。
    _assert_compact(result.next_compact, saved)  # 已交付 SACStep 也不能别名可复用 capture 数组。


def test_next_step_clears_old_capture_and_keeps_returned_flags(bridge):
    r"""新步没有 reset 时不复用旧终点，原地更新 raw flags 不得改写上一 SACStep。"""
    env, raw = bridge.env, bridge.raw  # 同一个实例连续执行两步。
    env.reset()  # 启用正常 capture。
    raw.terminated_ids, raw.truncated_ids = (0, 2), (1, 2)  # 同时覆盖 drop、timeout 与 both。
    first = env.step(torch.zeros(4, 16))  # 此时 capture mask 的前三行为真。
    raw.terminated_ids, raw.truncated_ids = (), ()  # 第二步所有行都连续推进。
    second = env.step(torch.zeros(4, 16))  # 不触发终点 compute。
    _assert_compact(second.next_compact, _frame(2))  # 旧捕获值不能覆盖新的普通 next。
    # 有效位只证明当前同步步有终点，不是“该环境曾经结束过”的永久标志。
    assert env._captured_mask is None or not env._captured_mask.any(), "新 step 未清除旧 capture 有效位"
    assert not second.terminated.any() and not second.truncated.any(), "连续步不应携带旧 done"  # 下一步不会再次 reset。
    assert first.terminated.tolist() == [True, False, True, False], "上一转移 terminated 别名了 raw buffer"
    assert first.truncated.tolist() == [False, True, True, False], "上一转移 truncated 别名了 raw buffer"
    assert raw.observation_manager.calls == [True, False, True, True], "连续步不应读取 pre-reset 观测"


@pytest.mark.parametrize("case", ["missing-all", "missing-one-row", "stale-same-row"])
def test_missing_terminal_capture_is_rejected_including_stale_rows(bridge, case):
    r"""done 后确实 reset，但漏调 hook 时必须报错；旧步 capture 不能当作本步证书。"""
    env, raw = bridge.env, bridge.raw  # 缺失整批与部分缺失均使用真实 step 校验逻辑。
    env.reset()  # 初始 reset 不提供任何可冒用的 terminal capture。
    if case == "stale-same-row":
        raw.terminated_ids = (0,)  # 先让同一行生成一份合法但将过期的 capture。
        env.step(torch.zeros(4, 16))  # 下一步必须清掉其有效位。
    raw.terminated_ids = (0, 1) if case == "missing-one-row" else (0,)  # 原始 done 事实。
    raw.skip_capture_ids = (1,) if case == "missing-one-row" else (0,)  # reset 仍执行，仅 hook 缺失。
    with pytest.raises(RuntimeError, match="missing.*pre-reset terminal observation"):
        env.step(torch.zeros(4, 16))  # 不能静默把 reset 初态或旧终点写入 replay。


@pytest.mark.parametrize(
    "terminated,truncated,reported",
    [((0,), (), ()), ((), (1,), ()), ((), (), (2,)), ((0,), (), (1,))],
    ids=["lost-drop", "lost-timeout", "spurious-done", "wrong-row"],
)
def test_transport_done_must_match_raw_flags(bridge, terminated, truncated, reported):
    r"""done 必须逐行等于 terminated OR truncated；数量相等而环境错位也必须拒绝。"""
    env, raw = bridge.env, bridge.raw  # 故障发生于 transport，而非任务原始 flags。
    env.reset()  # 对真实 done 仍正常产生 capture，以独立检验 flags 校验。
    raw.terminated_ids, raw.truncated_ids = terminated, truncated  # 原始物理事实。
    bridge.transport.done_override = torch.tensor([row in reported for row in range(4)])  # 不一致的 [N] done。
    with pytest.raises(RuntimeError, match="done disagrees with physical terminated/truncated"):
        env.step(torch.zeros(4, 16))  # 不得仅比较 done 数量或依靠隐式广播。


def test_column_transport_reward_is_exposed_as_one_dimensional_reward(bridge):
    r"""真实常规链路为 [N]；即使 transport 显式交付 [N,1]，SACStep.reward 仍为 [N]。"""
    bridge.env.reset()  # 构造合法 current。
    bridge.transport.column_reward = True  # 仅此用例验证桥接声明的 reshape 行为。
    result = bridge.env.step(torch.zeros(4, 16))  # 无结束事件，奖励来源仍是 post-physics。
    assert result.reward.shape == (4,), f"SACStep.reward 实际 {result.reward.shape}，期望 [N=4]"
    torch.testing.assert_close(result.reward, torch.tensor([-123.25, 0.25, 2.5, 1111.25]), rtol=0, atol=0)


def test_close_restores_original_callback_before_transport_close(bridge):
    r"""close 解除实例 hook，再关闭 transport；关闭后原始 reset 仍按其原语义工作。"""
    env, raw, transport = bridge.env, bridge.raw, bridge.transport  # 回调身份由绑定的实例和函数共同定义。
    env.reset()  # 初始化后产生真实非空 capture，以检查 close 清理。
    raw.terminated_ids = (0,)  # 至少捕获一行。
    env.step(torch.zeros(4, 16))  # _captured 与 mask 均已分配。
    env.close()  # 必须先恢复 raw 回调，后调用 transport.close。
    # Python 每次读取类方法均可生成新的 bound method 对象；== 核对同一 __self__ 与 __func__。
    assert raw._reset_idx == bridge.original, "close 没有恢复原始实例回调"
    assert transport.close_calls == 1 and transport.callback_at_close == bridge.original, (
        "transport.close 时回调尚未恢复"
    )
    assert not env._captured and env._captured_mask is None and not env._capture_enabled, "close 没有释放 capture"
    calls = list(raw.observation_manager.calls)  # 原始 reset 自身不计算终点观察。
    raw._reset_idx(torch.tensor([3]))  # 关闭后直接调用原回调，不再进入桥接。
    assert raw.observation_manager.calls == calls, "恢复后的原始回调仍触发 capture"


def test_bridge_to_compact_replay_rebuilds_terminal_and_reset_history(bridge):
    r"""三步小集成：真实桥接 → compact append → History30；结束不 bootstrap，新回合不串历史。"""
    env, raw = bridge.env, bridge.raw  # 保留真实 16/21/30 ABI，只缩小采样容量。
    replay = bridge.api.replay.CompactReplay(
        capacity_transitions=16,  # 只写三次 vector-step，不发生环覆盖。
        num_envs=4,  # 两资产、各两副本。
        asset_ids_by_env=torch.tensor([9, 2, 9, 2]),  # 两资产交错副本。
        history_steps=30,  # 保持真实 History30 ABI。
        n_step=1,  # 聚焦环境边界，不重复 n-step 算法合同。
        gamma=0.99,  # 有限时域结束的折扣系数应严格为零。
        device="cpu",  # 非锁页小池，不涉及 GPU。
    )
    live = env.reset()  # 初始状态只成为第一条 transition 的 current。
    first_frame = live["actor_jnt_current"].clone()  # [4,16,5]，独立历史参考。
    reset_frame = None  # 在第二次 step 的 mixed done 之后保存新回合初态。
    for tick in range(1, 4):
        raw.terminated_ids = (0, 2) if tick == 2 else ()  # 第二步 drop / both。
        raw.truncated_ids = (1, 2) if tick == 2 else ()  # 第二步 timeout / both。
        actions = torch.zeros(4, 16)  # [N,16] 未归一化的测试动作。
        result = env.step(actions)  # 返回 [N] reward 和显式 pre-reset next。
        replay.append(
            env.compact(live), actions, result.reward, result.terminated, result.truncated, result.next_compact
        )
        live = result.observation  # 下一条 current 来自 live，绝不能用 terminal compact 代替。
        if tick == 2:
            reset_frame = live["actor_jnt_current"].clone()  # 三行新回合初态与第四行连续物理态。
    assert replay.total_transitions == 12, "初始 reset 不得 append 为额外 transition"
    envs = torch.arange(4)  # 固定选择全部副本，不引入抽样随机性。
    terminal = replay.gather(envs, torch.ones(4, dtype=torch.long))  # sequence=1，对应第二次动作。
    expected_history = torch.cat(
        (
            first_frame[:, None].repeat(1, 28, 1, 1),  # 初态补齐 28 帧。
            _frame(1)["actor_jnt_current"][:, None].float(),  # 第一动作的物理终点。
            _frame(2)["actor_jnt_current"][:, None].float(),  # 第二动作的 pre-reset 终点。
        ),
        dim=1,  # [初态×28, 物理帧1, 物理终点2]，总计 History30。
    )
    # 终点历史最后一帧属于旧回合，不能从下一次 append 的 current 读取。
    torch.testing.assert_close(terminal["next_obs"]["actor_jnt_history"], expected_history, rtol=0, atol=0)
    assert terminal["terminated"].tolist() == [True, False, True, False], "replay 丢失 drop/both 标志"
    assert terminal["truncated"].tolist() == [False, True, True, False], "replay 丢失 timeout/both 标志"
    torch.testing.assert_close(terminal["discounts"], torch.tensor([0.0, 0.0, 0.0, 0.99]))  # 有限时域结束关 bootstrap。
    restarted = replay.gather(envs, torch.full((4,), 2, dtype=torch.long))  # 新回合第一条 current。
    assert reset_frame is not None, "参考轨迹没有保存 mixed reset 后的新初态"
    expected_history[:3] = reset_frame[:3, None]  # 只有 done 三行用新初态填满 History30。
    # 未结束行继续原历史；三种真实结束组合均不允许历史跨回合。
    torch.testing.assert_close(restarted["obs"]["actor_jnt_history"], expected_history, rtol=0, atol=0)
