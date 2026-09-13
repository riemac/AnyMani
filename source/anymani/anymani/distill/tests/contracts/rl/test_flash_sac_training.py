r"""FlashSACTrainingRun 的小预算 CPU 集成合同：采样时钟、时序证据与完整恢复。

真实 FlashSACLearner、CompactReplay、StaticHandBank、FirstThirtySecondsMetrics 和
checkpoint I/O 共同执行；替身仅提供可逐步复算的具名环境事实与冻结解析几何。
数值锚点：A=2、N=4、C=64、预热12、B=4、History30、n=3、容量136；MLP/Q 隐宽16、
残差块1、价值原子11。U(C)=floor(max(C-12,0)/4)，Actor 在第1/3/5/...次Q更新时优化。

合成环境的明确时间单位是10秒/步，因此3步完成首30秒；该时钟写入测试身份，不代表
生产20Hz物理仿真。两种手型各有两个交错副本，四行分别产生晚timeout、早drop、早timeout、
晚axis事件。reset首帧重复30次，普通步左移并追加；终点快照在auto-reset前冻结。
本文件只证明 CPU 算法编排、文件与统计合同，不作物理行为或策略能力判断。
"""

from __future__ import annotations

import importlib
import importlib.abc
import math
import random
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_flash_sac_identity import _identity_case
from test_flash_sac_learner import _assert_tree_equal, _observation
from test_flash_sac_learner import api as api
from test_flash_sac_learner import restore_cpu_rng as restore_cpu_rng
from test_flash_sac_networks import _FrozenQGeometry

# 导入拦截在模块体执行之前生效；CPU 模型可导入 torch，但不得加载任何物理运行时。
_PHYSICS = {"isaaclab", "isaacsim", "omni", "carb", "pxr"}  # 禁止的顶层运行时包。
_LABELS = (1, 0, 1, 0)  # 每资产两个交错副本，不能假设资产在环境轴连续排列。
_PERIODS = (5, 2, 2, 4)  # 每副本一个回合的合成策略步数。
_TURN_RATES = (0.25, -0.125, 0.5, 0.75)  # 圈/合成步，有符号且按副本不同。


class _RejectPhysicsImports(importlib.abc.MetaPathFinder):
    r"""拒绝 CPU 集成路径意外触发 Isaac/Kit 等模块，而非启动后再检查。"""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        r"""其余导入仍交给标准查找器，真实模型、优化器和文件实现不被替换。"""
        if fullname.split(".")[0] in _PHYSICS:  # 只按精确顶层包名阻止物理依赖。
            raise AssertionError(f"CPU training contract attempted physics import: {fullname}")


@pytest.fixture(scope="module", autouse=True)
def cpu_only() -> Iterator[None]:
    r"""单线程、隐藏CUDA并禁止初始化；生命周期覆盖依赖fixture的真实源码导入。"""
    before = set(sys.modules)  # 只核对本模块新加载的物理依赖。
    guard = _RejectPhysicsImports()  # MetaPath屏障先于任何生产源码import。
    threads = torch.get_num_threads()  # 测试结束恢复调用进程原有CPU线程数。
    sys.meta_path.insert(0, guard)  # 不让运行时模块执行初始化代码。
    try:
        with pytest.MonkeyPatch.context() as patches:
            patches.setenv("CUDA_VISIBLE_DEVICES", "")  # 本测试的全部张量都明确驻留CPU。
            patches.setenv("OMP_NUM_THREADS", "1")  # 命令行同样设置，避免导入前线程池扩张。

            def reject_cuda(*args: Any, **kwargs: Any) -> None:
                r"""真实算法若申请CUDA上下文立即失败，不接触GPU。"""
                raise AssertionError("CPU training contract attempted CUDA initialization")

            patches.setattr(torch.cuda, "_lazy_init", reject_cuda)  # 仅设备边界拦截，不改算法算子。
            torch.set_num_threads(1)  # 本来已导入torch时也落实单线程CPU预算。
            yield  # autouse模块fixture先于普通api依赖执行。
            added = set(sys.modules) - before  # 检查整个训练/恢复生命周期的新增模块。
            assert not [name for name in added if name.split(".")[0] in _PHYSICS], "加载了物理运行时"
            assert not torch.cuda.is_initialized(), "CPU 集成测试不得创建 CUDA 上下文"
    finally:
        sys.meta_path.remove(guard)  # 退出后恢复后续contract的标准导入路线。
        torch.set_num_threads(threads)  # 只恢复本测试修改的CPU并行度。


@pytest.fixture(scope="module")
def training_api(api: SimpleNamespace, cpu_only: None) -> Iterator[SimpleNamespace]:
    r"""复用真实learner加载边界；日志资源读取仍执行生产/proc实现。"""
    root = Path(__file__).resolve().parents[3]  # distill源码根，非另一个安装副本。
    with pytest.MonkeyPatch.context() as patches:
        # rl diagnostics包的聚合入口还导出资产分类；本合同只需其真实runtime叶模块。
        name = "anymani.distill.diagnostics.recording.rl"  # 不让无关聚合入口扩张导入依赖。
        package = ModuleType(name)  # 仅省略包入口，叶模块函数保持原样。
        package.__path__ = [str(root / "diagnostics/recording/rl")]  # 真实资源读取源码路径。
        patches.setitem(sys.modules, name, package)  # 退出fixture后恢复原有包身份。
        prefix = "anymani.distill.rl.flash_sac."  # 算法对象共用真实的生产模块身份。
        modules = {  # 缺失identity/checkpoint应明确导入失败，不能用空实现回退。
            field: importlib.import_module(prefix + field)
            for field in ("training", "identity", "checkpoint", "environment", "replay", "metrics")
        }
        yield SimpleNamespace(**vars(api), **modules)  # learner/replay等全部来自真实源码。


@pytest.fixture(autouse=True)
def restore_host_rng() -> Iterator[None]:
    r"""隔离Python/NumPy进程随机状态；PyTorch CPU随机流由既有restore_cpu_rng夹具恢复。"""
    python_state, numpy_state = random.getstate(), np.random.get_state()  # 两条环境侧随机流的独立快照。
    try:
        yield  # 不替换训练循环实际调用的get/set_state。
    finally:
        random.setstate(python_state)  # 测试结束恢复外部Python随机序列。
        np.random.set_state(numpy_state)  # 测试结束恢复外部NumPy随机序列。


def _config(api: SimpleNamespace) -> Any:
    r"""64条含预热新交互、13次Q更新、7次Actor更新；最小合法池为4*(30+3+1)。"""
    return api.config.FlashSACConfig(
        seed=281,  # 固定模型初始化、行为重复噪声与独立回放随机流。
        actor_variant="flash_mlp",  # 真实归一化MLP Actor，保留完整6585维输入。
        asset_count=2,  # 16/9 DoF两种手型，标签只用于路由和统计。
        num_envs=4,  # 同步采样轴N=4；预算必须精确止于该边界。
        total_transitions=64,  # 包含最初12条预热，不额外补采样。
        learning_starts=12,  # n=3所需的第一组未来成熟时，Q预算仍为零。
        replay_capacity=136,  # T=34，大于History30+n3，持久内容有限且完整。
        replay_device="cpu",  # 所有转移、抽样索引和重建历史驻留CPU。
        batch_size=4,  # 每资产严格抽两条，独立于每资产两个环境副本。
        history_steps=30,  # oldest-to-latest且包含当前帧。
        n_step=3,  # 最多三步的真实折扣回报，不跨物理或恢复边界。
        updates_per_transition=0.25,  # 每四条预热后的新交互一次Q优化。
        actor_update_period=2,  # 首次Q优化同时更新Actor，此后隔次。
        actor_mlp_hidden_dim=16,  # 小隐层只控制CPU成本，不缩减输入ABI。
        actor_mlp_num_blocks=1,  # 保留真实残差与归一化。
        critic_hidden_dim=16,  # 两套独立Q参数各16维。
        critic_num_blocks=1,  # 每个Q都运行真实反向传播。
        critic_bins=11,  # categorical支持仍为[-5,5]。
        console_interval=12,  # 12/24/36/48/60/64形成不同长度末窗口。
        checkpoint_interval=24,  # 24/48/64保留模型，resume维护最后完整边界。
        use_amp=False,  # CPU FP32，无AMP或CUDA上下文。
        compile_mode=None,  # eager真实算子，无编译耗时放大。
    )


class _Evidence:
    r"""只观察训练循环的证据drain时机；任务统计来自环境逐步事实。"""

    def __init__(self) -> None:
        r"""记录每次强制/普通drain，不伪造回合或学习样本计数。"""
        self.drains: list[bool] = []  # 强制drain应与完整checkpoint发布边界一一对应。

    def drain(self, *, force: bool = False) -> dict[str, Any]:
        r"""返回无附加字段的小载荷；训练循环在checkpoint前强制调用。"""
        self.drains.append(force)  # 只审计生命周期，不返回假学习结果。
        return {}  # 本合成环境无额外HDF5/Parquet载荷。


class _Transport:
    r"""最小CPU传输接口；逐资产统计的分母由真实fake step记录直接形成。"""

    def __init__(self, env: _Environment) -> None:
        r"""建立等额路由、课程状态与本窗口尚未drain的起点。"""
        self.env = env  # 只有此环境能产生新transition。
        self._rl_device = "cpu"  # training据此创建真实CPU learner。
        self.prototype_index = torch.tensor(_LABELS)  # [N=4]，交错的两资产轴。
        self.training_evidence = _Evidence()  # 只承接真实循环已完成边界的drain。
        self.versions: list[int] = []  # 每个动作使用前的累计新transition版本。
        self.configured: list[tuple[Path, str]] = []  # 证据目录和完整方法digest。
        self.restored: list[dict[str, Any]] = []  # 接收到的课程恢复状态。
        self.course = {"level": 3, "observed_vector_steps": 0}  # 课程和物理age刻意分别保存。
        self.cursor = 0  # 只计本进程已经排空的fake step记录。
        self.drained_at: list[int] = []  # 核对日志窗口不会重复消费原始数据。

    def set_train_info(self, count: int) -> None:
        r"""记录行为动作前的C，回放重用不能调用该接口伪造新版本。"""
        self.versions.append(count)  # [0,4,...]；resume从保存的绝对预算继续。

    def configure_training_evidence(self, directory: Path, digest: str) -> None:
        r"""绑定本段产物目录和方法身份，不自行启动记录后台线程。"""
        self.configured.append((Path(directory), digest))  # 一段物理重建对应一个新run目录。

    def get_env_state(self) -> dict[str, Any]:
        r"""完整保留课程状态；回合age、关节状态与History不属于可移植物理恢复。"""
        return {"curriculum": deepcopy(self.course)}  # 独立小字典，checkpoint可安全持久化。

    def set_env_state(self, state: dict[str, Any]) -> None:
        r"""课程必须在新环境reset前恢复，不能把旧回合history装回新物理段。"""
        assert self.env.reset_calls == 0, "课程应在物理reset前恢复"
        self.restored.append(deepcopy(state))  # 保存实际输入以核对checkpoint链路。
        self.course = deepcopy(state["curriculum"])  # 保留统计进度，不设置任何物理age。

    def drain_rollout_metrics(self) -> dict[str, torch.Tensor]:
        r"""逐资产返回未消费窗口的原始reward/net及自然结束分母，空终点均值显式为0。"""
        records = self.env.records[self.cursor :]  # 每条记录恰为一次已完成的vector step。
        assert records, "空窗口不能形成任务均值"
        result = {  # 与生产training._log实际读取的任务键逐一对齐。
            key: torch.zeros(2, dtype=torch.float64)
            for key in (
                "reward_mean",  # 原始逐步reward均值。
                "net_turns_mean",  # 当前回合pre-reset净圈均值。
                "terminal_net_turns_mean",  # 自然终点净圈均值。
                "rollout_sample_count",  # 此窗口的新交互分母。
                "completed_episode_count",  # 此窗口的自然结束回合分母。
            )
        }
        for record in records:  # 小fake逐行归约，独立于真实learner与回放采样。
            for env, asset in enumerate(_LABELS):  # 不借助连续资产切片假设。
                result["rollout_sample_count"][asset] += 1  # 每个环境当前步贡献一条新样本。
                result["reward_mean"][asset] += record["reward"][env]  # 未经过reward_normalizer缩放。
                result["net_turns_mean"][asset] += record["net"][env]  # pre-reset当前回合净圈。
                if record["terminated"][env] or record["truncated"][env]:  # 自然结束才计回合分母。
                    result["completed_episode_count"][asset] += 1  # 不将无终止副本填零计入。
                    result["terminal_net_turns_mean"][asset] += record["net"][env]  # 终点冻结事实。
        for key in ("reward_mean", "net_turns_mean"):  # rollout量按新交互分母求均值。
            result[key] /= result["rollout_sample_count"]  # 每资产此窗口恰有2*vector_steps个样本。
        result["terminal_net_turns_mean"] /= result["completed_episode_count"].clamp_min(1)  # 空分母占位0。
        self.cursor = len(self.env.records)  # 下个窗口不能重复使用这些原始记录。
        self.drained_at.append(self.cursor)  # 用于从文件坐标复算窗口覆盖。
        self.training_evidence.drain()  # 与真实transport相同的普通drain生命周期。
        return result  # 每项[A=2]，JSON写入只发生在被测训练循环。


class _Environment:
    r"""合成10秒/步CPU环境：named observation、History30、pre-reset七事实均自洽。

    q/pi由初态、回合号、age和上一动作解析确定；这只是有限张量动力学，不模拟刚体。
    第0行50秒timeout，第1行20秒drop，第2行20秒timeout，第3行40秒axis失败。
    首30秒完成后净圈冻结，晚失败不能改写已经完成的窗口。
    """

    def __init__(self, api: SimpleNamespace, *, initial_offset: float = 0.0, corrupt_at: int | None = None) -> None:
        r"""复用16/9 DoF真实named fixture的mask/限位/运动学图；动态历史由本环境重新产生。"""
        self.api = api  # SACStep与typed观测均使用真实生产数据合同。
        source = _observation()  # 既有网络fixture拥有真实21-owner/16-joint/30-history轴。
        rows = torch.tensor([2, 0, 2, 0])  # 资产1为9DoF、资产0为16DoF，各两副本。
        self.base = {name: value[rows].clone() for name, value in source.items()}  # 同资产静态字段逐位一致。
        self.provider = _FrozenQGeometry(api.observations.PalmRotationGeometry).eval()  # 无可训练参数。
        self.initial_offset, self.corrupt_at = initial_offset, corrupt_at  # 偏置区分恢复后的新物理初态。
        self.transport = _Transport(self)  # training只通过这个明确接口消费任务证据。
        self.raw = SimpleNamespace(command_manager=self)  # get_term返回下面同一个pre-reset command。
        self.command = SimpleNamespace(post_physics_evaluation_snapshot={})  # 每步冻结七项事实。
        self.ages = torch.zeros(4, dtype=torch.long)  # [N]，当前回合已完成合成步数。
        self.episodes = torch.zeros(4, dtype=torch.long)  # [N]，自然reset次数，不作为模型身份输入。
        self.history = torch.zeros(4, 30, 16, 5)  # reset先重复首帧，再用于真实Actor。
        self.live: dict[str, torch.Tensor] = {}  # 最新reset后named观察。
        self.records: list[dict[str, torch.Tensor]] = []  # 最多16条小型物理记录，独立审计源。
        self.histories: list[torch.Tensor] = []  # 每次动作前的干净History30，用于顺序审计oracle。
        self.reset_calls = 0  # 初始reset不是transition，不增加learner/replay预算。

    def get_term(self, name: str) -> SimpleNamespace:
        r"""训练循环只读取已冻结goal_pose快照，不允许再次推进任务状态。"""
        assert name == "goal_pose", f"未知command项：{name}"
        return self.command  # 同一对象被每次step写入post-physics事实。

    def _frame(self, actions: torch.Tensor) -> torch.Tensor:
        r"""形成[N,16,5]帧：q/pi、目标/pi、上一动作、零own-contact、二元TIP-contact。"""
        frame = torch.zeros(4, 16, 5)  # 动态字段全FP32，未使用的ghost槽初始为零。
        position = self.initial_offset + torch.arange(4) * 0.03 + self.episodes * 0.02 + self.ages * 0.01  # 归一q/pi。
        frame[..., 0] = position[:, None] + actions / (24 * math.pi)  # q/pi，显式有界动作项的量纲转换。
        frame[..., 1] = position[:, None] + 0.02  # 归一目标与当前q不同，可识别时序错位。
        frame[..., 2] = actions  # 上一策略归一动作，ghost动作必须为零。
        tips = (self.ages[:, None] % 2).float() * self.base["tip_valid"]  # [N,F]合法二元触觉。
        frame[..., 4] = tips.repeat(1, 4)  # depth-major，四深度共享所属手指TIP触觉。
        return frame.masked_fill(~self.base["jnt_valid"][..., None], 0)  # 合法物理关节以外精确为零。

    def _named(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""保持完整named ABI，几何按本次q解析，特权状态只进入critic字段。"""
        data = {name: value.clone() for name, value in self.base.items()}  # 不别名旧live，保障append读取旧current。
        data["actor_jnt_current"] = frame  # [N,16,5]，当前控制帧。
        data["actor_jnt_history"] = self.history.clone()  # oldest-to-latest，包括当前帧。
        data["actor_owner_contact"].zero_()  # Actor没有PALM/JOINT接触。
        data["actor_owner_contact"][:, 17:, 0] = (self.ages[:, None] % 2) * data["tip_valid"]  # 二元TIP。
        data["critic_jnt_state"] = frame[..., :4].clone()  # [N,16,4]，解析的合成特权关节状态。
        data["critic_owner_contact"] = data["actor_owner_contact"].repeat(1, 1, 2)  # [N,21,2]有限接触量。
        data["critic_obj"].zero_()  # 物体其他特权分量在合成模型内为零。
        data["critic_obj"][:, 0, 0] = self.ages * 0.1  # 用age产生可区分的动态物体状态。
        data["critic_task"].zero_()  # 不把资产标签混作任务神经特征。
        data["critic_task"][:, 0, 0] = self.ages / torch.tensor(_PERIODS)  # 已经过的回合比例。
        data["critic_reward_release"].fill_(0.5)  # [N,1]固定合法释放系数。
        with torch.no_grad():  # provider和real learner都按冻结几何边界调用。
            actor, _ = self.api.observations.actor_inputs(data)  # 真实typed验证与ghost/TIP净化。
            geometry = self.provider.resolve(self.transport.prototype_index, actor)  # 真实q对应的解析几何。
        data["geometry_tokens"] = geometry.tokens  # [N,21,128]，不是跨状态learned缓存。
        return data  # 静态图来自同一手型，全部实数FP32、全部tensor CPU。

    def reset(self) -> dict[str, torch.Tensor]:
        r"""物理重建从新初态开始；reset帧重复30次，不产生奖励、终止或新交互。"""
        self.reset_calls += 1  # 构造后首次reset应恰为1次。
        self.ages.zero_()  # 物理age不能从课程状态恢复。
        self.episodes.zero_()  # 合成回合身份也属于新物理段。
        frame = self._frame(torch.zeros(4, 16))  # 初始帧的上一动作全零。
        self.history = frame[:, None].repeat(1, 30, 1, 1)  # reset首帧重复，含当前帧。
        self.live = self._named(frame)  # 完整合法named观察交付真实Actor。
        return self.live  # 原始课程metadata不进入神经输入。

    def step(self, actions: torch.Tensor) -> Any:
        r"""推进一次真实fake交互，冻结终点后再auto-reset；SACStep清楚区分live/next_compact。"""
        assert actions.device.type == "cpu" and actions.shape == (4, 16), "动作必须是CPU [N=4,J=16]"
        assert torch.isfinite(actions).all() and torch.all(actions.abs() <= 1), "实际Actor动作必须有限有界"
        assert torch.count_nonzero(actions[~self.base["jnt_valid"]]) == 0, "ghost动作不得作用于合成动力学"
        self.histories.append(self.history.clone())  # 独立保存动作前干净时序，故障只污染对外观察。
        self.ages += 1  # 一个物理步恰好推进一次，不在日志/first30中重复推进。
        terminal_frame = self._frame(actions)  # 仍属于旧回合的post-physics终点。
        self.history = torch.cat((self.history[:, 1:], terminal_frame[:, None]), dim=1)  # 正常左移并追加。
        terminal = self._named(terminal_frame)  # 先形成终点的全部动态与特权状态。
        next_compact = {
            name: value.clone() for name, value in self.api.observations.compact_observation(terminal).items()
        }  # 独立克隆，后续auto-reset不能改写旧终点。
        periods = torch.tensor(_PERIODS)  # [N]，每行固定但互不相同的回合长度。
        ended = self.ages == periods  # [N]，仅自然结束产生done。
        terminated = ended & torch.tensor([False, True, False, True])  # drop/axis两类真实事件。
        truncated = ended & ~terminated  # 50秒timeout或20秒早timeout，语义保持不同。
        net = self.ages.double() * torch.tensor(_TURN_RATES, dtype=torch.float64)  # 未截断有符号净圈。
        snapshot = {  # 七事实全部[N]且在reset前冻结，时间/弧度/圈数不可混用。
            "episode_duration_s": self.ages.double() * 10.0,  # 10秒/合成步，与测试身份一致。
            "net_rotation_rad": net * (2 * math.pi),  # 圈数转回原始物理角度单位rad。
            "net_turns_first30": torch.where(self.ages >= 3, torch.tensor(_TURN_RATES) * 3, 0).double(),
            "first30_complete": self.ages >= 3,  # 当且仅当duration>=30秒。
            "termination_object_out_of_anchor": ended & torch.tensor([False, True, False, False]),
            "termination_goal_axis_misaligned": ended & torch.tensor([False, False, False, True]),
            "termination_time_out": truncated.clone(),  # 提前纯timeout由真实first30模块统计为删失。
        }
        self.command.post_physics_evaluation_snapshot = snapshot  # 不在get_term或metrics中重新计算。
        reward = torch.tensor([1.0, -2.0, 3.0, 4.0]) + 0.125 * len(self.records)  # 原始逐步reward，非缩放Q目标。
        self.records.append(
            {  # 只保留小记录，供日志分母与恢复边界的独立oracle使用。
                "reward": reward.clone(),  # [N]原始逐步reward。
                "net": net.clone(),  # [N]当前回合pre-reset净圈。
                "terminated": terminated.clone(),  # [N]drop/axis事实。
                "truncated": truncated.clone(),  # [N]计划时域结束事实。
                "actions": actions.clone(),  # [N,16]真实Actor输出。
                "next_frame": terminal_frame.clone(),  # [N,16,5]显式物理终点。
            }
        )
        self.transport.course["observed_vector_steps"] += 1  # 可恢复课程时钟与实际fake交互保持一一对应。

        # auto-reset只修改自然结束的副本；其他副本历史连续，next_compact保留旧回合终点。
        self.ages[ended] = 0  # 新物理回合age从零开始。
        self.episodes[ended] += 1  # 区分多次reset的首帧，防止偶然常量掩盖串段。
        live_frame = terminal_frame.clone()  # 非done行继续使用post-physics帧。
        live_frame[ended] = self._frame(torch.zeros_like(actions))[ended]  # done行上一动作归零。
        self.history[ended] = live_frame[ended, None].repeat(1, 30, 1, 1)  # reset首帧必须重复。
        self.live = self._named(live_frame)  # 后续Actor读取reset后实时状态。
        if len(self.records) == self.corrupt_at:  # 故障只改变历史次序，末帧与current仍保持一致。
            self.live["actor_jnt_history"][:, :-1] = self.live["actor_jnt_history"][:, :-1].flip(1)
        return self.api.environment.SACStep(self.live, reward, terminated, truncated, next_compact, {})  # 真实数据类。


def _new_run(api: SimpleNamespace, directory: Path, **env_kwargs: Any) -> Any:
    r"""建立明确合成物理身份，再构造真实训练循环；目录只属于当前测试临时case。"""
    config = _config(api)  # A2/N4/C64、固定小网络和最小合法池。
    directory.mkdir(parents=True, exist_ok=True)  # 身份fixture只写入这个测试目录。
    inputs = _identity_case(directory, config)  # 用真实identity builder消费小型锁文件和catalog字节。
    inputs["task_contract"].update(policy_hz=0.1, physics_hz=0.1, synthetic_dynamics=True)  # 10秒/步。
    inputs["provider_identity"] = {"kind": "frozen-analytic-cpu-fixture", "token_width": 128}  # 不冒称N040权重。
    identity = api.identity.build_method_identity(**inputs)  # 真实原生身份与checkpoint校验均执行。
    env = _Environment(api, **env_kwargs)  # 只替换物理producer，不替换learner或replay。
    return api.training.FlashSACTrainingRun(config, env, env.provider, identity, directory)  # 独立完整CPU run。


def _trace_updates(run: Any, patches: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    r"""透明记录真实update返回值与抽样配额；原forward/loss/backward/Adam按原顺序完整执行。"""
    original = run.learner.update  # 保存真正的生产bound method，不能返回占位loss。
    records: list[dict[str, Any]] = []  # 最多13条标量字典，不保存训练计算图。

    def observed_update(batch: dict[str, Any]) -> dict[str, float]:
        r"""检查银行重建后的实际batch，再转发给真实learner并复制其标量输出。"""
        counts = torch.bincount(batch["asset_index"], minlength=2)  # 元数据仅用于检查逐资产抽样配额。
        torch.testing.assert_close(counts, torch.tensor([2, 2]), rtol=0, atol=0)  # B4/A2严格等额。
        for side in ("obs", "next_obs"):  # current与n-step终点都必须完成真实静态/几何重建。
            assert batch[side]["actor_jnt_history"].shape == (4, 30, 16, 5), f"{side}: History30轴错误"
            assert batch[side]["geometry_tokens"].shape == (4, 21, 128), f"{side}: 几何轴错误"
            assert all(value.device.type == "cpu" for value in batch[side].values()), f"{side}: 非CPU张量"
        info = original(batch)  # 真正计算分布式Q、策略/温度、Adam和target EMA。
        records.append({"transitions": run.learner.collected_transitions, "info": dict(info)})  # 不重新计算loss。
        return info  # 原字典原样传给training._last_learning更新。

    patches.setattr(run.learner, "update", observed_update)  # 只透明观察这个真实实例的调用。
    return records  # 用独立保存的返回值验证learning_last不是窗口平均值。


def _json_records(path: Path) -> list[dict[str, Any]]:
    r"""严格检查有限、可解析且有界的JSON产物，包括解析后溢出为Inf的数值。"""
    import json

    assert path.is_file() and path.stat().st_size <= 256 * 1024, f"文件缺失或超出小预算：{path}"
    text = path.read_text(encoding="utf-8")  # 一次读取本测试拥有的小产物。
    records = [json.loads(line) for line in text.splitlines()] if path.suffix == ".jsonl" else [json.loads(text)]
    assert 0 < len(records) <= 32, f"JSON行数不符合小预算：{path}, rows={len(records)}"
    for record in records:  # allow_nan=False同时拒绝NaN、Infinity和解析后1e999溢出。
        assert isinstance(record, dict), f"JSON记录必须是具名字段对象：{path}"
        json.dumps(record, ensure_ascii=False, allow_nan=False)  # 递归检查全部数值，None保持未定义语义。
    return records  # 测试消费与用户实际查看的是同一份落盘内容。


@pytest.fixture(scope="module")
def finished_case(training_api: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    r"""预算/窗口/快照三个测试共享一次完整64条CPU闭环，避免重复训练扩大成本。"""
    with torch.random.fork_rng(devices=[]), pytest.MonkeyPatch.context() as patches:
        run = _new_run(training_api, tmp_path_factory.mktemp("flash_sac_training"))  # 仅16个vector steps。
        initial = deepcopy(run.learner.state_dict())  # 证明真实权重/Adam不是静态占位。
        updates = _trace_updates(run, patches)  # 透明收集真实算法输出，不改变数据与优化顺序。
        report = run.run()  # 真实循环、first30、文件保存与JSON写入。
        final = deepcopy(run.learner.state_dict())  # 在fork_rng退出前封存真实末次保存对应的全局随机流。
        return SimpleNamespace(run=run, initial=initial, final=final, updates=updates, report=report)  # 后续只读。


def test_real_sampling_and_optimizer_clocks_include_warmup(training_api: SimpleNamespace, finished_case: Any) -> None:
    r"""预热已计入64条；Q=13、Actor=7且Adam步数一致，历史审计覆盖全部成熟起点。"""
    run, report = finished_case.run, finished_case.report  # 同一个已完成的真实小闭环。
    assert type(run.learner) is training_api.learner.FlashSACLearner, "learner必须是真实算法实例"  # 完整算法owner。
    assert type(run.replay) is training_api.replay.CompactReplay, "回放不能以计数替身代替"  # 真实转移与序列存储。
    assert type(run.bank) is training_api.observations.StaticHandBank, "静态表必须实际重建观察"  # 按手型重建。
    # 首30秒窗口由生产统计owner结算，而非fake预先填入结果。
    assert type(run.first30) is training_api.metrics.FirstThirtySecondsMetrics, "first30必须消费真实七事实"
    assert report["status"] == "completed" and report["transitions"] == 64, report  # 恰好用完声明的新数据预算。
    assert len(run.env.records) == 16 and run.env.reset_calls == 1, "初始reset不能算新交互"  # 16*4=64。
    assert run.env.transport.versions == list(range(0, 64, 4)), "行为版本必须按真实vector step推进"  # 动作前的C。
    # learner和回放必须描述同一个已经发生的交互集合。
    assert run.learner.collected_transitions == run.replay.total_transitions == 64, "采样/回放预算不一致"
    assert int(run.learner.reward_normalizer.count) == 64, "预热/优化复用改变了reward样本分母"  # 不按batch复用增量。
    expected_q = run.config.critic_update_budget(64)  # 独立以公开配置定义应完成的预算。
    expected_actor = (expected_q + run.config.actor_update_period - 1) // run.config.actor_update_period  # ceil(U/2)。
    assert expected_q == report["critic_updates"] == len(finished_case.updates) == 13, "Critic更新预算错误"  # U(64)。
    assert expected_actor == report["actor_updates"] == 7, "Actor延迟相位必须从首个Q更新开始"  # Q1/Q3/.../Q13。
    assert [row["transitions"] for row in finished_case.updates] == list(range(16, 65, 4)), "预热前进行了优化"  # C>12。
    assert report["history_vector_steps_verified"] == 16 - 3 + 1, "每个成熟起点必须实际审计History30"  # S-n+1。
    # 独立事件表给出8次drop+4次axis、3次晚timeout+8次早timeout。
    assert report["physical_termination_rows"] == 12 and report["timeout_rows"] == 11, "终止事实计数错误"

    # Adam真实step和权重变化共同证明确实发生优化，不能只把计数器加到期望值。
    state = finished_case.final  # 包含真实optimizer moments与全部在线/目标网络及保存当刻RNG。
    for key, expected in (("actor_optimizer", 7), ("critic_optimizer", 13), ("temperature_optimizer", 7)):
        values = state[key]["state"].values()  # 每个实际被优化参数的Adam状态。
        # 每个实际参与优化的参数都应处于同一Adam时钟，不能只检查外层计数。
        assert values and {int(value["step"]) for value in values} == {expected}, f"{key}: Adam步数不匹配"
    for key in ("actor", "critic", "target_critic"):  # 包括目标网络EMA，非纯参数构造成功。
        # 在线网络和目标网络至少一个状态量改变，排除只推进计数的空更新。
        assert any(not torch.equal(value, finished_case.initial[key][name]) for name, value in state[key].items()), key
    assert not state["cuda_rng"], "CPU learner checkpoint不得保存CUDA随机流"  # 设备边界仍是CPU。
    assert all(not call[3] for call in run.provider.calls), "几何provider调用必须冻结"  # autograd不进入几何producer。
    assert not list(run.provider.parameters()), "合成冻结provider不得被优化器训练"  # 只有固定buffer。


def test_each_log_window_keeps_raw_asset_denominators_and_learning_last(finished_case: Any) -> None:
    r"""从独立逐步事实复算每窗口任务统计；learning_last严格等于真实最近update输出的逐键保留值。"""
    run = finished_case.run  # 一个完整64条轨迹，只读本次真实文件。
    metrics = _json_records(run.metrics_path)  # six窗口，末窗口只有4条。
    assets = _json_records(run.asset_metrics_path)  # 每个窗口始终有A=2条资产记录。
    assert [row["transitions"] for row in metrics] == [12, 24, 36, 48, 60, 64], "日志坐标不在真实边界"  # 末窗口4条。
    assert len(assets) == 2 * len(metrics), "每个日志窗口必须覆盖全部资产"  # A行/窗口。
    # 预热窗口尚未执行优化，学习损失应保持未定义，而不是填零。
    assert metrics[0]["learning_last"] == {} and metrics[0]["critic_updates"] == 0, "预热日志伪造了loss"
    assert run.env.transport.drained_at == [3, 6, 9, 12, 15, 16], "任务窗口被漏记或重复drain"  # 逐窗口排空游标。
    previous = 0  # 上个日志窗口的绝对transition结束坐标。
    for index, row in enumerate(metrics):  # 每个窗口独立核算，不只检查最终累计数。
        count = row["transitions"]  # 当前真实新交互总数。
        records = run.env.records[previous // 4 : count // 4]  # 与文件对应的原始物理步集合。
        assert row["window_transitions"] == count - previous, f"C={count}: 窗口新交互分母错误"  # Delta C，不是C。
        expected_q = run.config.critic_update_budget(count)  # 预算不由日志间隔决定。
        assert row["critic_updates"] == expected_q and row["actor_updates"] == (expected_q + 1) // 2, row  # 更新相位。
        assert row["replay_occupancy"] == count and row["replay_bytes"] == run.replay.storage_bytes, row  # 实际池容量。
        last = {}  # 以真实调用返回值逐键前向保留，Actor隔次缺项不能凭空重算。
        for update in finished_case.updates:
            if update["transitions"] <= count:  # 不能读未来版本的loss。
                last.update(update["info"])  # 只归约真实返回的标量字典。
        assert row["learning_last"] == last, f"C={count}: 最近学习量被错误平均或与任务数据混合"  # 逐键最近值。
        group = row["families"]["all"]  # 小cohort明确只有一个all组。
        all_ended, terminal_sum, reward_sum, net_sum = 0, 0.0, 0.0, 0.0  # 独立物理分母/分子。
        for asset in range(2):  # 用Python标量从原始事实复算，与fake transport张量归约独立。
            selected = [env for env, label in enumerate(_LABELS) if label == asset]  # 每资产两个副本。
            rewards = [float(item["reward"][env]) for item in records for env in selected]  # 未缩放reward。
            nets = [float(item["net"][env]) for item in records for env in selected]  # pre-reset净圈。
            ends = [
                float(item["net"][env])
                for item in records
                for env in selected
                if bool(item["terminated"][env] | item["truncated"][env])
            ]  # 仅自然终点，不用有条件均值的零填充替代分母。
            record = assets[2 * index + asset]  # 文件顺序以资产索引显式标注。
            expected = {  # 均值和分母共同验证，防止“看起来合理”的假统计。
                "transitions": count,  # 完整vector-step结束坐标。
                "asset_index": asset,  # 独立统计路由标签。
                "rollout_sample_count": len(rewards),  # 逐步reward/net的分母。
                "completed_episode_count": len(ends),  # 自然终点均值的分母。
                "reward_mean": math.fsum(rewards) / len(rewards),  # 原始reward/step。
                "net_turns_mean": math.fsum(nets) / len(nets),  # pre-reset当前净圈。
                "terminal_net_turns_mean": math.fsum(ends) / len(ends) if ends else 0,  # 空分母占位0。
            }
            assert record == pytest.approx(expected), f"C={count}, asset={asset}: 原始任务分母或均值错误"  # 独立重算。
            assert not (set(last) & record.keys()), "learner-last字段不得混入逐资产原始任务行"  # 统计样本空间不同。
            all_ended += len(ends)  # 终点分母不与rollout分母混合。
            terminal_sum += math.fsum(ends)  # 组终点均值必须按回合加权，而非资产均值再平均。
            reward_sum += math.fsum(rewards)  # 组原始reward总和。
            net_sum += math.fsum(nets)  # 组所有当前净圈总和。
        assert group["ended"] == all_ended, f"C={count}: 组回合分母错误"  # 各资产真实自然结束数之和。
        # 终点采用回合等权；逐步reward采用新交互等权，两个分母具有不同统计意义。
        assert group["terminal_net"] == pytest.approx(terminal_sum / all_ended), f"C={count}: 终点权重错误"
        # 此处核对环境原始reward/Delta C，Q的缩放只属于learner内部目标。
        assert group["reward_per_step"] == pytest.approx(reward_sum / (count - previous)), "原始reward被缩放"
        assert group["live_net"] == pytest.approx(net_sum / (count - previous)), "当前净圈窗口错误"  # rollout均值。
        previous = count  # 下个窗口不再消费任何已检查样本。

    # 16步的独立事件表：资产0有8个早drop与4个安全first30，资产1有3个安全first30、8个删失。
    summary = metrics[-1]["first30"]  # first30真实模块的累计近期窗口结果。
    assert summary == pytest.approx(
        {
            "assets_with_window": 2,  # 两种手型均形成至少一个有效窗口。
            "first30_episode_count": 15,  # 8个早drop+7个完整first30。
            "censored_early_timeouts": 8,  # 纯早timeout不进入净圈/安全分母。
            "first30_asset_mean_net": 2 / 3,  # ((8*(-.25)+4*2.25)/12+.75)/2。
            "first30_episode_mean_net": 37 / 60,  # (8*(-.25)+4*2.25+3*.75)/15。
            "first30_safe_fraction": 7 / 15,  # 晚失败不能回写30秒时已结算的安全结果。
        }
    ), "完成/早失败/早timeout没有按同一pre-reset事实结算"


def test_periodic_models_and_resume_have_distinct_replay_payloads(
    training_api: SimpleNamespace, finished_case: Any
) -> None:
    r"""模型保留24/48/64边界且不含replay；resume含64条完整回放及实验随机状态。"""
    run = finished_case.run  # 同一完整小闭环的真实文件，不另起保存替身。
    paths = sorted((run.run_dir / "nn").glob("*.pt"))  # 只检查当前临时run的产物。
    assert {path.name for path in paths} == {"resume.pt", *(f"transitions_{count:09d}.pt" for count in (24, 48, 64))}
    for count in (24, 48, 64):  # 所有周期模型都能被真实reader读取并通过身份检查。
        path = run.run_dir / "nn" / f"transitions_{count:09d}.pt"  # 文件名明确锚定真实新交互数。
        saved = training_api.checkpoint.read_checkpoint(path, expected_identity=run.identity)  # 真I/O。
        # 模型快照具有只读模型消费资格，完整恢复另由含回放产物声明。
        assert saved["metadata"] == {"kind": "model_snapshot", "has_replay": False}, "模型快照资格错误"
        assert saved["replay"] is None and "replay" not in saved["learner"], "模型意外携带整份回放"  # 两处均无整池。
        # 两个状态owner持久化同一C；保存前必须完成该数据预算应有的U(C)次优化。
        assert saved["learner"]["collected_transitions"] == saved["experiment"]["collected_transitions"] == count
        # 依据公共预算函数复算，避免把文件名当作已经完成相应优化的证明。
        assert saved["learner"]["critic_updates"] == run.config.critic_update_budget(count), "保存半次更新边界"
        _assert_tree_equal(saved["static_bank"], run.bank.state_dict(), f"C={count}: static bank")  # 小表逐位一致。
    resume = training_api.checkpoint.read_checkpoint(
        run.run_dir / "nn/resume.pt", expected_identity=run.identity, require_replay=True
    )  # 完整恢复资格不得由文件名推断。
    assert resume["metadata"] == {"kind": "full_resume", "has_replay": True}, "resume缺完整回放资格"  # 真实presence。
    _assert_tree_equal(resume["replay"], run.replay.state_dict(), "published full replay")  # 包括全部已写/未写环槽。
    _assert_tree_equal(resume["learner"], finished_case.final, "published full learner")  # 参数/Adam/温度/保存当刻RNG。
    _assert_tree_equal(resume["experiment"]["replay_rng"], run.replay_rng.get_state(), "published replay RNG")
    assert sum(run.env.transport.training_evidence.drains) == 3, "checkpoint前强制drain次数错误"  # 对应24/48/64。
    # 人类看到的终局文件和调用方获得的返回值必须描述同一组运行事实。
    assert _json_records(run.run_dir / "training_summary.json")[0] == finished_case.report, "落盘摘要与返回值矛盾"


def test_history_audit_rejects_oldest_to_latest_corruption(training_api: SimpleNamespace, tmp_path: Path) -> None:
    r"""第2步仅翻转历史前29帧，第5步起点成熟时审计必须捕获；末帧/current一致不能掩盖错序。"""
    run = _new_run(training_api, tmp_path / "bad_history", corrupt_at=2)  # 其余全部观察和算法保持合法。
    with pytest.raises(AssertionError, match="Mismatched elements"):  # 不接受无关构造/shape错误当作审计成功。
        run.run()  # n=3时sequence2在第五个vector step成熟。
    # s=2的三个未来转移成熟要求S=5，不能等到更晚窗口才报告错序。
    assert len(run.env.records) == 5 and run.learner.collected_transitions == 20, "故障没有在首个成熟边界报出"
    assert run._checked_histories == 2, "故障前的两个干净历史也必须通过真实审计"  # s0/s1排除其他接口错误。
    envs = torch.arange(4)  # 对全部副本复查sequence2的真实回放重建。
    clean = run.replay.gather(envs, torch.full_like(envs, 2))["obs"]["actor_jnt_history"]  # 真实回放不是故障源。
    _assert_tree_equal(clean, run.env.histories[2], "clean replay history")  # 干净的producer oracle。
    sequence, corrupt = run._history_checks[0]  # 异常保留尚未通过的producer参考。
    assert sequence == 2 and not torch.equal(clean[:, :-1], corrupt[:, :-1]), "故障未破坏历史次序"  # 唯一干预是过去帧。
    _assert_tree_equal(clean[:, -1], corrupt[:, -1], "latest frame remains correct")  # 最新帧仍完全正确。
    assert not (run.run_dir / "training_summary.json").exists(), "审计失败不能伪装完成"  # 异常出口不发布成功摘要。
    assert not (run.run_dir / "nn/resume.pt").exists(), "不完整边界不能发布resume"  # 最近24条周期尚未达到。


def test_full_resume_preserves_learning_but_starts_new_physical_history(
    training_api: SimpleNamespace, tmp_path: Path
) -> None:
    r"""28条中断再采36条：完整学习/回放/RNG保留，课程恢复，物理/History与reward trace显式开新段。"""
    first = _new_run(training_api, tmp_path / "first")  # 总方法预算仍是64，进程只走到28。
    report = first.run(stop_after_transitions=28)  # 非周期的完整边界也必须保存快照和resume。
    assert report["status"] == "stopped-at-declared-boundary" and report["transitions"] == 28, report
    assert report["critic_updates"] == first.config.critic_update_budget(28) == 4, "中断边界欠/超更新"
    resume_path = first.run_dir / "nn/resume.pt"  # 显式读完整恢复，不从模型快照猜测数据池。
    payload = training_api.checkpoint.read_checkpoint(
        resume_path, expected_identity=first.identity, require_replay=True
    )  # 完整读取同时验证刚发布的experiment随机状态可安全反序列化。
    old_bytes = {path.name: path.read_bytes() for path in (first.metrics_path, first.asset_metrics_path, resume_path)}
    assert payload["experiment"]["first30"]["seen"].any(), "中断fixture必须包含已结算但尚未结束的first30流"
    assert torch.count_nonzero(payload["learner"]["reward_normalizer"]["returns"]), "中断fixture必须有非空reward trace"
    assert _json_records(first.metrics_path)[-1]["families"]["all"]["terminal_net"] is None, "空终点窗口应输出null"
    python_rng = random.getstate()  # 保存时真实的Python环境侧随机流。
    numpy_rng = cast(tuple[Any, ...], np.random.get_state())  # NumPy默认legacy=True，返回完整MT19937状态元组。
    assert isinstance(numpy_rng, tuple), "本CPU实验明确使用NumPy传统MT19937状态"
    random.seed(812)  # 模拟新进程中不同的Python初始化状态，要求resume实际撤销该差异。
    np.random.seed(813)  # 模拟新进程中不同的NumPy状态，不能恰好相同而形成伪阳性。

    # 新环境使用明显不同的合法q初态；学习身份不改变，物理段绝不能拼接旧History。
    env = _Environment(training_api, initial_offset=0.4)  # 只属于重建物理初态的扰动，静态手型保持相同。
    resumed = training_api.training.FlashSACTrainingRun(
        first.config, env, env.provider, first.identity, tmp_path / "resumed", resume=resume_path
    )  # 新目录完整恢复。
    expected_learner = deepcopy(payload["learner"])  # 恢复允许改变的唯一learner字段是逐环境reward trace。
    expected_learner["reward_normalizer"]["returns"].zero_()  # 不将旧物理回合trace延伸进新环境。
    _assert_tree_equal(resumed.learner.state_dict(), expected_learner, "resumed learner except reset trace")
    expected_replay = deepcopy(payload["replay"])  # 旧转移、终点、done、动作全部逐位保留。
    expected_replay["next_episode_start"].fill_(7)  # 下一次append从S=28/4=7启动新历史段。
    _assert_tree_equal(resumed.replay.state_dict(), expected_replay, "resumed replay with pending new segment")
    expected_first30 = deepcopy(payload["experiment"]["first30"])  # 过去净圈窗口/删失计数继续保留。
    expected_first30["seen"].zero_()  # 所有新物理回合允许重新结算，不伪造终止事件。
    _assert_tree_equal(resumed.first30.state_dict(), expected_first30, "resumed first30 windows")
    _assert_tree_equal(resumed.bank.state_dict(), payload["static_bank"], "resumed static bank")
    _assert_tree_equal(resumed.replay_rng.get_state(), payload["experiment"]["replay_rng"], "resumed replay RNG")
    assert random.getstate() == python_rng, "Python随机流未恢复到保存边界"
    numpy_restored = cast(tuple[Any, ...], np.random.get_state())  # 正常API读回状态，不依赖checkpoint编码方式。
    assert isinstance(numpy_restored, tuple), "恢复不能改变NumPy随机算法状态类型"
    assert numpy_restored[0] == numpy_rng[0] and numpy_restored[2:] == numpy_rng[2:], "NumPy随机算法/游标/缓存错误"
    np.testing.assert_array_equal(numpy_restored[1], numpy_rng[1])  # MT19937完整624词状态逐位恢复。
    assert env.transport.restored == [payload["experiment"]["environment"]], "课程状态未恢复"
    assert env.reset_calls == 1 and not env.records and not env.ages.any(), "构造恢复期间伪造了物理交互"
    assert env.transport.configured == [(resumed.run_dir, first.identity["identity_digest"])], "新证据目录绑定错误"
    initial = resumed.observation["actor_jnt_current"].clone()  # 保存新物理首帧，后续独立核对回放段。
    _assert_tree_equal(
        resumed.observation["actor_jnt_history"], initial[:, None].repeat(1, 30, 1, 1), "reset History30"
    )  # 独立于真实replay重建的reset首帧oracle。
    assert not torch.equal(initial, first.observation["actor_jnt_current"]), "恢复fixture未形成不同物理初态"

    # 从旧实例相同学习状态查询新初态，独立得到下一行为动作；完整resume须保持该噪声相位。
    expected_action = first.learner.act(resumed.observation, explore=True).actions.clone()  # 行为RNG不使用全局RNG。
    result = resumed.run()  # 从28续到64，不重复预热，不扩张方法总交互预算。
    assert result["status"] == "completed" and result["transitions"] == 64, result
    assert len(first.env.records) == 7 and len(env.records) == 9, "中断+恢复的新交互总量不是16个vector steps"
    assert env.transport.versions == list(range(28, 64, 4)), "恢复后行为版本从零重新计数"
    assert result["critic_updates"] == 13 and result["actor_updates"] == 7, "恢复丢失更新相位或重复优化"
    _assert_tree_equal(env.records[0]["actions"], expected_action, "first resumed behavior sample")  # RNG实际续接证据。
    assert result["history_vector_steps_verified"] == 7, "新物理段的成熟History未重新审计"
    assert int(resumed.learner.reward_normalizer.count) == resumed.replay.total_transitions == 64, "恢复后预算分母错误"
    metrics = _json_records(resumed.metrics_path)  # 第一新窗口为28->36，共8条，而不是累计36条。
    assert [row["transitions"] for row in metrics] == [36, 48, 60, 64], "恢复窗口坐标错误"
    assert sum(row["window_transitions"] for row in metrics) == 36, "恢复日志重复消费旧样本"
    new_sample_count = sum(
        row["rollout_sample_count"] for row in _json_records(resumed.asset_metrics_path)
    )  # 逐资产计数，所有窗口合起来必须恰等于36条新增transition。
    assert new_sample_count == 36, "新段原始分母错误"
    assert _json_records(resumed.run_dir / "training_summary.json")[0] == result, "恢复摘要不一致"

    # 最后旧转移没有物理done；新流边界仅截短n-step，旧next仍bootstrap且不会变成新首帧。
    envs = torch.arange(4)  # 四个副本同时进行边界审计，避免只检查恰好done的行。
    old = resumed.replay.gather(envs, torch.full_like(envs, 6))  # 旧段最后一条current。
    _assert_tree_equal(old["steps"], torch.ones(4, dtype=torch.long), "old segment one-step target")
    torch.testing.assert_close(old["discounts"], torch.full((4,), first.config.gamma), rtol=0, atol=0)
    _assert_tree_equal(old["rewards"], first.env.records[-1]["reward"], "old segment raw return")
    _assert_tree_equal(old["next_obs"]["actor_jnt_current"], first.env.records[-1]["next_frame"], "old explicit next")
    assert not (old["terminated"] | old["truncated"]).any(), "物理重建被伪造成失败或timeout"
    new = resumed.replay.gather(envs, torch.full_like(envs, 7))  # 新物理段第一条已成熟回放。
    _assert_tree_equal(
        new["obs"]["actor_jnt_history"], initial[:, None].repeat(1, 30, 1, 1), "new replay reset history"
    )  # 新物理段首个成熟回放只能包含新初态重复。
    final = training_api.checkpoint.read_checkpoint(resumed.run_dir / "nn/resume.pt", require_replay=True)
    assert final["experiment"]["resume_source"] == str(resume_path.resolve()), "恢复lineage丢失"
    assert final["experiment"]["physical_resume"] == "new-strict-pregrasp-history-segment", "物理恢复声明错误"
    assert final["experiment"]["environment"]["curriculum"]["observed_vector_steps"] == 16, "课程计数没有续接"
    for path in (first.metrics_path, first.asset_metrics_path, resume_path):  # 新run不得改写作为恢复源的旧产物。
        assert path.read_bytes() == old_bytes[path.name], f"完整恢复改写了原始产物：{path}"


def test_stop_boundary_rejected_before_any_new_interaction(training_api: SimpleNamespace, tmp_path: Path) -> None:
    r"""非法2条边界、越界68条和零预算必须在step与文件打开之前失败。"""
    run = _new_run(training_api, tmp_path / "invalid_boundary")  # 合法完整配置只构造一次。
    for boundary in (0, 2, 68):  # 空区间、非vector边界、超方法总预算三类。
        with pytest.raises(ValueError, match="vector-step boundary"):
            run.run(stop_after_transitions=boundary)  # 不允许静默向上/向下取整预算。
        assert not run.env.records and run.replay.total_transitions == 0, f"边界{boundary}已产生新交互"
        assert not run.metrics_path.exists() and not run.asset_metrics_path.exists(), "非法预算提前发布了日志"


@pytest.mark.parametrize("filename", ["metrics.jsonl", "asset_metrics.jsonl"])
def test_existing_metrics_are_never_overwritten(training_api: SimpleNamespace, tmp_path: Path, filename: str) -> None:
    r"""任一现存训练指标文件即锁定目录；恢复/重跑应要求新目录而非覆盖或追加旧证据。"""
    run = _new_run(training_api, tmp_path / "immutable")  # 尚未run，所以没有任何训练日志。
    path = run.run_dir / filename  # 只注入测试拥有的既有证据，不触碰用户实验文件。
    original = b'{"transitions": 4, "sentinel": "immutable-original-evidence"}\n'  # 有限合法JSON行。
    path.write_bytes(original)  # 模拟该目录已经由某次训练发布指标。
    env = _Environment(training_api)  # 新请求不应越过目录闸门进入物理reset。
    with pytest.raises(FileExistsError, match="existing training metrics are immutable"):
        training_api.training.FlashSACTrainingRun(run.config, env, env.provider, run.identity, run.run_dir)
    assert path.read_bytes() == original, "拒绝新run时改写了既有metrics字节"
    assert env.reset_calls == 0 and not env.records, "既有指标闸门后仍发生了物理交互"
