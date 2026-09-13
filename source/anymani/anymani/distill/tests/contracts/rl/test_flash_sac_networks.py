r"""真实 FlashSAC 网络组合的 CPU 合同：信息边界、概率梯度与静态手型重建。

使用真实 structured/flash_mlp Actor 与真实双 Q；仅几何提供器替换为明确的冻结解析函数。
批次包含 9/12/16 DoF，canonical 轴固定为 16 JOINT、4 TIP、21 owner 与 History30。
小配置为 MLP/critic 宽度 16、各 1 个残差块、11 个价值原子；structured 保留生产宽度。
源码在独立 Python 包命名空间加载，包级环境注册被跳过，模型与数学实现不被替换。
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

# 尺寸表是测试的独立观测 ABI，不从被测拼接函数推导，以便捕获漏项和轴错位。
_DYNAMIC_SHAPES = {
    "actor_jnt_current": (16, 5),  # q/pi、目标/pi、上一动作、own-contact、TIP-contact
    "actor_jnt_history": (30, 16, 5),  # oldest-to-latest，最后一帧为当前状态
    "actor_owner_contact": (21, 1),  # 部署侧二元触觉
    "critic_jnt_state": (16, 4),  # 特权关节状态
    "critic_owner_contact": (21, 2),  # 特权接触力及接触位
    "critic_obj": (1, 15),  # 物体状态
    "critic_task": (1, 8),  # 任务状态
    "critic_reward_release": (1,),  # 奖励释放系数
}
_RUNTIME_PREFIXES = ("isaaclab", "isaacsim", "omni", "carb", "pxr")  # 本测试禁止新加载的物理运行时


class _RejectPhysicsImports(importlib.abc.MetaPathFinder):
    r"""在执行模块体之前拒绝物理运行时导入；纯 torch 神经网络照常加载。"""

    def find_spec(self, fullname, path=None, target=None):
        r"""只拦截运行时顶层包，其他导入交回标准查找器。"""
        if fullname.split(".")[0] in _RUNTIME_PREFIXES:  # 即使依赖意外新增环境 import，也不能初始化物理
            raise AssertionError(f"CPU network contract attempted physics import: {fullname}")
        # 隐式返回 None；数学与真实模型源码继续由标准查找器处理。


@pytest.fixture(scope="module")
def cpu_api():
    r"""载入真实源码但跳过 anymani/rl 包的环境注册，整个测试模块禁用 CUDA 初始化。

    仅将 networks/observations 的绝对模型导入指向同一份隔离加载的真实模型模块；
    所有 Linear、TCN、Transformer、BN、概率函数和网络 forward 均原样执行。
    """
    root = Path(__file__).resolve().parents[3]  # distill 根目录，与测试文件同属当前项目
    prefix = "_flash_sac_real_network_contract"  # 独立于其他合同测试的类/模块身份
    guard = _RejectPhysicsImports()  # 不允许导入时偷偷启动运行时依赖
    previous_modules = set(sys.modules)  # 用于确认整个验证没有新增物理模块
    sys.meta_path.insert(0, guard)  # 导入前设置屏障，而不是启动后再检查
    try:
        # 空包只提供实际源码搜索路径；不执行任何生产 __init__.py 的注册逻辑。
        for suffix, directory in (
            ("", root),  # 根命名空间
            (".models", root / "models"),  # 真实模型源码
            (".models.backbones", root / "models" / "backbones"),  # 真实图注意力源码
            (".flash_sac", root / "rl" / "flash_sac"),  # 真实 SAC 组合与数学源码
        ):
            package = ModuleType(prefix + suffix)  # 仅导入外壳，无运算替身
            package.__path__ = [str(directory)]  # 固定到当前项目，禁止意外加载另一份安装包
            sys.modules[package.__name__] = package  # dataclass 的模块身份必须可解析
        policy = importlib.import_module(prefix + ".models.palm_rotation_policy")  # 真实结构化模型
        with pytest.MonkeyPatch.context() as imports:
            for name in ("anymani", "anymani.distill", "anymani.distill.models"):
                package = ModuleType(name)  # 局部绝对导入外壳，退出上下文后恢复已有模块
                package.__path__ = []  # 禁止沿该外壳继续扫描环境包
                imports.setitem(sys.modules, name, package)  # 仅临时改 Python 模块映射
            imports.setitem(sys.modules, "anymani.distill.models.palm_rotation_policy", policy)  # 同一真实类
            api = SimpleNamespace(
                policy=policy,  # 真实 typed observation 与结构化 Actor
                config=importlib.import_module(prefix + ".flash_sac.config"),  # 真实配置校验
                observations=importlib.import_module(prefix + ".flash_sac.observations"),  # 真实净化和静态表
                networks=importlib.import_module(prefix + ".flash_sac.networks"),  # 两种真实 Actor 与双 Q
                probability=importlib.import_module(prefix + ".flash_sac.math"),  # 真实重参数概率函数
            )
        with pytest.MonkeyPatch.context() as devices:

            def reject_cuda(*args, **kwargs):
                r"""任何测试路径试图初始化 CUDA 都立即失败，不触及正在训练的 GPU。"""
                raise AssertionError("CPU network contract attempted CUDA initialization")

            devices.setattr(torch.cuda, "_lazy_init", reject_cuda)  # 仅影响本测试进程
            yield api  # 所有验证共享源码模块，但每个测试单独构造模型和张量
        loaded = set(sys.modules) - previous_modules  # 统计本测试新导入的模块
        assert not any(name.split(".")[0] in _RUNTIME_PREFIXES for name in loaded), "CPU 合同加载了物理运行时"
        assert not torch.cuda.is_initialized(), "CPU 网络验证不得创建 CUDA 上下文"
    finally:
        sys.meta_path.remove(guard)  # 还原导入路由，不干扰后续合同测试


def _small_config(api, variant="flash_mlp"):
    r"""仅改当前配置实例的网络容量；structured 骨架仍使用生产宽度。"""
    return api.config.FlashSACConfig(
        actor_variant=variant,  # 真实 structured 或 flash_mlp 分支
        actor_mlp_hidden_dim=16,  # CPU 小尺寸 MLP
        actor_mlp_num_blocks=1,  # 保留一个真实归一化残差块
        critic_hidden_dim=16,  # CPU 小尺寸独立双 Q
        critic_num_blocks=1,  # 保留一个真实集成残差块
        critic_bins=11,  # 支持仍覆盖配置中的 [-value_support, value_support]
    )


def _actor(api, variant):
    r"""构造真实 Actor，固定参数初始化且不污染其他测试的 CPU 随机状态。"""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(1701)  # 两种 Actor 的初始化均可复现
        return api.networks.FlashPalmActor(_small_config(api, variant))  # 不修补生产构造器


def _critic(api):
    r"""构造真实双 Q；参数独立抽样，所有计算位于 CPU。"""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(1702)  # 固定双 Q 初始化以稳定特权敏感性检查
        return api.networks.FlashPalmDoubleCritic(_small_config(api))  # 独立两组参数和运行统计


def _named_observation():
    r"""构造三种真实有效维数的 named tensor 输入，不调用环境或冻结模型。

    depth-major 槽位以每根手指的有效深度定义：$(3,3,3,0)$、$(4,3,3,2)$、$(4,4,4,4)$。
    这保证每根手指沿深度为连续前缀，实际 DoF 恰为 9/12/16；9 DoF 行含一个完全空手指。
    图是合法范围内的合成关系桶，只验证输入路由，不作为真实运动学图的物理证据。
    """
    rng = torch.Generator(device="cpu").manual_seed(1703)  # 数据采样不改变进程 RNG
    depths = torch.tensor([[3, 3, 3, 0], [4, 3, 3, 2], [4, 4, 4, 4]])  # [B=3, finger=4]
    joint = (torch.arange(4)[None, :, None] < depths[:, None, :]).reshape(3, 16)  # [B, depth*finger]
    tip = depths > 0  # 每根存在的手指具有一个有效 TIP
    owner = torch.cat((torch.ones(3, 1, dtype=torch.bool), joint, tip), dim=-1)  # [B, PALM+JOINT+TIP]
    data = {name: torch.randn(3, *shape, generator=rng) * 0.2 for name, shape in _DYNAMIC_SHAPES.items()}
    data["actor_jnt_current"].masked_fill_(~joint[..., None], 0)  # 规范化当前帧只保留物理关节
    data["actor_jnt_current"][..., 3] = 0  # TIP-only 基准下没有自身指节接触
    data["actor_jnt_history"].masked_fill_(~joint[:, None, :, None], 0)  # 历史 ghost 也为零
    data["actor_jnt_history"][..., 3] = 0  # 所有 30 帧均遵守 TIP-only 信息边界
    data["actor_jnt_history"][:, -1] = data["actor_jnt_current"]  # History30 包含当前帧
    data["actor_owner_contact"] = torch.zeros(3, 21, 1)  # 非 TIP owner 触觉清零
    data["actor_owner_contact"][:, 17:, 0] = tip.float()  # 有效 TIP 接触为 1
    data["critic_jnt_state"].masked_fill_(~joint[..., None], 0)  # 特权状态仍须屏蔽 ghost
    data["critic_owner_contact"].masked_fill_(~owner[..., None], 0)  # 特权接触可读真实非 TIP owner
    limits = torch.stack((-torch.ones(3, 16), torch.ones(3, 16)), dim=-1)  # qmin/pi、qmax/pi
    limits.mul_(1 + 0.1 * torch.arange(3)[:, None, None])  # 各资产限位不同，便于检查静态表路由
    data["actor_jnt_limits"] = limits.masked_fill(~joint[..., None], 0)  # [3,16,2]
    data.update(jnt_valid=joint, tip_valid=tip, owner_valid=owner)  # 同一 canonical 轴的三类 mask
    data["geometry_tokens"] = torch.randn(3, 21, 128, generator=rng).masked_fill(~owner[..., None], 0) * 0.2
    pair = owner[:, :, None] & owner[:, None, :]  # 仅两端均真实的 owner 关系合法
    distance = (torch.arange(21)[:, None] - torch.arange(21)[None, :]).abs().clamp_max(8)  # 合法距离桶
    for index, name in enumerate(("shortest_path", "parent_direction", "child_direction")):
        data[name] = (distance + index).clamp_max(8).expand(3, -1, -1).clone().masked_fill(~pair, 0)  # [3,21,21]
    return data  # 所有浮点张量 FP32，所有张量位于 CPU


def _assert_same_sample(actual, expected):
    r"""信息不变性必须覆盖动作、中心、两种概率归约、尺度与有效 DoF。"""
    for name in ("actions", "mean_action", "log_prob", "log_prob_per_active", "log_std", "active_count"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0, msg=name)  # 精确不变


class _ActorBoundaryView(Mapping):
    r"""即使输出偶然相同，也禁止 Actor 读取特权字段或资产索引 metadata。"""

    def __init__(self, data):
        r"""保存完整观测，并记录真实读取的字段名。"""
        self.data = data  # 包含任意额外 metadata 的原始字典
        self.reads = set()  # 只用于测试观察信息依赖，不参与网络运算

    def __getitem__(self, key):
        r"""Mapping.get 等接口也经此方法，避免 dict.get 绕开读取审计。"""
        assert not key.startswith("critic_") and key not in {"asset_indices", "asset_id", "metadata"}, key
        self.reads.add(key)  # 记录合法访问，用于确认断言实际覆盖了真实前向
        return self.data[key]  # 返回原始张量，不替换或伪造网络输入

    def __iter__(self):
        r"""暴露完整键集合；额外字段不能因迭代而进入 Actor。"""
        return iter(self.data)  # 与调用方真实字典具有相同字段集合

    def __len__(self):
        r"""返回完整观测字段数。"""
        return len(self.data)  # 元数据不属于数值特征


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_real_actor_constructor_and_capacity_contract(cpu_api, variant):
    r"""必须能调用真实生产构造器；structured 保留 128 维 contextual JOINT 读出。"""
    actor = _actor(cpu_api, variant)  # 构造错误直接失败，不以替代模型绕开
    assert isinstance(actor, cpu_api.networks.FlashPalmActor), "必须验证真实 SAC Actor"
    assert all(parameter.device.type == "cpu" for parameter in actor.parameters()), "Actor 参数必须在 CPU"
    if variant == "structured":  # TCN/整手图结构保持生产容量
        assert actor.body.direct_head[1].in_features == 128, "structured 的 contextual token 宽度必须保持 128"
    else:  # 小容量只作用于 MLP 分支
        assert actor.embedder.w.w.weight.shape == (16, 6585), "MLP 应读取完整合法 6585D 输入"
        assert len(actor.blocks) == 1, "小配置仍包含一个真实归一化残差块"


def test_flat_dimensions_and_real_concatenation_order(cpu_api):
    r"""独立拼接检查 $6585=80+2400+32+21+2688+3\cdot441+16+4+21$，特权增量为 130。"""
    data = _named_observation()  # 基准已清除 own-contact 与 ghost
    actor_fields = (
        "actor_jnt_current",  # 80D 当前控制信号
        "actor_jnt_history",  # 2400D History30
        "actor_jnt_limits",  # 32D 归一化限位
        "actor_owner_contact",  # 21D TIP-only 触觉槽
        "geometry_tokens",  # 2688D 冻结 owner 几何
        "shortest_path",  # 441D 距离关系
        "parent_direction",  # 441D 父方向关系
        "child_direction",  # 441D 子方向关系
        "jnt_valid",  # 16D 可控关节 mask
        "tip_valid",  # 4D 手指存在性 mask
        "owner_valid",  # 21D 几何实体 mask
    )
    privileged_fields = (
        "critic_jnt_state",  # 64D 特权关节状态
        "critic_owner_contact",  # 42D 特权接触信号
        "critic_obj",  # 15D 物体状态
        "critic_task",  # 8D 任务状态
        "critic_reward_release",  # 1D 奖励释放状态
    )  # 64+42+15+8+1=130
    actor_reference = torch.cat([data[key].reshape(3, -1).float() for key in actor_fields], dim=-1)  # 手工字段顺序
    critic_reference = torch.cat(
        [actor_reference, *[data[key].reshape(3, -1).float() for key in privileged_fields]], dim=-1
    )  # 特权字段只能出现在 Critic 后缀
    assert actor_reference.shape == (3, cpu_api.observations.ACTOR_FLAT_DIM) == (3, 6585), "Actor 拼接宽度"
    assert critic_reference.shape == (3, cpu_api.observations.CRITIC_FLAT_DIM) == (3, 6715), "Critic 拼接宽度"
    torch.testing.assert_close(  # 独立字段列表验证 Actor 实际拼接顺序
        cpu_api.observations.actor_flat_features(data), actor_reference, atol=0, rtol=0
    )  # 内容顺序
    torch.testing.assert_close(  # Critic 只能在合法输入后追加特权后缀
        cpu_api.observations.critic_flat_features(data), critic_reference, atol=0, rtol=0
    )  # 内容顺序


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
@pytest.mark.parametrize("training", [False, True])
def test_actor_mixed_dof_deterministic_center_and_rng(cpu_api, variant, training):
    r"""同批 9/12/16 DoF 正常前向；确定性返回 bounded center 且不消耗 RNG。"""
    actor, data = _actor(cpu_api, variant), _named_observation()  # B=3，训练 BN 有足够样本
    actor.train(not training)  # 显式模式不能由外层 Module.training 意外覆盖
    rng_before = torch.get_rng_state().clone()  # 仅比较前向自身的随机状态消耗
    sample = actor(data, training=training, deterministic=True)  # 完整真实 Actor 前向
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)  # 确定性路径不采样
    valid = data["jnt_valid"]  # 物理动作维，非统一 16 DoF 假设
    torch.testing.assert_close(sample.active_count, torch.tensor([9, 12, 16]), atol=0, rtol=0)  # 精确 DoF
    torch.testing.assert_close(sample.actions, sample.mean_action, atol=0, rtol=0)  # 保留原始有界中心
    assert sample.actions.shape == sample.log_std.shape == (3, 16), "动作和逐关节尺度必须为 [B,16]"
    assert sample.log_prob.shape == sample.log_prob_per_active.shape == (3,), "两种概率归约都必须为 [B]"
    assert torch.isfinite(sample.log_prob).all() and torch.all(sample.actions.abs() <= 1), "中心及密度必须有限有界"
    torch.testing.assert_close(  # ghost 不产生物理动作
        sample.actions[~valid], torch.zeros_like(sample.actions[~valid]), atol=0, rtol=0
    )  # ghost 零
    torch.testing.assert_close(  # 输出中的无效尺度采用规范占位值
        sample.log_std[~valid], torch.zeros_like(sample.log_std[~valid]), atol=0, rtol=0
    )  # ghost 占位
    torch.testing.assert_close(sample.log_prob_per_active, sample.log_prob / valid.sum(-1))  # 联合密度与熵控制的区别


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_actor_does_not_read_privilege_or_asset_metadata(cpu_api, variant):
    r"""固定合法张量后，特权 NaN、任意资产索引和非张量 metadata 均不能影响 Actor。"""
    actor, data = _actor(cpu_api, variant), _named_observation()  # 固定同一合法物理状态
    expected = actor(data, deterministic=True)  # 完整观测下的基准输出
    actor_only = {key: value for key, value in data.items() if not key.startswith("critic_")}  # 特权字段可完全省略
    _assert_same_sample(actor(actor_only, deterministic=True), expected)  # 不允许隐式要求特权键存在
    contaminated: dict[str, object] = dict(data)  # 额外 metadata 可为非张量，合法张量仍为相同对象
    contaminated.update(  # 真正不可读的特权张量，任何泄漏都会产生 NaN
        {key: torch.full_like(value, float("nan")) for key, value in data.items() if key.startswith("critic_")}
    )
    contaminated.update(  # 资产 ID 与 metadata 不能进入数值拼接
        asset_indices=torch.tensor([-7, 10**9, 1]), asset_id="not-a-feature", metadata={"privilege": object()}
    )
    view = _ActorBoundaryView(contaminated)  # 同时检测读取依赖和数值不变性
    _assert_same_sample(actor(view, deterministic=True), expected)  # NaN 特权不进入任何 Actor 算子
    assert "actor_jnt_history" in view.reads and "geometry_tokens" in view.reads, "必须实际执行合法历史与几何路径"


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_actor_tip_only_contact_boundary(cpu_api, variant):
    r"""所有 current/history 的第 3 通道与前 17 个 owner 接触都属于非 TIP 信息。"""
    actor, data = _actor(cpu_api, variant), _named_observation()  # 同一策略与合法观测
    polluted = {key: value.clone() for key, value in data.items()}  # 不改变基准输入
    polluted["actor_jnt_current"][..., 3] = float("nan")  # 真实关节自身触觉也必须被隔离
    polluted["actor_jnt_history"][..., 3] = float("inf")  # 全部历史时刻都遵守 TIP-only
    polluted["actor_owner_contact"][:, :17] = 1e9  # PALM/JOINT 接触不能混入部署 Actor
    noise = torch.linspace(-0.2, 0.2, 48).reshape(3, 16)  # 固定同一重参数路径
    _assert_same_sample(actor(polluted, noise=noise), actor(data, noise=noise))  # 检查尺度和概率，不只检查均值


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
@pytest.mark.parametrize("pollution", [1e9, float("nan"), float("inf")])
def test_actor_ghost_dynamic_geometry_and_graph_pollution(cpu_api, variant, pollution):
    r"""ghost 数值和涉及 ghost 的图桶可被污染，合法动作/概率仍应逐元素相同。"""
    actor, data = _actor(cpu_api, variant), _named_observation()  # 9 DoF 行同时含 ghost JOINT/TIP
    polluted = {key: value.clone() for key, value in data.items()}  # 污染前保留独立 oracle
    joint, owner = data["jnt_valid"], data["owner_valid"]  # 两类物理槽位掩码
    for key in ("actor_jnt_current", "actor_jnt_limits"):
        polluted[key].masked_fill_(~joint[..., None], pollution)  # 逐关节动态及限位
    polluted["actor_jnt_history"].masked_fill_(~joint[:, None, :, None], pollution)  # 30 帧全部覆盖
    for key in ("actor_owner_contact", "geometry_tokens"):
        polluted[key].masked_fill_(~owner[..., None], pollution)  # 包括完全空手指的 TIP token
    pair = owner[:, :, None] & owner[:, None, :]  # 任一端无效的关系都不可被读取
    for key in ("shortest_path", "parent_direction", "child_direction"):
        polluted[key].masked_fill_(~pair, 10**9)  # 非法 embedding 桶必须先净化，不依赖 clamp 挽救
    noise = torch.linspace(-0.2, 0.2, 48).reshape(3, 16)  # 固定 stochastic 路径进行精确比较
    _assert_same_sample(actor(polluted, noise=noise), actor(data, noise=noise))  # 连联合密度也必须不变


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_actor_reparameterized_probability_and_ghost_gradients(cpu_api, variant):
    r"""验证真实 Actor 的 tanh 密度与重参数梯度，并检查被隔离输入的梯度为零。

    $z=\mu+\sigma\epsilon$，$\log\pi=\sum_{j\in valid}[\log\mathcal N(z_j;\mu_j,\sigma_j)-\log(1-a_j^2)]$。
    $\partial a_j/\partial\epsilon_j=\sigma_j(1-a_j^2)$，
    $\partial\log\pi/\partial\epsilon_j=-\epsilon_j+2\sigma_j a_j$；ghost 的两项导数均为 0。
    用小幅固定噪声和 FP64 概率 oracle 避免直接 log(1-a²) 的浮点饱和。
    """
    actor, data = _actor(cpu_api, variant), _named_observation()  # 真实骨架保留自动微分
    differentiable = (
        "actor_jnt_current",  # 当前控制帧的合法/ghost 雅可比
        "actor_jnt_history",  # 全部历史帧的雅可比
        "actor_jnt_limits",  # 静态限位也必须屏蔽 ghost
        "actor_owner_contact",  # 只允许 TIP 接触产生梯度
        "geometry_tokens",  # 冻结输入的数值雅可比，不把 provider 纳入优化器
    )
    for key in differentiable:
        data[key].requires_grad_()  # 用输入雅可比证明信息边界，而非只依赖一次输出相等
    valid = data["jnt_valid"]  # [3,16]，计数为 9/12/16
    noise = torch.linspace(-0.15, 0.15, 48).reshape(3, 16).masked_fill(~valid, float("nan")).requires_grad_()
    rng_before = torch.get_rng_state().clone()  # 显式噪声也不得消费额外随机数
    sample = actor(data, noise=noise)  # 可微重参数采样，不使用确定性分支
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)  # 噪声来源完全由 caller 指定
    center = sample.mean_action.detach().double()  # bounded center 的独立概率 oracle
    mean = torch.atanh(center.clamp(-1 + 1e-6, 1 - 1e-6))  # structured 的数值保护与生产合同相同
    std = sample.log_std.detach().double().exp()  # 潜 Gaussian 尺度，不是 tanh 后动作标准差
    safe_noise = noise.detach().double().masked_fill(~valid, 0)  # oracle 同样只积分真实关节
    latent = mean + std * safe_noise  # FP64 减小 atanh/tanh 相消误差
    expected_actions = latent.tanh().masked_fill(~valid, 0)  # 有界样本，ghost 精确零
    reference_logp = torch.distributions.Normal(mean, std).log_prob(latent) - torch.log1p(-latent.tanh().square())
    reference_logp = reference_logp.masked_fill(~valid, 0).sum(-1)  # 联合密度，不能除以 canonical 16
    torch.testing.assert_close(sample.actions.double(), expected_actions, atol=2e-5, rtol=2e-5)  # FP32/64 往返容差
    torch.testing.assert_close(sample.log_prob.double(), reference_logp, atol=2e-4, rtol=2e-5)  # 完整 Jacobian 修正
    da = torch.autograd.grad(sample.actions.sum(), noise, retain_graph=True)[0]  # 逐噪声方向的动作梯度
    dlogp = torch.autograd.grad(sample.log_prob.sum(), noise, retain_graph=True)[0]  # 密度的重参数路径梯度
    expected_da = (sample.log_std.exp() * (1 - sample.actions.square())).masked_fill(~valid, 0)  # sigma*(1-a²)
    expected_dlogp = (-noise.masked_fill(~valid, 0) + 2 * sample.log_std.exp() * sample.actions).masked_fill(~valid, 0)
    torch.testing.assert_close(da, expected_da, atol=2e-6, rtol=2e-5)  # 有效/ghost 两类噪声梯度
    torch.testing.assert_close(dlogp, expected_dlogp, atol=2e-5, rtol=2e-5)  # -epsilon+2*sigma*a
    loss = (sample.actions * torch.linspace(0.1, 0.9, 16)).sum() + 0.03 * sample.log_prob.sum()  # 非均匀损失余切
    loss.backward()  # 完整 Actor 参数、输入与噪声的反向传播
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in actor.parameters()), "Actor 必须可训练"
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()), "所有 Actor 梯度必须有限"
    for key in differentiable:
        gradient = data[key].grad  # 输入字段的真实雅可比投影
        assert gradient is not None and torch.isfinite(gradient).all(), f"{variant}: {key} 梯度必须存在且有限"
        if key == "actor_jnt_history":  # 历史额外包含时间轴
            mask = valid[:, None, :, None]  # [B,1,J,1] -> [B,T,J,D]
        elif key in {"actor_owner_contact", "geometry_tokens"}:  # owner 字段涵盖 PALM/JOINT/TIP
            mask = data["owner_valid"][..., None]  # [B,owner,1]
        else:  # 当前帧和限位只具有 JOINT 轴
            mask = valid[..., None]  # [B,J,1]
        assert torch.count_nonzero(gradient.masked_select(~mask.expand_as(gradient))) == 0, f"{key}: ghost 梯度泄漏"
    current_gradient = data["actor_jnt_current"].grad  # 当前帧梯度 [B,J,5]
    history_gradient = data["actor_jnt_history"].grad  # 历史梯度 [B,30,J,5]
    contact_gradient = data["actor_owner_contact"].grad  # owner 接触梯度 [B,21,1]
    assert current_gradient is not None, "当前帧梯度缺失"  # 不能用 detach 伪造输入信息隔离
    assert history_gradient is not None, "历史帧梯度缺失"  # History30 保留真实自动微分路径
    assert contact_gradient is not None, "owner 触觉梯度缺失"  # TIP 触觉路径必须仍连接到策略
    assert torch.count_nonzero(current_gradient[..., 3]) == 0, "current own-contact 梯度必须为零"
    assert torch.count_nonzero(history_gradient[..., 3]) == 0, "history own-contact 梯度必须为零"
    assert torch.count_nonzero(contact_gradient[:, :17]) == 0, "非 TIP owner 接触梯度必须为零"


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_actor_implicit_random_sampling_is_replayable(cpu_api, variant):
    r"""省略 noise 时真实采样会推进 RNG，恢复同一 CPU 状态后可精确复现。"""
    actor, data = _actor(cpu_api, variant), _named_observation()  # eval 保持 BN 状态只读
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1704)  # 固定一次标准正态采样实验
        before = torch.get_rng_state().clone()  # 抽样前状态
        first = actor(data)  # 默认 stochastic=True 的真实采样
        after = torch.get_rng_state().clone()  # 抽样后的随机状态
        assert not torch.equal(before, after), "随机分支必须真正产生噪声"
        torch.set_rng_state(before)  # 同一概率模型、同一噪声序列
        _assert_same_sample(actor(data), first)  # 所有输出均精确重放
    assert torch.all(first.actions.abs() <= 1), "随机动作仍必须落在归一动作范围"
    assert torch.count_nonzero(first.actions[~data["jnt_valid"]]) == 0, "随机 ghost 动作必须为零"


@pytest.mark.parametrize("training", [False, True])
def test_double_q_distribution_shapes_probability_and_expected_value(cpu_api, training):
    r"""真实双 Q 输出 $[2,B]$ 和 $[2,B,K]$；价值必须等于同一支撑上的概率期望。"""
    critic, data = _critic(cpu_api), _named_observation()  # Q=2、B=3、K=11
    critic.train(not training)  # 显式 training 控制归一化统计
    actions = torch.linspace(-0.8, 0.8, 48).reshape(3, 16)  # 包含任意 ghost 动作，Q 应先清零
    values, log_probs = critic(data, actions, training=training)  # 真实 categorical Q 前向
    assert values.shape == (2, 3) and log_probs.shape == (2, 3, 11), "双 Q 必须保留 Q/B/K 三条轴"
    probabilities = log_probs.exp()  # [2,3,11]，每条 categorical 分布独立归一化
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3), atol=5e-7, rtol=5e-7)  # 概率质量守恒
    expected = (probabilities * critic.support[None, None, :]).sum(-1)  # E[Z]，不是 argmax 原子
    torch.testing.assert_close(values, expected, atol=1e-7, rtol=1e-6)  # 网络返回的同一分布期望
    torch.testing.assert_close(critic.support, torch.linspace(-5, 5, 11), atol=0, rtol=0)  # 仅 bins 缩小，物理支持未变
    assert critic.embedder.w.weight.shape == (2, 16, 6731), "Critic 输入为 6715D 状态加 16D 动作"


@pytest.mark.parametrize(
    "field", ["critic_jnt_state", "critic_owner_contact", "critic_obj", "critic_task", "critic_reward_release"]
)
def test_critic_really_reads_each_privileged_field(cpu_api, field):
    r"""每类特权输入均应有非零 Q 梯度；仅给 Actor 添加不可见字段不能替代真实 Critic 读取。"""
    critic, data = _critic(cpu_api), _named_observation()  # 固定网络与合法状态
    data[field].requires_grad_()  # 单独选择一类特权证据
    actions = torch.zeros(3, 16)  # 固定物理动作，隔离状态依赖
    values, _ = critic(data, actions)  # 期望 Q 的真实输入依赖
    coefficients = torch.tensor([[0.2, -0.7, 1.1], [0.9, 0.3, -0.4]])  # 防止对称求和抵消梯度
    gradient = torch.autograd.grad((values * coefficients).sum(), data[field])[0]  # dL/d(privilege)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0, f"Critic 未读取 {field}"
    changed = dict(data)  # 仅沿该特权字段的敏感方向改变状态
    changed[field] = data[field].detach() + 0.05 * gradient.sign()  # 小幅有向扰动
    changed_values, _ = critic(changed, actions)  # 真实前向变化证明依赖可观察
    assert not torch.equal(changed_values, values), f"改变 {field} 后 Q 完全不变"


@pytest.mark.parametrize("pollution", [1e9, float("nan"), float("inf")])
def test_critic_ignores_ghost_actions_in_values_and_gradients(cpu_api, pollution):
    r"""Q 的有效动作梯度与输出不依赖 ghost 填充值，ghost 本身的 Q 梯度严格为零。"""
    critic, data = _critic(cpu_api), _named_observation()  # 真实动作条件双 Q
    valid = data["jnt_valid"]  # 每行不同的物理动作子空间
    clean = torch.linspace(-0.4, 0.4, 48).reshape(3, 16).masked_fill(~valid, 0).requires_grad_()  # 有效动作固定
    dirty = clean.detach().masked_fill(~valid, pollution).requires_grad_()  # 仅污染不产生物理作用的维度
    expected_q, expected_logp = critic(data, clean)  # 干净动作 oracle
    actual_q, actual_logp = critic(data, dirty)  # 真实 Q 的 mask 入口
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)  # 期望价值不变
    torch.testing.assert_close(actual_logp, expected_logp, atol=0, rtol=0)  # 完整价值分布也不变
    clean_gradient = torch.autograd.grad(expected_q.square().sum(), clean)[0]  # 动作条件的参考导数
    dirty_gradient = torch.autograd.grad(actual_q.square().sum(), dirty)[0]  # 污染后的导数
    torch.testing.assert_close(dirty_gradient, clean_gradient, atol=0, rtol=0)  # 所有有效动作梯度保持一致
    assert torch.isfinite(dirty_gradient).all() and torch.count_nonzero(dirty_gradient[~valid]) == 0, "ghost Q 梯度泄漏"
    assert torch.count_nonzero(dirty_gradient[valid]) > 0, "Q 必须真实依赖有效动作"


@pytest.mark.parametrize("variant", ["structured", "flash_mlp"])
def test_actor_critic_parameter_storage_and_gradient_ownership_are_disjoint(cpu_api, variant):
    r"""输入可以共享，Actor/Critic 的 Parameter 身份、底层存储与优化梯度均须独立。"""
    actor, critic, data = _actor(cpu_api, variant), _critic(cpu_api), _named_observation()  # 两个独立真实网络
    actor_parameters, critic_parameters = list(actor.parameters()), list(critic.parameters())  # 包括所有可训练组件
    assert {id(p) for p in actor_parameters}.isdisjoint({id(p) for p in critic_parameters}), "Parameter 对象共享"
    actor_storage = {p.data_ptr() for p in actor_parameters}  # 独立 Parameter 对象也可能持有同一块存储
    critic_storage = {p.data_ptr() for p in critic_parameters}  # 检查底层可学习权重的别名
    assert actor_storage.isdisjoint(critic_storage), "Actor/Critic 参数存储共享"
    sample = actor(data, deterministic=True)  # 验证 Actor 自身梯度归属
    sample.actions.square().sum().backward()  # 仅策略目标
    assert all(p.grad is None for p in critic_parameters), "Actor backward 写入了 Critic 参数梯度"
    actor.zero_grad(set_to_none=True)  # 清除上一条反向路径，隔离 Critic 目标
    values, _ = critic(data, sample.actions.detach())  # 固定动作时 Q 更新不反传 Actor
    values.square().sum().backward()  # 仅 Critic 目标
    assert all(p.grad is None for p in actor_parameters), "Critic backward 写入了 Actor 参数梯度"
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in critic_parameters), "Critic 必须可训练"


class _FrozenQGeometry(torch.nn.Module):
    r"""冻结解析几何替身：按实际 $q=\pi\,current[...,0]$ 生成可追踪的 owner tokens。

    每个 owner 角度标量乘固定 128 通道系数；PALM 取全手角度和，TIP 取所属手指角度和。
    资产索引只模拟静态形态偏移，明确属于 provider 的查表输入，不作为 Actor 的额外神经输入。
    """

    channel: torch.Tensor  # 只读的固定通道系数 buffer [128]

    def __init__(self, geometry_type):
        r"""只注册常量 buffer，不含可训练参数或物理设备句柄。"""
        super().__init__()
        self.geometry_type = geometry_type  # 真实 PalmRotationGeometry 数据合同
        self.register_buffer("channel", torch.arange(1, 129, dtype=torch.float32) / 128)  # 无量纲固定通道系数
        self.calls = []  # 记录查询的资产、q、shape 和梯度模式，供独立断言核验

    def resolve(self, indices, actor):
        r"""从当前帧而非历史缓存读取 q，返回同设备 $[B,21,128]$ token。"""
        q = actor.jnt_current[..., 0].masked_fill(~actor.jnt_valid, 0) * math.pi  # q/pi -> rad
        self.calls.append(  # 记录真实的 q 解析量、路由索引和冻结调用边界
            (indices.clone(), q.detach().clone(), tuple(actor.jnt_history.shape), torch.is_grad_enabled())
        )
        owner_q = torch.cat((q.sum(-1, keepdim=True), q, q.reshape(-1, 4, 4).sum(1)), dim=-1)  # PALM/JOINT/TIP
        tokens = owner_q[..., None] * self.channel + 0.125 * indices[:, None, None]  # q 与静态形态共同确定几何
        tokens = tokens.masked_fill(~actor.owner_valid[..., None], 0)  # 无效实体没有几何信号
        graph = torch.zeros(indices.numel(), 21, 21, dtype=torch.long)  # provider 图不接管 bank 中的静态图
        return self.geometry_type(tokens, actor.owner_valid, graph, graph.clone(), graph.clone())  # 真实 typed 输出


def _live_replicas():
    r"""三资产各两个交错副本；静态证据相同，同资产动态状态允许不同。"""
    base = _named_observation()  # 行 0/1/2 分别拥有 9/12/16 DoF
    indices = torch.tensor([2, 0, 1, 2, 0, 1])  # 非分组顺序，防止依赖每资产连续存储
    live = {key: value[indices].clone() for key, value in base.items()}  # 同资产复制相同静态值
    live["actor_jnt_current"][3:, :, 0] += 0.05 * live["jnt_valid"][3:]  # 第二副本处于不同 q
    live["actor_jnt_history"][:, -1] = live["actor_jnt_current"]  # 仍保持 History30 包含当前帧
    return base, live, indices  # base 是按资产编号排序的独立静态表 oracle


def test_static_bank_builds_one_immutable_entry_per_asset(cpu_api):
    r"""静态表按资产去副本，允许副本动态不同；输入与导出的 state_dict 均不应别名修改表。"""
    base, live, indices = _live_replicas()  # 六环境、三资产
    bank = cpu_api.observations.StaticHandBank.from_live(live, indices)  # 真实一致性检查与静态抽取
    assert bank.asset_count == 3, "静态轴是资产数，不是环境副本数"
    for key in cpu_api.observations.STATIC_FIELDS:
        torch.testing.assert_close(bank.fields[key], base[key], atol=0, rtol=0, msg=key)  # 按真实资产索引恢复
        assert bank.fields[key].device.type == "cpu" and not bank.fields[key].requires_grad, key
    exported = bank.state_dict()  # 用户可能修改导出的检查点字典
    exported["actor_jnt_limits"].add_(5)  # 修改副本不应反向改写冻结静态表
    live["actor_jnt_limits"].add_(7)  # 采集侧后续内存复用也不能改写 bank
    torch.testing.assert_close(bank.fields["actor_jnt_limits"], base["actor_jnt_limits"], atol=0, rtol=0)  # 独立存储


@pytest.mark.parametrize(
    "field",
    [
        "actor_jnt_limits",  # 相同资产的静态运动范围
        "jnt_valid",  # 关节存在性
        "tip_valid",  # 手指存在性
        "owner_valid",  # 几何实体存在性
        "shortest_path",  # 无向关系
        "parent_direction",  # 父方向关系
        "child_direction",  # 子方向关系
    ],
)
def test_static_bank_rejects_replica_static_disagreement(cpu_api, field):
    r"""同一资产的任意静态字段不一致，都必须在建立 bank 时拒绝并指出字段名。"""
    _, live, indices = _live_replicas()  # 行 0 与行 3 都属于资产 2
    value = live[field]  # 测试覆盖限位、三类 mask、三类图
    if value.dtype == torch.bool:  # mask 的差异为一个真实有效性位
        value[3].reshape(-1)[0].logical_not_()  # 只改第二副本，第一副本作为静态 oracle
    else:  # 图或限位的差异为一个标量
        value[3].reshape(-1)[0] += 1  # 不能以近似静态相等接受该差异
    with pytest.raises(ValueError, match=field):  # 错误必须给出具体静态字段
        cpu_api.observations.StaticHandBank.from_live(live, indices)


@pytest.mark.parametrize("invalid", ["negative_alias", "fractional", "missing_asset"])
def test_static_bank_from_live_rejects_invalid_asset_indices(cpu_api, invalid):
    r"""资产编号不能由负索引别名或浮点截断产生；也不允许漏掉声明区间中的资产。"""
    _, live, indices = _live_replicas()  # 第一行是资产 2，另一合法资产 2 副本位于第 3 行
    if invalid == "negative_alias":  # -1 在 Python 索引中会误别名到最后一个资产
        indices[0] = -1  # 其余 0/1/2 都存在，专门检测负索引悄悄通过一致性检查
    elif invalid == "fractional":  # 2.5 不是资产 2，不能通过 .long() 截断接受
        indices = indices.float()  # 其余数值不变，只破坏一个索引的整数语义
        indices[0] = 2.5  # 单个非法查表编号
    else:  # 0 和 2 存在，声明区间中资产 1 缺失
        indices[indices == 1] = 2  # from_live 必须拒绝不完整静态表
    with pytest.raises((ValueError, TypeError)):  # 不能隐式改变资产身份后继续建表
        cpu_api.observations.StaticHandBank.from_live(live, indices)


def test_static_bank_assemble_uses_current_q_and_preserves_dynamic_history(cpu_api):
    r"""重建按请求资产顺序查静态表；provider 在 no_grad 下读取当前 q，不能复用旧几何。"""
    base, live, live_indices = _live_replicas()  # 每资产已有经过副本核对的静态证据
    bank = cpu_api.observations.StaticHandBank.from_live(live, live_indices)  # 真实静态表
    indices = torch.tensor([2, 0, 2, 1])  # 重排并重复某资产，batch=4 与 bank 资产数不同
    dynamic = {key: base[key][indices].clone().requires_grad_() for key in _DYNAMIC_SHAPES}  # 包含回放恢复的 History30
    before = {key: value.detach().clone() for key, value in dynamic.items()}  # 检查 assemble 不原位修改输入
    provider = _FrozenQGeometry(cpu_api.policy.PalmRotationGeometry).eval()  # 无训练参数的 CPU 几何提供器
    assembled = bank.assemble(dynamic, indices, provider, device="cpu")  # 真实动态+静态+冻结几何重建
    assert len(list(provider.parameters())) == 0 and len(provider.calls) == 1, "每批恰调用一次冻结 provider"
    routed_indices, q, history_shape, grad_enabled = provider.calls[0]  # 记录真实 resolve 的输入
    torch.testing.assert_close(routed_indices, indices, atol=0, rtol=0)  # 查表索引顺序不得重排
    expected_q = base["actor_jnt_current"][indices, :, 0] * math.pi  # current 的 q/pi 转成真实 q
    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)  # 不是用 history 中的旧帧计算几何
    assert history_shape == (4, 30, 16, 5) and not grad_enabled, "冻结 provider 的历史 shape/no_grad 合同"
    assert assembled["geometry_tokens"].shape == (4, 21, 128), "冻结几何必须恢复 canonical owner 宽度"
    assert not assembled["geometry_tokens"].requires_grad, "provider 输出必须切断自动微分图"
    for key, shape in _DYNAMIC_SHAPES.items():
        assert assembled[key].shape == (4, *shape), f"动态字段 {key} shape 错误"
        torch.testing.assert_close(assembled[key], before[key], atol=0, rtol=0, msg=key)  # 动态值和历史顺序保留
        torch.testing.assert_close(dynamic[key], before[key], atol=0, rtol=0, msg=key)  # 输入不被原位改写
    for key in cpu_api.observations.STATIC_FIELDS:
        torch.testing.assert_close(assembled[key], base[key][indices], atol=0, rtol=0, msg=key)  # 同一资产静态查表
    torch.testing.assert_close(assembled["actor_jnt_history"][:, -1], assembled["actor_jnt_current"], atol=0, rtol=0)
    actor_input, geometry = cpu_api.observations.actor_inputs(assembled)  # 重建结果能进入真实 Actor typed contract
    assert actor_input.jnt_current.shape == (4, 16, 5) and geometry.tokens.shape == (4, 21, 128), "typed 重建 shape"
    changed = {key: value.detach().clone() for key, value in dynamic.items()}  # 同资产、不同当前关节状态
    changed["actor_jnt_current"][..., 0] += 0.1 * assembled["jnt_valid"]  # 每个真实关节改变 0.1*pi rad
    changed["actor_jnt_history"][:, -1] = changed["actor_jnt_current"]  # 保持有效的 History30 末帧
    second = bank.assemble(changed, indices, provider, device="cpu")  # 必须再次按新的 q 查询
    assert len(provider.calls) == 2, "每个重建批次必须重新按 q 查询冻结 provider"
    assert not torch.equal(second["geometry_tokens"], assembled["geometry_tokens"]), "当前 q 改变后几何缓存 stale"
    expected_delta = 0.1 * math.pi * assembled["jnt_valid"] / 128  # JOINT token 第 0 通道的解析增量
    torch.testing.assert_close(
        second["geometry_tokens"][:, 1:17, 0] - assembled["geometry_tokens"][:, 1:17, 0],  # 实际 JOINT 几何增量
        expected_delta,  # q/pi -> rad -> 第 0 通道系数 1/128
        atol=3e-8,  # FP32 静态偏移与角度项相减的舍入误差
        rtol=2e-5,
    )  # 证明 provider 使用当前 q，并保持明确的 q/pi 量纲转换


@pytest.mark.parametrize(
    "indices",  # 冻结静态表的资产查询编号
    [[-1, 0, 1], [3, 0, 1], [-0.5, 0, 1], [1.5, 0, 2]],  # 后两种数值截断后会错误落入有效区间
    ids=["negative_integer", "out_of_range", "negative_fraction", "positive_fraction"],  # 失败证据按根因命名
)
def test_static_bank_assemble_rejects_invalid_indices_before_geometry(cpu_api, indices):
    r"""越界整数与小数索引均需在 provider 查询前拒绝；负小数不能截断为资产 0。"""
    base, live, live_indices = _live_replicas()  # 合法静态表的资产编号为 0/1/2
    bank = cpu_api.observations.StaticHandBank.from_live(live, live_indices)  # 三资产冻结表
    dynamic = {key: base[key].clone() for key in _DYNAMIC_SHAPES}  # 其他所有输入合法
    provider = _FrozenQGeometry(cpu_api.policy.PalmRotationGeometry).eval()  # 用调用次数检查拒绝时机
    with pytest.raises((ValueError, TypeError)):  # 不接受 int(-0.5)=0 或 int(1.5)=1 的静默身份替换
        bank.assemble(dynamic, torch.tensor(indices), provider, device="cpu")
    assert not provider.calls, "非法资产索引不能进入冻结几何查询"


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("actor_jnt_history", (3, 29, 16, 5)),  # 时间轴必须为 30
        ("actor_jnt_current", (3, 16, 4)),  # 当前控制帧必须含五通道
        ("actor_jnt_history", (4, 30, 16, 5)),  # 历史批轴必须与静态查表的请求批次相同
    ],
)
def test_static_bank_rejects_malformed_dynamic_before_geometry(cpu_api, field, shape):
    r"""动态当前帧和 History30 必须满足真实 typed shape 合同，再交付冻结几何查询。"""
    base, live, live_indices = _live_replicas()  # 三资产、六环境的有效静态来源
    bank = cpu_api.observations.StaticHandBank.from_live(live, live_indices)  # 构造合法 bank
    dynamic = {key: base[key].clone() for key in _DYNAMIC_SHAPES}  # 请求 B=3 的动态批次
    dynamic[field] = torch.zeros(shape)  # 每次只扰动一种物理轴约定
    provider = _FrozenQGeometry(cpu_api.policy.PalmRotationGeometry).eval()  # 正确形状的提供器本身不变
    with pytest.raises(ValueError, match="shape"):  # 真实 PalmRotationActorObservation 负责明确的轴检查
        bank.assemble(dynamic, torch.arange(3), provider, device="cpu")
    assert not provider.calls, "非法动态/history 形状不能进入冻结几何查询"
