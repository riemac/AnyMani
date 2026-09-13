r"""真实 FlashSACLearner 的独立 CPU 合同：时序、状态边界与随机重放。

语义参照：Holiday Robotics / FlashRL（MIT），flash_rl/agents/flashSAC/agent.py
的 _update_networks：Actor → temperature → critic → target EMA。
使用 B=4、完整 History30/21-owner/128维几何具名观察；MLP 隐宽16、残差块1、价值原子11。
结构化 Actor 另以 B=2 运行同一真实更新。合成观察只验证张量和算法合同，不证明学习能力。
所有前向、损失、Adam 更新和参数投影均执行真实实现；钩子只记录数值及调用时机。
"""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import io
import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def api() -> Iterator[SimpleNamespace]:
    r"""只略过父包环境注册，按真实源码导入 learner 及其全部 torch 依赖。

    空包仅提供 __path__，不替换任何类、数学函数或网络层；离开本文件测试后恢复导入空间。
    这样可以在没有 Isaac/Kit 的进程中检验真实相对导入和结构化策略实现。
    """
    root = Path(__file__).resolve().parents[4]  # anymani Python 包根目录
    before = set(sys.modules)  # 只清理由本文件新增的 anymani 子模块
    shells: set[str] = set()  # 由 MonkeyPatch 自行恢复的空父包
    with pytest.MonkeyPatch.context() as patches:  # 导入边界仅在此文件的测试生命周期内有效
        for suffix in (
            "",  # 顶层anymani通常触发环境注册
            ".distill",  # 训练/表征共同父包
            ".distill.models",  # 真实结构化网络所在目录
            ".distill.models.backbones",  # 真实图Transformer所在目录
            ".distill.rl",  # 略过训练环境别名注册
            ".distill.rl.flash_sac",  # 真实learner及相对依赖所在目录
        ):  # 只替代包入口，叶模块仍逐个执行真实源码
            name = "anymani" + suffix  # 生产源码使用的真实包名
            if name not in sys.modules:  # 已存在的真实模块保持身份，不影响其他合同测试
                package = ModuleType(name)  # 仅省略 __init__ 的运行时注册副作用
                directory = root.joinpath(*suffix.strip(".").split(".")) if suffix else root  # 真实源码路径
                package.__dict__["__path__"] = [str(directory)]  # 导入器只在对应生产目录查找模块
                package.__package__ = name  # 允许真实相对导入
                package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)  # 标准包 metadata
                patches.setitem(sys.modules, name, package)  # fixture 生命周期内有效
                shells.add(name)  # teardown 不重复删除 MonkeyPatch 拥有的键
        try:  # 导入失败也须恢复本文件新增的包级状态
            prefix = "anymani.distill.rl.flash_sac."  # 所有测试消费同一生产模块身份
            modules = {
                name: importlib.import_module(prefix + name)  # 真实算法与网络，不用测试替身
                for name in ("learner", "config", "networks", "normalized", "observations")
            }
            imported = set(sys.modules) - before  # 核对本次导入没有触发仿真依赖
            forbidden = [name for name in imported if name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"}]
            assert not forbidden, f"CPU 合同意外导入仿真依赖：{forbidden}"  # 仿真依赖不属于此证据边界
            yield SimpleNamespace(**modules)  # 类与函数全部来自生产源码
        finally:  # 每个测试文件拥有自己的隔离导入生命周期
            for name in set(sys.modules) - before - shells:  # 只枚举本文件创建的叶模块
                if name.startswith("anymani."):  # 恢复其他测试所见的包级状态
                    sys.modules.pop(name, None)  # 不删除任何本文件载入前存在的模块


@pytest.fixture(autouse=True)
def restore_cpu_rng() -> Iterator[None]:
    r"""每条合同保存并恢复 CPU RNG；devices=[] 不访问 CUDA 随机流。"""
    with torch.random.fork_rng(devices=[]):  # learner 构造中的 manual_seed 只影响本条测试
        yield  # 保留被测 learner 自身的采样行为


@pytest.fixture
def small_config(api: SimpleNamespace) -> Any:
    r"""保留真实输入宽度，仅缩小 MLP 隐宽与离散价值头，预算以 transition 计。"""
    return api.config.FlashSACConfig(
        seed=173,  # 模型初始化和独立随机重放的数值锚点
        actor_variant="flash_mlp",  # 上游式BN/UnitLinear残差Actor
        asset_count=4,  # 四种手型，身份不进入网络
        num_envs=4,  # 一个vector step含4条真实transition
        batch_size=4,  # B=4，每资产一条回放样本
        total_transitions=128,  # 预算单位为transition
        learning_starts=16,  # 预热交互仍计入总预算
        replay_capacity=256,  # T=64 > History30+n3
        updates_per_transition=0.25,  # 每4条新交互1次Q更新
        actor_update_period=2,  # Actor在第1/3/5次Q更新时更新
        actor_mlp_hidden_dim=16,  # 输入仍为6585维，仅缩小隐层
        actor_mlp_num_blocks=1,  # 一个真实残差块
        critic_hidden_dim=16,  # 每个critic独立16维隐层
        critic_num_blocks=1,  # 每个critic一个真实残差块
        critic_bins=11,  # [-5,5]支撑，Delta z=1
        target_tau=0.2,  # 明显但合法的EMA权重，便于核算
        use_amp=False,  # FP32 CPU验证
        compile_mode=None,  # eager路径保持算子与钩子可审计
    )


def _observation() -> dict[str, torch.Tensor]:
    r"""构造四种真实 depth-major prefix 手型，完整维度且严格 TIP-only。

    指长分别为 (4,4,4,4)、(4,3,2,3)、(3,0,3,3)、(1,2,4,3)。无效指没有 TIP。
    q/pi、u/pi、上一动作位于归一坐标；接触为二元值。History30 含当前帧。
    图由 PALM→各指关节链→TIP 形成，保证 owner masks 与图关系共同对应同一运动学树。
    """
    generator = torch.Generator(device="cpu").manual_seed(604)  # 独立于 learner 的策略随机流
    lengths = torch.tensor([[4, 4, 4, 4], [4, 3, 2, 3], [3, 0, 3, 3], [1, 2, 4, 3]])  # [B,F]
    joint = (torch.arange(4)[None, :, None] < lengths[:, None, :]).reshape(4, 16)  # [B,depth*F]
    tip = lengths > 0  # [B,4]，无关节的手指也没有有效指尖
    owner = torch.cat((torch.ones(4, 1, dtype=torch.bool), joint, tip), -1)  # [B,21] PALM/JOINT/TIP
    history = 0.2 * torch.randn(4, 30, 16, 5, generator=generator)  # 完整 [B,L,J,C]，归一物理量
    history[..., 3] = 0  # 非 TIP 自身接触通道必须恒零
    contacts = torch.randint(0, 2, (4, 30, 4), generator=generator).float() * tip[:, None, :]  # [B,L,F]
    history[..., 4] = contacts.repeat(1, 1, 4)  # depth-major：各深度共享所属手指的 TIP 触觉
    history.masked_fill_(~joint[:, None, :, None], 0)  # ghost 历史不承载状态
    current = history[:, -1].clone()  # latest frame 与当前物理观测逐位一致
    contact = torch.zeros(4, 21, 1)  # actor 仅 TIP 接触可非零
    contact[:, 17:, 0] = contacts[:, -1]  # PALM/JOINT 接触保持零
    limits = torch.tensor([-1.0, 1.0]).expand(4, 16, 2).clone().masked_fill(~joint[..., None], 0)  # qmin/pi,qmax/pi

    # 用祖先链独立计算无向距离与 parent/child 方向桶；不可达方向使用末桶8。
    shortest = torch.zeros(4, 21, 21, dtype=torch.long)  # ghost pair 始终为零
    parent = torch.zeros_like(shortest)  # [B,E,E]，从源 owner 向祖先的距离
    for row in range(4):  # 四种形态分别建立祖先树，不在batch间共享错误图结构
        ancestors = {0: [0]}  # PALM 是唯一根
        for finger, length in enumerate(lengths[row].tolist()):  # 逐指的真实关节链长度
            chain = [0]  # 同手指有效关节沿深度顺序连到根
            for depth in range(length):  # 只遍历物理关节，ghost不进入祖先关系
                node = 1 + 4 * depth + finger  # depth-major JOINT owner 编号
                chain = [node, *chain]  # 当前节点到 PALM 的祖先顺序
                ancestors[node] = chain  # 保存独立列表，不共享后续变更
            if length:  # 没有真实关节的手指不建立虚构TIP
                ancestors[17 + finger] = [17 + finger, *chain]  # TIP 接最后一个真实关节
        for source, source_path in ancestors.items():  # 图的源实体轴
            for target, target_path in ancestors.items():  # 图的目标实体轴
                common = next(node for node in source_path if node in target_path)  # 最近共同祖先
                distance = source_path.index(common) + target_path.index(common)  # 两段祖先距离之和
                shortest[row, source, target] = min(8, distance)  # 无向距离桶
                parent[row, source, target] = source_path.index(target) if target in source_path else 8  # 父向可达性
    tokens = (0.2 * torch.randn(4, 21, 128, generator=generator)).masked_fill(~owner[..., None], 0)  # 冻结几何[B,E,D]
    critic_joint = torch.randn(4, 16, 4, generator=generator).masked_fill(~joint[..., None], 0)  # 特权关节状态
    critic_contact = torch.randn(4, 21, 2, generator=generator).masked_fill(~owner[..., None], 0)  # 特权接触
    return {  # 完整named观察，所有浮点值均驻留CPU
        "actor_jnt_current": current,  # [4,16,5]归一状态与TIP接触
        "actor_jnt_history": history,  # [4,30,16,5]含当前帧
        "actor_jnt_limits": limits,  # [4,16,2]归一关节限位
        "actor_owner_contact": contact,  # [4,21,1]仅TIP触觉
        "jnt_valid": joint,  # [4,16]真实可控关节
        "tip_valid": tip,  # [4,4]真实手指末端
        "owner_valid": owner,  # [4,21]与几何同索引
        "geometry_tokens": tokens,  # [4,21,128]，ghost几何精确零
        "shortest_path": shortest,  # [4,21,21]无向图距离桶
        "parent_direction": parent,  # [4,21,21]父向距离桶
        "child_direction": parent.transpose(1, 2).contiguous(),  # 子向距离是父向的转置
        "critic_jnt_state": critic_joint,  # [4,16,4]特权关节状态
        "critic_owner_contact": critic_contact,  # [4,21,2]特权接触
        "critic_obj": torch.randn(4, 1, 15, generator=generator),  # 物体状态，Actor不可读
        "critic_task": torch.randn(4, 1, 8, generator=generator),  # 任务状态，Actor不可读
        "critic_reward_release": torch.tensor([[0.0], [0.1], [0.2], [0.3]]),  # 只进入 critic 的真实形状字段
    }


@pytest.fixture
def batch(small_config: Any) -> dict[str, Any]:
    r"""当前/下一状态确实不同；折扣同时覆盖终止、1步、2步与完整3步 bootstrap。"""
    observation = _observation()  # B=4，全 named observation
    following = {name: value.clone() for name, value in observation.items()}  # 同一手型的下一状态
    following["actor_jnt_current"][..., :3] += 0.03 * observation["jnt_valid"][..., None]  # 有效关节物理状态推进
    following["actor_jnt_history"] = torch.cat(
        (observation["actor_jnt_history"][:, 1:], following["actor_jnt_current"][:, None]), 1
    )  # [4,30,16,5]，沿时间轴推进一帧
    following["geometry_tokens"] += 0.02 * observation["owner_valid"][..., None]  # 下一 q 对应不同冻结几何
    following["critic_obj"] += 0.1  # 下一物体特权状态发生变化
    actions = torch.linspace(-0.4, 0.4, 64).reshape(4, 16).masked_fill(~observation["jnt_valid"], 0)  # replay 归一动作
    gamma = small_config.gamma  # n-step折扣使用任务声明的同一个gamma
    return {  # transition已包含终止/n-step的最终TD权重
        "obs": observation,  # 完整当前状态
        "next_obs": following,  # 完整下一状态
        "actions": actions,  # [4,16]回放动作
        "rewards": torch.tensor([0.25, -0.4, 0.2, 0.1]),  # [4]已累计的原始n-step奖励
        "discounts": torch.tensor([0.0, gamma, gamma**2, gamma**3]),  # [4]，已成形TD权重
    }  # batch 不依赖 learner 从 done 重建 n-step 折扣


def _parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    r"""冻结参数快照；与 BN buffers 分开，证伪 EMA/optimizer 混合更新。"""
    return {name: value.detach().clone() for name, value in module.named_parameters()}  # Parameter 身份不参与拷贝


def _assert_tree_equal(actual: Any, expected: Any, path: str = "state") -> None:
    r"""递归逐位核对模型、Adam moments、计数与随机流；错误保留完整状态键。"""
    if isinstance(expected, torch.Tensor):  # 模型/RNG/moments均按实际tensor值比较
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)  # 相同 CPU 随机流应逐位恢复
    elif isinstance(expected, Mapping):  # 同一个checkpoint层级必须含相同物理/算法状态
        assert actual.keys() == expected.keys(), f"{path} 的状态键不一致"  # 不允许缺项后只比较共有键
        for key in expected:  # 遍历全部状态，不只选择Actor权重
            _assert_tree_equal(actual[key], expected[key], f"{path}.{key}")  # 显式保留 checkpoint namespace
    elif isinstance(expected, (tuple, list)):  # optimizer分组和参数索引列表
        assert len(actual) == len(expected), f"{path} 的状态序列长度不一致"  # 无静默截断
        for index, (observed, reference) in enumerate(zip(actual, expected, strict=True)):  # 序列顺序同样重要
            _assert_tree_equal(observed, reference, f"{path}[{index}]")  # optimizer param_groups / CUDA空列表
    else:  # 普通配置标量、计数与None梯度
        assert actual == expected, f"{path}: actual={actual!r}, expected={expected!r}"  # 精确而非近似恢复


def _assert_projected(module: torch.nn.Module, normalized: ModuleType) -> None:
    r"""核对优化后真实约束：线性行半径1，BN联合仿射及RMS scale半径sqrt(d)。"""
    for name, child in module.named_modules():  # 组合层的每个实际单位投影叶子
        layer = cast(Any, child)  # 类由隔离导入取得，运行时isinstance给出真实投影语义
        if isinstance(layer, normalized.UnitLinear):  # 单网络线性层
            norms = layer.w.weight.detach().norm(dim=-1)  # [d_out]，每个输出行独立
            expected = torch.ones_like(norms)  # ||w_o||=1
        elif isinstance(layer, normalized.EnsembleUnitLinear):  # 双Q独立线性层
            norms = layer.weight.detach().norm(dim=-1)  # [Q,d_out]，不跨 Q 归约
            expected = torch.ones_like(norms)  # 每个 critic 独立的单位球面
        elif isinstance(layer, (normalized.UnitBatchNorm, normalized.EnsembleUnitBatchNorm)):  # 联合仿射约束
            norms = (layer.weight.detach().square() + layer.bias.detach().square()).sum(-1).sqrt()  # scale/bias联合
            expected = torch.full_like(norms, math.sqrt(layer.weight.shape[-1]))  # 半径sqrt(d)，不是分别投影
        elif isinstance(layer, (normalized.UnitRMSNorm, normalized.EnsembleUnitRMSNorm)):  # 只有scale的径向约束
            norms = layer.weight.detach().norm(dim=-1)  # RMS只约束scale
            expected = torch.full_like(norms, math.sqrt(layer.weight.shape[-1]))  # 同一特征尺度
        else:  # 没有单位参数约束的其他生产层
            continue  # 组合层和未声明单位约束的 bias 不纳入投影命题
        torch.testing.assert_close(norms, expected, rtol=2e-6, atol=2e-6, msg=f"{name} 参数投影范数")  # FP32归约误差界


def test_named_observation_contract_is_full_shape_and_tip_only(api: SimpleNamespace, batch: dict[str, Any]) -> None:
    r"""先验证证据输入有真实完整宽度、合法 owner 关系与 TIP-only 掩码。"""
    expected = {  # 真实接口尺寸是独立参照，而非从被测函数返回值反推
        "actor_jnt_current": (4, 16, 5),  # B/J/通道
        "actor_jnt_history": (4, 30, 16, 5),  # B/时间/J/通道
        "actor_jnt_limits": (4, 16, 2),  # 下限/上限
        "actor_owner_contact": (4, 21, 1),  # B/owner/触觉
        "jnt_valid": (4, 16),  # 真实关节轴
        "tip_valid": (4, 4),  # 真实指尖轴
        "owner_valid": (4, 21),  # PALM/JOINT/TIP共同轴
        "geometry_tokens": (4, 21, 128),  # 冻结几何宽度128
        "shortest_path": (4, 21, 21),  # 无向实体关系
        "parent_direction": (4, 21, 21),  # 父向实体关系
        "child_direction": (4, 21, 21),  # 子向实体关系
        "critic_jnt_state": (4, 16, 4),  # 特权关节状态
        "critic_owner_contact": (4, 21, 2),  # 特权接触状态
        "critic_obj": (4, 1, 15),  # 物体状态
        "critic_task": (4, 1, 8),  # 任务状态
        "critic_reward_release": (4, 1),  # 奖励释放状态
    }  # 尺度测试仍使用完整物理信息接口，不能压缩输入宽度通过
    for observation in (batch["obs"], batch["next_obs"]):  # 当前与下一状态均应满足同一物理合同
        assert {name: tuple(value.shape) for name, value in observation.items()} == expected, (
            "named observation形状不完整"  # 不得通过删掉宽维特征让小网络测试成立
        )
        actor, geometry = api.observations.actor_inputs(observation)  # 真实数据类构造验证owner一致性
        _assert_tree_equal(actor.owner_valid, geometry.owner_valid, "owner mask")  # 两种视图同一物理实体轴
        assert not actor.owner_contact[:, :17].any(), "TIP-only actor泄漏了PALM/JOINT接触"  # 仅owner17..20可非零
        assert not actor.jnt_history[..., 3].any(), "History30非TIP接触通道必须全零"  # 历史不能旁路接触信息边界
        _assert_tree_equal(actor.jnt_history[:, -1], actor.jnt_current, "history最新帧")  # 包含当前帧
        assert api.observations.actor_flat_features(observation).shape == (4, 6585), (
            "Actor不能缩小真实观察宽度"
        )  # B/feature
        assert api.observations.critic_flat_features(observation).shape == (4, 6715), (
            "Critic必须保留130维特权状态"
        )  # 6585+130


def test_update_order_disjoint_gradients_counts_and_parameter_ema(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any]
) -> None:
    r"""第1/3次Actor→温度，第2次跳过；每次critic优化/投影后逐参数EMA。"""
    learner = api.learner.FlashSACLearner(small_config, device="cpu")  # 真实小MLP，Adam和目标Q均正常构造
    groups = {
        "actor": list(learner.actor.parameters()),  # 行为分布参数
        "temperature": [learner.log_temperature],  # 独立标量log alpha
        "critic": list(learner.critic.parameters()),  # 在线分布双Q参数
        "target": list(learner.target_critic.parameters()),  # EMA目标参数
    }  # 独立状态集合
    ids = {name: {id(parameter) for parameter in parameters} for name, parameters in groups.items()}  # 优化变量身份
    for first, values in ids.items():  # Actor/温度/在线Q/目标Q四类状态
        for second, others in ids.items():  # 两两检查可训练变量归属
            assert first == second or values.isdisjoint(others), f"{first}/{second} 共享了Parameter"  # 不共享存储对象
    optimizers = {
        "actor": learner.actor_optimizer,  # Actor专属Adam
        "temperature": learner.temperature_optimizer,  # 温度专属Adam
        "critic": learner.critic_optimizer,  # 在线双Q专属Adam
    }
    for name, optimizer in optimizers.items():  # 三个Adam各自完整覆盖本组参数
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == ids[name], (
            f"{name} optimizer参数归属错误"  # 既不能漏参，也不能混入相邻网络
        )
    for value in vars(learner).values():  # 即使增加未知名称的optimizer，也必须检查其目标参数归属
        if isinstance(value, torch.optim.Optimizer):  # 审计实际optimizer对象，不只检查属性名
            assert not ids["target"] & {id(p) for group in value.param_groups for p in group["params"]}, (
                "target拥有optimizer"  # 目标网络只允许参数EMA
            )
    events: list[str] = []  # 混合记录前向、实际optimizer step、投影和目标参数lerp
    real_step, real_lerp = learner._backward_step, torch.Tensor.lerp_  # 包装保留真实数值路径

    def record_step(loss: torch.Tensor, optimizer: Any, parameters: Any, *, clip: float | None = None) -> None:
        r"""每个优化阶段前后检查全部组：非本组的参数和已有梯度都不应变化。"""
        label = next(name for name, candidate in optimizers.items() if candidate is optimizer)  # 当前阶段名称
        before = {name: [p.detach().clone() for p in values] for name, values in groups.items()}  # 参数快照
        gradients = {
            name: [None if p.grad is None else p.grad.clone() for p in values] for name, values in groups.items()
        }  # 保存上一阶段梯度，检查当前阶段没有跨组累积
        events.append(label)  # 与上游 _update_networks 的先后顺序对照
        real_step(loss, optimizer, parameters, clip=clip)  # 真正执行zero_grad/backward/Adam.step
        assert torch.isfinite(loss), f"{label} loss非有限"  # 有限loss是实际更新的最低数值前提
        assert any(not torch.equal(p, old) for p, old in zip(groups[label], before[label], strict=True)), (
            f"{label}未产生真实参数更新"  # 不把空优化器或伪造返回值当成功
        )
        for name, values in groups.items():  # 每一阶段之后审计全部参数组
            for index, parameter in enumerate(values):  # 包括未被当前目标直接读取的参数
                assert torch.isfinite(parameter).all(), f"{name}[{index}]参数非有限"  # 每个实际参数元素均检查
                assert parameter.grad is None or torch.isfinite(parameter.grad).all(), (
                    f"{name}[{index}]梯度非有限"
                )  # 不只查loss
                if name != label:  # 保留上一阶段残余梯度，但当前阶段不能再向其累积
                    _assert_tree_equal(
                        parameter, before[name][index], f"{label}修改了{name}[{index}]"
                    )  # 非本组权重不更新
                    _assert_tree_equal(
                        parameter.grad, gradients[name][index], f"{label}梯度泄漏到{name}[{index}]"
                    )  # 无额外累积

    def record_lerp(tensor: torch.Tensor, end: torch.Tensor, weight: Any) -> torch.Tensor:
        r"""只把目标Parameter的lerp计入EMA，BN运行统计的lerp另有自己的时钟。"""
        if id(tensor) in ids["target"]:  # 区分目标Parameter EMA与BN buffer自身的统计EMA
            events.append("ema")  # 每个target参数恰被更新一次
        return real_lerp(tensor, end, weight)  # 完整保留上游式原位线性插值

    def record_forward(label: str):
        r"""记录具名观察的batch长度与显式BN模式，区分current/next cross-batch。"""

        def hook(_module: Any, args: Any, kwargs: Any) -> None:
            r"""被测网络收到的真实观察轴应分别为B=4或2B=8。"""
            events.append(f"{label}:{args[0]['jnt_valid'].shape[0]}:{kwargs['training']}")  # 不更改输入

        return hook  # 返回只读pre-hook

    def project_with_record(label: str, original: Any) -> None:
        r"""调用真实范数投影，同时记录其相对optimizer/EMA的位置。"""
        events.append(label)  # 优化后立即投影
        original()  # 不替换任何投影数学

    with ExitStack() as stack:  # 所有临时钩子只观察这一真实learner实例
        stack.enter_context(patch.object(learner, "_backward_step", side_effect=record_step))  # 所有优化保持真实执行
        stack.enter_context(patch.object(torch.Tensor, "lerp_", record_lerp))  # 目标EMA数值仍用原算子
        actor_project = learner.actor.project_parameters  # 先保留真实bound method，避免wrapper递归
        critic_project = learner.critic.project_parameters  # 同样保留真实critic投影
        stack.enter_context(
            patch.object(
                learner.actor,  # 真实Actor对象
                "project_parameters",  # 仅包装其实际投影入口
                side_effect=lambda: project_with_record("actor_project", actor_project),  # 调用原方法
            )
        )  # 投影与step/EMA使用同一事件时钟
        stack.enter_context(
            patch.object(
                learner.critic,  # 真实在线双Q对象
                "project_parameters",  # 记录投影时机
                side_effect=lambda: project_with_record("critic_project", critic_project),  # 调用原方法
            )
        )  # 真实critic投影仍原位执行
        for name in ("actor", "critic", "target_critic"):  # 三个网络的实际B轴与BN模式
            handle = getattr(learner, name).register_forward_pre_hook(
                record_forward(name), with_kwargs=True
            )  # 记录实际调用
            stack.callback(handle.remove)  # 不把钩子留给其他测试
        for index in range(1, 4):  # 首次、延迟跳过、再次更新Actor三个相位
            events.clear()  # 每次update单独判定上游次序
            target_before = _parameters(learner.target_critic)  # EMA的旧端点
            info = learner.update(batch)  # 真实完整更新，含随机重参数采样
            prefix = (
                ["actor:8:True", "critic:4:False", "actor", "actor_project", "temperature"] if index % 2 else []
            )  # 延迟Actor
            expected = prefix + [
                "actor:4:False",  # 已更新策略采样下一状态动作
                "target_critic:8:True",  # 目标Q自有cross-batch统计
                "critic:8:True",  # 在线Q自己的cross-batch前向
                "critic",  # 真实critic优化
                "critic_project",  # critic单位范数投影
            ]  # 每次Q更新
            assert events == expected + ["ema"] * len(groups["target"]), (
                f"第{index}次更新次序错误：{events}"
            )  # EMA必须最后
            assert learner.critic_updates == index and learner.actor_updates == (index + 1) // 2, (
                f"第{index}次计数错误"
            )  # 1/0起算
            assert info["critic_updates"] == index and info["actor_updates"] == (index + 1) // 2, (
                "返回计数与内部状态不一致"  # 诊断不得另用不同的计数时钟
            )
            assert ("actor_loss" in info) == bool(index % 2), "延迟Actor的诊断键与实际更新周期不一致"  # 不能复用旧loss
            assert all(math.isfinite(value) for value in info.values()), f"非有限训练统计：{info}"  # 审计全部具名标量
            for name, parameter in learner.target_critic.named_parameters():  # 对每个参数独立核算凸组合
                online = dict(learner.critic.named_parameters())[name].detach()  # 已完成Adam及单位投影
                old = target_before[name]  # EMA前的目标参数
                expected_parameter = (1 - small_config.target_tau) * old + small_config.target_tau * online  # 凸组合
                torch.testing.assert_close(
                    parameter, expected_parameter, rtol=2e-6, atol=2e-7, msg=f"{name} EMA不是投影后的在线参数"
                )
                assert not parameter.requires_grad and parameter.grad is None, f"target {name}持有梯度"  # 目标不可微
            _assert_projected(learner.actor, api.normalized)  # Actor延迟步之间也保持约束
            _assert_projected(learner.critic, api.normalized)  # critic每次Adam之后完成投影


def test_target_batch_norm_uses_own_cross_batch_not_online_buffers(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any]
) -> None:
    r"""以不同初始running moments区分目标自更新、复制在线buffer和buffer EMA三种机制。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 初始化目标与在线结构相同
    with torch.no_grad():  # 只构造可辨识的合法统计初态
        for name, buffer in learner.target_critic.named_buffers():  # 目标自己的running moments
            if "running_mean" in name:  # location状态
                buffer.fill_(2.0)  # 目标均值的可辨识初态
            elif "running_var" in name:  # scale状态
                buffer.fill_(4.0)  # 目标方差的可辨识初态
        for name, buffer in learner.critic.named_buffers():  # 在线统计刻意与目标不同
            if "running_mean" in name:  # location状态
                buffer.fill_(-3.0)  # 在线统计刻意不同，但仍是合法有限buffer
            elif "running_var" in name:  # scale状态
                buffer.fill_(0.75)  # 在线初始方差仍为正
    expected: dict[str, torch.Tensor] = {}  # 按每个目标BN的真实输入独立计算更新
    seen: dict[str, torch.Tensor] = {}  # 目标前向结束、参数EMA之前的buffer快照

    def audit_bn(name: str):
        r"""使用完整2B=8个当前/下一样本，计算每个critic自己的无偏运行方差。"""

        def hook(module: Any, args: Any, kwargs: Any) -> None:
            r"""不改前向，只核对维度、模式并形成预期running moments。"""
            x = args[0].detach()  # [Q=2,2B=8,d]
            assert kwargs.get("training", args[1] if len(args) > 1 else None) is True, (
                f"{name}目标BN未用cross-batch模式"  # training必须显式为True
            )
            assert x.shape[:2] == (2, 8), f"{name}期望[Q,2B,d]，实际{x.shape}"  # Q轴不参与统计归约
            momentum = module.momentum  # 当前BN自己的统计EMA权重
            expected[name + ".running_mean"] = (1 - momentum) * module.running_mean + momentum * x.mean(1)  # μ更新
            sample_var = x.var(1, correction=1)  # 无偏方差B/(B-1)修正独立于实现的乘法形式
            expected[name + ".running_var"] = (1 - momentum) * module.running_var + momentum * sample_var  # v更新

        return hook  # 源数据来自本层真实输入，不从在线网络倒推

    def capture_after_forward(module: Any, args: Any, kwargs: Any, _output: Any) -> None:
        r"""完整target前向结束时保存buffers；后续EMA只能改Parameter。"""
        assert kwargs["training"] is True, "目标critic必须用自己的训练BN统计"  # 不读在线BN或固定推理统计
        for name, value in batch["obs"].items():  # 检查完整named输入的拼接顺序
            _assert_tree_equal(args[0][name], torch.cat((value, batch["next_obs"][name]), 0), f"target combined.{name}")
        _assert_tree_equal(args[1][:4], batch["actions"], "target cross-batch当前动作")  # 当前半批来自replay
        seen.update({name: value.clone() for name, value in module.named_buffers()})  # 参数EMA前的快照

    with ExitStack() as stack:  # 每层输入统计钩子只在该次更新期间存在
        for name, module in learner.target_critic.named_modules():  # 覆盖embedder和残差块内部BN
            if isinstance(module, api.normalized.EnsembleUnitBatchNorm):  # 只对有running moments的层形成参照
                stack.callback(
                    module.register_forward_pre_hook(audit_bn(name), with_kwargs=True).remove
                )  # 每层独立均值/方差
        stack.callback(
            learner.target_critic.register_forward_hook(capture_after_forward, with_kwargs=True).remove
        )  # EMA前时点
        learner.update(batch)  # 真正运行目标BN、在线BN和参数EMA
    assert len(expected) == 6, f"一层embedder加一个双BN block应有6个目标统计buffer，实际{len(expected)}"  # μ/v各三项
    current = dict(learner.target_critic.named_buffers())  # update结束后的目标buffer
    for name, value in expected.items():  # 每层/每种矩独立核算
        torch.testing.assert_close(
            current[name], value, rtol=3e-6, atol=3e-6, msg=f"{name}未按目标自己的cross-batch更新"
        )
        assert not torch.equal(current[name], dict(learner.critic.named_buffers())[name]), (
            f"{name}被在线BN覆盖"
        )  # 排除copy
    _assert_tree_equal(current, seen, "target buffers after EMA")  # 所有buffer包括support完全未被EMA修改


def test_discounts_consumed_directly_and_reward_stats_only_observe_real_steps(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any]
) -> None:
    r"""n-step/终止折扣来自batch；重复优化只读取尺度，不增加环境交互或RMS计数。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 初始count与交互计数均为0
    terminated = torch.tensor([False, True, False, False])  # 第二个环境终止
    truncated = torch.tensor([False, False, True, False])  # 第三个环境截断
    learner.observe_transition(torch.tensor([1.0, 2.0, -1.0, -0.5]), terminated, truncated)  # 恰一个真实vector step
    initial = copy.deepcopy(learner.reward_normalizer.state_dict())  # 每个env贡献一个trace样本
    assert learner.collected_transitions == 4 and learner.reward_normalizer.count == 4, (
        "真实step应新增4条transition"
    )  # N=4
    batch["terminated"] = torch.ones(4, dtype=torch.bool)  # 与非零discount故意冲突，learner应只消费成形discount
    batch["truncated"] = torch.ones(4, dtype=torch.bool)  # 不得再做另一次done乘法
    projected: list[torch.Tensor] = []  # 捕获真实投影结果，以独立三角基复算
    original = api.learner.project_categorical  # 本测试不替换C51数值路径

    def audit_projection(
        logs: torch.Tensor, rewards: torch.Tensor, discounts: torch.Tensor, costs: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        r"""在learner真正调用投影时核对折扣、熵符号与归一奖励，返回原实现结果。"""
        _assert_tree_equal(discounts, batch["discounts"], "batch discounts直接消费")  # 不再乘gamma**n或done
        _assert_tree_equal(
            rewards, learner.reward_normalizer.scale(batch["rewards"]), "Q目标的只读奖励尺度"
        )  # 不中心化
        result = original(logs, rewards, discounts, costs, support)  # 原投影
        atoms = (rewards[:, None] + discounts[:, None] * (support[None, :] - costs[:, None])).clamp(
            support[0], support[-1]
        )  # [B,K]，clip后的soft Bellman目标
        width = support[1] - support[0]  # 奖励单位，当前11-bin支撑步长为1
        weights = (1 - (atoms[:, :, None] - support[None, None, :]).abs() / width).clamp_min(0)  # C51三角基
        expected = (logs.exp()[:, :, None] * weights).sum(1)  # [B,K]，独立于scatter索引实现
        torch.testing.assert_close(result, expected, rtol=3e-6, atol=5e-7)  # 消费的全部TD项可复算
        terminal_expected = (1 - (rewards[0] - support).abs() / width).clamp_min(0)  # d0=0，全质量投到reward
        torch.testing.assert_close(result[0], terminal_expected, rtol=2e-6, atol=2e-7)  # future entropy不进入terminal
        assert not result.requires_grad, "目标分布不应沿target/alpha建立梯度图"  # TD目标梯度隔离
        projected.append(result.clone())  # 保留本次真实目标
        return result  # 网络仍用真实C51目标学习

    with patch.object(api.learner, "project_categorical", side_effect=audit_projection):  # 只增加审计，仍执行原投影
        infos = [learner.update(batch), learner.update(batch)]  # 两次读取同一回放，不是两次新交互
    assert len(projected) == 2 and learner.collected_transitions == 4, (
        "优化错误地推进了采样时钟"
    )  # 两次update不是两步环境
    _assert_tree_equal(
        learner.reward_normalizer.state_dict(), initial, "update只读reward统计"
    )  # 连trace和max_abs也不动
    amplitude = 0.5 * (small_config.learning_rate - small_config.final_learning_rate)  # 余弦下降幅度
    fraction = 4 / small_config.total_transitions  # 仅由真实采样进度定义余弦相位
    expected_rate = small_config.final_learning_rate + amplitude * (1 + math.cos(math.pi * fraction))  # 可核算学习率
    assert infos[0]["learning_rate"] == infos[1]["learning_rate"] == expected_rate, (
        "LR时钟混入了optimizer次数"
    )  # 同C同LR
    learner.observe_transition(torch.tensor([0.5, 0.0, 1.0, -1.0]), terminated, truncated)  # 再增加一个真实vector step
    assert learner.collected_transitions == 8 and learner.reward_normalizer.count == 8, (
        "只允许observe推进采样与统计计数"  # 两次真实vector step共有8个trace样本
    )


@pytest.mark.parametrize("completed_updates", [1, 2])
def test_checkpoint_restores_noise_rng_next_update_and_all_weights(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any], completed_updates: int
) -> None:
    r"""覆盖下一次跳过/执行Actor两种相位；相同随机状态必须重放动作、损失和完整权重。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 真正初始化Adam/目标Q/行为采样器
    learner.observe_transition(
        torch.tensor([1.0, 0.5, -0.5, 2.0]), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
    )  # 一个真实vector step，RMS与采样时钟都非空
    learner.act(batch["obs"], explore=True)  # checkpoint包含已经推进的行为噪声与保持时钟
    for _ in range(completed_updates):  # checkpoint分别位于Actor更新后的奇数/偶数相位
        learner.update(batch)  # 建立非空Adam moments和BN状态
    stream = io.BytesIO()  # checkpoint只在内存中序列化
    torch.save(learner.state_dict(), stream)  # 同步快照，避免标准state_dict共享引用被下一次更新修改
    stream.seek(0)  # 读取固定检查点
    checkpoint = torch.load(stream, map_location="cpu", weights_only=True)  # 不使用自定义pickle对象

    # 先运行参考续训，再构造并恢复另一实例；恢复必须撤销构造函数消耗的全局RNG。
    reference_behavior = [learner.act(batch["obs"], explore=True).actions.clone() for _ in range(3)]  # 行为专用随机流
    with torch.no_grad():  # 参考策略样本只用于随机重放，不引入额外优化
        reference_sample = learner.actor(batch["next_obs"], training=False)  # 目标策略的独立Gaussian随机流
    reference_info = learner.update(batch)  # 与checkpoint相同相位的下一次真实优化
    reference_state = copy.deepcopy(learner.state_dict())  # 保存参数、BN、optimizer、统计和全部RNG
    restored = api.learner.FlashSACLearner(small_config, "cpu")  # 构造过程会重置并消耗全局种子
    restored.load_state_dict(checkpoint)  # 被测标准恢复路径
    assert restored.critic_updates == completed_updates and restored.actor_updates == (completed_updates + 1) // 2, (
        "checkpoint更新相位未恢复"  # Actor延迟计数必须随checkpoint恢复
    )
    for reference in reference_behavior:  # 连续多步行为噪声，可能跨越保持段
        _assert_tree_equal(restored.act(batch["obs"], explore=True).actions, reference, "恢复后的行为噪声动作")
    with torch.no_grad():  # 恢复侧复现同一次独立目标策略采样
        sample = restored.actor(batch["next_obs"], training=False)  # 复现完全相同的目标策略随机样本
    for name in (
        "actions",  # 真实随机动作
        "mean_action",  # 确定性中心
        "log_prob",  # 有效关节联合log密度
        "log_prob_per_active",  # 有效关节平均log密度
        "log_std",  # 条件潜尺度
        "active_count",  # 每个样本的真实DoF
    ):  # 完整ActorSample
        _assert_tree_equal(getattr(sample, name), getattr(reference_sample, name), f"采样状态.{name}")
    _assert_tree_equal(restored.update(batch), reference_info, "下一update损失与诊断")  # 无下降假设，只核对重放
    _assert_tree_equal(restored.state_dict(), reference_state, "下一update完整state")  # 所有权重/Adam moments逐位一致


@pytest.mark.parametrize("change", [{"gamma": 0.8}, {"total_transitions": 256}, {"target_sigma": 0.2}])
def test_checkpoint_rejects_configuration_mismatch_before_loading(
    api: SimpleNamespace, small_config: Any, change: dict[str, Any]
) -> None:
    r"""网络shape相同但方法/预算不同也必须拒绝，且不能先写入部分checkpoint。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 形成原配置checkpoint
    checkpoint = copy.deepcopy(learner.state_dict())  # 标准快照
    restored = api.learner.FlashSACLearner(replace(small_config, **change), "cpu")  # 参数shape仍兼容
    before = copy.deepcopy(restored.state_dict())  # 错误加载前的全部状态
    with pytest.raises(ValueError, match="configuration mismatch"):  # gamma/预算/熵尺度虽不改shape仍必须拒绝
        restored.load_state_dict(checkpoint)  # config identity必须先于参数/optimizer恢复检查
    _assert_tree_equal(restored.state_dict(), before, "配置冲突后的完整状态")  # 拒绝时不发生部分恢复


def test_deterministic_act_is_readonly_and_does_not_advance_either_rng(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any]
) -> None:
    r"""确定性评价不采行为噪声，也不改变BN、reward统计或训练用全局随机流。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 单独实例避免沿用其他测试状态
    before = copy.deepcopy(learner.state_dict())  # 包含CPU RNG与两个行为RNG
    sample = learner.act(batch["obs"], explore=False)  # 实际部署评价路径
    _assert_tree_equal(sample.actions, sample.mean_action, "确定性动作中心")  # 不采样latent噪声
    assert not sample.actions.requires_grad and not sample.actions[~batch["obs"]["jnt_valid"]].any(), (
        "评价动作梯度或ghost不合法"  # 确定性动作仍保持物理mask
    )
    _assert_tree_equal(learner.state_dict(), before, "确定性评价只读state")  # 统计和所有随机流均不前进


@pytest.mark.parametrize("max_repeat", [1, 7, 16])
def test_repeated_noise_matches_truncated_zeta_and_exact_hold_clock(api: SimpleNamespace, max_repeat: int) -> None:
    r"""P(L=k)∝k^-s，k∈[1,M]；每段共用长度而四环境Gaussian向量独立。"""
    exponent, seed = 1.5, 83  # 长尾更明显但仍为同一截断Zeta定义
    noise = api.learner.RepeatedGaussianNoise(
        4, exponent=exponent, max_repeat=max_repeat, device=torch.device("cpu"), seed=seed
    )  # N=4，共享区间时钟但Gaussian逐环境独立
    mass = [length ** (-exponent) for length in range(1, max_repeat + 1)]  # Python双精度独立概率参照
    denominator = math.fsum(mass)  # 截断分布的有限配分函数
    cumulative = [math.fsum(mass[:index]) / denominator for index in range(1, max_repeat + 1)]  # 完整CDF
    torch.testing.assert_close(
        noise.cdf, torch.tensor(cumulative, dtype=torch.float64), rtol=1e-14, atol=1e-14
    )  # 有限Zeta归一化
    intervals = torch.Generator(device="cpu").manual_seed(seed)  # 独立重放共享区间抽样
    gaussian = torch.Generator(device="cpu").manual_seed(seed + 1)  # 独立重放环境级Gaussian抽样
    for interval in range(24):  # 固定数量的完整保持段，而非用有限样本拟合分布曲线
        draw = float(torch.rand((), generator=intervals, dtype=torch.float64))  # [0,1)连续样本
        length = next(index + 1 for index, probability in enumerate(cumulative) if draw < probability)  # 逆CDF
        expected_noise = torch.randn(4, 16, generator=gaussian)  # [N,J]，每环境独立，非全batch共用一行
        first = noise.sample().clone()  # 区间刷新时恰抽一次Gaussian
        assert first.shape == (4, 16) and torch.isfinite(first).all(), f"第{interval}段噪声shape/有限性错误"  # N/J布局
        _assert_tree_equal(first, expected_noise, "区间Gaussian随机流")  # 固定seed逐位可核算
        assert 1 <= length <= max_repeat and noise.remaining == length - 1, (
            f"第{interval}段保持长度错误"
        )  # 包含首次sample
        assert not torch.equal(first[0], first[1]), "不同环境错误共用了同一个Gaussian向量"  # 共享的是时钟而非噪声行
        state = noise.state_dict()  # 第一次sample之后的两个生成器状态
        for remaining in range(length - 2, -1, -1):  # 已消费首次sample，再保持L-1次
            _assert_tree_equal(noise.sample(), first, "保持期内噪声恒定")  # 总计L次sample使用同一个向量
            assert noise.remaining == remaining, "保持时钟有off-by-one"  # 每个vector step恰减一
            _assert_tree_equal(noise.interval_generator.get_state(), state["interval_rng"], "保持期不重抽长度")
            _assert_tree_equal(noise.noise_generator.get_state(), state["noise_rng"], "保持期不重抽Gaussian")
        assert noise.remaining == 0, "段末remaining必须归零，下次sample才刷新"  # 段末不提前重抽


def test_repeated_noise_restores_mid_interval_without_resampling(api: SimpleNamespace) -> None:
    r"""保持期中间恢复：当前向量、剩余时钟与未来跨段随机流都必须完整重放。"""
    kwargs = {"num_envs": 4, "exponent": 1.2, "max_repeat": 16, "device": torch.device("cpu")}  # 长度分布与环境轴合同
    original = api.learner.RepeatedGaussianNoise(**kwargs, seed=15)  # 固定种子、有限查找长段
    for _ in range(64):  # 有限确定的前提准备，避免无界查找
        original.sample()  # 进入有至少两步剩余的区间
        if original.remaining >= 2:  # checkpoint必须落在尚未结束的保持段内部
            break  # 后续第一次sample应完全不抽随机数
    assert original.remaining >= 2, "固定种子没有产生所需保持期，测试前提失效"  # 明确报告前提而非假通过
    saved = copy.deepcopy(original.state_dict())  # 保留非零remaining和当前noise
    restored = api.learner.RepeatedGaussianNoise(**kwargs, seed=999)  # 不同构造seed应被checkpoint覆盖
    global_rng = torch.get_rng_state().clone()  # 类的两个局部随机流不应影响全局策略采样
    restored.load_state_dict(saved)  # 恢复不采样
    _assert_tree_equal(restored.state_dict(), saved, "noise恢复无额外重抽")  # 连生成器字节状态也一致
    _assert_tree_equal(torch.get_rng_state(), global_rng, "noise恢复不消耗全局RNG")  # 全局训练RNG与行为生成器分离
    for step in range(48):  # 包含当前段后半段和后续多个完整区间
        _assert_tree_equal(restored.sample(), original.sample(), f"恢复后的第{step}步噪声")  # 含保持期结束和多个新段
        _assert_tree_equal(restored.state_dict(), original.state_dict(), f"恢复后的第{step}步时钟/RNG")
    incompatible = api.learner.RepeatedGaussianNoise(**{**kwargs, "num_envs": 2}, seed=15)  # 环境轴不可静默重排
    with pytest.raises(ValueError, match="configuration mismatch"):  # 不能把4环境噪声载入2环境流
        incompatible.load_state_dict(saved)  # N不一致时拒绝恢复


@pytest.mark.parametrize(
    ("collected", "expected"), [(0, 0), (15, 0), (16, 0), (19, 0), (20, 1), (32, 4), (128, 28), (999, 28)]
)
def test_config_budget_counts_transitions_not_vector_steps_or_epochs(
    small_config: Any, collected: int, expected: int
) -> None:
    r"""U(C)=floor(max(0,min(C,128)-16)/4)，改变并行环境数不改变同transition预算。"""
    assert small_config.critic_update_budget(collected) == expected, (
        f"C={collected}期望累计{expected}次critic更新"
    )  # 预热后取floor
    other_parallelism = replace(small_config, num_envs=8, replay_capacity=512)  # 同4资产但每资产两个环境副本
    assert other_parallelism.critic_update_budget(collected) == expected, (
        "同transition数量因vector-step数不同改变了更新预算"  # 不受并行环境数或epoch影响
    )
    assert small_config.to_dict()["updates_per_transition"] == 0.25, (
        "配置记录应保存明确的每transition更新率"
    )  # 更新/新交互


def test_structured_actor_real_backward_on_named_observation(
    api: SimpleNamespace, small_config: Any, batch: dict[str, Any]
) -> None:
    r"""结构化TCN/图Actor在真实SAC更新中反向，验证动作头、条件sigma与共享干路梯度。"""
    config = replace(small_config, actor_variant="structured")  # 使用实际生产结构化骨架，不替换为MLP
    learner = api.learner.FlashSACLearner(config, "cpu")  # 真实structured Actor与独立分布critic
    small = {
        name: {key: value[1:3].clone() for key, value in item.items()} if isinstance(item, dict) else item[1:3].clone()
        for name, item in batch.items()
    }  # B=2，含12/9-DoF与缺失整指，输入宽度完整
    before = _parameters(learner.actor)  # 真实optimizer更新前的Actor快照
    info = learner.update(small)  # 真正的alpha*mean_logp-minQ与分布critic损失、clip和Adam
    assert all(math.isfinite(value) for value in info.values()), f"structured诊断非有限：{info}"  # 全部真实训练统计
    assert learner.actor_updates == 1 and learner.critic_updates == 1, "structured更新未完成"  # 构造/前向成功还不够
    body = learner.actor.body  # 生产结构化策略对象
    for name in (
        "direct_head",  # 关节动作读出
        "conditional_sigma_head",  # 每关节条件探索尺度
        "history_encoder",  # 逐关节History30 TCN
        "local_encoder",  # 低维当前/历史状态编码
        "global_backbone",  # 全owner图注意力
    ):  # 完整语义路径
        parameters = list(getattr(body, name).parameters())  # 每个语义子模块都应接收到真实SAC梯度
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), (
            f"structured {name}梯度路径断开"
        )  # 非零反向
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters), (
            f"structured {name}梯度非有限"
        )  # 全组检查
    assert any(not torch.equal(parameter, before[name]) for name, parameter in learner.actor.named_parameters()), (
        "structured Actor没有真实更新"  # Adam后至少一个真实权重改变
    )
    assert all(torch.isfinite(parameter).all() for parameter in learner.actor.parameters()), (
        "structured参数非有限"
    )  # 不只查loss
    sample = learner.act(small["obs"], explore=False)  # 确定性评价不受N=4行为噪声形状约束
    assert sample.actions.shape == (2, 16) and not sample.actions[~small["obs"]["jnt_valid"]].any(), (
        "structured动作shape/ghost不合法"  # [B=2,J=16]，结构零不能输出物理动作
    )
    assert all(parameter.grad is None for parameter in learner.target_critic.parameters()), (
        "structured更新使目标Q持有梯度"  # 模型换成结构化Actor后仍保持target梯度隔离
    )
