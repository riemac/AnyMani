r"""从已保存PPO批次比较优势归一化的完整Actor梯度，不运行仿真或optimizer。

输入为post-dataset-prepare-before-optimizer证据包，包含实际Actor参数、缓存N040输入、动作和旧概率。
先重放Actor并检查与rollout均值/概率的一致性，再在相同参数和样本上比较三种clipped surrogate：
    global：生产者保存的全rollout归一化优势G；
    per_asset：生产者保存的逐资产归一化优势(G-mean_a(G))/(std_a(G)+epsilon)；
    centered_global_scale：G-mean_a(G)，仅移除逐资产std增益，保留任务中心化。

每个模式分别汇总两组replica的全Actor参数梯度，最终每个half都覆盖全部256资产。
两组仍共享生产者的完整rollout归一化统计；这是条件于同一批经验的方向一致性，不是独立梯度SNR估计。
仅比较PPO surrogate；entropy与bounds等共同正则项不进入本探针，以免共同方向掩盖回报信号差异。
梯度是Adam之前的证据，不宣称它代表实际参数位移、物理能力或已确定的学习根因。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

prefer_local_rl_games(strict=True)  # 概率适配器依赖rl_games，导入前固定项目后端

from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.algorithms.ppo_batch import normalize_advantages_per_asset  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationMaskedContinuousModel  # noqa: E402


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    r"""以FP64计算固定参数坐标中两个梯度方向的余弦。"""
    x, y = left.double(), right.double()  # 只提高归约精度，不改变已计算的FP32梯度
    denominator = x.norm() * y.norm()  # $\|g_1\|\|g_2\|$
    return float((x @ y / denominator.clamp_min(1e-30)).item())  # 零向量按0报告，并另存范数


def probe(packet_path: Path, output_dir: Path, *, device: torch.device, chunk_size: int, disable_tf32: bool = False) -> dict:
    r"""对一个真实更新前批次计算三种权重下的全Actor half-gradients。

    梯度按每个half的总样本数归约，activation分块只控制内存，不变更统计总体。
    输入旧均值和旧负logp在每块被数值复核；所有模型参数与buffer须在结束时逐值保持。
    """
    started = time.perf_counter()  # 独立计时，不混入原训练墙钟
    packet = torch.load(packet_path, map_location="cpu", weights_only=False, mmap=True)  # 信任本项目生成的只读证据包
    assert packet["capture_phase"] == "post-dataset-prepare-before-optimizer"  # 固定采样/优化时间边界
    identity = packet["identity"]  # 精确方法与实际超参数
    cfg = identity["training"]  # 模型构造采用保存值，不由CLI重新选结构
    root = Path(__file__).resolve().parents[6]  # AnyMani根
    for name, expected in identity["implementation"]["files"].items():  # 跨实现不静默比较
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    allow_tf32 = bool(cfg["allow_tf32"]) and not disable_tf32  # 可显式做只读数值敏感性复核
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # 默认与原训练允许的矩阵精度一致
    torch.backends.cudnn.allow_tf32 = allow_tf32  # 卷积与矩阵运算使用同一检查设置
    package = PalmRotationActorCritic(
        arm=identity["policy"]["arm"],
        initial_log_std=cfg["initial_log_std"],
        max_log_std=cfg["max_log_std"],
        base_action_limit=cfg["base_action_limit"],
        history_encoder=cfg["history_encoder"],
    )  # N040 encoder不属于该package，消费保存的几何tokens
    prefix = "a2c_network.package."  # 正式checkpoint命名空间
    state = {name[len(prefix):]: value for name, value in packet["model"].items() if name.startswith(prefix)}
    package.load_state_dict(state, strict=True)  # Actor/Critic权重都必须无缺项匹配
    package.to(device).eval()  # 本探针只使用冻结策略分布的可微前向
    actor = package.actor  # 梯度边界仅为Actor，Critic不参与计算
    parameters = tuple(actor.parameters())  # 覆盖全部Actor参数，包括共享logstd
    parameter_names = [name for name, _ in actor.named_parameters()]  # 与保存的向量坐标绑定
    data = packet["dataset"]  # 已排列但尚未执行PPO更新的完整rollout
    observations = data["obs"]  # CPU mmap具名张量，按块搬入GPU
    labels = observations["prototype_index"].reshape(-1).long()  # 仅定义统计归约，绝不进入Actor
    halves = data["replica_halves"].reshape(-1).long()  # 实际replica parity标签
    count = labels.numel()  # 本次新交互数NH
    asset_count = int(cfg["asset_count"])  # 固定支持集成员数
    assert count == cfg["num_envs"] * cfg["horizon_length"]  # 不把某个minibatch当完整rollout
    assert torch.equal(torch.unique(halves), torch.tensor([0, 1]))  # 必须存在两个非空half
    half_counts = torch.bincount(halves, minlength=2)  # 精确loss分母
    for half in (0, 1):  # 两组各自覆盖全部资产且每资产等量
        member_counts = torch.bincount(labels[halves == half], minlength=asset_count)
        assert bool((member_counts == member_counts[0]).all()) and int(member_counts[0]) > 0
    global_advantage = data["global_advantages"].reshape(-1).float()  # 原生产者全rollout估计
    local_advantage = data["per_asset_advantages"].reshape(-1).float()  # 原生产者逐资产估计
    rebuilt, means, standard_deviations = normalize_advantages_per_asset(
        global_advantage, labels, asset_count=asset_count
    )  # 同一生产数学，sample std分母n-1
    normalization_error = float((rebuilt - local_advantage).abs().max())  # CPU/GPU归约舍入范围
    assert normalization_error < 1e-4, normalization_error
    modes = ("per_asset", "centered_global_scale", "global")  # 局部std与局部中心化分别有对照
    advantages = torch.stack((local_advantage, global_advantage - means[labels], global_advantage))  # [3,B]
    size = sum(parameter.numel() for parameter in parameters)  # 完整Actor坐标维数P
    gradients = torch.zeros(3, 2, size, device=device)  # [mode,replica-half,P]，不写parameter.grad
    max_mean_error, max_logp_error, min_free = 0.0, 0.0, None  # 数值与资源见证
    epsilon = float(packet["loss_configuration"]["e_clip"])  # 真实PPO裁剪半宽
    for start in range(0, count, chunk_size):  # activation分块，统计量始终来自完整批次
        stop = min(start + chunk_size, count)  # 最后一块可以较短
        obs = {name: value[start:stop].to(device) for name, value in observations.items()}  # 保留存储dtype
        geometry = PalmRotationGeometry(
            tokens=obs["geometry_tokens"].float(), owner_valid=obs["owner_valid"].bool(),
            shortest_path=obs["shortest_path"].long(), parent_direction=obs["parent_direction"].long(),
            child_direction=obs["child_direction"].long(),
        )  # 与生产适配器同一几何/graph输入
        actor_obs = PalmRotationActorObservation(
            jnt_current=obs["actor_jnt_current"].float(), jnt_history=obs["actor_jnt_history"].float(),
            jnt_limits=obs["actor_jnt_limits"].float(), owner_contact=obs["actor_owner_contact"].float(),
            jnt_valid=obs["jnt_valid"].bool(), tip_valid=obs["tip_valid"].bool(), owner_valid=obs["owner_valid"].bool(),
        )  # 无critic_*或资产ID特征
        prediction = actor(actor_obs, geometry)  # 同一参数点的可微策略均值
        mean = prediction.mean  # 有界动作中心[B_chunk,16]
        logstd = prediction.log_std.expand_as(mean)  # 共享潜标准差的逐slot视图
        actions = data["actions"][start:stop].to(device)  # 原采样动作，不生成新动作
        per_joint = PalmRotationMaskedContinuousModel.Network._squashed_per_joint_neglogp(
            actions, mean, logstd.exp(), logstd
        )  # 原概率实现含tanh Jacobian与边界epsilon
        new_neglogp = (per_joint * actor_obs.jnt_valid).sum(-1)  # ghost不进入整手logp
        old_neglogp = data["old_logp_actions"][start:stop].to(device)  # rollout冻结的负logp
        max_logp_error = max(max_logp_error, float((new_neglogp.detach() - old_neglogp).abs().max()))
        max_mean_error = max(max_mean_error, float((mean.detach() - data["rollout_mu"][start:stop].to(device)).abs().max()))
        if not disable_tf32:  # 同精度模式必须通过原记录的重放一致性门
            assert max_mean_error < 5e-4 and max_logp_error < 0.02, (max_mean_error, max_logp_error)
        else:  # 显式精度干预允许输出变化；要求仍处于原PPO未裁剪邻域，不冒充原精度重放
            assert max_logp_error < float(np.log1p(epsilon)), max_logp_error  # rho保持在[1/(1+eps),1+eps]内
        ratio = torch.exp(old_neglogp - new_neglogp)  # $\rho=\pi_\theta/\pi_{rollout}$，正负号与生产PPO一致
        local_halves = halves[start:stop].to(device)  # 当前块的replica membership
        for mode_index in range(3):  # 每种scope共享同一个前向图
            advantage = advantages[mode_index, start:stop].to(device)  # 本模式的冻结逐样本权重
            loss = -torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)  # clipped surrogate
            for half in (0, 1):  # 相同轨迹half、相同全局分母
                objective = loss[local_halves == half].sum() / int(half_counts[half])  # 分块loss可线性相加
                partial = torch.autograd.grad(objective, parameters, retain_graph=not (mode_index == 2 and half == 1), allow_unused=True)
                flat = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for p, g in zip(parameters, partial, strict=True)])  # unused branch保留零坐标
                gradients[mode_index, half] += flat.detach()  # 仅写探针持有的梯度数组
        free, _ = torch.cuda.mem_get_info(device)  # driver余量包括非PyTorch占用
        min_free = free if min_free is None else min(min_free, free)  # 全部块的最小已采样余量
        assert free >= 2 * 1024**3, free  # 本机资源边界
    assert torch.isfinite(gradients).all()  # 非有限梯度不能进入方向比较
    assert all(parameter.grad is None for parameter in parameters)  # autograd.grad未污染生产梯度slot
    for name, value in package.state_dict().items():  # 前向与微分后模型逐值保持
        assert torch.equal(value.cpu(), state[name]), name
    aggregate = (gradients * half_counts.to(device)[None, :, None]).sum(1) / count  # 按实际half样本数合成
    result = {  # 只输出压缩结果；完整梯度写NPZ
        "update": int(packet["update"]), "policy_version": int(packet["policy_version"]),
        "sample_count": count, "asset_count": asset_count, "half_counts": half_counts.tolist(),
        "actor_parameter_count": size, "normalization_max_abs_error": normalization_error,
        "recorded_allow_tf32": bool(cfg["allow_tf32"]), "probe_allow_tf32": allow_tf32,
        "recorded_precision_replay_gate_passed": max_mean_error < 5e-4 and max_logp_error < 0.02,
        "mean_replay_max_abs_error": max_mean_error, "neglogp_replay_max_abs_error": max_logp_error,
        "scopes": {mode: {"half_cosine": cosine(gradients[i, 0], gradients[i, 1]), "half_norms": gradients[i].norm(dim=1).cpu().tolist(), "aggregate_norm": float(aggregate[i].norm())} for i, mode in enumerate(modes)},
        "aggregate_cross_cosines": {"local_vs_center_only": cosine(aggregate[0], aggregate[1]), "local_vs_global": cosine(aggregate[0], aggregate[2]), "center_only_vs_global": cosine(aggregate[1], aggregate[2])},
        "relative_local_scaling_q05_q50_q95": torch.quantile(1/(standard_deviations+1e-8), torch.tensor([.05,.5,.95])).tolist(),
        "minimum_driver_free_bytes": min_free, "wall_seconds": time.perf_counter() - started,
        "model_unchanged": True, "optimizer_steps": 0, "new_environment_steps": 0,
    }
    np.savez_compressed(output_dir / f"u{packet['update']:06d}-gradients.npz", gradients=gradients.cpu().numpy(), aggregate=aggregate.cpu().numpy(), modes=np.asarray(modes), parameter_names=np.asarray(parameter_names))  # 保存坐标和原始方向
    return result  # 聚合者仅解释此条件批次的梯度，不自动选择训练配置


def main() -> None:
    r"""执行明确指定的保存批次，发布全新结果目录。"""
    parser = argparse.ArgumentParser(description=__doc__)  # 无仿真或训练启动参数
    parser.add_argument("--packet", action="append", required=True, type=Path)  # 可比较多个已声明更新点
    parser.add_argument("--output_dir", required=True, type=Path)  # 新证据目录
    parser.add_argument("--device", default="cuda:0")  # 主代理负责独占GPU
    parser.add_argument("--chunk_size", default=512, type=int)  # 只控制activation内存
    parser.add_argument("--disable_tf32", action="store_true")  # 仅用于判断梯度结论的数值敏感性
    args = parser.parse_args()  # 每次探针的输入必须显式
    if args.output_dir.exists() or args.chunk_size < 1:
        parser.error("output_dir must be new and chunk_size positive")  # 不覆盖既有结果
    args.output_dir.mkdir(parents=True)  # 调用方已确认case父目录
    torch.set_num_threads(1)  # 限制CPU读批次/归约争用
    device = torch.device(args.device)  # 当前实现使用CUDA driver余量门
    if device.type != "cuda":
        parser.error("this bounded probe requires an explicitly assigned CUDA device")
    results = [probe(path.resolve(), args.output_dir, device=device, chunk_size=args.chunk_size, disable_tf32=args.disable_tf32) for path in args.packet]
    report = {"artifact_type": "saved_batch_advantage_scope_probe", "scope": "Full Actor clipped-surrogate gradients; no regularizers/Adam or physical sampling; replica halves share full-rollout normalization statistics", "packets": [str(p.resolve()) for p in args.packet], "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")  # 统计口径与路径一起保存
    for row in results:
        print(json.dumps({"update": row["update"], "scopes": row["scopes"], "aggregate_cross_cosines": row["aggregate_cross_cosines"]}))  # 不打印大张量


if __name__ == "__main__":
    main()  # standalone只读探针
