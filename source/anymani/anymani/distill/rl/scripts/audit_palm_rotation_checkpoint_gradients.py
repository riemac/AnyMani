#!/usr/bin/env python3
r"""在冻结checkpoint上审计异构PPO Actor梯度的跨rollout可靠性。

训练期``full_gradient_shadow``把每个资产的replicas按奇偶分半，回答同一rollout内部的方向自一致性；但它若只在
fresh initialization运行，不能判断技能形成后跨资产负余弦是否变成稳定冲突。本入口从一个完整PPO checkpoint
恢复Actor、Critic、value normalizer与概率分布，生成$R$个相互独立的30步stochastic rollouts，并在每个rollout
的第一个stratified minibatch上复用生产代码计算：

$$
g_{ihr}^{(s)}=\nabla_{\theta^a}\,\frac{1}{|\mathcal B_{ihr}|}
\sum_{x\in\mathcal B_{ihr}}\mathcal L_{\mathrm{PPO}}^{(s)}(x),
\qquad s\in\{\mathrm{global},\mathrm{per\_asset}\}.
$$

其中$i$是资产、$h$是replica half、$r$是独立rollout。脚本既不调用optimizer step，也不写原训练run；每次
审计前后逐参数验证Actor完全相等。跨rollout余弦比单次Gram更接近PCGrad类方法所需的前提：只有同一资产的
$g_{ir}$在不同采样下方向稳定，跨资产负余弦才可解释为任务冲突，而不是有限样本符号噪声。

实现通过进程内替换``PalmRotationPpoAgent.train``，只复用正式Runner的checkpoint identity gate、rollout、GAE、
逐资产advantage与PPO objective。训练实现文件不被修改；审计输出拥有独立目录与脚本SHA-256。窄
``torch.compile``在该进程内降为eager，因为AOTAutograd donated buffers不允许同一图上的多次
``autograd.grad(retain_graph=True)``。这只改变诊断执行方式，不改变被审计checkpoint。

``--family_groups --full_rollout``按LEAP/Allegro分组，分别覆盖Actor与Critic全部可训练参数。每个microbatch
只形成两族×replica两半的梯度和，完整H30归约后复用SSL的两任务解析FairGrad。候选只用于局部进展与
跨rollout稳定性比较；该模式在生产loss形成后、主backward之前退出，不写parameter.grad或执行optimizer。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _arguments() -> argparse.Namespace:
    r"""解析冻结checkpoint、cohort与独立rollout数量。

    ``rollouts=4``产生6个跨rollout pair；相对只比较两次，它能暴露单次偶然同向，同时仍只需一次Isaac启动和
    $4\times30$个policy steps。输出目录必须尚不存在，避免把不同checkpoint的梯度证据混入同一artifact。

    Returns:
        argparse.Namespace: 经范围验证的命令行参数。
    """

    parser = argparse.ArgumentParser(description=__doc__)  # 长模块说明同时作为CLI科研语义
    parser.add_argument("--checkpoint", type=Path, required=True, help="完整schema-3 PPO checkpoint。")
    parser.add_argument("--cohort_lock", type=Path, required=True, help="checkpoint训练时的canonical cohort lock。")
    parser.add_argument("--output", type=Path, required=True, help="全新、独立的gradient-audit输出目录。")
    parser.add_argument("--rollouts", type=int, default=4, help="冻结参数下独立H30 rollout数量；正式诊断默认4。")
    parser.add_argument("--seed_offset", type=int, default=10000, help="相对训练seed的诊断随机流偏移。")
    parser.add_argument("--warmup_steps", type=int, default=0, help="无更新预运行步数；0测冷启动，570测28.5–30秒占据。")
    parser.add_argument("--full_rollout", action="store_true", help="按half样本数合并全部分层minibatch的只读梯度。")
    parser.add_argument(
        "--family_groups", action="store_true", help="完整H30上的LEAP/Allegro Actor/Critic与FairGrad候选。"
    )
    args = parser.parse_args()
    if args.rollouts < 2:  # 单一rollout只有half consistency，不能形成跨rollout可靠性证据
        parser.error("--rollouts must be at least 2")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps must be non-negative")
    if args.family_groups and not args.full_rollout:
        parser.error("--family_groups requires --full_rollout")  # 不把一个激活切片冒充两族完整采样总体。
    if args.output.expanduser().exists():  # 不复用目录，防止新旧NPZ看似属于同一冻结参数总体
        parser.error("--output must not already exist")
    return args


def _sha256(path: Path) -> str:
    r"""计算输入或产出artifact的逐字节SHA-256。"""

    digest = hashlib.sha256()  # artifact identity，不使用mtime或文件名替代内容
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    r"""以temporary→replace发布JSON，避免进程退出留下可见的半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)  # 输出根由本次只读checkpoint probe独占
    temporary = path.with_suffix(path.suffix + ".tmp")  # 同文件系统保证replace原子性
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""沿最后一维计算余弦，任一零范数样本返回0而不伪造稳定方向。

    Args:
        left (np.ndarray): 形状$[...,P]$的梯度向量。
        right (np.ndarray): 与``left``逐项对齐的梯度向量。

    Returns:
        np.ndarray: 形状$[...]$的余弦值。
    """

    numerator = np.sum(left * right, axis=-1)  # $g_l^\top g_r$
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)  # $\|g_l\|\|g_r\|$
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1.0e-20)


def _quantiles(values: np.ndarray) -> list[float]:
    r"""返回min/q25/median/q75/max，固定报告轴便于不同checkpoint直接比较。"""

    return [float(value) for value in np.quantile(values.astype(np.float64), (0.0, 0.25, 0.5, 0.75, 1.0))]


def _family_half_gradient_sums(
    objective: torch.Tensor,
    groups: torch.Tensor,
    halves: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""返回两族两半的完整参数梯度和及样本数，不写主梯度。

    生产helper给出每个subset的均值梯度$g_{ihm}$；乘回$n_{ihm}$后以FP64跨microbatch相加，最后只除
    一次完整分母。这样每个microbatch的replica parity不均衡也不会改变完整H30估计量。

    Returns:
        tuple: FP64梯度和`[2,2,P]`与整数样本数`[2,2]`；参数声明顺序定义固定坐标。
    """

    from anymani.distill.rl.algorithms.gradient_audit import per_asset_replica_half_gradients  # noqa: PLC0415

    gradients, counts = per_asset_replica_half_gradients(
        objective, groups, halves, parameters, asset_count=2
    )  # 沿用生产的autograd.grad、unused参数零填充和half合法性检查。
    return gradients.detach().double() * counts[..., None], counts  # $\sum_x\nabla L(x)$，不保留计算图。


def _family_gradient_evidence(
    half_sums: torch.Tensor, half_counts: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    r"""归约完整两族梯度，比较普通平均、SSL FairGrad及等范数FairGrad候选。

    输入为`[2,2,P]`的梯度和与`[2,2]`计数。两族样本总数必须相等，普通平均才能同时表示原始PPO总体。
    $g_i=\sum_h s_{ih}/\sum_h n_{ih}$；候选$d$的$g_i^T d$是SGD方向下的一阶loss下降率，不是Adam实际
    步进或物理能力保证。保留FairGrad原始尺度，同时报告匹配$\|(g_0+g_1)/2\|$后的方向以分开尺度效应。
    """

    from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import combine_fairgrad  # noqa: PLC0415
    from anymani.distill.rl.algorithms.gradient_audit import compute_actor_gradient_scope_audit  # noqa: PLC0415

    if half_sums.ndim != 3 or half_sums.shape[:2] != (2, 2) or half_counts.shape != (2, 2):
        raise ValueError("family gradient sums/counts require [2,2,P] and [2,2]")
    if bool((half_counts <= 0).any()) or not bool(torch.isfinite(half_sums).all()):
        raise ValueError("family gradient evidence requires finite sums and positive half counts")
    if int(half_counts[0].sum()) != int(half_counts[1].sum()):
        raise ValueError("family probe requires balanced full-rollout group sample counts")
    halves = half_sums.double() / half_counts[..., None]  # 每族、每half的样本均值梯度。
    audit = compute_actor_gradient_scope_audit(halves, half_counts)  # 此处asset轴表示两族，而非256个成员。
    gradients = audit["asset_gradients"]  # `[2,P]`；Actor与Critic在各自调用内独立归约。
    mean = gradients.mean(dim=0)  # 原始均匀采样PPO在该冻结参数点的完整总体方向。
    result = combine_fairgrad((gradients[0],), (gradients[1],))  # flatten不改变Euclidean梯度几何。
    fairgrad = result.combined[0]
    if fairgrad is None:
        fairgrad = torch.zeros_like(mean)  # SSL的零梯度/近反向政策，状态在evidence中明确记录。
    fairgrad_norm = float(torch.linalg.vector_norm(fairgrad))
    mean_norm = float(torch.linalg.vector_norm(mean))
    matched = fairgrad * (mean_norm / fairgrad_norm) if fairgrad_norm > 0.0 else torch.zeros_like(mean)
    directions = {"mean": mean, "fairgrad": fairgrad, "fairgrad_mean_norm": matched}
    evidence = {
        "group_names": ["leap", "allegro"],
        "half_counts": half_counts.cpu().tolist(),
        "group_norms": audit["asset_norms"].cpu().tolist(),
        "gram": audit["asset_gram"].cpu().tolist(),
        "cosine": float(audit["asset_cosine"][0, 1]),
        "half_self_cosine": audit["half_self_cosine"].cpu().tolist(),
        "fairgrad": {
            "weights": [result.evidence.density_weight, result.evidence.kappa_weight],
            "active_tasks": result.evidence.active_tasks,
            "shared_conflict_blocked": result.evidence.shared_conflict_blocked,
            "near_opposition_tolerance": 1.0e-6,
            "scale_convention": "SSL analytic alpha=1; paper c=1; nondegenerate norm=sqrt(2)",
        },
        "candidates": {
            name: {
                "norm": float(torch.linalg.vector_norm(direction)),
                "group_loss_decrease_rates": (gradients @ direction).cpu().tolist(),
            }
            for name, direction in directions.items()
        },
        "projection_scope": "raw gradient space before clipping, Adam and parameter-group learning rates",
    }
    dense = {
        "half_gradients": halves,
        "half_counts": half_counts,
        "group_gradients": gradients,
        "half_self_cosine": audit["half_self_cosine"],
        **{f"{name}_direction": direction for name, direction in directions.items()},
    }
    return dense, evidence


def _cross_rollout_summary(npz_paths: Sequence[Path], *, family_groups: bool = False) -> dict[str, Any]:
    r"""从$R$份完整梯度形成同资产与合成Actor方向的跨rollout统计。

    对每个scope先从NPZ读取$G_r\in\mathbb R^{A\times P}$。每对rollout$(r,s)$计算逐资产
    $\cos(g_{ir},g_{is})$及等权合成方向$\cos(\sum_i g_{ir},\sum_i g_{is})$。除全体pair分位数外，另报告
    每资产pair-median为正和pair-q25为正的比例；后者要求至少75%的rollout pairs同向，是更保守的可靠性门。

    Args:
        npz_paths (Sequence[Path]): 每个独立rollout的``full-actor-gradient-shadow-v1``数组。

    Returns:
        dict[str, Any]: JSON-safe跨rollout证据及每个输入artifact摘要。
    """

    arrays = [np.load(path) for path in npz_paths]  # 每项含两种advantage scope的$[A,P]$完整梯度
    try:
        result: dict[str, Any] = {  # 输入哈希使summary不依赖目录名猜测
            "rollout_count": len(arrays),
            "rollout_pair_count": len(tuple(itertools.combinations(range(len(arrays)), 2))),
            "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in npz_paths],
            "scopes": {},
        }
        axis = "family" if family_groups else "asset"  # 两族模式不把group统计命名为逐资产证据。
        scopes = ("actor", "critic") if family_groups else ("global", "per_asset_rollout")
        for scope in scopes:
            gradient_key = f"{scope}_{'group' if family_groups else 'asset'}_gradients"
            gradients = [array[gradient_key].astype(np.float64) for array in arrays]  # $R\times[A,P]$或$R\times[2,P]$
            if any(value.shape != gradients[0].shape for value in gradients[1:]):
                raise RuntimeError(f"{scope} gradient arrays do not share one [A,P] coordinate system")

            # 所有无序rollout pairs共享同一冻结参数坐标；逐资产余弦保留$[K,A]$而不先混合资产。
            per_pair_asset = []  # $K=R(R-1)/2$份`[A]`同资产重复采样余弦
            aggregate_pairs = []  # 每个pair对比等权资产合成方向$\sum_i g_i$
            for left_index, right_index in itertools.combinations(range(len(gradients)), 2):
                left = gradients[left_index]  # `[A,P]`，冻结checkpoint上的第$r$次rollout
                right = gradients[right_index]  # `[A,P]`，独立action/minibatch随机流
                per_pair_asset.append(_cosine(left, right))  # `[A]`，不跨资产平均
                aggregate_pairs.append(float(_cosine(left.sum(axis=0), right.sum(axis=0))))  # scalar Actor update方向
            pair_asset = np.stack(per_pair_asset, axis=0)  # `[K,A]`
            asset_pair_median = np.median(pair_asset, axis=0)  # `[A]`，每资产跨rollout中心方向
            asset_pair_q25 = np.quantile(pair_asset, 0.25, axis=0)  # `[A]`，保守一致性下四分位

            # 同一rollout内部even/odd replica halves仍单独报告，避免跨rollout改善掩盖half样本不足。
            half_self = np.stack(
                [array[f"{scope}_half_self_cosine"].astype(np.float64) for array in arrays], axis=0
            )  # `[R,A]`
            result["scopes"][scope] = {
                f"{axis}_cross_rollout_cosine_min_q25_median_q75_max": _quantiles(pair_asset),
                f"{axis}_pair_median_min_q25_median_q75_max": _quantiles(asset_pair_median),
                f"{axis}_fraction_positive_pair_median": float(np.mean(asset_pair_median > 0.0)),
                f"{axis}_fraction_positive_pair_q25": float(np.mean(asset_pair_q25 > 0.0)),
                "aggregate_cross_rollout_cosines": aggregate_pairs,
                "aggregate_cross_rollout_cosine_median": float(np.median(aggregate_pairs)),
                "within_rollout_half_self_cosine_min_q25_median_q75_max": _quantiles(half_self),
                "within_rollout_positive_half_fraction": float(np.mean(half_self > 0.0)),
            }
            if family_groups:
                result["scopes"][scope]["family_cross_rollout_cosines"] = pair_asset.tolist()
                result["scopes"][scope]["candidate_cross_rollout_cosines"] = {
                    name: [
                        float(
                            _cosine(
                                arrays[left][f"{scope}_{name}_direction"], arrays[right][f"{scope}_{name}_direction"]
                            )
                        )
                        for left, right in itertools.combinations(range(len(arrays)), 2)
                    ]
                    for name in ("mean", "fairgrad", "fairgrad_mean_norm")
                }  # 跨采样稳定性与单次正投影分开报告。
        return result
    finally:
        for array in arrays:
            array.close()  # mmap/file descriptors不跨summary发布边界存活


def _training_argv(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> list[str]:
    r"""由checkpoint自身训练合同重建identity-compatible launcher参数。

    诊断不允许调用者手填arm、环境数或minibatch布局；这些字段全部来自checkpoint，避免在加载后才发现probe
    population已改变。``max_updates=epoch+1``只满足Runner配置正值，进程内替换的``train``不会执行该update。

    Returns:
        list[str]: 交给正式训练入口parser的完整参数序列。
    """

    identity = checkpoint.get("anymani_identity")
    if not isinstance(identity, Mapping) or identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
        raise RuntimeError("gradient audit requires a schema-3/4 palm-rotation checkpoint")
    training = identity.get("training")
    policy = identity.get("policy")
    if not isinstance(training, Mapping) or not isinstance(policy, Mapping):
        raise RuntimeError("checkpoint is missing policy/training identity")
    minibatch_count = int(training["minibatch_count"])  # 生产stratified activation slices数量
    argv = [
        "anymani.distill.rl.train_palm_rotation_mvp",
        "--headless",
        "--rl_games_strict",
        "--cohort_lock",
        str(args.cohort_lock.expanduser().resolve()),
        "--num_envs",
        str(int(training["num_envs"])),
        "--max_updates",
        str(int(checkpoint["epoch"]) + 1),
        "--reward_release_start_turns",
        str(float(training.get("reward_release_start_turns", 1.0))),
        "--reward_release_end_turns",
        str(float(training.get("reward_release_end_turns", 2.0))),
        "--actor_contact",
        str(training.get("actor_contact", "all")),
        "--episode_seconds_min",
        str(float(training.get("episode_seconds_min", 120.0))),
        "--episode_seconds_max",
        str(float(training.get("episode_seconds_max", 120.0))),
        "--reward_release_floor",
        str(float(training.get("reward_release_floor", 0.0))),
        "--reward_release_reference_seconds",
        str(float(training.get("reward_release_reference_seconds", 120.0))),
        "--learning_rate",
        str(float(training.get("actor_base_lr", 3.0e-4))),
        "--rotation_progress_clip_rad",
        str(float(training.get("rotation_progress_clip_rad_per_step", 0.025))),
        "--strict_goal_reward_weight",
        str(float(training.get("strict_goal_reward_weight", 10.0))),
        "--joint_pose_anchor_weight",
        str(float(training.get("joint_pose_anchor_weight", -0.5))),
        "--seed",
        str(int(training["seed"])),
        "--minibatches",
        str(minibatch_count),
        "--gradient_accumulation_steps",
        str(int(training["gradient_accumulation_steps"])),
        "--gradient_probe_frequency",
        str(int(training["gradient_probe_frequency"])),
        "--full_gradient_shadow_frequency",
        str(int(training["full_gradient_shadow_frequency"])),
        "--advantage_normalization_scope",
        str(training["advantage_normalization_scope"]),
        "--arm",
        str(policy["arm"]),
        "--history_encoder",
        str(training["history_encoder"]),
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--experiment_name",
        args.output.expanduser().resolve().name,
        "--device",
        str(training.get("device", "cuda:0")),
    ]
    if bool(training.get("allow_tf32", False)):
        argv.append("--tf32")  # 数值模式仍与被审计checkpoint一致
    compile_mode = training.get("torch_compile")
    if compile_mode is not None:
        argv.extend(("--torch_compile", str(compile_mode)))  # 只用于identity重建，实际forward在进程内转eager
    return argv


def run_checkpoint_audit(
    args: argparse.Namespace,
    audit_callback: Callable[[Any, argparse.Namespace, Mapping[str, Any]], None] | None = None,
) -> None:
    r"""按原始identity恢复Agent，在独立目录执行冻结检查点诊断。

    默认运行完整梯度审计；显式callback可复用同一资产/模型恢复门做动作响应等诊断。Callback负责采样与
    参数不变校验，且只向args.output写summary，不执行训练或覆盖source run。
    """

    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)  # exact source artifact
    cohort_path = args.cohort_lock.expanduser().resolve(strict=True)  # exact support/geometry order
    output_root = args.output.expanduser().resolve()  # 全新probe-owned目录
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # App启动前只读metadata/state
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint root must be a mapping")
    checkpoint_state = checkpoint  # nested log-dir hook需访问recorder inventory，避免与其checkpoint路径参数同名

    # 正式入口在module import时解析sys.argv并启动AppLauncher；先用checkpoint合同重写参数，杜绝CLI语义漂移。
    sys.argv = _training_argv(args, checkpoint)
    original_compile = torch.compile  # probe结束后恢复进程全局函数，即使当前进程随后立即关闭Kit
    torch.compile = lambda function, **_kwargs: function  # type: ignore[assignment]  # 多次autograd必须使用eager图

    import anymani.distill.rl.train_palm_rotation_mvp as training_entry  # noqa: PLC0415  # AppLauncher顺序真源
    from anymani.distill.rl.algorithms.gradient_audit import (  # noqa: PLC0415
        compute_actor_gradient_scope_audit,
        per_asset_replica_half_gradients,
    )
    from anymani.distill.rl.palm_rotation_ppo import (  # noqa: PLC0415
        PALM_ROTATION_PPO_ALGO,
        PalmRotationPpoAgent,
    )
    from anymani.distill.rl.runtime import palm_rotation_probes as probe_runtime  # noqa: PLC0415

    checkpoint_epoch = int(checkpoint["epoch"])  # 所有repeat均锚定同一冻结update
    training_seed = int(checkpoint["anymani_identity"]["training"]["seed"])  # 正式训练随机种子

    def configure_probe_dir(agent_cfg: dict[str, Any], *, checkpoint: str | None) -> tuple[Path, Path]:
        r"""把Runner/TensorBoard/params重定向到probe目录并镜像checkpoint的只读shard inventory。

        Full checkpoint restore会先核对``anymani_metrics_recorder``中每个Parquet shard的名称与SHA；这是防止训练
        run错误续接的必要门。Probe不能把writer留在source run，因此在独立目录为checkpoint声明的shards建立
        hard links。它们共享不可变inode、不复制数据且不写源文件；probe自身也从不调用recorder ``record``。
        """

        if checkpoint is None:
            raise RuntimeError("checkpoint gradient audit cannot run without full checkpoint restore")
        output_root.joinpath("params").mkdir(parents=True, exist_ok=False)  # output必须由本次probe首次创建
        source_run = checkpoint_path.parent.parent  # `<source-run>/nn/checkpoint.pth` -> exact shard owner
        source_shards = source_run / "metrics_shards"  # checkpoint recorder state所声明的不可变Parquet集合
        probe_shards = output_root / "metrics_shards"  # restore gate在重定向run目录中查找的镜像集合
        probe_shards.mkdir()
        recorder_state = checkpoint_state.get("anymani_metrics_recorder")
        if not isinstance(recorder_state, Mapping) or not isinstance(recorder_state.get("shards"), Sequence):
            raise RuntimeError("checkpoint gradient audit requires recorder shard inventory")
        for entry in recorder_state["shards"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                raise RuntimeError("checkpoint recorder contains a malformed shard entry")
            source = source_shards / entry["name"]  # exact checkpoint-declared source shard
            destination = probe_shards / entry["name"]  # 独立run中的同名只读hard link
            os.link(source, destination)  # 原inode不可变；restore随后逐SHA再次验证全部内容
        config = agent_cfg["params"]["config"]  # rl_games resolved运行配置
        config["train_dir"] = str(output_root.parent)  # writer根；不触碰source run
        config["full_experiment_name"] = output_root.name  # event与agent目录都落在probe-owned root
        config["rl_games_backend_file"] = str(training_entry.backend_info.package_file)
        config["rl_games_backend_commit"] = training_entry.backend_info.git_commit
        return output_root.parent, output_root

    def audit_train(agent: PalmRotationPpoAgent) -> None:
        r"""替代训练循环：只形成rollout objectives与完整梯度，不执行任何optimizer step。"""

        # rl_games把experience buffer、current episode accumulators与``dones``延迟到``train()``入口创建；
        # 本诊断覆盖了该入口，故必须显式执行同一初始化，不能只恢复checkpoint中的model/optimizer state。
        agent.init_tensors()
        vec_env = agent.vec_env
        if vec_env is None:
            raise RuntimeError("gradient audit requires the constructed vector environment")

        # 参数快照覆盖Actor与Critic；value RMS等buffers由完整model snapshot在每个repeat前恢复。
        frozen_model = {
            name: tensor.detach().clone() for name, tensor in agent.model.state_dict().items()
        }  # GPU-resident exact checkpoint state，总量远小于rollout buffer
        frozen_actor = [
            parameter.detach().clone() for parameter in agent.model.a2c_network.package.actor.parameters()
        ]  # 逐参数bitwise不变门
        frozen_critic = [
            parameter.detach().clone() for parameter in agent.model.a2c_network.package.critic.parameters()
        ]
        frozen_environment = deepcopy(vec_env.get_env_state())  # 恢复课程/diagnostic counter，不声称恢复PhysX状态
        print("[gradient-audit] checkpoint parameter snapshot complete", flush=True)
        original_epoch = int(agent.epoch_num)  # restore后的真实checkpoint update
        original_accumulation = int(agent._gradient_accumulation_steps)  # 正式值通常1；probe临时设2阻止step边界
        original_probe_frequency = int(agent.config.get("gradient_probe_frequency", 0))
        original_shadow_frequency = int(agent.config.get("full_gradient_shadow_frequency", 0))
        npz_paths: list[Path] = []  # $R$个full-gradient dense artifacts
        populations: list[dict[str, Any]] = []  # 记录实际episode age，区分elapsed-time与持续存活总体
        original_shadow = probe_runtime.run_full_actor_gradient_shadow
        original_probe = probe_runtime.run_gradient_probe
        half_sums: dict[str, torch.Tensor] = {}
        count_sums: dict[str, torch.Tensor] = {}
        shadow_calls = 0
        active_halves: torch.Tensor | None = None  # 当前slice的parity，仅供进程内probe闭包读取。
        family_reports: list[dict[str, Any]] = []
        minimum_driver_free: int | None = None
        family_by_asset: torch.Tensor | None = None
        family_counts: list[int] = []
        parameter_layout: dict[str, Any] = {}
        combiner_source_sha256: str | None = None
        if args.family_groups:
            from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import (  # noqa: PLC0415
                combine_fairgrad,
            )

            cohort = json.loads(cohort_path.read_text(encoding="utf-8"))  # 使用已通过Runner identity的同一成员锁。
            group_ids = {"single_palm_leap": 0, "single_palm_allegro": 1}
            names = [member["provenance"]["group_name"] for member in cohort["members"]]
            if len(names) != agent.asset_count or set(names) != set(group_ids):
                raise RuntimeError("family audit requires the complete pure LEAP/Allegro training cohort")
            family_counts = [names.count(name) for name in group_ids]
            if family_counts[0] != family_counts[1] or (agent.num_actors // agent.asset_count) % 2:
                raise RuntimeError("family audit requires balanced families and an even replica count")
            family_by_asset = torch.tensor([group_ids[name] for name in names], device=agent.ppo_device)
            combiner_source_sha256 = _sha256(Path(combine_fairgrad.__code__.co_filename))
            for side in ("actor", "critic"):
                module = getattr(agent.model.a2c_network.package, side)
                parameter_layout[side] = [
                    {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
                    for name, parameter in module.named_parameters()
                    if parameter.requires_grad
                ]  # flatten边界保留，后续不得把Actor与Critic坐标混合。

        class FamilyForwardCaptured(Exception):
            r"""只读family probe在生产loss形成后、主backward前结束当前调用。"""

        def accumulate_family_probe(
            current_agent: Any,
            *,
            actor_objective: torch.Tensor,
            critic_objective: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
            r"""复用生产的逐样本loss，累计两族两半的完整参数梯度和。

            Actor已包含当前优势scope、entropy和bounds；Critic已包含0.5系数、value clip和critic_coef。
            此处不重写loss，也不把单个microbatch的FairGrad方向相加来冒充完整采样段FairGrad。
            """

            nonlocal shadow_calls
            if family_by_asset is None or active_halves is None:
                raise RuntimeError("family gradient probe lacks frozen grouping/parity metadata")
            groups = family_by_asset[labels]  # `[M]`，不进入Actor或Critic的forward输入。
            for side, objective in (("actor", actor_objective), ("critic", critic_objective)):
                module = getattr(current_agent.model.a2c_network.package, side)
                parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
                sums, counts = _family_half_gradient_sums(objective, groups, active_halves, parameters)
                if side not in half_sums:
                    half_sums[side], count_sums[side] = sums, counts.clone()
                else:
                    half_sums[side].add_(sums)
                    count_sums[side].add_(counts)
            shadow_calls += 1
            raise FamilyForwardCaptured  # 不执行生产calc_gradients后续的backward、clipping或optimizer。

        def accumulate_shadow(
            current_agent: Any,
            *,
            global_objective: torch.Tensor,
            per_asset_objective: torch.Tensor,
            labels: torch.Tensor,
            replica_halves: torch.Tensor,
        ) -> None:
            r"""按$n_{imh}$加权合并microbatch均值梯度，得到完整rollout的$g_{ih}$。

            各slice的half数量不必相等；仅平均已有cosine或不加权平均half梯度都会改变估计量。
            """

            nonlocal shadow_calls
            parameters = tuple(current_agent.model.a2c_network.package.actor.parameters())
            for scope, objective in (("global", global_objective), ("per_asset_rollout", per_asset_objective)):
                gradients, counts = per_asset_replica_half_gradients(
                    objective,
                    labels,
                    replica_halves,
                    parameters,
                    asset_count=current_agent.asset_count,
                )
                weighted = gradients * counts[..., None]  # sum-objective梯度，`[A,2,P]`
                if scope not in half_sums:
                    half_sums[scope], count_sums[scope] = weighted, counts.clone()
                else:
                    half_sums[scope].add_(weighted)
                    count_sums[scope].add_(counts)
            shadow_calls += 1
            if shadow_calls != len(current_agent.dataset):
                return
            dense = {}
            scope_summary = {}
            expected_half_count = int(current_agent.batch_size) // (2 * current_agent.asset_count)
            for scope in half_sums:
                counts = count_sums[scope]
                if not bool((counts == expected_half_count).all()):
                    raise RuntimeError("full-rollout gradient counts do not match complete replica halves")
                gradients = half_sums[scope] / counts[..., None]
                audit = compute_actor_gradient_scope_audit(gradients, counts)
                dense[f"{scope}_half_gradients"] = gradients.cpu().numpy()
                dense.update({f"{scope}_{key}": value.cpu().numpy() for key, value in audit.items()})
                scope_summary[scope] = {
                    "half_self_cosine_median": float(audit["half_self_cosine"].median().item()),
                    "positive_half_fraction": float(audit["reliable_asset_fraction"].item()),
                    "aggregate_to_individual_norm_ratio": float(audit["aggregate_to_individual_norm_ratio"].item()),
                }
            path = Path(current_agent.experiment_dir) / "full_gradient_shadows"
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / f"update_{checkpoint_epoch:06d}.npz", **dense)
            _atomic_json(
                path / f"update_{checkpoint_epoch:06d}.json",
                {
                    "sample_population": "complete-H30-rollout-count-weighted-microbatches",
                    "activation_minibatches": shadow_calls,
                    "half_sample_count": expected_half_count,
                    "scopes": scope_summary,
                },
            )

        if args.family_groups:
            probe_runtime.run_gradient_probe = accumulate_family_probe
        elif args.full_rollout:
            probe_runtime.run_full_actor_gradient_shadow = accumulate_shadow

        try:
            for repeat_index in range(int(args.rollouts)):
                half_sums.clear()
                count_sums.clear()
                shadow_calls = 0
                # 每次从完全相同的模型/value-normalizer状态与环境reset边界开始，只改变诊断随机流。
                agent.model.load_state_dict(frozen_model, strict=True)
                vec_env.set_env_state(deepcopy(frozen_environment))
                agent.optimizer.zero_grad(set_to_none=True)
                agent.critic_optimizer.zero_grad(set_to_none=True)
                repeat_seed = training_seed + int(args.seed_offset) + repeat_index  # 独立Normal action/minibatch流
                torch.manual_seed(repeat_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(repeat_seed)
                print(f"[gradient-audit] rollout {repeat_index}: resetting environments", flush=True)
                agent.obs = agent.env_reset()  # 所有A×replicas重置到正式rank-0 pregrasp
                agent.dones.fill_(1)  # 与生产``init_current_rewards``一致：reset前状态是trajectory boundary
                agent.current_rewards.zero_()
                agent.current_shaped_rewards.zero_()
                agent.current_lengths.zero_()

                # 不调用额外play_steps或prepare_dataset：只推进原始动作/物理路径，不更新value RMS。
                resets = torch.zeros_like(agent.dones, dtype=torch.long)
                for _ in range(int(args.warmup_steps)):
                    values = agent.get_action_values(agent.obs)  # eval/no_grad，仍从同一随机策略采样
                    agent.obs, rewards, agent.dones, _infos = agent.env_step(values["actions"])
                    agent.current_rewards.add_(rewards)
                    agent.current_shaped_rewards.add_(agent.rewards_shaper(rewards))
                    agent.current_lengths.add_(1)
                    resets += agent.dones.long()
                    not_done = 1.0 - agent.dones.float()
                    agent.current_rewards.mul_(not_done.unsqueeze(1))
                    agent.current_shaped_rewards.mul_(not_done.unsqueeze(1))
                    agent.current_lengths.mul_(not_done)
                if not all(torch.equal(value, agent.model.state_dict()[name]) for name, value in frozen_model.items()):
                    raise RuntimeError("gradient-audit warmup changed model/value statistics")
                environment_now = vec_env.get_env_state()
                if not torch.equal(
                    environment_now["reward_release"]["env_lambda"], frozen_environment["reward_release"]["env_lambda"]
                ):
                    raise RuntimeError("gradient-audit warmup changed actual reward release")
                episode_ages = vec_env.env.unwrapped.episode_length_buf.detach().cpu()
                populations.append(
                    {
                        "warmup_steps": int(args.warmup_steps),
                        "episode_age_steps_at_h30_start": episode_ages.tolist(),
                        "warmup_resets_by_env": resets.cpu().tolist(),
                        "model_and_value_state_unchanged_after_warmup": True,
                    }
                )

                # ``play_steps``逐值复用生产H30 stochastic policy、reward、done、critic与GAE路径。
                print(f"[gradient-audit] rollout {repeat_index}: collecting H30", flush=True)
                batch = agent.play_steps()  # env-major flatten前的正式rollout batch
                agent._reset_optimization_metrics()
                agent.prepare_dataset(batch)  # global/per-asset GAE与stratified permutation
                print(f"[gradient-audit] rollout {repeat_index}: differentiating frozen objectives", flush=True)

                # Shadow在``calc_gradients``主backward之前运行；accumulation=2使第一个microbatch绝不触发step。
                repeat_root = output_root / f"rollout_{repeat_index:02d}"
                agent.experiment_dir = str(repeat_root)  # dense NPZ/JSON只写独立repeat目录
                agent.epoch_num = checkpoint_epoch  # artifact update含义始终是source checkpoint坐标
                agent.config["gradient_probe_frequency"] = 1 if args.family_groups else (0 if args.full_rollout else 1)
                agent.config["full_gradient_shadow_frequency"] = 0 if args.family_groups else 1
                agent._gradient_accumulation_steps = 2  # 当前只调用一个microbatch，故optimizer boundary不可达
                for minibatch_index in range(len(agent.dataset) if args.full_rollout else 1):
                    agent._gradient_microbatch_index = 0  # 每个slice均单独只读，永不进入第二个累积microstep
                    minibatch = agent.dataset[minibatch_index]
                    minibatch_halves = minibatch.get("replica_halves")
                    if not isinstance(minibatch_halves, torch.Tensor):
                        raise RuntimeError("gradient audit minibatch has no replica parity tensor")
                    active_halves = minibatch_halves  # 与同一slice的observations/actions/advantages对齐。
                    if args.family_groups:
                        try:
                            agent.calc_gradients(minibatch)
                        except FamilyForwardCaptured:
                            pass  # 仅消费loss图；正常return表示未进入只读hook，必须拒绝。
                        else:
                            raise RuntimeError("family probe did not stop before production backward")
                    else:
                        agent.calc_gradients(minibatch)
                    if agent._gradient_microbatch_index != (0 if args.family_groups else 1):
                        raise RuntimeError("gradient audit crossed an optimizer boundary")
                    if args.family_groups and torch.device(agent.ppo_device).type == "cuda":
                        free, total = torch.cuda.mem_get_info(agent.ppo_device)  # 包含PhysX与allocator外部占用。
                        minimum_driver_free = free if minimum_driver_free is None else min(minimum_driver_free, free)
                        required = int(agent.config.get("gpu_driver_free_memory_bytes_min", 0))
                        if free < required:
                            raise RuntimeError(
                                "palm-rotation training exhausted CUDA driver headroom: "
                                f"free={free}, required={required}, total={total}, scope=family-gradient-audit"
                            )  # 与正式训练保持相同安全门，父watchdog能识别该停止原因。
                if args.full_rollout and shadow_calls != len(agent.dataset):
                    raise RuntimeError("full-rollout shadow did not observe every minibatch")
                print(f"[gradient-audit] rollout {repeat_index}: gradient artifact complete", flush=True)
                if agent._gradient_microbatch_index != (0 if args.family_groups else 1):
                    raise RuntimeError("gradient audit unexpectedly executed more than one activation microbatch")

                if args.family_groups:
                    if any(parameter.grad is not None for parameter in agent.model.parameters()):
                        raise RuntimeError("family probe wrote a model parameter gradient")
                    replicas = agent.num_actors // agent.asset_count
                    expected_counts = torch.tensor(family_counts, device=agent.ppo_device)[:, None]
                    expected_counts = (expected_counts * (replicas // 2) * agent.horizon_length).expand(2, 2)
                    dense_family: dict[str, np.ndarray] = {}
                    report_family: dict[str, Any] = {}
                    for side in ("actor", "critic"):
                        if not torch.equal(count_sums[side], expected_counts):
                            raise RuntimeError("family gradient counts do not cover the full H30 population")
                        dense_side, report_family[side] = _family_gradient_evidence(half_sums[side], count_sums[side])
                        dense_family.update({f"{side}_{key}": value.cpu().numpy() for key, value in dense_side.items()})
                    family_root = repeat_root / "family_gradient_shadows"
                    family_root.mkdir(parents=True, exist_ok=True)
                    destination = family_root / f"update_{checkpoint_epoch:06d}.npz"
                    temporary = destination.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, **dense_family)
                    temporary.replace(destination)  # 完整梯度写完才发布可见artifact。
                    _atomic_json(family_root / f"update_{checkpoint_epoch:06d}.json", report_family)
                    family_reports.append(report_family)

                # 任何Actor参数变化都说明optimizer隔离失败；即刻拒绝发布summary。
                current_actor = tuple(agent.model.a2c_network.package.actor.parameters())
                if not all(
                    torch.equal(parameter.detach(), reference)
                    for parameter, reference in zip(current_actor, frozen_actor, strict=True)
                ):
                    raise RuntimeError("checkpoint gradient audit changed Actor parameters")
                if not all(
                    torch.equal(parameter.detach(), reference)
                    for parameter, reference in zip(
                        agent.model.a2c_network.package.critic.parameters(), frozen_critic, strict=True
                    )
                ):
                    raise RuntimeError("checkpoint gradient audit changed Critic parameters")
                shadow_directory = "family_gradient_shadows" if args.family_groups else "full_gradient_shadows"
                npz_path = repeat_root / shadow_directory / f"update_{checkpoint_epoch:06d}.npz"
                if not npz_path.is_file():
                    raise RuntimeError("full Actor gradient shadow did not publish its dense artifact")
                npz_paths.append(npz_path)
        finally:
            probe_runtime.run_full_actor_gradient_shadow = original_shadow
            probe_runtime.run_gradient_probe = original_probe
            # 内存状态也恢复到checkpoint，确保后续清理/observer不会看见probe更新后的value RMS或gradients。
            agent.model.load_state_dict(frozen_model, strict=True)
            vec_env.set_env_state(deepcopy(frozen_environment))
            agent.optimizer.zero_grad(set_to_none=True)
            agent.critic_optimizer.zero_grad(set_to_none=True)
            agent.epoch_num = original_epoch
            agent._gradient_accumulation_steps = original_accumulation
            agent.config["gradient_probe_frequency"] = original_probe_frequency
            agent.config["full_gradient_shadow_frequency"] = original_shadow_frequency

        summary = {
            "schema_version": (
                "checkpoint-family-actor-critic-gradient-reliability-v1"
                if args.family_groups
                else "checkpoint-full-actor-gradient-reliability-v1"
            ),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_frame": int(checkpoint["frame"]),
            "method_identity_digest": checkpoint["anymani_identity"]["identity_digest"],
            "cohort_lock": str(cohort_path),
            "cohort_lock_sha256": _sha256(cohort_path),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "optimizer_steps": 0,
            "actor_parameters_frozen_exact": True,
            "critic_parameters_frozen_exact": True,
            "warmup_steps": int(args.warmup_steps),
            "full_rollout": bool(args.full_rollout),
            "rollout_populations": populations,
            "source_metric_shards_hardlinked": len(checkpoint["anymani_metrics_recorder"]["shards"]),
            "rollout_horizon": int(agent.horizon_length),
            "rollout_seeds": [training_seed + int(args.seed_offset) + index for index in range(int(args.rollouts))],
            "sample_population": (
                "independent-H30-after-untrained-warmup-full-rollout-count-weighted-even-odd-halves"
                if args.full_rollout
                else "independent-H30-after-declared-untrained-warmup-first-stratified-minibatch-even-odd-halves"
            ),
            "cross_rollout": _cross_rollout_summary(npz_paths, family_groups=bool(args.family_groups)),
        }
        if args.family_groups:
            summary.update(
                {
                    "family_groups": ["leap", "allegro"],
                    "family_asset_counts": family_counts,
                    "parameter_layout": parameter_layout,
                    "parameter_gradients_untouched": True,
                    "minimum_driver_free_bytes": minimum_driver_free,
                    "combiner_source_sha256": combiner_source_sha256,
                    "advantage_scope": agent.advantage_normalization_scope,
                    "family_rollout_reports": family_reports,
                    "candidate_optimizer_updates_executed": False,
                }
            )
        _atomic_json(output_root / "summary.json", summary)
        print(json.dumps({"output": str(output_root / "summary.json"), "optimizer_steps": 0}, sort_keys=True))

    class CheckpointGradientAuditAgent(PalmRotationPpoAgent):
        r"""显式覆盖Runner factory所实例化的Agent，使恢复epoch不会进入原训练终止门。"""

        def train(self) -> None:
            r"""把rl_games无参数训练入口转发到冻结梯度审计闭包。"""

            print("[gradient-audit] entered frozen checkpoint audit agent", flush=True)
            try:
                if audit_callback is None:
                    audit_train(self)
                else:
                    audit_callback(self, args, checkpoint_state)  # 复用恢复门；不进入原PPO train loop。
            except BaseException as error:
                print(
                    f"[gradient-audit] aborted by {type(error).__name__}: {error!r}",
                    flush=True,
                )  # ``SystemExit(0)``也必须留下阶段证据，不能伪装成probe成功
                raise

    original_runner_init = training_entry.PalmRotationPpoRunner.__init__  # factory注册发生在Runner构造期

    def audit_runner_init(runner: Any, *runner_args: Any, **runner_kwargs: Any) -> None:
        r"""在正式Runner完成默认注册后，以诊断Agent覆盖同名algorithm factory。"""

        original_runner_init(runner, *runner_args, **runner_kwargs)
        runner.algo_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: CheckpointGradientAuditAgent(**factory_kwargs),
        )  # ObjectFactory同名赋值覆盖默认PalmRotationPpoAgent，不改变network/model registries

    # 两个进程内替换均只改变本次probe orchestration；checkpoint identity仍由未修改的正式源码逐文件验证。
    training_entry._configure_log_dir = configure_probe_dir  # type: ignore[attr-defined]  # noqa: SLF001
    training_entry.PalmRotationPpoRunner.__init__ = audit_runner_init  # type: ignore[method-assign]
    try:
        try:
            training_entry.main()
        except RuntimeError as error:
            # 正式入口要求``metrics.parquet``；probe刻意不伪造训练metrics，只接受该唯一预期的尾部拒绝。
            expected = "palm-rotation Runner returned without any finalized PPO update metrics"
            if str(error) != expected or not (output_root / "summary.json").is_file():
                raise
    finally:
        torch.compile = original_compile  # type: ignore[assignment]
        training_entry.simulation_app.close()  # import模式不会进入正式模块的``__main__`` finally


def main() -> None:
    r"""命令行入口仍执行原有冻结梯度审计。"""

    run_checkpoint_audit(_arguments())


if __name__ == "__main__":
    main()
