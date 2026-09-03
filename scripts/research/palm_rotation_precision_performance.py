r"""MVP80四层N040 encoder-only BF16数值与provider→actor性能门。

数值集包含每个资产的schema-3 rank-0 $q_0$和一个确定性小扰动$q_0+\delta q$。同一FP32 actor
分别读取FP32 N040与BF16 N040恢复的FP32 $Z^e$，检查：finite/mask完全一致、per-sample $Z$
relative-L2不超过2%、actor mean RMS差不超过初始$\sigma=e^{-0.5}$的10%。

性能集固定$B=2560$，CUDA Event覆盖``current q -> BF16 N040 -> FP32 residual actor mean``，
20次warmup、50次measurement；RTX 5070 Ti门为p95<50 ms且peak allocated memory<总显存85%。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml"

parser = argparse.ArgumentParser(description="Validate MVP80 N040 BF16 precision and provider-to-actor latency.")
parser.add_argument("--batch_size", type=int, default=2560, help="Formal performance batch, fixed to 2560 by default.")
parser.add_argument("--warmups", type=int, default=20, help="CUDA-event warmup forwards.")
parser.add_argument("--repeats", type=int, default=50, help="CUDA-event measured forwards.")
parser.add_argument("--output", type=Path, default=None, help="Optional JSON artifact path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ASSET_BINDING在import时冻结support axis；精度/performance必须与正式80-row manifest完全相同。
manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
rows = tuple(int(row) for row in manifest["selected_rows"])
if len(rows) != 80 or len(set(rows)) != 80:
    raise ValueError("precision gate requires exactly 80 unique MVP assets")
if args.batch_size % 80 != 0:
    raise ValueError("performance batch must contain an equal integer number of replicas for all 80 assets")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in rows)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = "80"  # 不创建scene，仅使config-level reset binding轴完整

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch  # noqa: E402
from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.structured_geometry import (  # noqa: E402
    build_structured_retained_geometry_provider,
)
from anymani.pregrasp.good_catalog import GoodPregraspCatalog  # noqa: E402
from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (  # noqa: E402
    GOOD_PREGRASP_RESET_CFG,
)
from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING  # noqa: E402


def _rank0_q_and_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""解析80个exact catalog entries，返回rank-0 q与JOINT/TIP/owner masks。"""

    catalog_root = Path(GOOD_PREGRASP_RESET_CFG.catalog_root)
    if not catalog_root.is_absolute():
        catalog_root = ROOT / catalog_root
    catalog = GoodPregraspCatalog(catalog_root)
    q_rows: list[tuple[float, ...]] = []
    joint_masks: list[tuple[bool, ...]] = []
    tip_masks: list[tuple[bool, ...]] = []
    for binding, artifact in zip(GOOD_PREGRASP_RESET_CFG.bindings, ASSET_BINDING.canonical_artifacts, strict=True):
        entry = catalog.resolve(binding.resolve_key())  # exact hand/object/scale/physics/generation key
        rank0 = entry.members[0]
        if rank0.rank != 0:
            raise RuntimeError("good-pregrasp entry does not begin with rank-0")
        q_rows.append(rank0.candidate.q_state_rad)  # canonical `[16]` rad，ghost=0
        joint_masks.append(rank0.candidate.active_joint_mask)
        tip_masks.append(tuple(artifact.routing.active_tip_mask))
    q = torch.tensor(q_rows, dtype=torch.float32, device=device)  # `[80,16]` rad
    joint_valid = torch.tensor(joint_masks, dtype=torch.bool, device=device)  # `[80,16]`
    tip_valid = torch.tensor(tip_masks, dtype=torch.bool, device=device)  # `[80,4]`
    owner_valid = torch.cat(
        (torch.ones(80, 1, dtype=torch.bool, device=device), joint_valid, tip_valid), dim=1
    )  # `[80,21]` PALM+JOINT+TIP
    return q, joint_valid, tip_valid, owner_valid


def _actor_observation(
    q_rad: torch.Tensor,
    joint_valid: torch.Tensor,
    tip_valid: torch.Tensor,
    owner_valid: torch.Tensor,
) -> PalmRotationActorObservation:
    r"""由$q$构造无contact、$u=q$、constant-History30的合法actor packet。"""

    q_normalized = q_rad / torch.pi  # joint coordinate无量纲化
    current = torch.zeros(q_rad.shape[0], 16, 5, device=q_rad.device)  # `[q/pi,u/pi,a,c_j,c_tip]`
    current[..., 0] = q_normalized
    current[..., 1] = q_normalized  # rank-0 preload满足$u_0=q_0$
    current *= joint_valid.unsqueeze(-1)  # ghost state严格为0
    history = current.unsqueeze(1).expand(-1, 30, -1, -1).clone()  # oldest-to-latest constant reset history
    limits = torch.stack((-torch.ones_like(q_rad), torch.ones_like(q_rad)), dim=-1)
    limits *= joint_valid.unsqueeze(-1)  # performance/precision共用宽松normalized limits
    return PalmRotationActorObservation(
        jnt_current=current,
        jnt_history=history,
        jnt_limits=limits,
        owner_contact=torch.zeros(q_rad.shape[0], 21, 1, device=q_rad.device),
        jnt_valid=joint_valid,
        tip_valid=tip_valid,
        owner_valid=owner_valid,
    )


def _expand_rows(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    r"""按80-asset round-robin顺序复制static或rank-0 tensor到正式batch。"""

    replicas = batch_size // 80
    return tensor.unsqueeze(1).expand(-1, replicas, *tensor.shape[1:]).reshape(batch_size, *tensor.shape[1:])


def main() -> dict[str, object]:
    r"""执行BF16/FP32数值门与$B=2560$ CUDA-event性能门。"""

    if not torch.cuda.is_available():
        raise RuntimeError("MVP80 BF16/performance gate requires CUDA")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 reference及actor不允许TF32近似
    torch.backends.cudnn.allow_tf32 = False
    q_rank0, joint_valid, tip_valid, owner_valid = _rank0_q_and_masks(device)

    # 每个asset加入一个0.02 rad以内的确定性active-joint perturbation，覆盖q-dependent encoder路径。
    phase = torch.arange(16, device=device).float().unsqueeze(0) + torch.arange(80, device=device).float().unsqueeze(1)
    perturbation = 0.02 * torch.sin(0.7 * phase) * joint_valid.float()  # rad
    q_precision = torch.cat((q_rank0, q_rank0 + perturbation), dim=0)  # `[160,16]`
    joint_precision = torch.cat((joint_valid, joint_valid), dim=0)
    tip_precision = torch.cat((tip_valid, tip_valid), dim=0)
    owner_precision = torch.cat((owner_valid, owner_valid), dim=0)
    prototype_precision = torch.arange(80, device=device).repeat(2)  # rank-0与perturbed同asset routing
    actor_observation = _actor_observation(q_precision, joint_precision, tip_precision, owner_precision)

    # 两个provider从同一FP32 retained artifact独立构造，避免BF16 static cache污染FP32 reference。
    fp32_provider = build_structured_retained_geometry_provider(ASSET_BINDING, device=device)
    bf16_provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=device)
    with torch.no_grad():
        fp32_batch = fp32_provider.resolve(prototype_precision, q_precision)
        bf16_geometry = bf16_provider.resolve(prototype_precision, actor_observation)
    fp32_geometry = PalmRotationGeometry(
        tokens=fp32_batch.geometry_entities.float(),
        owner_valid=fp32_batch.owner_valid_mask,
        shortest_path=fp32_batch.shortest_path,
        parent_direction=fp32_batch.parent_direction,
        child_direction=fp32_batch.child_direction,
    )

    # Relative L2按environment独立计算，防止少数手型的极大误差被全batch norm掩盖。
    token_error = torch.linalg.vector_norm((bf16_geometry.tokens - fp32_geometry.tokens).reshape(160, -1), dim=-1)
    token_reference = torch.linalg.vector_norm(fp32_geometry.tokens.reshape(160, -1), dim=-1).clamp_min(1.0e-12)
    relative_l2 = token_error / token_reference
    actor = PalmRotationActorCritic(residual_enabled=True).actor.to(device).eval()
    with torch.no_grad():
        mean_fp32 = actor(actor_observation, fp32_geometry).mean
        mean_bf16 = actor(actor_observation, bf16_geometry).mean
    mean_rms_error = torch.sqrt(torch.mean((mean_bf16 - mean_fp32).square()))
    initial_sigma = math.exp(-0.5)  # MVP shared Gaussian初始标准差
    masks_equal = bool(torch.equal(fp32_batch.owner_valid_mask, bf16_geometry.owner_valid))
    finite = bool(
        torch.isfinite(fp32_geometry.tokens).all().item()
        and torch.isfinite(bf16_geometry.tokens).all().item()
        and torch.isfinite(mean_fp32).all().item()
        and torch.isfinite(mean_bf16).all().item()
    )

    # 正式performance batch按asset-major 32 replicas展开，prototype index与q/masks同步。
    q_performance = _expand_rows(q_rank0, args.batch_size)
    joint_performance = _expand_rows(joint_valid, args.batch_size)
    tip_performance = _expand_rows(tip_valid, args.batch_size)
    owner_performance = _expand_rows(owner_valid, args.batch_size)
    prototype_performance = torch.arange(80, device=device).repeat_interleave(args.batch_size // 80)
    performance_observation = _actor_observation(
        q_performance,
        joint_performance,
        tip_performance,
        owner_performance,
    )

    def forward() -> torch.Tensor:
        r"""计时边界：physical q -> BF16 N040 -> FP32 actor mean。"""

        geometry = bf16_provider.resolve(prototype_performance, performance_observation)
        return actor(performance_observation, geometry).mean

    with torch.no_grad():
        for _ in range(int(args.warmups)):
            forward()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        for _ in range(int(args.repeats)):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()  # pyright: ignore[reportCallIssue]  # 当前PyTorch stub误要求显式stream
            output = forward()
            stop.record()  # pyright: ignore[reportCallIssue]  # 使用当前CUDA default stream
            stop.synchronize()
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("provider-to-actor performance forward produced non-finite output")
            timings.append(float(start.elapsed_time(stop)))  # milliseconds
    timing_tensor = torch.tensor(timings)
    p50_ms = float(torch.quantile(timing_tensor, 0.50).item())
    p95_ms = float(torch.quantile(timing_tensor, 0.95).item())
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    memory_fraction = peak_memory / total_memory

    result: dict[str, object] = {
        "artifact_type": "anymani.heterogeneous_palm_rotation_precision_performance",
        "schema_version": "1.0.0",
        "device": str(device),
        "asset_count": 80,
        "precision_samples": 160,
        "precision": {
            "finite": finite,
            "masks_equal": masks_equal,
            "z_relative_l2_mean": float(relative_l2.mean().item()),
            "z_relative_l2_max": float(relative_l2.max().item()),
            "z_relative_l2_limit": 0.02,
            "actor_mean_rms_error": float(mean_rms_error.item()),
            "actor_mean_rms_limit": 0.1 * initial_sigma,
            "initial_sigma": initial_sigma,
        },
        "performance": {
            "batch_size": int(args.batch_size),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "p95_limit_ms": 50.0,
            "peak_memory_bytes": peak_memory,
            "total_memory_bytes": total_memory,
            "peak_memory_fraction": memory_fraction,
            "memory_fraction_limit": 0.85,
        },
    }
    passed = finite and masks_equal and float(relative_l2.max().item()) <= 0.02
    passed = passed and float(mean_rms_error.item()) <= 0.1 * initial_sigma
    passed = passed and p95_ms < 50.0 and memory_fraction < 0.85
    result["passed"] = passed
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise RuntimeError("MVP80 BF16 precision/performance gate failed")
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
