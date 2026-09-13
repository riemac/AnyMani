r"""比较N040 backbone深度与推理精度对冻结PPO actor时延的独立影响。

Timed boundary与正式performance contract一致：RTX 5070 Ti、$B=4096$、GPU-resident observation、
20次warmup和50次CUDA Event。Provider指标只覆盖冻结N040的q-dependent cached-runtime路径；actor指标
覆盖provider、History30 stack-MLP、task adapter、一层policy Transformer与action/value heads。

三层候选从四层正式artifact删除第四个Transformer block；这只服务时延测量，不形成训练checkpoint。
最终可部署三层权重必须来自独立SSL训练。`*_full`条件改变整个actor的矩阵精度；`*_encoder`条件只改变
冻结N040 provider并把$Z$转回FP32，temporal/policy及rl_games边界继续使用FP32。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.methods.density_material_jacobian.artifact import (
    SE3RetainedEncoderArtifact,
    load_se3_retained_encoder_artifact,
)
from anymani.distill.models.input_adapters.evidence import StaticGeometryEvidence
from anymani.distill.models.input_adapters.se3_invariant_encoder import SE3InvariantGeometryEncoder
from anymani.distill.models.policy import CanonicalEvidenceBank
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_N040_HISTORY_OBS_DIM,
    HeterogeneousN040HistoryPpoBuilder,
)
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider
from anymani.distill.tests.performance.test_canonical_retained_geometry_encoder_latency import (
    _canonical_single_structure_evidence,
)

Precision = Literal["fp32", "tf32_full", "bf16_full", "tf32_encoder", "bf16_encoder"]

ARTIFACT_RELATIVE_PATH = Path(
    "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched/"
    "20260830T164445Z/retained_encoder.pt"
)
ARTIFACT_SHA256 = "cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea"


def _parse_args() -> argparse.Namespace:
    r"""解析批大小、计时重复数与可选JSON证据路径。"""

    parser = argparse.ArgumentParser(description="Benchmark N040 depth x inference precision.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _single_row_bank(evidence: StaticGeometryEvidence) -> CanonicalEvidenceBank:
    r"""为单结构性能fixture增加asset row轴，不改变其几何tensor。"""

    # 必需字段在类型合同中始终为tensor；三个valid masks允许fixture省略。
    row_evidence = StaticGeometryEvidence(
        anchors=evidence.anchors.unsqueeze(0),
        home_surface_points=evidence.home_surface_points.unsqueeze(0),
        home_surface_mask=evidence.home_surface_mask.unsqueeze(0),
        palm_normal=evidence.palm_normal.unsqueeze(0),
        space_screws=evidence.space_screws.unsqueeze(0),
        q_home=evidence.q_home.unsqueeze(0),
        entity_role=evidence.entity_role.unsqueeze(0),
        entity_joint_index=evidence.entity_joint_index.unsqueeze(0),
        joint_entity_index=evidence.joint_entity_index.unsqueeze(0),
        shortest_path=evidence.shortest_path.unsqueeze(0),
        parent_direction=evidence.parent_direction.unsqueeze(0),
        child_direction=evidence.child_direction.unsqueeze(0),
        entity_valid_mask=(
            evidence.entity_valid_mask.unsqueeze(0) if evidence.entity_valid_mask is not None else None
        ),
        joint_valid_mask=(
            evidence.joint_valid_mask.unsqueeze(0) if evidence.joint_valid_mask is not None else None
        ),
        anchor_valid_mask=(
            evidence.anchor_valid_mask.unsqueeze(0) if evidence.anchor_valid_mask is not None else None
        ),
    )
    return CanonicalEvidenceBank(
        evidence=row_evidence,
        asset_ids=("depth-precision-fixture",),
        physical_geometry_hashes=("depth-precision-physical",),
    )


def _artifact_for_depth(depth: int) -> SE3RetainedEncoderArtifact:
    r"""加载四层正式artifact，或截取前三层形成只用于计时的结构等价candidate。

    Args:
        depth (int): Graph-biased Transformer block数，只接受3或4。

    Returns:
        SE3RetainedEncoderArtifact: FP32 encoder及原artifact provenance。
    """

    artifact = load_se3_retained_encoder_artifact(
        resolve_anymani_root() / ARTIFACT_RELATIVE_PATH,
        expected_sha256=ARTIFACT_SHA256,
    )
    if depth == 4:
        return artifact
    if depth != 3:
        raise ValueError("N040 depth benchmark accepts only 3 or 4 blocks")

    # 前三层参数逐值继承正式artifact；第四层keys删除后必须严格匹配三层config。
    config = replace(
        artifact.encoder.se3_config,
        backbone=replace(artifact.encoder.se3_config.backbone, layers=3),
    )
    encoder = SE3InvariantGeometryEncoder(config)
    truncated_state = {
        name: value
        for name, value in artifact.encoder.state_dict().items()
        if not name.startswith("backbone.layers.3.")
    }
    encoder.load_state_dict(truncated_state, strict=True)
    return SE3RetainedEncoderArtifact(
        encoder=encoder,
        load_report=artifact.load_report,
        artifact_sha256=f"{artifact.artifact_sha256}:depth3-prefix",
        path=artifact.path,
        feature_spec=dict(artifact.feature_spec),
        input_contract=dict(artifact.input_contract),
        lineage={**dict(artifact.lineage), "latency_only_backbone_depth": 3},
    )


def _build_actor(depth: int, device: torch.device) -> tuple[torch.nn.Module, RetainedGeometryProvider]:
    r"""按正式History30性能配置构造指定N040深度的冻结actor。"""

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    artifact = _artifact_for_depth(depth)
    provider = RetainedGeometryProvider(
        artifact=artifact,
        evidence_bank=_single_row_bank(_canonical_single_structure_evidence(torch.device("cpu"))),
        dataset_digest="depth-precision-dataset",
        manifest_digest="depth-precision-manifest",
        canonical_schema_digest="depth-precision-canonical-schema",
        evidence_source_config={"fixture": "40-anchor-64-home"},
    )
    builder = HeterogeneousN040HistoryPpoBuilder()
    builder.load(
        {
            "retained_geometry_provider": provider,
            "parallel_geometry_temporal": True,
            "compile_policy_adapter": True,
            "temporal_encoder": "stack_mlp",
            "heterogeneous_policy": {
                "owner_feature_dim": 1,
                "joint_feature_dim": 6,
                "temporal_feature_dim": 32,
                "geometry_entity_width": 128,
                "hidden_width": 128,
                "layers": 1,
                "attention_heads": 4,
                "feedforward_width": 256,
                "dropout": 0.0,
                "initial_log_std": -0.5,
            },
        }
    )
    model = AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": (HETEROGENEOUS_N040_HISTORY_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    ).to(device)
    return model.a2c_network.eval(), provider


def _observation(batch_size: int, device: torch.device) -> torch.Tensor:
    r"""构造与正式性能合同相同shape和数值域的GPU-resident History30 observation。"""

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    obs = torch.zeros(batch_size, HETEROGENEOUS_N040_HISTORY_OBS_DIM, device=device)
    history = obs[:, : 30 * 16 * 4].reshape(batch_size, 30, 16, 4)
    history[:, :, :, 0].uniform_(-0.7 / torch.pi, 0.7 / torch.pi)  # 当前/历史$q/\pi$
    history[:, :, :, 1].uniform_(-0.7 / torch.pi, 0.7 / torch.pi)  # policy target$/\pi$
    history[:, :, :, 2].uniform_(-1.0, 1.0)  # 上一策略动作，无量纲
    history[:, :, :, 3].bernoulli_(0.4)  # 所属TIP接触bit
    limits = obs[:, 30 * 16 * 4 : 30 * 16 * 4 + 32].reshape(batch_size, 16, 2)
    limits[:, :, 0] = -1.0
    limits[:, :, 1] = 1.0
    obs[:, -17] = 0.0  # 所有环境共享single-structure evidence row
    obs[:, -16:] = 1.0  # fixture的16个JOINT均有效
    return obs


def _precision_context(precision: Precision):
    r"""返回全actor BF16 autocast上下文；encoder-only条件由provider wrapper实现。"""

    if precision == "bf16_full":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_encoder_precision(provider: RetainedGeometryProvider, precision: Precision) -> None:
    r"""为冻结provider安装局部precision wrapper，保持policy及输出$Z$为FP32。

    Provider始终持有FP32 master weights。BF16只在`resolve()`内部启用autocast；TF32只在该调用
    dispatch矩阵kernel时打开，函数返回前恢复全局backend状态。
    """

    original_resolve = provider.resolve
    if precision == "bf16_encoder":

        def bf16_resolve(asset_row: torch.Tensor, q_rad: torch.Tensor):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = original_resolve(asset_row, q_rad)
            return replace(batch, geometry_entities=batch.geometry_entities.float())

        provider.resolve = bf16_resolve  # type: ignore[method-assign]
    elif precision == "tf32_encoder":

        def tf32_resolve(asset_row: torch.Tensor, q_rad: torch.Tensor):
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            old_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                return original_resolve(asset_row, q_rad)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul
                torch.backends.cudnn.allow_tf32 = old_cudnn

        provider.resolve = tf32_resolve  # type: ignore[method-assign]


def _profile_cuda(callable_, *, warmups: int, events: int) -> dict[str, float]:
    r"""使用同步CUDA Events报告median/p95/max时延，单位ms。"""

    with torch.inference_mode():
        for _ in range(warmups):
            callable_()
        torch.cuda.synchronize()
        samples: list[float] = []
        stream = torch.cuda.current_stream()
        for _ in range(events):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            callable_()
            end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    r"""计算$\|x-x_0\|_2/\max(\|x_0\|_2,10^{-12})$。"""

    numerator = torch.linalg.vector_norm((actual - reference).double())
    denominator = torch.linalg.vector_norm(reference.double()).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _benchmark_case(
    depth: int,
    precision: Precision,
    obs: torch.Tensor,
    *,
    warmups: int,
    events: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    r"""测量一个depth×precision条件并返回FP32边界的Z与action mean。"""

    device = obs.device
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32_full"
    torch.backends.cudnn.allow_tf32 = precision == "tf32_full"
    try:
        network, provider = _build_actor(depth, device)
        _set_encoder_precision(provider, precision)
        history = obs[:, : 30 * 16 * 4].reshape(obs.shape[0], 30, 16, 4)
        rows = obs[:, -17].long()
        q_rad = history[:, -1, :, 0] * torch.pi

        def provider_forward() -> torch.Tensor:
            # 所有条件都以FP32 Z结束，保持policy融合边界一致。
            with _precision_context(precision):
                return provider.resolve(rows, q_rad).geometry_entities.float()

        def actor_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Full-precision条件也在action/value边界恢复FP32，保持rl_games接口不变。
            with _precision_context(precision):
                action_mean, action_log_std, value, _ = network({"obs": obs})
            return action_mean.float(), action_log_std.float(), value.float()

        torch.cuda.reset_peak_memory_stats(device)
        provider_latency = _profile_cuda(provider_forward, warmups=warmups, events=events)
        actor_latency = _profile_cuda(actor_forward, warmups=warmups, events=events)
        with torch.inference_mode():
            z = provider_forward().detach().cpu()
            action_mean = actor_forward()[0].detach().cpu()
        result = {
            "depth": depth,
            "precision": precision,
            "encoder_parameters": sum(parameter.numel() for parameter in provider.encoder.parameters()),
            "provider": provider_latency,
            "actor": actor_latency,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        }
        return result, z, action_mean
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def main() -> int:
    r"""运行深度、精度及精度作用域的时延与数值偏差矩阵。"""

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("N040 depth/precision benchmark requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        raise RuntimeError(f"benchmark is bound to RTX 5070 Ti, found {device_name}")
    if args.batch_size < 1 or args.warmups < 1 or args.events < 1:
        raise ValueError("batch_size, warmups and events must be positive")

    obs = _observation(args.batch_size, device)
    rows: list[dict[str, Any]] = []
    references: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for depth in (4, 3):
        for precision in ("fp32", "tf32_encoder", "bf16_encoder", "tf32_full", "bf16_full"):
            # 前一个compiled actor离开函数作用域后清理cache，避免六个条件累积占用显存。
            gc.collect()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            result, z, action_mean = _benchmark_case(
                depth,
                precision,
                obs,
                warmups=args.warmups,
                events=args.events,
            )
            if precision == "fp32":
                references[depth] = (z, action_mean)
                result["z_relative_l2_vs_fp32"] = 0.0
                result["action_relative_l2_vs_fp32"] = 0.0
            else:
                reference_z, reference_action = references[depth]
                result["z_relative_l2_vs_fp32"] = _relative_l2(z, reference_z)
                result["action_relative_l2_vs_fp32"] = _relative_l2(action_mean, reference_action)
            rows.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    report = {
        "schema_version": "1.0.0",
        "device": device_name,
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "events": args.events,
        "artifact": str(ARTIFACT_RELATIVE_PATH),
        "artifact_sha256": ARTIFACT_SHA256,
        "timed_boundary": "GPU-resident q/static-cache to FP32 Z; full History30 actor to FP32 outputs",
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
