r"""N031 retained encoder 的 SO(2)、origin-translation 与 full-SE(3) paired gauge audit。"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import torch
from anymani.distill.models.input_adapters.se3_gauge import rewrite_static_geometry_evidence_se3
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


def _axis_angle_rotation(axis: tuple[float, float, float], angle: float, *, device, dtype) -> torch.Tensor:
    r"""返回 det=+1 的 $R\in SO(3)$。"""

    vector = torch.tensor(axis, device=device, dtype=dtype)
    vector = vector / torch.linalg.vector_norm(vector)
    x, y, z = vector
    skew = torch.stack(
        (
            torch.stack((x * 0.0, -z, y)),
            torch.stack((z, y * 0.0, -x)),
            torch.stack((-y, x, z * 0.0)),
        )
    )
    identity = torch.eye(3, device=device, dtype=dtype)
    theta = torch.tensor(angle, device=device, dtype=dtype)
    return identity + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


class _Metric:
    r"""流式累计 paired tensor 的绝对/相对误差与 cosine。"""

    def __init__(self) -> None:
        self.error_square = 0.0
        self.reference_square = 0.0
        self.max_abs = 0.0
        self.cosine_sum = 0.0
        self.count = 0

    def update(self, reference: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            reference = reference[mask]
            actual = actual[mask]
        reference = reference.detach().double().reshape(reference.shape[0], -1)
        actual = actual.detach().double().reshape(actual.shape[0], -1)
        error = actual - reference
        self.error_square += float(error.square().sum())
        self.reference_square += float(reference.square().sum())
        self.max_abs = max(self.max_abs, float(error.abs().max()))
        cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=-1, eps=1.0e-30)
        self.cosine_sum += float(cosine.sum())
        self.count += len(reference)

    def report(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "max_abs": self.max_abs,
            "relative_l2": (self.error_square / max(self.reference_square, 1.0e-30)) ** 0.5,
            "rms_error": (self.error_square / max(1, self.count)) ** 0.5,
            "mean_cosine": self.cosine_sum / max(1, self.count),
        }


def main() -> None:
    r"""在 64 held-out morphologies × 2 q 上审计指定 retained feature hierarchy。"""

    parser = ArgumentParser()
    parser.add_argument("--encoder", choices=("legacy", "se3"), default="legacy")
    parser.add_argument("--config", default="geometry_ssl_density_material_jacobian_v0_8_0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "logs/ssl/geometry_ssl_density_material_jacobian_v0_8_0_extended384/"
            "20260830T073321Z/checkpoints/last.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-008-frame-gauge"),
    )
    args = parser.parse_args()
    config = compose_evaluation_cfg(config_ref=args.config)
    device = torch.device("cuda:0")
    catalog = build_runtime(config.data).resolve_evaluation()
    method = build_runtime(config.method)
    method.configure_source_artifacts(
        root=config.evaluation.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="evaluation",
    )
    method.prepare(catalog, role="evaluation", device=device, dtype=torch.float32)
    method.configure_execution(replace(config.evaluation.execution, compile_enabled=False, model_autocast_dtype="float32"))
    model = method.initialize_model(device=device, dtype=torch.float32)
    checkpoint = args.checkpoint
    payload = load_pretrain_checkpoint(checkpoint, map_location="cpu")
    method.load_training_state_dict(payload["method_state"])
    method.eval_mode()
    encoder = model.encoder
    if args.encoder == "se3" and not isinstance(model.encoder, SE3InvariantGeometryEncoder):
        legacy = config.method.model.encoder
        encoder = SE3InvariantGeometryEncoder(
            SE3InvariantGeometryEncoderCfg(
                frontend=SE3InvariantAnchorFrontendCfg(
                    relation_width=legacy.frontend.relation_width,
                    home_width=legacy.frontend.home_width,
                    screw_width=legacy.frontend.screw_width,
                    role_width=legacy.frontend.role_width,
                    length_scale_m=legacy.frontend.length_scale_m,
                ),
                backbone=legacy.backbone,
            )
        ).to(device=device, dtype=torch.float32)
        encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        encoder.eval()
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=config.run.seed + config.evaluation.evaluation_seed_offset,
        device=device,
        dtype=torch.float32,
        max_resident_assets=8,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        64,
        q_per_asset=2,
        assets_per_minibatch=8,
        q_per_asset_per_minibatch=2,
        max_resident_assets=8,
    )
    identity = torch.eye(3, device=device, dtype=torch.float32)
    transforms = {
        "so2": (
            _axis_angle_rotation((0.0, 0.0, 1.0), 1.137, device=device, dtype=torch.float32),
            torch.zeros(3, device=device),
        ),
        "origin_translation": (
            identity,
            torch.tensor((0.041, -0.027, 0.019), device=device),
        ),
        "full_se3": (
            _axis_angle_rotation((0.31, -0.72, 0.62), 0.83, device=device, dtype=torch.float32),
            torch.tensor((0.041, -0.027, 0.019), device=device),
        ),
    }
    metrics = {
        name: {level: _Metric() for level in ("home_features", "screw_features", "z")}
        for name in transforms
    }
    try:
        step = 0
        with torch.no_grad():
            while not schedule.complete:
                batch = session.realize(schedule.next(), schedule=schedule, step=step)
                reference_home = encoder._home_features(batch.evidence)
                reference_screw = encoder._screw_features(batch.evidence)
                reference_z = encoder(
                    batch.q,
                    batch.evidence,
                    evidence_row_index=batch.evidence_row_index,
                ).entities
                entity_valid = batch.evidence.entity_valid_mask[batch.evidence_row_index]
                joint_valid = batch.evidence.joint_valid_mask
                if joint_valid is None:
                    joint_valid = torch.ones(batch.evidence.space_screws.shape[:2], device=device, dtype=torch.bool)
                for name, (rotation, translation) in transforms.items():
                    rewritten = rewrite_static_geometry_evidence_se3(
                        batch.evidence,
                        rotation=rotation,
                        translation=translation,
                    )
                    actual_home = encoder._home_features(rewritten)
                    actual_screw = encoder._screw_features(rewritten)
                    actual_z = encoder(
                        batch.q,
                        rewritten,
                        evidence_row_index=batch.evidence_row_index,
                    ).entities
                    metrics[name]["home_features"].update(reference_home, actual_home, batch.evidence.entity_valid_mask)
                    metrics[name]["screw_features"].update(reference_screw, actual_screw, joint_valid)
                    metrics[name]["z"].update(reference_z, actual_z, entity_valid)
                step += 1
    finally:
        session.close()
        method.close()
    report = {
        "case": "AR-MPJ-008",
        "checkpoint": str(checkpoint),
        "encoder": args.encoder,
        "population": {"assets": 64, "q_per_asset": 2, "rows": 128},
        "coordinate_rewrite": "p'=Rp+t; omega'=Romega; v'=Rv-omega'xt",
        "metrics": {
            transform: {level: metric.report() for level, metric in levels.items()}
            for transform, levels in metrics.items()
        },
    }
    output = args.output / f"report_{args.encoder}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
