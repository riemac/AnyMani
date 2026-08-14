r"""21-asset same-topology canonical residual Geometry SSL pilot。

这是一个完整、自包含的声明式实验组合：资产 family、physical-group split、query/target、
canonical residual model、paired objective、window/q/epoch budget、calibration 与 evidence
规则在一个模块中可核对。runtime 只消费该 resolved config，不在运行时选择实验语义。
"""

from __future__ import annotations

from pathlib import Path

from anymani.distill.models.geometry_ssl import GeometrySSLModelConfig
from anymani.distill.models.input_adapters.geometry import GeometryEncoderConfig
from anymani.distill.objectives.representations.field_reconstruction import GeometrySSLWeights
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg
from anymani.distill.ssl.config import (
    GeometrySSLAssetCfg,
    GeometrySSLExperimentCfg,
    GeometrySSLTrainLoopCfg,
)

FORMAL_MOTHER = Path(
    "/home/hac/isaac/AnyMani/source/anymani/anymani/assets/generated/2026-08-12_18-16-48/"
    "single_palm_leap/right_t4_i4_m4_r4"
)
FORMAL_FAMILY = FORMAL_MOTHER / "2026-08-13_02-05-29"
MOTHER_ASSET_ID = "f5d8c069"
FORMAL_VARIANT_IDS = (
    "05c1db5c",
    "15c82d3f",
    "25496b19",
    "34266899",
    "4540d627",
    "49e9b654",
    "56c628f9",
    "624b70e9",
    "71cc91d9",
    "8e433022",
    "99272bfa",
    "ab3d4b97",
    "b22a3f60",
    "b336924f",
    "c2508da5",
    "c4f25d2e",
    "d84108ac",
    "ea1c252d",
    "ed6de294",
    "f25ede87",
)


def canonical_residual_family_experiment() -> GeometrySSLExperimentCfg:
    r"""返回正式 mother+20 variants 的 canonical residual pilot 配置。

    该配置只覆盖 right LEAP、同 topology、16 DOF family；不包含 official、PPO、cross-DOF
    或 cross-topology 语义。bundle 清单显式冻结，不在 import/runtime 阶段扫描目录猜资产。
    """

    return GeometrySSLExperimentCfg(
        schema_version="1.1.0",
        assets=GeometrySSLAssetCfg(
            family_paths=(str(FORMAL_MOTHER), *(str(FORMAL_FAMILY / asset_id) for asset_id in FORMAL_VARIANT_IDS)),
            mother_asset_id=MOTHER_ASSET_ID,
            validation_asset_count=4,
            split_seed=20260813,
        ),
        query=SpatialQuerySamplerCfg(query_count=64),
        target=GeometryFieldTargetCfg(
            bandwidths_m=(0.004, 0.012, 0.032, 0.064),
            edges_per_owner=2,
        ),
        model=GeometrySSLModelConfig(
            encoder=GeometryEncoderConfig(
                relation_width=64,
                home_width=64,
                screw_width=64,
                hidden_width=128,
                zero_order_width=128,
                first_order_width=64,
                transformer_layers=2,
                attention_heads=4,
                feedforward_width=256,
                dropout=0.0,
            ),
            decoder_hidden_width=128,
            decoder_residual_blocks=3,
            bandwidth_count=4,
        ),
        objective=GeometrySSLWeights(
            density=1.0,
            kappa=1.0,
            derived_field=1.0,
            sobolev=1.0,
            chain=1.0,
            paired=1.0,
        ),
        train=GeometrySSLTrainLoopCfg(
            steps=30_000,
            batch_size=4,
            assets_per_microbatch=2,
            q_per_asset_per_microbatch=2,
            max_resident_assets=20,
            q_per_asset_per_epoch=256,
            epochs=20,
            validation_q_per_asset=64,
            calibration_batches=8,
            gradient_accumulation_steps=4,
            seed=20260813,
            deterministic_algorithms=True,
            device="cuda:0",
            dtype="float32",
            experiment_name="canonical_residual_right_leap_family_20260813",
            resume_checkpoint="",
        ),
    )


__all__ = [
    "FORMAL_FAMILY",
    "FORMAL_MOTHER",
    "FORMAL_VARIANT_IDS",
    "MOTHER_ASSET_ID",
    "canonical_residual_family_experiment",
]
