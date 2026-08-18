r"""manifest-driven cross-mother canonical residual Geometry SSL 声明式实验。

本模块只声明资产 family、物理表征、模型、目标、协议与 run identity。Hydra defaults 选择
``trainer/single_gpu_16gb``；导入本模块不会扫描资产、初始化 CUDA 或创建输出目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.objectives.representations.field_reconstruction import GeometryFieldObjectiveCfg
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import (
    GaussianProximityFieldCfg,
    GeometryFieldTargetCfg,
)
from anymani.distill.ssl.config import (
    GeometryCalibrationCfg,
    GeometryCoverageCfg,
    GeometryReproducibilityCfg,
    GeometrySSLExperimentCfg,
    GeometrySSLProtocolCfg,
    GeometrySSLRunCfg,
    GeometrySSLTrainerCfg,
    GeometryValidationCfg,
)

FORMAL_ASSET_DATASET_MANIFEST = "source/anymani/anymani/assets/datasets/canonical_cross_mother_v1.yaml"
"""正式 cross-mother pilot 的唯一资产选择文件。"""


@dataclass(frozen=True)
class CanonicalSourceCfg(GeometrySourceCfg):
    r"""固定 owner boundary evidence 与 palm-seed anchor realization。"""

    home_points_per_owner: int = 64
    home_surface_oversample_factor: int = 8
    anchors_per_finger: int = 10
    anchor_radius_m: float = 0.05
    anchor_radial_decay_scale_m: float = 0.025
    anchor_surface_fraction: float = 0.5
    static_sampling_seed: int = 0


@dataclass(frozen=True)
class CanonicalFieldCfg(GaussianProximityFieldCfg):
    r"""固定 4/16/64 mm train centers、±10% jitter 与五尺度 validation grid。"""

    bandwidth_centers_m: tuple[float, ...] = (0.004, 0.016, 0.064)
    bandwidth_jitter_relative: float = 0.10
    validation_bandwidths_m: tuple[float, ...] = (0.004, 0.008, 0.016, 0.032, 0.064)


@dataclass(frozen=True)
class CanonicalQueryCfg(SpatialQuerySamplerCfg):
    r"""每 owner 64 点的 workspace/shell/adjacent = 32/16/16 测度。"""

    query_count: int = 64
    workspace_fraction: float = 0.50
    owner_shell_fraction: float = 0.25
    adjacent_fraction: float = 0.25
    workspace_radius_m: float = 0.05
    shell_offset_min_m: float = 0.0005
    shell_offset_max_m: float = 0.004
    adjacent_candidate_count: int = 4


@dataclass(frozen=True)
class CanonicalTargetCfg(GeometryFieldTargetCfg):
    r"""每 owner 两条 sampled κ edges 与保守局部非光滑 mask。"""

    edges_per_owner: int = 2
    distance_epsilon_m: float = 1.0e-6
    feature_margin_min_m: float = 1.0e-5


@dataclass(frozen=True)
class CanonicalLayoutCfg(GeometryPaddingCfg):
    r"""最多 20 JOINT、5 TIP、26 owners 的跨结构稠密容器。"""

    max_joint_count: int = 20
    max_tip_count: int = 5
    max_graph_distance: int = 8


@dataclass(frozen=True)
class CanonicalRepresentationCfg(GeometryRepresentationCfg):
    r"""组合 physical source、Gaussian field、query measure、target 与 layout。"""

    source: CanonicalSourceCfg = dataclass_field(default_factory=CanonicalSourceCfg)
    field: CanonicalFieldCfg = dataclass_field(default_factory=CanonicalFieldCfg)
    query: CanonicalQueryCfg = dataclass_field(default_factory=CanonicalQueryCfg)
    target: CanonicalTargetCfg = dataclass_field(default_factory=CanonicalTargetCfg)
    layout: CanonicalLayoutCfg = dataclass_field(default_factory=CanonicalLayoutCfg)


@dataclass(frozen=True)
class CanonicalFrontendCfg(SO2AnchorFrontendCfg):
    r"""共享点/旋量—anchor 前端与 owner 内集合聚合容量。"""

    relation_width: int = 64
    home_width: int = 64
    screw_width: int = 64
    role_width: int = 8
    length_scale_m: float = 0.1


@dataclass(frozen=True)
class CanonicalBackboneCfg(GraphBiasedTransformerCfg):
    r"""2 层 encoder-only 全连接 Transformer 与三种每头加性 graph bias。"""

    hidden_width: int = 128
    layers: int = 2
    attention_heads: int = 4
    feedforward_width: int = 256
    dropout: float = 0.0
    max_graph_distance: int = 8


@dataclass(frozen=True)
class CanonicalLatentHeadsCfg(GeometryLatentHeadsCfg):
    r"""每 owner $D_0=128$ 与逐 JOINT residual-screw $D_1=64$。"""

    zero_order_width: int = 128
    first_order_width: int = 64
    first_order_source: str = "residual_screw"


@dataclass(frozen=True)
class CanonicalDensityDecoderCfg(ScalarSigmaFiLMDensityDecoderCfg):
    r"""128 hidden、3 FiLM blocks、16 mm sigma reference 的 SSL-only scalar reader。"""

    hidden_width: int = 128
    residual_blocks: int = 3
    sigma_reference_m: float = 0.016


@dataclass(frozen=True)
class CanonicalSensitivityDecoderCfg(DistanceSensitivityDecoderCfg):
    r"""128 hidden coefficient 与无偏置 $1/\sqrt{D_1}$ 奇读取。"""

    coefficient_hidden_width: int = 128
    readout_bias: bool = False
    carrier_scale: str = "inverse_sqrt"


@dataclass(frozen=True)
class CanonicalDecoderCfg(GeometrySSLDecoderCfg):
    r"""聚合两个训练期 readers；两者均不进入 retained transfer。"""

    density: CanonicalDensityDecoderCfg = dataclass_field(default_factory=CanonicalDensityDecoderCfg)
    sensitivity: CanonicalSensitivityDecoderCfg = dataclass_field(default_factory=CanonicalSensitivityDecoderCfg)


@dataclass(frozen=True)
class CanonicalEncoderCfg(GeometryEncoderCfg):
    r"""组合 canonical frontend、graph-biased backbone 与 latent heads。"""

    frontend: CanonicalFrontendCfg = dataclass_field(default_factory=CanonicalFrontendCfg)
    backbone: CanonicalBackboneCfg = dataclass_field(default_factory=CanonicalBackboneCfg)
    heads: CanonicalLatentHeadsCfg = dataclass_field(default_factory=CanonicalLatentHeadsCfg)


@dataclass(frozen=True)
class CanonicalModelCfg(GeometrySSLModelCfg):
    r"""retained encoder 与 disposable SSL decoders 的完整模型配置。"""

    encoder: CanonicalEncoderCfg = dataclass_field(default_factory=CanonicalEncoderCfg)
    ssl_decoders: CanonicalDecoderCfg = dataclass_field(default_factory=CanonicalDecoderCfg)


@dataclass(frozen=True)
class CanonicalObjectiveCfg(GeometryFieldObjectiveCfg):
    r"""声明六项全部开启；runtime calibration 另存 evidence，不覆盖本对象。"""

    density: float = 1.0
    kappa: float = 1.0
    derived_field: float = 1.0
    sobolev: float = 1.0
    chain: float = 1.0
    paired: float = 1.0


@dataclass(frozen=True)
class CanonicalCoverageCfg(GeometryCoverageCfg):
    epochs: int = 20
    q_per_asset_per_epoch: int = 256
    q_per_asset_per_realization: int = 2


@dataclass(frozen=True)
class CanonicalCalibrationCfg(GeometryCalibrationCfg):
    batches: int = 8
    min_weight: float = 1.0e-2
    max_weight: float = 1.0e3


@dataclass(frozen=True)
class CanonicalValidationCfg(GeometryValidationCfg):
    q_per_asset: int = 64
    every_optimizer_updates: int = 250
    selection_metrics: tuple[str, ...] = ("density", "kappa", "derived_field")
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    )
    bootstrap_replicates: int = 2_000


@dataclass(frozen=True)
class CanonicalReproducibilityCfg(GeometryReproducibilityCfg):
    seed: int = 20260813
    deterministic_algorithms: bool = True
    seed_domains: tuple[str, ...] = ("model", "sobol_q", "query", "sigma", "edge", "validation", "bootstrap")


@dataclass(frozen=True)
class CanonicalProtocolCfg(GeometrySSLProtocolCfg):
    coverage: CanonicalCoverageCfg = dataclass_field(default_factory=CanonicalCoverageCfg)
    calibration: CanonicalCalibrationCfg = dataclass_field(default_factory=CanonicalCalibrationCfg)
    validation: CanonicalValidationCfg = dataclass_field(default_factory=CanonicalValidationCfg)
    reproducibility: CanonicalReproducibilityCfg = dataclass_field(default_factory=CanonicalReproducibilityCfg)
    run_safety_step_limit: int = 30_000


@dataclass(frozen=True)
class CanonicalRunCfg(GeometrySSLRunCfg):
    output_dir: str = "logs/ssl"
    experiment_name: str = "canonical_residual_right_leap_family_20260813"
    resume_checkpoint: str = ""


@dataclass(frozen=True)
class CanonicalResidualFamilyCfg(GeometrySSLExperimentCfg):
    r"""right LEAP 同 topology/16-DOF family 的最高声明式实验身份。"""

    defaults = ({"trainer": "single_gpu_16gb"}, "_self_")
    schema_version: str = "2.0.0"
    asset_dataset_manifest: str = FORMAL_ASSET_DATASET_MANIFEST
    representation: CanonicalRepresentationCfg = dataclass_field(default_factory=CanonicalRepresentationCfg)
    model: CanonicalModelCfg = dataclass_field(default_factory=CanonicalModelCfg)
    objective: CanonicalObjectiveCfg = dataclass_field(default_factory=CanonicalObjectiveCfg)
    protocol: CanonicalProtocolCfg = dataclass_field(default_factory=CanonicalProtocolCfg)
    trainer: GeometrySSLTrainerCfg = dataclass_field(default_factory=GeometrySSLTrainerCfg)
    run: CanonicalRunCfg = dataclass_field(default_factory=CanonicalRunCfg)


__all__ = [
    "CanonicalResidualFamilyCfg",
    "FORMAL_ASSET_DATASET_MANIFEST",
]
