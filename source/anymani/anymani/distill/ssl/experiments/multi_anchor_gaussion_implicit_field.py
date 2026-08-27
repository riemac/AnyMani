r"""多锚点 Gaussian 隐式场实验的完整 Python 装配。

所有科研参数、训练参数和运行阶段都在本模块逐行可读。Hydra 只注册完整 `EXPERIMENT`
并接受命令行覆盖，不再用分片 YAML 拼装 method/model/objective。

实验文件名保留历史拼写 `gaussion`，避免破坏已有入口。
"""

from anymani.distill.methods.multi_anchor_gaussian_implicit_field import (
    DensityObjectiveCfg,
    EntityPermutationCfg,
    FairGradCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianMethodCfg,
    MultiAnchorGaussianObjectivesCfg,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import (
    GaussianProximityFieldCfg,
    GeometryFieldTargetCfg,
)
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg
from anymani.distill.ssl.post_training import (
    EmbodimentEvaluationCfg,
    EmbodimentValidationCfg,
    EvaluationCfg,
    EvaluationRunCfg,
    ValidationCfg,
    ValidationRunCfg,
)
from anymani.distill.ssl.runtime.pretrainer import EmbodimentPretrainTrainerCfg
from anymani.distill.ssl.runtime.run import PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

# Dataset manifest 已完整冻结 train/validation/evaluation；实验层不再重复声明 partition。

###
#  资产数据集配置层
###
DATA_CFG = HandAssetCatalogCfg(
    manifest="source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml",
    expected_sha256="f1398417888e7c237cbb2583dcf8e9cd10bef7fee792b307c67dfa74fb6e0698",
)

###
#  方法配置层-多锚点隐式高斯密度场
###
## 状态度量配置
STATE_MEASURE_CFG = JointConfigurationMeasureCfg()

## 表征配置
REPRESENTATION_CFG = GeometryRepresentationCfg(
    source=GeometrySourceCfg(
        home_points_per_owner=64,
        home_surface_oversample_factor=8,
        static_sampling_seed=0,
        anchors=AnchorBankCfg(
            bank_size=8,
            anchors_per_finger=10,
            radius_m=0.05,
            radial_decay_scale_m=0.025,
            surface_fraction=0.5,
        ),
    ),
    field=GaussianProximityFieldCfg(
        bandwidth_centers_m=(0.004, 0.016, 0.064),
        bandwidth_jitter_relative=0.10,
        validation_bandwidths_m=(0.004, 0.016, 0.064),
    ),
    query=SpatialQuerySamplerCfg(
        query_count=64,
        workspace_fraction=0.50,
        owner_shell_fraction=0.25,
        adjacent_fraction=0.25,
        workspace_radius_m=0.05,
        shell_offset_min_m=0.0005,
        shell_offset_max_m=0.004,
        adjacent_candidate_count=4,
    ),
    target=GeometryFieldTargetCfg(
        train_active_per_joint=2,
        train_zero_per_joint=1,
        validation_active_per_joint=4,
        validation_zero_per_joint=4,
        distance_epsilon_m=1.0e-6,
        feature_margin_min_m=1.0e-5,
    ),
)

## 网络模型配置
MODEL_CFG = GeometrySSLModelCfg(
    encoder=GeometryEncoderCfg(
        frontend=SO2AnchorFrontendCfg(
            relation_width=64,
            home_width=64,
            screw_width=64,
            role_width=8,
            length_scale_m=0.1,
        ),
        backbone=GraphBiasedTransformerCfg(
            hidden_width=128,
            layers=4,
            attention_heads=4,
            feedforward_width=256,
            dropout=0.0,
            max_graph_distance=8,
        ),
    ),
    ssl_decoders=GeometrySSLDecoderCfg(
        density=ScalarSigmaFiLMDensityDecoderCfg(
            hidden_width=128,
            residual_blocks=2,
            sigma_reference_m=0.016,
        ),
        sensitivity=DistanceSensitivityDecoderCfg(
            hidden_width=128,
            residual_blocks=2,
            readout_rank=64,
            physical_scale_m=0.1,
        ),
    ),
)

## 目标配置
OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(
    density=DensityObjectiveCfg(),
    kappa=KappaObjectiveCfg(),
)

## 数据增强配置
JOINT_SIGN_REWRITE_CFG = JointSignRewriteCfg(probability=0.20, seed_offset=17)
ENTITY_PERMUTATION_CFG = EntityPermutationCfg(enabled=True, seed_offset=31_337)
FAIRGRAD_CFG = FairGradCfg(
    algorithm="fairgrad_alpha_1_two_task_analytic_v1",
    near_opposition_tolerance=1.0e-6,
)

## 方法聚合
METHOD_CFG = MultiAnchorGaussianMethodCfg(
    state_measure=STATE_MEASURE_CFG,
    representation=REPRESENTATION_CFG,
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    fairgrad=FAIRGRAD_CFG,
    entity_permutation=ENTITY_PERMUTATION_CFG,
    joint_sign_rewrite=JOINT_SIGN_REWRITE_CFG,
)

###
#  训练器配置层：预实验与正式实验复用同一套显式 epoch/minibatch/microbatch 接口。
###
## 训练器聚合
## 正式 preset：256 epochs × 4 minibatches = 1024 updates，覆盖 8 个完整 8192-asset catalog cycles。
TRAINER_CFG = EmbodimentPretrainTrainerCfg(
    sampling=OnlineSamplingCfg(
        assets_per_minibatch=64,
        q_per_asset_per_minibatch=8,
        shuffle_assets=True,
        seed=20260813,
    ),
    max_epochs=256,
    num_minibatches=4,
    mini_epochs=1,
    microbatch_size=64,
    max_resident_assets=64,
    checkpoint_every_epochs=4,
)

## 独立 validation 使用相同固定测度，但不声明训练 cadence。
VALIDATION_CFG = ValidationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    selection_metrics=("density", "kappa"),
    seed_offset=1_000_003,
    max_resident_assets=64,
)

## 独立 evaluation 保留当前 unseen suites、训练 q-bank 与六项消融语义。
EVALUATION_CFG = EvaluationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    final_ablations=(
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "joint_token_shuffle",
    ),
    bootstrap_replicates=2_000,
    evaluation_seed_offset=2_000_003,
    training_q_bank_seed_offset=3_000_003,
    bootstrap_seed_offset=4_000_003,
    max_resident_assets=64,
)

###
#  运行配置层：输出目录、实验名、随机种子与只读 source cache
###
RUN_CFG = PretrainRunCfg(
    output_dir="logs/ssl",
    experiment_name="canonical_multi_anchor_gaussian_fairgrad_v0_7_3",
    seed=20260813,
    source_cache_root="logs/ssl/_cache/geometry_source/v1",
    source_cache_mode="readonly",
)

###
#  完整实验语义配置
###
EXPERIMENT = EmbodimentPretrainCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    trainer=TRAINER_CFG,
    run=RUN_CFG,
)

VALIDATION_EXPERIMENT = EmbodimentValidationCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    validation=VALIDATION_CFG,
    run=ValidationRunCfg(),
)

EVALUATION_EXPERIMENT = EmbodimentEvaluationCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    evaluation=EVALUATION_CFG,
    run=EvaluationRunCfg(),
)

__all__ = [
    "DATA_CFG",
    "EVALUATION_CFG",
    "EVALUATION_EXPERIMENT",
    "EXPERIMENT",
    "JOINT_SIGN_REWRITE_CFG",
    "METHOD_CFG",
    "MODEL_CFG",
    "OBJECTIVES_CFG",
    "REPRESENTATION_CFG",
    "RUN_CFG",
    "STATE_MEASURE_CFG",
    "TRAINER_CFG",
    "VALIDATION_CFG",
    "VALIDATION_EXPERIMENT",
]
