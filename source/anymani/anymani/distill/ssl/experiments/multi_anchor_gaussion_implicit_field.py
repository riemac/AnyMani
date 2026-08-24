r"""多锚点 Gaussian 隐式场实验的完整 Python 装配。

所有科研参数、训练参数和运行阶段都在本模块逐行可读。Hydra 只注册完整 `EXPERIMENT`
并接受命令行覆盖，不再用分片 YAML 拼装 method/model/objective。

实验文件名保留历史拼写 `gaussion`，避免破坏已有入口。
"""

from anymani.distill.methods.multi_anchor_gaussian_implicit_field import (
    DensityObjectiveCfg,
    DerivedFieldObjectiveCfg,
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
    GeometryLatentHeadsCfg,
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
from anymani.distill.ssl.runtime.pretrainer import (
    EmbodimentPretrainTrainerCfg,
    FinalEvaluationCfg,
    ValidationCfg,
)
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
        train_active_per_joint=1,
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
            layers=2,
            attention_heads=4,
            feedforward_width=256,
            dropout=0.0,
            max_graph_distance=8,
        ),
        heads=GeometryLatentHeadsCfg(
            zero_order_width=128,
            first_order_width=64,
            first_order_source="residual_screw",
        ),
    ),
    ssl_decoders=GeometrySSLDecoderCfg(
        density=ScalarSigmaFiLMDensityDecoderCfg(
            hidden_width=128,
            residual_blocks=3,
            sigma_reference_m=0.016,
        ),
        sensitivity=DistanceSensitivityDecoderCfg(
            coefficient_hidden_width=128,
            readout_bias=False,
            carrier_scale="inverse_sqrt",
        ),
    ),
)

## 目标配置
OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(
    density=DensityObjectiveCfg(weight=1.0),
    kappa=KappaObjectiveCfg(weight=1.0),
    derived_field=DerivedFieldObjectiveCfg(weight=1.0),
)

## 数据增强配置
JOINT_SIGN_REWRITE_CFG = JointSignRewriteCfg(probability=0.20, seed_offset=17)

## 方法聚合
METHOD_CFG = MultiAnchorGaussianMethodCfg(
    state_measure=STATE_MEASURE_CFG,
    representation=REPRESENTATION_CFG,
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    joint_sign_rewrite=JOINT_SIGN_REWRITE_CFG,
)

###
#  训练器配置层：预实验与正式实验复用同一套显式 epoch/minibatch/microbatch 接口。
###
## 验证器配置
VALIDATION_CFG = ValidationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    every_epochs=8,
    selection_metrics=("density", "kappa", "derived_field"),
    seed_offset=1_000_003,
)

## 评估器配置
FINAL_EVALUATION_CFG = FinalEvaluationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    final_ablations=(
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    ),
    bootstrap_replicates=2_000,
    evaluation_seed_offset=2_000_003,
    training_q_bank_seed_offset=3_000_003,
    bootstrap_seed_offset=4_000_003,
)

## 训练器聚合
## preset：32 个 epoch，每轮生成 4 个 minibatch；每个 512-pair minibatch 独立更新一次。
## 当前总预算恰好走完一个 catalog cycle，因此每项资产只 realization $A^{(0)}$；后续 bank 只在新 cycle 轮换。
TRAINER_CFG = EmbodimentPretrainTrainerCfg(
    sampling=OnlineSamplingCfg(
        assets_per_minibatch=64,
        q_per_asset_per_minibatch=8,
        shuffle_assets=True,
        seed=20260813,
    ),
    max_epochs=32,
    num_minibatches=4,
    mini_epochs=1,
    microbatch_size=64,
    validation=VALIDATION_CFG,
    final_evaluation=FINAL_EVALUATION_CFG,
    max_resident_assets=64,
    checkpoint_every_epochs=8,
)

###
#  运行配置层：输出目录、实验名、随机种子、阶段
###
RUN_CFG = PretrainRunCfg(
    output_dir="logs/ssl",
    experiment_name="canonical_multi_anchor_gaussian",
    seed=20260813,
    phase="pretrain",
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

__all__ = [
    "DATA_CFG",
    "EXPERIMENT",
    "FINAL_EVALUATION_CFG",
    "JOINT_SIGN_REWRITE_CFG",
    "METHOD_CFG",
    "MODEL_CFG",
    "OBJECTIVES_CFG",
    "REPRESENTATION_CFG",
    "RUN_CFG",
    "STATE_MEASURE_CFG",
    "TRAINER_CFG",
    "VALIDATION_CFG",
]
