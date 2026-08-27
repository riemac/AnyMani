r"""Geometry SSL Multitask Representation v0.7.3 的冻结实验快照。

本文件定义上次研究讨论完成后的唯一正式 Geometry SSL 主实验。它研究一个明确问题：统一的
PALM/JOINT/TIP entity representation 是否能同时保存零阶空间占据场与逐 JOINT 距离 Jacobian
信息，并在跨手型留出测试与 PPO 消费中保持物理可读性。

实验对象是手型 $\mathfrak m$、当前关节构型 $q$、固定 hand-frame query $x$ 和真实 collision
surface。Teacher 计算 unsigned distance $d_o(x;q)$，并以实际米制带宽定义 Gaussian density：

$$
\rho_{\sigma,o}(x;q)=\exp\!\left[-\frac{d_o(x;q)^2}{2\sigma^2}\right].
$$

距离灵敏度为

$$
\kappa_{o,i}=\frac{\partial d_o}{\partial q_i}.
$$

Density 是 owner/query/sigma 标量；kappa 是
owner/query/JOINT 标量，单位分别为无量纲和 m/rad。训练只优化这两个平等主任务；derived field
与真实 density JVP 是事后诊断，不进入 optimizer。

本实验的 retained encoder 输入、模型容量和 supervision contract 均冻结在下方配置对象中；
训练预算和运行资源属于可调整的执行参数。

Retained encoder 输出唯一的 $Z\in\mathbb R^{B\times G\times128}$，其中 $G$ 按 PALM、JOINT、TIP
实体排列。输入为当前物理 $q$、$q_{home}$、空间 screw、home collision surface、8 套有限
physical anchor bank 的当前 realization 和 graph relation。4-layer Pre-LN graph-biased
Transformer 产生最终 unified entity tokens；JOINT view 通过 routing 从同一 $Z$ gather。
Density reader 使用 2-block owner-conditioned query/sigma FiLM；kappa reader 使用 2-block
owner-query FiLM 与无 bias rank-64 JOINT 双线性读取，固定 κ 物理尺度为 $0.1\,\mathrm m/\mathrm{rad}$。

每个有效 JOINT 和每个 q 固定采样 2 条 active edge 与 1 条 structural-zero edge，active:zero 按
2:1 归约，并覆盖 owner-shell、adjacent、workspace strata。每个资产的 8 个 q 共享一次合法
entity-axis permutation；joint-sign rewrite 以 0.20 概率同步改写 $q_i,q_{home,i},\mathcal S_i$
并只验收 observable density invariance 与 kappa sign equivariance。Shared retained encoder
使用精确两任务 alpha=1 FairGrad；两个 private readers 各自更新；三组梯度分别裁剪。

正式预算固定为 seed 20260813、256 epochs、每 epoch 4 个新 minibatches、每批 64 assets × 8 q，
共 1024 optimizer updates 和 8 个完整 8192-asset catalog cycles。每 4 epochs 保存一次 full
schema-8 checkpoint。source artifacts 先以 read-write 离线准备 train 全部 8 banks 与 validation/
evaluation bank 0，正式训练和事后阶段以 readonly 消费。训练完成后按 epoch-0 naive baseline 和
validation fixed bank 选择 checkpoint，再执行 held-out evaluation、PCA 32/64/96/128 replay、
retained artifact export 与 PPO transfer。

该文件是本实验的独立版本化记录。epochs、seed、device、cache 和训练预算等执行参数可以按实验需要
修订；涉及表示容量、teacher、objective、sampling 或监督语义的重大变化，应创建新的版本化快照。
registry、训练 CLI 和 checkpoint resolved_config 都记录这份快照的路径与内容 identity。
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
from anymani.distill.models.input_adapters.geometry import GeometryEncoderCfg, SO2AnchorFrontendCfg
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GaussianProximityFieldCfg, GeometryFieldTargetCfg
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

DATA_CFG = HandAssetCatalogCfg(
    manifest="source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml",
    expected_sha256="f1398417888e7c237cbb2583dcf8e9cd10bef7fee792b307c67dfa74fb6e0698",
)

STATE_MEASURE_CFG = JointConfigurationMeasureCfg()

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

OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(
    density=DensityObjectiveCfg(),
    kappa=KappaObjectiveCfg(),
)

JOINT_SIGN_REWRITE_CFG = JointSignRewriteCfg(probability=0.20, seed_offset=17)
ENTITY_PERMUTATION_CFG = EntityPermutationCfg(enabled=True, seed_offset=31_337)
FAIRGRAD_CFG = FairGradCfg(
    algorithm="fairgrad_alpha_1_two_task_analytic_v1",
    near_opposition_tolerance=1.0e-6,
)

METHOD_CFG = MultiAnchorGaussianMethodCfg(
    state_measure=STATE_MEASURE_CFG,
    representation=REPRESENTATION_CFG,
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    fairgrad=FAIRGRAD_CFG,
    entity_permutation=ENTITY_PERMUTATION_CFG,
    joint_sign_rewrite=JOINT_SIGN_REWRITE_CFG,
)

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

VALIDATION_CFG = ValidationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    selection_metrics=("density", "kappa"),
    seed_offset=1_000_003,
    max_resident_assets=64,
)

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

RUN_CFG = PretrainRunCfg(
    output_dir="logs/ssl",
    experiment_name="geometry_ssl_multitask_representation_v0_7_3",
    seed=20260813,
    source_cache_root="logs/ssl/_cache/geometry_source/v1",
    source_cache_mode="auto",
)

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
    run=ValidationRunCfg(experiment_name="geometry_ssl_multitask_representation_v0_7_3_validation"),
)

EVALUATION_EXPERIMENT = EmbodimentEvaluationCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    evaluation=EVALUATION_CFG,
    run=EvaluationRunCfg(experiment_name="geometry_ssl_multitask_representation_v0_7_3_evaluation"),
)

__all__ = [
    "DATA_CFG",
    "EVALUATION_CFG",
    "EVALUATION_EXPERIMENT",
    "ENTITY_PERMUTATION_CFG",
    "EXPERIMENT",
    "FAIRGRAD_CFG",
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
