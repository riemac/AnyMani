"""N040 proper-SE(3) density and material Jacobian snapshot. The selected formal run uses 512 epochs, 2,048 updates, and 1,048,576 fresh asset-state pairs."""


from anymani.distill.methods.density_material_jacobian import (
    DensityMaterialJacobianObjectivesCfg,
    DensityObjectiveCfg,
    GammaChannelScaleCfg,
    MaterialJacobianObjectiveCfg,
    MaterialPointSamplingCfg,
    SE3CoordinateRewriteCfg,
    SE3DensityMaterialJacobianMethodCfg,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import (
    EntityPermutationCfg,
    FairGradCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import ScalarSigmaFiLMDensityDecoderCfg
from anymani.distill.models.decoders.representations.material_point_jacobian import (
    AnchorRelationalJacobianDecoderCfg,
)
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoderCfg,
)
from anymani.distill.models.se3_density_material_jacobian_ssl import SE3DensityMaterialJacobianModelCfg
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GaussianProximityFieldCfg, GeometryFieldTargetCfg
from anymani.distill.representations.targets.material_point_jacobian import MaterialPointRelationJacobianCfg
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg
from anymani.distill.ssl.post_training import EmbodimentEvaluationCfg, EvaluationCfg, EvaluationRunCfg
from anymani.distill.ssl.runtime.pretrainer import EmbodimentPretrainTrainerCfg, ExecutionPrecisionCfg
from anymani.distill.ssl.runtime.run import PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

DATA_CFG = HandAssetCatalogCfg(
    manifest="source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml",
    expected_sha256="671e204e8542e69fab7adc05bb3516a28993a7aa744a333b31811eb2e9c0eeb8",
)

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
        fixed_bandwidths_m=(0.004, 0.016, 0.064),
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
    target=GeometryFieldTargetCfg(),
)

ENCODER_CFG = SE3InvariantGeometryEncoderCfg(
    frontend=SE3InvariantAnchorFrontendCfg(
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
)

MODEL_CFG = SE3DensityMaterialJacobianModelCfg(
    encoder=ENCODER_CFG,
    density=ScalarSigmaFiLMDensityDecoderCfg(
        hidden_width=128,
        residual_blocks=2,
        sigma_reference_m=0.016,
    ),
    material_jacobian=AnchorRelationalJacobianDecoderCfg(
        latent_width=128,
        relation_width=64,
        hidden_width=128,
    ),
)

OBJECTIVES_CFG = DensityMaterialJacobianObjectivesCfg(
    density=DensityObjectiveCfg(),
    material_jacobian=MaterialJacobianObjectiveCfg(
        channel_scale=GammaChannelScaleCfg(height=0.30, radius=0.30, dot=0.13, chirality=0.13)
    ),
)

METHOD_CFG = SE3DensityMaterialJacobianMethodCfg(
    state_measure=JointConfigurationMeasureCfg(),
    representation=REPRESENTATION_CFG,
    material_target=MaterialPointRelationJacobianCfg(
        length_scale_m=0.1,
        distance_epsilon_m=1.0e-9,
        plane_radius_epsilon_m=1.0e-9,
    ),
    material_sampling=MaterialPointSamplingCfg(
        train_active_per_joint=2,
        train_zero_per_joint=1,
        fixed_active_per_joint=4,
        fixed_zero_per_joint=4,
        points_per_edge=1,
        seed_offset=71_117,
    ),
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    fairgrad=FairGradCfg(),
    entity_permutation=EntityPermutationCfg(enabled=True, seed_offset=31_337),
    joint_sign_rewrite=JointSignRewriteCfg(probability=0.20, seed_offset=17),
    se3_coordinate_rewrite=SE3CoordinateRewriteCfg(
        probability=1.0,
        translation_half_extent_m=0.05,
        seed_offset=93_113,
    ),
)

TRAINER_CFG = EmbodimentPretrainTrainerCfg(
    sampling=OnlineSamplingCfg(
        assets_per_minibatch=64,
        q_per_asset_per_minibatch=8,
        shuffle_assets=True,
        seed=20260830,
    ),
    max_epochs=512,
    num_minibatches=4,
    mini_epochs=1,
    microbatch_size=64,
    checkpoint_every_epochs=32,
    emit_compression_basis=False,
    execution=ExecutionPrecisionCfg(
        teacher_dtype="float32",
        parameter_dtype="float32",
        model_autocast_dtype="bfloat16",
        loss_dtype="float32",
        fairgrad_accumulation_dtype="float64",
        allow_tf32=False,
        compile_enabled=True,
        compile_mode="reduce-overhead",
    ),
)

EVALUATION_CFG = EvaluationCfg(
    q_per_asset=64,
    assets_per_minibatch=2,
    q_per_asset_per_minibatch=2,
    final_ablations=("query_only", "same_asset_q_shuffle", "cross_asset_shuffle", "joint_token_shuffle"),
    bootstrap_replicates=2_000,
    evaluation_seed_offset=2_000_003,
    bootstrap_seed_offset=4_000_003,
    max_resident_assets=8,
    execution=TRAINER_CFG.execution,
)

RUN_CFG = PretrainRunCfg(
    output_dir="logs/ssl",
    experiment_name="geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched",
    seed=20260830,
    source_cache_root="logs/ssl/_cache/geometry_source/v2",
    source_cache_mode="auto",
)

EXPERIMENT = EmbodimentPretrainCfg(data=DATA_CFG, method=METHOD_CFG, trainer=TRAINER_CFG, run=RUN_CFG)
EVALUATION_EXPERIMENT = EmbodimentEvaluationCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    evaluation=EVALUATION_CFG,
    run=EvaluationRunCfg(
        experiment_name="geometry_ssl_density_material_jacobian_se3_v0_8_1_evaluation",
        seed=20260830,
    ),
)

__all__ = [
    "DATA_CFG",
    "ENCODER_CFG",
    "EVALUATION_CFG",
    "EVALUATION_EXPERIMENT",
    "EXPERIMENT",
    "METHOD_CFG",
    "MODEL_CFG",
    "OBJECTIVES_CFG",
    "REPRESENTATION_CFG",
    "RUN_CFG",
    "TRAINER_CFG",
]
