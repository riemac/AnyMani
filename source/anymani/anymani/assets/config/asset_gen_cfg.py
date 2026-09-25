"Declares the pre-made and post-mutate recipes. The CLI requires an explicit topology path for post-mutation."

from __future__ import annotations

from pathlib import Path

from ..asset_physics import AssetPhysicsCfg, DensityProfileCfg
from ..generator.hand_generator import HandGeneratorCfg
from ..generator.mutate import (
    HandMutatorCfg,
    LimitTweakCfg,
    LinkProximalOverlapCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from ..units import cm, deg, g_cm3, mm
from ..validator.hand_rules import HandValidatorCfg
from . import AssetRunStrategyCfg

ConnectivityFacade = dict[str, dict[str, list[str]]] | None  # hand_preset -> finger_slot -> allowed connectivity recipes
EditablePath = str | Path


# ============================================================================

# ============================================================================

ASSET_PHYSICS_CFG: AssetPhysicsCfg | None = AssetPhysicsCfg(
    density=DensityProfileCfg(

        palm=g_cm3(0.55),
        finger_link=g_cm3(1.2),
        fingertip=g_cm3(0.72),
        custom_tip=g_cm3(0.72),
    )
)


# ============================================================================

# ============================================================================

HAND_PRESETS: list[str] = ["single_palm_allegro", "single_palm_leap"]

CONNECTIVITY_PRESETS: ConnectivityFacade = None


# CONNECTIVITY_PRESETS: ConnectivityFacade = {
#     "single_palm_allegro": {
#         "thumb": [
#             "allegro_thumb_full",
#             "allegro_thumb_drop_j3",
#         ],
#         "index": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#         "middle": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#         "ring": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#     },
#     "single_palm_leap": {
#         "thumb": ["leap_thumb_full"],
#         "index": ["leap_non_thumb_full"],
#         "middle": ["leap_non_thumb_full"],
#         "ring": ["leap_non_thumb_full"],
#     },
# }

PRE_MADE_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "generated"


PRE_MADE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    pre_made=HandValidatorCfg.PreMadeCfg(
        finger_count_min=3,
        require_non_thumb_with_min_revolute_dof=3,
        check_palm_thumb_binding=True,
    )
)

PRE_MADE_SHOW_REGISTRY = True

PRE_MADE_PRINT_RESULT_LIMIT: int | None = 40

PRE_MADE_CFG = HandGeneratorCfg(
    mode="made",
    artifact_level="bundle",
    output_dir=PRE_MADE_OUTPUT_DIR,
    handedness="all",
    hand_presets=list(HAND_PRESETS),
    connectivity_presets=CONNECTIVITY_PRESETS,
    mixed=True,
    missing=True,
    Validate=PRE_MADE_VALIDATOR_CFG,  # pre-made hand-level validator
    Physics=ASSET_PHYSICS_CFG,
    recolored="anatomy_soft_v1",
    max_enumerate=None,
    premade_parallel=True,
    premade_parallel_workers=None,
    premade_parallel_fallback="serial"
)


# ============================================================================

# ============================================================================


POST_MUTATE_SOURCE_TOPOLOGY_PATH: EditablePath | None = None

POST_MUTATE_PRINT_RESULT_LIMIT: int | None = 10


class QuickPostMutateCfg(HandMutatorCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    link_scale = LinkScaleCfg(
        self_mode={"identity": 0.2, "general": 0.4, "only_length": 0.4},
        scale_type="rel",

        link_scale=(0.75, 1.25, 0.9, 1.1, 0.9, 1.1),

        distrib="uniform",
        boundary_policy="clip",
    )
    link_proximal_overlap = LinkProximalOverlapCfg(
        self_mode={"identity": 0.2, "disturb": 0.5, "homologous_non_thumb": 0.3},
        overhang_delta_ratio=(-1, 2),
        max_parent_overlap_ratio=0.4,
        distrib="uniform",
        boundary_policy="clip",
    )
    mount_perturb = MountPerturbCfg(
        self_mode={"identity": 0.2, "general": 0.2, "index_ring_x_pos": 0.2, "index_ring_yaw_rot": 0.2, "index_ring": 0.2},
        pos_radius=cm(0.8),
        rot_radius=deg(5),
        mirror_x_range=(cm(-1.0), cm(1.0)),
        mirror_yaw_range=(deg(-5), deg(5)),
        thumb_pos_radius=cm(1.0),
        thumb_rot_radius=deg(5.0),
        distrib="uniform",
        boundary_policy="clip",
    )
    limit_tweak = LimitTweakCfg(
        disturb_object="independent",
        disturb_type="add",
        joint_range=(deg(-10), deg(10)),
        self_mode={"identity":0.2, "disturb":0.5, "homologous_non_thumb":0.3},

        distrib={"type": "uniform"},
        boundary_policy="clip",
    )
    tip_replace = TipReplaceCfg(
        self_mode={"identity": 0.2, "same": 0.5, "general": 0.3},
        tip_range={"cs": 0.2, "leap_cube": 0.2, "round": 0.2, "wedge": 0.2, "thinner": 0.2},
        scale=(0.9, 1.1),
        cs_ratio={"add": (-0.15, 0.15)},
    )


POST_MUTATE_MUTATOR_CFG = QuickPostMutateCfg()


POST_MUTATE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    post_mutate=HandValidatorCfg.PostMutateCfg(
        finger_count_min=3,
        require_non_thumb_with_min_revolute_dof=3,
        check_finger_spacing=True,
        min_finger_spacing=mm(5),
        check_finger_length=True,
        max_thumb_length=cm(18.5),
        max_non_thumb_length=cm(18.0),
        check_mount_consistency=True,
        sdf_device="cuda",
        sdf_mesh_backend="warp",
    )
)



POST_MUTATE_CFG = HandGeneratorCfg(
    mode="mutate",
    artifact_level="bundle",
    source_topology_dir=Path("__post_mutate_topology_dir__"),
    output_dir=Path("__post_mutate_output_dir__"),
    n_samples=20,
    post_mutate_seed=20260813,
    post_mutate_attempts_per_variant=10,
    post_mutate_require_unique_geometry=True,
    post_mutate_sdf_execution="central_gpu_batch",
    Mutate=POST_MUTATE_MUTATOR_CFG,
    Validate=POST_MUTATE_VALIDATOR_CFG,
    Physics=ASSET_PHYSICS_CFG,
    recolored="anatomy_soft_v1",
)


# ============================================================================

# ============================================================================

ASSET_RUN_STRATEGY = AssetRunStrategyCfg(
    topology_selection_mode="all",
    topology_selection_count=None,
)


__all__ = [
    "ASSET_RUN_STRATEGY",
    "ASSET_PHYSICS_CFG",
    "POST_MUTATE_CFG",
    "POST_MUTATE_PRINT_RESULT_LIMIT",
    "POST_MUTATE_SOURCE_TOPOLOGY_PATH",
    "PRE_MADE_CFG",
    "PRE_MADE_PRINT_RESULT_LIMIT",
    "PRE_MADE_SHOW_REGISTRY",
]
