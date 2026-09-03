r"""已发布宽松v4 good-pregrasp的冻结复现身份。

v4只作为历史运行对照；正式MVP runtime切换strict v5后，旧生成脚本仍从本模块取得原digest，避免
同名active identity变化后无法重现既有v4 catalog。
"""

from __future__ import annotations

from typing import Any

from anymani.pregrasp.schema import stable_digest

from .pregrasp_identity import DEX_CUBE_SHA256, formal_physics_identity

GOOD_PREGRASP_OBJECT_SCALE = 1.1
GOOD_PREGRASP_SEED = 20260902
GOOD_PREGRASP_CANDIDATE_COUNT = 512
GOOD_PREGRASP_CANDIDATES_PER_BATCH = 32
GOOD_PREGRASP_REQUIRE_STRICT = False
GOOD_PREGRASP_CATALOG_ROOT = "outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v4"


def good_pregrasp_generation_identity() -> dict[str, Any]:
    r"""返回已冻结的v4宽松proposal/physics gate协议。"""

    return {
        "algorithm": "mvp80-good-pregrasp-v4",
        "seed": GOOD_PREGRASP_SEED,
        "candidate_count_per_asset": GOOD_PREGRASP_CANDIDATE_COUNT,
        "candidate_batch_size_per_asset": GOOD_PREGRASP_CANDIDATES_PER_BATCH,
        "q_seed_families": ["blend_0p55", "blend_0p70", "blend_0p85", "n000"],
        "q_noise_rad": [-0.1, 0.1],
        "object_xy_noise_m": [-0.005, 0.005],
        "object_scale": GOOD_PREGRASP_OBJECT_SCALE,
        "object_orientation_h_wxyz": [1.0, 0.0, 0.0, 0.0],
        "cold_reset_physics_steps": 120,
        "policy_substeps": 6,
        "hard_gate": {
            "joint_margin_fraction": 0.0,
            "tip_center_distance_m": 0.125,
            "sector_min_deg": 10.0,
            "penetration_depth_m": 0.0005,
            "object_displacement_m": 0.015,
            "object_tilt_deg": 10.0,
            "peak_linear_velocity_m_s": 0.5,
            "peak_off_axis_angular_velocity_rad_s": 8.0,
            "palm_contact_fraction": 0.5,
        },
    }


GOOD_PREGRASP_GENERATION_IDENTITY = good_pregrasp_generation_identity()
GOOD_PREGRASP_GENERATION_DIGEST = stable_digest(GOOD_PREGRASP_GENERATION_IDENTITY)
GOOD_PREGRASP_PHYSICS_IDENTITY = formal_physics_identity(
    object_scale=GOOD_PREGRASP_OBJECT_SCALE,
    cube_sha256=DEX_CUBE_SHA256,
)
GOOD_PREGRASP_PHYSICS_DIGEST = stable_digest(GOOD_PREGRASP_PHYSICS_IDENTITY)


__all__ = [
    "GOOD_PREGRASP_CANDIDATE_COUNT",
    "GOOD_PREGRASP_CANDIDATES_PER_BATCH",
    "GOOD_PREGRASP_CATALOG_ROOT",
    "GOOD_PREGRASP_GENERATION_DIGEST",
    "GOOD_PREGRASP_GENERATION_IDENTITY",
    "GOOD_PREGRASP_OBJECT_SCALE",
    "GOOD_PREGRASP_PHYSICS_DIGEST",
    "GOOD_PREGRASP_PHYSICS_IDENTITY",
    "GOOD_PREGRASP_REQUIRE_STRICT",
    "GOOD_PREGRASP_SEED",
    "good_pregrasp_generation_identity",
]
