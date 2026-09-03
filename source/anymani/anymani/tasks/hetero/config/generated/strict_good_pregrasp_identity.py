r"""MVP80 strict v5 Sobol→Top-32→3×128 CEM generation identity。

该模块先独立于active runtime identity存在，使v5未完成时现有v4 task仍可运行。只有80项Top-8全部通过
strict gate并发布后，runtime的``good_pregrasp_identity.py``才原子切换到本协议。
"""

from __future__ import annotations

from typing import Any

from anymani.pregrasp.schema import stable_digest
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE

from .pregrasp_identity import DEX_CUBE_SHA256, formal_physics_identity

STRICT_GOOD_PREGRASP_OBJECT_SCALE = 1.1
STRICT_GOOD_PREGRASP_SEED = 20260902
STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES = 256
STRICT_GOOD_PREGRASP_PHYSICS_TOP_K = 32
STRICT_GOOD_PREGRASP_CEM_ROUNDS = 3
STRICT_GOOD_PREGRASP_CEM_CANDIDATES = 128
STRICT_GOOD_PREGRASP_CEM_ELITES = 16
STRICT_GOOD_PREGRASP_REQUIRE_STRICT = True
STRICT_GOOD_PREGRASP_CATALOG_ROOT = "outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5"
STRICT_GOOD_PREGRASP_EVIDENCE_ROOT = "outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5"


def strict_good_pregrasp_generation_identity() -> dict[str, Any]:
    r"""返回strict v5 proposal、refinement、physics与gate完整协议。"""

    return {
        "algorithm": "mvp80-strict-good-pregrasp-v5",
        "seed": STRICT_GOOD_PREGRASP_SEED,
        "object_scale": STRICT_GOOD_PREGRASP_OBJECT_SCALE,
        "object_orientation_h_wxyz": [1.0, 0.0, 0.0, 0.0],
        "initial_proposal": {
            "type": "scrambled_sobol",
            "dimensions": 13,
            "candidates_per_asset": STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
            "joint_parameterization": "blend_plus_depth_and_finger_synergies",
            "object_parameterization": "opposition_mix_plus_xyz_clearance",
            "cheap_tip_clearance_band_m": [0.06, 0.10],
            "object_xy_offset_half_width_m": [0.04, 0.03],
            "object_xy_offset_distribution": "signed_cubic_center_dense",
            "object_position_bounds_h_m": [[-0.06, 0.06], [0.03, 0.14], [0.055, 0.065]],
            "comfort_joint_margin_fraction": 0.11,
        },
        "physics_screen": {
            "geometry_top_k_per_asset": STRICT_GOOD_PREGRASP_PHYSICS_TOP_K,
            "cold_reset_physics_steps": 120,
            "physics_hz": 120,
            "policy_hz": 20,
            "early_peak_seconds": 0.2,
            "palm_tail_seconds": 0.5,
        },
        "refinement": {
            "type": "per_asset_elite_mixture_low_rank_gaussian_cem",
            "position_center_feedback": "minus_physx_contact_normal_times_1p10_depth_plus_0p25mm",
            "settle_height_feedback": "if_initial_penetration_le_0p5mm_lower_by_clamped_displacement_minus_4p5mm",
            "random_stream_key": "formal_dataset_row",
            "strict_mode_exploitation": {
                "max_proposals_per_round": 96,
                "joint_std_rad": 0.0005,
                "position_std_m": [0.00005, 0.00005, 0.000025],
                "activation": "one_to_seven_strict_or_normalized_gate_violation_le_0p35",
            },
            "joint_pca_dimensions": 4,
            "object_position_dimensions": 3,
            "rounds_max": STRICT_GOOD_PREGRASP_CEM_ROUNDS,
            "proposals_per_round_per_failed_asset": STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
            "full_physics_candidates_per_round": STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
            "elite_physical_candidates": STRICT_GOOD_PREGRASP_CEM_ELITES,
            "elite_proposal_counts_descending": [24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3],
            "per_asset_manual_parameters": False,
        },
        "hard_gate": {
            "angular_velocity_kind": "total_l2",
            **MVP80_STRICT_GOOD_PREGRASP_GATE.to_dict(),
            "gate_digest": MVP80_STRICT_GOOD_PREGRASP_GATE.digest,
        },
        "publication": {"top_k_per_asset": 8, "require_all_80_assets": True},
    }


STRICT_GOOD_PREGRASP_GENERATION_IDENTITY = strict_good_pregrasp_generation_identity()
STRICT_GOOD_PREGRASP_GENERATION_DIGEST = stable_digest(STRICT_GOOD_PREGRASP_GENERATION_IDENTITY)
STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY = formal_physics_identity(
    object_scale=STRICT_GOOD_PREGRASP_OBJECT_SCALE,
    cube_sha256=DEX_CUBE_SHA256,
)
STRICT_GOOD_PREGRASP_PHYSICS_DIGEST = stable_digest(STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY)


__all__ = [
    "STRICT_GOOD_PREGRASP_CATALOG_ROOT",
    "STRICT_GOOD_PREGRASP_CEM_CANDIDATES",
    "STRICT_GOOD_PREGRASP_CEM_ELITES",
    "STRICT_GOOD_PREGRASP_CEM_ROUNDS",
    "STRICT_GOOD_PREGRASP_EVIDENCE_ROOT",
    "STRICT_GOOD_PREGRASP_GENERATION_DIGEST",
    "STRICT_GOOD_PREGRASP_GENERATION_IDENTITY",
    "STRICT_GOOD_PREGRASP_OBJECT_SCALE",
    "STRICT_GOOD_PREGRASP_PHYSICS_DIGEST",
    "STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY",
    "STRICT_GOOD_PREGRASP_PHYSICS_TOP_K",
    "STRICT_GOOD_PREGRASP_REQUIRE_STRICT",
    "STRICT_GOOD_PREGRASP_SEED",
    "STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES",
    "strict_good_pregrasp_generation_identity",
]
