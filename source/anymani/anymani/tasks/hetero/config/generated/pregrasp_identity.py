r"""Generated heterogeneous formal pregrasp的gate、physics与search协议身份。

Cache record自洽只说明“某个gate/physics/search下有效”，不能说明它属于当前任务。Formal task必须从自己的scene
与sealed科学门构造期望身份，再验证record；禁止从cache命中的record反向接受任意宽松gate或不同物理配置。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from anymani.pregrasp import PregraspGate, PregraspLookupKey
from anymani.pregrasp.schema import stable_digest

DEX_CUBE_ASSET_ID = "DexCube"
DEX_CUBE_SHA256 = "7a5c015690652f4ca1d62ed757d494c51f052067bd037372c8fd581bc92d437b"
FORMAL_OBJECT_SCALE = 1.2
FORMAL_OBJECT_DENSITY_KG_M3 = 400.0  # authored cfg；DexCube USD fixed mass在PhysX中优先
FORMAL_STATIC_FRICTION = 1.0
FORMAL_DYNAMIC_FRICTION = 1.0
FORMAL_RESTITUTION = 0.0
FORMAL_SOLVER_POSITION_ITERATIONS = 8
FORMAL_SOLVER_VELOCITY_ITERATIONS = 0
FORMAL_PHYSICS_DT_S = 1.0 / 120.0
FORMAL_POLICY_DT_S = 0.05
FORMAL_CONTACT_FORCE_THRESHOLD_N = 0.25
FORMAL_CONTACT_EMA_ALPHA = 0.5

# 三个独立prestartup probes读取PhysX view；不能用理论$s^3/s^5$替换实际fixed-mass/$s^2$行为。
_OBSERVED_OBJECT_PROPERTIES = {
    1.1: (0.2160000056028366, 0.00015681602235417813),
    1.2: (0.2160000056028366, 0.0001866240199888125),
    1.25: (0.2160000056028366, 0.000202499984879978),
}


def formal_pregrasp_gate() -> PregraspGate:
    r"""返回contact/support/gravity与basin共同使用的唯一formal数值门。"""

    return PregraspGate(
        min_tip_ge_2_fraction=0.8,  # 6 s认证尾窗至少80% samples具有两个TIP
        min_tip_ge_3_fraction=0.8,  # gravity-robust tier的三个TIP persistence
        max_finger_non_tip_fraction=0.0,  # contact tier不接受任何>0.25 N finger non-tip sample
        max_penetration_depth_m=0.001,  # 1 mm；probe数值噪声约1--3 um
        max_anchor_distance_m=0.025,  # 与任务position success anchor同为25 mm
        max_linear_velocity_rms_m_s=0.05,
        max_angular_velocity_rms_rad_s=0.5,
        max_object_orientation_drift_rad=0.5,
        min_joint_limit_margin_rad=0.0,
        max_target_tracking_error_rms_rad=0.1,
        max_joint_effort_rms_N_m=2.0,
        min_basin_success_fraction=0.8,
        required_gravity_directions=6,
    )


FORMAL_PREGRASP_GATE = formal_pregrasp_gate()


def formal_physics_identity(*, object_scale: float, cube_sha256: str) -> dict[str, Any]:
    r"""由scene常量与三scale实测PhysX属性构造exact lookup identity。

    Args:
        object_scale (float): prestartup absolute DexCube scale，只接受1.1、1.2、1.25 anchors。
        cube_sha256 (str): 当前进程通过``retrieve_file_path``解析后计算的真实USD bytes SHA-256。

    Returns:
        dict[str, Any]: 完整solver/material/mass/inertia/contact/controller identity。
    """

    scale = float(object_scale)
    if scale not in _OBSERVED_OBJECT_PROPERTIES:
        raise ValueError("formal pregrasp physics identity only covers scale anchors 1.1, 1.2, and 1.25")
    if cube_sha256 != DEX_CUBE_SHA256:
        raise ValueError("resolved DexCube bytes disagree with formal SHA-256")
    mass_kg, principal_inertia = _OBSERVED_OBJECT_PROPERTIES[scale]
    return {
        "identity_version": "hetero-pregrasp-physics-v2",
        "isaac_sim": "5.1.0",
        "object_asset_id": DEX_CUBE_ASSET_ID,
        "object_sha256": cube_sha256,
        "object_absolute_scale": scale,
        "object_authored_density_kg_m3": FORMAL_OBJECT_DENSITY_KG_M3,
        "object_observed_mass_kg": mass_kg,
        "object_observed_principal_inertia_kg_m2": [principal_inertia] * 3,
        "object_mass_policy": "usd_fixed_mass_overrides_authored_density",
        "object_inertia_scale_law": "observed_approximately_s_squared",
        "physics_dt_s": FORMAL_PHYSICS_DT_S,
        "policy_dt_s": FORMAL_POLICY_DT_S,
        "solver": {
            "position_iterations": FORMAL_SOLVER_POSITION_ITERATIONS,
            "velocity_iterations": FORMAL_SOLVER_VELOCITY_ITERATIONS,
            "bounce_threshold_velocity_m_s": 0.2,
        },
        "material": {
            "static_friction": FORMAL_STATIC_FRICTION,
            "dynamic_friction": FORMAL_DYNAMIC_FRICTION,
            "restitution": FORMAL_RESTITUTION,
            "friction_combine_mode": "average",
            "restitution_combine_mode": "average",
        },
        "contact": {
            "sensor_count": 24,
            "filter": "robot_link_to_object_only",
            "force_threshold_N": FORMAL_CONTACT_FORCE_THRESHOLD_N,
            "ema_alpha": FORMAL_CONTACT_EMA_ALPHA,
            "friction_forces_included": True,
        },
        "structural_collision": "filter_palm_finger_and_same_finger_keep_cross_finger",
        "controller": {
            "type": "implicit_pd",
            "candidate_state": "separate_actual_q_state_and_pd_preload_target",
            "effort_source": "implicit_actuator_computed_torque",
        },
        "action_lifecycle": "one_1_over_24_rad_target_update_per_20Hz_policy_step_hold_6x120Hz",
        "hand_frame": "fixed_palm_up_semantic_plus_z",
    }


FORMAL_SEARCH_PROTOCOL = {
    "certifier": "hetero-contact-basin-certification-v1",
    "point_basin_algorithm": "hetero-point-basin-v3",
    "candidate_state": "separate_actual_q_state_and_pd_preload_target",
    "minimum_trials": 64,
    "minimum_success_fraction": 0.8,
    "center_control_trials_per_asset": 1,
    "q_perturbation_rad": [-0.02, 0.02],
    "position_perturbation_h_m": [-0.002, 0.002],
    "rotation_vector_perturbation_h_rad": [-0.05235987755982989, 0.05235987755982989],  # exact serialized ±3°
    "linear_velocity_perturbation_h_m_s": [-0.01, 0.01],
    "angular_velocity_perturbation_h_rad_s": [-0.1, 0.1],
    "settle_policy_steps": 120,
    "settle_prefix_policy_steps": 80,
    "certification_tail_policy_steps": 40,
    "physics_substeps_per_policy_step": 6,
}
FORMAL_SEARCH_PROTOCOL_DIGEST = stable_digest(FORMAL_SEARCH_PROTOCOL)


def _plain_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    r"""验证nested search字段是mapping，避免缺字段被``None``静默接受。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"formal pregrasp {field_name} must be a mapping")
    return value


def _require_exact(actual: Any, expected: Any, field_name: str) -> None:
    r"""对协议离散/数值字段执行exact JSON equality。"""

    if stable_digest({"value": actual}) != stable_digest({"value": expected}):
        raise ValueError(f"formal pregrasp {field_name} disagrees with expected protocol")


def validate_formal_search_identity(search_identity: Mapping[str, Any]) -> None:
    r"""验证允许不同seed/lineage但不允许改变扰动与时序语义的basin协议。"""

    _require_exact(search_identity.get("algorithm"), FORMAL_SEARCH_PROTOCOL["certifier"], "certifier")
    _require_exact(
        search_identity.get("candidate_state_semantics"), FORMAL_SEARCH_PROTOCOL["candidate_state"], "candidate state"
    )
    protocol = _plain_mapping(search_identity.get("basin_protocol"), "basin protocol")
    exact_fields = {
        "algorithm": FORMAL_SEARCH_PROTOCOL["point_basin_algorithm"],
        "basin_center_control_trials_per_asset": FORMAL_SEARCH_PROTOCOL["center_control_trials_per_asset"],
        "refinement_q_offset_rad": FORMAL_SEARCH_PROTOCOL["q_perturbation_rad"],
        "refinement_object_offset_h_m": FORMAL_SEARCH_PROTOCOL["position_perturbation_h_m"],
        "basin_rotation_vector_h_rad": FORMAL_SEARCH_PROTOCOL["rotation_vector_perturbation_h_rad"],
        "basin_linear_velocity_h_m_s": FORMAL_SEARCH_PROTOCOL["linear_velocity_perturbation_h_m_s"],
        "basin_angular_velocity_h_rad_s": FORMAL_SEARCH_PROTOCOL["angular_velocity_perturbation_h_rad_s"],
        "settle_policy_steps": FORMAL_SEARCH_PROTOCOL["settle_policy_steps"],
        "settle_prefix_policy_steps": FORMAL_SEARCH_PROTOCOL["settle_prefix_policy_steps"],
        "certification_tail_policy_steps": FORMAL_SEARCH_PROTOCOL["certification_tail_policy_steps"],
        "physics_substeps_per_policy_step": FORMAL_SEARCH_PROTOCOL["physics_substeps_per_policy_step"],
    }
    for field_name, expected in exact_fields.items():
        _require_exact(protocol.get(field_name), expected, field_name)
    trials = int(search_identity.get("perturbation_trials", 0))
    successes = int(search_identity.get("perturbation_successes", -1))
    protocol_trials = int(protocol.get("basin_trial_count", 0))
    if trials != protocol_trials or trials < int(FORMAL_SEARCH_PROTOCOL["minimum_trials"]):
        raise ValueError("formal pregrasp basin trial count is insufficient or inconsistent")
    if successes < math.ceil(float(FORMAL_SEARCH_PROTOCOL["minimum_success_fraction"]) * trials):
        raise ValueError("formal pregrasp search identity reports insufficient perturbation successes")
    for digest_field in ("basin_artifact_sha256", "nominal_artifact_sha256", "nominal_record_digest"):
        if not isinstance(search_identity.get(digest_field), str) or re.fullmatch(
            r"[0-9a-f]{64}", str(search_identity[digest_field])
        ) is None:
            raise ValueError(f"formal pregrasp {digest_field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class FormalPregraspCatalogIdentity:
    r"""Current task-owned expected identity，先于provider query验证cache candidate。"""

    object_scale: float
    cube_sha256: str
    gate_digest: str
    physics_identity: Mapping[str, Any]
    search_protocol_digest: str

    @classmethod
    def build(cls, *, object_scale: float, cube_sha256: str) -> FormalPregraspCatalogIdentity:
        r"""由当前scene verified bytes与固定任务常量构造identity。"""

        physics = formal_physics_identity(object_scale=object_scale, cube_sha256=cube_sha256)
        return cls(
            object_scale=float(object_scale),
            cube_sha256=cube_sha256,
            gate_digest=FORMAL_PREGRASP_GATE.digest,
            physics_identity=physics,
            search_protocol_digest=FORMAL_SEARCH_PROTOCOL_DIGEST,
        )

    def validate_lookup_key(self, lookup_key: PregraspLookupKey) -> None:
        r"""拒绝object、gate、physics或search协议任一漂移的完整lookup key。"""

        if lookup_key.cube_asset_id != DEX_CUBE_ASSET_ID or lookup_key.cube_asset_sha256 != self.cube_sha256:
            raise ValueError("pregrasp lookup does not bind current resolved DexCube bytes")
        if lookup_key.support_mode != "palm_supported":
            raise ValueError("formal heterogeneous pregrasp requires palm_supported mode")
        if lookup_key.gate_digest != self.gate_digest:
            raise ValueError("pregrasp lookup gate differs from formal task gate")
        if stable_digest(lookup_key.physics_identity) != stable_digest(self.physics_identity):
            raise ValueError("pregrasp lookup physics differs from current formal scene identity")
        validate_formal_search_identity(lookup_key.search_identity)


__all__ = [
    "DEX_CUBE_ASSET_ID",
    "DEX_CUBE_SHA256",
    "FORMAL_CONTACT_EMA_ALPHA",
    "FORMAL_CONTACT_FORCE_THRESHOLD_N",
    "FORMAL_DYNAMIC_FRICTION",
    "FORMAL_OBJECT_DENSITY_KG_M3",
    "FORMAL_OBJECT_SCALE",
    "FORMAL_POLICY_DT_S",
    "FORMAL_PREGRASP_GATE",
    "FORMAL_RESTITUTION",
    "FORMAL_SEARCH_PROTOCOL",
    "FORMAL_SEARCH_PROTOCOL_DIGEST",
    "FORMAL_SOLVER_POSITION_ITERATIONS",
    "FORMAL_SOLVER_VELOCITY_ITERATIONS",
    "FORMAL_STATIC_FRICTION",
    "FormalPregraspCatalogIdentity",
    "formal_physics_identity",
    "formal_pregrasp_gate",
    "validate_formal_search_identity",
]
