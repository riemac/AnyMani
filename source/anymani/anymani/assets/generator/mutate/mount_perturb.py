"Perturbs finger mounts in the declared palm-local frame while preserving finger topology."

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector3, _ensure_tuple
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


_MODE_GENERAL = "general"
_MODE_IDENTITY = "identity"
_MODE_INDEX_RING_YAW = "index_ring_yaw_rot"
_MODE_INDEX_RING_X = "index_ring_x_pos"
_MODE_INDEX_RING_BOTH = "index_ring"

_ALL_SELF_MODES = {
    _MODE_GENERAL,
    _MODE_IDENTITY,
    _MODE_INDEX_RING_YAW,
    _MODE_INDEX_RING_X,
    _MODE_INDEX_RING_BOTH,
}

_INDEX_RING_MODES = {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}

_MODE_TOLERANCE = 1e-9


@dataclass
class MountPerturbCfg(MutatorBaseCfg):
    "Proposal settings for finger-root position and rotation in the palm-local frame."

    class_type: type["MountPerturbMutator"] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."

    self_mode: Literal[
        "identity",
        "general",
        "index_ring_yaw_rot",
        "index_ring_x_pos",
        "index_ring",
    ] | dict[str, float] | None = _MODE_GENERAL
    "Identity, shared, or per-finger mutation proposal mode."

    pos_radius: float | Vector3 | None = None
    "Position perturbation radius in meters."

    rot_radius: float | Vector3 | None = None
    "Rotation perturbation radius in radians."

    mirror_yaw_range: Vector2 | None = None
    "Finger-mount yaw interval for mirrored proposals, in radians."

    mirror_x_range: Vector2 | None = None
    "Palm-frame x-position interval for mirrored finger-mount proposals, in meters."

    thumb_pos_radius: float | Vector3 | None = None
    "Thumb position-proposal radius in meters."

    thumb_rot_radius: float | Vector3 | None = None
    "Thumb rotation-proposal radius in radians."

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    "Proposal distribution used inside the declared parameter interval."

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    "Action for proposals outside their permitted domain."

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    "Resolved proposal modes with positive probability."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        self.class_type = MountPerturbMutator
        self._active_modes = _resolve_active_modes(self.self_mode)
        _validate_mode_fields(self)


class MountPerturbMutator(MutatorBase):
    "Applies a typed geometry proposal to a hand copy and records sampled values."

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        "Stores the palm-local mount proposal config for later sampling."

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        'Summarizes proposal settings and sample outcomes.'

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        "Plans finger-root pose changes in the palm-local frame without changing the finger chain."

        sample = _normalize_sample_payload(sampled_params, self.cfg)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["mount_perturb"] = sample
        patch.metadata["post_mutate_mount_perturb"] = sample


        if resolved_mode == _MODE_IDENTITY:
            return patch

        if resolved_mode == _MODE_GENERAL:
            self._plan_general_patch(target, sample=sample, patch=patch)
            return patch

        self._plan_index_ring_patch(target, sample=sample, patch=patch)
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        'Samples parameters for the resolved mutation mode.'

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb resolved mode: {resolved_mode!r}")

        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY}

        if resolved_mode == _MODE_GENERAL:
            return {
                "resolved_self_mode": _MODE_GENERAL,
                "finger_deltas": {
                    finger.name: {
                        "delta_pos_local": _sample_ball_vector(
                            self.cfg.pos_radius,
                            distrib=self.cfg.distrib,
                            boundary_policy=self.cfg.boundary_policy,
                        ),
                        "delta_rotvec_local": _sample_ball_vector(
                            self.cfg.rot_radius,
                            distrib=self.cfg.distrib,
                            boundary_policy=self.cfg.boundary_policy,
                        ),
                    }
                    for finger in target.fingers
                },
            }

        sample: dict[str, Any] = {"resolved_self_mode": resolved_mode}
        if resolved_mode in {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH}:
            sample["mirror_yaw"] = _sample_scalar(
                self.cfg.mirror_yaw_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        if resolved_mode in {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}:
            sample["mirror_x"] = _sample_scalar(
                self.cfg.mirror_x_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        if _hand_has_finger(target, "thumb"):
            sample["thumb_delta_pos_local"] = _sample_ball_vector(
                self.cfg.thumb_pos_radius,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
            sample["thumb_delta_rotvec_local"] = _sample_ball_vector(
                self.cfg.thumb_rot_radius,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        return sample

    def _plan_general_patch(self, target: HandCfg, *, sample: dict[str, Any], patch: HandPatch) -> None:

        finger_deltas = dict(sample.get("finger_deltas", {}))
        for finger_index, finger in enumerate(target.fingers):
            finger_payload = dict(finger_deltas.get(finger.name, {}))
            delta_pos_local = _as_vector3(finger_payload.get("delta_pos_local", (0.0, 0.0, 0.0)))
            delta_rotvec_local = _as_vector3(finger_payload.get("delta_rotvec_local", (0.0, 0.0, 0.0)))
            new_mount = _apply_local_mount_delta(
                finger.mount,
                delta_pos_local=delta_pos_local,
                delta_rotvec_local=delta_rotvec_local,
            )
            patch.add(
                ("finger", finger_index, "mount"),
                _mount_replacer(finger_index=finger_index, new_mount=new_mount),
            )

    def _plan_index_ring_patch(self, target: HandCfg, *, sample: dict[str, Any], patch: HandPatch) -> None:

        finger_index_map = {finger.name: index for index, finger in enumerate(target.fingers)}
        mirror_yaw = float(sample.get("mirror_yaw", 0.0))
        mirror_x = float(sample.get("mirror_x", 0.0))



        sample["mirror_pair_applied"] = "index" in finger_index_map and "ring" in finger_index_map

        if sample["mirror_pair_applied"]:
            if "index" in finger_index_map:
                index_mount = target.fingers[finger_index_map["index"]].mount
                index_new_mount = _apply_mount_delta_with_palm_shift(
                    index_mount,
                    delta_pos_palm=(mirror_x, 0.0, 0.0),
                    delta_rotvec_local=(0.0, 0.0, -mirror_yaw),
                )
                patch.add(
                    ("finger", finger_index_map["index"], "mount"),
                    _mount_replacer(finger_index=finger_index_map["index"], new_mount=index_new_mount),
                )
            if "ring" in finger_index_map:
                ring_mount = target.fingers[finger_index_map["ring"]].mount
                ring_new_mount = _apply_mount_delta_with_palm_shift(
                    ring_mount,
                    delta_pos_palm=(-mirror_x, 0.0, 0.0),
                    delta_rotvec_local=(0.0, 0.0, mirror_yaw),
                )
                patch.add(
                    ("finger", finger_index_map["ring"], "mount"),
                    _mount_replacer(finger_index=finger_index_map["ring"], new_mount=ring_new_mount),
                )


        if "thumb" in finger_index_map:
            thumb_delta_pos_local = _as_vector3(sample.get("thumb_delta_pos_local", (0.0, 0.0, 0.0)))
            thumb_delta_rotvec_local = _as_vector3(sample.get("thumb_delta_rotvec_local", (0.0, 0.0, 0.0)))
            thumb_mount = target.fingers[finger_index_map["thumb"]].mount
            thumb_new_mount = _apply_local_mount_delta(
                thumb_mount,
                delta_pos_local=thumb_delta_pos_local,
                delta_rotvec_local=thumb_delta_rotvec_local,
            )
            patch.add(
                ("finger", finger_index_map["thumb"], "mount"),
                _mount_replacer(finger_index=finger_index_map["thumb"], new_mount=thumb_new_mount),
            )


def _resolve_active_modes(self_mode: Any) -> tuple[str, ...]:

    if self_mode is None:
        return (_MODE_GENERAL,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError(f"mount_perturb.self_mode must be str | dict[str, float] | None, got {type(self_mode).__name__}")

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"mount_perturb.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("mount_perturb.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"mount_perturb.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _validate_mode_fields(cfg: MountPerturbCfg) -> None:

    active_modes = set(cfg._active_modes)


    _normalize_radius(cfg.pos_radius, field_name="mount_perturb.pos_radius")
    _normalize_radius(cfg.rot_radius, field_name="mount_perturb.rot_radius")
    _normalize_radius(cfg.thumb_pos_radius, field_name="mount_perturb.thumb_pos_radius")
    _normalize_radius(cfg.thumb_rot_radius, field_name="mount_perturb.thumb_rot_radius")
    _normalize_range(cfg.mirror_yaw_range, field_name="mount_perturb.mirror_yaw_range")
    _normalize_range(cfg.mirror_x_range, field_name="mount_perturb.mirror_x_range")

    if _MODE_GENERAL in active_modes and cfg.pos_radius is None and cfg.rot_radius is None:
        raise ValueError("mount_perturb general mode requires at least one of pos_radius or rot_radius")

    if active_modes & {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH} and cfg.mirror_yaw_range is None:
        raise ValueError("mount_perturb index_ring_yaw_rot/index_ring requires mirror_yaw_range")
    if active_modes & {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH} and cfg.mirror_x_range is None:
        raise ValueError("mount_perturb index_ring_x_pos/index_ring requires mirror_x_range")

    if _MODE_GENERAL not in active_modes and (cfg.pos_radius is not None or cfg.rot_radius is not None):
        raise ValueError("mount_perturb pos_radius/rot_radius are only valid when general mode is active")
    if not (active_modes & _INDEX_RING_MODES) and (cfg.thumb_pos_radius is not None or cfg.thumb_rot_radius is not None):
        raise ValueError("mount_perturb thumb_pos_radius/thumb_rot_radius are only valid for index/ring modes")
    if not (active_modes & {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH}) and cfg.mirror_yaw_range is not None:
        raise ValueError("mount_perturb mirror_yaw_range is only valid for index_ring_yaw_rot/index_ring")
    if not (active_modes & {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}) and cfg.mirror_x_range is not None:
        raise ValueError("mount_perturb mirror_x_range is only valid for index_ring_x_pos/index_ring")

    if active_modes == {_MODE_IDENTITY}:
        any_payload = any(
            value is not None
            for value in (
                cfg.pos_radius,
                cfg.rot_radius,
                cfg.mirror_yaw_range,
                cfg.mirror_x_range,
                cfg.thumb_pos_radius,
                cfg.thumb_rot_radius,
            )
        )
        if any_payload:
            raise ValueError("mount_perturb identity mode must not carry perturbation payload fields")


def _draw_resolved_mode(cfg: MountPerturbCfg) -> str:

    if cfg.self_mode is None:
        return _MODE_GENERAL
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_GENERAL
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _normalize_sample_payload(sampled_params: dict[str, Any] | None, cfg: MountPerturbCfg) -> dict[str, Any]:

    if not sampled_params:
        return {"resolved_self_mode": _draw_resolved_mode(cfg)}
    if "sample" in sampled_params and isinstance(sampled_params["sample"], dict):
        return dict(sampled_params["sample"])
    if "resolved_self_mode" in sampled_params:
        return dict(sampled_params)
    raise ValueError("mount_perturb sampled_params must provide either {'sample': {...}} or a structured payload")


def _normalize_radius(value: float | Vector3 | None, *, field_name: str) -> Vector3 | None:

    if value is None:
        return None
    if isinstance(value, (int, float)):
        radius = float(value)
        if radius < 0.0:
            raise ValueError(f"{field_name} must be non-negative, got {value!r}")
        return (radius, radius, radius)
    radius_vec = _ensure_tuple(value, length=3, field_name=field_name)
    if any(component < 0.0 for component in radius_vec):
        raise ValueError(f"{field_name} components must be non-negative, got {radius_vec!r}")
    return radius_vec


def _normalize_range(value: Vector2 | None, *, field_name: str) -> Vector2 | None:

    if value is None:
        return None
    return _ensure_tuple(value, length=2, field_name=field_name)


def _as_vector3(value: Any) -> Vector3:

    return _ensure_tuple(value, length=3, field_name="mount_perturb.sample_vector")


def _hand_has_finger(hand: HandCfg, finger_name: str) -> bool:

    return any(finger.name == finger_name for finger in hand.fingers)


def _sample_scalar(
    value_range: Vector2 | None,
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> float:

    if value_range is None:
        return 0.0
    return float(_make_range_sampler(value_range, distrib=distrib, boundary_policy=boundary_policy)())


def _sample_ball_vector(
    radius: float | Vector3 | None,
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> Vector3:

    radius_vec = _normalize_radius(radius, field_name="mount_perturb.sample_radius")
    if radius_vec is None:
        return (0.0, 0.0, 0.0)

    distrib_type = distrib.get("type", "uniform") if isinstance(distrib, dict) else distrib
    distrib_type = str(distrib_type).lower()
    if distrib_type == "uniform":
        unit_vector = _sample_uniform_unit_ball()
    elif distrib_type == "normal":
        unit_vector = _sample_normalized_gaussian_ball(distrib=distrib, boundary_policy=boundary_policy)
    else:
        raise ValueError(f"unsupported mount_perturb vector distribution type: {distrib_type!r}")
    return (
        radius_vec[0] * unit_vector[0],
        radius_vec[1] * unit_vector[1],
        radius_vec[2] * unit_vector[2],
    )


def _sample_uniform_unit_ball() -> Vector3:

    direction = _sample_unit_sphere_direction()
    radius = random.random() ** (1.0 / 3.0)
    return (radius * direction[0], radius * direction[1], radius * direction[2])


def _sample_normalized_gaussian_ball(
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> Vector3:

    sigma_rule = float(distrib.get("sigma_rule", 3.0)) if isinstance(distrib, dict) else 3.0
    sigma = 1.0 / max(abs(sigma_rule), 1e-12)
    if isinstance(distrib, dict) and "sigma" in distrib:
        sigma = float(distrib["sigma"])

    policy = boundary_policy or "clip"
    for _ in range(32):
        sample = (
            random.gauss(0.0, sigma),
            random.gauss(0.0, sigma),
            random.gauss(0.0, sigma),
        )
        norm = _vector_norm(sample)
        if norm <= 1.0 + _MODE_TOLERANCE or policy == "none":
            return sample
        if policy in {"resample"}:
            continue
        return _project_to_unit_ball(sample)
    return _project_to_unit_ball(sample)


def _sample_unit_sphere_direction() -> Vector3:

    while True:
        candidate = (
            random.gauss(0.0, 1.0),
            random.gauss(0.0, 1.0),
            random.gauss(0.0, 1.0),
        )
        norm = _vector_norm(candidate)
        if norm > _MODE_TOLERANCE:
            return (candidate[0] / norm, candidate[1] / norm, candidate[2] / norm)


def _project_to_unit_ball(vector: Vector3) -> Vector3:

    norm = _vector_norm(vector)
    if norm <= 1.0 + _MODE_TOLERANCE:
        return vector
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _vector_norm(vector: Vector3) -> float:

    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _mount_replacer(*, finger_index: int, new_mount: PoseCfg):

    def apply_mount(hand: HandCfg, *, fi=finger_index, mount=new_mount) -> None:
        hand.fingers[fi].mount = mount

    return apply_mount


def _apply_local_mount_delta(
    mount: PoseCfg,
    *,
    delta_pos_local: Vector3,
    delta_rotvec_local: Vector3,
) -> PoseCfg:

    base_rotation = _rpy_rotation_matrix(mount.rpy)
    delta_pos_palm = _apply_rotation(base_rotation, delta_pos_local)
    delta_rotation = _rotvec_to_matrix(delta_rotvec_local)
    new_rotation = _matrix_multiply(base_rotation, delta_rotation)
    return PoseCfg(
        pos=(
            mount.pos[0] + delta_pos_palm[0],
            mount.pos[1] + delta_pos_palm[1],
            mount.pos[2] + delta_pos_palm[2],
        ),
        rpy=_matrix_to_rpy(new_rotation),
    )


def _apply_mount_delta_with_palm_shift(
    mount: PoseCfg,
    *,
    delta_pos_palm: Vector3,
    delta_rotvec_local: Vector3,
) -> PoseCfg:

    rotated_mount = _apply_local_mount_delta(
        mount,
        delta_pos_local=(0.0, 0.0, 0.0),
        delta_rotvec_local=delta_rotvec_local,
    )
    return PoseCfg(
        pos=(
            rotated_mount.pos[0] + delta_pos_palm[0],
            rotated_mount.pos[1] + delta_pos_palm[1],
            rotated_mount.pos[2] + delta_pos_palm[2],
        ),
        rpy=rotated_mount.rpy,
    )


def _rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def _rotvec_to_matrix(rotvec: Vector3) -> tuple[Vector3, Vector3, Vector3]:

    theta = _vector_norm(rotvec)
    if theta <= _MODE_TOLERANCE:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ux, uy, uz = rotvec[0] / theta, rotvec[1] / theta, rotvec[2] / theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_minus_c = 1.0 - c
    return (
        (
            c + ux * ux * one_minus_c,
            ux * uy * one_minus_c - uz * s,
            ux * uz * one_minus_c + uy * s,
        ),
        (
            uy * ux * one_minus_c + uz * s,
            c + uy * uy * one_minus_c,
            uy * uz * one_minus_c - ux * s,
        ),
        (
            uz * ux * one_minus_c - uy * s,
            uz * uy * one_minus_c + ux * s,
            c + uz * uz * one_minus_c,
        ),
    )


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:

    rhs_cols = (
        (rhs[0][0], rhs[1][0], rhs[2][0]),
        (rhs[0][1], rhs[1][1], rhs[2][1]),
        (rhs[0][2], rhs[1][2], rhs[2][2]),
    )
    rows: list[Vector3] = []
    for row in lhs:
        rows.append(
            (
                row[0] * rhs_cols[0][0] + row[1] * rhs_cols[0][1] + row[2] * rhs_cols[0][2],
                row[0] * rhs_cols[1][0] + row[1] * rhs_cols[1][1] + row[2] * rhs_cols[1][2],
                row[0] * rhs_cols[2][0] + row[1] * rhs_cols[2][1] + row[2] * rhs_cols[2][2],
            )
        )
    return (rows[0], rows[1], rows[2])


def _matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:

    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
        return (roll, pitch, yaw)


    roll = math.atan2(-matrix[0][1], matrix[1][1])
    yaw = 0.0
    return (roll, pitch, yaw)


__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
