"""Builder-side runtime objects for hand-asset generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Literal, cast

from .asset_schema_core import (
    AssetCfgBase,
    BoxGeometryCfg,
    CollisionGeometryCfg,
    CylinderGeometryCfg,
    InertiaTensorCfg,
    Handedness,
    InertialCfg,
    PoseCfg,
    SphereGeometryCfg,
)
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

HandRule = Callable[[HandCfg], None]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> Mat3:
    r"""Compute the rotation matrix corresponding to URDF RPY angles."""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _transpose(matrix: Mat3) -> Mat3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(left: Mat3, right: Mat3) -> Mat3:
    return (
        (
            sum(left[0][k] * right[k][0] for k in range(3)),
            sum(left[0][k] * right[k][1] for k in range(3)),
            sum(left[0][k] * right[k][2] for k in range(3)),
        ),
        (
            sum(left[1][k] * right[k][0] for k in range(3)),
            sum(left[1][k] * right[k][1] for k in range(3)),
            sum(left[1][k] * right[k][2] for k in range(3)),
        ),
        (
            sum(left[2][k] * right[k][0] for k in range(3)),
            sum(left[2][k] * right[k][1] for k in range(3)),
            sum(left[2][k] * right[k][2] for k in range(3)),
        ),
    )


def _matrix_add(left: Mat3, right: Mat3) -> Mat3:
    return (
        (left[0][0] + right[0][0], left[0][1] + right[0][1], left[0][2] + right[0][2]),
        (left[1][0] + right[1][0], left[1][1] + right[1][1], left[1][2] + right[1][2]),
        (left[2][0] + right[2][0], left[2][1] + right[2][1], left[2][2] + right[2][2]),
    )


def _matrix_scale(matrix: Mat3, scale: float) -> Mat3:
    return (
        (scale * matrix[0][0], scale * matrix[0][1], scale * matrix[0][2]),
        (scale * matrix[1][0], scale * matrix[1][1], scale * matrix[1][2]),
        (scale * matrix[2][0], scale * matrix[2][1], scale * matrix[2][2]),
    )


def _outer(vec: tuple[float, float, float]) -> Mat3:
    return (
        (vec[0] * vec[0], vec[0] * vec[1], vec[0] * vec[2]),
        (vec[1] * vec[0], vec[1] * vec[1], vec[1] * vec[2]),
        (vec[2] * vec[0], vec[2] * vec[1], vec[2] * vec[2]),
    )


def _identity3() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _diag(ixx: float, iyy: float, izz: float) -> Mat3:
    return ((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz))


def _inertia_tensor_from_matrix(matrix: Mat3) -> InertiaTensorCfg:
    return InertiaTensorCfg(
        ixx=matrix[0][0],
        iyy=matrix[1][1],
        izz=matrix[2][2],
        ixy=matrix[0][1],
        ixz=matrix[0][2],
        iyz=matrix[1][2],
    )


def _primitive_mass_and_inertia_at_centroid(
    collision: CollisionGeometryCfg,
    density: float,
) -> tuple[float, Mat3]:
    r"""Compute primitive mass and centroid inertia in the primitive-local frame."""

    geometry = collision.geometry
    if isinstance(geometry, BoxGeometryCfg):
        inertial = InertialCfg.from_box(geometry.size, density=density, inertia_padding=0.0)
    elif isinstance(geometry, CylinderGeometryCfg):
        inertial = InertialCfg.from_cylinder(
            geometry.radius,
            geometry.length,
            density=density,
            principal_axis="z",
            inertia_padding=0.0,
        )
    elif isinstance(geometry, SphereGeometryCfg):
        inertial = InertialCfg.from_sphere(geometry.radius, density=density, inertia_padding=0.0)
    else:
        raise TypeError("aggregate_primitive_inertial only supports primitive collision geometry in v1")

    tensor = cast(InertiaTensorCfg, inertial.inertia)
    return inertial.mass, _diag(tensor.ixx, tensor.iyy, tensor.izz)


def aggregate_primitive_inertial(
    collisions: Sequence[CollisionGeometryCfg],
    *,
    density: float,
    inertia_padding: float = 1e-8,
) -> InertialCfg:
    r"""Aggregate multiple primitive collisions into one link-level inertial descriptor.

    This function implements the builder-layer counterpart of the plan:
    multiple primitive collision elements are collapsed into one URDF inertial
    element by computing a combined center of mass and inertia tensor.

    Args:
        collisions (Sequence[CollisionGeometryCfg]): Primitive collision elements.
        density (float): Shared density used for primitive mass approximation.
        inertia_padding (float): Diagonal padding for numerical robustness.

    Returns:
        InertialCfg: Aggregated inertial descriptor.

    Raises:
        ValueError: If no collision is provided.
        TypeError: If any collision uses non-primitive geometry.
    """

    if not collisions:
        raise ValueError("aggregate_primitive_inertial requires at least one collision element")
    if density <= 0.0:
        raise ValueError("density must be positive")

    primitive_terms: list[tuple[float, tuple[float, float, float], Mat3]] = []
    total_mass = 0.0
    weighted_pos = [0.0, 0.0, 0.0]
    for collision in collisions:
        mass_i, inertia_local = _primitive_mass_and_inertia_at_centroid(collision, density)
        origin = PoseCfg.from_value(collision.origin)
        rotation = _rotation_matrix_from_rpy(origin.rpy)
        inertia_rotated = _matmul(_matmul(rotation, inertia_local), _transpose(rotation))
        pos = origin.pos
        primitive_terms.append((mass_i, pos, inertia_rotated))
        total_mass += mass_i
        weighted_pos[0] += mass_i * pos[0]
        weighted_pos[1] += mass_i * pos[1]
        weighted_pos[2] += mass_i * pos[2]

    com = (weighted_pos[0] / total_mass, weighted_pos[1] / total_mass, weighted_pos[2] / total_mass)
    inertia_about_com = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    identity = _identity3()
    for mass_i, pos, inertia_rotated in primitive_terms:
        rel = (pos[0] - com[0], pos[1] - com[1], pos[2] - com[2])
        rel_sq = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]
        parallel_axis = _matrix_scale(
            _matrix_add(_matrix_scale(identity, rel_sq), _matrix_scale(_outer(rel), -1.0)),
            mass_i,
        )
        inertia_about_com = _matrix_add(inertia_about_com, _matrix_add(inertia_rotated, parallel_axis))

    return InertialCfg(
        mass=total_mass,
        origin=PoseCfg(pos=com),
        inertia=_inertia_tensor_from_matrix(inertia_about_com),
        inertia_padding=inertia_padding,
    )


@dataclass
class BuilderCfg(AssetCfgBase):
    r"""Base config for builder runtime objects."""

    class_type: type["Builder"] | None = None
    """Associated builder runtime class."""


class Builder:
    r"""Base runtime object for asset builders."""

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg

    def build(self) -> HandCfg:
        raise NotImplementedError


@dataclass
class HandBuilderCfg(BuilderCfg):
    r"""Top-level config for hand assembly in generator v1."""

    class_type: type["Builder"] | None = None
    """Associated hand-builder runtime class."""

    hand_name: str = "generated_hand"
    """Name used for the resulting `HandCfg`."""

    family: str = "generic"
    """Family tag for the resulting `HandCfg`."""

    handedness: Handedness = "unknown"
    """Handedness tag for the resulting `HandCfg`."""

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    """Palm schema input or palm mapping."""

    fingers: list[FingerCfg | Mapping[str, Any]] = field(default_factory=list)
    """Finger schema inputs or finger mappings."""

    auto_compute_missing_inertial: bool = False
    """Whether to infer missing inertial terms from primitive collision geometry."""

    default_density: float = 500.0
    """Default density used when inferring inertial terms from primitive collisions."""

    inertia_padding: float = 1e-8
    """Diagonal padding used by inferred inertial terms."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Reserved builder metadata."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandBuilder


class HandBuilder(Builder):
    r"""Top-level hand builder.

    v1 keeps the runtime hierarchy intentionally shallow:
    only `HandBuilder` is public, while joint/finger/palm handling stays as
    internal assembly steps so the project can iterate on the algorithm freely.
    """

    cfg: HandBuilderCfg

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)

    def _coerce_palm(self) -> PalmCfg:
        return self.cfg.palm if isinstance(self.cfg.palm, PalmCfg) else PalmCfg(**self.cfg.palm)

    def _coerce_fingers(self) -> list[FingerCfg]:
        return [finger if isinstance(finger, FingerCfg) else FingerCfg(**finger) for finger in self.cfg.fingers]

    def _fill_missing_inertial(self, hand: HandCfg) -> None:
        r"""Infer missing inertial terms from primitive collisions when requested."""

        palm = cast(PalmCfg, hand.palm)
        if palm.inertial is None and palm.collisions:
            palm.inertial = aggregate_primitive_inertial(
                palm.collisions,
                density=self.cfg.default_density,
                inertia_padding=self.cfg.inertia_padding,
            )

        for finger in hand.fingers:
            for joint in finger.joints:
                if joint.inertial is None and joint.collisions:
                    joint.inertial = aggregate_primitive_inertial(
                        joint.collisions,
                        density=self.cfg.default_density,
                        inertia_padding=self.cfg.inertia_padding,
                    )

    def build(self) -> HandCfg:
        palm = self._coerce_palm()
        fingers = self._coerce_fingers()
        hand = HandCfg(
            name=self.cfg.hand_name,
            palm=palm,
            fingers=fingers,
            family=self.cfg.family,
            handedness=self.cfg.handedness,
            metadata=self.cfg.metadata.copy(),
        )
        if self.cfg.auto_compute_missing_inertial:
            self._fill_missing_inertial(hand)
        return hand


__all__ = [
    "BuilderCfg",
    "Builder",
    "HandBuilderCfg",
    "HandBuilder",
    "HandRule",
    "aggregate_primitive_inertial",
]
