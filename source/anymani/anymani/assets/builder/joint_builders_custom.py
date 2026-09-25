"Builds custom tip meshes by mapping a mesh-local base anchor to the tip-joint frame. Mesh unit conversion is applied before user scale."

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..asset_base import JointCfg
from ..asset_builders import JointBuilder, JointBuilderCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    JointLimitCfg,
    PoseCfg,
    Vector3,
    VisualGeometryCfg,
    _ensure_tuple,
)
from .joint_builders_primitive import _add_rpy


_CUSTOM_TIP_DIR = Path(__file__).resolve().parents[1] / "custom" / "tips"
"Directory containing the default custom fingertip meshes."


_DEFAULT_BASE_RPY = (0.0, -math.pi / 2.0, 0.0)
"Canonical base rotation for custom tip meshes, in radians."


THUMB_FUNCTIONAL_TIP_PHASE_RPY = (0.0, -math.pi / 2.0, 0.0)
"Thumb custom-tip functional rotation in radians about the configured local axes."


_CUSTOM_TIP_PRESETS: dict[str, dict[str, object]] = {
    "leap_cube": {
        "file_name": "finger_tip_soft.stl",
        "anchor_point": (9.48570692492, 0.0, -16.4999999586),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "round": {
        "file_name": "round_finger_tip_soft.stl",
        "anchor_point": (9.50986387389, 0.0, -16.4913187022),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "wedge": {
        "file_name": "wedge_finger_tip_soft.stl",
        "anchor_point": (9.5, 0.0, -16.5),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "thinner": {
        "file_name": "thinner_finger_tip_soft.stl",
        "anchor_point": (9.5, 0.0, -16.5),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
}
"Mesh filename, anchor point, unit scale, and canonical orientation for each tip preset."


def _pose_from_value(value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:

    return PoseCfg.from_value(value)


def apply_thumb_functional_tip_phase(offset: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
    'Applies thumb functional tip phase.'

    pose = _pose_from_value(offset)
    return PoseCfg(pos=pose.pos, rpy=_add_rpy(pose.rpy, THUMB_FUNCTIONAL_TIP_PHASE_RPY))


def _scale_to_vector(value: float | Sequence[float]) -> Vector3:

    if isinstance(value, (int, float)):
        scale = float(value)
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {value}")
        return (scale, scale, scale)
    scale = _ensure_tuple(value, length=3, field_name="custom_tip.scale")
    if any(component <= 0.0 for component in scale):
        raise ValueError(f"custom tip scale must be positive, got {scale}")
    return scale


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


def _resolve_tip_preset(tip_type: str) -> dict[str, object]:

    try:
        return dict(_CUSTOM_TIP_PRESETS[tip_type])
    except KeyError as exc:
        raise KeyError(f"Unknown custom tip preset: {tip_type!r}") from exc


@dataclass
class CustomJointBuilderCfg(JointBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["CustomJointBuilder"] | None = None
    "Associated runtime implementation for this configuration class."

    name: str = "joint"
    "Stable semantic identifier preserved in the output metadata."

    parent: str = "palm"
    "Parent link or owner identifier in the source topology."

    child: str | None = None
    "Child link name in the source kinematic tree."

    joint_type: str = "fixed"
    "Whether the link connection is fixed or actuated by a revolute coordinate."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    axis: Vector3 = (0.0, 0.0, 0.0)
    "Unit direction expressed in the owning local frame."

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = None
    "Lower and upper joint-coordinate bounds in radians."

    is_tip: bool = True
    "Whether this generation or validation condition is active."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Tip geometry provenance and contact metadata for the exported sidecar."

    is_customized: bool = True
    "Whether this generation or validation condition is active."

    def __post_init__(self):
        super().__post_init__()
        self.origin = _pose_from_value(self.origin)
        self.axis = _ensure_tuple(self.axis, length=3, field_name="custom_joint.axis")
        if self.class_type in {None, JointBuilder}:
            self.class_type = CustomJointBuilder


@dataclass
class CustomTipBuilderCfg(CustomJointBuilderCfg):
    """Defines a custom fingertip mesh, its local anchor, and target pose.

    anchor_point is expressed in mesh-local coordinates before unit conversion. unit_scale converts the source mesh to meters; scale is dimensionless; base_rpy is in radians. The anchor is placed at mesh_offset in the tip-joint frame.
    """

    tip_type: str = "round"
    "Terminal tip recipe selected for collision, visual, and contact geometry."

    mesh_path: str | Path | None = None
    "Source mesh path; geometry bytes are preserved during export."

    mesh_offset: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Target pose of the semantic mesh anchor in the joint frame; it is not the mesh-origin pose."

    scale: float | Sequence[float] = 1.0
    "Dimensionless scale applied after source mesh unit conversion."

    unit_scale: float | None = None
    "Conversion from source mesh units to meters."

    anchor_point: Vector3 | Sequence[float] | None = None
    "Semantic point in the owning mesh-local frame, before mesh unit scaling."

    base_rpy: Vector3 | Sequence[float] | None = None
    "Canonical custom-tip rotation in radians."

    _mesh_scale_xyz: Vector3 = field(init=False, default=(1.0, 1.0, 1.0))
    "Final dimensionless scale written to the URDF mesh element."

    def __post_init__(self):
        super().__post_init__()

        preset = _resolve_tip_preset(str(self.tip_type).lower())
        self.tip_type = str(self.tip_type).lower()
        self.mesh_offset = _pose_from_value(self.mesh_offset)
        user_scale = _scale_to_vector(self.scale)

        default_mesh_path = _CUSTOM_TIP_DIR / str(preset["file_name"])
        self.mesh_path = Path(self.mesh_path) if self.mesh_path is not None else default_mesh_path
        self.unit_scale = float(self.unit_scale if self.unit_scale is not None else preset["unit_scale"])
        if self.unit_scale <= 0.0:
            raise ValueError("unit_scale must be positive")

        self.anchor_point = _ensure_tuple(
            self.anchor_point if self.anchor_point is not None else preset["anchor_point"],
            length=3,
            field_name="custom_tip.anchor_point",
        )
        self.base_rpy = _ensure_tuple(
            self.base_rpy if self.base_rpy is not None else preset["base_rpy"],
            length=3,
            field_name="custom_tip.base_rpy",
        )


        self._mesh_scale_xyz = tuple(self.unit_scale * component for component in user_scale)


class CustomJointBuilder(JointBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: CustomJointBuilderCfg

    def __init__(self, cfg: CustomJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        if not isinstance(self.cfg, CustomTipBuilderCfg):
            raise NotImplementedError("CustomJointBuilder v1 currently only supports CustomTipBuilderCfg.")

        mesh_origin = self._build_mesh_origin()
        geometry = {"type": "mesh", "file_path": str(self.cfg.mesh_path), "scale": self.cfg._mesh_scale_xyz}
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_mesh_col",
                geometry=geometry,
                origin=mesh_origin,
            )
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_mesh_vis",
                geometry=geometry,
                origin=mesh_origin,
            )
        ]

        metadata = {
            **self.cfg.metadata,
            "custom_tip_type": self.cfg.tip_type,
            "mesh_path": str(self.cfg.mesh_path),
            "anchor_point": self.cfg.anchor_point,
            "mesh_scale": self.cfg._mesh_scale_xyz,
            "mesh_origin_rpy": mesh_origin.rpy,
        }
        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=None,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=metadata,
        )

    def _build_mesh_origin(self) -> PoseCfg:

        assert isinstance(self.cfg, CustomTipBuilderCfg)
        total_rpy = _add_rpy(self.cfg.base_rpy, self.cfg.mesh_offset.rpy)
        scaled_anchor = (
            self.cfg.anchor_point[0] * self.cfg._mesh_scale_xyz[0],
            self.cfg.anchor_point[1] * self.cfg._mesh_scale_xyz[1],
            self.cfg.anchor_point[2] * self.cfg._mesh_scale_xyz[2],
        )
        rotated_anchor = _apply_rotation(_rpy_rotation_matrix(total_rpy), scaled_anchor)
        return PoseCfg(
            pos=(
                self.cfg.mesh_offset.pos[0] - rotated_anchor[0],
                self.cfg.mesh_offset.pos[1] - rotated_anchor[1],
                self.cfg.mesh_offset.pos[2] - rotated_anchor[2],
            ),
            rpy=total_rpy,
        )


__all__ = [
    "THUMB_FUNCTIONAL_TIP_PHASE_RPY",
    "CustomJointBuilderCfg",
    "CustomTipBuilderCfg",
    "CustomJointBuilder",
    "apply_thumb_functional_tip_phase",
]
