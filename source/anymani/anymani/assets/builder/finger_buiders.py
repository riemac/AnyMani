"Builds serial fingers in proximal-to-distal order from link lengths, local joint axes, and tip geometry. Lengths are in meters and angles in radians. The thumb distal link labels remain a review point."

from __future__ import annotations

# FIXME: Review whether the thumb's last two child-link labels should be PIP/DIP or thumb-specific names.

from dataclasses import dataclass, field
from typing import Any

from ..asset_base import FingerCfg
from ..asset_builders import FingerBuilder, FingerBuilderCfg
from ..asset_schema_core import PoseCfg, Vector2, Vector3, Vector6, _ensure_tuple, _normalize_axis
from ._utils import (
    _build_box_mesh,
    _build_cylinder_mesh,
    _mesh_cross_section,
    _mesh_length,
    _normalize_joint_limits,
    _normalize_joint_properties,
    _normalize_pose_list,
    _normalize_pose_value,
    _to_si,
)
from .joint_builders_custom import CustomTipBuilderCfg, apply_thumb_functional_tip_phase
from .joint_builders_primitive import PrimJointBuilderCfg





#




_NON_THUMB_CHILD_LINK_SUFFIXES: tuple[str, ...] = ("mcp1", "mcp2", "pip", "dip")
_THUMB_CHILD_LINK_SUFFIXES: tuple[str, ...] = ("cmc1", "cmc2", "mcp", "dip")


def _normalize_tip_dict(tip: dict[str, Any] | None) -> dict[str, Any]:

    tip = dict(tip or {"type": "cs", "radius": 0.012, "height": 0.01})
    tip_type = str(tip.get("type", tip.get("kind", "cs"))).lower()
    normalized: dict[str, Any] = {"type": tip_type}
    if tip_type == "cs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
    elif tip_type == "bs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
        normalized["width"] = _to_si(tip.get("width", tip.get("depth", 0.02)))
        normalized["depth"] = _to_si(tip.get("depth", tip.get("width", 0.02)))
    elif tip_type in {"mesh", "custom"}:
        normalized["type"] = "mesh"
        normalized["tip_type"] = str(tip.get("tip_type", tip.get("preset", "round"))).lower()
        if "path" in tip:
            normalized["path"] = str(tip["path"])
        if "file_path" in tip:
            normalized["path"] = str(tip["file_path"])
        if "scale" in tip:
            scale_value = tip["scale"]
            normalized["scale"] = (
                float(scale_value)
                if isinstance(scale_value, (int, float))
                else _ensure_tuple(scale_value, length=3, field_name="tip.scale")
            )
        if "unit_scale" in tip:
            normalized["unit_scale"] = float(tip["unit_scale"])
        if "anchor_point" in tip:
            normalized["anchor_point"] = _ensure_tuple(tip["anchor_point"], length=3, field_name="tip.anchor_point")
        if "base_rpy" in tip:
            normalized["base_rpy"] = _ensure_tuple(tip["base_rpy"], length=3, field_name="tip.base_rpy")
    else:
        raise ValueError(f"Only cs/bs/mesh tip recipes are supported in v1, got {tip_type!r}")
    return normalized







#






#



@dataclass
class RegularFingerBuilderCfg(FingerBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["RegularFingerBuilder"] | None = None
    "Associated runtime implementation for this configuration class."

    name: str = "finger"
    "Stable semantic identifier preserved in the output metadata."

    parent_link: str = "palm"
    "Source parent link for this joint or finger root."

    num_joints: int = 4
    "Number of active revolute joints in the source finger chain."

    mesh_shape: list[dict[str, Any]] = field(default_factory=list)
    "Mesh geometric classification used by mass and inertia closure."

    mesh_offsets: list[Any] = field(default_factory=list)
    "Per-link mesh anchor offsets in the owning joint frames."

    _mesh_offsets_6d: list[Vector6] = field(init=False, default_factory=list)
    "Source tip offsets ordered as translation and rotation components."

    tip: dict[str, Any] = field(default_factory=dict)
    "Terminal child-link configuration attached after the last active joint."

    tip_offset: Any = None
    "Terminal mesh-anchor pose in the tip-joint frame."

    _tip_offset_6d: Vector6 = field(init=False, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    "Tip offset ordered as translation and rotation components."

    axes: list[Vector3] = field(default_factory=list)
    "Revolute axes expressed in their joint-local frames."

    joint_limits: list[Any] = field(default_factory=list)
    "Lower and upper joint-coordinate bounds aligned to the full source chain, in radians."

    joint_properties: list[Any] = field(default_factory=list)
    "Optional effort, velocity, and friction values aligned to the source chain."

    def __post_init__(self):
        super().__post_init__()
        if self.num_joints < 1:
            raise ValueError("num_joints must be >= 1")

        self._mesh_offsets_6d = _normalize_pose_list(self.mesh_offsets, count=self.num_joints, field_name="mesh_offsets")
        self._tip_offset_6d = _normalize_pose_value(self.tip_offset, field_name="tip_offset")
        self.tip = _normalize_tip_dict(self.tip)
        if not self.axes:
            self.axes = [(0.0, 0.0, 1.0) for _ in range(self.num_joints)]
        if len(self.axes) != self.num_joints:
            raise ValueError(f"axes length must equal num_joints={self.num_joints}")
        self.axes = [_normalize_axis(_ensure_tuple(axis, length=3, field_name="axes")) for axis in self.axes]
        self.joint_limits = _normalize_joint_limits(self.joint_limits, count=self.num_joints)
        self.joint_properties = _normalize_joint_properties(
            self.joint_properties,
            count=self.num_joints,
        )
        if len(self.mesh_shape) != self.num_joints:
            raise ValueError(f"mesh_shape length must equal num_joints={self.num_joints}")
        self.class_type = RegularFingerBuilder

@dataclass
class AllegroFingerBuilderCfg(RegularFingerBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    width: float | None = None
    "Link cross-section dimension in meters."

    height: float | None = None
    "Link cross-section dimension in meters."

    radius: float | None = None
    "Primitive radius in meters."

    length: list[float] = field(default_factory=lambda: [0.018, 0.054, 0.038, 0.022])
    "Primary link span along the declared anatomical or joint-local axis, in meters."

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]
        width = _to_si(self.width or 0.027)
        height = _to_si(self.height or 0.020)
        radius = _to_si(self.radius) if self.radius is not None else None
        defaults = _normalize_pose_list([0.0, 0.0, -0.006, 0.0][: self.num_joints], count=self.num_joints, field_name="allegro_default_offsets")
        merged_offsets = self.mesh_offsets or defaults
        self.mesh_offsets = merged_offsets
        if not self.axes:
            self.axes = [(0.0, 1.0, 0.0)] + [(1.0, 0.0, 0.0)] * max(self.num_joints - 1, 0)
        if not self.tip:
            self.tip = {"type": "cs", "radius": 0.012, "height": 0.010}
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class LeapFingerBuilderCfg(RegularFingerBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    width: float | None = None
    "Link cross-section dimension in meters."

    height: float | None = None
    "Link cross-section dimension in meters."

    radius: float | None = None
    "Primitive radius in meters."

    length: list[float] = field(default_factory=lambda: [0.039, 0.015, 0.036, 0.020])
    "Primary link span along the declared anatomical or joint-local axis, in meters."
    fixed_part: float | None = None
    "Fixed geometry between the palm mount and the first active joint."

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]
        width = _to_si(self.width or 0.034)
        height = _to_si(self.height or 0.0205)
        radius = _to_si(self.radius) if self.radius is not None else None
        self.fixed_part = _to_si(self.fixed_part or 0.013)
        if not self.axes:
            defaults = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
            self.axes = defaults[: self.num_joints]
        if not self.tip:
            # User confirmed that the first testing path may use cylinder+sphere.
            self.tip = {"type": "cs", "radius": 0.012, "height": 0.010}
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class RegularThumbBuilderCfg(RegularFingerBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    cmc1_width: float | None = None
    "Count or linear dimension in the units declared by the associated schema."

    cmc1_height: float | None = None
    "Count or linear dimension in the units declared by the associated schema."

    width: float | None = None
    "Link cross-section dimension in meters."

    height: float | None = None
    "Link cross-section dimension in meters."

    lengths: list[float] = field(default_factory=lambda: [0.045, 0.017, 0.043, 0.040])
    "Count or linear dimension in the units declared by the associated schema."

    cmc1_offset: float | Vector2 | Vector3 = (0.009, 0.0145)
    "CMC1 mesh-anchor offset in its joint frame, with translation in meters and rotation in radians."

    non_cmc1_offset: list[Any] = field(default_factory=lambda: [-0.002, 0.0, -0.009])
    "Tip mesh-anchor offset for joints after thumb CMC1, in meters and radians."

    def __post_init__(self):
        self.num_joints = len(self.lengths)
        lengths = [_to_si(value) for value in self.lengths]
        cmc1_width = _to_si(self.cmc1_width or 0.035)
        cmc1_height = _to_si(self.cmc1_height or 0.034)
        width = _to_si(self.width or 0.019)
        height = _to_si(self.height or 0.027)

        cmc1_pose = _normalize_pose_value(self.cmc1_offset, field_name="cmc1_offset")
        other_offsets = _normalize_pose_list(self.non_cmc1_offset, count=self.num_joints - 1, field_name="non_cmc1_offset")
        self.mesh_offsets = [cmc1_pose] + other_offsets
        if not self.axes:

            #



            #



            self.axes = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ]
        if not self.tip:
            self.tip = {"type": "cs", "radius": 0.012, "height": 0.010}
        if not self.mesh_shape:
            self.mesh_shape = [
                _build_box_mesh(length=lengths[0], width=cmc1_width, height=cmc1_height, offset=cmc1_pose, center_on_joint=True),
                *[
                    _build_box_mesh(length=lengths[idx], width=width, height=height, offset=other_offsets[idx - 1])
                    for idx in range(1, self.num_joints)
                ],
            ]
        super().__post_init__()
        self.mesh_shape[0]["center_on_joint"] = True
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class SphericalFingerBuilderCfg(FingerBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."


class RegularFingerBuilder(FingerBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: RegularFingerBuilderCfg

    def __init__(self, cfg: RegularFingerBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> FingerCfg:
        "Builds the configured geometry component from typed dimensions and local frames."




        if isinstance(self.cfg, RegularThumbBuilderCfg):
            joints = self._build_thumb_chain()
        else:
            root_fixed_length = self.cfg.fixed_part if isinstance(self.cfg, LeapFingerBuilderCfg) else 0.0
            joints = self._build_serial_chain(
                root_fixed_length=root_fixed_length,
                emit_root_fixed_segment=isinstance(self.cfg, LeapFingerBuilderCfg) and root_fixed_length > 0.0,
            )
        return FingerCfg(
            name=self.cfg.name,
            parent_link=self.cfg.parent_link,
            mount=PoseCfg(),
            joints=joints,
            metadata={"builder": self.cfg.__class__.__name__},
        )

    def _build_serial_chain(self, *, root_fixed_length: float, emit_root_fixed_segment: bool = False) -> list[Any]:



        #




        #


        #


        #



        joints = []
        parent_link = self.cfg.parent_link
        if emit_root_fixed_segment:
            root_fixed_joint = self._build_root_fixed_segment(length=root_fixed_length)
            joints.append(root_fixed_joint)
            parent_link = root_fixed_joint.child
        previous_valid_length = root_fixed_length
        for index in range(self.cfg.num_joints):
            origin = PoseCfg(pos=(0.0, previous_valid_length, 0.0)) if index > 0 or root_fixed_length > 0.0 else PoseCfg()
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)
            joints.append(joint)
            parent_link = joint.child
            previous_valid_length = _mesh_length(self.cfg.mesh_shape[index]) + self.cfg._mesh_offsets_6d[index][1]

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=previous_valid_length))
        return joints

    def _build_root_fixed_segment(self, *, length: float):

        first_mesh = dict(self.cfg.mesh_shape[0])
        mesh_kind = str(first_mesh.get("type", first_mesh.get("kind", "box"))).lower()
        if mesh_kind == "box":
            root_mesh = _build_box_mesh(
                length=length,
                width=float(first_mesh["width"]),
                height=float(first_mesh["height"]),
                offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        elif mesh_kind == "cylinder":
            root_mesh = _build_cylinder_mesh(
                length=length,
                radius=float(first_mesh["radius"]),
                offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        else:
            raise ValueError(f"Unsupported first mesh kind for fixed root segment: {mesh_kind!r}")

        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_root_fixed",
            parent=self.cfg.parent_link,
            child=f"{self.cfg.name}_root_fixed_link",
            joint_type="fixed",
            origin=PoseCfg(),
            axis=(0.0, 0.0, 0.0),
            limit=None,
            mesh=root_mesh,
            metadata={
                "finger_name": self.cfg.name,
                "joint_index": "root_fixed",
                "fixed_root_segment": True,
            },
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()

    def _build_thumb_chain(self) -> list[Any]:





        #

        #








        #


        cfg = self.cfg
        assert isinstance(cfg, RegularThumbBuilderCfg)

        joints = [self._build_joint(index=0, parent_link=cfg.parent_link, origin=PoseCfg())]
        parent_link = joints[0].child

        cmc1_mesh_pose = cfg._mesh_offsets_6d[0]
        cmc1_length = _mesh_length(cfg.mesh_shape[0])
        cmc1_width, cmc1_height = _mesh_cross_section(cfg.mesh_shape[0])
        next_width, next_height = _mesh_cross_section(cfg.mesh_shape[1])
        cmc2_origin = PoseCfg(
            pos=(
                (cmc1_width - next_width) / 2.0,
                cmc1_mesh_pose[1] + cmc1_length / 2.0,
                cmc1_mesh_pose[2] - (cmc1_height - next_height) / 2.0,
            )
        )
        joint_1 = self._build_joint(index=1, parent_link=parent_link, origin=cmc2_origin)
        joints.append(joint_1)
        parent_link = joint_1.child

        next_joint_y = _mesh_length(cfg.mesh_shape[1]) + cfg._mesh_offsets_6d[1][1]
        for index in range(2, cfg.num_joints):
            origin = PoseCfg(pos=(0.0, next_joint_y, 0.0))
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)
            joints.append(joint)
            parent_link = joint.child
            next_joint_y = _mesh_length(cfg.mesh_shape[index]) + cfg._mesh_offsets_6d[index][1]

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=next_joint_y))
        return joints

    def _build_joint(self, *, index: int, parent_link: str, origin: PoseCfg):
        mesh = dict(self.cfg.mesh_shape[index])
        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_j{index}",
            parent=parent_link,
            child=self._revolute_child_link_name(index=index),
            joint_type="revolute",
            origin=origin,
            axis=self.cfg.axes[index],
            limit=self.cfg.joint_limits[index],
            joint_properties=self.cfg.joint_properties[index],
            mesh=mesh,
            metadata={
                "finger_name": self.cfg.name,
                "joint_index": index,
                "allow_zero_origin": index == 0 and origin.pos == (0.0, 0.0, 0.0),
            },
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()

    def _build_tip_joint(self, *, parent_link: str, tip_origin_y: float):
        tip_recipe = dict(self.cfg.tip)
        tip_recipe["offset"] = self.cfg._tip_offset_6d
        is_thumb = isinstance(self.cfg, RegularThumbBuilderCfg)
        common_kwargs = {
            "name": f"{self.cfg.name}_tip",
            "parent": parent_link,
            "child": self._tip_child_link_name(),
            "joint_type": "fixed",
            "origin": PoseCfg(pos=(0.0, tip_origin_y, 0.0)),
            "axis": (0.0, 0.0, 0.0),
            "limit": None,
            "is_tip": True,
            "metadata": {"finger_name": self.cfg.name, "joint_index": "tip"},
        }

        if tip_recipe["type"] == "mesh":




            mesh_offset = (
                apply_thumb_functional_tip_phase(tip_recipe["offset"])
                if is_thumb
                else tip_recipe["offset"]
            )
            builder_cfg = CustomTipBuilderCfg(
                tip_type=str(tip_recipe.get("tip_type", "round")),
                mesh_path=tip_recipe.get("path"),
                mesh_offset=mesh_offset,
                scale=tip_recipe.get("scale", 1.0),
                unit_scale=tip_recipe.get("unit_scale"),
                anchor_point=tip_recipe.get("anchor_point"),
                base_rpy=tip_recipe.get("base_rpy"),
                **common_kwargs,
            )
        else:
            builder_cfg = PrimJointBuilderCfg(
                mesh=tip_recipe,
                **common_kwargs,
            )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()

    def _revolute_child_link_name(self, *, index: int) -> str:

        semantic_suffixes = (
            _THUMB_CHILD_LINK_SUFFIXES
            if isinstance(self.cfg, RegularThumbBuilderCfg)
            else _NON_THUMB_CHILD_LINK_SUFFIXES
        )
        if not 0 <= index < len(semantic_suffixes):
            raise ValueError(
                f"RegularFingerBuilder currently only defines semantic child-link names for "
                f"{len(semantic_suffixes)} revolute joints, got index={index} for finger {self.cfg.name!r}"
            )
        return f"{self.cfg.name}_{semantic_suffixes[index]}"

    def _tip_child_link_name(self) -> str:

        return f"{self.cfg.name}_tip"


__all__ = [
    "RegularFingerBuilderCfg",
    "AllegroFingerBuilderCfg",
    "LeapFingerBuilderCfg",
    "RegularThumbBuilderCfg",
    "SphericalFingerBuilderCfg",
    "RegularFingerBuilder",
]
