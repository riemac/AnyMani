r"""基础几何关节构建器：把 primitive recipe 落为 `JointCfg`。

这一层只回答一个问题：

“给定一个 joint frame，以及一段 box / cylinder / sphere / tip 复合几何，
应该如何把 child link 的 collision / visual / inertial 写到 `JointCfg` 里？”

注意这里并不决定整根 finger 的运动学链组织。那部分逻辑属于 finger builder。
本层只负责“本 joint 之后这段 link 的局部几何语义”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

from ..asset_base import JointCfg
from ..asset_builders import JointBuilder, JointBuilderCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    InertialCfg,
    JointLimitCfg,
    PoseCfg,
    Vector3,
    Vector6,
    VisualGeometryCfg,
    _ensure_tuple,
)


_DEFAULT_DENSITY = 650.0
"""默认密度 $\\rho$ [kg/m^3]。

这是首轮实现的工程近似值，用于在用户没有显式给出质量时，根据几何体积
自动补一个正定、量级合理的惯量。
"""


def _pose_from_value(value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
    r"""把宽松位姿输入统一规范为 `PoseCfg`。"""

    return PoseCfg.from_value(value)


def _add_rpy(lhs: Vector3, rhs: Vector3) -> Vector3:
    r"""把两个小角度 `rpy` 增量直接按分量相加。

    当前 primitive builder 只需要处理很少数的轴对齐旋转，例如把 cylinder
    从 URDF 默认的 $z$ 轴朝向旋到 finger builder 需要的 $y$ 轴朝向。
    这里先不引入完整的 $SO(3)$ 组合工具，避免把首轮实现做得过重。
    """

    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _make_geometry_pose(
    *,
    offset: PoseCfg,
    default_pos: Vector3,
    default_rpy: Vector3 = (0.0, 0.0, 0.0),
    center_on_joint: bool = False,
) -> PoseCfg:
    r"""在当前 joint 约定下构造 primitive 的局部几何位姿。

    对 regular link，默认采用你草图中的“新约”：

    - 当偏移为 0 时，几何底面落在 joint frame 的 $x-z$ 平面
    - 几何沿 $+y$ 方向生长

    对 `CMC1` 这类特例，允许 `center_on_joint=True`，表示“偏移为 0 时，
    mesh frame 与 joint frame 完全重合”。
    """

    if center_on_joint:
        base = offset.pos
    else:
        base = default_pos
    return PoseCfg(pos=base, rpy=_add_rpy(default_rpy, offset.rpy))


def _box_inertia(size: Vector3, mass: float) -> dict[str, float]:
    sx, sy, sz = size
    return {
        "ixx": mass * (sy * sy + sz * sz) / 12.0,
        "iyy": mass * (sx * sx + sz * sz) / 12.0,
        "izz": mass * (sx * sx + sy * sy) / 12.0,
    }


def _cylinder_inertia(radius: float, length: float, mass: float) -> dict[str, float]:
    return {
        "ixx": mass * (3.0 * radius * radius + length * length) / 12.0,
        "iyy": mass * radius * radius / 2.0,
        "izz": mass * (3.0 * radius * radius + length * length) / 12.0,
    }


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    moment = 2.0 * mass * radius * radius / 5.0
    return {"ixx": moment, "iyy": moment, "izz": moment}


def _estimate_mass(*, volume: float, cfg_mass: float | None, density: float) -> float:
    r"""当 cfg 没有显式指定质量时，用体积和默认密度估质量。"""

    return float(cfg_mass) if cfg_mass is not None else max(volume * density, 1e-6)


@dataclass
class PrimJointBuilderCfg(JointBuilderCfg):
    r"""基础几何关节构建器配置。

    这个 cfg 故意暴露 joint-centric 字段，让 finger builder 可以把“本段 link
    的几何构造”委托给这一层，而不用在 finger 层反复重写 primitive 逻辑。
    """

    class_type: type["PrimJointBuilder"] | type["ComPrimJointBuilder"] | None = None
    """Associated runtime builder type."""

    name: str = "joint"
    """Joint name used in the emitted ``JointCfg``."""

    parent: str = "palm"
    """Parent link name in the emitted ``JointCfg``."""

    child: str | None = None
    """Optional explicit child link name."""

    mesh: dict[str, Any] = field(default_factory=dict)
    """Primitive mesh recipe.

    Supported ``type`` values:
    - ``box``: expects ``length``/``width``/``height`` or ``size``
    - ``cylinder``: expects ``length`` and ``radius``
    - ``sphere``: expects ``radius``
    - ``cs``: cylinder + sphere composite tip
    - ``bs``: box + sphere composite tip
    """

    joint_type: Literal["revolute", "fixed"] = "revolute"
    """Joint type emitted into the resulting ``JointCfg``."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Joint frame pose relative to the parent link."""

    axis: Vector3 = (0.0, 0.0, 1.0)
    """Joint axis used for ``revolute`` joints."""

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = (-math.pi, math.pi)
    """Optional joint limit information."""

    density: float = _DEFAULT_DENSITY
    """Fallback density used to derive inertial mass from primitive volume."""

    mass: float | None = None
    """Optional explicit mass override for the synthesized child link."""

    is_tip: bool = False
    """Whether the emitted joint/link should be marked as fingertip related."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata forwarded to the resulting ``JointCfg``."""

    def __post_init__(self):
        super().__post_init__()
        self.origin = _pose_from_value(self.origin)
        self.axis = _ensure_tuple(self.axis, length=3, field_name="prim_joint.axis")
        self.density = float(self.density)
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.mass is not None:
            self.mass = float(self.mass)
            if self.mass <= 0.0:
                raise ValueError("mass must be positive")
        if self.class_type in {None, JointBuilder}:
            mesh_kind = str(self.mesh.get("type", self.mesh.get("kind", "box"))).lower()
            self.class_type = ComPrimJointBuilder if mesh_kind in {"cs", "bs"} else PrimJointBuilder


class PrimJointBuilder(JointBuilder):
    r"""单一 primitive 关节构建器。"""

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据单一 primitive recipe 构建 `JointCfg`。

        Returns:
            JointCfg: joint-centric 的 child link 描述。
        """

        geom_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind", "box"))).lower()
        if geom_kind == "box":
            collisions, visuals, inertial = self._build_box()
        elif geom_kind == "cylinder":
            collisions, visuals, inertial = self._build_cylinder()
        elif geom_kind == "sphere":
            collisions, visuals, inertial = self._build_sphere()
        else:
            raise ValueError(f"Unsupported primitive joint mesh type: {geom_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=self.cfg.metadata.copy(),
        )

    def _build_box(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 box link。

        默认采用你 TODO 里的“新约”：

        $$
        x_m = d_x,\quad y_m = \frac{l}{2} + d_y,\quad z_m = d_z.
        $$
        """
        mesh = self.cfg.mesh
        if "size" in mesh:
            size = _ensure_tuple(mesh["size"], length=3, field_name="box.size")
        else:
            size = (
                float(mesh["width"]),
                float(mesh["length"]),
                float(mesh["height"]),
            )

        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], size[1] / 2.0 + offset.pos[1], offset.pos[2]),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(volume=size[0] * size[1] * size[2], cfg_mass=self.cfg.mass, density=self.cfg.density)
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_box_inertia(size, mass))
        collision = CollisionGeometryCfg(
            name=f"{self.cfg.name}_col",
            geometry={"type": "box", "size": size},
            origin=origin,
        )
        visual = VisualGeometryCfg(
            name=f"{self.cfg.name}_vis",
            geometry={"type": "box", "size": size},
            origin=origin,
        )
        return [collision], [visual], inertial

    def _build_cylinder(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 cylinder link。

        当前 finger 语义要求 cylinder 沿 $+y$ 生长，但 URDF primitive cylinder
        默认沿 $z$ 轴，因此这里额外施加一个 $(-\\pi/2, 0, 0)$ 的旋转，把几何
        从默认朝向旋到 finger builder 需要的朝向。
        """
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["length"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        # URDF cylinders are aligned with +z by default, so we rotate them into +y.
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),
            default_rpy=(-math.pi / 2.0, 0.0, 0.0),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(
            volume=math.pi * radius * radius * length,
            cfg_mass=self.cfg.mass,
            density=self.cfg.density,
        )
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_cylinder_inertia(radius, length, mass))
        geometry = {"type": "cylinder", "radius": radius, "length": length}
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)
        return [collision], [visual], inertial

    def _build_sphere(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 sphere link。"""
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], radius + offset.pos[1], offset.pos[2]),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=self.cfg.mass,
            density=self.cfg.density,
        )
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_sphere_inertia(radius, mass))
        geometry = {"type": "sphere", "radius": radius}
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)
        return [collision], [visual], inertial


class ComPrimJointBuilder(JointBuilder):
    r"""复合 primitive 构建器，当前主要服务于指尖。"""

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据复合指尖 recipe 构建 `JointCfg`。"""

        mesh_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind"))).lower()
        if mesh_kind == "cs":
            collisions, visuals, inertial = self._build_cylinder_sphere_tip()
        elif mesh_kind == "bs":
            collisions, visuals, inertial = self._build_box_sphere_tip()
        else:
            raise ValueError(f"Unsupported composite primitive mesh type: {mesh_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=self.cfg.metadata.copy(),
        )

    def _build_cylinder_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 `cylinder + sphere` 指尖。

        这里保持你原始 TODO 的语义：球心落在圆柱顶面中心，使球的最大截面与
        圆柱顶面重合，从而形成自然的 fingertip 过渡。
        """
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["height"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))

        cyl_origin = PoseCfg(
            pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),
            rpy=_add_rpy((-math.pi / 2.0, 0.0, 0.0), offset.rpy),
        )
        sph_origin = PoseCfg(pos=(offset.pos[0], length + offset.pos[1], offset.pos[2]), rpy=offset.rpy)

        cyl_mass = _estimate_mass(
            volume=math.pi * radius * radius * length,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,
            density=self.cfg.density,
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,
            density=self.cfg.density,
        )
        total_mass = cyl_mass + sph_mass
        com_y = (cyl_mass * cyl_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass

        equivalent_length = length + 2.0 * radius
        inertial = InertialCfg(
            mass=total_mass,
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),
            inertia=_cylinder_inertia(radius, equivalent_length, total_mass),
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,
            ),
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_cap_col",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_body_vis",
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial

    def _build_box_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 `box + sphere` 指尖。"""
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        height = float(mesh["height"])
        width = float(mesh["width"])
        depth = float(mesh.get("depth", width))
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))

        box_origin = PoseCfg(pos=(offset.pos[0], height / 2.0 + offset.pos[1], offset.pos[2]), rpy=offset.rpy)
        sph_origin = PoseCfg(pos=(offset.pos[0], height + offset.pos[1], offset.pos[2]), rpy=offset.rpy)

        box_mass = _estimate_mass(
            volume=width * height * depth,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,
            density=self.cfg.density,
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,
            density=self.cfg.density,
        )
        total_mass = box_mass + sph_mass
        com_y = (box_mass * box_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass

        inertial = InertialCfg(
            mass=total_mass,
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),
            inertia=_box_inertia((width, height + 2.0 * radius, depth), total_mass),
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,
            ),
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_cap_col",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_body_vis",
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial


__all__ = ["PrimJointBuilderCfg", "PrimJointBuilder", "ComPrimJointBuilder"]
