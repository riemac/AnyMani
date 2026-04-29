r"""基础几何关节构建器：把 primitive recipe 落为 `JointCfg`。

这一层只回答一个问题：

“给定一个 joint frame，以及一段 box / cylinder / sphere / tip 复合几何，
应该如何把 child link 的 collision / visual / inertial 写到 `JointCfg` 里？”

注意这里并不决定整根 finger 的运动学链组织。那部分逻辑属于 finger builder。
本层只负责“本 joint 之后这段 link 的局部几何语义”。

这里保留大量注释的原因，是 joint-level 的“旧约 / 新约”、偏移语义、指尖
复合几何规则，都会直接影响后续 finger-level 的串联算法是否可读、可核对。
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

    return PoseCfg.from_value(value)  # 兼容 tuple / dict / PoseCfg 等多种写法


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
        base = offset.pos  # CMC1 特例：零偏移时 mesh frame 与 joint frame 重合
    else:
        base = default_pos  # regular joint：零偏移时 primitive 从 joint frame 的 $x-z$ 平面向 $+y$ 长出
    return PoseCfg(pos=base, rpy=_add_rpy(default_rpy, offset.rpy))  # 平移采用新约，姿态采用增量叠加


def _box_inertia(size: Vector3, mass: float) -> dict[str, float]:
    r"""计算均质长方体的质心惯量对角项。"""
    sx, sy, sz = size  # 分别对应 $x/y/z$ 三个方向的边长
    return {
        "ixx": mass * (sy * sy + sz * sz) / 12.0,  # $I_{xx} = \frac{m}{12}(s_y^2+s_z^2)$
        "iyy": mass * (sx * sx + sz * sz) / 12.0,  # $I_{yy} = \frac{m}{12}(s_x^2+s_z^2)$
        "izz": mass * (sx * sx + sy * sy) / 12.0,  # $I_{zz} = \frac{m}{12}(s_x^2+s_y^2)$
    }


def _cylinder_inertia(radius: float, length: float, mass: float) -> dict[str, float]:
    r"""计算均质圆柱体的质心惯量对角项。

    这里的 `length` 指圆柱轴向长度。由于下游我们会把 URDF 默认沿 $z$ 的圆柱
    旋到 finger builder 所需的 $y$ 向，所以惯量依然按“轴向长度为 `length`”
    的标准公式计算。
    """
    return {
        "ixx": mass * (3.0 * radius * radius + length * length) / 12.0,  # 横向惯量
        "iyy": mass * radius * radius / 2.0,  # 轴向惯量
        "izz": mass * (3.0 * radius * radius + length * length) / 12.0,  # 横向惯量
    }


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    r"""计算均质球体的质心惯量对角项。"""
    moment = 2.0 * mass * radius * radius / 5.0  # 球体三个主轴惯量完全相同
    return {"ixx": moment, "iyy": moment, "izz": moment}  # $I_{xx}=I_{yy}=I_{zz}$


def _estimate_mass(*, volume: float, cfg_mass: float | None, density: float) -> float:
    r"""当 cfg 没有显式指定质量时，用体积和默认密度估质量。"""

    return float(cfg_mass) if cfg_mass is not None else max(volume * density, 1e-6)  # 加最小值，避免零质量导致惯量退化


@dataclass
class PrimJointBuilderCfg(JointBuilderCfg):
    r"""基础几何关节构建器配置。

    这个 cfg 故意暴露 joint-centric 字段，让 finger builder 可以把“本段 link
    的几何构造”委托给这一层，而不用在 finger 层反复重写 primitive 逻辑。
    """

    class_type: type["PrimJointBuilder"] | type["ComPrimJointBuilder"] | None = None  # 单体 primitive / 复合 tip 两类运行时 builder
    """关联的 primitive joint 运行时构建器。"""

    name: str = "joint"  # joint 逻辑名
    """输出到 `JointCfg` 中的 joint 名。"""

    parent: str = "palm"  # parent link 名
    """输出到 `JointCfg` 中的 parent link 名。"""

    child: str | None = None  # child link 名
    """可选显式 child link 名。"""

    mesh: dict[str, Any] = field(default_factory=dict)
    """primitive mesh 配方。

    当前支持的 ``type``：
    - ``box``：需要 ``length``/``width``/``height`` 或 ``size``
    - ``cylinder``：需要 ``length`` 和 ``radius``
    - ``sphere``：需要 ``radius``
    - ``cs``：cylinder + sphere 复合指尖
    - ``bs``：box + sphere 复合指尖
    """

    joint_type: Literal["revolute", "fixed"] = "revolute"  # 关节类型
    """输出到 `JointCfg` 中的 joint 类型。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)  # joint frame 位姿
    """joint frame 相对于 parent link frame 的位姿。"""

    axis: Vector3 = (0.0, 0.0, 1.0)  # 默认转轴
    """转轴方向，仅对 ``revolute`` joint 有意义。"""

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = (-math.pi, math.pi)  # 默认宽松限位
    """可选关节限位信息。"""

    density: float = _DEFAULT_DENSITY  # 默认密度
    """未显式给出质量时，用于由体积反推质量的默认密度。"""

    mass: float | None = None  # 显式质量覆盖
    """可选显式质量覆盖。若给出，则不再按体积估质量。"""

    is_tip: bool = False  # 是否标记为指尖相关
    """该 joint/link 是否应被标记为指尖相关。"""

    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元数据
    """附加元数据，会原样转发到结果 `JointCfg`。"""

    def __post_init__(self):
        super().__post_init__()
        self.origin = _pose_from_value(self.origin)  # joint frame 相对 parent link frame 的位姿
        self.axis = _ensure_tuple(self.axis, length=3, field_name="prim_joint.axis")  # 先保证是合法三元组
        self.density = float(self.density)  # 自动质量估计用的密度
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.mass is not None:
            self.mass = float(self.mass)  # 显式质量覆盖优先级高于密度估计
            if self.mass <= 0.0:
                raise ValueError("mass must be positive")
        if self.class_type in {None, JointBuilder}:
            mesh_kind = str(self.mesh.get("type", self.mesh.get("kind", "box"))).lower()  # 兼容历史 `kind=` 写法
            self.class_type = ComPrimJointBuilder if mesh_kind in {"cs", "bs"} else PrimJointBuilder  # 指尖复合几何单独走组合 builder


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

        geom_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind", "box"))).lower()  # primitive 类型分发键
        if geom_kind == "box":
            collisions, visuals, inertial = self._build_box()  # 长方体 link
        elif geom_kind == "cylinder":
            collisions, visuals, inertial = self._build_cylinder()  # 圆柱体 link
        elif geom_kind == "sphere":
            collisions, visuals, inertial = self._build_sphere()  # 球体 link
        else:
            raise ValueError(f"Unsupported primitive joint mesh type: {geom_kind}")

        return JointCfg(
            name=self.cfg.name,  # joint 名
            parent=self.cfg.parent,  # parent link 名
            child=self.cfg.child,  # child link 名
            joint_type=self.cfg.joint_type,  # revolute / fixed
            axis=self.cfg.axis,  # 转轴方向
            limit=self.cfg.limit,  # 关节限位
            origin=self.cfg.origin,  # joint frame 相对 parent 的位姿
            inertial=inertial,  # child link 的惯量描述
            collisions=collisions,  # child link collision 几何
            visuals=visuals,  # child link visual 几何
            is_tip=self.cfg.is_tip,  # 是否标记为指尖相关
            metadata=self.cfg.metadata.copy(),  # 附加 provenance / 调试信息
        )

    def _build_box(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 box link。

        默认采用你原始建模草案里的“新约”：

        $$
        x_m = d_x,\quad y_m = \frac{l}{2} + d_y,\quad z_m = d_z.
        $$
        """
        # --- 算法之一: Box（最常用，一般用作手指 link / palm 的构成）
        # 输入：偏移量 $d=(d_x,d_y,d_z)$，box 尺寸 $s=(s_x,s_y,s_z)$。
        # 输出：joint frame 下的 box mesh frame：
        # $x_m=d_x,\ y_m=s_y/2+d_y,\ z_m=d_z$。
        mesh = self.cfg.mesh
        if "size" in mesh:
            size = _ensure_tuple(mesh["size"], length=3, field_name="box.size")  # 直接接收完整 size
        else:
            size = (
                float(mesh["width"]),  # $x$ 向宽度
                float(mesh["length"]),  # $y$ 向长度
                float(mesh["height"]),  # $z$ 向高度
            )

        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))  # mesh 相对 joint frame 的增量位姿
        center_on_joint = bool(mesh.get("center_on_joint", False))  # CMC1 等特例可直接与 joint frame 重合
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], size[1] / 2.0 + offset.pos[1], offset.pos[2]),  # “新约”：底面贴 joint frame 的 $x-z$ 平面
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(volume=size[0] * size[1] * size[2], cfg_mass=self.cfg.mass, density=self.cfg.density)  # 体积估质量
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_box_inertia(size, mass))  # 惯量在质心 frame 下计算
        collision = CollisionGeometryCfg(
            name=f"{self.cfg.name}_col",  # collision 名
            geometry={"type": "box", "size": size},  # box primitive 参数
            origin=origin,  # box mesh frame
        )
        visual = VisualGeometryCfg(
            name=f"{self.cfg.name}_vis",  # visual 名
            geometry={"type": "box", "size": size},  # visual 与 collision 先保持一致
            origin=origin,  # visual frame
        )
        return [collision], [visual], inertial  # box link 的完整局部描述

    def _build_cylinder(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 cylinder link。

        当前 finger 语义要求 cylinder 沿 $+y$ 生长，但 URDF primitive cylinder
        默认沿 $z$ 轴，因此这里额外施加一个 $(-\\pi/2, 0, 0)$ 的旋转，把几何
        从默认朝向旋到 finger builder 需要的朝向。
        """
        # --- 算法之二: Cylinder（Box 的替代，一般也用作手指 link）
        # 输入：偏移量 $d=(d_x,d_y,d_z)$，cylinder 尺寸 $(r,h)$。
        # 输出：joint frame 下的 cylinder mesh frame：
        # $x_m=d_x,\ y_m=h/2+d_y,\ z_m=d_z$。
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])  # 圆柱半径
        length = float(mesh["length"])  # 圆柱轴向长度
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))  # 增量位姿
        center_on_joint = bool(mesh.get("center_on_joint", False))
        # URDF 默认 cylinder 轴向沿 +z，因此这里额外旋到 finger builder 需要的 +y 语义。
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
        geometry = {"type": "cylinder", "radius": radius, "length": length}  # cylinder primitive 参数
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)  # collision 几何
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)  # visual 几何
        return [collision], [visual], inertial  # cylinder link 的完整局部描述

    def _build_sphere(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 sphere link。"""
        # --- 算法之三: Sphere（当前很少单独用作 finger link，本轮保留接口）
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])  # 球半径
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))  # 增量位姿
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
        geometry = {"type": "sphere", "radius": radius}  # sphere primitive 参数
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)  # collision 几何
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)  # visual 几何
        return [collision], [visual], inertial  # sphere link 的完整局部描述


class ComPrimJointBuilder(JointBuilder):
    r"""复合 primitive 构建器，当前主要服务于指尖。"""

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据复合指尖 recipe 构建 `JointCfg`。"""

        mesh_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind"))).lower()  # 指尖复合类型分发键
        if mesh_kind == "cs":
            collisions, visuals, inertial = self._build_cylinder_sphere_tip()
        elif mesh_kind == "bs":
            collisions, visuals, inertial = self._build_box_sphere_tip()
        else:
            raise ValueError(f"Unsupported composite primitive mesh type: {mesh_kind}")

        return JointCfg(
            name=self.cfg.name,  # joint 名
            parent=self.cfg.parent,  # parent link 名
            child=self.cfg.child,  # child link 名
            joint_type=self.cfg.joint_type,  # fixed / revolute
            axis=self.cfg.axis,  # 这里对 fixed tip 实际无效，但保留 schema 完整性
            limit=self.cfg.limit,  # limit 同理
            origin=self.cfg.origin,  # tip joint frame 相对 parent 的位姿
            inertial=inertial,  # 复合 tip 的整体惯量
            collisions=collisions,  # 复合 collision 列表
            visuals=visuals,  # 复合 visual 列表
            is_tip=self.cfg.is_tip,  # 显式标记 tip
            metadata=self.cfg.metadata.copy(),  # 附加元数据
        )

    def _build_cylinder_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 `cylinder + sphere` 指尖。

        这里保持你原始算法说明的语义：球心落在圆柱顶面中心，使球的最大截面与
        圆柱顶面重合，从而形成自然的 fingertip 过渡。
        """
        # --- 算法之一 ---：cylinder + sphere 构造指尖的复合 mesh（最常用）
        # 输入：半径 $r$，高度 $h$，偏移 $d$。
        # 输出：
        # - 圆柱中心：$(d_x,\ d_y+h/2,\ d_z)$
        # - 球心：$(d_x,\ d_y+h,\ d_z)$
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])  # 半球帽半径
        length = float(mesh["height"])  # 圆柱体部分长度
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))  # 复合 tip 整体的增量位姿

        cyl_origin = PoseCfg(
            pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),  # 圆柱中心落在主体中点
            rpy=_add_rpy((-math.pi / 2.0, 0.0, 0.0), offset.rpy),  # 把 URDF 默认沿 $z$ 的圆柱旋到沿 $y$
        )
        sph_origin = PoseCfg(pos=(offset.pos[0], length + offset.pos[1], offset.pos[2]), rpy=offset.rpy)  # 球心落在圆柱顶面中心

        cyl_mass = _estimate_mass(
            volume=math.pi * radius * radius * length,  # 圆柱主体体积
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,  # 若显式给总质量，则按经验比例分配
            density=self.cfg.density,  # 否则按密度估质量
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,  # 球帽体积近似成整球
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,  # 给球帽分配剩余质量份额
            density=self.cfg.density,
        )
        total_mass = cyl_mass + sph_mass  # 复合 tip 总质量
        com_y = (cyl_mass * cyl_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass  # 沿 $y$ 方向求加权质心

        equivalent_length = length + 2.0 * radius  # 用“圆柱主体 + 球帽直径”近似整体轴向长度
        inertial = InertialCfg(
            mass=total_mass,  # 复合 tip 总质量
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),  # 整体质心
            inertia=_cylinder_inertia(radius, equivalent_length, total_mass),  # 首轮用等效圆柱近似整体惯量
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",  # 圆柱主体 collision
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,  # 圆柱主体位姿
            ),
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_cap_col",  # 半球帽 collision
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,  # 球帽位姿
            ),
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_body_vis",  # 圆柱主体 visual
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",  # 半球帽 visual
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial  # `cs` 指尖的完整局部描述

    def _build_box_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        r"""构建 `box + sphere` 指尖。"""
        # --- 算法之二 ---：box + sphere 构造指尖的复合 mesh
        # 输入：半径 $r$，高度 $h$，宽度 $w$，必要时加 depth。
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])  # 半球帽半径
        height = float(mesh["height"])  # box 主体高度（沿 $y$）
        width = float(mesh["width"])  # box 横向宽度（沿 $x$）
        depth = float(mesh.get("depth", width))  # 若不显式给 depth，则默认用 width 形成方截面
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))  # 复合 tip 的整体增量位姿

        box_origin = PoseCfg(pos=(offset.pos[0], height / 2.0 + offset.pos[1], offset.pos[2]), rpy=offset.rpy)  # box 主体中心
        sph_origin = PoseCfg(pos=(offset.pos[0], height + offset.pos[1], offset.pos[2]), rpy=offset.rpy)  # 球帽中心

        box_mass = _estimate_mass(
            volume=width * height * depth,  # box 主体体积
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,  # 若显式给总质量，则按经验比例切分
            density=self.cfg.density,
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,  # 球帽体积近似成整球
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,  # 球帽质量份额
            density=self.cfg.density,
        )
        total_mass = box_mass + sph_mass  # 复合 tip 总质量
        com_y = (box_mass * box_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass  # 沿 $y$ 方向求加权质心

        inertial = InertialCfg(
            mass=total_mass,  # 复合 tip 总质量
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),  # 整体质心
            inertia=_box_inertia((width, height + 2.0 * radius, depth), total_mass),  # 首轮用等效 box 近似整体惯量
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",  # box 主体 collision
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,  # box 主体位姿
            ),
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_cap_col",  # 球帽 collision
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,  # 球帽位姿
            ),
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_body_vis",  # box 主体 visual
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",  # 球帽 visual
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial  # `bs` 指尖的完整局部描述


__all__ = ["PrimJointBuilderCfg", "PrimJointBuilder", "ComPrimJointBuilder"]
