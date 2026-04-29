r"""URDF 导出器：把 HandCfg 序列化为标准 URDF XML 文件。

URDF（Unified Robot Description Format）是当前仿真管线（Isaac Lab / pybullet / MuJoCo）
导入机器人模型的通用格式。本模块负责把内存中的 `HandCfg` 按照 URDF 1.0 规范
写出到文件。

URDF 结构对应关系
-----------------

.. code-block:: xml

    <robot name="{hand.name}">
      <!-- palm link -->
      <link name="{palm.name}">
        <inertial>...</inertial>
        <visual>...</visual>
        <collision>...</collision>
      </link>

      <!-- 每个 finger 的每个 joint + child link -->
      <joint name="{joint.name}" type="{joint.joint_type}">
        <parent link="{joint.parent}"/>
        <child link="{joint.child}"/>
        <origin xyz="..." rpy="..."/>
        <axis xyz="..."/>
        <limit lower="..." upper="..." effort="..." velocity="..."/>
      </joint>
      <link name="{joint.child}">
        <inertial>...</inertial>
        <visual>...</visual>
        <collision>...</collision>
      </link>
      ...
    </robot>

设计说明
--------

### finger mount 的处理

`FingerCfg.mount` 描述 finger 挂载位姿（palm frame 下的局部变换）。
在 URDF 里，这通过一个 ``fixed`` joint 实现：

- parent = palm.name
- child = ``{finger.name}_mount_link``（虚拟 link，零质量）
- first real joint 的 parent 改为 ``{finger_name}_mount_link``

这样可以干净地把 mount 的位姿语义嵌入 URDF 拓扑，而不是把 mount 变换
"消解"进第一个 joint 的 origin 里。

另一种做法（简化版）是直接把 ``mount.pos + joint.origin.pos`` 叠加到
第一个 joint 的 origin，不引入虚拟 link。`cfg.use_mount_link` 控制选择哪种。

### 几何类型支持

当前 URDF 写入器支持以下 collision/visual 基元：

- ``box``：``<geometry><box size="wx wy wz"/></geometry>``
- ``cylinder``：``<geometry><cylinder radius="r" length="h"/></geometry>``
- ``sphere``：``<geometry><sphere radius="r"/></geometry>``
- ``mesh``：``<geometry><mesh filename="..." scale="..."/></geometry>``

### effort / velocity 默认值

URDF 要求 ``<limit>`` 有 effort 和 velocity，但 HandCfg 没有存储它们。
默认填入 ``cfg.default_effort`` 和 ``cfg.default_velocity``，可按 preset 覆盖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_core import CollisionGeometryCfg, InertialCfg, MaterialCfg, MeshGeometryCfg, PoseCfg, VisualGeometryCfg
from ._base import ExporterBase, ExportResult


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class UrdfWriterCfg(AssetCfgBase):
    r"""URDF 导出器配置。"""

    class_type: type["UrdfWriter"] | None = None
    """关联的运行时类。"""

    filename: str = "hand.urdf"
    """输出文件名；相对于传入的 output_dir。"""

    use_mount_link: bool = True
    """是否为每根 finger 的挂载位姿插入一个虚拟 fixed joint + mount link。
    开启（默认）：拓扑更清晰，mount 位姿与第一个真实 joint 解耦。
    关闭：把 mount 变换叠加进第一个 joint origin，不引入额外 link。"""

    include_inertial: bool = True
    """是否在 URDF link 里写入 ``<inertial>``；若 HandCfg 中 inertial 为
    None，则填入 ``cfg.default_inertial`` 的占位值。"""

    default_effort: float = 10.0
    """``<limit>`` required 字段 effort 的默认值（N·m）。"""

    default_velocity: float = 3.14
    """``<limit>`` required 字段 velocity 的默认值（rad/s）。"""

    mesh_package_prefix: str | None = None
    """mesh 文件路径的 ROS package 前缀（如 ``"package://my_robot"``）。
    为 ``None`` 时使用相对路径或绝对路径（取决于 mesh 原路径）。"""

    overwrite: bool = True
    """若目标文件已存在，是否覆盖。``False`` 时记入 skipped 并跳过。"""

    recolored_materials: dict[str, MaterialCfg] = field(default_factory=dict)
    """按 link 名覆盖 visual material 的映射。

    key 是最终写进 URDF 的 `<link name="...">`，例如：

    - `palm`
    - `index_mcp1`
    - `thumb_tip`

    value 是要注入到该 link **所有 `<visual>`** 上的材质。collision 不消费这个字段。
    """

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = UrdfWriter
        normalized_materials: dict[str, MaterialCfg] = {}
        for link_name, material in self.recolored_materials.items():
            if isinstance(material, MaterialCfg):
                material_cfg = material.copy()
            elif isinstance(material, Mapping):
                material_cfg = MaterialCfg(**material)
            else:
                raise TypeError(
                    f"recolored_materials[{link_name!r}] must be MaterialCfg or mapping, got {material!r}"
                )
            normalized_materials[str(link_name)] = material_cfg
        self.recolored_materials = normalized_materials


# ============================================================================
#  运行时壳
# ============================================================================


class UrdfWriter(ExporterBase):
    r"""URDF 写入器。

    把 `HandCfg` 渲染为符合 URDF 1.0 规范的 XML 文件。
    """

    cfg: UrdfWriterCfg

    def __init__(self, cfg: UrdfWriterCfg):
        self.cfg = cfg

    def export(self, target: HandCfg, output_dir: Path) -> ExportResult:  # type: ignore[override]
        r"""把 `HandCfg` 写出为 URDF 文件。

        Args:
            target (HandCfg): 待导出的整手配置。
            output_dir (Path): 产物落盘目录；不存在时自动创建。

        Returns:
            ExportResult: 含写入路径或错误信息的结果包。
        """

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        robot = _build_robot_elem(target, self.cfg)
        ET.indent(robot)
        tree = ET.ElementTree(robot)
        tree.write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path])

    def to_urdf_string(self, target: HandCfg) -> str:
        r"""把 `HandCfg` 渲染为 URDF XML 字符串（不落盘）。

        适合调试预览或传给仿真器的内存接口。

        Args:
            target (HandCfg): 待渲染的整手配置。

        Returns:
            str: 完整的 URDF XML 字符串。
        """

        robot = _build_robot_elem(target, self.cfg)
        ET.indent(robot)
        return ET.tostring(robot, encoding="unicode", xml_declaration=True)


# ============================================================================
#  内部构建辅助
# ============================================================================


def _build_robot_elem(target: HandCfg, cfg: UrdfWriterCfg) -> ET.Element:
    r"""Build the top-level ``robot`` XML element."""

    robot = ET.Element("robot", attrib={"name": target.name})
    robot.append(_build_link_elem(target.palm.name, target.palm.inertial, target.palm.collisions, target.palm.visuals, cfg))

    for finger in target.fingers:
        parent_name = target.palm.name
        joints = finger.joints
        if cfg.use_mount_link:
            mount_link_name = f"{finger.name}_mount_link"
            robot.append(_build_fixed_joint(f"{finger.name}_mount_joint", parent_name, mount_link_name, finger.mount))
            robot.append(_build_link_elem(mount_link_name, None, [], [], cfg))
            parent_name = mount_link_name
        else:
            joints = [_copy_joint_with_mount(finger, joint) if index == 0 else joint for index, joint in enumerate(finger.joints)]

        for joint in joints:
            robot.append(_build_joint_elem(joint, parent_name, cfg))
            robot.append(_build_link_elem(joint.child, joint.inertial, joint.collisions, joint.visuals, cfg))
            parent_name = joint.child
    return robot


def _copy_joint_with_mount(finger, joint):
    r"""Return a shallow joint copy with the finger mount folded into the first joint."""

    mount = finger.mount or PoseCfg()
    origin = PoseCfg(
        pos=(
            mount.pos[0] + joint.origin.pos[0],
            mount.pos[1] + joint.origin.pos[1],
            mount.pos[2] + joint.origin.pos[2],
        ),
        rpy=(
            mount.rpy[0] + joint.origin.rpy[0],
            mount.rpy[1] + joint.origin.rpy[1],
            mount.rpy[2] + joint.origin.rpy[2],
        ),
    )
    return joint.replace(origin=origin)


def _build_link_elem(
    name: str,
    inertial: InertialCfg | None,
    collisions: list[CollisionGeometryCfg],
    visuals: list[VisualGeometryCfg],
    cfg: UrdfWriterCfg,
) -> ET.Element:
    r"""构建 ``<link>`` XML 元素。

    Returns:
        xml.etree.ElementTree.Element: 构建好的 link 元素。
    """

    link = ET.Element("link", attrib={"name": name})
    if cfg.include_inertial:
        inertial_cfg = inertial or InertialCfg(
            mass=1e-6,
            origin=PoseCfg(),
            inertia={"ixx": 1e-9, "iyy": 1e-9, "izz": 1e-9},
        )
        inertial_elem = ET.SubElement(link, "inertial")
        ET.SubElement(inertial_elem, "origin", attrib=_pose_attrib(inertial_cfg.origin))
        ET.SubElement(inertial_elem, "mass", attrib={"value": _fmt_scalar(inertial_cfg.mass)})
        ET.SubElement(
            inertial_elem,
            "inertia",
            attrib={
                "ixx": _fmt_scalar(inertial_cfg.inertia.ixx),
                "ixy": _fmt_scalar(inertial_cfg.inertia.ixy),
                "ixz": _fmt_scalar(inertial_cfg.inertia.ixz),
                "iyy": _fmt_scalar(inertial_cfg.inertia.iyy),
                "iyz": _fmt_scalar(inertial_cfg.inertia.iyz),
                "izz": _fmt_scalar(inertial_cfg.inertia.izz),
            },
        )

    for visual in visuals:
        visual_elem = ET.SubElement(link, "visual")
        if visual.name:
            visual_elem.attrib["name"] = visual.name
        ET.SubElement(visual_elem, "origin", attrib=_pose_attrib(visual.origin))
        visual_elem.append(_build_geometry_elem(visual.geometry, cfg))
        material = cfg.recolored_materials.get(name) or visual.material
        if material is not None:
            visual_elem.append(_build_material_elem(material))

    for collision in collisions:
        collision_elem = ET.SubElement(link, "collision")
        if collision.name:
            collision_elem.attrib["name"] = collision.name
        ET.SubElement(collision_elem, "origin", attrib=_pose_attrib(collision.origin))
        collision_elem.append(_build_geometry_elem(collision.geometry, cfg))

    return link


def _build_joint_elem(joint, parent_override: str, cfg: UrdfWriterCfg) -> ET.Element:
    r"""构建 ``<joint>`` XML 元素。

    Returns:
        xml.etree.ElementTree.Element: 构建好的 joint 元素。
    """

    joint_elem = ET.Element("joint", attrib={"name": joint.name, "type": joint.joint_type})
    ET.SubElement(joint_elem, "parent", attrib={"link": parent_override})
    ET.SubElement(joint_elem, "child", attrib={"link": joint.child})
    ET.SubElement(joint_elem, "origin", attrib=_pose_attrib(joint.origin))

    if joint.joint_type != "fixed":
        ET.SubElement(joint_elem, "axis", attrib={"xyz": _fmt_triplet(joint.axis)})
        if joint.limit is not None:
            ET.SubElement(
                joint_elem,
                "limit",
                attrib={
                    "lower": _fmt_scalar(joint.limit.lower),
                    "upper": _fmt_scalar(joint.limit.upper),
                    "effort": _fmt_scalar(
                        joint.limit.effort if joint.limit.effort is not None else cfg.default_effort
                    ),
                    "velocity": _fmt_scalar(
                        joint.limit.velocity if joint.limit.velocity is not None else cfg.default_velocity
                    ),
                },
            )

    return joint_elem


def _build_fixed_joint(name: str, parent: str, child: str, origin: PoseCfg) -> ET.Element:
    r"""构建一个 fixed joint XML 元素（用于 mount link 插入）。

    Returns:
        xml.etree.ElementTree.Element: 构建好的 fixed joint 元素。
    """

    joint_elem = ET.Element("joint", attrib={"name": name, "type": "fixed"})
    ET.SubElement(joint_elem, "parent", attrib={"link": parent})
    ET.SubElement(joint_elem, "child", attrib={"link": child})
    ET.SubElement(joint_elem, "origin", attrib=_pose_attrib(origin))
    return joint_elem


def _build_geometry_elem(geom, cfg: UrdfWriterCfg) -> ET.Element:
    r"""根据 collision/visual 几何类型构建对应的 ``<geometry>`` XML 元素。

    Returns:
        xml.etree.ElementTree.Element: 构建好的 geometry 元素。
    """

    geometry_elem = ET.Element("geometry")
    kind = geom.kind
    if kind == "box":
        ET.SubElement(geometry_elem, "box", attrib={"size": _fmt_triplet(geom.size)})
    elif kind == "cylinder":
        ET.SubElement(
            geometry_elem,
            "cylinder",
            attrib={"radius": _fmt_scalar(geom.radius), "length": _fmt_scalar(geom.length)},
        )
    elif kind == "sphere":
        ET.SubElement(geometry_elem, "sphere", attrib={"radius": _fmt_scalar(geom.radius)})
    elif kind == "mesh":
        filename = geom.file_path
        if cfg.mesh_package_prefix and not filename.startswith(("package://", "/")):
            filename = f"{cfg.mesh_package_prefix.rstrip('/')}/{filename.lstrip('./')}"
        mesh_attrib = {"filename": filename}
        if isinstance(geom, MeshGeometryCfg) and geom.scale != (1.0, 1.0, 1.0):
            mesh_attrib["scale"] = _fmt_triplet(geom.scale)
        ET.SubElement(geometry_elem, "mesh", attrib=mesh_attrib)
    else:
        raise ValueError(f"Unsupported URDF geometry kind: {kind}")
    return geometry_elem


def _build_material_elem(material: MaterialCfg) -> ET.Element:
    r"""把 `MaterialCfg` lower 成 URDF `<material>` 元素。"""

    material_elem = ET.Element("material")
    if material.name:
        material_elem.attrib["name"] = material.name
    ET.SubElement(material_elem, "color", attrib={"rgba": _fmt_triplet(material.rgba)})
    return material_elem


def _pose_attrib(pose: PoseCfg) -> dict[str, str]:
    return {"xyz": _fmt_triplet(pose.pos), "rpy": _fmt_triplet(pose.rpy)}


def _fmt_triplet(values) -> str:
    return " ".join(_fmt_scalar(value) for value in values)


def _fmt_scalar(value: float) -> str:
    return f"{float(value):.9g}"


__all__ = ["UrdfWriterCfg", "UrdfWriter"]
