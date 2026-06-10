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
        <joint_properties friction="..."/>
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
当前 AnyMani 已明确收敛到**官方 Allegro / LEAP URDF 同款语义**：

- 不再额外插入 ``*_mount_link`` 这类虚拟 link；
- 不再额外插入 ``*_mount_joint`` 这类 fixed joint；
- finger 相对 palm 的默认挂载位姿，统一写进该 finger 链第一个 joint 的
  ``origin``。

也就是说，导出时执行的是：

$$
{}^{palm}\mathbf{T}_{j_0}
=
{}^{palm}\mathbf{T}_{mount}
\cdot
{}^{mount}\mathbf{T}_{j_0}
$$

这里当前仍采用项目既有的近似实现：位置分量直接相加，RPY 角按分量叠加。
这不是最一般的 SE(3) 严格复合，但与本项目当前 mount preset 和 finger root
建模约定一致；更重要的是，它与用户要求对齐的官方 URDF 表达形式一致。

### 几何类型支持

当前 URDF 写入器支持以下 collision/visual 基元：

- ``box``：``<geometry><box size="wx wy wz"/></geometry>``
- ``cylinder``：``<geometry><cylinder radius="r" length="h"/></geometry>``
- ``sphere``：``<geometry><sphere radius="r"/></geometry>``
- ``mesh``：``<geometry><mesh filename="..." scale="..."/></geometry>``

### effort / velocity 默认值

URDF 要求 ``<limit>`` 有 effort 和 velocity，但 HandCfg 没有存储它们。
默认填入 ``cfg.default_effort`` 和 ``cfg.default_velocity``，可按 preset 覆盖。

### joint_properties

LEAP 官方 URDF 使用 ``<joint_properties friction=\"...\"/>`` 表达关节摩擦。
AnyMani v1 为了保持来源一致，若 `JointCfg.joint_properties.friction` 存在，
就按同一标签写出；不额外写 ``<dynamics>``，避免 importer 重复解释 friction。
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    EllipticCylinderGeometryCfg,
    InertialCfg,
    MaterialCfg,
    MeshGeometryCfg,
    PoseCfg,
    VisualGeometryCfg,
)
from ..procedural_meshes import (
    is_procedural_cs_tip_uri,
    materialize_procedural_cs_tip_mesh,
    parse_procedural_cs_tip_uri,
)
from ._base import ExporterBase, ExportResult

# ============================================================================
#  配置类
# ============================================================================


@dataclass
class UrdfWriterCfg(AssetCfgBase):
    r"""URDF 导出器配置。"""

    class_type: type[UrdfWriter] | None = None
    """关联的运行时类。"""

    filename: str = "hand.urdf"
    """输出文件名；相对于传入的 output_dir。"""

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

    canonical_mesh_dirname: str = "meshes"
    r"""自动生成 canonical mesh 时，相对样本目录的子目录名。

    当前主要服务于 `elliptic_cylinder` 这类 URDF 原生 primitive 无法直接表达的
    几何：内部仍保留解析 primitive，导出时再 lower 成当前样本目录下的一份
    canonical cylinder mesh，加上三轴 `scale`。
    """

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

    def export(
        self,
        target: HandCfg,
        output_dir: Path,
        *,
        mesh_root_dir: Path | None = None,
    ) -> ExportResult:  # type: ignore[override]
        r"""把 `HandCfg` 写出为 URDF 文件。

        Args:
            target (HandCfg): 待导出的整手配置。
            output_dir (Path): 产物落盘目录；不存在时自动创建。
            mesh_root_dir (Path | None): 真实 mesh 落盘根目录。为 `None` 时退化为
                当前 `output_dir` 下的 `canonical_mesh_dirname/`。

        Returns:
            ExportResult: 含写入路径或错误信息的结果包。
        """

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_state = _MeshExportState(
            output_dir=output_dir,
            mesh_dirname=self.cfg.canonical_mesh_dirname,
            mesh_root_dir=mesh_root_dir,
        )
        robot = _build_robot_elem(target, self.cfg, mesh_state=mesh_state)
        ET.indent(robot)
        tree = ET.ElementTree(robot)
        tree.write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path, *mesh_state.written])

    def to_urdf_string(self, target: HandCfg) -> str:
        r"""把 `HandCfg` 渲染为 URDF XML 字符串（不落盘）。

        适合调试预览或传给仿真器的内存接口。

        Args:
            target (HandCfg): 待渲染的整手配置。

        Returns:
            str: 完整的 URDF XML 字符串。
        """

        mesh_state = _MeshExportState(
            output_dir=Path("."),
            mesh_dirname=self.cfg.canonical_mesh_dirname,
            write_enabled=False,
        )  # 预览字符串路径里只保留相对文件名，不真的向当前工作目录写 mesh
        robot = _build_robot_elem(target, self.cfg, mesh_state=mesh_state)
        ET.indent(robot)
        return ET.tostring(robot, encoding="unicode", xml_declaration=True)


# ============================================================================
#  内部构建辅助
# ============================================================================


@dataclass
class _MeshExportState:
    r"""一次 URDF 导出调用内部共享的自动 mesh 生成状态。

    之所以单独保留这层状态，而不是在 `_build_geometry_elem()` 里每次直接写文件，
    是因为同一个 `HandCfg` 里可能有多处 `elliptic_cylinder`。我们希望：

    1. 同一份 canonical cylinder mesh 只生成一次；
    2. 导出结果能把附带写出的 mesh 文件路径也收进 `ExportResult.written`；
    3. `to_urdf_string()` 路径里仍能构造一致的相对文件名，而不真的写文件。
    """

    output_dir: Path
    mesh_dirname: str
    write_enabled: bool = True
    written: list[Path] = field(default_factory=list)
    mesh_root_dir: Path | None = None
    _unit_cylinder_relpath: str | None = None
    _materialized_mesh_relpaths: dict[Path, str] = field(default_factory=dict)
    _materialized_mesh_names: dict[str, Path] = field(default_factory=dict)


def _build_robot_elem(target: HandCfg, cfg: UrdfWriterCfg, *, mesh_state: _MeshExportState) -> ET.Element:
    r"""构建顶层 ``robot`` XML 元素。

    # NOTE:
    这里故意不再生成任何 `*_mount_link` / `*_mount_joint` 辅助拓扑。
    对每根 finger，我们都把 hand-level `mount` 直接折叠进该 finger 链的第一个
    joint `origin`，从而与官方 Allegro / LEAP URDF 的挂载语义保持一致。
    """

    robot = ET.Element("robot", attrib={"name": target.name})
    robot.append(
        _build_link_elem(
            target.palm.name,
            target.palm.inertial,
            target.palm.collisions,
            target.palm.visuals,
            cfg,
            mesh_state=mesh_state,
        )
    )

    for finger in target.fingers:
        parent_name = target.palm.name  # finger 根的真实 parent 仍然是 palm；不再经由虚拟 mount link 中转
        joints = [
            _copy_joint_with_mount(finger, joint) if index == 0 else joint
            for index, joint in enumerate(finger.joints)
        ]  # 只把挂载位姿折叠进该 finger 链第一个 joint；后续 joint 保持原局部语义不变

        for joint in joints:
            robot.append(_build_joint_elem(joint, parent_name, cfg))
            robot.append(
                _build_link_elem(
                    joint.child,
                    joint.inertial,
                    joint.collisions,
                    joint.visuals,
                    cfg,
                    mesh_state=mesh_state,
                )
            )
            parent_name = joint.child
    return robot


def _copy_joint_with_mount(finger, joint):
    r"""返回一份把 hand-level mount 折叠进 joint origin 的浅拷贝。

    这里服务的不是“任意 joint”的一般变换，而是 finger 链**第一个** joint 的
    导出语义。对 Allegro 非拇指，这通常是 `index_j0` / `middle_j0` 一类根关节；
    对 LEAP non-thumb，则可能是那段真实存在的 `root_fixed` 根部段。

    # NOTE:
    项目当前的 pose 组合仍沿用既有近似：

    - 平移：逐分量相加；
    - 姿态：RPY 逐分量相加。

    这与 builder / preset 侧当前的局部建模约定一致，也正是用户要求恢复的
    “first joint origin 表达挂载位姿”语义。
    """

    mount = finger.mount or PoseCfg()  # finger 若未显式给 mount，则退化为零位姿，不改变原 joint origin
    origin = PoseCfg(
        pos=(
            mount.pos[0] + joint.origin.pos[0],  # ${}^{palm}x_{j_0} = x_{mount} + x_{local}$
            mount.pos[1] + joint.origin.pos[1],  # ${}^{palm}y_{j_0} = y_{mount} + y_{local}$
            mount.pos[2] + joint.origin.pos[2],  # ${}^{palm}z_{j_0} = z_{mount} + z_{local}$
        ),
        rpy=(
            mount.rpy[0] + joint.origin.rpy[0],  # roll 分量直接叠加
            mount.rpy[1] + joint.origin.rpy[1],  # pitch 分量直接叠加
            mount.rpy[2] + joint.origin.rpy[2],  # yaw 分量直接叠加
        ),
    )
    return joint.replace(origin=origin)


def _build_link_elem(
    name: str,
    inertial: InertialCfg | None,
    collisions: list[CollisionGeometryCfg],
    visuals: list[VisualGeometryCfg],
    cfg: UrdfWriterCfg,
    *,
    mesh_state: _MeshExportState,
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
        visual_elem.append(_build_geometry_elem(visual.geometry, cfg, mesh_state=mesh_state))
        material = cfg.recolored_materials.get(name) or visual.material
        if material is not None:
            visual_elem.append(_build_material_elem(material))

    for collision in collisions:
        collision_elem = ET.SubElement(link, "collision")
        if collision.name:
            collision_elem.attrib["name"] = collision.name
        ET.SubElement(collision_elem, "origin", attrib=_pose_attrib(collision.origin))
        collision_elem.append(_build_geometry_elem(collision.geometry, cfg, mesh_state=mesh_state))

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
        if joint.joint_properties is not None and joint.joint_properties.friction is not None:
            ET.SubElement(
                joint_elem,
                "joint_properties",
                attrib={"friction": _fmt_scalar(joint.joint_properties.friction)},
            )

    return joint_elem


def _build_fixed_joint(name: str, parent: str, child: str, origin: PoseCfg) -> ET.Element:
    r"""构建一个通用 fixed joint XML 元素。

    Returns:
        xml.etree.ElementTree.Element: 构建好的 fixed joint 元素。
    """

    joint_elem = ET.Element("joint", attrib={"name": name, "type": "fixed"})
    ET.SubElement(joint_elem, "parent", attrib={"link": parent})
    ET.SubElement(joint_elem, "child", attrib={"link": child})
    ET.SubElement(joint_elem, "origin", attrib=_pose_attrib(origin))
    return joint_elem


def _build_geometry_elem(geom, cfg: UrdfWriterCfg, *, mesh_state: _MeshExportState) -> ET.Element:
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
    elif kind == "elliptic_cylinder":
        mesh_geom = _lower_elliptic_cylinder_to_mesh(geom, mesh_state=mesh_state)
        filename = mesh_geom.file_path
        if cfg.mesh_package_prefix and not filename.startswith(("package://", "/")):
            filename = f"{cfg.mesh_package_prefix.rstrip('/')}/{filename.lstrip('./')}"
        mesh_attrib = {"filename": filename}
        if mesh_geom.scale != (1.0, 1.0, 1.0):
            mesh_attrib["scale"] = _fmt_triplet(mesh_geom.scale)
        ET.SubElement(geometry_elem, "mesh", attrib=mesh_attrib)
    elif kind == "sphere":
        ET.SubElement(geometry_elem, "sphere", attrib={"radius": _fmt_scalar(geom.radius)})
    elif kind == "mesh":
        filename = _materialize_mesh_geometry(geom, mesh_state=mesh_state)
        if cfg.mesh_package_prefix and not filename.startswith(("package://", "/")):
            filename = f"{cfg.mesh_package_prefix.rstrip('/')}/{filename.lstrip('./')}"
        mesh_attrib = {"filename": filename}
        if isinstance(geom, MeshGeometryCfg) and geom.scale != (1.0, 1.0, 1.0):
            mesh_attrib["scale"] = _fmt_triplet(geom.scale)
        ET.SubElement(geometry_elem, "mesh", attrib=mesh_attrib)
    else:
        raise ValueError(f"Unsupported URDF geometry kind: {kind}")
    return geometry_elem


def _lower_elliptic_cylinder_to_mesh(
    geom: EllipticCylinderGeometryCfg,
    *,
    mesh_state: _MeshExportState,
) -> MeshGeometryCfg:
    r"""把内部椭圆柱 primitive lower 成 canonical cylinder mesh。

    这里的 lower 策略是：

    1. 自动生成一份单位 canonical cylinder mesh，主轴沿局部 $+y$；
    2. 用三轴缩放
       $$
       (2r_x,\ l,\ 2r_z)
       $$
       把它映射到目标椭圆柱。

    之所以不直接在导出时生成“最终尺寸 mesh”，是为了让 URDF 里仍保留显式 scale，
    便于人工巡检“椭圆柱到底被拉成了什么样子”。
    """

    rel_path = _ensure_unit_cylinder_mesh(mesh_state)  # 每个样本目录复用同一份 canonical cylinder mesh
    return MeshGeometryCfg(
        file_path=rel_path,
        scale=(2.0 * geom.radius_x, geom.length, 2.0 * geom.radius_z),  # unit cylinder → 椭圆柱的三轴缩放
    )


def _materialize_mesh_geometry(geom: MeshGeometryCfg, *, mesh_state: _MeshExportState) -> str:
    r"""把真实 mesh 几何 materialize 到当前导出边界，并返回 URDF 相对路径。

    当前 mesh 导出 contract 是：

    - primitive 继续原样写 `<box>/<sphere>/<cylinder>`；
    - procedural `cs` URI 在这里兜底物化为当前导出边界下的 OBJ；
    - 只有真实 `<mesh filename=...>` 会进入这里；
    - 同一导出边界内，相同源 mesh 只复制一份，尺寸差异继续由 URDF `scale`
      表达，而不是通过复制多份 STL 表达。
    """

    if is_procedural_cs_tip_uri(geom.file_path):
        return _materialize_procedural_cs_tip_geometry(geom, mesh_state=mesh_state)

    source_path = Path(geom.file_path).expanduser()
    if not source_path.is_absolute():
        return geom.file_path  # 相对路径或 package:// 暂不重写；当前项目真实 mesh 都来自本地绝对源路径

    cached = mesh_state._materialized_mesh_relpaths.get(source_path)
    if cached is not None:
        return cached

    mesh_root = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)
    target_name = _resolve_materialized_mesh_name(source_path, mesh_state=mesh_state)
    target_path = mesh_root / target_name

    if mesh_state.write_enabled:
        mesh_root.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            mesh_state.written.append(target_path)

    rel_path = os.path.relpath(target_path, start=mesh_state.output_dir)
    mesh_state._materialized_mesh_relpaths[source_path] = rel_path
    return rel_path


def _materialize_procedural_cs_tip_geometry(geom: MeshGeometryCfg, *, mesh_state: _MeshExportState) -> str:
    r"""导出器兜底物化 `procedural://anymani/cs_tip`。

    generator 主链会在 physics closure 前先完成这件事；这里保留兜底，是为了让
    单元测试、交互式调试或直接调用 `UrdfWriter.export(hand, ...)` 时，也不会把
    运行时中间 URI 泄漏进 URDF。URDF 的 `<mesh filename=...>` 必须指向真实文件，
    否则 IsaacLab / URDF Visualizer 都无法加载该 fingertip。
    """

    spec = parse_procedural_cs_tip_uri(geom.file_path)  # 从 URI 还原 $(r,h,N_\theta,N_\phi)$ 参数
    mesh_root = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)  # 当前导出边界共享 mesh 目录
    mesh_path, written = materialize_procedural_cs_tip_mesh(
        spec,
        mesh_root,
        write_enabled=mesh_state.write_enabled,
    )  # `write_enabled=False` 时只返回应写路径，用于 `to_urdf_string()`
    if written:
        mesh_state.written.append(mesh_path)  # ExportResult 记录新写出的 procedural OBJ
    return os.path.relpath(mesh_path, start=mesh_state.output_dir)  # URDF 始终写相对当前 hand.urdf 的路径


def _resolve_materialized_mesh_name(source_path: Path, *, mesh_state: _MeshExportState) -> str:
    r"""为 materialized mesh 分配稳定文件名。

    同一导出边界内：

    - 同源文件固定复用同一文件名；
    - basename 不冲突时直接保留；
    - basename 冲突但源文件不同，则追加稳定短 hash。
    """

    basename = source_path.name
    occupied = mesh_state._materialized_mesh_names.get(basename)
    if occupied is None or occupied == source_path:
        mesh_state._materialized_mesh_names[basename] = source_path
        return basename

    stem = source_path.stem
    suffix = source_path.suffix
    digest = hashlib.md5(str(source_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{stem}_{digest}{suffix}"
    mesh_state._materialized_mesh_names[candidate] = source_path
    return candidate


def _ensure_unit_cylinder_mesh(mesh_state: _MeshExportState) -> str:
    r"""确保当前导出上下文可见一份主轴沿局部 $+y$ 的单位圆柱 OBJ。"""

    if mesh_state._unit_cylinder_relpath is not None:
        return mesh_state._unit_cylinder_relpath  # 同一轮导出已生成过 canonical mesh 时直接复用

    mesh_dir = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)  # 当前导出边界下的共享 mesh 目录
    mesh_path = mesh_dir / "unit_cylinder_y.obj"  # 当前统一采用一份 y-axis cylinder 基底网格
    if mesh_state.write_enabled:
        mesh_dir.mkdir(parents=True, exist_ok=True)  # 只有真实导出时才在样本目录里创建 mesh 子目录
        if not mesh_path.exists():
            mesh_path.write_text(_unit_cylinder_y_obj_text(), encoding="utf-8")  # 首次遇到椭圆柱时再真正写文件
            mesh_state.written.append(mesh_path)
    mesh_state._unit_cylinder_relpath = os.path.relpath(mesh_path, start=mesh_state.output_dir)  # URDF 使用相对当前 hand.urdf 的路径
    return mesh_state._unit_cylinder_relpath


def _unit_cylinder_y_obj_text(*, segments: int = 24) -> str:
    r"""生成一份主轴沿局部 $+y$ 的单位圆柱 OBJ 文本。

    canonical cylinder mesh 约定如下：

    - 半径为 $1$；
    - 长度为 $1$；
    - 中心位于原点；
    - 主轴沿局部 $+y$；
    - 底面中心 $y=-0.5$，顶面中心 $y=+0.5$。

    这样导出到 URDF 时，只需施加
    $$
    (2r_x,\ l,\ 2r_z)
    $$
    的三轴 scale，就能得到目标椭圆柱。
    """

    segments = max(int(segments), 3)  # OBJ 圆周最少需要 3 段，避免退化成非法 mesh
    lines: list[str] = ["# canonical unit cylinder aligned with +y"]  # 头注释便于人工识别该文件由导出器自动生成
    lines.append("v 0 -0.5 0")  # 底面圆心
    lines.append("v 0 0.5 0")  # 顶面圆心

    # 先写底/顶圆环顶点。未缩放横截面位于 $(x,z)$ 平面，非均匀 scale 后自然得到椭圆。
    for ring_y in (-0.5, 0.5):
        for index in range(segments):
            theta = 2.0 * math.pi * index / segments  # 第 index 个圆周角
            x = math.cos(theta)  # 单位圆横截面在局部 $x$ 上的坐标
            z = math.sin(theta)  # 单位圆横截面在局部 $z$ 上的坐标
            lines.append(f"v {_fmt_scalar(x)} {_fmt_scalar(ring_y)} {_fmt_scalar(z)}")

    bottom_center_index = 1  # OBJ 使用 1-based 索引；第 1 个顶点是底面圆心
    top_center_index = 2  # 第 2 个顶点是顶面圆心
    bottom_start = 3  # 底面圆环顶点起始索引
    top_start = 3 + segments  # 顶面圆环顶点起始索引

    # 底面三角扇。为了让外法向朝 $-y$，底面索引顺序取反。
    for index in range(segments):
        current = bottom_start + index
        nxt = bottom_start + ((index + 1) % segments)
        lines.append(f"f {bottom_center_index} {nxt} {current}")

    # 顶面三角扇。顶面外法向朝 $+y$。
    for index in range(segments):
        current = top_start + index
        nxt = top_start + ((index + 1) % segments)
        lines.append(f"f {top_center_index} {current} {nxt}")

    # 侧面用两个三角形拼成一块矩形 patch。
    for index in range(segments):
        bottom_current = bottom_start + index
        bottom_next = bottom_start + ((index + 1) % segments)
        top_current = top_start + index
        top_next = top_start + ((index + 1) % segments)
        lines.append(f"f {bottom_current} {bottom_next} {top_next}")
        lines.append(f"f {bottom_current} {top_next} {top_current}")

    return "\n".join(lines) + "\n"


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
