"""分层 quick-check exporter 回归测试。

这组测试锁住当前新增的三个并列 exporter：

1. `JointExporter`
2. `FingerExporter`
3. `PalmExporter`

测试重点不是文件名细节，而是：

- 是否真能导出独立 URDF
- 是否符合约定的 preview 语义
- 是否没有偷偷退回到整手导出路径
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import yaml
from assets.asset_schema_core import CollisionGeometryCfg, PoseCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.builder.palm_builders import ComPalmBuilder, ComPalmBuilderCfg
from assets.exporter import (
    FingerExporter,
    FingerExporterCfg,
    JointExporter,
    JointExporterCfg,
    PalmExporter,
    PalmExporterCfg,
    SidecarCfg,
    SidecarExporter,
)
from assets.presets import get_finger_builder_preset


def _build_allegro_finger():
    """构造一根稳定的 Allegro 非拇指 finger。"""

    cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(name="index", parent_link="palm")
    return cfg.class_type(cfg).build()


def _build_simple_hand_for_sidecar() -> HandCfg:
    r"""构造一只长度可解析的简化整手，用于 sidecar 统计回归。

    这只手只保留一根 index finger，长度构造与 validator 单测一致：

    - 第一段 box：区间 $[-2, 2]$cm；
    - 第二段 box：区间 $[2, 6]$cm；
    - 指尖 sphere：区间 $[7, 9]$cm。

    因此 `total_length_cm` 的新语义应稳定为 $11$cm，而不是旧的 joint-origin 求和近似。
    """

    return HandCfg(
        name="sidecar_demo",
        palm=PalmCfg(
            name="palm",
            collisions=[
                CollisionGeometryCfg(
                    name="palm_col",
                    geometry={"type": "box", "size": (0.08, 0.06, 0.02)},
                )
            ],
        ),
        fingers=[
            FingerCfg(
                name="index",
                parent_link="palm",
                mount=PoseCfg(),
                joints=[
                    JointCfg(
                        name="index_j0",
                        parent="palm",
                        child="index_link_0",
                        origin=PoseCfg(),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_link_0_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="index_j1",
                        parent="index_link_0",
                        child="index_link_1",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_link_1_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="index_tip_fixed",
                        parent="index_link_1",
                        child="index_tip",
                        joint_type="fixed",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_tip_col",
                                geometry={"type": "sphere", "radius": 0.01},
                            )
                        ],
                        is_tip=True,
                    ),
                ],
            )
        ],
        family="unit_test",
        handedness="right",
    )


def test_joint_exporter_writes_standalone_joint_preview(tmp_path):
    """joint exporter 应写出 `stub base -> target joint -> child link` 的局部 URDF。"""

    joint = _build_allegro_finger().joints[0]
    result = JointExporter(JointExporterCfg()).export(joint, tmp_path)

    assert result.ok is True
    assert len(result.written) == 1

    root = ET.parse(result.written[0]).getroot()
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint_elem.attrib["name"] for joint_elem in root.findall("joint")}

    assert "joint_preview_base" in link_names
    assert joint.child in link_names
    assert joint.name in joint_names


def test_finger_exporter_writes_standalone_finger_preview(tmp_path):
    """finger exporter 应保留整根 finger 的 joint 链，并记录附带写出的 procedural mesh。"""

    finger = _build_allegro_finger()
    result = FingerExporter(FingerExporterCfg()).export(finger, tmp_path)

    assert result.ok is True
    urdf_path = tmp_path / "finger.urdf"
    assert urdf_path in result.written
    assert any(path.parent.name == "meshes" and path.suffix == ".obj" for path in result.written)

    root = ET.parse(urdf_path).getroot()
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint_elem.attrib["name"] for joint_elem in root.findall("joint")}

    assert "finger_preview_base" in link_names
    assert set(finger.joint_names).issubset(joint_names)
    assert "index_mount_joint" not in joint_names


def test_palm_exporter_draws_mount_markers_and_stub_roots(tmp_path):
    """palm exporter 应从 `finger_mounts` metadata 生成 marker + stub-root 预览 link。"""

    palm = ComPalmBuilder(ComPalmBuilderCfg(preset="allegro")).build()
    result = PalmExporter(PalmExporterCfg()).export(palm, tmp_path)

    assert result.ok is True
    assert len(result.written) == 1

    root = ET.parse(result.written[0]).getroot()
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint_elem.attrib["name"] for joint_elem in root.findall("joint")}

    for finger_name in ("index", "middle", "ring", "thumb"):
        assert f"{finger_name}_mount_preview_joint" in joint_names
        assert f"{finger_name}_mount_preview_link" in link_names

    index_preview_link = next(link for link in root.findall("link") if link.attrib["name"] == "index_mount_preview_link")
    preview_geometry_kinds = {
        child[0].tag
        for child in (*index_preview_link.findall("visual/geometry"), *index_preview_link.findall("collision/geometry"))
    }
    assert preview_geometry_kinds == {"sphere", "box"}


def test_sidecar_uses_real_axial_geometry_length_for_total_length_cm(tmp_path):
    r"""sidecar `total_length_cm` 应与新轴向几何长度语义保持一致。"""

    hand = _build_simple_hand_for_sidecar()
    result = SidecarExporter(SidecarCfg()).export(hand, tmp_path)

    assert result.ok is True
    doc = yaml.safe_load(result.written[0].read_text(encoding="utf-8"))

    assert doc["fingers"][0]["name"] == "index"
    assert doc["fingers"][0]["total_length_cm"] == 11.0
    assert doc["geometry_semantics"]["schema_version"] == "1.0.0"
    assert len(doc["geometry_semantics"]["components"]) == 4
