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

from assets.builder.palm_builders import ComPalmBuilder, ComPalmBuilderCfg
from assets.exporter import (
    FingerExporter,
    FingerExporterCfg,
    JointExporter,
    JointExporterCfg,
    PalmExporter,
    PalmExporterCfg,
)
from assets.presets import get_finger_builder_preset


def _build_allegro_finger():
    """构造一根稳定的 Allegro 非拇指 finger。"""

    cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(name="index", parent_link="palm")
    return cfg.class_type(cfg).build()


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
    """finger exporter 应保留整根 finger 的 joint 链，但不引入 hand-level mount。"""

    finger = _build_allegro_finger()
    result = FingerExporter(FingerExporterCfg()).export(finger, tmp_path)

    assert result.ok is True
    assert len(result.written) == 1

    root = ET.parse(result.written[0]).getroot()
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
