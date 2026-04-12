"""pre-made 闭环的 validator / exporter / generator 测试。

这组测试把首轮最关键的纵向契约锁住：

1. `HandCfg -> HandValidator`
2. `HandCfg -> UrdfWriter`
3. `HandGenerator.generate()` 在不启用 mutate 时能稳定产出 bundle

测试设计上尽量复用同一份 Allegro 锚点 hand，避免因为测试样本漂移掩盖接口问题。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.exporter.urdf_writer import UrdfWriter, UrdfWriterCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.validator.hand_rules import HandValidator, HandValidatorCfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """与 builder 测试共用的一份 Allegro hand recipe。"""

    return HumanLikeHandBuilderCfg(
        name="allegro_demo",
        family="allegro",
        handedness="right",
        palm_cfg=ComPalmBuilderCfg(preset="allegro"),
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供纵向测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def test_hand_validator_reports_mount_spacing_warning_without_rejecting():
    """validator 应允许 warning 放行，但把 spacing 问题记录下来。"""

    hand = _build_allegro_hand()
    validator = HandValidator(HandValidatorCfg(min_finger_spacing=0.05))

    result = validator.validate(hand)

    assert result.passed is True
    assert result.errors == []
    assert any("finger spacing" in warning for warning in result.warnings)


def test_urdf_writer_inserts_mount_link_when_enabled():
    """开启 `use_mount_link` 时，URDF 里应显式插入 mount joint / mount link。"""

    hand = _build_allegro_hand()
    writer = UrdfWriter(UrdfWriterCfg(use_mount_link=True))

    root = ET.fromstring(writer.to_urdf_string(hand))
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_elems = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert "index_mount_link" in link_names
    assert "index_mount_joint" in joint_elems
    assert joint_elems["index_mount_joint"].attrib["type"] == "fixed"
    assert joint_elems["index_j0"].find("parent").attrib["link"] == "index_mount_link"


def test_urdf_writer_folds_mount_into_first_joint_when_mount_link_disabled():
    """关闭 `use_mount_link` 时，mount 应折叠进第一关节 origin。"""

    hand = _build_allegro_hand()
    writer = UrdfWriter(UrdfWriterCfg(use_mount_link=False))

    root = ET.fromstring(writer.to_urdf_string(hand))
    joint_elems = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    # 关闭 mount link 后，不应再看到虚拟 mount joint；
    # 第一关节的 parent 直接回到 palm。
    assert "index_mount_joint" not in joint_elems
    assert joint_elems["index_j0"].find("parent").attrib["link"] == hand.palm.name


def test_hand_generator_returns_bundle_and_exports_to_configured_directory(tmp_path):
    """generator 在 pre-made 路线上应能返回 bundle，并把产物写到指定目录。"""

    cfg = HandGeneratorCfg(
        mode="full",
        artifact_level="bundle",
        output_dir=tmp_path,
        Made=_make_allegro_builder_cfg(),
    )

    result = HandGenerator(cfg).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    assert result.tree_txt is not None
    assert result.tree_mermaid is not None
    assert result.urdf_path.parent.parent == tmp_path

