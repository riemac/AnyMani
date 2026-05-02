"""URDF recolored 导出测试。

这组测试锁住的是本轮新接通的 visual recolor contract：

1. `HandGeneratorCfg.recolored="anatomy_v1"` 时，会按 anatomy 语义给各 link 的
   `<visual>` 注入 `<material><color .../></material>`；
2. LEAP non-thumb 的 `root_fixed_link` 会和 palm 一样被染成红色；
3. `recolored={link_name: rgba}` 时，只覆盖指定 link；
4. `<collision>` 永远不写 material。
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg():
    return make_human_like_builder_cfg(
        name="allegro_recolor_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _make_leap_builder_cfg():
    return make_human_like_builder_cfg(
        name="leap_recolor_demo",
        family="leap",
        handedness="right",
        palm_cfg="single_box_leap",
        finger_cfg="leap_non_thumb_v1",
        thumb_cfg="leap_thumb_v1",
    )


def _visual_color_map(urdf_path: Path) -> dict[str, list[str]]:
    root = ET.parse(urdf_path).getroot()
    color_map: dict[str, list[str]] = {}
    for link in root.findall("link"):
        colors = [
            color.attrib["rgba"]
            for visual in link.findall("visual")
            if (material := visual.find("material")) is not None
            if (color := material.find("color")) is not None
        ]
        if colors:
            color_map[str(link.attrib["name"])] = colors
    return color_map


def _collision_material_count(urdf_path: Path) -> int:
    root = ET.parse(urdf_path).getroot()
    return sum(
        1
        for link in root.findall("link")
        for collision in link.findall("collision")
        if collision.find("material") is not None
    )


def _rgba_string_to_tuple(rgba: str) -> tuple[float, float, float, float]:
    r"""把 URDF color 字符串转回浮点 tuple，避免测试被格式化精度绑死。"""

    return tuple(float(value) for value in rgba.split())  # type: ignore[return-value]


def test_named_recolored_palette_injects_visual_materials_and_colors_leap_root_fixed(tmp_path):
    r"""命名 palette 应按 anatomy 语义落到 visual 上。"""

    result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="urdf",
            output_dir=tmp_path,
            Made=_make_leap_builder_cfg(),
            recolored="anatomy_v1",
        )
    ).generate()

    assert result is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()

    color_map = _visual_color_map(result.urdf_path)
    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    assert all(rgba == "1 0 0 1" for rgba in color_map["palm"])
    assert all(rgba == "1 0 0 1" for rgba in color_map["index_root_fixed_link"])
    assert all(rgba == "1 1 0 1" for rgba in color_map["index_mcp1"])
    assert all(rgba == "0 1 1 1" for rgba in color_map["thumb_cmc2"])
    assert all(rgba == "1 0 1 1" for rgba in color_map["thumb_tip"])
    assert sidecar["hand_cfg"]["family"] == "leap"
    assert sidecar["hand_cfg"]["handedness"] == "right"
    assert _collision_material_count(result.urdf_path) == 0


def test_recolored_dict_only_overrides_requested_visual_link(tmp_path):
    r"""显式 override 字典只应影响用户点名的 child link。"""

    result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="urdf",
            output_dir=tmp_path,
            Made=_make_allegro_builder_cfg(),
            recolored={"index_tip": (0.25, 0.5, 0.75, 1.0)},
        )
    ).generate()

    assert result is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()

    color_map = _visual_color_map(result.urdf_path)
    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    assert color_map["index_tip"]
    assert all(rgba == "0.25 0.5 0.75 1" for rgba in color_map["index_tip"])
    assert "index_dip" not in color_map
    assert "palm" not in color_map
    assert sidecar["hand_cfg"]["family"] == "allegro"
    assert sidecar["hand_cfg"]["fingers"]
    assert _collision_material_count(result.urdf_path) == 0


def test_soft_recolored_palette_preserves_anatomy_hues_with_lower_saturation(tmp_path):
    r"""`anatomy_soft_v1` 应保留 anatomy 色相语义，但避免 RGB 原色刺眼。

    这里不测试“审美好坏”，只锁住两个科研可复现事实：

    - palm / root_fixed 仍是红色语义；
    - mcp1 / cmc2 / pip / dip / tip 仍对应黄 / 青 / 绿 / 蓝 / 紫语义，
      但数值不再是 `0/1` 饱和原色。
    """

    result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="urdf",
            output_dir=tmp_path,
            Made=_make_leap_builder_cfg(),
            recolored="anatomy_soft_v1",
        )
    ).generate()

    assert result is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()

    color_map = _visual_color_map(result.urdf_path)
    palm_rgba = _rgba_string_to_tuple(color_map["palm"][0])
    cyan_rgba = _rgba_string_to_tuple(color_map["thumb_cmc2"][0])
    green_rgba = _rgba_string_to_tuple(color_map["index_pip"][0])
    blue_rgba = _rgba_string_to_tuple(color_map["index_dip"][0])
    purple_rgba = _rgba_string_to_tuple(color_map["thumb_tip"][0])

    assert palm_rgba[0] > palm_rgba[1] and palm_rgba[0] > palm_rgba[2]
    assert cyan_rgba[1] > cyan_rgba[0] and cyan_rgba[2] > cyan_rgba[0]
    assert green_rgba[1] > green_rgba[0] and green_rgba[1] > green_rgba[2]
    assert blue_rgba[2] > blue_rgba[0] and blue_rgba[2] > blue_rgba[1]
    assert purple_rgba[2] > purple_rgba[0] and purple_rgba[0] > purple_rgba[1]
    assert all(channel < 0.95 for rgba in [palm_rgba, cyan_rgba, green_rgba, blue_rgba, purple_rgba] for channel in rgba[:3])
