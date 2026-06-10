"""URDF recolored 导出测试。

这组测试锁住的是本轮新接通的 visual recolor contract：

1. `HandGeneratorCfg.recolored="anatomy_soft_v1"` 时，会按 anatomy 语义给各 link 的
   `<visual>` 注入 `<material><color .../></material>`；
2. LEAP non-thumb 的 `root_fixed_link` 会和 palm 一样被染成红色；
3. `recolored={link_name: rgba}` 时，只覆盖指定 link；
4. `<collision>` 永远不写 material。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg


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


def _make_mixed_tip_leap_builder_cfg():
    r"""构造一只同时含 custom mesh tip 与 procedural `cs` tip 的 LEAP 手。

    这个测试锚点对应用户截图中的真实问题：post-mutate 后同一只手可能同时出现
    custom mesh fingertip 与 `cs` fingertip。recolor contract 不应让两类 tip 因
    几何来源不同而变成不同颜色；它们都是末端接触皮肤，应共享 warm ivory。
    """

    custom_non_thumb = get_finger_builder_preset("leap_non_thumb_v1").replace(
        tip={"type": "mesh", "tip_type": "round"},  # non-thumb 指尖强制走 custom mesh tip 路线
    )
    return make_human_like_builder_cfg(
        name="leap_mixed_tip_recolor_demo",
        family="leap",
        handedness="right",
        palm_cfg="single_box_leap",
        finger_cfg=custom_non_thumb,
        thumb_cfg="leap_thumb_v1",  # thumb 保持默认 procedural `cs`，形成 custom/cs 混合样本
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
            recolored="anatomy_soft_v1",
        )
    ).generate()

    assert result is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()

    color_map = _visual_color_map(result.urdf_path)
    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    assert all(rgba == "0.603921569 0.149019608 0.149019608 1" for rgba in color_map["palm"])
    assert all(rgba == "0.603921569 0.149019608 0.149019608 1" for rgba in color_map["index_root_fixed_link"])
    assert all(rgba == "0.866666667 0.866666667 0.0509803922 1" for rgba in color_map["index_mcp1"])
    assert all(rgba == "0.0470588235 0.439215686 0.48627451 1" for rgba in color_map["thumb_cmc2"])
    assert all(rgba == "0.352941176 0.231372549 0.447058824 1" for rgba in color_map["index_dip"])
    assert all(rgba == "0.92 0.88 0.78 1" for rgba in color_map["thumb_tip"])
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


def test_soft_recolored_palette_uses_ivory_tip_and_purple_dip(tmp_path):
    r"""`anatomy_soft_v1` 应把接触皮肤和 distal link 视觉语义分开。

    这里不测试“审美好坏”，只锁住两个科研可复现事实：

    - palm / root_fixed 仍是红色语义；
    - `*_dip` 使用紫色，避免继续和第二段深青/靛蓝视觉重复；
    - `*_tip` 使用 warm ivory，统一 custom tip 与 procedural `cs` 的接触皮肤观感。
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
    purple_rgba = _rgba_string_to_tuple(color_map["index_dip"][0])
    ivory_rgba = _rgba_string_to_tuple(color_map["thumb_tip"][0])

    assert palm_rgba[0] > palm_rgba[1] and palm_rgba[0] > palm_rgba[2]
    assert cyan_rgba[1] > cyan_rgba[0] and cyan_rgba[2] > cyan_rgba[0]
    assert green_rgba[1] > green_rgba[0] and green_rgba[1] > green_rgba[2]
    assert purple_rgba[2] > purple_rgba[0] and purple_rgba[0] > purple_rgba[1]
    assert ivory_rgba == (0.92, 0.88, 0.78, 1.0)
    assert all(channel < 0.95 for rgba in [palm_rgba, cyan_rgba, green_rgba, purple_rgba, ivory_rgba] for channel in rgba[:3])


def test_soft_recolored_palette_gives_custom_and_cs_tips_same_ivory(tmp_path):
    r"""custom mesh tip 与 procedural `cs` tip 都应使用同一个 warm ivory。"""

    result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="urdf",
            output_dir=tmp_path,
            Made=_make_mixed_tip_leap_builder_cfg(),
            recolored="anatomy_soft_v1",
        )
    ).generate()

    assert result is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()

    color_map = _visual_color_map(result.urdf_path)
    assert _rgba_string_to_tuple(color_map["index_tip"][0]) == (0.92, 0.88, 0.78, 1.0)
    assert _rgba_string_to_tuple(color_map["thumb_tip"][0]) == (0.92, 0.88, 0.78, 1.0)
    assert _rgba_string_to_tuple(color_map["index_dip"][0]) == (0.352941176, 0.231372549, 0.447058824, 1.0)
