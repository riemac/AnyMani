r"""颜色恢复开关的纯配置合同；用设置桩替代Kit，不启动模拟器或GPU。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from anymani.robots.visual_material_policy import generated_hand_visual_materials_enabled


@pytest.mark.parametrize(
    "enabled_setting",
    [None, "/app/window/enabled", "/app/livestream/enabled", "/app/xr/enabled", "/isaaclab/render/offscreen"],
)
def test_render_intent_controls_material_restore(monkeypatch, enabled_setting: str | None) -> None:
    r"""纯headless关闭；窗口、直播、XR和headless摄像机均启用URDF颜色。"""
    settings = {enabled_setting: True} if enabled_setting else {}
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))

    assert generated_hand_visual_materials_enabled() is (enabled_setting is not None)


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_explicit_heterogeneous_override_is_preserved(monkeypatch, value: str, expected: bool) -> None:
    r"""已有异构可视化工具的显式开关优先，且不需要加载Kit。"""
    monkeypatch.setenv("ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS", value)
    monkeypatch.setitem(sys.modules, "carb", None)

    assert generated_hand_visual_materials_enabled(override_env="ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS") is expected


def test_unset_override_uses_offscreen_state(monkeypatch) -> None:
    r"""环境变量缺省时自动识别headless录像，而非继续沿用灰色默认值。"""
    monkeypatch.delenv("ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS", raising=False)
    settings = {"/isaaclab/render/offscreen": True}
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))

    assert generated_hand_visual_materials_enabled(override_env="ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS")


def test_without_kit_defaults_to_no_material_work(monkeypatch) -> None:
    r"""离线配置检查不应为判断颜色而加载Isaac运行时。"""
    monkeypatch.setitem(sys.modules, "carb", None)

    assert not generated_hand_visual_materials_enabled()


@pytest.mark.parametrize(
    "config_path",
    [
        "hetero/config/generated/asset_binding.py",
        "gm/config/single_asset/single_asset_env_cfg.py",
        "inhand/config/generated_right_t4_i4_m4_r4/generated_right_t4_i4_m4_r4_adr_env_cfg.py",
    ],
)
def test_generated_scene_configs_use_shared_render_policy(config_path: str) -> None:
    r"""三条generated场景均接入共享开关；通过AST核对而不导入Isaac场景。"""
    source = (Path(__file__).resolve().parents[3] / "tasks" / config_path).read_text(encoding="utf-8")
    bindings = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.keyword) and node.arg == "restore_visual_materials"
    ]

    assert len(bindings) == 1
    assert isinstance(bindings[0], ast.Call)
    assert isinstance(bindings[0].func, ast.Name)
    assert bindings[0].func.id == "generated_hand_visual_materials_enabled"
