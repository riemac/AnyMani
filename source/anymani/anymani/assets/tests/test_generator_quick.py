"""generator/quick.py façade 测试。

这组测试不追求覆盖 quick.py 里每一行打印，而是锁住两件真正重要的事：

1. quick façade 的顶部少量字段，确实会正确 lower 到正式 `HandGenerator` 主线；
2. mixed / missing / recolor 这些当前研究核心语义，不会因为“只是 quick 脚本”
   就悄悄绕开正式实现。
"""

from __future__ import annotations

import assets.generator.quick as quick_module
from assets.generator.hand_generator import HandGeneratorCfg
from assets.generator.quick import enumerate_premade_bundles, main


def _single_family_full_pool(hand_preset: str, family: str) -> dict[str, dict[str, list[str]]]:
    r"""构造 quick façade 现在直接接受的 slot-level full-chain pool。"""

    thumb_recipe = f"{family}_thumb_full"
    non_thumb_recipe = f"{family}_non_thumb_full"
    return {
        hand_preset: {
            "thumb": [thumb_recipe],
            "index": [non_thumb_recipe],
            "middle": [non_thumb_recipe],
            "ring": [non_thumb_recipe],
        }
    }


def test_quick_facade_enumerates_small_recolored_space_and_writes_bundle(tmp_path):
    r"""quick façade 应直接用 `HandGeneratorCfg` 驱动正式 pre-made bundle 导出。"""

    cfg = HandGeneratorCfg(
        mode="made",
        hand_presets=["single_palm_allegro"],
        connectivity_presets=_single_family_full_pool("single_palm_allegro", "allegro"),
        mixed=False,
        missing=False,
        recolored="anatomy_v1",
        artifact_level="bundle",
        output_dir=tmp_path,
        max_enumerate=1,
    )

    results = enumerate_premade_bundles(cfg)

    assert len(results) == 1
    result = results[0]
    assert result.hand_cfg is not None
    assert result.metadata["topology_kind"] == "single_family"
    assert result.metadata["connectivity_preset"] == "thumb-full__index-full__middle-full__ring-full"
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()


def test_quick_run_cfg_is_direct_hand_generator_cfg():
    r"""quick.py 顶部的唯一正式运行入口应直接是 `HandGeneratorCfg`。"""

    assert isinstance(quick_module.RUN_CFG, HandGeneratorCfg)


def test_quick_facade_main_accepts_small_custom_cfg(monkeypatch, tmp_path):
    r"""`main(cfg)` 路径应允许测试 / notebook 直接传入 `HandGeneratorCfg`。"""

    monkeypatch.setattr(quick_module, "_SHOW_REGISTRY", False)  # 测试里关闭 registry 打印，避免噪声淹没失败信息
    monkeypatch.setattr(quick_module, "_PRINT_RESULT_LIMIT", 0)  # 测试里只保留 summary，避免终端 preview 干扰断言阅读

    cfg = HandGeneratorCfg(
        mode="made",
        hand_presets=["single_palm_leap"],
        connectivity_presets=_single_family_full_pool("single_palm_leap", "leap"),
        mixed=False,
        missing=True,
        recolored=False,
        artifact_level="hand_cfg",
        output_dir=tmp_path,
        max_enumerate=2,
    )

    exit_code = main(cfg)

    assert exit_code == 0
