"""统一资产生成配置与 runner 回归测试。"""

from __future__ import annotations

from importlib import import_module

from anymani.assets.config import asset_gen_cfg as asset_cfg_module
from anymani.assets.generator.hand_generator import HandGeneratorCfg
from anymani.assets.scripts import _asset_generate_runner as runner_module
from anymani.assets.scripts import generate as generate_module
from anymani.assets.scripts._asset_generate_runner import enumerate_premade_bundles
from anymani.assets.validator.hand_rules import HandValidatorCfg


def _single_family_full_pool(hand_preset: str, family: str) -> dict[str, dict[str, list[str]]]:
    r"""构造 slot-level full-chain pool。"""

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


def test_premade_runner_enumerates_small_recolored_space_and_writes_bundle(tmp_path):
    r"""pre-made runner helper 应直接用 `HandGeneratorCfg` 驱动正式 bundle 导出。"""

    cfg = HandGeneratorCfg(
        mode="made",
        hand_presets=["single_palm_allegro"],
        connectivity_presets=_single_family_full_pool("single_palm_allegro", "allegro"),
        mixed=False,
        missing=False,
        recolored="anatomy_soft_v1",
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


def test_asset_config_exposes_direct_premade_hand_generator_cfg():
    r"""配置模块中的 `PRE_MADE_CFG` 应直接是 `HandGeneratorCfg`。"""

    assert isinstance(asset_cfg_module.PRE_MADE_CFG, HandGeneratorCfg)
    assert isinstance(asset_cfg_module.PRE_MADE_CFG.Validate, HandValidatorCfg)
    assert asset_cfg_module.PRE_MADE_CFG.recolored == "anatomy_soft_v1"
    assert asset_cfg_module.PRE_MADE_CFG.premade_parallel is True
    assert asset_cfg_module.PRE_MADE_CFG.premade_parallel_fallback == "serial"
    assert asset_cfg_module.PRE_MADE_CFG.Validate.pre_made.finger_count_min == 3
    assert asset_cfg_module.PRE_MADE_CFG.Validate.pre_made.require_non_thumb_with_min_revolute_dof == 3
    assert asset_cfg_module.PRE_MADE_CFG.Validate.pre_made.check_palm_thumb_binding is True


def test_unified_generate_runner_accepts_premade_cli(monkeypatch, tmp_path):
    r"""统一 runner 应支持通过 CLI 选择 pre-made 阶段。"""

    custom_module = import_module("anymani.assets.config.asset_gen_cfg")
    monkeypatch.setattr(custom_module, "PRE_MADE_SHOW_REGISTRY", False)
    monkeypatch.setattr(custom_module, "PRE_MADE_PRINT_RESULT_LIMIT", 0)
    monkeypatch.setattr(
        custom_module,
        "PRE_MADE_CFG",
        HandGeneratorCfg(
            mode="made",
            hand_presets=["single_palm_leap"],
            connectivity_presets=_single_family_full_pool("single_palm_leap", "leap"),
            mixed=False,
            missing=True,
            recolored=False,
            artifact_level="hand_cfg",
            output_dir=tmp_path,
            max_enumerate=2,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate.py",
            "--stage",
            "pre-made",
            "--config-module",
            "anymani.assets.config.asset_gen_cfg",
        ],
    )
    monkeypatch.setattr(runner_module, "enumerate_premade_bundles", lambda cfg: [])
    monkeypatch.setattr(generate_module, "enumerate_premade_bundles", lambda cfg: [])

    exit_code = generate_module.main()

    assert exit_code == 0
