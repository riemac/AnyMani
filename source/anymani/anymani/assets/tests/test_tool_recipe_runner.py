"""tooling 层回归测试：RecipeLoader + GeneratorRunner。

这组测试锁住当前首轮真正需要的运行时入口契约：

1. recipe 可从 dict / YAML 收敛到 typed `HandGeneratorCfg`
2. 历史 `export_dir` 写法仍能兼容到当前 `output_dir`
3. runner 能接受 cfg / dict / path 三种入口
4. `bundle` 和 `hand_cfg` 两种产物粒度都能得到稳定文件组织
"""

from __future__ import annotations

import yaml

from assets.builder.hand_builders import HumanLikeHandBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.exporter.hand_exporter import HandExporterCfg
from assets.generator.hand_generator import HandGeneratorCfg
from assets.tool.recipe_loader import RecipeLoader
from assets.tool.runner import GeneratorRunner
from assets.validator.hand_rules import HandValidatorCfg


def _made_recipe_dict() -> dict:
    """返回一份最小但完整的 Allegro pre-made recipe dict。"""

    return {
        "name": "allegro_tooling_demo",
        "family": "allegro",
        "handedness": "right",
        "palm_cfg": "com_allegro",
        "finger_cfg": "allegro_non_thumb_v1",
        "thumb_cfg": "allegro_thumb_v1",
    }


def _tool_recipe_dict(output_dir: str, *, artifact_level: str) -> dict:
    """返回一份可直接喂给 RecipeLoader / Runner 的完整 recipe。"""

    return {
        "mode": "full",
        "artifact_level": artifact_level,
        "sampling_strategy": "sample",
        "n_samples": 1,
        # 这里故意继续用历史字段名，锁住 loader 的兼容桥接语义。
        "export_dir": output_dir,
        "Made": _made_recipe_dict(),
        "Validate": {
            "strict": False,
            "finger": {
                "strict": False,
                "joint": {
                    "strict": False,
                    "min_link_length": 1e-5,
                },
            },
        },
        "Export": {
            "artifact_level": artifact_level,
            "Sidecar": {
                "experiment_tag": "tooling_regression",
            },
        },
    }


def test_recipe_loader_builds_typed_cfg_and_bridges_legacy_export_dir(tmp_path):
    """dict recipe 应被解析为 typed cfg，并把 `export_dir` 兼容到 `output_dir`。"""

    cfg = RecipeLoader.load_dict(_tool_recipe_dict(str(tmp_path), artifact_level="bundle"))

    assert isinstance(cfg, HandGeneratorCfg)
    assert cfg.output_dir == tmp_path
    assert isinstance(cfg.Made, HumanLikeHandBuilderCfg)
    assert isinstance(cfg.Made.palm_cfg, ComPalmBuilderCfg)
    assert cfg.Made.palm_cfg.preset == "allegro"
    assert cfg.Made.finger_cfg.__class__.__name__ == "AllegroFingerBuilderCfg"
    assert cfg.Made.thumb_cfg.__class__.__name__ == "RegularThumbBuilderCfg"
    assert isinstance(cfg.Validate, HandValidatorCfg)
    assert cfg.Validate.finger.joint.min_link_length == 1e-5
    assert isinstance(cfg.Export, HandExporterCfg)
    assert cfg.Export.Sidecar.experiment_tag == "tooling_regression"


def test_recipe_loader_save_and_load_round_trip_keeps_current_contract(tmp_path):
    """save/load round-trip 应保留当前 tooling 关心的 typed contract。"""

    cfg = HandGeneratorCfg(
        mode="full",
        artifact_level="bundle",
        output_dir=tmp_path / "generated",
        Made=HumanLikeHandBuilderCfg(
            name="allegro_round_trip",
            family="allegro",
            handedness="right",
            palm_cfg=ComPalmBuilderCfg(preset="allegro"),
            finger_cfg="allegro_non_thumb_v1",
            thumb_cfg="allegro_thumb_v1",
        ),
    )
    recipe_path = tmp_path / "recipe.yaml"

    RecipeLoader.save(cfg, recipe_path)
    dumped = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    loaded = RecipeLoader.load(recipe_path)

    assert dumped["output_dir"] == str(tmp_path / "generated")
    assert "class_type" not in recipe_path.read_text(encoding="utf-8")
    assert isinstance(loaded, HandGeneratorCfg)
    assert loaded.output_dir == tmp_path / "generated"
    assert isinstance(loaded.Made, HumanLikeHandBuilderCfg)
    assert isinstance(loaded.Made.palm_cfg, ComPalmBuilderCfg)
    assert loaded.Made.finger_cfg.__class__.__name__ == "AllegroFingerBuilderCfg"
    assert loaded.Made.thumb_cfg.__class__.__name__ == "RegularThumbBuilderCfg"


def test_generator_runner_runs_from_dict_and_persists_hand_cfg_sidecar_and_trees(tmp_path):
    """runner 在 `hand_cfg` 模式下也应补齐 sidecar / tree 文件。"""

    out_dir = tmp_path / "hand_cfg_outputs"
    runner = GeneratorRunner(_tool_recipe_dict(str(tmp_path / "ignored"), artifact_level="hand_cfg"), output_dir=out_dir)

    results = runner.run()

    assert len(results) == 1
    result = results[0]
    assert result.hand_cfg is not None
    assert result.urdf_path is None
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    assert (result.sidecar_path.parent / "tree.txt").is_file()
    assert (result.sidecar_path.parent / "tree.mmd").is_file()
    assert result.sidecar_path.parent.parent == out_dir


def test_generator_runner_loads_yaml_recipe_and_respects_output_override_for_bundle(tmp_path):
    """runner 从 YAML 路径启动时，应把 bundle 产物写到覆写目录。"""

    recipe_path = tmp_path / "bundle_recipe.yaml"
    RecipeLoader.save(
        RecipeLoader.load_dict(_tool_recipe_dict(str(tmp_path / "legacy_out"), artifact_level="bundle")),
        recipe_path,
    )
    out_dir = tmp_path / "bundle_outputs"

    results = list(GeneratorRunner(recipe_path, output_dir=out_dir).stream())

    assert len(results) == 1
    result = results[0]
    assert result.hand_cfg is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    assert result.urdf_path.parent.parent == out_dir
    assert result.sidecar_path.parent.parent == out_dir
