r"""generator recipe helper 回归测试：`RecipeLoader -> HandGenerator` 直连契约。

这组测试锁住 generator 内部 recipe helper 的边界：

1. `RecipeLoader` 仍负责 YAML / dict → typed `HandGeneratorCfg`
2. 历史 `export_dir` 仍兼容到当前 `output_dir`
3. recipe helper 不再通过 `GeneratorRunner` 额外包一层运行逻辑
4. 声明式 recipe 加载后，应直接驱动 `HandGenerator`

也就是说，这里验证的是：

$$
\text{recipe} \xrightarrow{\text{RecipeLoader}} \text{HandGeneratorCfg}
\xrightarrow{\text{HandGenerator}} \text{artifact}
$$

而不是旧的：

$$
\text{recipe} \xrightarrow{\text{RecipeLoader}} \text{GeneratorRunner}
\xrightarrow{\text{内部再转}} \text{HandGenerator}
$$
"""

from __future__ import annotations

import pytest
import yaml

from assets.builder.hand_builders import HumanLikeHandBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.asset_physics import AssetPhysicsCfg
from assets.exporter.hand_exporter import HandExporterCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutatorCfg, MountPerturbCfg
from assets.presets import make_human_like_builder_cfg
from assets.generator.runtime.recipe_loader import RecipeLoader
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
    """返回一份可直接喂给 `RecipeLoader` 的完整 recipe。"""

    return {
        "mode": "made",
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
    assert cfg.Validate.pre_made.finger.joint.min_link_length == 1e-5
    assert cfg.Validate.post_mutate.finger.joint.min_link_length == 1e-5
    assert isinstance(cfg.Export, HandExporterCfg)
    assert cfg.Export.Sidecar.experiment_tag == "tooling_regression"


def test_recipe_loader_builds_physics_cfg_block(tmp_path):
    r"""`Physics: {...}` 应被解析为 typed `AssetPhysicsCfg`。"""

    cfg = RecipeLoader.load_dict(
        {
            "mode": "made",
            "artifact_level": "hand_cfg",
            "output_dir": str(tmp_path / "generated"),
            "Made": _made_recipe_dict(),
            "Physics": {
                "density": {
                    "default": 700.0,
                    "custom_tip": 500.0,
                }
            },
        }
    )

    assert isinstance(cfg.Physics, AssetPhysicsCfg)
    assert cfg.Physics.density.default == 700.0
    assert cfg.Physics.density.custom_tip == 500.0


def test_recipe_loader_save_and_load_round_trip_keeps_current_contract(tmp_path):
    """save/load round-trip 应保留当前 tooling 关心的 typed contract。"""

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=tmp_path / "generated",
        Made=make_human_like_builder_cfg(
            name="allegro_round_trip",
            family="allegro",
            handedness="right",
            palm_cfg="com_allegro",
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
    assert loaded.Validate is None
    assert isinstance(loaded.Made, HumanLikeHandBuilderCfg)
    assert isinstance(loaded.Made.palm_cfg, ComPalmBuilderCfg)
    assert loaded.Made.finger_cfg.__class__.__name__ == "AllegroFingerBuilderCfg"
    assert loaded.Made.thumb_cfg.__class__.__name__ == "RegularThumbBuilderCfg"


def test_loaded_recipe_drives_hand_generator_directly_for_hand_cfg_mode(tmp_path):
    """`hand_cfg` 模式下，recipe helper 应只返回内存中的 `HandCfg`。"""

    cfg = RecipeLoader.load_dict(_tool_recipe_dict(str(tmp_path / "ignored"), artifact_level="hand_cfg"))
    cfg = cfg.replace(output_dir=tmp_path / "hand_cfg_outputs")

    result = HandGenerator(cfg).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.urdf_path is None
    assert result.sidecar_path is None


def test_loaded_recipe_drives_hand_generator_directly_for_bundle_mode(tmp_path):
    """`bundle` 模式下，recipe 加载后应直接得到完整落盘产物。"""

    recipe_path = tmp_path / "bundle_recipe.yaml"
    RecipeLoader.save(
        RecipeLoader.load_dict(_tool_recipe_dict(str(tmp_path / "legacy_out"), artifact_level="bundle")),
        recipe_path,
    )
    out_dir = tmp_path / "bundle_outputs"

    cfg = RecipeLoader.load(recipe_path).replace(output_dir=out_dir)
    result = HandGenerator(cfg).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    run_root = result.urdf_path.parent
    assert run_root.parent == out_dir
    assert result.sidecar_path.parent == run_root
    assert (run_root / "summary.yaml").is_file()


def test_recipe_loader_rejects_removed_output_layout_field(tmp_path):
    r"""`RecipeLoader` 应对已删除的 `output_layout` 字段给出清晰报错。

    这轮目录 contract 已经固定，不再接受旧的 layout 风格字段。
    """

    with pytest.raises(ValueError, match="Removed HandGeneratorCfg fields"):
        RecipeLoader.load_dict(
            {
                "mode": "made",
                "artifact_level": "hand_cfg",
                "output_dir": str(tmp_path / "generated"),
                "sampling_strategy": "enumerate",
                "hand_presets": ["single_palm_allegro"],
                "connectivity_presets": {
                    "single_palm_allegro": {
                        "thumb": ["allegro_thumb_full"],
                        "index": ["allegro_non_thumb_full"],
                        "middle": ["allegro_non_thumb_drop_j2"],
                        "ring": ["allegro_non_thumb_drop_j2_j3"],
                    }
                },
                "output_layout": "recursive",
            }
        )


def test_recipe_loader_builds_mutate_block_into_isaaclab_style_cfg(tmp_path):
    r"""`Mutate.mount_perturb: {...}` 应直接桥接成 cfg 类属性 term。"""

    cfg = RecipeLoader.load_dict(
        {
            "mode": "made",
            "artifact_level": "hand_cfg",
            "output_dir": str(tmp_path / "generated"),
            "Made": _made_recipe_dict(),
            "Mutate": {
                "mount_perturb": {
                    "self_mode": "general",
                    "pos_radius": [0.001, 0.001, 0.001],
                }
            },
        }
    )

    assert isinstance(cfg.Mutate, HandMutatorCfg)
    assert cfg.Mutate.has_terms() is True
    assert [name for name, _ in cfg.Mutate.ordered_terms()] == ["mount_perturb"]
    assert isinstance(cfg.Mutate.mount_perturb, MountPerturbCfg)
    assert cfg.Mutate.mount_perturb.pos_radius == (0.001, 0.001, 0.001)


def test_recipe_loader_rejects_removed_disturb_unit_field_in_mutate_block(tmp_path):
    r"""recipe 层也不再兼容 `disturb_unit`，避免 YAML 与 Python cfg 语义分叉。"""

    with pytest.raises(TypeError, match="disturb_unit"):
        RecipeLoader.load_dict(
            {
                "mode": "made",
                "artifact_level": "hand_cfg",
                "output_dir": str(tmp_path / "generated"),
                "Made": _made_recipe_dict(),
                "Mutate": {
                    "mount_perturb": {
                        "disturb_unit": "rad",
                        "self_mode": "general",
                        "pos_range": [0.001, 0.001],
                    }
                },
            }
        )


def test_recipe_loader_rejects_removed_sample_space_and_legacy_range_fields(tmp_path):
    r"""recipe 层也应一次性拒绝旧 `sample_space/pos_range` 写法。"""

    with pytest.raises(TypeError, match="sample_space"):
        RecipeLoader.load_dict(
            {
                "mode": "made",
                "artifact_level": "hand_cfg",
                "output_dir": str(tmp_path / "generated"),
                "Made": _made_recipe_dict(),
                "Mutate": {
                    "mount_perturb": {
                        "sample_space": {"pos": "ellipsoid", "rot": "ellipsoid"},
                        "self_mode": "general",
                        "pos_radius": [0.001, 0.001, 0.001],
                    }
                },
            }
        )

    with pytest.raises(TypeError, match="pos_range"):
        RecipeLoader.load_dict(
            {
                "mode": "made",
                "artifact_level": "hand_cfg",
                "output_dir": str(tmp_path / "generated"),
                "Made": _made_recipe_dict(),
                "Mutate": {
                    "mount_perturb": {
                        "self_mode": "general",
                        "pos_range": [0.001, 0.001],
                    }
                },
            }
        )
