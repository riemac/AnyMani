r"""Hand asset dataset YAML 的层级展开、partition 与 lineage contract。

测试在 ``tmp_path`` 中复现正式目录：generation run -> groups/mixed -> mother ->
variant set -> variant。这里验证的是数据集声明如何选择已生成资产，不重复测试 generator、
URDF exporter 或 post-mutate 数值算法。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from anymani.assets.bank.dataset import HandAssetDataset


def _write_bundle(path: Path, asset_id: str) -> Path:
    r"""写一个无需 mesh 的最小 right-hand bundle。

    Args:
        path (Path): mother 或 variant bundle 目录。
        asset_id (str): HandBank 暴露的稳定资产 ID。

    Returns:
        Path: 已写入 ``hand.urdf`` 与 ``hand.yaml`` 的 bundle 路径。
    """

    path.mkdir(parents=True, exist_ok=True)  # bundle root 是 HandContainer 的直接入口
    (path / "hand.urdf").write_text('<robot name="fixture"><link name="palm"/></robot>', encoding="utf-8")
    (path / "hand.yaml").write_text(
        yaml.safe_dump(
            {
                "id": asset_id,
                "handedness": "right",
                "topology_name": path.name,
                "hand_cfg": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_generation_run(path: Path) -> Path:
    r"""写 pre-made generation run 的最小 summary。"""

    path.mkdir(parents=True, exist_ok=True)  # generation run 同时容纳 groups 与 mixed
    (path / "summary.yaml").write_text(
        yaml.safe_dump({"run": {"mode": "made", "root_dir": str(path)}}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_variant_set(mother: Path, name: str, asset_ids: tuple[str, ...]) -> Path:
    r"""写一个 source lineage 明确、variant 数量闭合的 post-mutate run。"""

    variant_set = mother / name  # variant set 必须是 mother 的直接子目录
    variant_set.mkdir(parents=True, exist_ok=True)
    for asset_id in asset_ids:
        _write_bundle(variant_set / asset_id, asset_id)  # 每个直接子目录是一项独立 variant
    (variant_set / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"mode": "mutate", "root_dir": str(variant_set)},
                "config": {"source_topology_dir": str(mother)},
                "stats": {"succeeded": len(asset_ids)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return variant_set


def _write_dataset(path: Path, payload: dict) -> Path:
    r"""把测试声明写成与正式 dataset manifest 相同的 YAML。"""

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_dataset_resolves_default_and_overridden_runs_with_named_evaluation_suites(tmp_path: Path) -> None:
    r"""同一 manifest 应稳定展开 groups、多 run 与两种 generated evaluation 关系。"""

    default_run = _write_generation_run(tmp_path / "generated_default")
    secondary_run = _write_generation_run(tmp_path / "generated_secondary")

    # mother A 同时提供 train、validation 与 unseen-variant-set，只有 train 显式包含 mother 本体。
    mother_a = _write_bundle(default_run / "single_palm_leap" / "right_t4_i4_m4_r4", "mother_a")
    _write_variant_set(mother_a, "train_set", ("train_b", "train_a"))
    _write_variant_set(mother_a, "validation_set", ("validation_a",))
    _write_variant_set(mother_a, "evaluation_set", ("evaluation_a",))

    # mother B 位于第二个 generation run，证明 run_dir 只在 run block 边界覆盖。
    mother_b = _write_bundle(secondary_run / "single_palm_allegro" / "right_t3_i3_m3_r3", "mother_b")
    _write_variant_set(mother_b, "train_set", ("train_secondary",))

    # mother C 从未进入 train，因而可作为 unseen_mother suite；mixed 多一层 composition group。
    mother_c = _write_bundle(default_run / "mixed" / "leap_thumb_allegro_ring" / "left_t3_i2_m4_r4", "mother_c")
    _write_variant_set(mother_c, "evaluation_set", ("evaluation_mother",))

    manifest = _write_dataset(
        tmp_path / "dataset.yaml",
        {
            "schema_version": "1.0.0",
            "default_run_dir": str(default_run),
            "train": {
                "runs": {
                    "default": {
                        "groups": {
                            "single_palm_leap": {
                                "right_t4_i4_m4_r4": {
                                    "include_mother": True,
                                    "variant_sets": ["train_set"],
                                }
                            }
                        }
                    },
                    "secondary": {
                        "run_dir": str(secondary_run),
                        "groups": {
                            "single_palm_allegro": {
                                "right_t3_i3_m3_r3": {
                                    "include_mother": True,
                                    "variant_sets": ["train_set"],
                                }
                            }
                        },
                    },
                }
            },
            "validation": {
                "runs": {
                    "default": {
                        "groups": {
                            "single_palm_leap": {
                                "right_t4_i4_m4_r4": {
                                    "include_mother": False,
                                    "variant_sets": ["validation_set"],
                                }
                            }
                        }
                    }
                }
            },
            "evaluation": {
                "unseen_variant_set": {
                    "runs": {
                        "default": {
                            "groups": {
                                "single_palm_leap": {
                                    "right_t4_i4_m4_r4": {
                                        "include_mother": False,
                                        "variant_sets": ["evaluation_set"],
                                    }
                                }
                            }
                        }
                    }
                },
                "unseen_mother": {
                    "runs": {
                        "default": {
                            "mixed": {
                                "leap_thumb_allegro_ring": {
                                    "left_t3_i2_m4_r4": {
                                        "include_mother": True,
                                        "variant_sets": ["evaluation_set"],
                                    }
                                }
                            }
                        }
                    }
                },
                "official_zero_shot": {"assets": []},
            },
        },
    )

    resolved = HandAssetDataset.from_yaml(manifest).resolve()

    assert tuple(record.container.asset_id for record in resolved.train.records) == (
        "mother_a",
        "train_a",
        "train_b",
        "mother_b",
        "train_secondary",
    )
    assert tuple(record.container.asset_id for record in resolved.validation.records) == ("validation_a",)
    assert tuple(record.container.asset_id for record in resolved.evaluation["unseen_variant_set"].records) == (
        "evaluation_a",
    )
    assert tuple(record.container.asset_id for record in resolved.evaluation["unseen_mother"].records) == (
        "mother_c",
        "evaluation_mother",
    )
    assert resolved.train.records[-1].provenance.run_alias == "secondary"
    assert resolved.evaluation["unseen_mother"].records[-1].provenance.collection_kind == "mixed"


def test_dataset_rejects_evaluation_suite_with_wrong_relation_to_train(tmp_path: Path) -> None:
    r"""suite 名称必须由 mother lineage 相对 train 的真实关系支持，而不是任意标签。"""

    run = _write_generation_run(tmp_path / "generated")
    trained_mother = _write_bundle(run / "single_palm_leap" / "right_t4_i4_m4_r4", "mother")
    _write_variant_set(trained_mother, "train_set", ("train",))
    _write_variant_set(trained_mother, "wrong_unseen_mother", ("evaluation",))
    manifest = _write_dataset(
        tmp_path / "wrong_relation.yaml",
        {
            "schema_version": "1.0.0",
            "default_run_dir": str(run),
            "train": {
                "runs": {
                    "default": {
                        "groups": {
                            "single_palm_leap": {
                                "right_t4_i4_m4_r4": {
                                    "include_mother": True,
                                    "variant_sets": ["train_set"],
                                }
                            }
                        }
                    }
                }
            },
            "validation": {"runs": {}},
            "evaluation": {
                "unseen_variant_set": {"runs": {}},
                "unseen_mother": {
                    "runs": {
                        "default": {
                            "groups": {
                                "single_palm_leap": {
                                    "right_t4_i4_m4_r4": {
                                        "include_mother": False,
                                        "variant_sets": ["wrong_unseen_mother"],
                                    }
                                }
                            }
                        }
                    }
                },
                "official_zero_shot": {"assets": []},
            },
        },
    )

    with pytest.raises(ValueError, match="unseen_mother.*already appears in train"):
        HandAssetDataset.from_yaml(manifest).resolve()


def test_dataset_rejects_variant_set_whose_summary_points_to_another_mother(tmp_path: Path) -> None:
    r"""目录位置看似合法但 provenance 指向其他 mother 时必须 fail-closed。"""

    run = _write_generation_run(tmp_path / "generated")
    mother = _write_bundle(run / "single_palm_leap" / "right_t4_i4_m4_r4", "mother")
    other = _write_bundle(run / "single_palm_leap" / "right_t3_i3_m3_r3", "other")
    variant_set = _write_variant_set(mother, "foreign_set", ("variant",))
    summary = yaml.safe_load((variant_set / "summary.yaml").read_text(encoding="utf-8"))
    summary["config"]["source_topology_dir"] = str(other)  # 模拟复制目录后未更新 lineage provenance
    (variant_set / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    manifest = _write_dataset(
        tmp_path / "wrong_source.yaml",
        {
            "schema_version": "1.0.0",
            "default_run_dir": str(run),
            "train": {
                "runs": {
                    "default": {
                        "groups": {
                            "single_palm_leap": {
                                "right_t4_i4_m4_r4": {
                                    "include_mother": True,
                                    "variant_sets": ["foreign_set"],
                                }
                            }
                        }
                    }
                }
            },
            "validation": {"runs": {}},
            "evaluation": {
                "unseen_variant_set": {"runs": {}},
                "unseen_mother": {"runs": {}},
                "official_zero_shot": {"assets": []},
            },
        },
    )

    with pytest.raises(ValueError, match="source_topology_dir.*mother"):
        HandAssetDataset.from_yaml(manifest).resolve()
