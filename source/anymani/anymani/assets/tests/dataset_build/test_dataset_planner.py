r"""Dataset build template、mirror-pair inventory 与分层 selection planner 合同。"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
import yaml
from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.config import asset_gen_cfg as asset_cfg_module
from anymani.assets.dataset_build.builder import build_dataset_from_lock, derive_ppo_manifest_from_lock
from anymani.assets.dataset_build.planner import build_dataset_selection_plan, write_selection_lock
from anymani.assets.dataset_build.schema import load_dataset_build_template
from anymani.assets.generator.hand_generator import PostMutateVariantSetResult
from anymani.assets.generator.runtime.recipe_loader import RecipeLoader


def test_planner_selects_disjoint_pairs_and_derives_exact_variant_budget(tmp_path: Path) -> None:
    r"""小型 inventory 应复现正式五通道关系、pair 隔离和总资产语义。

    fixture 含四个 macro family、每类六个 canonical pairs；每类四个 full、两个
    missing。模板最终选择 16 train mothers、两条 validation 各 8、两条 evaluation
    各 8，并把两个 seen cohorts 的并集作为 16-mother PPO train。
    """

    run_root = _write_inventory(tmp_path / "generated")
    template_path = _write_template(tmp_path / "template.yaml", run_root=run_root)
    template, template_sha = load_dataset_build_template(template_path)

    first = build_dataset_selection_plan(template, template_sha256=template_sha)
    second = build_dataset_selection_plan(template, template_sha256=template_sha)

    assert first.to_lock_document() == second.to_lock_document()
    assert {role: len(pairs) * 2 for role, pairs in first.pairs.items()} == {
        "train": 16,
        "validation.unseen_variant_set": 8,
        "validation.unseen_mother": 8,
        "evaluation.unseen_variant_set": 8,
        "evaluation.unseen_mother": 8,
    }

    train = {pair.pair_key for pair in first.pairs["train"]}
    validation_seen = {pair.pair_key for pair in first.pairs["validation.unseen_variant_set"]}
    validation_unseen = {pair.pair_key for pair in first.pairs["validation.unseen_mother"]}
    evaluation_seen = {pair.pair_key for pair in first.pairs["evaluation.unseen_variant_set"]}
    evaluation_unseen = {pair.pair_key for pair in first.pairs["evaluation.unseen_mother"]}
    assert validation_seen <= train
    assert evaluation_seen <= train
    assert validation_seen.isdisjoint(evaluation_seen)
    assert set(first.ppo_train_pair_keys) == validation_seen | evaluation_seen
    assert train.isdisjoint(validation_unseen | evaluation_unseen)
    assert validation_unseen.isdisjoint(evaluation_unseen)

    # 每个 role 都以 pair 为原子，因而 four macro family 在本 fixture 中严格等额。
    for selected in first.pairs.values():
        assert Counter(pair.macro_family for pair in selected) == {
            "single_allegro": len(selected) // 4,
            "single_leap": len(selected) // 4,
            "mixed_allegro_base": len(selected) // 4,
            "mixed_leap_base": len(selected) // 4,
        }
        assert all(pair.left.handedness == "left" and pair.right.handedness == "right" for pair in selected)

    assert first.quota_report["planned_variants"] == 128
    assert first.quota_report["planned_assets"] == 160
    assert all(len({lineage.mutation_seed for lineage in role}) == len(role) for role in first.lineages.values())

    lock_path = tmp_path / "selection.lock.yaml"
    assert write_selection_lock(first, lock_path) == lock_path
    assert write_selection_lock(first, lock_path) == lock_path
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert lock["schema_version"] == "1.0.0"
    assert lock["quota_report"]["planned_variants"] == 128


def test_template_rejects_unknown_fields_and_odd_mother_counts(tmp_path: Path) -> None:
    r"""拼写错误与拆开 mirror pair 的数量必须在扫描 inventory 前失败。"""

    run_root = _write_inventory(tmp_path / "generated")
    template_path = _write_template(tmp_path / "template.yaml", run_root=run_root)
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    payload["partitions"]["train"]["mother_count"] = 15
    payload["balance"]["unknown_ratio"] = 1
    template_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        load_dataset_build_template(template_path)

    payload["balance"].pop("unknown_ratio")
    template_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="positive even"):
        load_dataset_build_template(template_path)


def test_builder_publishes_manifests_and_resume_skips_completed_tasks(tmp_path: Path) -> None:
    r"""全部 task exact-complete 后才发布 manifests，第二次运行应纯 resume。"""

    run_root = _write_inventory(tmp_path / "generated")
    template_path = _write_template(tmp_path / "dataset" / "template.yaml", run_root=run_root)
    template, template_sha = load_dataset_build_template(template_path)
    post_cfg = asset_cfg_module.POST_MUTATE_CFG
    plan = build_dataset_selection_plan(
        template,
        template_sha256=template_sha,
        generator_config_module="anymani.assets.config.asset_gen_cfg",
        generator_config_snapshot=RecipeLoader.dump(post_cfg),
    )
    lock_path = write_selection_lock(plan, template_path.parent / "selection.lock.yaml")
    calls: list[tuple[str, ...]] = []

    def fake_run_batch(_cfg, tasks, _workers):
        calls.append(tuple(task.task_id for task in tasks))
        return tuple(_write_fake_variant_set(task) for task in tasks)

    report = build_dataset_from_lock(
        template,
        template_sha256=template_sha,
        lock_path=lock_path,
        post_mutate_cfg=post_cfg,
        workers=2,
        run_batch=fake_run_batch,
    )
    first_call_count = len(calls)

    assert report["published"] is True
    assert report["status_counts"] == {"completed": 48}
    assert (template_path.parent / "ssl.yaml").is_file()
    assert (template_path.parent / "ppo.yaml").is_file()
    ssl = yaml.safe_load((template_path.parent / "ssl.yaml").read_text(encoding="utf-8"))
    ppo = yaml.safe_load((template_path.parent / "ppo.yaml").read_text(encoding="utf-8"))
    assert ssl["schema_version"] == "2.0.0"
    assert len(_manifest_mothers(ssl["train"])) == 16
    assert len(_manifest_mothers(ppo["train"])) == 16
    resolved = HandAssetDataset.from_yaml(template_path.parent / "ssl.yaml").resolve(
        allow_legacy_left_handedness=True
    )
    assert len(resolved.train.records) == 64
    assert {name: len(partition.records) for name, partition in resolved.validation.items()} == {
        "unseen_variant_set": 16,
        "unseen_mother": 16,
    }
    assert len(resolved.evaluation["unseen_variant_set"].records) == 32
    assert len(resolved.evaluation["unseen_mother"].records) == 32

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    state = yaml.safe_load((template_path.parent / ".build_state.yaml").read_text(encoding="utf-8"))
    derived = derive_ppo_manifest_from_lock(
        template,
        lock=lock,
        state=state,
        mother_count=8,
        selection_seed=99,
        reuse_ssl_holdouts=False,
    )
    assert len(_manifest_mothers(derived["train"])) == 8
    with pytest.raises(ValueError, match="smaller than reused SSL"):
        derive_ppo_manifest_from_lock(
            template,
            lock=lock,
            state=state,
            mother_count=8,
            selection_seed=99,
            reuse_ssl_holdouts=True,
        )

    resumed = build_dataset_from_lock(
        template,
        template_sha256=template_sha,
        lock_path=lock_path,
        post_mutate_cfg=post_cfg,
        workers=2,
        run_batch=fake_run_batch,
    )
    assert resumed["published"] is True
    assert len(calls) == first_call_count


def test_builder_quarantines_shortfall_then_retries_with_new_seed(tmp_path: Path) -> None:
    r"""单个 mother shortfall 不阻塞同阶段其它任务，并以 retry seed 精确补建。"""

    run_root = _write_inventory(tmp_path / "generated")
    template_path = _write_template(tmp_path / "dataset" / "template.yaml", run_root=run_root)
    template, template_sha = load_dataset_build_template(template_path)
    post_cfg = asset_cfg_module.POST_MUTATE_CFG
    plan = build_dataset_selection_plan(
        template,
        template_sha256=template_sha,
        generator_config_module="anymani.assets.config.asset_gen_cfg",
        generator_config_snapshot=RecipeLoader.dump(post_cfg),
    )
    lock_path = write_selection_lock(plan, template_path.parent / "selection.lock.yaml")
    failed_once: set[str] = set()
    quarantined_runs: list[Path] = []

    def fake_run_batch(_cfg, tasks, _workers):
        reports = []
        for task in tasks:
            if not failed_once:
                failed_once.add(task.task_id)
                report = _write_fake_variant_set(task, successful_variants=task.n_samples - 1)
                quarantined_runs.append(report.run_dir)
            else:
                report = _write_fake_variant_set(task)
            reports.append(report)
        return tuple(reports)

    report = build_dataset_from_lock(
        template,
        template_sha256=template_sha,
        lock_path=lock_path,
        post_mutate_cfg=post_cfg,
        workers=2,
        run_batch=fake_run_batch,
    )
    retried_task = next(iter(failed_once))

    assert report["published"] is True
    assert len(report["tasks"][retried_task]["attempts"]) == 2
    assert report["tasks"][retried_task]["attempts"][0]["status"] == "quarantined"
    assert report["tasks"][retried_task]["attempts"][1]["status"] == "accepted"
    assert (quarantined_runs[0] / "QUARANTINED.yaml").is_file()

def _write_inventory(run_root: Path) -> Path:
    r"""写 24 个 canonical pairs、48 个 mother bundles 的受控 inventory。"""

    run_root.mkdir(parents=True)
    count = 0
    definitions = (
        ("single_allegro", "single_family", "allegro"),
        ("single_leap", "single_family", "leap"),
        ("mixed_allegro_base", "mixed", "allegro"),
        ("mixed_leap_base", "mixed", "leap"),
    )
    for macro_index, (macro, composition, family) in enumerate(definitions):
        for pair_index in range(6):
            missing = pair_index >= 4
            missing_slot = ("index", "middle")[pair_index - 4] if missing else None
            group_name = (
                f"{macro}_composition_{pair_index % 3}"
                if composition == "mixed"
                else f"single_palm_{family}"
            )
            canonical_name = f"t4_i{2 + pair_index % 3}_m{2 + (pair_index + 1) % 3}_r{2 + (pair_index + 2) % 3}"
            if missing_slot == "index":
                canonical_name = f"t4_m{2 + pair_index % 3}_r{2 + (pair_index + 1) % 3}"
            elif missing_slot == "middle":
                canonical_name = f"t4_i{2 + pair_index % 3}_r{2 + (pair_index + 1) % 3}"
            canonical_name = f"{canonical_name}_v{pair_index}"
            for handedness in ("left", "right"):
                mother_name = f"{handedness}_{canonical_name}"
                mother_root = (
                    run_root / "mixed" / group_name / mother_name
                    if composition == "mixed"
                    else run_root / group_name / mother_name
                )
                mother_root.mkdir(parents=True)
                slot_family_map = {"thumb": family}
                surviving = [slot for slot in ("index", "middle", "ring") if slot != missing_slot]
                for slot_index, slot in enumerate(surviving):
                    slot_family_map[slot] = (
                        family
                        if composition == "single_family" or slot_index % 2 == 0
                        else ("leap" if family == "allegro" else "allegro")
                    )
                (mother_root / "hand.urdf").write_text('<robot name="fixture"/>', encoding="utf-8")
                (mother_root / "hand.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "id": f"{macro}-{pair_index}-{handedness}",
                            "family": family,
                            "handedness": handedness,
                            "dof": 8 + pair_index % 3,
                            "finger_count": 3 if missing else 4,
                            "family_composition": composition,
                            "missing_slots": [missing_slot] if missing_slot else [],
                            "slot_family_map": slot_family_map,
                            "selected_slot_recipes": {
                                slot: f"{slot_family}_recipe_{pair_index % 2}"
                                for slot, slot_family in slot_family_map.items()
                            },
                            "geometry_semantics": _geometry_semantics(
                                asset_id=f"{macro}-{pair_index}-{handedness}",
                                handedness=handedness,
                                size=0.01 + (macro_index * 6 + pair_index) * 1.0e-5,
                            ),
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                count += 1
    (run_root / "summary.yaml").write_text(
        yaml.safe_dump(
            {"run": {"mode": "made", "root_dir": str(run_root)}, "stats": {"succeeded": count}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_root


def _geometry_semantics(*, asset_id: str, handedness: str, size: float) -> dict:
    r"""提供 build-time fingerprint 所需的最小静态几何 payload。"""

    return {
        "schema_version": "1.0.0",
        "migration_version": "fixture",
        "source_kind": "generated",
        "asset_id": asset_id,
        "asset_name": asset_id,
        "topology_key": asset_id,
        "family": "fixture",
        "handedness": handedness,
        "units": {"length": "m", "angle": "rad"},
        "asset_to_hand_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "asset_to_hand_translation_m": [0.0, 0.0, 0.0],
        "palm_link": "palm",
        "palm_origin_pos_m": [0.0, 0.0, 0.0],
        "palm_origin_rpy_rad": [0.0, 0.0, 0.0],
        "kinematic_joints": [],
        "active_joint_names": [],
        "q_home_rad": [],
        "joint_limits_rad": [],
        "owners": [],
        "components": [
            {
                "component_id": "palm/collision/0",
                "owner_id": "palm",
                "carrier_link": "palm",
                "collision_index": 0,
                "collision_name": "palm",
                "geometry_kind": "box",
                "geometry_payload": {"type": "box", "size": [size, size, size]},
                "origin_pos_m": [0.0, 0.0, 0.0],
                "origin_rpy_rad": [0.0, 0.0, 0.0],
                "source_joint_name": None,
            }
        ],
        "anchor_seeds": [],
        "content_hash": hashlib.sha256(asset_id.encode()).hexdigest(),
    }


def _write_fake_variant_set(task, *, successful_variants: int | None = None) -> PostMutateVariantSetResult:
    r"""按 source task 写 exact-count fake run，隔离 builder 与真实 mutator 数值。"""

    source = Path(task.source_topology_dir)
    run_dir = source / f"fake_{task.seed}"
    run_dir.mkdir(parents=True)
    source_doc = yaml.safe_load((source / "hand.yaml").read_text(encoding="utf-8"))
    handedness = str(source_doc["handedness"])
    sidecars: list[Path] = []
    urdfs: list[Path] = []
    succeeded = task.n_samples if successful_variants is None else successful_variants
    for index in range(succeeded):
        asset_id = hashlib.sha256(f"{task.task_id}:{task.seed}:{index}".encode()).hexdigest()[:8]
        sample_dir = run_dir / asset_id
        sample_dir.mkdir()
        urdf = sample_dir / "hand.urdf"
        sidecar = sample_dir / "hand.yaml"
        urdf.write_text('<robot name="variant"/>', encoding="utf-8")
        sidecar.write_text(
            yaml.safe_dump(
                {
                    "id": asset_id,
                    "handedness": handedness,
                    "geometry_semantics": _geometry_semantics(
                        asset_id=asset_id,
                        handedness=handedness,
                        size=0.02 + int(asset_id, 16) * 1.0e-12,
                    ),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        sidecars.append(sidecar)
        urdfs.append(urdf)
    (run_dir / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"mode": "mutate", "root_dir": str(run_dir)},
                "config": {
                    "source_topology_dir": str(source),
                    "post_mutate_seed": task.seed,
                    "post_mutate_sources": [],
                },
                "stats": {"succeeded": succeeded},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return PostMutateVariantSetResult(
        task_id=task.task_id,
        source_topology_dir=source,
        run_dir=run_dir,
        planned_variants=task.n_samples,
        successful_variants=succeeded,
        shortfall=task.n_samples - succeeded,
        mutation_seed=task.seed,
        sidecar_paths=tuple(sidecars),
        urdf_paths=tuple(urdfs),
    )


def _manifest_mothers(partition: dict) -> set[tuple[str, str, str]]:
    r"""返回 manifest partition 中 collection/group/mother identity 集。"""

    result: set[tuple[str, str, str]] = set()
    for run in partition["runs"].values():
        for collection in ("groups", "mixed"):
            for group, mothers in run.get(collection, {}).items():
                result.update((collection, group, mother) for mother in mothers)
    return result


def _write_template(path: Path, *, run_root: Path) -> Path:
    r"""写与正式模板同 shape、但数量缩小的测试模板。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "template_id": "fixture_balanced_v1",
                "inventory": {"run_dir": str(run_root)},
                "seeds": {"selection": 20260819, "mutation": 20260820},
                "balance": {
                    "selection_unit": "canonical_mirror_pair",
                    "macro_family": {
                        "single_allegro": 1,
                        "single_leap": 1,
                        "mixed_allegro_base": 1,
                        "mixed_leap_base": 1,
                    },
                    "topology_shape": {"full": 2, "missing": 1},
                    "missing_slot": "uniform",
                    "mixed_composition_group": "uniform",
                    "dof": "uniform_available",
                },
                "partitions": {
                    "train": {"mother_count": 16, "assets_per_lineage": 4},
                    "validation": {
                        "unseen_variant_set": {"mother_count": 8, "assets_per_lineage": 2},
                        "unseen_mother": {"mother_count": 8, "assets_per_lineage": 2},
                    },
                    "evaluation": {
                        "unseen_variant_set": {"mother_count": 8, "assets_per_lineage": 4},
                        "unseen_mother": {"mother_count": 8, "assets_per_lineage": 4},
                        "official_zero_shot": [],
                    },
                },
                "generation_policy": {
                    "dataset_retry_rounds": 3,
                    "uniqueness": "resample",
                    "failed_run_policy": "quarantine",
                },
                "manifests": {
                    "ssl": {"enabled": True},
                    "ppo": {"enabled": True, "train_mother_count": 16, "reuse_ssl_holdouts": True},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
