"""CPU-only contracts for paper model/data paths and ordered runtime member views."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/anymani"))

from anymani.assets.bank.cohort import (  # noqa: E402
    HandAssetCohortMember,
    HandAssetCohortSource,
    ResolvedHandAssetCohort,
    parse_hand_asset_cohort_document,
)
from anymani.assets.bank.dataset import ResolvedHandAssetPartition, ResolvedHandAssetRecord  # noqa: E402
from anymani.publication.paper_data import PaperAssetBundle  # noqa: E402
from anymani.publication.runtime import (  # noqa: E402
    PaperModelBundle,
    _validate_reference,
    evaluator_arguments_for_policy,
    resolve_student_runtime_artifacts,
    run_module_main,
)
import anymani.publication.runtime as publication_runtime  # noqa: E402
import anymani.distill.rl.rl_games_backend as rl_games_backend  # noqa: E402
from anymani.assets.bank.hand_container import PortableMeshBinding  # noqa: E402

def _module(name: str, relative_path: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {relative_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _member(index: int, source_row: int, asset_id: str) -> HandAssetCohortMember:
    return HandAssetCohortMember(
        cohort_index=index,
        source_alias="source",
        source_row=source_row,
        asset_id=asset_id,
        content_hash=f"content-{asset_id}",
        provenance=SimpleNamespace(group_name="single_palm_leap", mother_name=f"mother-{index}"),
        mutation_descriptor={"kind": "mother"},
        configuration_domain_hash=f"domain-{asset_id}",
        physical_geometry_hash=f"physical-{asset_id}",
        canonical_schema_digest=f"schema-{asset_id}",
    )


def _raw_member(member: HandAssetCohortMember) -> dict:
    return {
        "cohort_index": member.cohort_index,
        "source_alias": member.source_alias,
        "source_row": member.source_row,
        "asset_id": member.asset_id,
        "content_hash": member.content_hash,
        "provenance": {"group_name": "single_palm_leap", "mother_name": f"mother-{member.cohort_index}"},
        "mutation_descriptor": {"kind": "mother"},
        "configuration_domain_hash": member.configuration_domain_hash,
        "physical_geometry_hash": member.physical_geometry_hash,
        "canonical_schema_digest": member.canonical_schema_digest,
    }


def _fake_bundle(tmp_path: Path) -> tuple[PaperAssetBundle, ResolvedHandAssetCohort]:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    members = (_member(0, 101, "hand-a"), _member(1, 207, "hand-b"))
    source_document = {
        "schema_version": "1.2.0",
        "cohort_id": "demo-parent",
        "selection": {
            "purpose": "test-parent",
            "nominal_assets": 2,
            "missing_members": [],
            "pregrasp_generation_identity": {
                "refinement": {"strict_mode_exploitation": {"position_std_m": [5e-05, 5e-05, 2.5e-05]}}
            },
        },
        "sources": {"source": {"manifest_path": "source.yaml", "manifest_sha256": "a" * 64}},
        "canonical_binding": {"schema_digest": "b" * 64},
        "members": [_raw_member(member) for member in members],
    }
    lock_path = assets_root / "parent.lock.yaml"
    lock_bytes = (json.dumps(source_document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    lock_path.write_bytes(lock_bytes)
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    manifest = {
        "artifact_type": "anymani.paper_asset_bundle",
        "schema_version": "1.0.0",
        "cohorts": {
            "demo": {
                "nominal_count": 2,
                "ready_count": 2,
                "members": [{"ready": True}, {"ready": True}],
                "locks": {
                    "runtime_canonical": {"path": "parent.lock.yaml", "published_sha256": lock_sha256}
                },
                "parent_manifests": [],
            }
        },
    }
    (assets_root / "paper_bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = tuple(
        ResolvedHandAssetRecord(
            container=SimpleNamespace(asset_id=member.asset_id),
            provenance=member.provenance,
            content_hash=member.content_hash,
        )
        for member in members
    )
    cohort = ResolvedHandAssetCohort(
        cohort_id="demo-parent",
        lock_path=lock_path,
        lock_sha256=lock_sha256,
        selection=source_document["selection"],
        canonical_binding=source_document["canonical_binding"],
        sources={
            "source": HandAssetCohortSource(
                alias="source", manifest_path=assets_root / "source.yaml", manifest_sha256="a" * 64
            )
        },
        members=members,
        partition=ResolvedHandAssetPartition(name="demo", records=records),
    )
    bundle = PaperAssetBundle(assets_root)
    bundle.resolve = lambda name, ready_only=True: cohort
    return bundle, cohort


def test_runtime_member_view_keeps_parent_coordinates_and_local_order(tmp_path: Path) -> None:
    bundle, parent = _fake_bundle(tmp_path)
    view = bundle.create_view("demo", (1, 0), lock_dir=tmp_path / "views")
    reopened = bundle.resolve_view(view.lock_path)

    assert view.parent_member_indices == (1, 0)
    assert view.selected_source_keys == ("source#207", "source#101")
    assert view.cohort.source_keys == view.selected_source_keys
    assert tuple(member.cohort_index for member in view.cohort.members) == (0, 1)
    assert tuple(member.source_row for member in view.cohort.members) == (207, 101)
    assert view.parent_lock_sha256 == parent.lock_sha256
    assert view.cohort.lock_sha256 == hashlib.sha256(view.lock_path.read_bytes()).hexdigest()
    assert reopened.view_sha256 == view.view_sha256
    assert reopened.lock_sha256 == view.lock_sha256
    values = reopened.cohort.selection["pregrasp_generation_identity"]["refinement"][
        "strict_mode_exploitation"
    ]["position_std_m"]
    assert values == [5e-05, 5e-05, 2.5e-05]
    assert all(type(value) is float for value in values)


def test_runtime_member_view_rejects_member_mutation(tmp_path: Path) -> None:
    bundle, _ = _fake_bundle(tmp_path)
    view = bundle.create_view("demo", (0,), lock_dir=tmp_path / "views")
    document = yaml.safe_load(view.lock_path.read_text(encoding="utf-8"))
    document["members"][0]["source_row"] += 1
    damaged = tmp_path / "damaged.lock.yaml"
    damaged.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Runtime view digest mismatch|mutated the source member|parent selection provenance was changed",
    ):
        bundle.resolve_view(damaged)


def test_model_bundle_checks_paths_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    actor = root / "student.ts"
    actor.write_bytes(b"frozen actor bytes")
    digest = hashlib.sha256(actor.read_bytes()).hexdigest()
    (root / "model_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "anymani.paper_model_bundle",
                "schema_version": "1.0.0",
                "models": {"student_torchscript": {"path": "student.ts", "sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    (root / ".bundle.json").write_text("{}", encoding="utf-8")
    bundle = PaperModelBundle.open(root)
    assert bundle.path("student_torchscript") == actor

    actor.write_bytes(b"modified actor bytes")
    with pytest.raises(ValueError, match="Model checksum mismatch"):
        bundle.path("student_torchscript")


def test_frozen_reference_is_30_second_r16_protocol() -> None:
    from anymani.publication.runtime import DEFAULT_REFERENCE_DOCUMENT

    _validate_reference(DEFAULT_REFERENCE_DOCUMENT)
    assert DEFAULT_REFERENCE_DOCUMENT["completed_steps"] == 600
    assert DEFAULT_REFERENCE_DOCUMENT["num_replicas"] == 16
    assert DEFAULT_REFERENCE_DOCUMENT["protocol"]["horizon_s"] == 30.0


def test_runtime_import_uses_this_public_checkout() -> None:
    assert Path(publication_runtime.__file__).resolve().is_relative_to(ROOT / "source/anymani")


def test_student_override_pair_uses_adjacent_sidecar(tmp_path: Path) -> None:
    actor = tmp_path / "best.pt"
    torchscript = tmp_path / "actor.ts"
    sidecar = Path(f"{torchscript}.json")
    for path in (actor, torchscript, sidecar):
        path.write_bytes(b"artifact")

    resolved = resolve_student_runtime_artifacts(
        SimpleNamespace(), checkpoint=actor, torchscript=torchscript
    )
    assert resolved.checkpoint == actor.resolve()
    assert resolved.torchscript == torchscript.resolve()
    assert resolved.sidecar == sidecar.resolve()
    with pytest.raises(ValueError, match="must be supplied together"):
        resolve_student_runtime_artifacts(SimpleNamespace(), checkpoint=actor)


def test_student_evaluator_does_not_receive_teacher_checkpoint_flag() -> None:
    common = ("--cohort_lock", "/tmp/cohort.lock", "--output", "/tmp/eval.json")
    assert "--checkpoint" not in evaluator_arguments_for_policy("student", Path("teacher.pth"), common)
    teacher_args = evaluator_arguments_for_policy("teacher", Path("teacher.pth"), common)
    assert teacher_args[:2] == ("--checkpoint", "teacher.pth")


def test_runtime_module_runner_uses_child_exit_and_checkout_first(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "publication_probe.py"
    output = tmp_path / "module-path.txt"
    module.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROBE_OUTPUT']).write_text(str(Path(__file__).resolve()))\n"
        "raise SystemExit(int(os.environ.get('PROBE_EXIT', '0')))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PROBE_OUTPUT", str(output))

    assert run_module_main("publication_probe", ()) == 0
    assert Path(output.read_text(encoding="utf-8")).resolve() == module.resolve()
    monkeypatch.setenv("PROBE_EXIT", "7")
    assert run_module_main("publication_probe", ()) == 7


def test_rl_games_backend_does_not_claim_a_parent_git_checkout(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "AnyMani" / ".venv" / "lib" / "python3.11" / "site-packages" / "rl_games"
    nested.mkdir(parents=True)
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(tuple(arguments))
        return SimpleNamespace(stdout=f"{tmp_path / 'AnyMani'}\n")

    monkeypatch.setattr(rl_games_backend.subprocess, "run", fake_run)
    assert rl_games_backend._git_commit(nested.parent) is None
    assert calls == [("git", "rev-parse", "--show-toplevel")]


def test_rl_games_backend_reads_vcs_revision_from_distribution_metadata(monkeypatch) -> None:
    commit = "36edd38823197e6e20c6cc4531765e654d13b80f"

    class Distribution:
        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps({"vcs_info": {"vcs": "git", "commit_id": commit}})

    monkeypatch.setattr(rl_games_backend.metadata, "distribution", lambda name: Distribution())
    assert rl_games_backend._direct_url_commit() == commit


def test_paper_bundle_resolves_old_mesh_locators_to_verified_member_uris(tmp_path: Path) -> None:
    bundle_root = tmp_path / "assets"
    asset_root = bundle_root / "source_trees/tree-a/hand-a"
    (asset_root / "meshes").mkdir(parents=True)
    mesh_bytes = b"verified mesh payload"
    mesh_path = asset_root / "meshes/cap.obj"
    mesh_path.write_bytes(mesh_bytes)
    mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
    sidecar_path = asset_root / "hand.yaml"
    locator = "/old/research/assets/generated/run/hand-a/meshes/cap.obj"
    sidecar_path.write_text(
        yaml.safe_dump(
            {
                "geometry_semantics": {
                    "components": [
                        {"geometry_kind": "mesh", "geometry_payload": {"file_path": locator}}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    bundle = PaperAssetBundle.__new__(PaperAssetBundle)
    bundle.root = bundle_root.resolve()
    bundle._verified = {}
    member = {
        "asset_id": "hand-a",
        "files": [{"path": "source_trees/tree-a/hand-a/hand.yaml", "published_sha256": sidecar_sha}],
        "mesh_dependencies": [
            {
                "uri": "../meshes/cap.obj",
                "path": "source_trees/tree-a/hand-a/meshes/cap.obj",
                "published_sha256": mesh_sha,
            }
        ],
    }

    bindings = bundle._portable_mesh_bindings(member)

    assert bindings == {
        locator: PortableMeshBinding(
            uri="../meshes/cap.obj", sha256=mesh_sha, path=mesh_path.resolve()
        )
    }
    assert mesh_path.read_bytes() == mesh_bytes


def test_paper_bundle_rejects_ambiguous_custom_tip_mesh_basename(tmp_path: Path) -> None:
    bundle_root = tmp_path / "assets"
    asset_root = bundle_root / "source_trees/tree-a/hand-a"
    (asset_root / "meshes").mkdir(parents=True)
    (asset_root / "custom/tips").mkdir(parents=True)
    for path in (asset_root / "meshes/cap.obj", asset_root / "custom/tips/cap.obj"):
        path.write_bytes(path.name.encode("utf-8"))
    locator = "/old/research/assets/custom/tips/cap.obj"
    sidecar_path = asset_root / "hand.yaml"
    sidecar_path.write_text(
        yaml.safe_dump(
            {"geometry_semantics": {"components": [{"geometry_kind": "mesh", "geometry_payload": {"file_path": locator}}]}}
        ),
        encoding="utf-8",
    )
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    specs = []
    for uri, relative_path in (
        ("../meshes/cap.obj", "meshes/cap.obj"),
        ("custom/tips/cap.obj", "custom/tips/cap.obj"),
    ):
        path = asset_root / relative_path
        specs.append({
            "uri": uri,
            "path": f"source_trees/tree-a/hand-a/{relative_path}",
            "published_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    bundle = PaperAssetBundle.__new__(PaperAssetBundle)
    bundle.root = bundle_root.resolve()
    bundle._verified = {}
    member = {
        "asset_id": "hand-a",
        "files": [{"path": "source_trees/tree-a/hand-a/hand.yaml", "published_sha256": sidecar_sha}],
        "mesh_dependencies": specs,
    }

    with pytest.raises(ValueError, match="matched 2 packaged dependencies"):
        bundle._portable_mesh_bindings(member)


def test_allegro_json_lock_with_yaml_suffix_preserves_scientific_float_types() -> None:
    lock_bytes = (ROOT / "tests/fixtures/allegro_training_runtime_lock_excerpt.yaml").read_bytes()
    yaml_values = yaml.safe_load(lock_bytes)["selection"]["pregrasp_generation_identity"]["refinement"][
        "strict_mode_exploitation"
    ]["position_std_m"]
    parsed = parse_hand_asset_cohort_document(lock_bytes)
    position_std_m = parsed["selection"]["pregrasp_generation_identity"]["refinement"][
        "strict_mode_exploitation"
    ]["position_std_m"]

    assert [type(value).__name__ for value in yaml_values] == ["str", "str", "float"]
    assert position_std_m == [5e-05, 5e-05, 2.5e-05]
    assert all(type(value) is float for value in position_std_m)


def test_cli_defaults_and_population_mapping_are_paper_scoped() -> None:
    collect = _module("paper_collect_cli", "scripts/collect.py")
    evaluate = _module("paper_evaluate_cli", "scripts/evaluate.py")
    replay = _module("paper_replay_cli", "scripts/replay.py")

    collect_args = collect.build_parser().parse_args([])
    assert collect_args.family == "all" and collect_args.mode == "all"
    assert collect._jobs("all", "all", smoke=False) == (
        ("leap", "mean"), ("leap", "sample"), ("allegro", "mean"), ("allegro", "sample")
    )
    assert collect._jobs("all", "all", smoke=True) == (("leap", "mean"),)
    assert collect._child_status_code(0, False) == 1
    assert collect._child_status_code(0, True) == 0
    assert collect._child_status_code(3, False) == 3
    assert evaluate._select_cohorts("unseen", None, "all") == (
        "leap_right_variant", "leap_right_mother", "allegro_right_variant", "allegro_right_mother"
    )
    replay_args = replay.build_parser().parse_args([])
    assert replay_args.policy == "student" and replay_args.cohort == "leap_training"
    assert replay_args.video_preset == "standard"
    recording_args = replay.build_parser().parse_args(
        [
            "--record-video",
            "--video-preset", "paper_white",
            "--video-resolution", "1920", "1080",
            "--video-eye", "0.3", "-0.4", "0.5",
            "--video-lookat", "0.0", "0.0", "0.4",
        ]
    )
    assert recording_args.video_resolution == [1920, 1080]
    assert recording_args.video_eye == [0.3, -0.4, 0.5]
    assert replay_args.asset_index == 0
