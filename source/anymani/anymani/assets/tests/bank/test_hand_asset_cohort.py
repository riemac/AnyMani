r"""Member-level cohort locks preserve parent-manifest identity and local ordering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from anymani.assets.bank.cohort import (
    finalize_hand_asset_cohort_lock,
    load_hand_asset_cohort,
    write_hand_asset_cohort_lock,
)
from anymani.assets.bank.cohort_selection import (
    PURE_LEAP_RIGHT_A64_RECIPE,
    PURE_LEAP_RIGHT_A128_RECIPE,
    LineageDescriptor,
    MutationVariantDescriptor,
    select_diverse_lineages,
    select_diverse_variants,
)
from anymani.assets.bank.dataset import HandAssetDataset


def _write_bundle(path: Path, asset_id: str, *, variant: bool = False) -> Path:
    r"""Write one minimal generated bundle for cohort resolver tests."""

    path.mkdir(parents=True, exist_ok=True)
    (path / "hand.urdf").write_text('<robot name="fixture"><link name="palm"/></robot>', encoding="utf-8")
    sidecar = {"id": asset_id, "handedness": "right", "topology_name": path.parent.name, "hand_cfg": {}}
    if variant:
        sidecar["post_mutate_samples"] = {"fixture": {"resolved_self_mode": "disturb", "value": 0.25}}
    (path / "hand.yaml").write_text(
        yaml.safe_dump(sidecar, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_source_dataset(root: Path, *, mother_id: str, variant_id: str) -> Path:
    r"""Write one two-record train manifest with mother-before-variant ordering."""

    generated = root / "generated"
    generated.mkdir(parents=True)
    (generated / "summary.yaml").write_text(yaml.safe_dump({"run": {"mode": "made"}}), encoding="utf-8")
    mother = _write_bundle(generated / "single_palm_leap" / "right_t4_i4_m4_r4", mother_id)
    variant_set = mother / "variants"
    _write_bundle(variant_set / variant_id, variant_id, variant=True)
    (variant_set / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"mode": "mutate"},
                "config": {"source_topology_dir": str(mother)},
                "stats": {"succeeded": 1},
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0.0",
                "default_run_dir": str(generated),
                "train": {
                    "runs": {
                        "default": {
                            "groups": {
                                "single_palm_leap": {
                                    "right_t4_i4_m4_r4": {
                                        "include_mother": True,
                                        "variant_sets": ["variants"],
                                    }
                                }
                            }
                        }
                    }
                },
                "validation": {"unseen_variant_set": {"runs": {}}, "unseen_mother": {"runs": {}}},
                "evaluation": {
                    "unseen_variant_set": {"runs": {}},
                    "unseen_mother": {"runs": {}},
                    "official_zero_shot": {"assets": []},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_cohort_lock_allows_overlapping_numeric_rows_across_qualified_sources(tmp_path: Path) -> None:
    r"""``ppo#0`` and ``ssl#0`` are distinct coordinates while local indices remain dense."""

    ppo = _write_source_dataset(tmp_path / "ppo", mother_id="ppo-mother", variant_id="ppo-variant")
    ssl = _write_source_dataset(tmp_path / "ssl", mother_id="ssl-mother", variant_id="ssl-variant")
    output = tmp_path / "cohort.yaml"
    write_hand_asset_cohort_lock(
        output,
        cohort_id="fixture",
        source_manifests={"ppo": ppo, "ssl": ssl},
        member_coordinates=(("ppo", 0), ("ssl", 0), ("ppo", 1)),
        selection={"algorithm": "explicit-test", "seed": 7},
        require_geometry_semantics=False,
    )

    cohort = load_hand_asset_cohort(output, require_geometry_semantics=False)
    assert cohort.source_keys == ("ppo#0", "ssl#0", "ppo#1")
    assert tuple(member.cohort_index for member in cohort.members) == (0, 1, 2)
    assert tuple(asset.asset_id for asset in cohort.assets) == ("ppo-mother", "ssl-mother", "ppo-variant")
    assert cohort.selection == {"algorithm": "explicit-test", "seed": 7}


def test_cohort_lock_rejects_member_or_parent_manifest_identity_drift(tmp_path: Path) -> None:
    r"""Asset metadata and parent YAML bytes are both fail-closed lock dependencies."""

    manifest = _write_source_dataset(tmp_path / "source", mother_id="mother", variant_id="variant")
    output = tmp_path / "cohort.yaml"
    write_hand_asset_cohort_lock(
        output,
        cohort_id="fixture",
        source_manifests={"ppo": manifest},
        member_coordinates=(("ppo", 0),),
        selection={"algorithm": "explicit-test"},
        require_geometry_semantics=False,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["members"][0]["asset_id"] = "wrong"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="asset/content identity"):
        load_hand_asset_cohort(output, require_geometry_semantics=False)

    # Republish a valid lock, then mutate only parent manifest bytes; source SHA must fail before bundle use.
    write_hand_asset_cohort_lock(
        output,
        cohort_id="fixture",
        source_manifests={"ppo": manifest},
        member_coordinates=(("ppo", 0),),
        selection={"algorithm": "explicit-test"},
        require_geometry_semantics=False,
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert HandAssetDataset.from_yaml(manifest).source_sha256 != json.loads(output.read_text())["sources"]["ppo"][
        "manifest_sha256"
    ]
    with pytest.raises(ValueError, match="manifest SHA mismatch"):
        load_hand_asset_cohort(output, require_geometry_semantics=False)


def test_canonical_final_lock_binds_source_bytes_and_ordered_physical_hashes(tmp_path: Path) -> None:
    r"""Schema-1.2保留1.1父lock并逐成员冻结canonical physical/schema identity。"""

    manifest = _write_source_dataset(tmp_path / "source", mother_id="mother", variant_id="variant")
    source_lock = tmp_path / "source.lock.yaml"
    write_hand_asset_cohort_lock(
        source_lock,
        cohort_id="fixture",
        source_manifests={"ppo": manifest},
        member_coordinates=(("ppo", 0), ("ppo", 1)),
        selection={"algorithm": "explicit-test"},
        require_geometry_semantics=False,
    )
    source_sha = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    final_lock = tmp_path / "canonical.lock.yaml"
    finalize_hand_asset_cohort_lock(
        source_lock,
        final_lock,
        canonical_identities=(("6" * 64, "8" * 64, "a" * 64), ("7" * 64, "9" * 64, "a" * 64)),
        canonical_schema_version="1.0.0",
        require_geometry_semantics=False,
    )

    resolved = load_hand_asset_cohort(final_lock, require_geometry_semantics=False)
    assert resolved.canonical_binding["source_lock_sha256"] == source_sha
    assert tuple(member.configuration_domain_hash for member in resolved.members) == ("6" * 64, "7" * 64)
    assert tuple(member.physical_geometry_hash for member in resolved.members) == ("8" * 64, "9" * 64)
    assert {member.canonical_schema_digest for member in resolved.members} == {"a" * 64}


def _lineage(key: str, missing: str, dofs: tuple[int, int, int, int], x: float) -> LineageDescriptor:
    r"""构造一个cell内的合成mother descriptor。"""

    digest = f"{int(key):064x}"
    return LineageDescriptor(
        key=key,
        source_alias="ssl",
        mother_row=int(key),
        mother_name=f"right-{key}",
        mother_asset_id=f"asset-{key}",
        mother_content_hash=digest,
        static_geometry_fingerprint=f"{int(key) + 100:064x}",
        cell=(3, 3),
        topology=f"topology-{key}",
        missing_slots=(missing,),
        finger_dofs=dofs,
        descriptor=(x, 0.0),
    )


def test_diverse_lineage_selector_prioritizes_missing_slot_and_dof_coverage() -> None:
    r"""离散topology覆盖优先于仅在连续描述上更远的重复类别。"""

    candidates = (
        _lineage("1", "index", (0, 4, 1, 3), 10.0),
        _lineage("2", "index", (0, 4, 1, 3), -10.0),
        _lineage("3", "ring", (3, 4, 0, 3), 0.1),
    )
    selected = select_diverse_lineages(candidates, quota=2)
    assert len({lineage.missing_slots for lineage in selected}) == 2
    assert {lineage.key for lineage in selected} & {"1", "2"}
    assert "3" in {lineage.key for lineage in selected}


def test_scale_recipe_cardinalities_and_cell_totals_are_exact() -> None:
    r"""A64为16×4；A128为32×4且PPO+SSL四cell均恰有8条mother。"""

    assert PURE_LEAP_RIGHT_A64_RECIPE.mother_count == 16
    assert PURE_LEAP_RIGHT_A64_RECIPE.asset_count == 64
    assert PURE_LEAP_RIGHT_A128_RECIPE.mother_count == 32
    assert PURE_LEAP_RIGHT_A128_RECIPE.asset_count == 128
    combined = tuple(
        sum(quota.for_cell(cell) for quota in PURE_LEAP_RIGHT_A128_RECIPE.source_quotas)
        for cell in ((3, 3), (3, 4), (4, 3), (4, 4))
    )
    assert combined == (8, 8, 8, 8)


def _variant(key: str, modes: tuple[str, ...], tips: tuple[str, ...], x: float) -> MutationVariantDescriptor:
    r"""构造lineage-local mutation/TIP/geometry合成variant。"""

    return MutationVariantDescriptor(
        key=key,
        source_row=int(key),
        asset_id=f"variant-{key}",
        content_hash=f"{int(key):064x}",
        mutation_mode_tokens=modes,
        tip_signature_tokens=tips,
        descriptor=(x, 0.0),
    )


def test_variant_selector_prioritizes_mutation_and_tip_coverage_before_distance() -> None:
    r"""连续几何很远的重复mode不能挤掉提供新mutation/TIP类别的variant。"""

    variants = (
        _variant("1", ("scale:general",), ("index:round",), -10.0),
        _variant("2", ("scale:general",), ("index:round",), 10.0),
        _variant("3", ("scale:only_length",), ("index:cs",), 0.1),
        _variant("4", ("mount:disturb",), ("index:leap_cube",), 0.2),
    )
    selected = select_diverse_variants((0.0, 0.0), variants, count=3)
    assert {variant.key for variant in selected} >= {"3", "4"}
    assert len({token for variant in selected for token in variant.mutation_mode_tokens}) == 3
