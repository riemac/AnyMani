r"""Member-level cohort locks preserve parent-manifest identity and local ordering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from anymani.assets.bank import cohort as cohort_module
from anymani.assets.bank import cohort_selection
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
    PureLeapRightLineageRecipe,
    SourceCellMotherQuota,
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
    assert (
        HandAssetDataset.from_yaml(manifest).source_sha256
        != json.loads(output.read_text())["sources"]["ppo"]["manifest_sha256"]
    )
    with pytest.raises(ValueError, match="manifest SHA mismatch"):
        load_hand_asset_cohort(output, require_geometry_semantics=False)


def test_cohort_json_round_trip_preserves_scientific_notation_numbers(tmp_path: Path) -> None:
    r"""canonical JSON中的5e-05必须仍为float，不能被YAML1.1解析成字符串而改变协议摘要。"""

    manifest = _write_source_dataset(tmp_path / "scientific", mother_id="mother", variant_id="variant")
    path = tmp_path / "numeric.lock.yaml"  # 后缀不决定内容语法；正式writer实际写canonical JSON
    selection = {"position_std_m": [5e-5, 5e-5, 2.5e-5], "literal": "5e-05"}
    write_hand_asset_cohort_lock(path, cohort_id="numeric-contract", source_manifests={"one": manifest},
                                 member_coordinates=(("one", 0),), selection=selection,
                                 require_geometry_semantics=False)
    loaded = load_hand_asset_cohort(path, require_geometry_semantics=False)
    assert loaded.selection == selection  # quoted string仍为string；不靠强制float转换“修复”协议
    assert all(isinstance(x, float) for x in loaded.selection["position_std_m"])

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

    # 同名canonical锁是已发布的物理身份；再次finalize不能原地替换其证据。
    published_bytes = final_lock.read_bytes()  # 独立保留原始字节，不用重新序列化比较
    with pytest.raises(FileExistsError):
        finalize_hand_asset_cohort_lock(
            source_lock,
            final_lock,
            canonical_identities=(("6" * 64, "8" * 64, "a" * 64), ("7" * 64, "9" * 64, "a" * 64)),
            canonical_schema_version="1.0.0",
            require_geometry_semantics=False,
        )
    assert final_lock.read_bytes() == published_bytes


def test_canonical_subset_reuses_verified_parent_without_resolving_full_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""生成分片继承已验证父集合的成员与物理证书，发布时不重复解析完整训练库。"""

    manifest = _write_source_dataset(tmp_path / "source", mother_id="mother", variant_id="variant")
    source = tmp_path / "parent.lock.yaml"
    canonical = tmp_path / "parent.canonical.lock.yaml"
    write_hand_asset_cohort_lock(
        source,
        cohort_id="parent",
        source_manifests={"ppo": manifest},
        member_coordinates=(("ppo", 0), ("ppo", 1)),
        selection={"algorithm": "fixture"},
        require_geometry_semantics=False,
    )
    finalize_hand_asset_cohort_lock(
        source,
        canonical,
        canonical_identities=(("6" * 64, "8" * 64, "a" * 64), ("7" * 64, "9" * 64, "a" * 64)),
        canonical_schema_version="v1",
        require_geometry_semantics=False,
    )
    parent = load_hand_asset_cohort(canonical, require_geometry_semantics=False)
    subset_source = tmp_path / "subset.lock.yaml"
    subset_canonical = tmp_path / "subset.canonical.lock.yaml"

    # 完整源解析在父集合加载时已发生。子集合只重新编号，不允许再次进入昂贵的全库解析。
    with monkeypatch.context() as patch:

        def forbidden_resolution(*_args, **_kwargs):
            raise AssertionError("subset publication must not resolve the complete source partition")

        patch.setattr(cohort_module, "resolve_prepared_train", forbidden_resolution)
        cohort_module.write_hand_asset_cohort_subset(
            parent,
            subset_source,
            subset_canonical,
            cohort_id="subset",
            member_indices=(1,),
            selection={"purpose": "pregrasp-generation-only"},
        )

    # 独立consumer仍走完整验证，确认轻量发布没有放松source或canonical合同。
    child = load_hand_asset_cohort(subset_canonical, require_geometry_semantics=False)
    assert child.source_keys == ("ppo#1",)
    assert child.members[0] == replace(parent.members[1], cohort_index=0)
    assert child.selection["parent_asset_indices"] == [1]
    assert child.selection["parent_cohort_lock_sha256"] == parent.lock_sha256
    assert child.canonical_binding["source_lock_sha256"] == hashlib.sha256(subset_source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        cohort_module.write_hand_asset_cohort_subset(
            parent, subset_source, subset_canonical, cohort_id="subset", member_indices=(1,), selection={}
        )
    for invalid in ((), (0, 0), (-1,), (2,)):
        with pytest.raises(ValueError, match="subset indices"):
            cohort_module.write_hand_asset_cohort_subset(
                parent,
                tmp_path / "invalid.lock.yaml",
                tmp_path / "invalid.canonical.lock.yaml",
                cohort_id="invalid",
                member_indices=invalid,
                selection={},
            )


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


def _canonical_parent(root: Path, alias: str, physical_start: int):
    r"""为跨父集合发布构造两个成员的已验证canonical fixture。"""

    manifest = _write_source_dataset(root, mother_id=f"{root.name}-mother", variant_id=f"{root.name}-variant")
    # 多父合并测试需要明确的源内容身份；旧最小fixture保留无content hash的legacy覆盖。
    mother_dir = root / "generated/single_palm_leap/right_t4_i4_m4_r4"
    for directory in (mother_dir, mother_dir / "variants" / f"{root.name}-variant"):
        sidecar_path = directory / "hand.yaml"
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["geometry_semantics"] = {"content_hash": hashlib.sha256(sidecar["id"].encode()).hexdigest()}
        sidecar_path.write_text(yaml.safe_dump(sidecar))
    source, final = root / "source.lock.yaml", root / "canonical.lock.yaml"
    write_hand_asset_cohort_lock(
        source,
        cohort_id=root.name,
        source_manifests={alias: manifest},
        member_coordinates=((alias, 0), (alias, 1)),
        selection={},
        require_geometry_semantics=False,
    )
    finalize_hand_asset_cohort_lock(
        source,
        final,
        canonical_identities=tuple(
            (f"{n + 100:064x}", f"{n:064x}", "a" * 64) for n in (physical_start, physical_start + 1)
        ),
        canonical_schema_version="v1",
        require_geometry_semantics=False,
    )
    return load_hand_asset_cohort(final, require_geometry_semantics=False)


def test_canonical_union_preserves_parent_coordinates_without_source_reresolution(tmp_path, monkeypatch) -> None:
    r"""多父已验证快照仅重排成员，并保留每个成员的真实canonical证书。"""

    left, right = _canonical_parent(tmp_path / "left", "ppo", 1), _canonical_parent(tmp_path / "right", "ssl", 3)
    source, final = tmp_path / "union.lock.yaml", tmp_path / "union.canonical.lock.yaml"

    def forbidden_resolution(*_args, **_kwargs):
        raise AssertionError("union publication must reuse validated canonical parents")

    with monkeypatch.context() as scoped:
        scoped.setattr(cohort_module, "resolve_prepared_train", forbidden_resolution)
        cohort_module.write_hand_asset_cohort_union(
            {"left": left, "right": right},
            source,
            final,
            cohort_id="union",
            member_coordinates=(("left", 1), ("right", 0), ("left", 0)),
            selection={"purpose": "fixture"},
        )
    combined = load_hand_asset_cohort(final, require_geometry_semantics=False)
    assert combined.source_keys == ("ppo#1", "ssl#0", "ppo#0")
    assert tuple(member.physical_geometry_hash for member in combined.members) == (
        f"{2:064x}",
        f"{3:064x}",
        f"{1:064x}",
    )
    assert combined.selection["parent_cohorts"]["left"]["lock_sha256"] == left.lock_sha256
    assert combined.selection["parent_cohorts"]["right"]["lock_sha256"] == right.lock_sha256
    with pytest.raises(FileExistsError):
        cohort_module.write_hand_asset_cohort_union(
            {"left": left},
            source,
            final,
            cohort_id="again",
            member_coordinates=(("left", 0),),
            selection={},
        )


def test_canonical_union_rejects_alias_and_physical_collisions(tmp_path) -> None:
    r"""不同源不能共用含糊alias，物理相同的两个成员不能靠两个父名称重复进入新集合。"""

    left = _canonical_parent(tmp_path / "left", "ppo", 1)
    wrong_alias = _canonical_parent(tmp_path / "alias", "ppo", 3)
    same_physics = _canonical_parent(tmp_path / "physical", "ssl", 1)
    for other, message in ((wrong_alias, "source alias"), (same_physics, "physical")):
        with pytest.raises(ValueError, match=message):
            cohort_module.write_hand_asset_cohort_union(
                {"left": left, "other": other},
                tmp_path / "bad.lock.yaml",
                tmp_path / "bad.canonical.lock.yaml",
                cohort_id="bad",
                member_coordinates=(("left", 0), ("other", 0)),
                selection={},
            )
    assert not (tmp_path / "bad.canonical.lock.yaml").exists()


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


def test_allegro_recipe_balances_cells_after_protecting_old_transfer_lineage() -> None:
    r"""纯Allegro从15条PPO与17条SSL母体形成32×4，保留整个旧迁移留出谱系。"""

    recipe = cohort_selection.PURE_ALLEGRO_RIGHT_A128_RECIPE  # 新族的源配额，不是PPO-ready声明
    assert isinstance(recipe, cohort_selection.PureFamilyLineageRecipe)
    assert cohort_selection.PureLeapRightLineageRecipe is cohort_selection.PureFamilyLineageRecipe
    assert (recipe.family, recipe.production_group, recipe.handedness) == ("allegro", "single_palm_allegro", "right")
    assert recipe.excluded_mother_names == ("right_t4_m4_r3",)
    assert (recipe.mother_count, recipe.asset_count, recipe.members_per_lineage) == (32, 128, 4)

    # PPO原四cell为1/4/4/7；移除3TIP/4DoF-thumb旧留出后，由SSL-only补到各8条。
    quotas = {quota.source_alias: quota.counts for quota in recipe.source_quotas}
    assert quotas == {"ppo": (1, 3, 4, 7), "ssl": (7, 5, 4, 1)}
    assert tuple(
        sum(quota.for_cell(cell) for quota in recipe.source_quotas) for cell in ((3, 3), (3, 4), (4, 3), (4, 4))
    ) == (8, 8, 8, 8)


@pytest.mark.parametrize("excluded", [(), ("right-1",)])
def test_lineage_exclusion_preserves_quota_and_records_only_explicit_exclusions(
    monkeypatch: pytest.MonkeyPatch, excluded: tuple[str, ...]
) -> None:
    r"""排除研究留出母体后仍选择完整lineage；未排除的历史recipe保持原有序列化字段。"""

    # 两条同cell母体各含四个不同资产；测试只回答选择集合，不加载真实几何或Isaac。
    lineages = []
    for key in (1, 2):
        records = tuple(
            SimpleNamespace(
                container=SimpleNamespace(asset_id=f"asset-{key}-{j}", sidecar_path=f"geometry-{key}-{j}"),
                content_hash=f"content-{key}-{j}",
                provenance=SimpleNamespace(asset_role="mother" if j == 0 else "variant"),
            )
            for j in range(4)
        )
        descriptor = replace(
            _lineage(str(key), "index", (0, 4, 1, 3), 0.0),
            mother_row=4 * key,
            mother_asset_id=records[0].container.asset_id,
        )  # 相同几何描述时，较小identity的母体1本来会先被选中。
        lineages.append(
            SimpleNamespace(
                descriptor=descriptor, member_rows=tuple(range(4 * key, 4 * key + 4)), member_records=records
            )
        )
    monkeypatch.setattr(
        cohort_selection.HandAssetDataset, "from_yaml", lambda _path: SimpleNamespace(source_sha256="a" * 64)
    )
    monkeypatch.setattr(cohort_selection, "resolve_prepared_train", lambda *_args, **_kwargs: (object(), True))
    monkeypatch.setattr(cohort_selection, "_resolved_lineages", lambda *_args: tuple(lineages))
    monkeypatch.setattr(cohort_selection, "geometry_fingerprint_from_sidecar", str)

    # 排除名单作用于母体选择前，不能仅删掉最终资产而令cell配额或4成员语义缩水。
    recipe = PureLeapRightLineageRecipe(
        cohort_id="fixture",
        source_quotas=(SourceCellMotherQuota("ssl", (1, 0, 0, 0)),),
        excluded_mother_names=excluded,
    )
    result = cohort_selection.resolve_lineage_cohort_selection(recipe, source_manifests={"ssl": "fixture.yaml"})
    assert len(result.member_coordinates) == 4
    selected = result.selection_document["selected_lineages"]
    assert len(selected) == 1
    assert selected[0]["mother_name"] == ("right-2" if excluded else "right-1")
    serialized = result.selection_document["recipe"]
    if excluded:
        assert serialized["excluded_mother_names"] == excluded
    else:
        assert "excluded_mother_names" not in serialized  # 已发布无排除recipe不因新增可选字段改变身份。

    # 拼错留出母体必须明确失败，不能因为没有匹配到名字就悄悄继续训练。
    with pytest.raises(ValueError, match="excluded mother names.*not found"):
        cohort_selection.resolve_lineage_cohort_selection(
            replace(recipe, excluded_mother_names=("right-unknown",)), source_manifests={"ssl": "fixture.yaml"}
        )


@pytest.mark.parametrize("excluded", [("",), (" right-1",), ("right-1", "right-1")])
def test_lineage_exclusion_rejects_ambiguous_names(excluded: tuple[str, ...]) -> None:
    r"""拒绝空白或重复的排除名，使冻结recipe清楚表示每条研究留出母体。"""

    with pytest.raises(ValueError, match="excluded mother names"):
        replace(PURE_LEAP_RIGHT_A128_RECIPE, excluded_mother_names=excluded)


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
