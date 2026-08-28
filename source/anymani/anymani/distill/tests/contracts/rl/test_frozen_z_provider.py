r"""冻结 $Z$ provider 的 shape、identity、mask 与 checkpoint buffer 合同。"""

from __future__ import annotations

import torch
from anymani.assets.canonical_runtime import CanonicalHandArtifact, CanonicalHandRouting
from anymani.distill.rl.frozen_z import FrozenZProvider, build_frozen_z_provider_from_canonical_artifacts


def _provider(*, seed: int = 7, physical_hashes: tuple[str, ...] = ("p0", "p1")) -> FrozenZProvider:
    r"""构造两资产、不同 active joints/tips 的最小真实轴 provider。"""

    joint_mask = torch.tensor(
        [
            [True, True, False, False] * 4,
            [True, False, False, False] * 4,
        ],
        dtype=torch.bool,
    )  # `[2,16]`
    owner_mask = torch.zeros(2, 21, dtype=torch.bool)
    owner_mask[:, 0] = True  # PALM
    owner_mask[:, 1:17] = joint_mask
    owner_mask[:, 17:] = torch.tensor([[True, True, True, True], [True, True, True, True]])
    graph = torch.arange(21 * 21, dtype=torch.long).reshape(1, 21, 21).repeat(2, 1, 1)
    return FrozenZProvider(
        asset_ids=("a0", "a1"),
        physical_geometry_hashes=physical_hashes,
        owner_valid_mask=owner_mask,
        joint_valid_mask=joint_mask,
        shortest_path=graph,
        parent_direction=graph + 1,
        child_direction=graph + 2,
        dataset_digest="dataset",
        manifest_digest="manifest",
        seed=seed,
    )


def test_valid_owner_z_is_normalized_and_ghost_is_exact_zero() -> None:
    r"""有效 token 匹配 final-norm 尺度，ghost token 不携带随机噪声。"""

    provider = _provider()
    valid = provider.z_table[provider.owner_valid_mask]
    invalid = provider.z_table[~provider.owner_valid_mask]

    assert torch.allclose(valid.mean(dim=-1), torch.zeros(valid.shape[0]), atol=1.0e-6)
    assert torch.allclose(torch.sqrt(valid.square().mean(dim=-1)), torch.ones(valid.shape[0]), atol=1.0e-6)
    assert torch.count_nonzero(invalid) == 0
    assert tuple(provider.parameters()) == ()


def test_same_identity_is_bitwise_stable_and_identity_changes_with_inputs() -> None:
    r"""重复构造逐 bit 一致；seed/physical hash 改变必须改变 artifact identity。"""

    first = _provider()
    repeated = _provider()
    other_seed = _provider(seed=8)
    other_physical = _provider(physical_hashes=("p0", "changed"))

    assert torch.equal(first.z_table, repeated.z_table)
    assert first.identity == repeated.identity
    assert first.identity["identity_digest"] != other_seed.identity["identity_digest"]
    assert first.identity["identity_digest"] != other_physical.identity["identity_digest"]


def test_resolve_gathers_all_manifest_tables_by_asset_row() -> None:
    r"""asset row 必须同步 gather Z、mask 与 graph，不能只路由单一 tensor。"""

    provider = _provider()
    rows = torch.tensor([1, 0, 1], dtype=torch.long)
    batch = provider.resolve(rows)

    assert torch.equal(batch.geometry_entities, provider.z_table[rows])
    assert torch.equal(batch.owner_valid_mask, provider.owner_valid_mask[rows])
    assert torch.equal(batch.joint_valid_mask, provider.joint_valid_mask[rows])
    assert torch.equal(batch.shortest_path, provider.shortest_path[rows])
    assert batch.geometry_entities.requires_grad is False


def test_provider_rejects_inconsistent_joint_owner_mask() -> None:
    r"""manifest 的 JOINT owner 与 active-joint mask 不一致时必须 fail closed。"""

    joint = torch.ones(1, 16, dtype=torch.bool)
    owner = torch.ones(1, 21, dtype=torch.bool)
    owner[:, 1] = False
    graph = torch.zeros(1, 21, 21, dtype=torch.long)
    try:
        FrozenZProvider(
            asset_ids=("a",),
            physical_geometry_hashes=("p",),
            owner_valid_mask=owner,
            joint_valid_mask=joint,
            shortest_path=graph,
            parent_direction=graph,
            child_direction=graph,
            dataset_digest="d",
            manifest_digest="m",
        )
    except ValueError as exc:
        assert "JOINT owner mask" in str(exc)
    else:
        raise AssertionError("inconsistent manifest masks must be rejected")


def test_artifact_builder_lowers_compact_active_chains_to_real_owner_graph() -> None:
    r"""manifest active chains 必须产生 PALM→JOINT→TIP 图，ghost owner 保持 invalid/unreachable。"""

    active = (True, False, False, False, True, False, False, False, *([False] * 8))  # index j0/j1
    routing = CanonicalHandRouting(
        asset_id="index-two-dof",
        source_dof_count=2,
        source_joint_names=("index_j0", "index_j1"),
        active_joint_names=("index_j0", "index_j1"),
        active_joint_mask=active,
        active_tip_mask=(True, False, False, False),
        source_to_canonical=(("index_j0", "index_j0"), ("index_j1", "index_j1")),
    )
    artifact = CanonicalHandArtifact(
        schema_version="v1",
        schema_digest="schema",
        asset_id="index-two-dof",
        source_content_hash="source-content",
        source_urdf_hash="source-urdf",
        physical_geometry_hash="physical",
        canonical_urdf_hash="canonical-urdf",
        canonical_urdf_path="/unused/hand.urdf",
        manifest_path="/unused/canonical_runtime.json",
        routing=routing,
    )

    provider = build_frozen_z_provider_from_canonical_artifacts(
        (artifact,),
        dataset_digest="dataset",
        manifest_digest="manifest",
    )

    assert provider.owner_valid_mask[0, [0, 1, 5, 17]].all()  # palm、index j0/j1、index tip
    assert not provider.owner_valid_mask[0, 2]  # middle j0 ghost
    assert provider.shortest_path[0, 0, 17] == 3
    assert provider.parent_direction[0, 17, 0] == 3
    assert provider.child_direction[0, 0, 17] == 3
    assert provider.shortest_path[0, 0, 2] == 8  # invalid owner 不进入图
