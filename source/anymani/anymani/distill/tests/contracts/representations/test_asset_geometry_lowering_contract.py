"""assets 静态几何语义到 representation source 运动学规格的职责边界合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.sources.kinematics import (
    forward_owner_transforms,
    lower_hand_geometry_semantics,
)

_MOTHER_ROOT = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "generated"
    / "2026-06-10_11-30-08"
    / "single_palm_leap"
    / "right_t4_i4_m4_r4"
)
_requires_local_mother = pytest.mark.skipif(
    not _MOTHER_ROOT.is_dir(),
    reason="generated LEAP mother asset is a local research artifact",
)


@_requires_local_mother
def test_mother_bank_semantics_lower_to_branched_owner_kinematics() -> None:
    """mother 必须 lower 为 16 JOINT、21 owner、24 collision 的分支树。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    semantics = container.geometry_semantics
    spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)

    assert spec.space_screws.shape == (16, 6)
    assert spec.owner_home_transforms.shape == (21, 4, 4)
    assert spec.owner_ancestor_mask.shape == (21, 16)
    assert spec.joint_ancestor_mask.shape == (16, 16)
    assert spec.joint_limits is not None and spec.joint_limits.shape == (16, 2)
    assert spec.component_owner_indices is not None and spec.component_owner_indices.shape == (24,)
    assert spec.component_owner_local_transforms is not None
    assert spec.component_owner_local_transforms.shape == (24, 4, 4)

    index_j0 = spec.owner_ids.index("joint/index_j0")
    index_j3 = spec.owner_ids.index("joint/index_j3")
    middle_j0 = spec.owner_ids.index("joint/middle_j0")
    index_tip = spec.owner_ids.index("tip/index")
    middle_tip = spec.owner_ids.index("tip/middle")
    assert spec.owner_ancestor_mask[index_j0].sum().item() == 1
    assert spec.owner_ancestor_mask[index_j3].sum().item() == 4
    assert spec.owner_ancestor_mask[middle_j0].sum().item() == 1
    assert not spec.owner_ancestor_mask[middle_j0, spec.joint_names.index("index_j0")]
    assert spec.owner_ancestor_mask[index_tip].sum().item() == 4

    assert spec.owner_graph_shortest is not None
    assert spec.owner_graph_parent is not None
    assert spec.owner_graph_child is not None
    assert spec.owner_graph_shortest[index_tip, middle_tip].item() == 8  # 真距离 10，在末桶截断
    assert spec.owner_graph_parent[index_tip, 0].item() == 5
    assert spec.owner_graph_child[0, index_tip].item() == 5


@_requires_local_mother
def test_lowered_home_pose_matches_explicit_q_home_and_anchor_seed() -> None:
    """在显式 home 上，POE 必须还原 owner home；首 JOINT 原点应对齐资产 seed。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    semantics = container.geometry_semantics
    spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)

    reconstructed_home = forward_owner_transforms(spec, spec.q_home.unsqueeze(0))[0]
    assert torch.allclose(reconstructed_home, spec.owner_home_transforms, atol=1.0e-12, rtol=1.0e-12)

    seed_by_finger = {seed.finger_name: seed for seed in semantics.anchor_seeds}
    for finger_name in ("index", "middle", "ring", "thumb"):
        joint_name = seed_by_finger[finger_name].first_active_joint_name
        owner_index = spec.owner_ids.index(f"joint/{joint_name}")
        assert spec.owner_home_transforms[owner_index, :3, 3].tolist() == pytest.approx(
            seed_by_finger[finger_name].position_a_m
        )
