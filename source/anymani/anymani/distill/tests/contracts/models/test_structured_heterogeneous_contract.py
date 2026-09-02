r"""Structured heterogeneous模型输入ABI合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredCriticObservation,
)


def _masks(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""构造10-DoF/4-TIP canonical masks。"""

    joint = torch.tensor(
        (True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False)
    ).repeat(batch, 1)
    tip = torch.ones(batch, 4, dtype=torch.bool)
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint, tip), dim=-1)
    return joint, tip, owner


def test_actor_restores_named_task_dict_and_numeric_masks() -> None:
    r"""Task float0/1 masks在模型边界恢复为bool，不依赖flat offsets。"""

    joint, tip, owner = _masks()
    task = {
        "jnt_current": torch.zeros(2, 16, 3),
        "jnt_history": torch.zeros(2, 30, 16, 4),
        "jnt_limits": torch.zeros(2, 16, 2),
        "tip_contact": torch.zeros(2, 4, 1),
        "jnt_valid": joint.float(),
        "tip_valid": tip.float(),
        "owner_valid": owner.float(),
        "palm_valid": torch.ones(2, 1),  # 模型无需伪PALM feature，额外named term不影响解析
    }
    observation = StructuredActorObservation.from_task_dict(task)
    assert observation.jnt_valid.dtype == torch.bool
    assert observation.jnt_history.shape == (2, 30, 16, 4)


def test_actor_rejects_owner_mask_inconsistent_with_joint_tip_axes() -> None:
    r"""Owner mask不能静默脱离JOINT/TIP routing。"""

    joint, tip, owner = _masks()
    owner[:, 3] = ~owner[:, 3]
    with pytest.raises(RuntimeError, match="owner_valid"):
        StructuredActorObservation(
            jnt_current=torch.zeros(2, 16, 3),
            jnt_history=torch.zeros(2, 30, 16, 4),
            jnt_limits=torch.zeros(2, 16, 2),
            tip_contact=torch.zeros(2, 4, 1),
            jnt_valid=joint,
            tip_valid=tip,
            owner_valid=owner,
        )


def test_critic_keeps_privileged_roles_separate() -> None:
    r"""JOINT/owner/object/task保持独立rank，不恢复127D flat state。"""

    joint, tip, owner = _masks()
    observation = StructuredCriticObservation(
        jnt_state=torch.zeros(2, 16, 4),
        owner_contact=torch.zeros(2, 21, 2),
        obj=torch.zeros(2, 1, 15),
        task=torch.zeros(2, 1, 8),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    assert observation.obj.shape == (2, 1, 15)
    assert observation.task.shape == (2, 1, 8)


def test_geometry_contract_allows_poisoned_ghost_for_mask_invariance_tests() -> None:
    r"""Ghost token可被finite poison，后续网络必须靠mask消除影响。"""

    _, _, owner = _masks()
    tokens = torch.randn(2, 21, 128)
    tokens[~owner] = 1.0e6
    geometry = GeometryTokenBatch(tokens=tokens, owner_valid=owner)
    assert float(geometry.tokens[~owner].max().item()) == 1.0e6
