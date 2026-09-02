r"""Nested task observation到one-level RL transport合同。"""

from __future__ import annotations

import torch
from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
from anymani.distill.rl.structured_transport import StructuredRlTransport


def _nested(batch: int = 3) -> dict[str, dict[str, torch.Tensor]]:
    r"""构造最小完整policy与示意critic nested groups。"""

    joint = torch.ones(batch, 16)
    tip = torch.ones(batch, 4)
    owner = torch.ones(batch, 21)
    return {
        "policy": {
            "jnt_current": torch.full((batch, 16, 3), 200.0),
            "jnt_history": torch.zeros(batch, 30, 16, 4),
            "jnt_limits": torch.zeros(batch, 16, 2),
            "tip_contact": torch.zeros(batch, 4, 1),
            "jnt_valid": joint,
            "tip_valid": tip,
            "owner_valid": owner,
        },
        "critic": {"jnt_state": torch.zeros(batch, 16, 4), "owner_valid": owner},
    }


def test_transport_removes_only_top_group_and_preserves_term_ranks() -> None:
    r"""History等高rank terms不flatten，prototype route保持long。"""

    route = torch.tensor((2, 0, 1), dtype=torch.long)
    transport = StructuredRlTransport.from_nested_observation(_nested(), route, floating_clip=100.0)
    assert transport.policy_terms["jnt_history"].shape == (3, 30, 16, 4)
    assert float(transport.policy_terms["jnt_current"].max()) == 100.0
    storage = transport.policy_storage()
    assert storage["prototype_index"].dtype == torch.long
    observation = StructuredActorObservation.from_task_dict(storage)
    assert observation.jnt_current.shape == (3, 16, 3)  # route不进入模型字段


def test_minibatch_selection_keeps_route_aligned_with_every_leaf() -> None:
    r"""Shuffle/slice后geometry route与actor/critic rows同序。"""

    route = torch.tensor((7, 8, 9), dtype=torch.long)
    nested = _nested()
    nested["policy"]["jnt_current"][:, 0, 0] = torch.tensor((70.0, 80.0, 90.0))
    transport = StructuredRlTransport.from_nested_observation(nested, route)
    selected = transport.select(torch.tensor((2, 0), dtype=torch.long))
    assert selected.prototype_index.tolist() == [9, 7]
    assert selected.policy_terms["jnt_current"][:, 0, 0].tolist() == [90.0, 70.0]
