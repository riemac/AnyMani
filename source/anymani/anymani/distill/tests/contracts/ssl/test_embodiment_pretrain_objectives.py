r"""六项独立 objective 与旧联合数值公式的等价合同。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjective,
    GeometryFieldObjectiveCfg,
)
from anymani.distill.objectives.representations.geometry_terms import (
    ChainObjectiveTermCfg,
    DensityObjectiveTermCfg,
    DerivedFieldObjectiveTermCfg,
    KappaObjectiveTermCfg,
    PairedParityObjectiveTermCfg,
    SobolevObjectiveTermCfg,
)
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch
from anymani.distill.ssl.contracts import build_runtime


@dataclass
class _Context:
    """把同一联合 objective 的共享节点暴露给六个独立 term。"""

    density_prediction: torch.Tensor
    density_target: torch.Tensor
    density_valid_mask: torch.Tensor
    kappa_prediction: torch.Tensor
    kappa_target: torch.Tensor
    edge_valid_mask: torch.Tensor
    field_sensitivity_target: torch.Tensor
    derived_field_sensitivity: torch.Tensor
    auto_field_sensitivity: torch.Tensor
    paired_additive_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def test_split_terms_match_joint_values_and_gradients() -> None:
    r"""固定 batch 上六项值、总和和 q/kappa 梯度必须逐元素等于联合实现。"""

    dtype = torch.float64
    q = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=dtype, requires_grad=True)
    coefficients = torch.tensor(
        [
            [[[0.2, -0.1], [0.3, 0.4]], [[-0.2, 0.5], [0.1, 0.2]]],
            [[[0.4, 0.1], [-0.3, 0.2]], [[0.5, -0.2], [0.2, 0.3]]],
        ],
        dtype=dtype,
    )  # `[G,N_Q,L,N_J]`
    density_prediction = 0.5 + torch.einsum("bj,grlj->bgrl", q, coefficients)
    kappa_prediction = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=dtype, requires_grad=True)
    density_target = torch.full_like(density_prediction, 0.45)
    field_targets = FieldTargetBatch(
        query_points=torch.zeros(2, 2, 2, 3, dtype=dtype),
        query_stratum=torch.zeros(2, 2, 2, dtype=torch.long),
        distance=torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.1], [0.5, 0.3]]], dtype=dtype),
        density=density_target,
        valid_mask=torch.tensor([[[True, True], [True, False]], [[True, True], [False, True]]]),
        owner_role=torch.tensor([0, 1]),
        bandwidths=torch.tensor([0.5, 1.0], dtype=dtype),
        provenance={"frame": "h", "length_unit": "m"},
    )
    owner_index = torch.tensor([0, 1])
    query_index = torch.tensor([1, 0])
    joint_index = torch.tensor([0, 1])
    sensitivity_targets = SensitivityTargetBatch(
        owner_index=owner_index,
        query_index=query_index,
        joint_index=joint_index,
        ancestor_mask=torch.tensor([True, True]),
        closest_point=torch.zeros(2, 2, 3, dtype=dtype),
        closest_source=torch.zeros(2, 2, dtype=torch.long),
        uniqueness_margin=torch.ones(2, 2, dtype=dtype),
        kappa=torch.tensor([[0.05, -0.1], [0.2, 0.25]], dtype=dtype),
        field_sensitivity=torch.tensor([[[0.02, -0.03], [0.01, 0.04]], [[-0.02, 0.05], [0.03, -0.01]]], dtype=dtype),
        valid_mask=torch.tensor([[True, True], [True, False]]),
        provenance={"frame": "h", "distance_unit": "m", "joint_unit": "rad"},
    )
    paired = (
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    paired_loss = paired[0] / paired[1] + paired[2] / paired[3]
    joint = GeometryFieldObjective(GeometryFieldObjectiveCfg())(
        q=q,
        density_prediction=density_prediction,
        kappa_prediction=kappa_prediction,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        paired_loss=paired_loss,
        paired_components=(torch.tensor(5.0, dtype=dtype), torch.tensor(6.0, dtype=dtype)),
        paired_additive_components=paired,
    )
    context = _Context(
        density_prediction=density_prediction,
        density_target=density_target,
        density_valid_mask=field_targets.valid_mask,
        kappa_prediction=kappa_prediction,
        kappa_target=sensitivity_targets.kappa,
        edge_valid_mask=sensitivity_targets.valid_mask,
        field_sensitivity_target=sensitivity_targets.field_sensitivity,
        derived_field_sensitivity=joint.derived_field_sensitivity,
        auto_field_sensitivity=joint.auto_field_sensitivity,
        paired_additive_components=paired,
    )
    configs = {
        "density": DensityObjectiveTermCfg(),
        "kappa": KappaObjectiveTermCfg(),
        "derived_field": DerivedFieldObjectiveTermCfg(),
        "sobolev": SobolevObjectiveTermCfg(),
        "chain": ChainObjectiveTermCfg(),
        "paired": PairedParityObjectiveTermCfg(),
    }
    results = {name: build_runtime(config).evaluate(context) for name, config in configs.items()}
    for name, result in results.items():
        torch.testing.assert_close(result.metrics["loss"], getattr(joint, name), atol=0.0, rtol=0.0)
    split_total = sum((result.metrics["loss"] for result in results.values()), torch.zeros((), dtype=dtype))
    torch.testing.assert_close(split_total, joint.total, atol=0.0, rtol=0.0)
    joint_gradients = torch.autograd.grad(joint.total, (q, kappa_prediction), retain_graph=True)
    split_gradients = torch.autograd.grad(split_total, (q, kappa_prediction))
    for actual, expected in zip(split_gradients, joint_gradients):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
