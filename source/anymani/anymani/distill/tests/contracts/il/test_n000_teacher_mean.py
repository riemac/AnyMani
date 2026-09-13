r"""Accepted teacher环境动作mean imitation的split与指标合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.il.n000_teacher_mean import (
    replica_modulo_split,
    teacher_mean_loss,
    teacher_mean_metrics,
    training_batch_indices,
)


def test_replica_split_is_deterministic_and_disjoint() -> None:
    r"""R16按env ID跨全轴取validation，且任何replica只属于一个split。"""

    train, validation = replica_modulo_split(16, validation_modulus=4, validation_remainder=3)
    assert validation == (3, 7, 11, 15)
    assert train == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(16))


def test_teacher_mean_metrics_separate_saturation_and_switch_states() -> None:
    r"""总体MSE不得遮蔽非饱和切换状态，并显式比较previous-action baseline。"""

    target = torch.tensor([[-1.0, 1.0, 0.25, -0.25], [1.0, -1.0, 0.75, -0.75]])
    prediction = target.clone()
    previous_action = torch.zeros_like(target)

    metrics = teacher_mean_metrics(prediction, target, previous_action=previous_action)

    assert metrics.mse == 0.0 and metrics.mae == 0.0
    assert metrics.saturated_mse == 0.0 and metrics.transition_mse == 0.0
    assert metrics.sign_accuracy == 1.0
    assert metrics.skill_vs_zero == 1.0 and metrics.skill_vs_previous_action == 1.0
    assert metrics.transition_skill_vs_zero == 1.0
    assert metrics.transition_skill_vs_previous_action == 1.0
    assert metrics.previous_action_saturated_mse == 1.0
    assert metrics.previous_action_transition_mse == pytest.approx(0.3125)
    assert metrics.saturated_fraction == pytest.approx(0.5)


def test_teacher_mean_metrics_reject_nonfinite_or_shape_mismatch() -> None:
    r"""监督诊断对shape漂移和非有限预测fail closed。"""

    target = torch.zeros(2, 16)
    with pytest.raises(ValueError, match="shape"):
        teacher_mean_metrics(torch.zeros(2, 15), target)
    prediction = target.clone()
    prediction[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        teacher_mean_metrics(prediction, target)


def test_balanced_action_regime_loss_does_not_let_saturation_count_dominate() -> None:
    r"""Balanced reduction分别平均饱和与切换元素，使85%的饱和标签不能压低切换误差权重。"""

    target = torch.tensor([[-1.0, 1.0, -1.0, 0.25]])
    prediction = torch.tensor([[0.0, 0.0, 0.0, 3.25]])

    element_mean = teacher_mean_loss(prediction, target, reduction="element_mean")
    balanced = teacher_mean_loss(prediction, target, reduction="balanced_action_regimes")

    assert element_mean.item() == pytest.approx(3.0)  # $(3\times1+1\times9)/4$
    assert balanced.item() == pytest.approx(5.0)  # $(\operatorname{mean}_{sat}1+\operatorname{mean}_{switch}9)/2$
    with pytest.raises(ValueError, match="reduction"):
        teacher_mean_loss(prediction, target, reduction="unknown")  # type: ignore[arg-type]


def test_explicit_full_batch_visits_every_fixed_sample_once() -> None:
    r"""Full-batch语义必须是无放回完整轴，不能由同尺寸randint冒充。"""

    index = training_batch_indices(
        sample_count=4,
        batch_size=4,
        full_batch=True,
        generator=None,
        device=torch.device("cpu"),
    )
    assert torch.equal(index, torch.arange(4))
    with pytest.raises(ValueError, match="equal"):
        training_batch_indices(
            sample_count=4,
            batch_size=3,
            full_batch=True,
            generator=None,
            device=torch.device("cpu"),
        )
