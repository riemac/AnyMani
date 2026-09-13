r"""固定检查点动作响应探针的纯张量/归约合同，不启动仿真。"""

import numpy as np
import pytest
import torch
from anymani.distill.rl.scripts.probe_palm_rotation_action_response import (
    _action_conditions,
    _initial_matches,
    _sign_contrast,
)


def test_action_conditions_are_signed_masked_and_reproducible() -> None:
    r"""12条条件保留反馈与固定向量的区别；随机正负方向与ghost语义严格对应。"""

    mean = torch.tensor([[0.2, 0.8, -0.3], [0.4, -0.5, 0.7]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    conditions = _action_conditions(mean, mask, amplitude=0.5, seed=73)
    repeat = _action_conditions(mean, mask, amplitude=0.5, seed=73)
    assert len(conditions) == 12 and conditions["feedback"] is None
    zero = conditions["zero"]
    assert isinstance(zero, torch.Tensor) and torch.count_nonzero(zero) == 0
    for name, action in conditions.items():
        if action is not None:
            assert torch.count_nonzero(action[~mask]) == 0
            torch.testing.assert_close(action, repeat[name])
            assert float(action.abs().max()) <= 1.0
    for prefix in ("frozenmean", "rademacher_0", "rademacher_1", "rademacher_2", "rademacher_3"):
        positive, negative = conditions[f"{prefix}_positive"], conditions[f"{prefix}_negative"]
        assert isinstance(positive, torch.Tensor) and isinstance(negative, torch.Tensor)
        torch.testing.assert_close(positive, -negative)


def test_initial_fingerprint_marks_only_changed_environments() -> None:
    r"""相同记录初态才能作配对contrast；一个环境变化不应隐藏或污染其余环境。"""

    reference = {"q": torch.zeros(3, 2), "mask": torch.ones(3, 2, dtype=torch.bool)}
    current = {name: value.clone() for name, value in reference.items()}
    current["q"][1, 0] = 0.01
    matched, errors = _initial_matches(reference, current, atol=1.0e-6)
    assert matched.tolist() == [True, False, True]
    assert errors["q"] == pytest.approx(0.01)


def test_sign_contrast_requires_matched_full_survival_and_uses_seconds() -> None:
    r"""终止或初态不匹配者不进入单位动作幅值下的rad/s响应归约。"""

    positive = {
        "net_rotation_rad": np.array([1.0, 2.0, 3.0]),
        "full_survival": np.array([True, False, True]),
        "initial_matched": np.array([True, True, False]),
    }
    negative = {
        "net_rotation_rad": np.array([-1.0, 0.0, 1.0]),
        "full_survival": np.ones(3, dtype=bool),
        "initial_matched": np.ones(3, dtype=bool),
    }
    contrast = _sign_contrast(positive, negative, amplitude=0.5, horizon_seconds=1.0)
    assert contrast["valid"].tolist() == [True, False, False]
    assert contrast["response_rad_s_per_action_unit"][0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        _sign_contrast(positive, negative, amplitude=0.0, horizon_seconds=1.0)
