r"""前向预实验 artifact 合同：复用训练数据、统计五项、不更新参数或权重。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from anymani.distill.ssl.runtime.lifecycle import _require_calibration_identity, _write_calibration_artifact
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule, OnlineSamplingCfg


class _Term:
    def __init__(self, value: float | tuple[torch.Tensor, float]) -> None:
        if isinstance(value, tuple):
            numerator, denominator_value = value
            denominator = torch.tensor(denominator_value)
            mean = numerator / denominator
        else:
            numerator = torch.tensor(value)
            denominator = torch.tensor(1.0)
            mean = numerator
        self.metrics = {"loss": mean}
        self.components = (SimpleNamespace(name="", numerator=numerator, denominator=denominator),)


class _Step:
    def __init__(self, terms: dict[str, float | tuple[torch.Tensor, float]]) -> None:
        self.objectives = {name: _Term(value) for name, value in terms.items()}
        for name, objective in self.objectives.items():
            objective.components[0].name = name
        self.sample_count = 1


class _Session:
    def __init__(self, batches: tuple[dict[str, float | tuple[torch.Tensor, float]], ...]) -> None:
        self.batches = batches
        self.index = 0
        self.asset_count = len(batches)

    def realize(self, _item, *, schedule, step: int):
        del schedule, step
        result = self.batches[self.index]
        self.index += 1
        return result


class _Method:
    def declared_objective_weights(self) -> dict[str, float]:
        return {"density": 1.0, "kappa": 1.0, "derived_field": 1.0, "sobolev": 1.0, "chain": 1.0}

    def formula_identity(self) -> dict[str, str]:
        prefix = "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives"
        return {name: f"{prefix}.{name}_objective" for name in self.declared_objective_weights()}

    def train_mode(self) -> None:
        return None

    def forward_objectives(self, batch, *, step: int, mode: str = "eval"):
        del step, mode
        return _Step(batch)


def _resolved(*, num_minibatches: int = 2, mini_epochs: int = 2) -> dict[str, object]:
    r"""构造只含 artifact 记录所需字段的最小 schema-5 配置。"""

    return {
        "method": {
            "state_measure": {"kind": "scrambled_sobol_joint_limits"},
            "representation": {"field": {"bandwidth_centers_m": [0.004, 0.016, 0.064]}},
            "model": {"encoder": {"heads": {"zero_order_width": 128}}},
            "objectives": {"density": {"weight": 1.0}},
            "joint_sign_rewrite": {"probability": 0.2},
        },
        "trainer": {
            "sampling": {"assets_per_minibatch": 1, "q_per_asset_per_minibatch": 1, "seed": 0},
            "num_minibatches": num_minibatches,
            "mini_epochs": mini_epochs,
            "gradient_accumulation_steps": 2,
        },
    }


def _schedule(session: _Session) -> OnlineMinibatchSchedule:
    r"""让 synthetic session 的每个 dict 对应一个新 minibatch。"""

    return OnlineMinibatchSchedule(
        session.asset_count,
        OnlineSamplingCfg(assets_per_minibatch=1, q_per_asset_per_minibatch=1, shuffle_assets=False),
        num_minibatches=session.asset_count,
    )


def test_calibration_records_new_data_and_reused_forward_counts(tmp_path: Path) -> None:
    r"""两批新数据复用两遍时，应记录 2 次 realization、4 次 forward。"""

    method = _Method()
    batches = (
        {"density": 0.4, "kappa": 0.2, "derived_field": 0.1, "sobolev": 0.3, "chain": 0.5},
        {"density": 0.6, "kappa": 0.4, "derived_field": 0.3, "sobolev": 0.1, "chain": 0.1},
    )
    session = _Session(batches)
    path = tmp_path / "loss_calibration.yaml"
    digest = _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        mini_epochs=2,
        gradient_accumulation_steps=2,
        manifest_hash="abc",
        resolved_config=_resolved(),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "5.0.0"
    assert payload["execution"]["minibatch_count"] == 2
    assert payload["execution"]["forward_count"] == 4
    assert payload["execution"]["new_sample_count"] == 2
    assert payload["execution"]["forward_sample_count"] == 4
    assert payload["term_means"]["density"] == pytest.approx(0.5)
    assert payload["term_traces"]["density"] == pytest.approx([0.4, 0.6, 0.4, 0.6])
    assert len(digest) == 64
    assert _require_calibration_identity(path, method=method, manifest_hash="abc") == digest


def test_calibration_reference_allows_preset_and_weight_changes_but_rejects_formula_drift(tmp_path: Path) -> None:
    r"""预实验只提供权重判断证据；正式 preset 可变，但损失公式身份必须可审计。"""

    method = _Method()
    session = _Session(({"density": 0.4, "kappa": 0.2, "derived_field": 0.1, "sobolev": 0.3, "chain": 0.5},))
    path = tmp_path / "loss_calibration.yaml"
    _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        mini_epochs=1,
        gradient_accumulation_steps=1,
        manifest_hash="abc",
        resolved_config=_resolved(num_minibatches=1, mini_epochs=1),
    )
    method.declared_objective_weights = lambda: {  # type: ignore[method-assign]
        "density": 0.2,
        "kappa": 1.0,
        "derived_field": 1.0,
        "sobolev": 1.0,
        "chain": 1.0,
    }
    assert _require_calibration_identity(path, method=method, manifest_hash="abc")

    drifted = _Method()
    drifted.formula_identity = lambda: {  # type: ignore[method-assign]
        **method.formula_identity(),
        "density": "somewhere.else.density_objective",
    }
    with pytest.raises(ValueError, match="formula identity"):
        _require_calibration_identity(path, method=drifted, manifest_hash="abc")


def test_calibration_uses_additive_statistics_and_never_builds_parameter_gradients(tmp_path: Path) -> None:
    r"""不同 denominator 按充分统计合并；只 forward 不应写入参数 ``.grad``。"""

    method = _Method()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    shared = {"kappa": 1.0, "derived_field": 1.0, "sobolev": 1.0, "chain": 1.0}
    session = _Session(
        (
            {"density": (parameter * 1.0, 1.0), **shared},
            {"density": (parameter * 9.0, 3.0), **shared},
        )
    )
    path = tmp_path / "loss_calibration.yaml"
    _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        mini_epochs=1,
        gradient_accumulation_steps=2,
        manifest_hash="abc",
        resolved_config=_resolved(mini_epochs=1),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["term_means"]["density"] == pytest.approx(5.0)
    assert payload["term_traces"]["density"] == pytest.approx([2.0, 6.0])
    assert parameter.detach().item() == pytest.approx(2.0)
    assert parameter.grad is None
