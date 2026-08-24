r"""前向预实验 artifact 合同：复用训练数据、统计三项、不更新参数或权重。"""

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
        return {"density": 1.0, "kappa": 1.0, "derived_field": 1.0}

    def formula_identity(self) -> dict[str, str]:
        prefix = "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives"
        return {name: f"{prefix}.{name}_objective" for name in self.declared_objective_weights()}

    def train_mode(self) -> None:
        return None

    def forward_objectives(self, batch, *, step: int, mode: str = "eval", microbatch_size: int | None = None):
        del step, mode, microbatch_size
        return _Step(batch)


class _FailAfterOneForward(_Method):
    r"""在第二个 group 模拟进程内异常，用于验证最近完整 partial 边界。"""

    def __init__(self) -> None:
        self.forward_count = 0

    def forward_objectives(self, batch, *, step: int, mode: str = "eval", microbatch_size: int | None = None):
        self.forward_count += 1
        if self.forward_count > 1:
            raise RuntimeError("synthetic interruption")
        return super().forward_objectives(batch, step=step, mode=mode, microbatch_size=microbatch_size)


def _resolved(*, max_epochs: int = 1, num_minibatches: int = 2) -> dict[str, object]:
    r"""构造只含 artifact 记录所需字段的最小 schema-6 配置。"""

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
            "max_epochs": max_epochs,
            "num_minibatches": num_minibatches,
            "mini_epochs": 1,
            "microbatch_size": 1,
        },
    }


def _schedule(session: _Session, *, max_epochs: int = 1, num_minibatches: int | None = None) -> OnlineMinibatchSchedule:
    r"""让 synthetic session 的每个 dict 对应一个新 minibatch。"""

    return OnlineMinibatchSchedule(
        session.asset_count,
        OnlineSamplingCfg(assets_per_minibatch=1, q_per_asset_per_minibatch=1, shuffle_assets=False),
        max_epochs=max_epochs,
        num_minibatches=num_minibatches or session.asset_count,
    )


def test_calibration_records_epoch_minibatches_without_reuse(tmp_path: Path) -> None:
    r"""一个 epoch 的两批新数据各前向一次，不产生 optimizer update。"""

    method = _Method()
    batches = (
        {"density": 0.4, "kappa": 0.2, "derived_field": 0.1},
        {"density": 0.6, "kappa": 0.4, "derived_field": 0.3},
    )
    session = _Session(batches)
    path = tmp_path / "loss_calibration.yaml"
    digest = _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        max_epochs=1,
        num_minibatches=2,
        microbatch_size=1,
        manifest_hash="abc",
        resolved_config=_resolved(),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "6.0.0"
    assert payload["status"] == "complete"
    assert payload["execution"]["global_minibatches"] == 2
    assert payload["execution"]["forward_count"] == 2
    assert payload["execution"]["new_pairs_realized"] == 2
    assert payload["execution"]["pair_uses"] == 2
    assert payload["execution"]["optimizer_updates"] == 0
    assert payload["term_means"]["density"] == pytest.approx(0.5)
    assert payload["term_traces"]["density"] == pytest.approx([0.4, 0.6])
    assert set(payload["term_means"]) == {"density", "kappa", "derived_field"}
    assert set(payload["term_traces"]) == {"density", "kappa", "derived_field"}
    assert len(digest) == 64
    assert _require_calibration_identity(path, method=method, manifest_hash="abc") == digest
    assert not (tmp_path / "loss_calibration.partial.yaml").exists()


def test_calibration_interruption_keeps_only_the_last_complete_epoch(tmp_path: Path) -> None:
    r"""第二个 epoch forward 失败时，partial 只能包含首个完整 epoch。"""

    method = _FailAfterOneForward()
    batch = {"density": 0.4, "kappa": 0.2, "derived_field": 0.1}
    session = _Session((batch, batch))
    output = tmp_path / "loss_calibration.yaml"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _write_calibration_artifact(
            method,
            session,
            _schedule(session, max_epochs=2, num_minibatches=1),
            output,
            max_epochs=2,
            num_minibatches=1,
            microbatch_size=1,
            manifest_hash="abc",
            resolved_config=_resolved(max_epochs=2, num_minibatches=1),
        )

    partial = yaml.safe_load((tmp_path / "loss_calibration.partial.yaml").read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["execution"]["completed_epochs"] == 1
    assert partial["execution"]["global_minibatches"] == 1
    assert partial["execution"]["forward_count"] == 1
    assert partial["term_means"]["density"] == pytest.approx(0.4)
    assert not output.exists()  # 未完成 run 不得发布可供正式训练引用的最终 artifact


def test_calibration_reference_allows_preset_and_weight_changes_but_rejects_formula_drift(tmp_path: Path) -> None:
    r"""预实验只提供权重判断证据；正式 preset 可变，但损失公式身份必须可审计。"""

    method = _Method()
    session = _Session(({"density": 0.4, "kappa": 0.2, "derived_field": 0.1},))
    path = tmp_path / "loss_calibration.yaml"
    _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        max_epochs=1,
        num_minibatches=1,
        microbatch_size=1,
        manifest_hash="abc",
        resolved_config=_resolved(num_minibatches=1),
    )
    method.declared_objective_weights = lambda: {  # type: ignore[method-assign]
        "density": 0.2,
        "kappa": 1.0,
        "derived_field": 1.0,
    }
    assert _require_calibration_identity(path, method=method, manifest_hash="abc")

    drifted = _Method()
    drifted.formula_identity = lambda: {  # type: ignore[method-assign]
        **method.formula_identity(),
        "density": "somewhere.else.density_objective",
    }
    with pytest.raises(ValueError, match="formula identity"):
        _require_calibration_identity(path, method=drifted, manifest_hash="abc")


def test_calibration_rejects_superseded_five_term_formula_identity(tmp_path: Path) -> None:
    r"""含额外损失身份的旧 artifact 不得被三项正式训练接受。"""

    method = _Method()
    session = _Session(({"density": 0.4, "kappa": 0.2, "derived_field": 0.1},))
    path = tmp_path / "loss_calibration.yaml"
    _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        max_epochs=1,
        num_minibatches=1,
        microbatch_size=1,
        manifest_hash="abc",
        resolved_config=_resolved(num_minibatches=1),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["formula_identity"]["sobolev"] = "superseded.sobolev_objective"
    payload["formula_identity"]["chain"] = "superseded.chain_objective"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="formula identity"):
        _require_calibration_identity(path, method=method, manifest_hash="abc")


def test_calibration_rejects_schema_five_artifact(tmp_path: Path) -> None:
    r"""schema-5 calibration 没有迁移路径，即使三项公式和 dataset hash 相同也必须拒绝。"""

    method = _Method()
    session = _Session(({"density": 0.4, "kappa": 0.2, "derived_field": 0.1},))
    path = tmp_path / "loss_calibration.yaml"
    _write_calibration_artifact(
        method,
        session,
        _schedule(session),
        path,
        max_epochs=1,
        num_minibatches=1,
        microbatch_size=1,
        manifest_hash="abc",
        resolved_config=_resolved(num_minibatches=1),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "5.0.0"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema must be 6.0.0"):
        _require_calibration_identity(path, method=method, manifest_hash="abc")


def test_calibration_uses_additive_statistics_and_never_builds_parameter_gradients(tmp_path: Path) -> None:
    r"""不同 denominator 按充分统计合并；只 forward 不应写入参数 ``.grad``。"""

    method = _Method()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    shared = {"kappa": 1.0, "derived_field": 1.0}
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
        max_epochs=1,
        num_minibatches=2,
        microbatch_size=1,
        manifest_hash="abc",
        resolved_config=_resolved(),
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["term_means"]["density"] == pytest.approx(5.0)
    assert payload["term_traces"]["density"] == pytest.approx([2.0, 6.0])
    assert parameter.detach().item() == pytest.approx(2.0)
    assert parameter.grad is None
