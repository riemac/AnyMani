r"""前向预实验 artifact 合同：不更新参数、不改权重。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from anymani.distill.ssl.runtime.lifecycle import _require_calibration_identity, _write_calibration_artifact


class _Term:
    def __init__(self, value: float) -> None:
        self.metrics = {"loss": __import__("torch").tensor(value)}


class _Step:
    def __init__(self, terms: dict[str, float]) -> None:
        self.objectives = {name: _Term(value) for name, value in terms.items()}


class _Method:
    def declared_objective_weights(self) -> dict[str, float]:
        return {"density": 1.0, "kappa": 1.0, "derived_field": 1.0, "sobolev": 1.0, "chain": 1.0}

    def formula_identity(self) -> dict[str, str]:
        return {
            "density": "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives.density_objective",
            "kappa": "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives.kappa_objective",
            "derived_field": "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives.derived_field_objective",
            "sobolev": "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives.sobolev_objective",
            "chain": "anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives.chain_objective",
        }

    def require_model(self):
        class _Model:
            def eval(self) -> None:
                return None

        return _Model()

    def forward_objectives(self, batch, *, step: int, mode: str = "eval"):
        del step, mode
        return _Step(batch)


def test_calibration_artifact_records_five_terms_without_weight_rewrite(tmp_path: Path) -> None:
    """前向预实验写出五项均值，并保持声明权重不变。"""

    method = _Method()
    batches = (
        {"density": 0.4, "kappa": 0.2, "derived_field": 0.1, "sobolev": 0.3, "chain": 0.5},
        {"density": 0.6, "kappa": 0.4, "derived_field": 0.3, "sobolev": 0.1, "chain": 0.1},
    )
    path = tmp_path / "loss_calibration.yaml"
    resolved = {
        "method": {
            "state_measure": {"kind": "scrambled_sobol_joint_limits"},
            "representation": {"field": {"bandwidth_centers_m": [0.004, 0.016, 0.064]}},
            "model": {"encoder": {"heads": {"zero_order_width": 128}}},
            "objectives": {"density": {"weight": 1.0}},
        },
        "trainer": {"sampling": {"epochs": 20, "q_per_asset_per_epoch": 256}},
    }
    digest = _write_calibration_artifact(
        method, batches, path, manifest_hash="abc", resolved_config=resolved
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "4.0.0"
    assert payload["declared_objective"]["density"] == 1.0
    assert payload["formula_identity"]["density"].endswith("density_objective")
    assert payload["scientific_identity"]["sampling"]["q_per_asset_per_epoch"] == 256
    assert payload["term_means"]["density"] == pytest.approx(0.5)
    assert payload["term_means"]["kappa"] == pytest.approx(0.3)
    assert len(digest) == 64
    assert _require_calibration_identity(
        path, method=method, manifest_hash="abc", resolved_config=resolved
    ) == digest

    payload.pop("scientific_identity")
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific identity"):
        _require_calibration_identity(path, method=method, manifest_hash="abc", resolved_config=resolved)


def test_calibration_accepts_changed_weights_but_rejects_formula_drift(tmp_path: Path) -> None:
    """pretrain 允许事后改 OBJECTIVES_CFG 权重，但公式身份必须 fail-closed。"""

    method = _Method()
    path = tmp_path / "loss_calibration.yaml"
    resolved = {
        "method": {
            "state_measure": {"kind": "scrambled_sobol_joint_limits"},
            "representation": {"field": {"bandwidth_centers_m": [0.004, 0.016, 0.064]}},
            "model": {"encoder": {"heads": {"zero_order_width": 128}}},
            "objectives": {"density": {"weight": 1.0}},
        },
        "trainer": {"sampling": {"epochs": 20, "q_per_asset_per_epoch": 256}},
    }
    _write_calibration_artifact(
        method,
        ({"density": 0.4, "kappa": 0.2, "derived_field": 0.1, "sobolev": 0.3, "chain": 0.5},),
        path,
        manifest_hash="abc",
        resolved_config=resolved,
    )
    method.declared_objective_weights = lambda: {  # type: ignore[method-assign]
        "density": 0.2,
        "kappa": 1.0,
        "derived_field": 1.0,
        "sobolev": 1.0,
        "chain": 1.0,
    }
    assert _require_calibration_identity(path, method=method, manifest_hash="abc", resolved_config=resolved)

    drifted = _Method()
    drifted.formula_identity = lambda: {  # type: ignore[method-assign]
        **method.formula_identity(),
        "density": "somewhere.else.density_objective",
    }
    with pytest.raises(ValueError, match="formula identity"):
        _require_calibration_identity(path, method=drifted, manifest_hash="abc", resolved_config=resolved)
