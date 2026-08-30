r"""SSL lifecycle 的 objective 名称必须由 concrete method 声明。"""

from __future__ import annotations

import pytest

from anymani.distill.ssl.runtime.lifecycle import _format_objective_terms

pytestmark = pytest.mark.contract


def test_epoch_summary_accepts_density_and_relational_jacobian_terms() -> None:
    r"""新 method 的 Gamma term 不应触发 Trainer 中历史 κ 名称假设。"""

    summary = _format_objective_terms({"density": 0.125, "material_jacobian": 0.03125})
    assert summary == "density=1.250000e-01 material_jacobian=3.125000e-02"


def test_epoch_summary_rejects_empty_method_update() -> None:
    r"""没有 objective 的 update 属于 method/runtime 合同错误。"""

    with pytest.raises(ValueError, match="at least one"):
        _format_objective_terms({})
