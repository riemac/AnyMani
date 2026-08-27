r"""Held-out morphology 的分层、形态等权 checkpoint-selection 合同。"""

from __future__ import annotations

import pytest
from anymani.distill.diagnostics.evaluation.geometry_ssl import aggregate_geometry_ssl_stratified_components

pytestmark = pytest.mark.contract


def test_stratified_selection_balances_q_morphologies_bins_and_axes() -> None:
    r"""聚合顺序必须为 q-denominator → morphology → bin → axis，不能按 block 或标量总数加权。"""

    # asset-a 分散在两个 q blocks：bin-0 的 numerator/denominator 合并为 10/2=5；
    # asset-b 为 10/1=10。因此 bin-0 morphology 等权为 7.5，而不是全标量 20/3。
    block_a0 = {
        "density": {"axis-0": {"bin-0": ((1.0,), (1.0,)), "bin-1": ((3.0,), (1.0,))}},
        "kappa": {"axis-0": {"bin-0": ((2.0,), (1.0,))}},
        "derived_field": {"axis-0": {"bin-0": ((4.0,), (1.0,))}},
    }
    block_a1_b0 = {
        "density": {
            "axis-0": {"bin-0": ((9.0, 10.0), (1.0, 1.0)), "bin-1": ((5.0, 7.0), (1.0, 1.0))}
        },
        "kappa": {"axis-0": {"bin-0": ((4.0, 8.0), (1.0, 1.0))}},
        "derived_field": {"axis-0": {"bin-0": ((6.0, 12.0), (1.0, 1.0))}},
    }

    evidence = aggregate_geometry_ssl_stratified_components(
        ((('asset-a',), block_a0), (("asset-a", "asset-b"), block_a1_b0))
    )

    density_bins = evidence["bin_scores"]["density"]["axis-0"]
    assert density_bins["bin-0"]["mse"] == pytest.approx(7.5)
    assert density_bins["bin-1"]["mse"] == pytest.approx(5.5)  # asset-a=(3+5)/2=4，asset-b=7
    assert evidence["axis_scores"]["density"]["axis-0"] == pytest.approx(6.5)  # 两个 bin 等权
    assert evidence["metric_scores"]["density"] == pytest.approx(6.5)  # 当前只有一个 axis
    assert evidence["morphology_scores"]["density"] == {"asset-a": 4.5, "asset-b": 8.5}


def test_stratified_selection_ignores_empty_bins_without_calling_them_zero() -> None:
    r"""某个 bin 无有效 denominator 时不得以 0 降低 axis score。"""

    components = {
        "density": {"axis": {"valid": ((2.0,), (1.0,)), "empty": ((0.0,), (0.0,))}},
        "kappa": {"axis": {"valid": ((3.0,), (1.0,))}},
        "derived_field": {"axis": {"valid": ((4.0,), (1.0,))}},
    }

    evidence = aggregate_geometry_ssl_stratified_components(((('asset-a',), components),))

    assert evidence["metric_scores"] == {"density": 2.0, "kappa": 3.0, "derived_field": 4.0}
    assert "empty" not in evidence["bin_scores"]["density"]["axis"]
