r"""Density + Gamma asset-level paired ablation bootstrap 合同。"""

from __future__ import annotations

import pytest
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime


def test_asset_level_ablation_reports_paired_delta_and_ci() -> None:
    r"""先在 morphology 内聚合 q 后，candidate-full 差值应按 asset 做 bootstrap。"""

    config = compose_evaluation_cfg(config_ref="geometry_ssl_density_material_jacobian_v0_8_0")
    method = build_runtime(config.method)
    evidence = {
        "ablations": ("full", "query_only"),
        "aggregate_metrics": {
            "full": {"density": 1.5, "material_jacobian": 2.5},
            "query_only": {"density": 2.5, "material_jacobian": 4.5},
        },
        "records": [
            {
                "asset_id": "a",
                "metrics": {
                    "full": {"density": 1.0, "material_jacobian": 2.0},
                    "query_only": {"density": 2.0, "material_jacobian": 4.0},
                },
            },
            {
                "asset_id": "b",
                "metrics": {
                    "full": {"density": 2.0, "material_jacobian": 3.0},
                    "query_only": {"density": 3.0, "material_jacobian": 5.0},
                },
            },
        ],
    }
    report = method.analyze_ablations(evidence, bootstrap_replicates=200, seed=17)

    assert report["record_count"] == 2
    assert report["paired_delta"]["query_only"]["density"]["mean_delta"] == pytest.approx(1.0)
    assert report["paired_delta"]["query_only"]["material_jacobian"]["mean_delta"] == pytest.approx(2.0)
    assert report["paired_delta"]["query_only"]["density"]["ci95"] == pytest.approx([1.0, 1.0])
