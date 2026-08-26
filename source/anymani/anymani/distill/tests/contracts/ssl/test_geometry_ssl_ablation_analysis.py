r"""Geometry SSL ablation 的 asset-balanced 配对 bootstrap 合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from anymani.distill.diagnostics.analysis.geometry_ssl import analyze_geometry_ssl_ablation_file

pytestmark = pytest.mark.contract


def _metrics(full: float, zero: float) -> dict[str, dict[str, float] | None]:
    """让三项 metric 共享手工可算数值，并保留一个全缺测 ablation。"""

    return {
        "full": {"density": full, "kappa": full, "derived_field": full},
        "joint_token_shuffle": {"density": zero, "kappa": zero, "derived_field": zero},
        "cross_asset_shuffle": None,
    }


def test_ablation_analysis_balances_assets_and_bootstraps_paired_differences(tmp_path: Path) -> None:
    r"""q 数不同时必须先 asset 内平均，且 95% CI 对 asset cluster 重采样。"""

    evidence = {
        "pairing_key": ["asset_id", "q_index"],
        "ablations": ["full", "joint_token_shuffle", "cross_asset_shuffle"],
        "records": [
            {"asset_id": "asset-a", "q_index": 0, "metrics": _metrics(1.0, 3.0)},
            {"asset_id": "asset-a", "q_index": 1, "metrics": _metrics(3.0, 5.0)},
            {"asset_id": "asset-b", "q_index": 0, "metrics": _metrics(10.0, 14.0)},
        ],
    }
    path = tmp_path / "validation_ablations.yaml"
    path.write_text(yaml.safe_dump(evidence, sort_keys=False), encoding="utf-8")

    analysis = analyze_geometry_ssl_ablation_file(path, bootstrap_samples=1_000, seed=17)

    # full 的 asset 均值分别为 2 和 10，morphology 等权均值应为 6，而不是逐 q 均值 14/3。
    assert analysis["metrics"]["full"]["density"]["asset_balanced_mean"] == pytest.approx(6.0)
    paired = analysis["paired_differences"]["joint_token_shuffle"]["density"]
    assert paired["estimate"] == pytest.approx(3.0)  # asset A 差 2、asset B 差 4，再等权平均
    assert paired["ci95_low"] == pytest.approx(2.0)
    assert paired["ci95_high"] == pytest.approx(4.0)
    assert paired["full_better"] is True
    missing = analysis["paired_differences"]["cross_asset_shuffle"]["density"]
    assert missing == {
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "asset_count": 0,
        "full_better": False,
    }


def test_ablation_analysis_rejects_duplicate_asset_q_pair(tmp_path: Path) -> None:
    """重复 `(asset_id,q_index)` 不得通过增加权重影响 CI。"""

    record = {"asset_id": "asset-a", "q_index": 0, "metrics": _metrics(1.0, 2.0)}
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pairing_key": ["asset_id", "q_index"],
                "ablations": ["full", "joint_token_shuffle", "cross_asset_shuffle"],
                "records": [record, record],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate ablation pairing key"):
        analyze_geometry_ssl_ablation_file(path, bootstrap_samples=10, seed=17)
