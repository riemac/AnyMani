"""异构PPO八组TensorBoard numerator/count恢复合同。"""

from __future__ import annotations

from types import SimpleNamespace

from anymani.distill.diagnostics.analysis.rl.heterogeneous_ppo import _cell_series_summary


def test_cell_metric_sum_is_normalized_only_on_observed_reset_cohorts() -> None:
    r"""固定keys的无样本零值必须丢弃；sum/count恢复cell内mean。"""

    count_events = [
        SimpleNamespace(step=1, value=0.0),
        SimpleNamespace(step=2, value=0.25),
        SimpleNamespace(step=3, value=0.5),
    ]
    sum_events = [
        SimpleNamespace(step=1, value=0.0),
        SimpleNamespace(step=2, value=0.5),
        SimpleNamespace(step=3, value=1.5),
    ]

    summary = _cell_series_summary(sum_events, count_events, metric_is_sum=True)

    assert summary["count"] == 2
    assert summary["first_value"] == 2.0
    assert summary["last_value"] == 3.0
    assert summary["logging_schema"] == "sum_over_episode_count"
