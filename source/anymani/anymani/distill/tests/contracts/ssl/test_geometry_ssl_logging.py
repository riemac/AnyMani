from __future__ import annotations

import json
from pathlib import Path

from anymani.distill.diagnostics.recording.geometry_ssl import GeometrySSLRunLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def test_logger_preserves_update_counters_and_new_pair_epoch_axis(tmp_path: Path) -> None:
    r"""逐 update JSONL 保存全部预算轴，epoch TensorBoard 以新 pair 数为横轴。"""

    logger = GeometrySSLRunLogger(tmp_path)
    logger.log_terms(
        optimizer_update=1,
        epoch=1,
        mini_epoch=0,
        minibatch_in_epoch=0,
        global_minibatch=0,
        new_pairs_seen=512,
        pair_uses=512,
        teacher_pairs_realized=2_048,
        microbatches_consumed=8,
        wall_time_seconds=1.25,
        split="train",
        terms={"density": 2.0, "kappa": 0.5, "derived_field": 0.25},
        denominators={"density": 512.0, "kappa": 512.0, "derived_field": 512.0},
        asset_ids=("a",),
        gradient_norm=3.0,
        total=2.75,
    )
    logger.log_epoch_terms(
        epoch=1,
        new_pairs_seen=2_048,
        pair_uses=2_048,
        optimizer_updates=4,
        terms={"density": 1.5, "kappa": 0.4, "derived_field": 0.2},
    )
    logger.close()

    record = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["optimizer_update"] == 1
    assert record["minibatch_reuse_identity"] == [0, 0]
    assert record["new_pairs_seen"] == 512
    assert record["pair_uses"] == 512
    assert record["teacher_pairs_realized"] == 2_048
    assert record["denominators"]["density"] == 512.0
    assert "step" not in record

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    epoch_density = events.Scalars("train_epoch/density")
    assert [(item.step, item.value) for item in epoch_density] == [(2_048, 1.5)]
