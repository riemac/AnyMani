"""归纳首轨迹的早、中、晚段净旋转，避免把自动重置后的运动并入原回合。

这是一项训练/开发集行为诊断，不改变正式一圈门，也不用于最终未见手选模。
每段先在同一副本内求增量，再跨R16取中位数；它不同于两个独立中位数相减。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    """从经过身份绑定的逐步轨迹计算三段10秒进展。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.evaluation.read_text())
    identity = document["evaluation_identity"]
    trace = document["step_trace"]
    if identity["protocol"]["policy_steps"] != 600 or identity["protocol"]["replicas_per_asset"] != 16:
        raise ValueError("phase dynamics requires fixed30s/R16")
    path = Path(trace["path"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != trace["sha256"]:
        raise ValueError("trace changed before phase analysis")
    with h5py.File(path) as handle:
        active = np.asarray(handle["active"], dtype=bool)
        radians = np.asarray(handle["net_rotation_rad"], dtype=np.float64)
    with h5py.File(document["trajectory_hdf5"]) as handle:
        final_net = np.asarray(handle["signed_net_turns"], dtype=np.float64)
        safe = ~np.asarray(handle["termination_drop"], dtype=bool) & ~np.asarray(handle["termination_axis"], dtype=bool)
    if active.shape != radians.shape or active.shape[0] != 600 or active.shape[-1] != 16:
        raise ValueError("phase trace axes changed")
    if not active[0].all() or np.any(active[1:] & ~active[:-1]):
        raise ValueError("first-trajectory activity must begin true and never reactivate")

    # active 在 terminal step 仍为真，随后永久为假；用最后有效索引冻结终局。
    valid_index = np.where(active, np.arange(600)[:, None, None], 0)
    valid_index = np.maximum.accumulate(valid_index, axis=0)
    frozen_net = np.take_along_axis(radians, valid_index, axis=0) / (2*np.pi)
    if not np.allclose(frozen_net[-1], final_net, rtol=0.0, atol=2e-6):
        raise ValueError("frozen step trajectory does not reproduce terminal net turns")
    phases = np.stack((frozen_net[199], frozen_net[399]-frozen_net[199], frozen_net[599]-frozen_net[399]))
    medians = np.median(phases, axis=2)
    members = identity["evaluated_cohort"]["members"]
    rows = []
    for index, member in enumerate(members):
        qualified_copy = safe[index] & (final_net[index] >= 1.0)
        stopped_late = qualified_copy & (phases[2, index] < 0.1)
        rows.append(dict(asset_id=member["asset_id"], evaluated_row=index,
                         net_turns_median=float(np.median(final_net[index])),
                         first10s_net_increment_median=float(medians[0, index]),
                         middle10s_net_increment_median=float(medians[1, index]),
                         last10s_net_increment_median=float(medians[2, index]),
                         safe_replicas=int(safe[index].sum()), safe_one_turn_replicas=int(qualified_copy.sum()),
                         safe_one_turn_but_last10s_below_point1_replicas=int(stopped_late.sum())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "audited-descriptive-dynamics", "assets": len(rows),
               "phase_seconds": [10, 10, 10], "first_trajectory_reconstruction": "freeze at last active step",
               "median_asset_phase_increments": np.median(medians, axis=1).tolist(),
               "criterion_unchanged": True, "trace_sha256": trace["sha256"]}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
