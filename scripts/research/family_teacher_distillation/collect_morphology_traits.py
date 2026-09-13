"""保存完整评价成员的静态构型特征，供结果后的失败类型分析使用。

关节数来自 typed HandCfg 的真实活动链。这里不读取策略成绩，也不改动
成员轴、初始化或筛选规则；同样保存初始化失败的资产。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path

from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot
from anymani.assets.bank.cohort import load_hand_asset_cohort


def main() -> None:
    """逐人口读静态源数据，并发布按资产身份可连接的分析表。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    registry_path = case / "evaluation-populations/registry.json"
    populations = json.loads(registry_path.read_text())["populations"]
    rows, inputs = [], {}
    for population in populations:
        path = Path(population["nominal_cohort"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != population["nominal_cohort_sha256"]:
            raise ValueError("nominal population changed before trait analysis")
        inputs[str(path)] = digest
        cohort = load_hand_asset_cohort(path, require_geometry_semantics=True)
        for index, (member, record) in enumerate(zip(cohort.members, cohort.partition.records)):
            hand = restore_hand_cfg_snapshot(record.container.sidecar["hand_cfg"])
            moving = {finger.name: [joint for joint in finger.joints if joint.joint_type == "revolute"] for finger in hand.fingers}
            counts = {name: len(moving.get(name, ())) for name in ("thumb", "index", "middle", "ring")}
            two = [name for name in ("index", "middle", "ring") if counts[name] == 2]
            upper = [moving[name][1].limit.upper for name in two]
            row = {
                "population": population["role"], "split": population["split"], "family": population["family"],
                "asset_row": index, "asset_id": member.asset_id, "base_design": member.provenance.mother_name,
                "finger_count": sum(value > 0 for value in counts.values()), "active_joint_count": sum(counts.values()),
                "thumb_joints": counts["thumb"], "index_joints": counts["index"],
                "middle_joints": counts["middle"], "ring_joints": counts["ring"],
                "nonthumb_one_joint_fingers": sum(counts[name] == 1 for name in ("index", "middle", "ring")),
                "nonthumb_two_joint_fingers": len(two),
                "nonthumb_two_joint_second_upper_min_deg": min(upper)*180/math.pi if upper else None,
                "nonthumb_two_joint_second_upper_max_deg": max(upper)*180/math.pi if upper else None,
                "source_urdf": str(record.container.urdf_path),
            }
            rows.append(row)
        if len(cohort.members) != population["nominal_assets"]:
            raise ValueError("trait table lost nominal members")
        del cohort
        gc.collect()
        print(json.dumps({"population": population["role"], "count": population["nominal_assets"]}), flush=True)
    output = case / "analysis/morphology-traits.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({"status": "complete-static-traits", "rows": len(rows),
        "source_cohorts": inputs, "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "policy_results_read": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
