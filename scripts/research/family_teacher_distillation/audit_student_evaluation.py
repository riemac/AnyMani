r"""从原始 R16 轨迹重计共享学生成绩，并保留完整预注册资产分母。

评价的实际成员来自其独立 evaluated_cohort，教师权重只作为环境来源。
原始净圈、绝对路径和终止位在 CPU float64 归约；初始化失败的成员单独
标为未取得轨迹，达标布尔值为 False，连续运动量保持空值。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml


def sha256(path: Path) -> str:
    """绑定真实文件字节；路径不充当检查点或原始轨迹的身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """核查同一模型与标准物理协议，输出全分母 CSV 和确定性统计。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--nominal-cohort", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--teacher-evaluation", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.evaluation.read_text())
    identity = document["evaluation_identity"]
    protocol = identity["protocol"]
    if document["artifact_type"] != "anymani.family_student_fixed_evaluation":
        raise ValueError("input is not a frozen offline-student evaluation")
    if (protocol["policy_steps"], protocol["replicas_per_asset"], protocol["policy_dt_s"]) != (600, 16, 0.05):
        raise ValueError("comparison requires fixed 30-second R16 evaluation")
    if not protocol["first_trajectory_only"] or not protocol["deterministic_actor_mean"] or protocol["adr_enabled"]:
        raise ValueError("comparison protocol changed")
    if protocol["actor_contact"] != "tip-only-binary" or protocol["residual_off_intervention"]:
        raise ValueError("comparison Actor information/control route changed")
    if protocol.get("actor_contact_intervention", "none") != "none" or protocol.get("direct_logit_gain_intervention", 1.0) != 1.0 or protocol.get("actor_relay") is not None:
        raise ValueError("comparison contains a diagnostic intervention")

    # 模型来源与 seed 从训练运行记录读取，随后绑定对应 best.pt 的实际 SHA。
    report_path = args.model_run / "model/training-report.json"
    report = json.loads(report_path.read_text())
    model = args.model_run / "model/best.pt"
    if sha256(model) != identity["checkpoint_sha256"] or Path(document["checkpoint"]).resolve() != model.resolve():
        raise ValueError("evaluation and training run refer to different checkpoint files")
    run = report["run_identity"]
    if run["dataset_sha256"] != identity["dataset_sha256"] or run["representation"] != identity["variant"]:
        raise ValueError("evaluation and training data/condition disagree")
    nominal = yaml.safe_load(args.nominal_cohort.read_text())["members"]
    actual = identity["evaluated_cohort"]["members"]
    actual_path = Path(identity["evaluated_cohort"]["path"])
    if sha256(actual_path) != identity["evaluated_cohort"]["sha256"] or sha256(actual_path) != identity["manifest_sha256"]:
        raise ValueError("evaluated cohort file/identity digests disagree")
    if yaml.safe_load(actual_path.read_text())["members"] != actual:
        raise ValueError("serialized evaluated members disagree with the actual cohort file")
    nominal_ids = [row["asset_id"] for row in nominal]
    actual_ids = [row["asset_id"] for row in actual]
    if len(set(nominal_ids)) != len(nominal_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("asset axes must be unique")
    if not set(actual_ids).issubset(nominal_ids):
        raise ValueError("evaluated population is outside the fixed nominal cohort")

    # 原始终局数组为 [A,16]；中位数比值与冻结正式统计保持同一含义。
    trajectory = Path(document["trajectory_hdf5"])
    if sha256(trajectory) != document["trajectory_hdf5_sha256"]:
        raise ValueError("raw trajectory hash disagrees with evaluation")
    with h5py.File(trajectory) as handle:
        terminal_metadata = json.loads(handle.attrs["metadata_json"])
        if terminal_metadata["identity_digest"] != identity["identity_digest"]:
            raise ValueError("terminal trajectory metadata belongs to another evaluation")
        net = np.asarray(handle["signed_net_turns"], dtype=np.float64)
        path = np.asarray(handle["absolute_path_turns"], dtype=np.float64)
        safe = ~np.asarray(handle["termination_drop"], dtype=bool) & ~np.asarray(handle["termination_axis"], dtype=bool)
    if net.shape != (len(actual), 16) or path.shape != net.shape or safe.shape != net.shape:
        raise ValueError("raw trajectory axes disagree with the evaluated population")
    if not np.isfinite(net).all() or not np.isfinite(path).all() or (path < 0).any():
        raise ValueError("raw physical measurements are invalid")
    # 完整逐步轨迹是交付证据：核对真实20Hz时轴、动作张量与同一评价身份。
    trace = document["step_trace"]
    trace_path = Path(trace["path"])
    if protocol["trace_stride"] != 1 or trace["samples"] != 600 or sha256(trace_path) != trace["sha256"]:
        raise ValueError("full20Hz step trajectory is missing or changed")
    with h5py.File(trace_path) as handle:
        trace_metadata = json.loads(handle.attrs["metadata_json"])
        if trace_metadata["identity_digest"] != identity["identity_digest"]:
            raise ValueError("step trajectory metadata belongs to another evaluation")
        if not np.array_equal(handle["policy_step"][...], np.arange(1, 601)):
            raise ValueError("step trajectory does not cover the exact1..600 time axis")
        if handle["active"].shape != (600, len(actual), 16) or handle["action"].shape != (600, len(actual), 16, 16):
            raise ValueError("step trajectory asset/replica/action axes disagree")
        for start in range(0, 600, 64):
            actions = handle["action"][start:start+64]
            if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.0 + 1e-6:
                raise ValueError("recorded policy actions are nonfinite or outside [-1,1]")
    medians = np.median(net, axis=1)
    path_medians = np.median(path, axis=1)
    direction = np.minimum(np.maximum(medians, 0.0) / np.maximum(path_medians, 2.0**-23), 1.0)
    safe_counts = safe.sum(axis=1)
    passed = (medians >= 1.0) & (direction >= 0.7) & (safe_counts >= 12)
    formal = document["reliable_topology_coverage"]
    if not formal["finite"] or int(passed.sum()) != formal["passed_asset_count"]:
        raise ValueError("raw recount disagrees with the frozen evaluator")
    recorded = document["physical_rotation"]["asset_results"]
    for index, result in enumerate(recorded):
        if not np.allclose(
            [medians[index], path_medians[index], direction[index], safe_counts[index] / 16],
            [result["net_turns_median"], result["absolute_path_turns_median"], result["directional_consistency"], result["safe_replica_fraction"]],
            rtol=0.0, atol=1e-9,
        ):
            raise ValueError(f"raw per-asset recount differs at evaluated row {index}")
    if [row["dataset_row"] for row, accepted in zip(recorded, passed) if accepted] != formal["passed_asset_rows"]:
        raise ValueError("raw passed-asset identities disagree with the evaluator")
    indexed = {asset_id: index for index, asset_id in enumerate(actual_ids)}
    rows = []
    for row_index, member in enumerate(nominal):
        index = indexed.get(member["asset_id"])
        provenance = member["provenance"]
        group = f"{provenance['group_name']}/{provenance['mother_name']}"
        row = dict(variant=run["representation"], seed=run["seed"], asset_row=row_index,
                   asset_id=member["asset_id"], base_design=group, evaluated=index is not None,
                   outcome="measured" if index is not None else "pregrasp-unavailable",
                   passed=False, net_turns_median=None, absolute_path_median=None,
                   direction=None, safe_replicas=None, two_turn_passed=False)
        if index is not None:
            row.update(passed=bool(passed[index]), net_turns_median=float(medians[index]),
                       absolute_path_median=float(path_medians[index]), direction=float(direction[index]),
                       safe_replicas=int(safe_counts[index]),
                       two_turn_passed=bool(passed[index] and medians[index] >= 2.0))
        rows.append(row)

    # 按完整母型分母统计；未初始化成员仍占据其预注册位置。
    totals = Counter(row["base_design"] for row in rows)
    successes = Counter(row["base_design"] for row in rows if row["passed"])
    summary = dict(status="audited", variant=run["representation"], seed=run["seed"],
                   nominal_assets=len(rows), evaluated_assets=len(actual), initialization_failures=len(rows)-len(actual),
                   passed_assets=sum(row["passed"] for row in rows), two_turn_assets=sum(row["two_turn_passed"] for row in rows),
                   base_design_count=len(totals), passed_base_designs=sum(successes[key] >= math.ceil(value/2) for key, value in totals.items()),
                   checkpoint_sha256=identity["checkpoint_sha256"], evaluation_sha256=sha256(args.evaluation),
                   nominal_cohort_sha256=sha256(args.nominal_cohort), runtime_cohort_sha256=sha256(actual_path),
                   training_report_sha256=sha256(report_path),
                   trajectory_sha256=document["trajectory_hdf5_sha256"], step_trace_sha256=trace["sha256"],
                   raw_step_trace_verified=True)
    if args.teacher_evaluation is not None:
        teacher = json.loads(args.teacher_evaluation.read_text())
        if teacher["evaluation_identity"]["manifest_sha256"] != sha256(args.nominal_cohort):
            raise ValueError("teacher retention requires the exact full nominal training cohort")
        teacher_rows = set(teacher["reliable_topology_coverage"]["passed_asset_rows"])
        summary.update(teacher_original_passed=len(teacher_rows), teacher_original_retained=sum(row["passed"] and row["asset_row"] in teacher_rows for row in rows))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["asset_table_sha256"] = sha256(args.output_prefix.with_suffix(".csv"))
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
