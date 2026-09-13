"""合并配对种子的完整评价表；缺少实验时保持 pending，不补造结果。

波动以训练种子为单位统计。R16 副本先在每个资产内归约，因此不会被当作
额外的独立训练种子。初始化失败保留在各预注册分母中，连续量仍为空值。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    """保存可直接绘图的平表，仅在有真实记录时写入数据行。"""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_quantile(values: list[float], quantile: float) -> float | None:
    """采用位置 (n-1)q 的线性分位数；未测量资产不补成零。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_endpoint_tables(output: Path, trials: list[dict], assets: list[dict], populations: list[dict]) -> None:
    """归并两族与留出类别；每个端点须具有该种子的全部组成集合。"""
    endpoints = {row["role"]: [row["role"]] for row in populations}
    for split in ("training", "development", "strict-unseen"):
        endpoints[split + "_all"] = [row["role"] for row in populations if row["split"] == split]
    for family in ("leap", "allegro"):
        endpoints["strict-unseen_" + family] = [family + "_right_variant", family + "_right_mother"]
    endpoints["strict-unseen_variants"] = ["leap_right_variant", "allegro_right_variant"]
    endpoints["strict-unseen_base_designs"] = ["leap_right_mother", "allegro_right_mother"]
    rows = []
    for endpoint, roles in endpoints.items():
        for variant in ("n040", "no_z", "fk"):
            for seed in (42, 43, 44):
                parts = [row for row in trials if row["variant"] == variant and row["seed"] == seed and row["population"] in roles]
                if len(parts) != len(roles):
                    continue  # 未完成的一个族不能被隐式当作零，或从分母中消失。
                members = [row for row in assets if row["variant"] == variant and int(row["seed"]) == seed and row["population"] in roles]
                nominal = sum(row["nominal_assets"] for row in parts)
                assert len(members) == nominal
                measured = [row for row in members if row["evaluated"] == "True"]
                net = [float(row["net_turns_median"]) for row in measured]
                assert all(math.isfinite(value) for value in net)
                all_teacher = all(row["teacher_original_passed"] is not None for row in parts)
                teacher_passed = sum(row["teacher_original_passed"] for row in parts) if all_teacher else None
                retained = sum(row["teacher_original_retained"] for row in parts) if all_teacher else None
                passed = sum(row["passed_assets"] for row in parts)
                rows.append(dict(endpoint=endpoint, variant=variant, seed=seed, nominal_assets=nominal,
                                 evaluated_assets=len(measured), initialization_failures=nominal-len(measured),
                                 passed_assets=passed, success_rate=passed/nominal,
                                 two_turn_assets=sum(row["two_turn_assets"] for row in parts),
                                 teacher_original_passed=teacher_passed, teacher_original_retained=retained,
                                 teacher_retention_fraction=retained/teacher_passed if teacher_passed else None,
                                 newly_passing_assets=passed-retained if retained is not None else None,
                                 teacher_passing_assets_lost=teacher_passed-retained if retained is not None else None,
                                 measured_mean_asset_net_turns=statistics.mean(net) if net else None,
                                 measured_asset_net_p10=linear_quantile(net, .1),
                                 measured_asset_net_median=linear_quantile(net, .5),
                                 measured_asset_net_p90=linear_quantile(net, .9),
                                 measured_mean_direction=statistics.mean(float(row["direction"]) for row in measured) if measured else None,
                                 measured_safe_replica_fraction=sum(int(row["safe_replicas"]) for row in measured)/(16*len(measured)) if measured else None,
                                 measured_net_gate_failures=sum(value < 1 for value in net),
                                 measured_direction_gate_failures=sum(float(row["direction"]) < .7 for row in measured),
                                 measured_safety_gate_failures=sum(int(row["safe_replicas"]) < 12 for row in measured),
                                 measured_low_path_assets_025=sum(float(row["absolute_path_median"]) < .25 for row in measured)))
    metrics = ("passed_assets", "success_rate", "two_turn_assets", "teacher_original_retained", "teacher_retention_fraction",
               "measured_mean_asset_net_turns", "measured_asset_net_p10", "measured_asset_net_median",
               "measured_safe_replica_fraction", "measured_mean_direction")
    aggregates, paired = [], []
    for endpoint in endpoints:
        for variant in ("n040", "no_z", "fk"):
            selected = [row for row in rows if row["endpoint"] == endpoint and row["variant"] == variant]
            for metric in metrics:
                values = [row[metric] for row in selected if row[metric] is not None]
                if values:
                    aggregates.append(dict(endpoint=endpoint, variant=variant, metric=metric, seed_count=len(values),
                                           nominal_assets=selected[0]["nominal_assets"], mean=statistics.mean(values),
                                           sample_std=statistics.stdev(values) if len(values)>1 else None,
                                           minimum=min(values), maximum=max(values)))
        for seed in (42, 43, 44):
            selected = {row["variant"]: row for row in rows if row["endpoint"] == endpoint and row["seed"] == seed}
            for baseline in ("no_z", "fk"):
                if "n040" not in selected or baseline not in selected:
                    continue
                for metric in metrics:
                    ours, other = selected["n040"][metric], selected[baseline][metric]
                    if ours is not None and other is not None:
                        paired.append(dict(endpoint=endpoint, seed=seed, comparison="Ours-minus-"+baseline,
                                           metric=metric, difference=ours-other))
    write_csv(output / "endpoint-per-seed.csv", rows)
    write_csv(output / "endpoint-summary.csv", aggregates)
    write_csv(output / "endpoint-paired-differences.csv", paired)


def main() -> None:
    """核查固定九组与八个人口集合，生成逐种子结果和配对差值。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    case = args.case.resolve()
    populations = json.loads((case / "evaluation-populations/registry.json").read_text())["populations"]
    trials, missing, asset_rows, base_rows = [], [], [], []
    model_hashes = {}
    for variant in ("n040", "no_z", "fk"):
        for seed in (42, 43, 44):
            name = f"formal-v1-{variant}-s{seed}"
            for population in populations:
                role = population["role"]
                suffix = f"{population['family']}-training" if population["split"] == "training" else role
                prefix = case / "analysis" / f"{name}-{suffix}"
                if not prefix.with_suffix(".json").exists():
                    missing.append({"variant": variant, "seed": seed, "population": role})
                    continue
                result = json.loads(prefix.with_suffix(".json").read_text())
                if result.get("raw_step_trace_verified") is not True or not result.get("step_trace_sha256"):
                    missing.append({"variant": variant, "seed": seed, "population": role,
                                    "reason": "full raw step-trajectory audit required"})
                    continue
                if result["variant"] != variant or result["seed"] != seed:
                    raise ValueError(f"trial identity mismatch: {prefix}")
                if result["nominal_assets"] != population["nominal_assets"] or result["evaluated_assets"] != population["ready_assets"]:
                    raise ValueError(f"trial population differs from the frozen registry: {prefix}")
                if result["nominal_cohort_sha256"] != population["nominal_cohort_sha256"]:
                    raise ValueError(f"nominal cohort changed: {prefix}")
                if result.get("runtime_cohort_sha256") != population["runtime_cohort_sha256"]:
                    raise ValueError(f"runtime cohort differs from the exact frozen ready subset: {prefix}")
                if result.get("asset_table_sha256") != hashlib.sha256(prefix.with_suffix(".csv").read_bytes()).hexdigest():
                    raise ValueError(f"asset table differs from the audited evaluation: {prefix}")
                previous = model_hashes.setdefault(name, result["checkpoint_sha256"])
                if previous != result["checkpoint_sha256"]:
                    raise ValueError(f"different checkpoints were used across populations: {name}")
                trials.append(dict(variant=variant, seed=seed, population=role, split=population["split"],
                                   passed_assets=result["passed_assets"], nominal_assets=result["nominal_assets"],
                                   success_rate=result["passed_assets"]/result["nominal_assets"],
                                   evaluated_assets=result["evaluated_assets"], initialization_failures=result["initialization_failures"],
                                   two_turn_assets=result["two_turn_assets"], passed_base_designs=result["passed_base_designs"],
                                   base_design_count=result["base_design_count"],
                                   teacher_original_retained=result.get("teacher_original_retained"),
                                   teacher_original_passed=result.get("teacher_original_passed"),
                                   checkpoint_sha256=result["checkpoint_sha256"]))
                with prefix.with_suffix(".csv").open() as stream:
                    members = list(csv.DictReader(stream))
                if len(members) != result["nominal_assets"] or len({row["asset_id"] for row in members}) != len(members):
                    raise ValueError(f"asset table has an invalid nominal member axis: {prefix}")
                bases = defaultdict(list)
                for row in members:
                    if row["variant"] != variant or int(row["seed"]) != seed:
                        raise ValueError(f"asset table has a different model identity: {prefix}")
                    asset_rows.append({"population": role, "split": population["split"], **row})
                    bases[row["base_design"]].append(row)
                trial_bases = []
                for base, rows in sorted(bases.items()):
                    passed = sum(row["passed"] == "True" for row in rows)
                    measured = [row for row in rows if row["evaluated"] == "True"]
                    # 与原审计相同：每个母型至少一半的名义成员通过，未初始化成员不剔除。
                    trial_bases.append(dict(variant=variant, seed=seed, population=role, split=population["split"],
                                            base_design=base, nominal_assets=len(rows), evaluated_assets=len(measured),
                                            initialization_failures=len(rows)-len(measured), passed_assets=passed,
                                            passed_base_design=passed >= math.ceil(len(rows)/2),
                                            two_turn_assets=sum(row["two_turn_passed"] == "True" for row in rows),
                                            measured_median_asset_net_turns=statistics.median(float(row["net_turns_median"]) for row in measured) if measured else None))
                if sum(row["passed_assets"] for row in trial_bases) != result["passed_assets"] or sum(row["passed_base_design"] for row in trial_bases) != result["passed_base_designs"]:
                    raise ValueError(f"base-design recount disagrees with the raw audit: {prefix}")
                base_rows.extend(trial_bases)
    aggregates = []
    for variant in ("n040", "no_z", "fk"):
        for population in populations:
            rows = [row for row in trials if row["variant"] == variant and row["population"] == population["role"]]
            if rows:
                values = [row["passed_assets"] for row in rows]
                aggregates.append(dict(variant=variant, population=population["role"], seed_count=len(values),
                                       mean_passed=statistics.mean(values), sample_std_passed=statistics.stdev(values) if len(values)>1 else None,
                                       minimum_passed=min(values), maximum_passed=max(values), nominal_assets=population["nominal_assets"]))
    paired = []
    for population in populations:
        for seed in (42, 43, 44):
            rows = {row["variant"]: row for row in trials if row["seed"]==seed and row["population"]==population["role"]}
            for baseline in ("no_z", "fk"):
                if "n040" in rows and baseline in rows:
                    paired.append(dict(population=population["role"], seed=seed, comparison=f"Ours-minus-{baseline}",
                                       passed_asset_difference=rows["n040"]["passed_assets"]-rows[baseline]["passed_assets"],
                                       two_turn_asset_difference=rows["n040"]["two_turn_assets"]-rows[baseline]["two_turn_assets"]))
    output = case / "comparison"
    output.mkdir(exist_ok=True)
    write_csv(output / "per-seed.csv", trials)
    write_csv(output / "per-asset.csv", asset_rows)
    write_csv(output / "per-base-design.csv", base_rows)
    write_csv(output / "seed-summary.csv", aggregates)
    write_csv(output / "paired-differences.csv", paired)
    write_endpoint_tables(output, trials, asset_rows, populations)
    summary = {"status": "complete" if not missing else "pending", "expected_trials": 9*len(populations),
               "completed_trials": len(trials), "missing_trials": missing,
               "uncertainty_unit": "training seed; sample standard deviation across seeds", "model_hashes": model_hashes}
    (output / "completeness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "expected_trials", "completed_trials")}))
    if args.require_complete and missing:
        raise SystemExit("formal comparison is incomplete")


if __name__ == "__main__":
    main()
