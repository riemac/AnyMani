"""按真实手型属性归纳已完成的比较，区分初始化失败与策略失败。

这是描述性构型分析。每格保存原资产分母与训练种子，不把小分组中的
差异解释为独立机制证据，也不使用这些结果改写最终留出成员。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取显式分析输入，不扫描未完成运行或猜测资产类型。"""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """连接固定资产身份，输出可复核的逐种子构型分层表。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    traits = read_csv(case / "analysis/morphology-traits.csv")
    by_asset = {(row["population"], row["asset_id"]): row for row in traits}
    if len(by_asset) != len(traits):
        raise ValueError("static trait identities repeat")
    results = read_csv(case / "comparison/per-asset.csv")
    grouped = defaultdict(lambda: dict(assets=0, evaluated=0, passed=0, two_turn=0, net=[]))
    for row in results:
        trait = by_asset[(row["population"], row["asset_id"])]
        short_column = "nonthumb_one_joint_fingers" if trait["family"] == "leap" else "nonthumb_two_joint_fingers"
        categories = {
            "finger_count": trait["finger_count"],
            "active_joint_count": trait["active_joint_count"],
            "family_specific_short_nonthumb_count": trait[short_column],
        }
        for category, value in categories.items():
            key = (row["variant"], int(row["seed"]), row["population"], row["split"], trait["family"], category, int(value))
            group = grouped[key]
            group["assets"] += 1
            group["evaluated"] += row["evaluated"] == "True"
            group["passed"] += row["passed"] == "True"
            group["two_turn"] += row["two_turn_passed"] == "True"
            if row["net_turns_median"]:
                group["net"].append(float(row["net_turns_median"]))
    rows = []
    for key, group in sorted(grouped.items()):
        variant, seed, population, split, family, category, value = key
        rows.append(dict(variant=variant, seed=seed, population=population, split=split, family=family,
                         category=category, category_value=value, nominal_assets=group["assets"],
                         evaluated_assets=group["evaluated"], initialization_failures=group["assets"]-group["evaluated"],
                         passed_assets=group["passed"], policy_failures=group["evaluated"]-group["passed"],
                         two_turn_assets=group["two_turn"], success_rate=group["passed"]/group["assets"],
                         evaluated_mean_asset_net_turns=sum(group["net"])/len(group["net"]) if group["net"] else None))
    output = case / "comparison/morphology-strata.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observed morphology strata; no missing runs were imputed.")


if __name__ == "__main__":
    main()
