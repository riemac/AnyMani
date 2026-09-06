r"""确定性构建pure LEAP-right A64/A128 member-level cohort locks。

该入口只编排``assets.bank.cohort_selection``中的版本化recipe与统一lock writer；不读取RL checkpoint，
不启动Isaac，也不允许通过CLI临时改写cell配额或成员数。研究留出可通过具名派生recipe排除整个mother，
仍从同一train池补足原配额；派生集合使用新名称与新文件，发布后重新加载并验证lineage分布。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from anymani.assets.bank.cohort import load_hand_asset_cohort
from anymani.assets.bank.cohort_selection import (
    PURE_LEAP_RIGHT_A64_RECIPE,
    PURE_LEAP_RIGHT_A128_RECIPE,
    write_lineage_cohort_lock,
)
from anymani.assets.bank.path_utils import resolve_anymani_root, resolve_bank_path

ROOT = resolve_anymani_root()  # AnyMani仓库根，所有默认路径由此形成absolute identity
DATASET_ROOT = ROOT / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
COHORT_ROOT = DATASET_ROOT / "cohorts"


def _parse_args() -> argparse.Namespace:
    r"""解析规模、具名研究留出与输出路径；cell配额由已有recipe确定。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(64, 128), required=True)
    parser.add_argument("--output", type=Path, default=None, help="缺省写入dataset的cohorts目录。")
    parser.add_argument("--cohort-id", type=str, default=None, help="派生集合的新名称，不复用历史集合ID。")
    parser.add_argument(
        "--exclude-mother", action="append", default=[], help="可重复指定完整mother名称，排除其全部变体。"
    )
    return parser.parse_args()


def main() -> None:
    r"""按冻结recipe发布lock，并重新解析以验证父manifest和逐成员identity。"""

    args = _parse_args()
    recipe = PURE_LEAP_RIGHT_A64_RECIPE if args.scale == 64 else PURE_LEAP_RIGHT_A128_RECIPE
    if args.exclude_mother and (not args.cohort_id or args.cohort_id == recipe.cohort_id):
        raise ValueError("--exclude-mother requires a new --cohort-id")
    if args.cohort_id is not None:
        recipe = replace(
            recipe, cohort_id=args.cohort_id, excluded_mother_names=tuple(args.exclude_mother)
        )  # 派生支持集记录自己的排除名单，原A64/A128 recipe保持不变。
    output = resolve_bank_path(args.output or COHORT_ROOT / f"{recipe.cohort_id}.lock.yaml")
    if output.exists():
        raise FileExistsError(f"cohort output already exists: {output}")
    source_manifests: dict[str, Path] = {"ppo": DATASET_ROOT / "ppo.yaml"}
    if args.scale == 128:
        source_manifests["ssl"] = DATASET_ROOT / "ssl.yaml"  # A128才允许引入SSL-only topology

    # 所有名称都代表冻结集合。先在独占目录验证，再原子新建目标，保护并发出现的同名文件。
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lineage-cohort-", dir=output.parent) as temporary:
        staged = write_lineage_cohort_lock(
            Path(temporary) / "source.lock.yaml",
            recipe=recipe,
            source_manifests=source_manifests,
        )
        os.link(staged, output)
    published = output
    cohort = load_hand_asset_cohort(published, require_geometry_semantics=True)
    mothers = Counter(member.provenance.mother_name for member in cohort.members)
    sources = Counter(member.source_alias for member in cohort.members)
    print(
        json.dumps(
            {
                "cohort_id": cohort.cohort_id,
                "asset_count": len(cohort.members),
                "mother_count": len(mothers),
                "members_per_mother": sorted(set(mothers.values())),
                "source_member_counts": dict(sorted(sources.items())),
                "lock_path": str(cohort.lock_path),
                "lock_sha256": cohort.lock_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
