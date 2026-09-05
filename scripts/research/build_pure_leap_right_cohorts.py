r"""确定性构建pure LEAP-right A64/A128 member-level cohort locks。

该入口只编排``assets.bank.cohort_selection``中的版本化recipe与统一lock writer；不读取RL checkpoint，
不启动Isaac，也不允许通过CLI临时改写cell配额或成员数。发布后重新加载lock，打印其byte SHA与lineage分布。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from anymani.assets.bank.cohort import load_hand_asset_cohort
from anymani.assets.bank.cohort_selection import (
    PURE_LEAP_RIGHT_A64_RECIPE,
    PURE_LEAP_RIGHT_A128_RECIPE,
    write_lineage_cohort_lock,
)
from anymani.assets.bank.path_utils import resolve_anymani_root

ROOT = resolve_anymani_root()  # AnyMani仓库根，所有默认路径由此形成absolute identity
DATASET_ROOT = ROOT / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
COHORT_ROOT = DATASET_ROOT / "cohorts"


def _parse_args() -> argparse.Namespace:
    r"""解析唯一允许变化的scale与可选输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(64, 128), required=True)
    parser.add_argument("--output", type=Path, default=None, help="缺省写入dataset的cohorts目录。")
    return parser.parse_args()


def main() -> None:
    r"""按冻结recipe发布lock，并重新解析以验证父manifest和逐成员identity。"""

    args = _parse_args()
    recipe = PURE_LEAP_RIGHT_A64_RECIPE if args.scale == 64 else PURE_LEAP_RIGHT_A128_RECIPE
    output = args.output or COHORT_ROOT / f"{recipe.cohort_id}.lock.yaml"
    source_manifests: dict[str, Path] = {"ppo": DATASET_ROOT / "ppo.yaml"}
    if args.scale == 128:
        source_manifests["ssl"] = DATASET_ROOT / "ssl.yaml"  # A128才允许引入SSL-only topology

    published = write_lineage_cohort_lock(
        output,
        recipe=recipe,
        source_manifests=source_manifests,
    )
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
