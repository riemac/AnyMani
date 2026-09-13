r"""按显式父顺序合并已验证canonical集合，保留source与物理证书。

每个父集合只解析一次，发布由cohort union API完成。命令不改父集合、不重做canonical lowering；
预抓取与训练角色的准入由各自流程验证。输出使用独立source/canonical文件，已有证据拒绝覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_union
from anymani.assets.bank.path_utils import resolve_anymani_root


def main() -> None:
    r"""按NAME=PATH父列表顺序建立联合成员轴，并输出父系与真实基数摘要。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", action="append", required=True, help="NAME=canonical-lock，可重复指定且顺序有意义。"
    )
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--purpose", default="joint-training-support")
    args = parser.parse_args()
    root = resolve_anymani_root()
    output = args.output_dir or root / "source/anymani/anymani/assets/datasets/cohorts/merged" / args.cohort_id
    output = output.expanduser().resolve()
    source_path, canonical_path = output / "source.lock.yaml", output / "canonical.lock.yaml"
    if source_path.exists() or canonical_path.exists():
        raise FileExistsError("merge output must be a new source/canonical pair")

    # 父名称只标识本次拼接来源，source alias/row仍来自各父锁，不能用父名称掩盖来源冲突。
    parents = {}
    for specification in args.parent:
        name, separator, path = specification.partition("=")
        if not separator or not name.strip() or name in parents:
            raise ValueError("parents must use unique non-empty NAME=PATH specifications")
        parents[name] = load_hand_asset_cohort(path, require_geometry_semantics=True)
    coordinates = tuple((name, index) for name, parent in parents.items() for index in range(len(parent.members)))
    write_hand_asset_cohort_union(
        parents,
        source_path,
        canonical_path,
        cohort_id=args.cohort_id,
        member_coordinates=coordinates,
        selection={"purpose": args.purpose, "algorithm": "canonical-parent-union-v1"},
    )
    print(
        json.dumps(
            {
                "cohort_id": args.cohort_id,
                "asset_count": len(coordinates),
                "parent_counts": {name: len(parent.members) for name, parent in parents.items()},
                "source_lock": str(source_path),
                "canonical_lock": str(canonical_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
