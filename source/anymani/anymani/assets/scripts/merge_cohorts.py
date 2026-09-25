"Merges compatible source cohort locks while retaining ordered member identities."

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_union
from anymani.assets.bank.path_utils import resolve_anymani_root


def main() -> None:
    "Dispatches the selected asset command and returns its process status."

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", action="append", required=True, help='NAME=canonical-lock; repeat as needed, and preserve the order.'
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
