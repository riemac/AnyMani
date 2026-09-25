"Provides cohort lock, audit, and selection commands."

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from anymani.assets.bank import cohort_selection
from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock
from anymani.assets.bank.path_utils import resolve_anymani_root, resolve_bank_path


def main() -> None:
    "Dispatches the selected asset command and returns its process status."


    root = resolve_anymani_root()
    dataset_root = root / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("leap", "allegro"), required=True)
    parser.add_argument("--cohort-id", required=True, help='Unique name for this candidate preparation.')
    parser.add_argument("--output", type=Path, default=None, help='Explicit candidate-lock output path; overrides the tiered default.')
    parser.add_argument("--ppo-manifest", type=Path, default=dataset_root / "ppo.yaml")
    parser.add_argument("--ssl-manifest", type=Path, default=dataset_root / "ssl.yaml")
    parser.add_argument("--exclude-mother", action="append", default=[], help='Add an entire study holdout lineage to the exclusion list.')
    args = parser.parse_args()


    base = (
        cohort_selection.PURE_LEAP_RIGHT_A128_RECIPE
        if args.family == "leap"
        else cohort_selection.PURE_ALLEGRO_RIGHT_A128_RECIPE
    )
    exclusions = tuple(dict.fromkeys((*base.excluded_mother_names, *args.exclude_mother)))
    recipe = replace(base, cohort_id=args.cohort_id, excluded_mother_names=exclusions)
    sources = {"ppo": resolve_bank_path(args.ppo_manifest), "ssl": resolve_bank_path(args.ssl_manifest)}
    output = resolve_bank_path(
        args.output or dataset_root / "cohorts" / "pure_family" / f"{recipe.cohort_id}.lock.yaml"
    )
    if output.exists():
        raise FileExistsError(output)


    resolved = cohort_selection.resolve_lineage_cohort_selection(recipe, source_manifests=sources)
    selection = {
        **resolved.selection_document,
        "purpose": "strict-pregrasp-preparation-candidates",
        "policy_exposure": "selection-does-not-read-policy-results",
        "selector_source_sha256": hashlib.sha256(Path(cohort_selection.__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".family-cohort-", dir=output.parent) as directory:
        staged = write_hand_asset_cohort_lock(
            Path(directory) / "source.lock.yaml",
            cohort_id=recipe.cohort_id,
            source_manifests=sources,
            member_coordinates=resolved.member_coordinates,
            selection=selection,
            require_geometry_semantics=True,
        )
        os.link(staged, output)


    cohort = load_hand_asset_cohort(output, require_geometry_semantics=True)
    mothers = Counter(member.provenance.mother_path for member in cohort.members)
    source_counts = Counter(member.source_alias for member in cohort.members)
    print(
        json.dumps(
            {
                "cohort_id": cohort.cohort_id,
                "family": recipe.family,
                "asset_count": len(cohort.members),
                "mother_count": len(mothers),
                "members_per_mother": sorted(set(mothers.values())),
                "source_member_counts": dict(sorted(source_counts.items())),
                "lock_path": str(cohort.lock_path),
                "lock_sha256": cohort.lock_sha256,
                "canonical_physics": "pending",
                "strict_pregrasp": "pending",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
