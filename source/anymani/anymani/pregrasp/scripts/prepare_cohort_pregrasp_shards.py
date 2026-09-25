"""Check a resolved cohort's strict pregrasp cache and shard only its misses.

Cache hits must match hand, physical, canonical, routing, object, scale, physics, and generation identities, then pass the complete Top-8 gate again. Existing records are read-only; missing members are interleaved by morphology cell and written as bounded cohort shards.

AppLauncher builds the training-path asset binding, but this planner does not create a scene or step PhysX. Each output directory preserves the full parent mapping in source/canonical locks and ``preparation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from anymani.assets.bank.cohort import parse_hand_asset_cohort_document


def main() -> None:
    """Build bounded tasks for missing members without changing the search or strict gate.

    Only ``GoodPregraspMissError`` creates a task. Corrupt entries, identity mismatches, and strict-gate failures remain errors. Canonical identities come from the resolved binding, never from bundle content hashes.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, required=True, help="Canonical-final lock for the resolved parent cohort.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory for this preparation pass.")
    parser.add_argument("--shard-assets", type=int, default=16, help="Maximum members per generation shard; default: 16.")
    parser.add_argument("--catalog", type=Path, default=None, help="Strict catalog root; defaults to the training runtime path.")
    args = parser.parse_args()
    if args.shard_assets < 1:
        parser.error("--shard-assets must be positive")
    parent_path = args.cohort_lock.expanduser().resolve(strict=True)  # Preserve the resolved training member axis.
    output_root = args.output_dir.expanduser().resolve()  # Use a dedicated preparation evidence directory.
    if output_root.exists():
        raise FileExistsError(f"pregrasp preparation output already exists: {output_root}")
    parent = parse_hand_asset_cohort_document(parent_path.read_bytes())
    if not isinstance(parent, dict) or parent.get("schema_version") != "1.2.0":
        raise ValueError("pregrasp preparation requires a canonical-final cohort lock")

    # Fix the support set before importing task modules; this binding step does not simulate.
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(parent_path)
    os.environ["ANYMANI_HETERO_NUM_ENVS"] = "1"
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_subset
        from anymani.pregrasp.good_catalog import GoodPregraspCatalog, GoodPregraspMissError
        from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
        from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

        parent_cohort = load_hand_asset_cohort(parent_path)  # Share the verified parent across all shards.
        binding = build_generated_asset_binding()  # Use the same source/canonical/routing binding as PPO.
        assert parent_cohort.lock_sha256 == binding.cohort_lock_sha256
        reset_cfg = binding.build_good_pregrasp_reset_cfg(
            num_envs=binding.asset_count,
            catalog_root=args.catalog.expanduser().resolve() if args.catalog is not None else None,
        )  # Build runtime lookup keys without expanding the logical cohort axis.
        catalog = GoodPregraspCatalog(reset_cfg.catalog_root)
        cached_indices: list[int] = []  # Parent indices with a valid certified Top-8.
        missing_by_cell: dict[int, list[int]] = defaultdict(list)  # Group misses by morphology cell, not policy label.
        keys = tuple(item.resolve_key() for item in reset_cfg.bindings)  # Exact keys bind generation and physics identity.
        for index, key in enumerate(keys):
            try:
                entry = catalog.resolve(key)
            except GoodPregraspMissError:
                missing_by_cell[binding.morphology_cell_ids[index]].append(index)
            else:
                MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)  # Recheck hits; invalid entries are errors, not misses.
                cached_indices.append(index)

        # Interleave cells so early shards cover distinct finger counts and thumb DOFs; never pad with repeats.
        ordered_missing = [
            index
            for row in zip_longest(*(missing_by_cell[cell] for cell in sorted(missing_by_cell)))
            for index in row
            if index is not None
        ]
        assert len(cached_indices) + len(ordered_missing) == binding.asset_count
        assert len(set(cached_indices + ordered_missing)) == binding.asset_count  # Preserve the complete parent denominator.
        output_root.mkdir(parents=True)
        inspection = {
            "parent_cohort_lock": str(parent_path),
            "parent_cohort_lock_sha256": binding.cohort_lock_sha256,
            "catalog_root": str(catalog.root),
            "expected_keys": [key.to_dict() for key in keys],
            "cached_parent_asset_indices": cached_indices,
            "ordered_missing_parent_asset_indices": ordered_missing,
        }  # Save the full query result before writing shards.
        (output_root / "inspection.json").write_text(
            json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shards = []
        for start in range(0, len(ordered_missing), args.shard_assets):
            indices = ordered_missing[start : start + args.shard_assets]  # Map each shard back to the parent axis.
            shard_id = f"{output_root.name}-shard-{len(shards):03d}"  # Keep each preparation pass uniquely named.
            source_path = output_root / f"{shard_id}.lock.yaml"
            canonical_path = output_root / f"{shard_id}.canonical.lock.yaml"
            write_hand_asset_cohort_subset(
                parent_cohort,
                source_path,
                canonical_path,
                cohort_id=shard_id,
                member_indices=indices,
                selection={
                    "algorithm": "missing-exact-pregrasp-cell-interleaved-v1",
                    "purpose": "pregrasp-generation-only",
                    "catalog_root": str(catalog.root),  # Keep the explicit catalog path in every shard lock.
                    **(
                        {"pregrasp_generation_identity": parent_cohort.selection["pregrasp_generation_identity"]}
                        if "pregrasp_generation_identity" in parent_cohort.selection
                        else {}
                    ),
                },
            )  # Keep full source/canonical validation for the later generator.
            shards.append(
                {
                    "cohort_id": shard_id,
                    "cohort_lock": str(canonical_path),
                    "asset_count": len(indices),
                    "parent_asset_indices": indices,
                }
            )

        # This is a preparation plan, not a learning result; train only after every miss is resolved.
        document = {
            "artifact_type": "anymani.good_pregrasp.cohort_preparation",
            "schema_version": "1.0.0",
            "parent_cohort_lock": str(parent_path),
            "parent_cohort_lock_sha256": binding.cohort_lock_sha256,
            "catalog_root": str(catalog.root),
            "asset_count": binding.asset_count,
            "cached_count": len(cached_indices),
            "cached_parent_asset_indices": cached_indices,
            "missing_count": len(ordered_missing),
            "shard_assets_max": args.shard_assets,
            "shards": shards,
        }
        plan_path = output_root / "preparation.json"
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(plan_path)
        print(
            json.dumps(
                {
                    "preparation": str(plan_path),
                    "assets": binding.asset_count,
                    "cached": len(cached_indices),
                    "missing": len(ordered_missing),
                    "shards": len(shards),
                }
            ),
            flush=True,
        )
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
