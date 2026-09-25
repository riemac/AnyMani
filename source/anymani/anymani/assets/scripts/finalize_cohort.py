"""Finalize a source cohort lock with runtime-equivalent canonical identities.

The tool builds the same generated-hand binding used at runtime, then writes a separate canonical-final lock. It does not create a scene or step physics; the source lock remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parse the source lock and optional canonical output path before importing Isaac Lab."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True, help="Validated source cohort lock.")
    parser.add_argument("--output", type=Path, default=None, help="Canonical-final lock output path.")
    return parser.parse_args()


ARGS = _parse_args()
SOURCE_LOCK = ARGS.source_lock.expanduser().resolve(strict=True)
OUTPUT_LOCK = (
    ARGS.output.expanduser().resolve()
    if ARGS.output is not None
    else SOURCE_LOCK.with_name(f"{SOURCE_LOCK.name.removesuffix('.lock.yaml')}.canonical.lock.yaml")
)
os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(SOURCE_LOCK)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> None:
    """Compute canonical member identities and write the finalized lock."""

    from anymani.assets.bank.cohort import finalize_hand_asset_cohort_lock
    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

    binding = build_generated_asset_binding()
    artifacts = binding.canonical_artifacts
    schema_versions = {artifact.schema_version for artifact in artifacts}
    if len(schema_versions) != 1:
        raise RuntimeError(f"cohort canonical artifacts disagree on schema version: {sorted(schema_versions)}")
    output = finalize_hand_asset_cohort_lock(
        SOURCE_LOCK,
        OUTPUT_LOCK,
        canonical_identities=tuple(
            (artifact.source_content_hash, artifact.physical_geometry_hash, artifact.schema_digest)
            for artifact in artifacts
        ),
        canonical_schema_version=next(iter(schema_versions)),
    )
    print(
        json.dumps(
            {
                "source_lock": str(SOURCE_LOCK),
                "source_lock_sha256": binding.cohort_lock_sha256,
                "canonical_lock": str(output),
                "asset_count": binding.asset_count,
                "physical_geometry_hashes_unique": len({artifact.physical_geometry_hash for artifact in artifacts}),
                "canonical_schema_version": next(iter(schema_versions)),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
