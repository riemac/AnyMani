"""Copy certified Top-8 entries into a separate, content-addressed catalog.

Source entries are checked against the shared strict gate before one atomic publish. Copying preserves physics and generation identity; it does not search for new initial states.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from anymani.pregrasp.good_catalog import GoodPregraspCatalog, GoodPregraspIndexEntry, GoodPregraspKey
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE


def copy_strict_catalog(
    source: GoodPregraspCatalog,
    target: GoodPregraspCatalog,
    *,
    keys: Sequence[GoodPregraspKey] | None = None,
) -> tuple[GoodPregraspIndexEntry, ...]:
    """Copy complete entries that pass the shared strict gate into a distinct catalog.

    Args:
        source: Read-only source catalog.
        target: Separate target; identical keys are idempotent and conflicting bytes are rejected.
        keys: Optional exact-key subset. A requested missing key stops the copy.

    Returns:
        Published index entries in request order, with their original Top-8 identity.
    """

    if source.root.resolve() == target.root.resolve():
        raise ValueError("source and target catalogs must be distinct")
    if not source.index_path.is_file():
        raise FileNotFoundError(source.index_path)  # Distinguish a missing source from an empty catalog.

    # Validate every entry before making any target update visible.
    entries = source.read_entries() if keys is None else source.resolve_many(tuple(keys))
    for entry in entries:
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)  # Apply the same cold-reset gate to all eight members.
    return target.publish_many(entries)  # Copy complete entries; never reconstruct a partial Top-8.


def main() -> None:
    """Parse source, target, and optional exact keys, then copy validated entries."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Root of an existing certified catalog.")
    parser.add_argument("--target", type=Path, required=True, help="Root of the new preparation or role catalog.")
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="Exact-key list or inspection JSON with expected_keys.",
    )
    args = parser.parse_args()

    # Keys come from binding or preparation output and use the shared identity schema.
    keys = None
    if args.keys_file is not None:
        document = json.loads(args.keys_file.read_text(encoding="utf-8"))
        records = document.get("expected_keys") if isinstance(document, Mapping) else document
        if not isinstance(records, list):
            raise ValueError("keys file must contain an exact-key list or expected_keys")
        keys = tuple(GoodPregraspKey.from_dict(record) for record in records)
    source, target = GoodPregraspCatalog(args.source), GoodPregraspCatalog(args.target)
    copied = copy_strict_catalog(source, target, keys=keys)

    # Report the publisher's content identities without rereading every file.
    print(
        json.dumps(
            {
                "source_catalog": str(source.root.resolve()),
                "target_catalog": str(target.root.resolve()),
                "copied_count": len(copied),
                "entry_digests": [entry.entry_digest for entry in copied],
                "strict_top8_validated": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
