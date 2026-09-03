r"""由预排序pair manifest与strict-v5 Top-8 catalog发布最终80-row selection。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.representative_selection import finalize_representative_selection
from anymani.pregrasp.good_catalog import GoodPregraspEntry
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
from anymani.tasks.hetero.config.generated.strict_good_pregrasp_identity import (
    STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
    STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
)

DEFAULT_CANDIDATES = Path(
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80_candidates.yaml"
)
DEFAULT_OUTPUT = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml")
DEFAULT_CATALOG = Path("outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5")
DEFAULT_SUMMARY = Path("outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/summary.json")


def _parse_args() -> argparse.Namespace:
    r"""解析候选manifest、一个或多个生成summary与最终输出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--summary", type=Path, action="append", default=None)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    r"""逐项重验640个strict members后原子发布最终MVP80 YAML。"""

    args = _parse_args()
    root = resolve_anymani_root()
    candidate_path = args.candidates if args.candidates.is_absolute() else root / args.candidates
    output_path = args.output if args.output.is_absolute() else root / args.output
    catalog_path = args.catalog if args.catalog.is_absolute() else root / args.catalog
    summary_arguments = args.summary or [DEFAULT_SUMMARY]
    summary_paths = tuple(path if path.is_absolute() else root / path for path in summary_arguments)
    candidate_document = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    summaries = tuple(json.loads(path.read_text(encoding="utf-8")) for path in summary_paths)
    if len(summaries) != 1 or summaries[0].get("formal_all_80_top8_passed") is not True:
        raise RuntimeError("final MVP80 manifest requires one complete strict-v5 80×Top-8 summary")
    if summaries[0].get("strict_gate_digest") != MVP80_STRICT_GOOD_PREGRASP_GATE.digest:
        raise RuntimeError("strict pregrasp summary gate digest disagrees with current protocol")
    if summaries[0].get("generation_identity_digest") != STRICT_GOOD_PREGRASP_GENERATION_DIGEST:
        raise RuntimeError("strict pregrasp summary generation identity disagrees with current protocol")
    if summaries[0].get("physics_identity_digest") != STRICT_GOOD_PREGRASP_PHYSICS_DIGEST:
        raise RuntimeError("strict pregrasp summary physics identity disagrees with current protocol")

    # Summary count不是充分证据；直接读取catalog index/payload并对640个metrics重放同一strict predicate。
    index_document = json.loads((catalog_path / "index.json").read_text(encoding="utf-8"))
    index_entries = index_document.get("entries", ())
    if len(index_entries) != 80:
        raise RuntimeError(f"strict catalog must contain exactly 80 entries, got {len(index_entries)}")
    strict_rows: list[int] = []
    asset_to_row = {str(item["asset_id"]): int(item["dataset_row"]) for item in summaries[0]["published"]}
    for index_entry in index_entries:
        payload_path = catalog_path / str(index_entry["payload_relpath"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != index_entry["entry_digest"]:
            raise RuntimeError(f"strict catalog payload digest mismatch: {payload_path}")
        entry = GoodPregraspEntry.from_dict(payload)
        if entry.key.digest != index_entry["key_digest"]:
            raise RuntimeError(f"strict catalog key digest mismatch: {payload_path}")
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
        try:
            strict_rows.append(asset_to_row[entry.key.asset_id])
        except KeyError as error:
            raise RuntimeError(f"catalog asset {entry.key.asset_id} is absent from strict summary") from error
    if len(strict_rows) != 80 or len(set(strict_rows)) != 80:
        raise RuntimeError("strict catalog does not map one-to-one onto 80 unique dataset rows")
    passed_rows = tuple(
        int(item["dataset_row"])
        for summary in summaries
        for item in summary["published"]
    )
    if set(passed_rows) != set(strict_rows):
        raise RuntimeError("strict catalog rows disagree with generation summary rows")
    document = finalize_representative_selection(
        candidate_document,
        passed_rows=passed_rows,
        pregrasp_catalog_root=str(catalog_path.relative_to(root)),
        pregrasp_summary_paths=[str(path.relative_to(root)) for path in summary_paths],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(output_path)
    print(
        {
            "output": str(output_path),
            "selected_asset_count": document["selected_asset_count"],
            "rejected_pair_count": len(document["rejected_pairs"]),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
