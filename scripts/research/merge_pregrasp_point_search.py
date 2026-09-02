r"""按formal dataset row合并已验证point-search artifacts，保留完整content lineage。

该工具只替换整条per-asset record/point证据，不平均q、pose或metrics。所有输入必须共享scale、gate、cube和physics
identity；nested :class:`PregraspRecord`逐条重验digest，合并后的每个selected row必须达到support或更高tier。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from anymani.pregrasp import PregraspRecord, PregraspTier, tier_satisfies


def _parse_args() -> argparse.Namespace:
    r"""解析base、重复``ROW=PATH`` replacements与输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replace", action="append", default=[], metavar="ROW=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    r"""读取point-search JSON object。"""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("artifact_type") != "anymani.pregrasp.point_search":
        raise ValueError(f"{path} is not a pregrasp point-search artifact")
    return document


def _sha256(path: Path) -> str:
    r"""返回artifact bytes SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_replacement(specification: str) -> tuple[int, Path]:
    r"""解析``ROW=PATH``且拒绝空字段。"""

    row_text, separator, path_text = specification.partition("=")
    if not separator or not row_text or not path_text:
        raise ValueError(f"invalid replacement specification: {specification!r}")
    return int(row_text), Path(path_text).expanduser().resolve()


def _unique_row_item(items: list[dict[str, Any]], row: int, label: str) -> dict[str, Any]:
    r"""取得唯一per-row item。"""

    matches = [item for item in items if int(item["dataset_row"]) == row]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one row={row} item")
    return deepcopy(matches[0])


def main() -> int:
    r"""合并replacements、重验records并写durable artifact。"""

    args = _parse_args()
    base_path = args.base.resolve()
    base = _load(base_path)
    output = deepcopy(base)
    replacement_specs = [_parse_replacement(specification) for specification in args.replace]
    if len({row for row, _ in replacement_specs}) != len(replacement_specs):
        raise ValueError("replacement rows must be unique")
    per_asset_lists = ("selected", "contact_frontier", "gate_frontier", "support_frontier")
    lineage: list[dict[str, Any]] = [
        {"kind": "base", "path": str(base_path), "sha256": _sha256(base_path)}
    ]
    for row, replacement_path in replacement_specs:
        replacement = _load(replacement_path)
        for field in ("scale", "gate_digest", "cube_sha256", "physics_identity"):
            if replacement.get(field) != base.get(field):
                raise ValueError(f"replacement row={row} disagrees with base field {field}")
        if row not in tuple(int(value) for value in base["dataset_rows"]):
            raise ValueError(f"replacement row={row} is absent from base dataset axis")
        for list_name in per_asset_lists:
            replacement_item = _unique_row_item(replacement[list_name], row, list_name)
            output[list_name] = [
                replacement_item if int(item["dataset_row"]) == row else item for item in output[list_name]
            ]
        replacement_points = [point for point in replacement["points"] if int(point["dataset_row"]) == row]
        if len(replacement_points) != 1:
            raise ValueError("replacement verify artifact must contain exactly one point for its row")
        output["points"] = [
            deepcopy(replacement_points[0]) if int(point["dataset_row"]) == row else point
            for point in output["points"]
        ]
        lineage.append(
            {"kind": "replacement", "dataset_row": row, "path": str(replacement_path), "sha256": _sha256(replacement_path)}
        )

    # 每个base row必须恰好保留一个strict selected record，并至少达到support point tier。
    for row in tuple(int(value) for value in output["dataset_rows"]):
        selected = _unique_row_item(output["selected"], row, "merged selected")
        record = PregraspRecord.from_dict(selected["record"])
        if not tier_satisfies(record.tier, PregraspTier.SUPPORT_BASIN):
            raise ValueError(f"merged selected row={row} does not reproduce support tier")
        for list_name in per_asset_lists[1:]:
            PregraspRecord.from_dict(_unique_row_item(output[list_name], row, list_name)["record"])
    output["portfolio"] = "merged_verified_support_centers"
    output["candidate_count_per_asset"] = 1
    output["merge_lineage"] = lineage
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset_rows": output["dataset_rows"],
                "replacement_rows": [row for row, _ in replacement_specs],
                "selected_tiers": [item["record"]["tier"] for item in output["selected"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
