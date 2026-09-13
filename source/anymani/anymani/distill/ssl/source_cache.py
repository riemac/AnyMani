r"""Geometry Source Cache v2 的显式 status/audit/prune 维护入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from anymani.distill.representations.sources.artifacts import (
    audit_source_cache,
    prune_source_cache,
    source_cache_status,
)


def _build_parser() -> argparse.ArgumentParser:
    r"""构造不依赖 dataset/model 的 cache 维护命令。"""

    parser = argparse.ArgumentParser(description="Inspect or maintain AnyMani Geometry Source Cache v2.")
    parser.add_argument("command", choices=("status", "audit", "prune"))
    parser.add_argument("--root", default="logs/ssl/_cache/geometry_source/v2")
    parser.add_argument("--apply", action="store_true", help="apply prune deletions; default is report-only")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    r"""执行一次显式 cache status、full checksum audit 或受保护 prune。"""

    args = _build_parser().parse_args(argv)
    root = Path(args.root).expanduser()
    if args.command == "status":
        result = source_cache_status(root)
    elif args.command == "audit":
        result = audit_source_cache(root)
    else:
        result = prune_source_cache(root, apply=bool(args.apply))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return result


if __name__ == "__main__":
    main()


__all__ = ["main"]
