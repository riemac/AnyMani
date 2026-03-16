from __future__ import annotations

import sys
from pathlib import Path


def _ensure_path():
    repo_root = Path(__file__).resolve().parents[4]
    source_anymani = repo_root / "source" / "anymani"
    if str(source_anymani) not in sys.path:
        sys.path.insert(0, str(source_anymani))


_ensure_path()
