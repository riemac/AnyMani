from __future__ import annotations

import sys
from pathlib import Path


ASSETS_PARENT = Path(__file__).resolve().parents[2]
if str(ASSETS_PARENT) not in sys.path:
    sys.path.insert(0, str(ASSETS_PARENT))
