"""Compatibility facade for asset schema modules.

This module preserves the historical import path `anymani.assets.asset_base`
while delegating the actual schema definitions to:

- `asset_schema_core.py`
- `asset_schema_embodiment.py`
"""

from .asset_schema_core import *  # noqa: F401,F403
from .asset_schema_embodiment import *  # noqa: F401,F403
