"Loads restricted YAML documents for asset schemas and sidecars."

from __future__ import annotations

from typing import Any

import yaml


def safe_load(data: bytes | str) -> Any:
    "Loads a restricted YAML document and rejects unsupported tags."

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(data, Loader=loader)


__all__ = ["safe_load"]
