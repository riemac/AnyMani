"""Records generated hand bundles, provenance, and pipeline status.

Topology rendering is a lazy, presentation-only helper on the result object. It reuses the HandCfg source and does not participate in geometry identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..asset_base import HandCfg


@dataclass
class HandGenerationResult:
    "Result record with ordered geometry, provenance, and validation status."

    hand_cfg: HandCfg | None = None
    "Typed source hand configuration before export or local mutation."

    urdf_path: Path | None = None
    "Filesystem location resolved relative to the declared asset root when not absolute."

    sidecar_path: Path | None = None
    "Filesystem location resolved relative to the declared asset root when not absolute."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Stage results and source identities associated with this generated hand."

    tree_txt: str | None = None
    "Text representation of the ordered hand topology."

    def render_trees(self) -> "HandGenerationResult":
        "Adds a text topology rendering without changing geometry or source identity."

        if self.hand_cfg is not None:
            # Keep visualization imports outside the generation and validation dependency path.
            from .presentation.tree_render import render_hand_tree_txt

            self.tree_txt = render_hand_tree_txt(self.hand_cfg)
        return self


__all__ = [
    "HandGenerationResult",
]
