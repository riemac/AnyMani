"Defines the asset-run strategy shared by the asset generation entry points."

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..asset_base import AssetCfgBase


@dataclass
class AssetRunStrategyCfg(AssetCfgBase):
    "Runner options for bounded topology enumeration and stage dispatch."

    topology_selection_mode: Literal["all", "random_subset", "random_subset_with_full_hand"] = "all"
    "Discrete topology selection policy; currently only complete enumeration is implemented."

    topology_selection_count: int | None = None
    "Count or linear dimension in the units declared by the associated schema."


__all__ = ["AssetRunStrategyCfg"]
