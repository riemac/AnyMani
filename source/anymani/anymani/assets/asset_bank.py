"TODO: Expand the generic query interface only when multiple bank types share a stable contract. This module inventories bundles and returns downstream-neutral references; task spawning and training stay outside the asset layer."

# Keep the public contract notes current when the implementation changes.

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any


@dataclass
class AssetBankCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."


class AssetBank:
    "Base interface for asset collection discovery and selection; concrete banks own their schemas and paths."

    def __init__(self, cfg: AssetBankCfg):
        self.cfg = cfg


if TYPE_CHECKING:
    from .bank.dataset import HandAssetDataset, HandAssetDatasetCfg, ResolvedHandAssetDataset
    from .bank.hand_bank import HandBank, HandBankCfg, HandSelection, HandSelectionMode, HandSourceMode
    from .bank.hand_container import HandContainer, HandContainerCfg, HandContainerLike, UrdfMeshRef, UrdfRgba


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HandAssetDataset": ("anymani.assets.bank.dataset", "HandAssetDataset"),
    "HandAssetDatasetCfg": ("anymani.assets.bank.dataset", "HandAssetDatasetCfg"),
    "HandBank": ("anymani.assets.bank.hand_bank", "HandBank"),
    "HandBankCfg": ("anymani.assets.bank.hand_bank", "HandBankCfg"),
    "HandSelection": ("anymani.assets.bank.hand_bank", "HandSelection"),
    "HandSelectionMode": ("anymani.assets.bank.hand_bank", "HandSelectionMode"),
    "HandSourceMode": ("anymani.assets.bank.hand_bank", "HandSourceMode"),
    "ResolvedHandAssetDataset": ("anymani.assets.bank.dataset", "ResolvedHandAssetDataset"),
    "HandContainer": ("anymani.assets.bank.hand_container", "HandContainer"),
    "HandContainerCfg": ("anymani.assets.bank.hand_container", "HandContainerCfg"),
    "HandContainerLike": ("anymani.assets.bank.hand_container", "HandContainerLike"),
    "UrdfMeshRef": ("anymani.assets.bank.hand_container", "UrdfMeshRef"),
    "UrdfRgba": ("anymani.assets.bank.hand_container", "UrdfRgba"),
}
"Mapping from public asset-bank names to their lazy import modules."


def __getattr__(name: str) -> Any:

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)


__all__ = [
    "AssetBankCfg",
    "AssetBank",
    "HandAssetDataset",
    "HandAssetDatasetCfg",
    "HandBank",
    "HandBankCfg",
    "HandSelection",
    "HandSelectionMode",
    "HandSourceMode",
    "ResolvedHandAssetDataset",
    "HandContainer",
    "HandContainerCfg",
    "HandContainerLike",
    "UrdfMeshRef",
    "UrdfRgba",
]
