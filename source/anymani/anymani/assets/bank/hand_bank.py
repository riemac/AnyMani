"Resolves explicit hand bundles into ordered selections and validates sidecars and URDF mesh closure."

# Keep the public contract notes current when the implementation changes.

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..asset_bank import AssetBank, AssetBankCfg
from .hand_container import HandContainer, HandContainerCfg, HandContainerLike, coerce_hand_container_cfg
from .path_utils import resolve_post_mutate_root

HandSourceMode = Literal["post_mutate", "pre_made", "mixed"]
"Post-mutate, pre-made, or explicit mixed bundle source mode."

HandSelectionMode = Literal["explicit", "sample", "all"]
"Explicit, deterministic sample, or all-candidate selection mode."

@dataclass(frozen=True)
class HandSelection:
    "Ordered selection result with source mode, selection mode, and optional sampling seed."

    assets: tuple[HandContainer, ...]
    "Ordered validated hand containers returned by this selection."

    source_mode: HandSourceMode
    "Source organization used to discover or resolve hand bundles."

    selection_mode: HandSelectionMode
    "Explicit list, deterministic sample, or complete source enumeration."

    sample_seed: int | None = None
    "Root seed for deterministic sampling."

    source_root: Path | None = None
    "Root used to resolve a relative asset path."


@dataclass
class HandBankCfg(AssetBankCfg):
    "Declares explicit bundle paths and validation behavior for an ordered hand selection."

    class_type: type[HandBank] | None = None
    "Associated runtime implementation for this configuration class."

    source_mode: HandSourceMode = "post_mutate"
    "Source organization used to discover or resolve hand bundles."

    selection_mode: HandSelectionMode = "explicit"
    "Explicit list, deterministic sample, or complete source enumeration."

    pre_made_path: str | Path | None = None
    "Filesystem location resolved relative to the declared asset root when not absolute."

    post_mutate_path: str | Path | None = None
    "Filesystem location resolved relative to the declared asset root when not absolute."

    post_mutate_name: str | None = None
    "Stable semantic identifier preserved in the output metadata."

    include_source_topology: bool = True
    "Whether a pre-made mother is included with post-mutate variants."

    containers: tuple[HandContainerLike, ...] = ()
    "Ordered explicit bundle entries; each entry resolves its own local mesh closure."

    sample_count: int | None = None
    "Count or linear dimension in the units declared by the associated schema."

    sample_seed: int = 0
    "Root seed for deterministic sampling."

    require_sidecar: bool = True
    "Whether hand.yaml must accompany the URDF."

    validate_mesh_relpaths: bool = True
    "Whether every relative URDF mesh reference must resolve to a local file."

    parse_visual_rgba: bool = True
    "Whether named visual RGBA values are read from the URDF."

    require_geometry_semantics: bool = False
    "Whether a bundle must expose validated typed static geometry semantics."

    allow_legacy_left_handedness: bool = False
    "Explicitly permits a legacy left bundle without the strict mirror certificate."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.class_type is None:
            self.class_type = HandBank
        self.containers = tuple(coerce_hand_container_cfg(container) for container in self.containers)


class HandBank(AssetBank):
    "Resolves an ordered set of explicit hand bundles and validates each URDF, sidecar, and mesh reference."

    cfg: HandBankCfg

    def __init__(self, cfg: HandBankCfg):
        "Stores the typed selection config; path IO and bundle validation occur only in resolve()."

        super().__init__(cfg)

    def resolve(self) -> HandSelection:
        "Resolves declared source paths, typed sidecars, and the required mesh closure."

        candidates = self.discover()
        return self.select(candidates)

    def discover(self) -> tuple[HandContainer, ...]:
        "Finds only the bundles permitted by the declared source mode."

        source_root = self._resolve_optional_source_root()
        if self.cfg.selection_mode == "explicit":
            if not self.cfg.containers:
                raise ValueError("selection_mode='explicit' requires at least one HandContainerCfg")  # fail-fast
            return tuple(
                self._container_from_cfg(container_cfg, source_root=source_root)  # bundle -> typed container
                for container_cfg in self.cfg.containers
            )

        if self.cfg.source_mode == "pre_made":
            return self._discover_pre_made(source_root)
        if self.cfg.source_mode == "mixed":
            if not self.cfg.containers:
                raise ValueError("source_mode='mixed' requires an explicit cross-family containers manifest")
            return tuple(
                self._container_from_cfg(container_cfg, source_root=None)
                for container_cfg in self.cfg.containers
            )
        if source_root is None:
            raise ValueError(f"selection_mode={self.cfg.selection_mode!r} requires a post_mutate source root")
        if not source_root.is_dir():
            raise FileNotFoundError(f"post-mutate source root does not exist: {source_root}")
        candidates: list[HandContainer] = []
        if self.cfg.include_source_topology:

            source_topology_root = source_root.parent
            if not self._has_hand_bundle_contract(source_topology_root):
                raise FileNotFoundError(
                    "include_source_topology=True requires a source topology bundle at "
                    f"post_mutate_path.parent: {source_topology_root / 'hand.urdf'}"
                )
            candidates.append(
                self._container_from_cfg(
                    HandContainerCfg(path=source_topology_root), source_root=None
                )
            )

        candidates.extend(
            self._container_from_cfg(HandContainerCfg(path=child), source_root=None)
            for child in source_root.iterdir()
            if child.is_dir() and (child / "hand.urdf").is_file()
        )
        return tuple(sorted(candidates, key=lambda container: container.asset_id))

    def _discover_pre_made(self, source_root: Path | None) -> tuple[HandContainer, ...]:

        if source_root is None or not source_root.is_dir():
            raise FileNotFoundError(f"pre-made source root does not exist: {source_root}")
        bundle_roots = (
            (source_root,)
            if self._has_hand_bundle_contract(source_root)
            else tuple(sorted({path.parent for path in source_root.rglob("hand.urdf")}))
        )
        candidates: list[HandContainer] = []
        for bundle_root in bundle_roots:
            container = self._container_from_cfg(
                HandContainerCfg(path=bundle_root), source_root=None
            )
            resolved_bundle_root = bundle_root.resolve(strict=False)
            if all(
                real_path.is_relative_to(resolved_bundle_root) for real_path in container.virtual_to_real.values()
            ):
                candidates.append(container)
        if not candidates:
            raise FileNotFoundError(f"no self-contained pre-made hand bundles found under: {source_root}")
        return tuple(sorted(candidates, key=lambda container: container.asset_id))

    def select(self, candidates: tuple[HandContainer, ...]) -> HandSelection:
        "Applies the declared explicit, deterministic-sample, or full-selection policy."

        source_root = self._resolve_optional_source_root()
        if self.cfg.selection_mode == "explicit":
            return HandSelection(
                assets=candidates,  # resolved containers
                source_mode=self.cfg.source_mode,  # provenance
                selection_mode=self.cfg.selection_mode,  # explicit
                sample_seed=None,
                source_root=source_root,
            )
        if self.cfg.selection_mode == "all":
            return HandSelection(
                assets=tuple(sorted(candidates, key=lambda container: container.asset_id)),
                source_mode=self.cfg.source_mode,  # provenance
                selection_mode=self.cfg.selection_mode,  # all
                sample_seed=None,
                source_root=source_root,  # collection root
            )
        if self.cfg.selection_mode == "sample":
            if self.cfg.sample_count is None:
                raise ValueError("selection_mode='sample' requires sample_count")
            if self.cfg.sample_count < 0:
                raise ValueError("sample_count must be non-negative")
            sorted_candidates = tuple(
                sorted(candidates, key=lambda container: container.asset_id)
            )
            if self.cfg.sample_count > len(sorted_candidates):
                raise ValueError(
                    f"sample_count={self.cfg.sample_count} exceeds available hand assets={len(sorted_candidates)}"
                )
            selected = tuple(
                random.Random(self.cfg.sample_seed).sample(sorted_candidates, self.cfg.sample_count)
            )
            return HandSelection(
                assets=selected,
                source_mode=self.cfg.source_mode,  # provenance
                selection_mode=self.cfg.selection_mode,  # sample
                sample_seed=self.cfg.sample_seed,
                source_root=source_root,  # collection root
            )
        raise ValueError(f"unknown HandBank selection_mode: {self.cfg.selection_mode!r}")

    def _resolve_optional_source_root(self) -> Path | None:

        if self.cfg.source_mode == "mixed":
            return None
        if self.cfg.source_mode == "pre_made":
            if self.cfg.pre_made_path is None:
                if self.cfg.selection_mode == "explicit" and all(
                    Path(cfg.path).expanduser().is_absolute() for cfg in self.cfg.containers  # absolute
                ):
                    return None
                raise ValueError("source_mode='pre_made' requires pre_made_path or absolute explicit containers")
            return Path(self.cfg.pre_made_path).expanduser().resolve(strict=False)

        try:
            return resolve_post_mutate_root(self.cfg)
        except ValueError:
            if self.cfg.selection_mode == "explicit" and all(
                Path(cfg.path).expanduser().is_absolute() for cfg in self.cfg.containers  # absolute manifest
            ):
                return None
            raise

    def _container_from_cfg(self, cfg: HandContainerCfg, *, source_root: Path | None) -> HandContainer:

        return HandContainer.from_cfg(
            cfg,  # path/ID/source kind/topology hint
            source_root=source_root,
            require_sidecar=self.cfg.require_sidecar,  # hand.yaml contract
            validate_mesh_relpaths=self.cfg.validate_mesh_relpaths,  # URDF mesh closure
            parse_visual_rgba=self.cfg.parse_visual_rgba,
            require_geometry_semantics=self.cfg.require_geometry_semantics,
            allow_legacy_left_handedness=self.cfg.allow_legacy_left_handedness,
        )

    @staticmethod
    def _has_hand_bundle_contract(candidate_root: Path) -> bool:

        return (candidate_root / "hand.urdf").is_file()


__all__ = [
    "HandBank",  # runtime discovery/selection
    "HandBankCfg",  # declarative config
    "HandSelection",  # resolved result/provenance
    "HandSelectionMode",  # explicit/sample/all
    "HandSourceMode",  # post_mutate/pre_made/mixed
]
