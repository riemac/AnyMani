"Represents one hand as a URDF, sidecar, and mesh dependency closure without copying or rewriting geometry."

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from ..asset_schema_geometry import HandGeometrySemanticsCfg
from .geometry_semantics import HandAssetSourceKind, resolve_hand_geometry_semantics
from .path_utils import resolve_container_entry_path
from .yaml_utils import safe_load

UrdfRgba = tuple[float, float, float, float]
"Four visual color channels in the normalized interval from zero to one."

@dataclass(frozen=True)
class UrdfMeshRef:
    "Original URDF URI, normalized virtual path, and resolved local mesh path."

    raw_uri: str
    "Exact mesh filename written in the source URDF."

    virtual_path: PurePosixPath
    "Normalized bundle-relative path used by the downstream hand container."

    real_path: Path
    "Resolved source file path for the mesh dependency."


@dataclass(frozen=True)
class PortableMeshBinding:
    "Verified bundle file, URI, and byte digest for one frozen geometry locator."

    uri: str
    sha256: str
    path: Path


@dataclass
class HandContainerCfg:
    "Explicit bundle path, sidecar contract, and source kind for one hand."

    path: str | Path
    "Filesystem location resolved relative to the declared asset root when not absolute."

    asset_id: str | None = None
    "Asset identifier from hand.yaml; compare content and geometry hashes for full identity."

    urdf_file: str = "hand.urdf"
    "URDF filename inside the selected bundle, normally hand.urdf."

    sidecar_file: str = "hand.yaml"
    "Stable semantic identifier preserved in the output metadata."

    source_kind: HandAssetSourceKind = "generated"
    "Selects the typed sidecar contract used to resolve this bundle."

    topology_key: str | None = None
    "Optional morphology identity used when resolving typed sidecar semantics."

    portable_mesh_bindings: Mapping[str, PortableMeshBinding] | None = None
    "Explicit legacy-locator bindings for a verified portable asset bundle."


HandContainerLike: TypeAlias = str | Path | HandContainerCfg
"Accepted explicit form for one hand bundle or its configuration."


def coerce_hand_container_cfg(value: HandContainerLike) -> HandContainerCfg:
    "Converts an explicit path or config into one typed hand-container entry."

    if isinstance(value, HandContainerCfg):
        return value
    if isinstance(value, str | Path):
        return HandContainerCfg(path=value)
    raise TypeError(f"Unsupported hand container entry: {value!r}")


@dataclass(frozen=True)
class HandContainer:
    "One hand asset presented as a virtual URDF, sidecar, and mesh bundle."

    asset_id: str
    "Asset identifier from hand.yaml; compare content and geometry hashes for full identity."

    virtual_to_real: dict[PurePosixPath, Path]
    "Mapping from normalized bundle paths to their source files, including shared meshes."

    real_to_virtual: dict[Path, PurePosixPath]
    "Reverse mapping from source files to normalized bundle-relative paths."

    sidecar: dict[str, Any] = field(default_factory=dict)
    "Stable semantic identifier preserved in the output metadata."

    source_kind: HandAssetSourceKind = "generated"
    "Selects the typed sidecar contract used to resolve this bundle."

    geometry_semantics: HandGeometrySemanticsCfg | None = None
    "Versioned static geometry and kinematic semantics emitted to the sidecar."

    mesh_refs: tuple[UrdfMeshRef, ...] = ()
    "All source URDF mesh references and their resolved files."

    visual_rgba_by_name: dict[str, UrdfRgba] = field(default_factory=dict)
    "Stable semantic identifier preserved in the output metadata."

    portable_mesh_bindings: dict[str, tuple[Path, str]] | None = None
    "Verified frozen-locator to bundle-file map; None keeps strict source-path behavior."

    @classmethod
    def from_cfg(
        cls,
        cfg: HandContainerCfg,
        *,
        source_root: Path | None = None,
        require_sidecar: bool = True,
        validate_mesh_relpaths: bool = True,
        parse_visual_rgba: bool = True,
        require_geometry_semantics: bool = False,
        allow_legacy_left_handedness: bool = False,
    ) -> HandContainer:
        "Resolves a hand URDF, sidecar, and every local mesh into a typed virtual bundle without rewriting source geometry."

        entry_path = resolve_container_entry_path(cfg.path, source_root=source_root)
        urdf_path, sidecar_path = _resolve_urdf_and_sidecar_paths(
            entry_path,
            urdf_file=cfg.urdf_file,
            sidecar_file=cfg.sidecar_file,
        )
        if not urdf_path.is_file():
            raise FileNotFoundError(f"hand URDF does not exist: {urdf_path}")
        sidecar = _load_sidecar(sidecar_path, require_sidecar=require_sidecar)
        _validate_generated_handedness_contract(
            sidecar,
            source_kind=cfg.source_kind,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )


        from .urdf_utils import parse_urdf_metadata

        mesh_refs, parsed_visual_rgba = parse_urdf_metadata(
            urdf_path,
            require_existing=validate_mesh_relpaths,
            parse_visual_rgba=parse_visual_rgba,
        )
        visual_rgba_by_name = parsed_visual_rgba if parse_visual_rgba else {}
        asset_id = str(cfg.asset_id or sidecar.get("id") or urdf_path.parent.name)
        geometry_semantics = (
            resolve_hand_geometry_semantics(
                sidecar,
                source_kind=cfg.source_kind,
                asset_id=asset_id,
                topology_key=cfg.topology_key,
            )
            if require_geometry_semantics
            else None
        )
        virtual_to_real, real_to_virtual = _build_virtual_path_bijection(
            urdf_path=urdf_path,
            sidecar_path=sidecar_path if sidecar_path.is_file() else None,
            mesh_refs=mesh_refs,
        )
        portable_mesh_bindings = _resolve_portable_mesh_bindings(
            sidecar,
            cfg.portable_mesh_bindings,
            mesh_refs=mesh_refs,
            virtual_to_real=virtual_to_real,
        )
        return cls(
            asset_id=asset_id,
            virtual_to_real=virtual_to_real,
            real_to_virtual=real_to_virtual,
            sidecar=sidecar,
            source_kind=cfg.source_kind,
            geometry_semantics=geometry_semantics,
            mesh_refs=mesh_refs,
            visual_rgba_by_name=visual_rgba_by_name,
            portable_mesh_bindings=portable_mesh_bindings,
        )


    @property
    def urdf_path(self) -> Path:
        "Returns the resolved hand.urdf file for this bundle."

        return self.real_path("hand.urdf")

    @property
    def sidecar_path(self) -> Path:
        "Returns the resolved hand.yaml file for this bundle."

        return self.real_path("hand.yaml")

    def real_path(self, virtual_path: str | PurePosixPath) -> Path:
        "Resolves one virtual bundle path to the source file used by the hand container."

        key = _normalize_virtual_path(virtual_path)
        try:
            return self.virtual_to_real[key]
        except KeyError as exc:
            raise KeyError(f"unknown virtual hand asset path {str(key)!r} for asset {self.asset_id!r}") from exc

    def virtual_path(self, real_path: str | Path) -> PurePosixPath:
        "Maps one source dependency to its normalized package-relative bundle path."

        key = Path(real_path).expanduser().resolve(strict=False)
        try:
            return self.real_to_virtual[key]
        except KeyError as exc:
            raise KeyError(f"real path {key} is not part of hand asset {self.asset_id!r}") from exc

    def resolve_mesh_locator(self, locator: str) -> Path:
        """Resolve one geometry locator without guessing across bundles."""

        if self.portable_mesh_bindings is not None:
            binding = self.portable_mesh_bindings.get(str(locator))
            if binding is None:
                raise FileNotFoundError(
                    f"portable bundle {self.asset_id!r} has no verified mesh binding for {locator!r}"
                )
            path, expected_sha256 = binding
            _verify_mesh_digest(path, expected_sha256, context=f"portable mesh locator {locator!r}")
            return path

        source_path = Path(locator).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"hand geometry mesh does not exist: {source_path}")
        return source_path.resolve(strict=True)


def _resolve_portable_mesh_bindings(
    sidecar: Mapping[str, Any],
    bindings: Mapping[str, PortableMeshBinding] | None,
    *,
    mesh_refs: tuple[UrdfMeshRef, ...],
    virtual_to_real: Mapping[PurePosixPath, Path],
) -> dict[str, tuple[Path, str]] | None:
    """Bind frozen absolute mesh locators only to this container's verified URDF closure."""

    if bindings is None:
        return None
    required = _absolute_mesh_locators(sidecar)
    supplied = {str(locator) for locator in bindings}
    if supplied != required:
        missing = sorted(required - supplied)
        extra = sorted(supplied - required)
        raise ValueError(
            f"portable mesh bindings must cover exactly this sidecar's absolute locators: missing={missing}, extra={extra}"
        )

    resolved: dict[str, tuple[Path, str]] = {}
    for locator, binding in bindings.items():
        if not isinstance(binding, PortableMeshBinding):
            raise TypeError(f"portable mesh binding for {locator!r} must be PortableMeshBinding")
        expected_sha256 = str(binding.sha256)
        if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise ValueError(f"portable mesh binding for {locator!r} requires a lowercase SHA-256")
        matches = {
            reference.virtual_path
            for reference in mesh_refs
            if reference.raw_uri == binding.uri or str(reference.virtual_path) == binding.uri
        }
        if len(matches) != 1:
            raise ValueError(
                f"portable mesh URI {binding.uri!r} for locator {locator!r} must resolve to one bundle path; "
                f"found={sorted(map(str, matches))}"
            )
        virtual_path = next(iter(matches))
        try:
            real_path = virtual_to_real[virtual_path].resolve(strict=True)
        except (KeyError, FileNotFoundError) as exc:
            raise FileNotFoundError(
                f"portable mesh URI {binding.uri!r} is absent from asset {sidecar.get('id', '<unknown>')!r}"
            ) from exc
        verified_path = Path(binding.path).expanduser().resolve(strict=True)
        if verified_path != real_path:
            raise ValueError(
                f"portable mesh URI {binding.uri!r} path disagrees with the member URDF closure: "
                f"manifest={verified_path}, urdf={real_path}"
            )
        _verify_mesh_digest(real_path, expected_sha256, context=f"portable mesh URI {binding.uri!r}")
        resolved_locator = str(locator)
        if resolved_locator in resolved and resolved[resolved_locator] != (real_path, expected_sha256):
            raise ValueError(f"portable mesh locator {resolved_locator!r} has conflicting bindings")
        resolved[resolved_locator] = (real_path, expected_sha256)
    return resolved


def _absolute_mesh_locators(sidecar: Mapping[str, Any]) -> set[str]:
    """Collect frozen absolute mesh paths from the typed geometry and builder snapshot."""

    locators: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "file_path" and isinstance(child, str) and Path(child).expanduser().is_absolute():
                    locators.add(child)
                visit(child)
        elif isinstance(value, list | tuple):
            for child in value:
                visit(child)

    visit(sidecar)
    return locators


def _verify_mesh_digest(path: Path, expected_sha256: str, *, context: str) -> None:
    """Verify one resolved mesh against its bundle-published byte identity."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{context} SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}")


def _validate_generated_handedness_contract(
    sidecar: dict[str, Any],
    *,
    source_kind: HandAssetSourceKind,
    allow_legacy_left_handedness: bool,
) -> None:

    if source_kind != "generated":
        return

    from ..handedness import validate_generated_handedness_contract

    validate_generated_handedness_contract(
        sidecar,
        allow_legacy_left_handedness=allow_legacy_left_handedness,
    )


def _resolve_urdf_and_sidecar_paths(entry_path: Path, *, urdf_file: str, sidecar_file: str) -> tuple[Path, Path]:

    if entry_path.suffix == ".urdf":
        urdf_path = entry_path.resolve(strict=False)
        bundle_dir = urdf_path.parent
    else:
        bundle_dir = entry_path.resolve(strict=False)
        urdf_path = (bundle_dir / urdf_file).resolve(strict=False)
    return urdf_path, (bundle_dir / sidecar_file).resolve(strict=False)


def _load_sidecar(sidecar_path: Path, *, require_sidecar: bool) -> dict[str, Any]:

    if not sidecar_path.is_file():
        if require_sidecar:
            raise FileNotFoundError(f"hand sidecar does not exist: {sidecar_path}")
        return {}
    data = safe_load(sidecar_path.read_bytes())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"hand sidecar must be a YAML mapping: {sidecar_path}")
    return data


def _build_virtual_path_bijection(
    *,
    urdf_path: Path,
    sidecar_path: Path | None,
    mesh_refs: tuple[UrdfMeshRef, ...],
) -> tuple[dict[PurePosixPath, Path], dict[Path, PurePosixPath]]:

    virtual_to_real: dict[PurePosixPath, Path] = {}
    real_to_virtual: dict[Path, PurePosixPath] = {}
    _add_virtual_mapping(virtual_to_real, real_to_virtual, PurePosixPath("hand.urdf"), urdf_path)
    if sidecar_path is not None:
        _add_virtual_mapping(virtual_to_real, real_to_virtual, PurePosixPath("hand.yaml"), sidecar_path)
    for mesh_ref in mesh_refs:
        _add_virtual_mapping(virtual_to_real, real_to_virtual, mesh_ref.virtual_path, mesh_ref.real_path)
    return virtual_to_real, real_to_virtual


def _add_virtual_mapping(
    virtual_to_real: dict[PurePosixPath, Path],
    real_to_virtual: dict[Path, PurePosixPath],
    virtual_path: PurePosixPath,
    real_path: Path,
) -> None:

    normalized_virtual = _normalize_virtual_path(virtual_path)
    normalized_real = Path(real_path).expanduser().resolve(strict=False)
    existing_real = virtual_to_real.get(normalized_virtual)
    if existing_real is not None and existing_real != normalized_real:
        raise ValueError(
            f"virtual hand asset path {normalized_virtual} maps to both {existing_real} and {normalized_real}"
        )
    existing_virtual = real_to_virtual.get(normalized_real)
    if existing_virtual is not None and existing_virtual != normalized_virtual:
        raise ValueError(
            f"real hand asset path {normalized_real} maps to both {existing_virtual} and {normalized_virtual}"
        )
    virtual_to_real[normalized_virtual] = normalized_real
    real_to_virtual[normalized_real] = normalized_virtual


def _normalize_virtual_path(path: str | PurePosixPath) -> PurePosixPath:

    virtual_path = PurePosixPath(path)
    if virtual_path.is_absolute() or ".." in virtual_path.parts:
        raise ValueError(f"virtual hand asset path must stay inside the virtual bundle: {path!r}")
    return virtual_path


__all__ = [
    "HandContainerLike",
    "HandContainer",
    "HandContainerCfg",
    "PortableMeshBinding",
    "UrdfMeshRef",
    "UrdfRgba",
    "coerce_hand_container_cfg",
]
