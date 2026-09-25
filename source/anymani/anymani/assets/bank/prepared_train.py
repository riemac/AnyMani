"Caches resolved train rows only while the manifest, resolver, URDFs, sidecars, and mesh hashes remain valid."

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, cast

from anymani.assets.asset_schema_geometry import geometry_semantics_from_dict, geometry_semantics_to_dict

from .dataset import (
    HandAssetDataset,
    HandAssetProvenance,
    ResolvedHandAssetPartition,
    ResolvedHandAssetRecord,
)
from .geometry_semantics import HandAssetSourceKind
from .hand_container import HandContainer, UrdfMeshRef

PREPARED_TRAIN_CACHE_SCHEMA_VERSION = "1.0.0"
"Version of the removable validated-train cache."


def _resolver_implementation_sha256() -> str:

    bank_root = Path(__file__).resolve().parent
    sources = (
        bank_root / "dataset.py",
        bank_root / "geometry_semantics.py",
        bank_root / "hand_container.py",
        bank_root / "urdf_utils.py",
        bank_root.parent / "asset_schema_geometry.py",
        Path(__file__).resolve(),
    )
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_paths(dataset: HandAssetDataset, cache_root: Path | None) -> tuple[Path, Path]:

    root = cache_root or Path(os.environ.get("ANYMANI_CACHE_DIR", "~/.cache/anymani"))
    cache_dir = root.expanduser() / "assets" / "datasets" / "prepared" / "v1" / dataset.source_sha256
    return cache_dir / "train-index.json", cache_dir / "train-payload.json"


def _dependency_index(partition: ResolvedHandAssetPartition) -> list[dict[str, str]]:

    paths = {
        real_path.expanduser().resolve(strict=True)
        for record in partition.records
        for real_path in record.container.virtual_to_real.values()
    }
    return [{"path": str(path), "sha256": _sha256_file(path)} for path in sorted(paths)]


def _serialize_partition(partition: ResolvedHandAssetPartition) -> dict[str, Any]:

    records: list[dict[str, Any]] = []
    for record in partition.records:
        container = record.container
        records.append(
            {
                "asset_id": container.asset_id,
                "source_kind": container.source_kind,
                "virtual_to_real": {
                    str(virtual): str(real.expanduser().resolve(strict=True))
                    for virtual, real in sorted(container.virtual_to_real.items(), key=lambda item: str(item[0]))
                },
                "sidecar": container.sidecar,
                "geometry_semantics": (
                    geometry_semantics_to_dict(container.geometry_semantics)
                    if container.geometry_semantics is not None
                    else None
                ),
                "mesh_refs": [
                    {
                        "raw_uri": mesh.raw_uri,
                        "virtual_path": str(mesh.virtual_path),
                        "real_path": str(mesh.real_path.expanduser().resolve(strict=True)),
                    }
                    for mesh in container.mesh_refs
                ],
                "visual_rgba_by_name": {
                    name: list(rgba) for name, rgba in sorted(container.visual_rgba_by_name.items())
                },
                "provenance": asdict(record.provenance),
                "content_hash": record.content_hash,
            }
        )
    return {
        "schema_version": PREPARED_TRAIN_CACHE_SCHEMA_VERSION,
        "partition_name": partition.name,
        "records": records,
    }


def _deserialize_partition(document: dict[str, Any]) -> ResolvedHandAssetPartition:

    if document.get("schema_version") != PREPARED_TRAIN_CACHE_SCHEMA_VERSION:
        raise ValueError("prepared train payload schema mismatch")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("prepared train payload must contain non-empty records")
    records: list[ResolvedHandAssetRecord] = []
    for row, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"prepared train record {row} must be a mapping")
        virtual_to_real = {
            PurePosixPath(virtual): Path(real).expanduser().resolve(strict=True)
            for virtual, real in cast(dict[str, str], raw["virtual_to_real"]).items()
        }
        real_to_virtual = {real: virtual for virtual, real in virtual_to_real.items()}
        if len(real_to_virtual) != len(virtual_to_real):
            raise ValueError(f"prepared train record {row} virtual path mapping is not bijective")
        raw_geometry = raw.get("geometry_semantics")
        geometry = geometry_semantics_from_dict(raw_geometry) if isinstance(raw_geometry, dict) else None
        mesh_refs = tuple(
            UrdfMeshRef(
                raw_uri=str(mesh["raw_uri"]),
                virtual_path=PurePosixPath(str(mesh["virtual_path"])),
                real_path=Path(mesh["real_path"]).expanduser().resolve(strict=True),
            )
            for mesh in cast(list[dict[str, Any]], raw.get("mesh_refs", []))
        )
        rgba = {
            str(name): tuple(float(value) for value in values)
            for name, values in cast(dict[str, list[float]], raw.get("visual_rgba_by_name", {})).items()
        }
        container = HandContainer(
            asset_id=str(raw["asset_id"]),
            virtual_to_real=virtual_to_real,
            real_to_virtual=real_to_virtual,
            sidecar=cast(dict[str, Any], raw["sidecar"]),
            source_kind=cast(HandAssetSourceKind, raw["source_kind"]),
            geometry_semantics=geometry,
            mesh_refs=mesh_refs,
            visual_rgba_by_name=cast(dict[str, tuple[float, float, float, float]], rgba),
        )
        records.append(
            ResolvedHandAssetRecord(
                container=container,
                provenance=HandAssetProvenance(**cast(dict[str, Any], raw["provenance"])),
                content_hash=str(raw.get("content_hash", "")),
            )
        )
    return ResolvedHandAssetPartition(name=str(document.get("partition_name", "train")), records=tuple(records))


def _load_if_valid(
    dataset: HandAssetDataset,
    index_path: Path,
    payload_path: Path,
    *,
    require_geometry_semantics: bool,
    allow_legacy_left_handedness: bool,
) -> ResolvedHandAssetPartition | None:

    if not index_path.is_file() or not payload_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("schema_version") != PREPARED_TRAIN_CACHE_SCHEMA_VERSION
            or index.get("dataset_source_sha256") != dataset.source_sha256
            or index.get("resolver_implementation_sha256") != _resolver_implementation_sha256()
            or index.get("resolver_options")
            != {
                "require_geometry_semantics": require_geometry_semantics,
                "allow_legacy_left_handedness": allow_legacy_left_handedness,
            }
        ):
            return None
        dependencies = index.get("dependencies")
        if not isinstance(dependencies, list):
            return None
        for dependency in dependencies:
            path = Path(dependency["path"]).expanduser()
            if not path.is_file() or _sha256_file(path) != dependency.get("sha256"):
                return None
        payload_bytes = payload_path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != index.get("payload_sha256"):
            return None
        document = json.loads(payload_bytes)
        partition = _deserialize_partition(document)
        if len(partition.records) != int(index.get("record_count", -1)):
            return None
        return partition
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def resolve_prepared_train(
    dataset: HandAssetDataset,
    *,
    require_geometry_semantics: bool = True,
    allow_legacy_left_handedness: bool = False,
    max_assets: int | None = None,
    cache_root: Path | None = None,
) -> tuple[ResolvedHandAssetPartition, bool]:
    'Resolves prepared train.'

    if max_assets is not None:
        return (
            dataset.resolve_train(
                require_geometry_semantics=require_geometry_semantics,
                allow_legacy_left_handedness=allow_legacy_left_handedness,
                max_assets=max_assets,
            ),
            False,
        )
    index_path, payload_path = _cache_paths(dataset, cache_root)
    cached = _load_if_valid(
        dataset,
        index_path,
        payload_path,
        require_geometry_semantics=require_geometry_semantics,
        allow_legacy_left_handedness=allow_legacy_left_handedness,
    )
    if cached is not None:
        return cached, True

    partition = dataset.resolve_train(
        require_geometry_semantics=require_geometry_semantics,
        allow_legacy_left_handedness=allow_legacy_left_handedness,
    )
    payload = json.dumps(
        _serialize_partition(partition),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    index = {
        "schema_version": PREPARED_TRAIN_CACHE_SCHEMA_VERSION,
        "dataset_source_path": str(dataset.source_path),
        "dataset_source_sha256": dataset.source_sha256,
        "resolver_implementation_sha256": _resolver_implementation_sha256(),
        "resolver_options": {
            "require_geometry_semantics": require_geometry_semantics,
            "allow_legacy_left_handedness": allow_legacy_left_handedness,
        },
        "record_count": len(partition.records),
        "dependencies": _dependency_index(partition),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload_tmp = payload_path.with_suffix(".json.tmp")
    index_tmp = index_path.with_suffix(".json.tmp")
    payload_tmp.write_bytes(payload)
    index_tmp.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    payload_tmp.replace(payload_path)
    index_tmp.replace(index_path)
    return partition, False


__all__ = ["PREPARED_TRAIN_CACHE_SCHEMA_VERSION", "resolve_prepared_train"]
