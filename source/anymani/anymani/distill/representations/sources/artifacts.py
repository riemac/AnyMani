r"""跨 run 几何 source base 与 selected anchor shard 的静态 artifact。

格式只使用 ``.npy`` 与 canonical JSON。Base 保存 q-independent POE、owner mesh、home surface 和
provenance；每个 anchor shard 只保存一个 $A^{(k)}$。Warp BVH、CUDA handle、Trimesh 对象、RNG、
query、teacher target、batch 与 learned state 都在加载时重建或由运行期产生，绝不进入磁盘格式。
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from anymani.assets.bank import HandContainer

from .anchor_sampling import (
    AnchorClassificationStats,
    AnchorRealization,
    AnchorSamples,
    _anchor_realization_hash,
)
from .collision_geometry import (
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    OwnerSurfaceRecord,
    OwnerSurfaceSamplingArrays,
    WarpSurfaceAudit,
    WarpSurfaceView,
    prepare_owner_surface_sampling_arrays,
    prepare_warp_surface_view,
)
from .geometry_source import GeometrySourceCfg, GeometrySourceCore
from .kinematics import EmbodimentGeometrySpec

SOURCE_ARTIFACT_SCHEMA_VERSION = "1.0.0"
"""显式磁盘语义版本；任何数组或算法含义变化都必须升级。"""

_SOURCE_ALGORITHM_IDENTITY = {
    "owner_surface": "owner-surface-v2",
    "home_surface": "area-candidate-fps-v1",
    "query_surface_sampling": "owner-triangle-area-barycentric-v1",
    "anchor_sampling": "palm-seed-radial-gaussian-fps-fast-winding-v2",
}


@dataclass(frozen=True)
class SourceArtifactReference:
    """run lineage 记录的一项完整 artifact 身份。"""

    artifact_key: str
    manifest_digest: str
    relative_path: str


def _canonical_json(value: object) -> bytes:
    """以 UTF-8、稳定 key 顺序和无空白分隔符编码 JSON。"""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def source_artifact_key(
    container: HandContainer,
    config: GeometrySourceCfg,
    *,
    dataset_manifest_sha256: str = "",
    producer_device: str = "cpu",
) -> str:
    r"""由 dataset/content/source-config/schema 形成 base 与 shard 共用 key。"""

    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError("source artifact key requires typed geometry semantics")
    identity = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "asset_id": container.asset_id,
        "asset_content_hash": semantics.content_hash,
        "asset_bytes": _asset_byte_identity(container),
        "source_config": asdict(config),
        "unit_frame_contract": "length=m,joint=rad,hand_frame=h",
        "producer": {
            "algorithms": _SOURCE_ALGORITHM_IDENTITY,
            "trimesh": _package_version("trimesh"),
            "manifold3d": _package_version("manifold3d"),
            "warp": _package_version("warp-lang"),
            "torch": str(torch.__version__),
            "cuda_compute_capability": _cuda_compute_capability(producer_device),
        },
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _asset_byte_identity(container: HandContainer) -> dict[str, object]:
    r"""哈希 canonical URDF 与去重后的实际 mesh bytes；绝对路径不进入跨机器身份。"""

    urdf_path = getattr(container, "urdf_path", None)
    urdf_digest = _sha256_file(Path(urdf_path)) if urdf_path is not None else None
    unique_meshes: dict[str, Path] = {}
    for reference in getattr(container, "mesh_refs", ()):
        real_path = getattr(reference, "real_path", None)
        if real_path is not None:
            unique_meshes.setdefault(str(getattr(reference, "virtual_path", "")), Path(real_path))
    return {
        "urdf_sha256": urdf_digest,
        "meshes": [
            {"virtual_path": virtual_path, "sha256": _sha256_file(path)}
            for virtual_path, path in sorted(unique_meshes.items())
        ],
    }


def _sha256_file(path: Path) -> str:
    """流式读取可能较大的 mesh，避免 key 构造产生整文件峰值副本。"""

    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return _sha256_file_snapshot(str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=65_536)
def _sha256_file_snapshot(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """同一 preparation 进程复用未变化文件摘要；stat 四元组只作 memo key，不进入 identity。"""

    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    """缺少 optional backend 时也把 absent 明确写入 producer identity。"""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _cuda_compute_capability(device: str) -> str:
    """把 anchor/Warp float32 数值后端的 CUDA 架构写入 artifact key。"""

    parsed = torch.device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        return "not-applicable"
    major, minor = torch.cuda.get_device_capability(parsed)
    return f"sm_{major}{minor}"


class GeometrySourceArtifactStore:
    r"""严格读取或原子构建 base/anchor-shard artifact。"""

    def __init__(
        self,
        root: Path | str,
        *,
        mode: str,
        dataset_manifest_sha256: str = "",
        producer_device: str = "cpu",
    ) -> None:
        if mode not in {"readonly", "read-write", "off"}:
            raise ValueError("source artifact mode must be readonly, read-write, or off")
        self.root = Path(root)
        self.mode = mode
        self.dataset_manifest_sha256 = str(dataset_manifest_sha256)
        self.producer_device = str(producer_device)

    def identity(self) -> dict[str, object]:
        r"""返回不含机器本地 root、但足以重建所有 per-asset keys 的 store 身份。"""

        return {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "mode": self.mode,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "producer_device_type": torch.device(self.producer_device).type,
            "producer_compute_capability": _cuda_compute_capability(self.producer_device),
            "producer_versions": {
                "trimesh": _package_version("trimesh"),
                "manifold3d": _package_version("manifold3d"),
                "warp": _package_version("warp-lang"),
                "torch": str(torch.__version__),
            },
            "algorithms": dict(_SOURCE_ALGORITHM_IDENTITY),
        }

    def key(self, container: HandContainer, config: GeometrySourceCfg) -> str:
        """返回当前 store dataset identity 下的稳定 source key。"""

        return source_artifact_key(
            container,
            config,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            producer_device=self.producer_device,
        )

    def base_path(self, key: str) -> Path:
        return self.root / key / "base"

    def anchor_path(self, key: str, bank_index: int) -> Path:
        return self.root / key / "anchors" / f"bank_{bank_index:06d}"

    def load_base(self, container: HandContainer, config: GeometrySourceCfg) -> tuple[GeometrySourceCore, SourceArtifactReference]:
        """校验并重建 CPU core；readonly miss/corruption 不做任何 fallback。"""

        key = self.key(container, config)
        arrays, metadata, digest = self._read_directory(self.base_path(key))
        if metadata.get("kind") != "geometry_source_base" or metadata.get("artifact_key") != key:
            raise ValueError("source base manifest identity does not match requested artifact key")
        if metadata.get("source_config") != asdict(config):
            raise ValueError("source base manifest config does not match requested source config")
        core = _decode_base(container, arrays, metadata)
        return core, SourceArtifactReference(key, digest, f"{key}/base")

    def write_base(self, core: GeometrySourceCore, config: GeometrySourceCfg) -> SourceArtifactReference:
        """把已物化 CPU core 原子发布为 base artifact。"""

        if self.mode != "read-write":
            raise PermissionError("source base writes require read-write artifact mode")
        key = self.key(core.container, config)
        arrays, metadata = _encode_base(core)
        metadata.update(
            {
                "kind": "geometry_source_base",
                "artifact_key": key,
                "source_config": asdict(config),
            }
        )
        digest = self._write_directory(self.base_path(key), arrays, metadata)
        return SourceArtifactReference(key, digest, f"{key}/base")

    def load_anchor(
        self,
        container: HandContainer,
        config: GeometrySourceCfg,
        bank_index: int,
    ) -> tuple[AnchorRealization, AnchorClassificationStats, SourceArtifactReference]:
        """校验并恢复一个 selected anchor shard。"""

        key = self.key(container, config)
        arrays, metadata, digest = self._read_directory(self.anchor_path(key, bank_index))
        if metadata.get("kind") != "geometry_anchor_shard" or metadata.get("artifact_key") != key:
            raise ValueError("anchor shard manifest identity does not match requested artifact key")
        realization, stats = _decode_anchor(arrays, metadata)
        if realization.bank_index != bank_index or realization.bank_size != config.anchors.bank_size:
            raise ValueError("anchor shard bank identity does not match requested configuration")
        return realization, stats, SourceArtifactReference(
            key,
            digest,
            f"{key}/anchors/bank_{bank_index:06d}",
        )

    def write_anchor(
        self,
        container: HandContainer,
        config: GeometrySourceCfg,
        realization: AnchorRealization,
        stats: AnchorClassificationStats,
    ) -> SourceArtifactReference:
        """原子发布一个 selected anchor shard；wall time 不进入 deterministic manifest。"""

        if self.mode != "read-write":
            raise PermissionError("anchor shard writes require read-write artifact mode")
        key = self.key(container, config)
        arrays, metadata = _encode_anchor(realization, stats)
        metadata.update({"kind": "geometry_anchor_shard", "artifact_key": key})
        path = self.anchor_path(key, realization.bank_index)
        digest = self._write_directory(path, arrays, metadata)
        return SourceArtifactReference(key, digest, str(path.relative_to(self.root)))

    def _read_directory(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
        """逐文件验证一个完整目录，禁止 incomplete marker 或 pickle/object arrays。"""

        try:
            manifest_bytes = (path / "manifest.json").read_bytes()
            complete = (path / "COMPLETE").read_text(encoding="ascii").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"source artifact is missing or incomplete: {path}") from exc
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        if complete != digest:
            raise ValueError(f"source artifact COMPLETE digest mismatch: {path}")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported source artifact schema")
        metadata = manifest.get("metadata")
        records = manifest.get("arrays")
        if not isinstance(metadata, dict) or not isinstance(records, dict):
            raise ValueError("source artifact manifest lacks metadata/arrays mappings")
        arrays: dict[str, np.ndarray] = {}
        for name, record in records.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ValueError("source artifact array manifest is malformed")
            array_path = path / str(record["path"])
            payload = array_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                raise ValueError(f"source artifact array digest mismatch: {array_path}")
            if len(payload) != int(record.get("byte_count", -1)):
                raise ValueError(f"source artifact array byte count mismatch: {array_path}")
            array = np.load(array_path, allow_pickle=False)
            if str(array.dtype) != record.get("dtype") or list(array.shape) != record.get("shape"):
                raise ValueError(f"source artifact array dtype/shape mismatch: {array_path}")
            if array.dtype.hasobject:
                raise ValueError("source artifact object arrays are forbidden")
            arrays[name] = array
        return arrays, metadata, digest

    def _write_directory(self, target: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
        """在同一文件系统写临时目录、fsync 并原子 rename；并发结果必须 digest 相同。"""

        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.parent / f".{target.name}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.exists():
                try:
                    existing_arrays, existing_metadata, existing_digest = self._read_directory(target)
                    same_arrays = existing_arrays.keys() == arrays.keys() and all(
                        existing_arrays[name].dtype == np.asarray(arrays[name]).dtype
                        and existing_arrays[name].shape == np.asarray(arrays[name]).shape
                        and np.array_equal(existing_arrays[name], np.asarray(arrays[name]))
                        for name in existing_arrays
                    )
                    if existing_metadata != metadata or not same_arrays:
                        raise RuntimeError("existing source artifact differs from deterministic rebuild")
                    return existing_digest
                except RuntimeError:
                    raise
                except (OSError, ValueError):
                    quarantine = self.root / ".quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    target.rename(quarantine / f"{int(time.time())}-{target.parent.name}-{target.name}-{uuid.uuid4().hex}")
            temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            arrays_dir = temporary / "arrays"
            arrays_dir.mkdir(parents=True)
            try:
                records: dict[str, dict[str, object]] = {}
                for name in sorted(arrays):
                    array = np.ascontiguousarray(arrays[name])
                    if array.dtype.hasobject:
                        raise ValueError(f"source artifact array {name!r} cannot use object dtype")
                    relative = Path("arrays") / f"{name}.npy"
                    path = temporary / relative
                    np.save(path, array, allow_pickle=False)
                    _fsync_file(path)
                    payload = path.read_bytes()
                    records[name] = {
                        "path": relative.as_posix(),
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "order": "C",
                        "byte_count": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                manifest = {
                    "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                    "metadata": metadata,
                    "arrays": records,
                }
                manifest_bytes = _canonical_json(manifest)
                manifest_path = temporary / "manifest.json"
                manifest_path.write_bytes(manifest_bytes)
                _fsync_file(manifest_path)
                digest = hashlib.sha256(manifest_bytes).hexdigest()
                complete_path = temporary / "COMPLETE"
                complete_path.write_text(digest + "\n", encoding="ascii")
                _fsync_file(complete_path)
                _fsync_directory(arrays_dir)
                _fsync_directory(temporary)
                try:
                    temporary.rename(target)
                except FileExistsError:
                    _arrays, _metadata, concurrent_digest = self._read_directory(target)
                    if concurrent_digest != digest:
                        raise RuntimeError("concurrent source artifact writers produced different digests")
                    shutil.rmtree(temporary)
                    return concurrent_digest
                _fsync_directory(target.parent)
                return digest
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise


def _encode_base(core: GeometrySourceCore) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    optional_spec_fields: list[str] = []
    for field_info in fields(core.spec_cpu):
        value = getattr(core.spec_cpu, field_info.name)
        if isinstance(value, torch.Tensor):
            arrays[f"spec_{field_info.name}"] = value.detach().cpu().numpy()
        elif value is not None and field_info.name not in {
            "owner_ids",
            "joint_names",
            "owner_roles",
            "owner_finger_names",
            "owner_joint_indices",
        }:
            optional_spec_fields.append(field_info.name)
    arrays.update(
        {
            "home_points": core.home_surface.points_owner_local_m,
            "home_faces": core.home_surface.face_indices,
            "home_barycentric": core.home_surface.barycentric,
        }
    )
    sampling_arrays = core.surface_sampling_arrays or prepare_owner_surface_sampling_arrays(core.geometry_cache)
    warp_views = core.warp_surface_views or tuple(
        prepare_warp_surface_view(record.surface_mesh, owner_id=record.owner_id)
        for record in core.geometry_cache.records
    )
    if len(sampling_arrays.triangles_owner_local_m) != len(core.geometry_cache.records):
        raise ValueError("source artifact query-sampling owner count mismatch")
    if len(warp_views) != len(core.geometry_cache.records):
        raise ValueError("source artifact Warp-view owner count mismatch")
    record_metadata: list[dict[str, Any]] = []
    for index, record in enumerate(core.geometry_cache.records):
        arrays[f"owner_{index:03d}_surface_vertices"] = np.asarray(record.surface_mesh.vertices)
        arrays[f"owner_{index:03d}_surface_faces"] = np.asarray(record.surface_mesh.faces)
        if record.solid_mesh is not None:
            arrays[f"owner_{index:03d}_solid_vertices"] = np.asarray(record.solid_mesh.vertices)
            arrays[f"owner_{index:03d}_solid_faces"] = np.asarray(record.solid_mesh.faces)
        arrays[f"owner_{index:03d}_query_triangles"] = sampling_arrays.triangles_owner_local_m[index]
        arrays[f"owner_{index:03d}_query_normals"] = sampling_arrays.face_normals_owner_local[index]
        arrays[f"owner_{index:03d}_query_area_cdf"] = sampling_arrays.face_area_cdf[index]
        warp_view = warp_views[index]
        arrays[f"owner_{index:03d}_warp_vertices"] = warp_view.vertices
        arrays[f"owner_{index:03d}_warp_faces"] = warp_view.faces
        arrays[f"owner_{index:03d}_warp_source_faces"] = warp_view.source_face_indices
        arrays[f"owner_{index:03d}_warp_face_altitudes"] = warp_view.face_altitudes_m
        record_metadata.append(
            {
                "owner_id": record.owner_id,
                "owner_index": record.owner_index,
                "role": record.role,
                "finger_name": record.finger_name,
                "component_ids": list(record.component_ids),
                "boolean_applied": record.boolean_applied,
                "has_solid": record.solid_mesh is not None,
                "warp_surface_audit": asdict(warp_view.audit),
            }
        )
    metadata = {
        "asset_id": core.asset_id,
        "asset_content_hash": core.geometry_cache.asset_content_hash,
        "identity": asdict(core.identity),
        "spec_strings": {
            "owner_ids": list(core.spec_cpu.owner_ids),
            "joint_names": list(core.spec_cpu.joint_names),
            "owner_roles": list(core.spec_cpu.owner_roles),
            "owner_finger_names": list(core.spec_cpu.owner_finger_names),
            "owner_joint_indices": list(core.spec_cpu.owner_joint_indices),
        },
        "unsupported_optional_spec_fields": optional_spec_fields,
        "geometry_cache": {
            "boolean_backend": core.geometry_cache.boolean_backend,
            "surface_geometry_hash": core.geometry_cache.surface_geometry_hash,
            "surface_processing_version": core.geometry_cache.surface_processing_version,
            "records": record_metadata,
        },
        "home_surface": {
            "owner_ids": list(core.home_surface.owner_ids),
            "sampling_seed": core.home_surface.sampling_seed,
            "oversample_factor": core.home_surface.oversample_factor,
        },
    }
    if optional_spec_fields:
        raise TypeError(f"source artifact cannot encode spec fields: {optional_spec_fields}")
    return arrays, metadata


def _decode_base(
    container: HandContainer,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> GeometrySourceCore:
    if metadata.get("asset_id") != container.asset_id:
        raise ValueError("source base asset ID does not match requested container")
    spec_values: dict[str, Any] = {}
    string_fields = metadata["spec_strings"]
    for field_info in fields(EmbodimentGeometrySpec):
        array = arrays.get(f"spec_{field_info.name}")
        if array is not None:
            spec_values[field_info.name] = torch.from_numpy(np.array(array, copy=True))
        elif field_info.name in string_fields:
            values = string_fields[field_info.name]
            spec_values[field_info.name] = tuple(values)
    spec = EmbodimentGeometrySpec(**spec_values)
    records = []
    query_triangles: list[np.ndarray] = []
    query_normals: list[np.ndarray] = []
    query_cdfs: list[np.ndarray] = []
    warp_views: list[WarpSurfaceView] = []
    for index, record in enumerate(metadata["geometry_cache"]["records"]):
        surface = trimesh.Trimesh(
            vertices=arrays[f"owner_{index:03d}_surface_vertices"],
            faces=arrays[f"owner_{index:03d}_surface_faces"],
            process=False,
        )
        solid = None
        if record["has_solid"]:
            solid = trimesh.Trimesh(
                vertices=arrays[f"owner_{index:03d}_solid_vertices"],
                faces=arrays[f"owner_{index:03d}_solid_faces"],
                process=False,
            )
        records.append(
            OwnerSurfaceRecord(
                owner_id=record["owner_id"],
                owner_index=int(record["owner_index"]),
                role=record["role"],
                finger_name=record["finger_name"],
                component_ids=tuple(record["component_ids"]),
                surface_mesh=surface,
                solid_mesh=solid,
                boolean_applied=bool(record["boolean_applied"]),
            )
        )
        query_triangles.append(arrays[f"owner_{index:03d}_query_triangles"])
        query_normals.append(arrays[f"owner_{index:03d}_query_normals"])
        query_cdfs.append(arrays[f"owner_{index:03d}_query_area_cdf"])
        warp_views.append(
            WarpSurfaceView(
                vertices=arrays[f"owner_{index:03d}_warp_vertices"],
                faces=arrays[f"owner_{index:03d}_warp_faces"],
                source_face_indices=arrays[f"owner_{index:03d}_warp_source_faces"],
                face_altitudes_m=arrays[f"owner_{index:03d}_warp_face_altitudes"],
                audit=WarpSurfaceAudit(**record["warp_surface_audit"]),
            )
        )
    cache_metadata = metadata["geometry_cache"]
    cache = OwnerGeometryCache(
        asset_id=container.asset_id,
        asset_content_hash=metadata["asset_content_hash"],
        boolean_backend=cache_metadata["boolean_backend"],
        records=tuple(records),
        surface_geometry_hash=cache_metadata["surface_geometry_hash"],
        surface_processing_version=cache_metadata["surface_processing_version"],
    )
    home_metadata = metadata["home_surface"]
    home = HomeSurfaceSamples(
        owner_ids=tuple(home_metadata["owner_ids"]),
        points_owner_local_m=arrays["home_points"],
        face_indices=arrays["home_faces"],
        barycentric=arrays["home_barycentric"],
        sampling_seed=int(home_metadata["sampling_seed"]),
        oversample_factor=int(home_metadata["oversample_factor"]),
    )
    identity = GeometryIdentity(**metadata["identity"])
    sampling_arrays = OwnerSurfaceSamplingArrays(tuple(query_triangles), tuple(query_normals), tuple(query_cdfs))
    return GeometrySourceCore(container, spec, cache, home, identity, sampling_arrays, tuple(warp_views))


def _encode_anchor(
    realization: AnchorRealization,
    stats: AnchorClassificationStats,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    samples = realization.samples
    arrays = {
        "anchors_hand_m": samples.anchors_hand_m,
        "surface_mask": samples.surface_mask,
    }
    metadata = {
        "bank_index": realization.bank_index,
        "bank_size": realization.bank_size,
        "root_seed": realization.root_seed,
        "derived_seed": realization.derived_seed,
        "realization_hash": realization.realization_hash,
        "sampling_version": realization.sampling_version,
        "finger_names": list(samples.finger_names),
        "seed_ids": list(samples.seed_ids),
        "radial_support_radius_m": samples.radial_support_radius_m,
        "radial_decay_scale_m": samples.radial_decay_scale_m,
        "surface_fraction": samples.surface_fraction,
        "classifier": {
            "query_point_count": stats.query_point_count,
            "kernel_launch_count": stats.kernel_launch_count,
            "boundary_recheck_count": stats.boundary_recheck_count,
            "boundary_disagreement_count": stats.boundary_disagreement_count,
        },
        "producer": {
            "warp": _package_version("warp-lang"),
            "torch": str(torch.__version__),
            "cuda_compute_capability": (
                f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
                if torch.cuda.is_available()
                else "not-applicable"
            ),
        },
    }
    return arrays, metadata


def _decode_anchor(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[AnchorRealization, AnchorClassificationStats]:
    samples = AnchorSamples(
        anchors_hand_m=arrays["anchors_hand_m"],
        finger_names=tuple(metadata["finger_names"]),
        seed_ids=tuple(metadata["seed_ids"]),
        surface_mask=arrays["surface_mask"],
        radial_support_radius_m=float(metadata["radial_support_radius_m"]),
        radial_decay_scale_m=float(metadata["radial_decay_scale_m"]),
        surface_fraction=float(metadata["surface_fraction"]),
        sampling_seed=int(metadata["derived_seed"]),
        algorithm_version=str(metadata["sampling_version"]),
    )
    if _anchor_realization_hash(samples) != metadata["realization_hash"]:
        raise ValueError("anchor shard realization hash does not match reconstructed samples")
    realization = AnchorRealization(
        bank_index=int(metadata["bank_index"]),
        bank_size=int(metadata["bank_size"]),
        root_seed=int(metadata["root_seed"]),
        derived_seed=int(metadata["derived_seed"]),
        samples=samples,
        realization_hash=str(metadata["realization_hash"]),
        sampling_version=str(metadata["sampling_version"]),
    )
    classifier = metadata["classifier"]
    stats = AnchorClassificationStats(
        query_point_count=int(classifier["query_point_count"]),
        kernel_launch_count=int(classifier["kernel_launch_count"]),
        boundary_recheck_count=int(classifier["boundary_recheck_count"]),
        boundary_disagreement_count=int(classifier["boundary_disagreement_count"]),
        elapsed_seconds=0.0,
    )
    return realization, stats


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "GeometrySourceArtifactStore",
    "SOURCE_ARTIFACT_SCHEMA_VERSION",
    "SourceArtifactReference",
    "source_artifact_key",
]
