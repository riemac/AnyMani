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
import sqlite3
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
    WarpSurfaceAudit,
    WarpSurfaceView,
    prepare_owner_surface_sampling_arrays,
    prepare_warp_surface_view,
)
from .geometry_source import GeometrySourceCfg, GeometrySourceCore
from .kinematics import EmbodimentGeometrySpec

SOURCE_ARTIFACT_SCHEMA_VERSION = "2.0.0"
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
    r"""由资产 bytes/typed semantics/source config/schema 形成跨实验复用的 canonical base key。

    Dataset manifest、模型、loss、optimizer、seed、epochs、实验快照与 producer GPU 均不改变
    q-independent canonical source，因此明确排除在 base identity 之外。
    """

    del dataset_manifest_sha256, producer_device
    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError("source artifact key requires typed geometry semantics")
    identity = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "asset_content_hash": semantics.content_hash,
        "asset_bytes": _asset_byte_identity(container),
        "source_config": asdict(config),
        "unit_frame_contract": "length=m,joint=rad,hand_frame=h",
        "algorithms": {
            "owner_surface": _SOURCE_ALGORITHM_IDENTITY["owner_surface"],
            "home_surface": _SOURCE_ALGORITHM_IDENTITY["home_surface"],
            "query_surface_sampling": _SOURCE_ALGORITHM_IDENTITY["query_surface_sampling"],
        },
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def anchor_artifact_key(
    container: HandContainer,
    config: GeometrySourceCfg,
    bank_index: int,
    *,
    producer_device: str,
) -> str:
    r"""形成 backend-specific selected-anchor object key。

    Anchor inside-classification 依赖 Warp/Torch/CUDA capability 与 bank index；这些执行身份只进入
    accelerator key，不污染可跨硬件复用的 canonical base key。
    """

    identity = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "base_key": source_artifact_key(container, config),
        "bank_index": int(bank_index),
        "bank_size": int(config.anchors.bank_size),
        "sampling_algorithm": _SOURCE_ALGORITHM_IDENTITY["anchor_sampling"],
        "producer": {
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
        role: str = "train",
        soft_limit_bytes: int = 32 * 1024**3,
        hard_limit_bytes: int = 48 * 1024**3,
        minimum_filesystem_reserve_bytes: int = 100 * 1024**3,
    ) -> None:
        if mode not in {"readonly", "read-write", "off"}:
            raise ValueError("source artifact mode must be readonly, read-write, or off")
        # object/index/lock/relative lineage 必须共享同一绝对根；相对 cwd 只允许在 CLI 配置边界出现。
        self.root = Path(root).expanduser().resolve(strict=False)
        self.mode = mode
        self.dataset_manifest_sha256 = str(dataset_manifest_sha256)
        self.producer_device = str(producer_device)
        if role not in {"train", "evaluation"}:
            raise ValueError("source artifact role must be train or evaluation")
        if not 0 < soft_limit_bytes <= hard_limit_bytes or minimum_filesystem_reserve_bytes < 0:
            raise ValueError("source cache capacity limits must satisfy 0 < soft <= hard and reserve >= 0")
        self.role = role
        self.soft_limit_bytes = int(soft_limit_bytes)
        self.hard_limit_bytes = int(hard_limit_bytes)
        self.minimum_filesystem_reserve_bytes = int(minimum_filesystem_reserve_bytes)
        self._index_read_count = 0
        if self.mode != "off":
            self.root.mkdir(parents=True, exist_ok=True)
            self._initialize_indexes()

    def identity(self) -> dict[str, object]:
        r"""返回不含机器本地 root、但足以重建所有 per-asset keys 的 store 身份。"""

        return {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "mode": self.mode,
            "role": self.role,
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

    def anchor_key(self, container: HandContainer, config: GeometrySourceCfg, bank_index: int) -> str:
        r"""返回当前 backend/bank 的 accelerator object key。"""

        return anchor_artifact_key(
            container,
            config,
            bank_index,
            producer_device=self.producer_device,
        )

    def base_path(self, key: str) -> Path:
        return self.root / "objects" / "base" / key[:2] / key

    def anchor_path(self, key: str) -> Path:
        return self.root / "objects" / "anchor" / key[:2] / key

    def load_base(self, container: HandContainer, config: GeometrySourceCfg) -> tuple[GeometrySourceCore, SourceArtifactReference]:
        """校验并重建 CPU core；readonly miss/corruption 不做任何 fallback。"""

        key = self.key(container, config)
        path = self._resolve_role_object(container.asset_id, "base", -1, key, self.base_path(key))
        arrays, metadata, digest = self._read_directory(path)
        if metadata.get("kind") != "geometry_source_base" or metadata.get("artifact_key") != key:
            raise ValueError("source base manifest identity does not match requested artifact key")
        if metadata.get("source_config") != asdict(config):
            raise ValueError("source base manifest config does not match requested source config")
        core = _decode_base(container, arrays, metadata)
        reference = SourceArtifactReference(key, digest, str(path.relative_to(self.root)))
        self._register_role_reference(container.asset_id, "base", -1, reference)
        return core, reference

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
        path = self.base_path(key)
        # 容量账本与对象发布共用一把锁；否则两个并发 miss 都可能在同一旧账本上通过 hard-limit gate。
        with (self.root / ".capacity.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # 已完成的 deterministic object 不产生新字节；并发 writer 不能因账本已接近 hard limit
            # 而在本应复用现有 object 时误报容量失败。损坏/半成品仍必须重新通过容量 gate。
            try:
                self._read_directory(path)
            except (FileNotFoundError, OSError, ValueError):
                self._require_capacity(arrays, metadata)
            digest = self._write_directory(path, arrays, metadata)
            reference = SourceArtifactReference(key, digest, str(path.relative_to(self.root)))
            self._register_object(reference, kind="base")
            self._register_role_reference(core.asset_id, "base", -1, reference)
        return reference

    def load_anchor(
        self,
        container: HandContainer,
        config: GeometrySourceCfg,
        bank_index: int,
    ) -> tuple[AnchorRealization, AnchorClassificationStats, SourceArtifactReference]:
        """校验并恢复一个 selected anchor shard。"""

        key = self.anchor_key(container, config, bank_index)
        path = self._resolve_role_object(container.asset_id, "anchor", bank_index, key, self.anchor_path(key))
        arrays, metadata, digest = self._read_directory(path)
        if metadata.get("kind") != "geometry_anchor_shard" or metadata.get("artifact_key") != key:
            raise ValueError("anchor shard manifest identity does not match requested artifact key")
        realization, stats = _decode_anchor(arrays, metadata)
        if realization.bank_index != bank_index or realization.bank_size != config.anchors.bank_size:
            raise ValueError("anchor shard bank identity does not match requested configuration")
        reference = SourceArtifactReference(key, digest, str(path.relative_to(self.root)))
        self._register_role_reference(container.asset_id, "anchor", bank_index, reference)
        return realization, stats, reference

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
        key = self.anchor_key(container, config, realization.bank_index)
        arrays, metadata = _encode_anchor(realization, stats)
        metadata.update({"kind": "geometry_anchor_shard", "artifact_key": key})
        path = self.anchor_path(key)
        # anchor shard 与 base 共用容量锁，保证 projected bytes 是对象库的线性化视图。
        with (self.root / ".capacity.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._read_directory(path)
            except (FileNotFoundError, OSError, ValueError):
                self._require_capacity(arrays, metadata)
            digest = self._write_directory(path, arrays, metadata)
            reference = SourceArtifactReference(key, digest, str(path.relative_to(self.root)))
            self._register_object(reference, kind="anchor")
            self._register_role_reference(container.asset_id, "anchor", realization.bank_index, reference)
        return reference

    def index_evidence(self) -> dict[str, object]:
        r"""返回当前角色 index 读取次数与共享 object store 账面占用。"""

        with self._connect(self.root / "objects.sqlite") as connection:
            row = connection.execute("SELECT COUNT(*), COALESCE(SUM(byte_count), 0) FROM objects").fetchone()
        return {
            "role": self.role,
            "role_index_read_count": self._index_read_count,
            "object_count": int(row[0]),
            "object_bytes": int(row[1]),
        }

    def _connect(self, path: Path) -> sqlite3.Connection:
        r"""打开一个短事务 SQLite connection；WAL schema 初始化由文件锁串行完成。"""

        connection = sqlite3.connect(path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize_indexes(self) -> None:
        r"""初始化共享 object catalog 与当前角色唯一 index；不会打开另一角色数据库。"""

        indexes = self.root / "indexes"
        indexes.mkdir(parents=True, exist_ok=True)
        with (self.root / ".indexes.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with self._connect(self.root / "objects.sqlite") as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS objects ("
                    "object_key TEXT PRIMARY KEY, kind TEXT NOT NULL, relative_path TEXT NOT NULL UNIQUE, "
                    "manifest_digest TEXT NOT NULL, byte_count INTEGER NOT NULL, schema_version TEXT NOT NULL)"
                )
            with self._connect(indexes / f"{self.role}.sqlite") as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS role_objects ("
                    "asset_id TEXT NOT NULL, kind TEXT NOT NULL, bank_index INTEGER NOT NULL, "
                    "object_key TEXT NOT NULL, manifest_digest TEXT NOT NULL, relative_path TEXT NOT NULL, "
                    "PRIMARY KEY(asset_id, kind, bank_index))"
                )

    def _resolve_role_object(
        self,
        asset_id: str,
        kind: str,
        bank_index: int,
        expected_key: str,
        expected_path: Path,
    ) -> Path:
        r"""只查询当前 role index；miss 时按内容 key 复用共享 object，不扫描其他 role。"""

        self._index_read_count += 1
        with self._connect(self.root / "indexes" / f"{self.role}.sqlite") as connection:
            row = connection.execute(
                "SELECT object_key, relative_path FROM role_objects WHERE asset_id=? AND kind=? AND bank_index=?",
                (asset_id, kind, bank_index),
            ).fetchone()
        if row is not None and str(row[0]) != expected_key:
            raise ValueError(f"source role index identity mismatch for asset={asset_id!r} kind={kind!r}")
        if row is not None:
            relative_path = str(row[1])
        else:
            # role miss 只查询共享 object catalog；绝不打开另一 role index，因此 evaluation 写入的
            # immutable object 可以被 train 复用，而 train 的访问证据仍只落在 train.sqlite。
            with self._connect(self.root / "objects.sqlite") as connection:
                object_row = connection.execute(
                    "SELECT kind, relative_path FROM objects WHERE object_key=?",
                    (expected_key,),
                ).fetchone()
            if object_row is None:
                return expected_path
            if str(object_row[0]) != kind:
                raise ValueError(
                    f"shared source object kind mismatch for key={expected_key!r}: "
                    f"expected={kind!r}, actual={object_row[0]!r}"
                )
            relative_path = str(object_row[1])
        candidate = (self.root / relative_path).resolve(strict=False)
        if self.root.resolve() not in candidate.parents:
            raise ValueError(f"source role index path escapes cache root: {relative_path!r}")
        return candidate

    def _register_role_reference(
        self,
        asset_id: str,
        kind: str,
        bank_index: int,
        reference: SourceArtifactReference,
    ) -> None:
        r"""事务发布当前角色到 immutable object 的引用；readonly 不改写 index。"""

        if self.mode != "read-write":
            return
        with self._connect(self.root / "indexes" / f"{self.role}.sqlite") as connection:
            connection.execute(
                "INSERT INTO role_objects VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(asset_id,kind,bank_index) DO UPDATE SET "
                "object_key=excluded.object_key, manifest_digest=excluded.manifest_digest, "
                "relative_path=excluded.relative_path",
                (
                    asset_id,
                    kind,
                    bank_index,
                    reference.artifact_key,
                    reference.manifest_digest,
                    reference.relative_path,
                ),
            )

    def _register_object(self, reference: SourceArtifactReference, *, kind: str) -> None:
        r"""把 COMPLETE object 的实际目录字节数写入共享容量账本。"""

        path = self.root / reference.relative_path
        byte_count = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        with self._connect(self.root / "objects.sqlite") as connection:
            connection.execute(
                "INSERT OR IGNORE INTO objects VALUES(?,?,?,?,?,?)",
                (
                    reference.artifact_key,
                    kind,
                    reference.relative_path,
                    reference.manifest_digest,
                    byte_count,
                    SOURCE_ARTIFACT_SCHEMA_VERSION,
                ),
            )

    def _require_capacity(self, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        r"""在写 temp object 前执行 32/48 GiB 与文件系统 reserve gate。"""

        with self._connect(self.root / "objects.sqlite") as connection:
            current = int(connection.execute("SELECT COALESCE(SUM(byte_count), 0) FROM objects").fetchone()[0])
        estimated = sum(np.asarray(array).nbytes + 256 for array in arrays.values()) + len(_canonical_json(metadata)) + 4096
        projected = current + estimated
        disk_free = shutil.disk_usage(self.root).free
        if projected > self.hard_limit_bytes:
            raise OSError(f"source cache hard limit would be exceeded: projected={projected}, hard={self.hard_limit_bytes}")
        if disk_free - estimated < self.minimum_filesystem_reserve_bytes:
            raise OSError(
                "source cache minimum filesystem reserve would be violated: "
                f"free={disk_free}, estimated={estimated}, reserve={self.minimum_filesystem_reserve_bytes}"
            )
        if projected > self.soft_limit_bytes:
            print(f"[SSL cache] soft limit exceeded: projected={projected} soft={self.soft_limit_bytes}")

    def audit_directory(self, path: Path) -> dict[str, int]:
        r"""显式深审计一个 COMPLETE object 的全部 payload SHA-256。"""

        arrays, _metadata, _digest = self._read_directory(path, verify_checksums=True)
        return {"array_count": len(arrays), "payload_bytes": sum(array.nbytes for array in arrays.values())}

    def _read_directory(
        self,
        path: Path,
        *,
        verify_checksums: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
        r"""快速验证 COMPLETE/schema/size；显式 audit 才读取全 payload checksum。"""

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
            if array_path.stat().st_size != int(record.get("byte_count", -1)):
                raise ValueError(f"source artifact array byte count mismatch: {array_path}")
            if verify_checksums:
                payload = array_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                    raise ValueError(f"source artifact array digest mismatch: {array_path}")
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
    warp_views = core.warp_surface_views or tuple(
        prepare_warp_surface_view(record.surface_mesh, owner_id=record.owner_id)
        for record in core.geometry_cache.records
    )
    if len(warp_views) != len(core.geometry_cache.records):
        raise ValueError("source artifact Warp-view owner count mismatch")
    record_metadata: list[dict[str, Any]] = []
    for index, record in enumerate(core.geometry_cache.records):
        surface_vertices = np.asarray(record.surface_mesh.vertices)
        surface_faces = np.asarray(record.surface_mesh.faces, dtype=np.int32)
        arrays[f"owner_{index:03d}_surface_vertices"] = surface_vertices
        arrays[f"owner_{index:03d}_surface_faces"] = surface_faces
        solid_storage = "none"
        if record.solid_mesh is not None:
            solid_vertices = np.asarray(record.solid_mesh.vertices)
            solid_faces = np.asarray(record.solid_mesh.faces, dtype=np.int32)
            if np.array_equal(surface_vertices, solid_vertices) and np.array_equal(surface_faces, solid_faces):
                solid_storage = "alias_surface"
            else:
                solid_storage = "separate"
                arrays[f"owner_{index:03d}_solid_vertices"] = solid_vertices
                arrays[f"owner_{index:03d}_solid_faces"] = solid_faces
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
                "solid_storage": solid_storage,
                "warp_surface_audit": asdict(warp_view.audit),
            }
        )
    metadata = {
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
    semantics = container.geometry_semantics
    if semantics is None or metadata.get("asset_content_hash") != semantics.content_hash:
        raise ValueError("source base asset content identity does not match requested container")
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
    warp_views: list[WarpSurfaceView] = []
    for index, record in enumerate(metadata["geometry_cache"]["records"]):
        surface = trimesh.Trimesh(
            vertices=arrays[f"owner_{index:03d}_surface_vertices"],
            faces=arrays[f"owner_{index:03d}_surface_faces"],
            process=False,
        )
        solid = None
        solid_storage = record.get("solid_storage")
        if solid_storage == "alias_surface":
            solid = surface.copy()
        elif solid_storage == "separate":
            solid = trimesh.Trimesh(
                vertices=arrays[f"owner_{index:03d}_solid_vertices"],
                faces=arrays[f"owner_{index:03d}_solid_faces"],
                process=False,
            )
        elif solid_storage != "none":
            raise ValueError(f"unknown compact solid storage mode={solid_storage!r}")
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
    sampling_arrays = prepare_owner_surface_sampling_arrays(cache)
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


def source_cache_status(root: Path | str) -> dict[str, object]:
    r"""读取共享 object catalog 与两个 role index 的容量/引用统计。

    ``status`` 只读取 SQLite 账本和文件元数据，不加载 ``.npy`` payload，因此不会把 cache 维护误作
    深度 checksum audit。role index 分开统计，能够直接核对 train/evaluation 是否发生越权引用。
    """

    cache_root = Path(root).expanduser().resolve()
    object_db = cache_root / "objects.sqlite"
    if not object_db.is_file():
        return {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "root": str(cache_root),
            "object_count": 0,
            "object_bytes": 0,
            "objects_by_kind": {},
            "roles": {},
        }
    with sqlite3.connect(object_db) as connection:
        rows = connection.execute(
            "SELECT kind, schema_version, COUNT(*), COALESCE(SUM(byte_count), 0) "
            "FROM objects GROUP BY kind, schema_version ORDER BY kind, schema_version"
        ).fetchall()
        object_count, object_bytes = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(byte_count), 0) FROM objects"
        ).fetchone()
    objects_by_kind = {
        f"{str(kind)}:{str(schema)}": {"count": int(count), "bytes": int(byte_count)}
        for kind, schema, count, byte_count in rows
    }
    roles: dict[str, object] = {}
    for role in ("train", "evaluation"):
        index_path = cache_root / "indexes" / f"{role}.sqlite"
        if not index_path.is_file():
            continue
        with sqlite3.connect(index_path) as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM role_objects").fetchone()[0])
            by_kind = connection.execute(
                "SELECT kind, COUNT(*) FROM role_objects GROUP BY kind ORDER BY kind"
            ).fetchall()
        roles[role] = {
            "reference_count": count,
            "references_by_kind": {str(kind): int(value) for kind, value in by_kind},
        }
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "root": str(cache_root),
        "object_count": int(object_count),
        "object_bytes": int(object_bytes),
        "objects_by_kind": objects_by_kind,
        "roles": roles,
    }


def audit_source_cache(root: Path | str) -> dict[str, object]:
    r"""对 object catalog 中每个 COMPLETE object 执行显式 payload checksum audit。"""

    cache_root = Path(root).expanduser().resolve()
    object_db = cache_root / "objects.sqlite"
    if not object_db.is_file():
        return {"schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION, "root": str(cache_root), "object_count": 0}
    # 维护命令只读取共享对象目录；不初始化或打开任一 role index，避免审计产生新的运行时引用。
    store = GeometrySourceArtifactStore.__new__(GeometrySourceArtifactStore)
    store.root = cache_root
    with sqlite3.connect(object_db) as connection:
        rows = connection.execute("SELECT relative_path FROM objects ORDER BY relative_path").fetchall()
    payload_bytes = 0
    for (relative_path,) in rows:
        evidence = store.audit_directory(cache_root / str(relative_path))
        payload_bytes += int(evidence["payload_bytes"])
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "root": str(cache_root),
        "object_count": len(rows),
        "payload_bytes": payload_bytes,
        "status": "ok",
    }


def prune_source_cache(root: Path | str, *, apply: bool = False) -> dict[str, object]:
    r"""报告或删除无 train/evaluation role 引用的旧 object；默认只报告不改盘。

    当前 schema 的被引用 object 永远保留。只有不被任一 role index 引用、或已属于旧 schema 的对象才
    进入候选集合；``apply=True`` 才执行目录删除和 object catalog 行删除，并以维护锁串行化该动作。
    """

    cache_root = Path(root).expanduser().resolve()
    object_db = cache_root / "objects.sqlite"
    if not object_db.is_file():
        return {"schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION, "root": str(cache_root), "candidates": [], "applied": apply}
    referenced: set[str] = set()
    for role in ("train", "evaluation"):
        index_path = cache_root / "indexes" / f"{role}.sqlite"
        if not index_path.is_file():
            continue
        with sqlite3.connect(index_path) as connection:
            referenced.update(str(row[0]) for row in connection.execute("SELECT DISTINCT object_key FROM role_objects"))
    with (cache_root / ".maintenance.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with sqlite3.connect(object_db) as connection:
            rows = connection.execute(
                "SELECT object_key, relative_path, schema_version, byte_count FROM objects ORDER BY object_key"
            ).fetchall()
            candidates = [
                {
                    "object_key": str(key),
                    "relative_path": str(relative_path),
                    "schema_version": str(schema),
                    "byte_count": int(byte_count),
                    "reason": "unreferenced",
                }
                for key, relative_path, schema, byte_count in rows
                if str(key) not in referenced
            ]
            stale_schema_references = [
                {"object_key": str(key), "schema_version": str(schema)}
                for key, _relative_path, schema, _byte_count in rows
                if str(key) in referenced and str(schema) != SOURCE_ARTIFACT_SCHEMA_VERSION
            ]
            if apply:
                for item in candidates:
                    relative_path = Path(str(item["relative_path"]))
                    target = (cache_root / relative_path).resolve()
                    if cache_root not in target.parents:
                        raise ValueError(f"refusing to prune path outside source cache root: {target}")
                    if target.is_dir():
                        shutil.rmtree(target)
                    connection.execute("DELETE FROM objects WHERE object_key=?", (item["object_key"],))
                connection.commit()
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "root": str(cache_root),
        "candidates": candidates,
        "candidate_bytes": sum(int(item["byte_count"]) for item in candidates),
        "stale_schema_references": stale_schema_references,
        "applied": apply,
    }


__all__ = [
    "audit_source_cache",
    "GeometrySourceArtifactStore",
    "prune_source_cache",
    "SOURCE_ARTIFACT_SCHEMA_VERSION",
    "SourceArtifactReference",
    "source_cache_status",
    "source_artifact_key",
]
