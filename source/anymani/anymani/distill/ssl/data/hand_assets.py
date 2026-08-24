r"""HandAssetDataset 到平等 embodiment catalog 的 data role。

Data role 只回答“本次实验有哪些资产以及它们属于哪个 partition”。它不采样 $q$，不生成 query、
sigma 或监督目标，也不把 mother/family 等 provenance 转成隐藏训练权重。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from anymani.assets.bank.dataset import (
    HandAssetDataset,
    ResolvedHandAssetDataset,
    ResolvedHandAssetPartition,
)
from anymani.assets.bank.hand_container import HandContainer

_CATALOG_CACHE_SCHEMA = "hand-asset-catalog-cache-v3"
_CATALOG_CACHE_MAX_BYTES = 512 * 1024 * 1024  # 最多保留 512 MiB 本机解析索引
_STALE_TEMPORARY_AGE_SECONDS = 24 * 60 * 60  # 只回收确认不属于活跃写入的临时文件


def _release_catalog_allocator_slack() -> bool:
    r"""在 cold resolve 转为 slim catalog 后归还完整 sidecar 的 CPython/glibc 空闲页。

    该回收发生在 CUDA/Warp 初始化之前，只处理已经没有对象引用的 host allocation；不改变
    ``ResolvedHandAssetDataset`` 的资产顺序、typed semantics、路径、identity 或 provenance。
    """

    import ctypes
    import gc

    gc.collect()  # 清理 worker/result 与完整 HandContainer 字典可能形成的 generation-2 cycles
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
    except (AttributeError, OSError):
        return False
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))  # GNU libc 以 pad=0 归还全部可释放 top chunks


def _cache_root() -> Path:
    r"""返回跨进程共享的本地 catalog cache 根目录。

    cache 只保存由本机已解析资产重新构造出的 Python 数据对象，不属于实验结果或
    数据集版本控制内容。默认遵循 ``XDG_CACHE_HOME``，因此新的 SSL 进程可以复用同一
    份 catalog；测试和受限环境可以通过环境变量改变位置。
    """

    configured = os.environ.get("ANYMANI_CACHE_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser() / "anymani"
    return root / "ssl" / "asset_catalog"


def _file_signature(path: Path) -> tuple[str, int, int] | None:
    r"""返回单文件的快速身份元组 ``(path, size, mtime_ns)``。

    cache 命中检查只做 stat，不重新读取 YAML/XML/mesh 内容。生成资产树是不可变发布
    产物时，size 与纳秒级修改时间足以捕获实际替换；任何缺失文件都使 cache 失效并回到
    完整解析路径，由 ``HandContainer.from_cfg`` 执行严格内容校验。
    """

    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def _directory_signature(path: Path) -> tuple[str, int, int] | None:
    r"""返回目录 mtime 身份，用于发现 variant set 中新增或删除的 leaf。"""

    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_dir():
        return None
    return str(path), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _catalog_paths(dataset: ResolvedHandAssetDataset) -> tuple[tuple[str, ...], tuple[str, ...]]:
    r"""收集 cache 命中检查所需的文件和声明目录路径。

    catalog 只冻结 URDF、sidecar 与 partition 展开结果；mesh freshness 在按需 materialize
    physical source 时核对。目录路径覆盖 manifest provenance 中的 run/mother/leaf 层级，
    从而能发现不触碰 manifest 的资产增删。
    """

    files: set[str] = set()
    directories: set[str] = set()
    partitions = (dataset.train, *dataset.validation.values(), *dataset.evaluation.values())
    for partition in partitions:
        for record in partition.records:
            container = record.container
            files.add(str(container.urdf_path))
            files.add(str(container.sidecar_path))
            directories.add(str(container.urdf_path.parent))
            provenance = record.provenance
            for raw_path in (provenance.run_dir, provenance.mother_path):
                if raw_path:
                    directories.add(str(Path(raw_path)))
    return tuple(sorted(files)), tuple(sorted(directories))


def _catalog_fingerprint(
    dataset: ResolvedHandAssetDataset,
    *,
    files: tuple[str, ...],
    directories: tuple[str, ...],
) -> str:
    r"""计算不读取资产内容的 cache freshness fingerprint。"""

    digest = hashlib.sha256()
    digest.update(dataset.source_sha256.encode("ascii"))
    for path in files:
        signature = _file_signature(Path(path))
        digest.update(repr(signature).encode("utf-8"))
    for path in directories:
        signature = _directory_signature(Path(path))
        digest.update(repr(signature).encode("utf-8"))
    return digest.hexdigest()


def _cache_file(manifest: str, *, allow_legacy_left_handedness: bool) -> Path:
    r"""按 manifest 路径和解析安全选项定位 cache 文件。"""

    identity = hashlib.sha256(str(Path(manifest).expanduser().resolve()).encode("utf-8")).hexdigest()
    option = "legacy" if allow_legacy_left_handedness else "strict"
    return _cache_root() / f"{identity}.{option}.pkl"


def _prune_catalog_cache(
    root: Path,
    *,
    keep: Path | None = None,
    max_bytes: int = _CATALOG_CACHE_MAX_BYTES,
    now: float | None = None,
) -> None:
    r"""把可重建 catalog 索引限制在固定磁盘预算内。

    当前 manifest 对应文件即使单独超过预算也保留；其余 ``.pkl`` 按最旧访问时间回收。
    原子写临时文件只有在超过 24 小时后才删除，避免并行 SSL 进程互相破坏活跃写入。

    Args:
        root (Path): ``asset_catalog`` 根目录。
        keep (Path | None): 当前命中或刚写完的索引，不参与本轮驱逐。
        max_bytes (int): 全部完整 ``.pkl`` 的软上限，默认 512 MiB。
        now (float | None): 测试可注入的 POSIX 时间；正式路径使用当前时间。
    """

    if max_bytes < 1:
        raise ValueError("catalog cache byte limit must be positive")
    if not root.is_dir():
        return
    protected = keep.resolve(strict=False) if keep is not None else None  # 当前运行使用的完整索引
    current_time = time.time() if now is None else float(now)  # stale 判断统一使用 wall-clock 秒

    # ``NamedTemporaryFile`` 产物以点开头且含 ``.pkl.``；只清理跨日残留，活跃写入保持不动。
    for path in root.iterdir():
        if not path.name.startswith(".") or ".pkl." not in path.name:
            continue
        try:
            if current_time - path.stat().st_mtime >= _STALE_TEMPORARY_AGE_SECONDS:
                path.unlink(missing_ok=True)
        except OSError:
            continue  # cache 回收失败不能改变 catalog 的科研输入或中止训练

    # 访问时间决定跨 manifest 的 LRU；禁用 atime 的文件系统会自然退化为 mtime 次序。
    entries: list[tuple[int, int, Path]] = []  # ``(atime_ns, size_bytes, path)``
    for path in root.glob("*.pkl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            entries.append((int(stat.st_atime_ns), int(stat.st_size), path))
    total_bytes = sum(size for _atime, size, _path in entries)  # 完整 catalog pickle 总占用
    for _atime, size, path in sorted(entries):
        if total_bytes <= max_bytes:
            break
        if protected is not None and path.resolve(strict=False) == protected:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        total_bytes -= size  # 只在实际 unlink 成功后更新账面占用


def _load_cached_dataset(
    path: Path,
    *,
    source_sha256: str,
    allow_legacy_left_handedness: bool,
) -> ResolvedHandAssetDataset | None:
    r"""加载并验证本地 catalog cache；任何异常均按 cache miss 处理。"""

    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("schema") != _CATALOG_CACHE_SCHEMA:
            return None
        if payload.get("source_sha256") != source_sha256:
            return None
        if payload.get("allow_legacy_left_handedness") != allow_legacy_left_handedness:
            return None
        dataset = payload.get("dataset")
        if not isinstance(dataset, ResolvedHandAssetDataset):
            return None
        files = tuple(payload.get("files", ()))
        directories = tuple(payload.get("directories", ()))
        expected = payload.get("fingerprint")
        if not isinstance(expected, str) or _catalog_fingerprint(dataset, files=files, directories=directories) != expected:
            return None
        _prune_catalog_cache(path.parent, keep=path)
        return dataset
    except (OSError, EOFError, pickle.PickleError, TypeError, ValueError, KeyError, ImportError):
        return None


def _write_cached_dataset(
    path: Path,
    dataset: ResolvedHandAssetDataset,
    *,
    allow_legacy_left_handedness: bool,
) -> None:
    r"""原子写入 slim catalog；首次解析进程会在释放 full heap 后重新加载。

    cache miss 的完整 ``parsed.resolve()`` 含原始 sidecar/visual debug 字典。若在 full 对象仍存活
    时构造并保留 slim 对象，两类小对象会交错占据 glibc heap 页面，full 释放后仍无法 trim。
    因此本函数只负责原子写盘，让 caller 先释放整批解析 heap，再从同一 pickle 构造常驻 catalog。
    """

    slim_dataset = _slim_dataset(dataset)
    files, directories = _catalog_paths(slim_dataset)
    payload = {
        "schema": _CATALOG_CACHE_SCHEMA,
        "source_sha256": dataset.source_sha256,
        "allow_legacy_left_handedness": allow_legacy_left_handedness,
        "files": files,
        "directories": directories,
        "fingerprint": _catalog_fingerprint(dataset, files=files, directories=directories),
        "dataset": slim_dataset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary_path = Path(stream.name)
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
    _prune_catalog_cache(path.parent, keep=path)


def _slim_dataset(dataset: ResolvedHandAssetDataset) -> ResolvedHandAssetDataset:
    r"""为 SSL catalog cache 去除重复原始 ``hand_cfg`` 与 visual debug 字典。

    SSL 下游只消费 typed geometry semantics、mesh refs、路径双射和 asset identity；
    原始 sidecar 只保存在发布目录中，由 robots/tasks 继续按需读取。
    """

    def slim_partition(partition: ResolvedHandAssetPartition) -> ResolvedHandAssetPartition:
        records = tuple(
            replace(
                record,
                container=replace(
                    record.container,
                    sidecar={
                        key: record.container.sidecar[key]
                        for key in ("id", "handedness", "topology_name")
                        if key in record.container.sidecar
                    },
                    visual_rgba_by_name={},
                ),
            )
            for record in partition.records
        )
        return replace(partition, records=records)

    return replace(
        dataset,
        train=slim_partition(dataset.train),
        validation={name: slim_partition(partition) for name, partition in dataset.validation.items()},
        evaluation={name: slim_partition(partition) for name, partition in dataset.evaluation.items()},
    )


@dataclass(frozen=True)
class EmbodimentCatalog:
    r"""一份 resolved dataset 的平等资产轴与完整 partition/provenance 证据。"""

    dataset: ResolvedHandAssetDataset  # YAML identity、typed config 与全部 partition provenance

    @property
    def train(self) -> tuple[HandContainer, ...]:
        r"""返回 Trainer 唯一可用于参数更新的有序资产轴。"""

        return self.dataset.train.assets

    @property
    def validation(self) -> dict[str, tuple[HandContainer, ...]]:
        r"""返回 checkpoint selection 使用的两条具名 held-out 资产轴。

        Assets schema 2.0 保留 ``unseen_variant_set`` 与 ``unseen_mother``，训练侧
        必须先在各 suite 内聚合，再等权形成 checkpoint score，不能按资产数量扁平加权。
        """

        return {name: partition.assets for name, partition in self.dataset.validation.items()}

    @property
    def evaluation(self) -> dict[str, tuple[HandContainer, ...]]:
        r"""返回训练冻结后使用的具名 evaluation suites。"""

        return {name: partition.assets for name, partition in self.dataset.evaluation.items()}

    def training_dataset_identity(self) -> dict[str, object]:
        r"""冻结无需物化 collision source 的训练数据身份。

        完整 ``physical_geometry_hash`` 依赖 Method 对 collision union、运动学与静态采样的物化，
        属于显式 physical audit。纯训练 checkpoint 只冻结原始 dataset bytes 与有序
        ``(asset_id, content_hash)`` 轴；任何资产替换、重排或 typed semantics 内容变化都会改变摘要。

        Returns:
            dict[str, object]: schema、dataset SHA、训练资产数和有序资产轴 SHA-256。
        """

        digest = hashlib.sha256(b"anymani-training-dataset-axis-v1\0")
        for record in self.dataset.train.records:
            asset_id = str(record.container.asset_id)  # dataset 轴上的稳定资产身份
            content_hash = str(record.content_hash)  # typed geometry semantics 的内容身份
            if not asset_id or not content_hash:
                raise ValueError("training dataset identity requires non-empty asset_id and content_hash")
            for value in (asset_id, content_hash):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        return {
            "schema_version": "1.0.0",
            "source_sha256": self.dataset.source_sha256,
            "train_asset_count": len(self.dataset.train.records),
            "train_asset_axis_sha256": digest.hexdigest(),
        }


class HandAssetCatalog:
    r"""解析一份 hand asset dataset；构造阶段不读取文件系统。"""

    def __init__(self, config: HandAssetCatalogCfg) -> None:
        r"""保存 manifest identity 与安全选项。"""

        self.config = config

    def resolve(self) -> EmbodimentCatalog:
        r"""读取 manifest、展开固定 roles 并验证可选的预期 SHA-256。"""

        parsed = HandAssetDataset.from_yaml(self.config.manifest)
        parsed_source_sha256 = parsed.source_sha256  # cache identity 来自本次实际读取的 manifest bytes
        cache_path = _cache_file(
            str(parsed.source_path),
            allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
        )
        dataset = _load_cached_dataset(
            cache_path,
            source_sha256=parsed.source_sha256,
            allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
        )
        if dataset is not None:
            print(f"[SSL] Catalog cache hit: {cache_path}")
        else:
            print(f"[SSL] Catalog cache miss: resolving assets from {parsed.source_path}")
            resolved_dataset = parsed.resolve(
                require_geometry_semantics=True,
                allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
            )
            _write_cached_dataset(
                cache_path,
                resolved_dataset,
                allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
            )
            del resolved_dataset  # pickle 已原子发布；解除完整 sidecar/visual catalog 强引用
            del parsed  # 原始 manifest façade 在 resolve 后不再参与 scientific identity 或训练
            _release_catalog_allocator_slack()
            dataset = _load_cached_dataset(
                cache_path,
                source_sha256=parsed_source_sha256,
                allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
            )
            if dataset is None:
                raise RuntimeError("catalog cache could not reload the slim dataset written by this process")
        if self.config.expected_sha256 and dataset.source_sha256 != self.config.expected_sha256:
            raise ValueError(
                "hand asset dataset SHA-256 mismatch: "
                f"expected={self.config.expected_sha256}, actual={dataset.source_sha256}"
            )
        return EmbodimentCatalog(dataset)


@dataclass(frozen=True)
class HandAssetCatalogCfg:
    r"""固定消费一份 HandAssetDataset 的 data role 配置。"""

    runtime_type: ClassVar[type[HandAssetCatalog]] = HandAssetCatalog  # Hydra 不序列化 runtime 绑定
    manifest: str = ""  # 相对 AnyMani 根或绝对 dataset YAML 路径
    expected_sha256: str = ""  # 正式 recipe 可钉住原始 YAML bytes；空值只记录实际 hash
    allow_legacy_left_handedness: bool = False  # 历史审计专用，正式训练保持 false

    def __post_init__(self) -> None:
        r"""拒绝空 manifest 和格式错误的显式 SHA-256。"""

        if not self.manifest:
            raise ValueError("hand asset catalog requires one dataset manifest")
        if self.expected_sha256 and (
            len(self.expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be an empty string or one lowercase SHA-256 digest")


__all__ = ["EmbodimentCatalog", "HandAssetCatalog", "HandAssetCatalogCfg"]
