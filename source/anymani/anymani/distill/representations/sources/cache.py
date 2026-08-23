r"""CPU GeometrySource 的进程内有界复用 arena。

``GeometrySource.materialize`` 是固定资产与 ``GeometrySourceCfg`` 的确定性静态映射；缓存命中只
改变计算是否重复，不改变 $q$、query、teacher 或 objective。完整 source 含 owner 三角面、home
surface 与 anchor bank，直接 pickle 会把 8192 项资产扩张到数十 GiB，因此正式 SSL 不持久化该对象。

arena 同时施加 16 项与 512 MiB 两个上限。条目驱逐只丢弃 CPU 强引用；已经上传的
``DeviceGeometrySource`` 继续拥有自己的 source 引用，直到 resident window 释放 Warp lease。
进程正常结束、异常退出或被系统终止时，内存由进程生命周期回收，不产生 stale 临时目录。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, fields
from threading import Lock, RLock

import numpy as np
import torch

from anymani.assets.bank.hand_container import HandContainer

from .geometry_source import GeometrySource, GeometrySourceCfg

_SOURCE_ARENA_ALGORITHM = "geometry-source-materialization-v1"
_DEFAULT_MAX_ENTRIES = 16  # 两个 8-asset device subwindow，允许当前窗与下一窗交叠
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # CPU source 数组的近似 512 MiB 上限
_KEY_LOCK_STRIPES = 64  # 固定锁带使并发元数据不随访问过的资产总数增长


def _cache_key(container: HandContainer, config: GeometrySourceCfg) -> str:
    r"""生成当前进程内的 source identity，不读取或 stat 资产文件。

    ``HandContainer.geometry_semantics.content_hash`` 已覆盖运动学、limits、碰撞 payload、frame 与
    provenance。发布资产在一次 run 中保持不可变，因此无需把路径 mtime 混入运行时随机身份。
    """

    semantics = container.geometry_semantics  # assets 层已经验证的静态内容合同
    if semantics is None:
        raise ValueError("geometry source arena requires typed geometry semantics")
    identity = repr(
        (
            _SOURCE_ARENA_ALGORITHM,
            container.asset_id,
            semantics.content_hash,
            asdict(config),
        )
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()  # arena key 不进入模型输入或采样 seed


def geometry_source_array_nbytes(source: GeometrySource) -> int:
    r"""估计一项 source 的主要 tensor/mesh/NumPy payload 字节数。

    该值覆盖决定内存量级的连续数组，不声称包含 Python object header、字符串 intern 或第三方库
    私有索引。16 项 hard cap 与 512 MiB 数组 cap 共同限制实际 RSS；运行基准另行记录进程 RSS。

    Args:
        source (GeometrySource): 已完成 CPU physical materialization 的静态 source。

    Returns:
        int: 去重后的 tensor、owner mesh、home surface 与 anchor bank 数组字节数。
    """

    total = 0  # 主要连续 payload 的累计字节数
    seen: set[int] = set()  # surface/solid 共用同一数组时只计一次

    def add_array(value: object) -> None:
        r"""按对象 identity 去重累加 NumPy/Torch 连续存储。"""

        nonlocal total
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, np.ndarray):
            seen.add(identity)
            total += int(value.nbytes)
        elif isinstance(value, torch.Tensor):
            seen.add(identity)
            total += int(value.numel() * value.element_size())

    # ``spec_cpu`` 全部张量均为运动学/graph 静态真值，字符串元组相对 mesh 可忽略。
    for field_info in fields(source.spec_cpu):
        add_array(getattr(source.spec_cpu, field_info.name))

    # Trimesh 的 vertices/faces 是 source RSS 主项；surface 与 solid 可能共享或复制存储。
    for record in source.geometry_cache.records:
        for mesh in (record.surface_mesh, record.solid_mesh):
            if mesh is not None:
                add_array(mesh.vertices)
                add_array(mesh.faces)

    # retained home realization 与所有 $A^{(k)}$ bank 数组必须计入同一 source 生命周期。
    for field_info in fields(source.home_surface):
        add_array(getattr(source.home_surface, field_info.name))
    for anchors in source.anchor_bank:
        for field_info in fields(anchors):
            add_array(getattr(anchors, field_info.name))
    return total


class GeometrySourceArena:
    r"""按 source identity 复用并 LRU 驱逐 CPU physical source。

    ``load_or_create`` 对同一 key 使用固定锁带，两个 prefetch worker 不会重复物化同一资产；落入
    不同锁带的资产可并行执行 mesh/Boolean/home/anchor 构造。LRU 元数据由 ``RLock`` 保护，
    materialize 本身不占全局锁。arena 不创建目录、不序列化对象，也不消费任何随机状态。
    """

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        size_of: Callable[[GeometrySource], int] = geometry_source_array_nbytes,
    ) -> None:
        r"""构造空 arena；任何资产 IO 均延迟到第一次 ``load_or_create``。

        Args:
            max_entries (int): 同时保留的 CPU source 项数，正式默认 16。
            max_bytes (int): 主要连续数组的近似字节上限，正式默认 512 MiB。
            size_of (Callable): 测试可注入的 source 大小测量函数。
        """

        if max_entries < 1 or max_bytes < 1:
            raise ValueError("geometry source arena limits must be positive")
        self.max_entries = int(max_entries)  # hard entry cap，防止小对象无限累积
        self.max_bytes = int(max_bytes)  # mesh/tensor/array payload cap
        self.size_of = size_of  # 不通过 pickle 估算，避免缓存本身引入序列化成本
        self.hits = 0  # 当前进程内复用次数
        self.misses = 0  # 实际执行 source materialization 的次数
        self.evictions = 0  # 因容量约束丢弃的 CPU 强引用次数
        self._resident_bytes = 0  # 与 ``_entries`` 严格同步的数组估算值
        self._entries: OrderedDict[str, tuple[GeometrySource, int]] = OrderedDict()
        self._key_locks = tuple(Lock() for _ in range(_KEY_LOCK_STRIPES))  # 固定容量的同 key 去重锁带
        self._lock = RLock()  # 只保护 LRU/统计元数据，不包围昂贵物化

    @property
    def resident_count(self) -> int:
        r"""返回当前 arena 持有的 CPU source 项数。"""

        with self._lock:
            return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        r"""返回当前 source 主要连续数组的估算字节数。"""

        with self._lock:
            return self._resident_bytes

    def load_or_create(
        self,
        container: HandContainer,
        *,
        config: GeometrySourceCfg,
        materialize: Callable[[], GeometrySource],
    ) -> GeometrySource:
        r"""返回一项进程内 source，cache miss 时只物化一次。

        Args:
            container (HandContainer): 已完成 typed geometry semantics 解析的资产 bundle。
            config (GeometrySourceCfg): 固定 home-surface/anchor realization 配置。
            materialize (Callable): miss 时执行的严格物理 source 构造函数。
        """

        key = _cache_key(container, config)  # 不含线程完成顺序或全局 RNG
        with self._lock:
            resident = self._entries.get(key)
            if resident is not None:
                self._entries.move_to_end(key)  # 最近使用项位于 OrderedDict 尾部
                self.hits += 1
                return resident[0]
            stripe = int.from_bytes(hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest(), "little")
            key_lock = self._key_locks[stripe % len(self._key_locks)]

        # materialize 不持有 arena 全局锁；同 key 的并发请求在独立锁后再次检查命中。
        with key_lock:
            with self._lock:
                resident = self._entries.get(key)
                if resident is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return resident[0]
            source = materialize()  # 固定输入/seed 的 q-independent CPU physical oracle
            size_bytes = max(0, int(self.size_of(source)))  # 测量失败应由 size_of 明确抛出
            with self._lock:
                self.misses += 1
                if size_bytes > self.max_bytes:
                    return source  # 超大单项只由当前 caller 持有，不突破 arena 字节硬界
                self._entries[key] = (source, size_bytes)
                self._resident_bytes += size_bytes
                self._evict_to_limits(protected_key=key)
            return source

    def clear(self) -> None:
        r"""幂等释放全部 CPU source 强引用；固定锁带不携带资产状态。"""

        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def stats(self) -> dict[str, int]:
        r"""返回可写入 runtime evidence 的容量与命中统计。"""

        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "resident_count": len(self._entries),
                "resident_bytes": self._resident_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
            }

    def _evict_to_limits(self, *, protected_key: str) -> None:
        r"""从 LRU 端驱逐，保证刚物化项可被本次 caller 上传到 device。"""

        while len(self._entries) > self.max_entries or self._resident_bytes > self.max_bytes:
            oldest_key = next(iter(self._entries))
            if oldest_key == protected_key:
                self._entries.move_to_end(oldest_key)
                continue
            _source, size_bytes = self._entries.pop(oldest_key)
            self._resident_bytes -= size_bytes
            self.evictions += 1


__all__ = ["GeometrySourceArena", "geometry_source_array_nbytes"]
