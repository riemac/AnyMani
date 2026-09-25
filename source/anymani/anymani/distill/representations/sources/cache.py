"""Bounded geometry source storage and memory accounting."""


from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, fields
from threading import Lock, RLock

import numpy as np
import torch

from anymani.assets.bank.hand_container import HandContainer

from .geometry_source import GeometrySource, GeometrySourceCfg, GeometrySourceCore

_SOURCE_ARENA_ALGORITHM = "geometry-source-materialization-v1"
_DEFAULT_MAX_ENTRIES = 16
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_KEY_LOCK_STRIPES = 64


def _cache_key(container: HandContainer, config: GeometrySourceCfg) -> str:


    semantics = container.geometry_semantics
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
    return hashlib.sha256(identity).hexdigest()


def geometry_source_array_nbytes(source: GeometrySource | GeometrySourceCore) -> int:


    total = 0
    seen: set[int] = set()

    def add_array(value: object) -> None:


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


    for field_info in fields(source.spec_cpu):
        add_array(getattr(source.spec_cpu, field_info.name))


    for record in source.geometry_cache.records:
        for mesh in (record.surface_mesh, record.solid_mesh):
            if mesh is not None:
                add_array(mesh.vertices)
                add_array(mesh.faces)


    for field_info in fields(source.home_surface):
        add_array(getattr(source.home_surface, field_info.name))
    sampling_arrays = source.surface_sampling_arrays
    if sampling_arrays is not None:
        for name in (
            "vertices_owner_local_m",
            "faces",
            "face_normals_owner_local",
            "face_area_cdf",
        ):
            for value in getattr(sampling_arrays, name):
                add_array(value)
    warp_views = source.warp_surface_views
    if warp_views is not None:
        for view in warp_views:
            for name in ("vertices", "faces", "source_face_indices", "face_altitudes_m"):
                add_array(getattr(view, name))
    if isinstance(source, GeometrySource):
        for anchors in source.anchor_bank:
            for field_info in fields(anchors):
                add_array(getattr(anchors, field_info.name))
    return total


class GeometrySourceArena:


    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        size_of: Callable[[GeometrySource | GeometrySourceCore], int] = geometry_source_array_nbytes,
    ) -> None:


        if max_entries < 1 or max_bytes < 1:
            raise ValueError("geometry source arena limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)  # mesh/tensor/array payload cap
        self.size_of = size_of
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._resident_bytes = 0
        self._entries: OrderedDict[str, tuple[GeometrySource | GeometrySourceCore, int]] = OrderedDict()
        self._key_locks = tuple(Lock() for _ in range(_KEY_LOCK_STRIPES))
        self._lock = RLock()

    @property
    def resident_count(self) -> int:


        with self._lock:
            return len(self._entries)

    @property
    def resident_bytes(self) -> int:


        with self._lock:
            return self._resident_bytes

    def load_or_create(
        self,
        container: HandContainer,
        *,
        config: GeometrySourceCfg,
        materialize: Callable[[], GeometrySource | GeometrySourceCore],
    ) -> GeometrySource | GeometrySourceCore:


        key = _cache_key(container, config)
        with self._lock:
            resident = self._entries.get(key)
            if resident is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return resident[0]
            stripe = int.from_bytes(hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest(), "little")
            key_lock = self._key_locks[stripe % len(self._key_locks)]


        with key_lock:
            with self._lock:
                resident = self._entries.get(key)
                if resident is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return resident[0]
            source = materialize()
            size_bytes = max(0, int(self.size_of(source)))
            with self._lock:
                self.misses += 1
                if size_bytes > self.max_bytes:
                    return source
                self._entries[key] = (source, size_bytes)
                self._resident_bytes += size_bytes
                self._evict_to_limits(protected_key=key)
            return source

    def clear(self) -> None:


        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def stats(self) -> dict[str, int]:


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


        while len(self._entries) > self.max_entries or self._resident_bytes > self.max_bytes:
            oldest_key = next(iter(self._entries))
            if oldest_key == protected_key:
                self._entries.move_to_end(oldest_key)
                continue
            _source, size_bytes = self._entries.pop(oldest_key)
            self._resident_bytes -= size_bytes
            self.evictions += 1


__all__ = ["GeometrySourceArena", "geometry_source_array_nbytes"]
