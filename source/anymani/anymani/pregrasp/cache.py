r"""Pregrasp schema-2的原子content-addressed cache与index。

Payload按record digest写入``records/<sha256>.json``，index是完整cache的commit marker。Provider只信任
index引用的payload；进程在payload发布后、index提交前崩溃产生的孤儿文件不会进入查询语义。同一lookup
domain的闭scale intervals禁止重叠，避免runtime在多个candidate之间做未声明的“最近anchor”选择。
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    PREGRASP_INDEX_ARTIFACT_TYPE,
    PREGRASP_SCHEMA_VERSION,
    PregraspCoverage,
    PregraspRecord,
    PregraspTier,
    canonical_json_bytes,
    stable_digest,
)


class PregraspCacheError(RuntimeError):
    r"""所有cache storage/index错误的共同基类。"""


class PregraspConflictError(PregraspCacheError):
    r"""同一lookup domain出现重叠scale或同key不同payload。"""


class PregraspIndexError(PregraspCacheError):
    r"""Index schema、digest、path或payload引用损坏。"""


@dataclass(frozen=True)
class PregraspIndexEntry:
    r"""Index中一个不可变record引用与查询充分字段。"""

    lookup_digest: str  # 不含scale interval的物理查询SHA-256
    record_digest: str  # 完整record payload SHA-256
    payload_relpath: str  # cache root内POSIX相对路径
    tier: PregraspTier  # 中心candidate达到的最高接触等级
    coverage: PregraspCoverage  # point或basin
    anchor: str  # canonical scale anchor字符串
    scale_min: float  # 闭区间下界
    scale_max: float  # 闭区间上界

    def __post_init__(self) -> None:
        r"""验证digest、相对路径和闭区间。"""

        for field_name in ("lookup_digest", "record_digest"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise PregraspIndexError(f"{field_name} must be a lowercase SHA-256 digest")
        path = Path(self.payload_relpath)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("records",):
            raise PregraspIndexError("payload_relpath must stay inside the cache records directory")
        if not (self.scale_min > 0.0 and self.scale_min <= self.scale_max):
            raise PregraspIndexError("index scale interval must be positive and ordered")
        object.__setattr__(self, "tier", PregraspTier(self.tier))
        object.__setattr__(self, "coverage", PregraspCoverage(self.coverage))

    @classmethod
    def from_record(cls, record: PregraspRecord) -> PregraspIndexEntry:
        r"""从严格record提取查询充分字段。"""

        certificate = record.scale_certificate
        scale_min = certificate.scale_min if certificate is not None else record.candidate.object_scale
        scale_max = certificate.scale_max if certificate is not None else record.candidate.object_scale
        anchor = certificate.anchor if certificate is not None else _canonical_anchor(record.candidate.object_scale)
        return cls(
            lookup_digest=record.lookup_key.digest,
            record_digest=record.digest,
            payload_relpath=f"records/{record.digest}.json",
            tier=record.tier,
            coverage=record.coverage,
            anchor=anchor,
            scale_min=scale_min,
            scale_max=scale_max,
        )

    def overlaps(self, other: PregraspIndexEntry) -> bool:
        r"""判断同一lookup domain的两个闭区间是否有交集。"""

        return self.lookup_digest == other.lookup_digest and max(self.scale_min, other.scale_min) <= min(
            self.scale_max, other.scale_max
        )  # 闭区间共享端点也会产生runtime歧义

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe index entry。"""

        return {
            "lookup_digest": self.lookup_digest,
            "record_digest": self.record_digest,
            "payload_relpath": self.payload_relpath,
            "tier": self.tier.value,
            "coverage": self.coverage.value,
            "anchor": self.anchor,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PregraspIndexEntry:
        r"""从index document恢复entry。"""

        return cls(
            lookup_digest=str(payload["lookup_digest"]),
            record_digest=str(payload["record_digest"]),
            payload_relpath=str(payload["payload_relpath"]),
            tier=PregraspTier(str(payload["tier"])),
            coverage=PregraspCoverage(str(payload["coverage"])),
            anchor=str(payload["anchor"]),
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
        )


def _canonical_anchor(scale: float) -> str:
    r"""把point scale规约成稳定十进制字符串，不用于basin interval推断。"""

    text = format(float(scale), ".12g")  # 去掉二进制浮点尾噪声，同时保留非常规probe scale
    return text


@dataclass(frozen=True)
class PregraspIndex:
    r"""完整cache index；其digest覆盖所有有序entries。"""

    entries: tuple[PregraspIndexEntry, ...] = ()  # 按lookup/scale/record稳定排序

    def __post_init__(self) -> None:
        r"""排序并拒绝重复record与重叠查询区间。"""

        entries = tuple(
            sorted(
                self.entries,
                key=lambda entry: (entry.lookup_digest, entry.scale_min, entry.scale_max, entry.record_digest),
            )
        )
        if len({entry.record_digest for entry in entries}) != len(entries):
            raise PregraspIndexError("pregrasp index contains duplicate record digests")
        for index, left in enumerate(entries):
            for right in entries[index + 1 :]:
                if right.lookup_digest != left.lookup_digest:
                    break  # 稳定排序后下一lookup domain不可能再与left冲突
                if left.overlaps(right):
                    raise PregraspConflictError("pregrasp index contains overlapping scale intervals")
        object.__setattr__(self, "entries", entries)

    def payload_dict(self) -> dict[str, Any]:
        r"""返回不含self digest的index payload。"""

        return {
            "artifact_type": PREGRASP_INDEX_ARTIFACT_TYPE,
            "schema_version": PREGRASP_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        r"""返回完整index content digest。"""

        return stable_digest(self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        r"""返回带index digest的完整document。"""

        return {**self.payload_dict(), "index_digest": self.digest}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PregraspIndex:
        r"""严格恢复index并复核digest。"""

        if payload.get("artifact_type") != PREGRASP_INDEX_ARTIFACT_TYPE:
            raise PregraspIndexError("unsupported pregrasp index artifact_type")
        if payload.get("schema_version") != PREGRASP_SCHEMA_VERSION:
            raise PregraspIndexError("unsupported pregrasp index schema_version")
        index = cls(entries=tuple(PregraspIndexEntry.from_dict(item) for item in payload.get("entries", ())))
        if payload.get("index_digest") != index.digest:
            raise PregraspIndexError("pregrasp index digest mismatch")
        return index


class AtomicPregraspCache:
    r"""用文件锁、content-addressed payload和原子index发布维护cache。

    Cache只保证单机多进程writer的一致性。Payload先发布、index后提交；若进程在两步之间终止，孤儿payload
    被provider忽略。Index发布使用同目录temporary file、flush、fsync与``os.replace``，父目录随后fsync。
    """

    def __init__(self, root: Path | str) -> None:
        r"""初始化cache路径；不在构造时信任或自动修复既有index。"""

        self.root = Path(root).expanduser().resolve()  # cache identity使用绝对root，禁止cwd漂移
        self.records_dir = self.root / "records"  # immutable content-addressed payloads
        self.index_path = self.root / "index.json"  # provider唯一commit marker
        self.lock_path = self.root / ".lock"  # Linux advisory writer lock
        self.records_dir.mkdir(parents=True, exist_ok=True)  # payload目录可包含未提交孤儿文件

    def payload_path(self, entry: PregraspIndexEntry) -> Path:
        r"""解析并验证一个entry的payload绝对路径。"""

        path = (self.root / entry.payload_relpath).resolve()
        if self.root not in path.parents or path.parent != self.records_dir:
            raise PregraspIndexError("pregrasp payload path escapes cache records directory")
        return path

    def load_index(self) -> PregraspIndex:
        r"""读取完整index；cache尚未发布时返回空index。"""

        if not self.index_path.exists():
            return PregraspIndex()  # 空cache是合法状态，provider查询会typed miss
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PregraspIndexError(f"cannot read pregrasp index: {exc}") from exc
        if not isinstance(payload, dict):
            raise PregraspIndexError("pregrasp index root must be a JSON object")
        return PregraspIndex.from_dict(payload)

    def publish(self, record: PregraspRecord) -> PregraspIndexEntry:
        r"""原子发布一个严格record并返回其index entry。

        同一record重复发布幂等；同lookup domain的另一payload若scale interval重叠则抛
        :class:`PregraspConflictError`，不做last-writer-wins。
        """

        validated = PregraspRecord.from_dict(record.to_dict())  # writer边界重新执行所有科研不变量
        entry = PregraspIndexEntry.from_record(validated)  # index只保存查询充分字段
        payload = canonical_json_bytes(validated.to_dict()) + b"\n"  # 人类文本末尾换行不进入record digest
        self.root.mkdir(parents=True, exist_ok=True)  # lock与index父目录在首次发布时创建
        with self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)  # index read-modify-write临界区
            index = self.load_index()
            for existing in index.entries:
                if existing.record_digest == entry.record_digest:
                    return existing  # 相同content重复发布不触碰index
                if existing.overlaps(entry):
                    raise PregraspConflictError("overlapping scale interval has a different pregrasp payload")
            payload_path = self.payload_path(entry)
            if payload_path.exists():
                if payload_path.read_bytes() != payload:
                    raise PregraspConflictError("content-addressed pregrasp payload bytes disagree with digest path")
            else:
                self._atomic_write(payload_path, payload)  # payload必须先于index durable
            updated = PregraspIndex(entries=(*index.entries, entry))
            self._atomic_write(self.index_path, canonical_json_bytes(updated.to_dict()) + b"\n")
            return entry

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        r"""在同目录flush/fsync后以``os.replace``原子发布文件。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")  # 同filesystem保证replace原子性
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)  # 一次写入canonical bytes
                stream.flush()  # Python buffer提交给kernel
                os.fsync(stream.fileno())  # payload/index bytes durable
            os.replace(temporary, path)  # 原子切换可见版本
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)  # directory entry durable
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)  # 失败后不保留本进程temporary文件


__all__ = [
    "AtomicPregraspCache",
    "PregraspCacheError",
    "PregraspConflictError",
    "PregraspIndex",
    "PregraspIndexEntry",
    "PregraspIndexError",
]
