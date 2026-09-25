(
    'Atomic content-addressed cache and index for pregrasp schema 2. Store '
    'payloads as records/<sha256>.json; the index is the cache commit marker, and '
    'providers trust only payloads it references. Crashes before index commit '
    'leave ignored orphans. Closed scale intervals within one lookup domain may '
    'not overlap, avoiding an undeclared nearest-anchor choice.'
)

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
    'Base class for cache storage/index errors.'


class PregraspConflictError(PregraspCacheError):
    'Overlapping scale intervals or different payloads for the same lookup domain.'


class PregraspIndexError(PregraspCacheError):
    'Corrupt index schema, digest, path, or payload reference.'


@dataclass(frozen=True)
class PregraspIndexEntry:
    'Immutable record reference and query-sufficient fields from the index.'

    lookup_digest: str  # SHA-256 of the physical query, excluding scale interval.
    record_digest: str  # SHA-256 of the complete record payload.
    payload_relpath: str  # POSIX-relative path under the cache root.
    tier: PregraspTier  # Highest contact tier reached by the center candidate.
    coverage: PregraspCoverage  # Coverage protocol: point or basin.
    anchor: str  # Canonical scale-anchor string.
    scale_min: float  # Closed interval lower bound.
    scale_max: float  # Closed interval upper bound.

    def __post_init__(self) -> None:
        'Validate digest, relative path, and closed interval.'

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
        'Extract query-sufficient fields from a strict record.'

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
        'Check whether two closed intervals in the same lookup domain intersect.'

        return self.lookup_digest == other.lookup_digest and max(self.scale_min, other.scale_min) <= min(
            self.scale_max, other.scale_max
        )  # Shared endpoints of closed intervals are ambiguous at runtime.

    def to_dict(self) -> dict[str, Any]:
        'Return a JSON-safe index entry.'

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
        'Restore an entry from an index document.'

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
    'Normalize a point scale to a stable decimal string; do not infer basin intervals from it.'

    text = format(float(scale), ".12g")  # Remove binary-float tail noise while preserving unusual probe scales.
    return text


@dataclass(frozen=True)
class PregraspIndex:
    'Complete cache index; digest covers every ordered entry.'

    entries: tuple[PregraspIndexEntry, ...] = ()  # Stable sort by lookup, scale, and record.

    def __post_init__(self) -> None:
        'Sort entries and reject duplicate records or overlapping query intervals.'

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
                    break  # Stable ordering prevents a later lookup domain from colliding with this one.
                if left.overlaps(right):
                    raise PregraspConflictError("pregrasp index contains overlapping scale intervals")
        object.__setattr__(self, "entries", entries)

    def payload_dict(self) -> dict[str, Any]:
        'Return index payload without its own digest.'

        return {
            "artifact_type": PREGRASP_INDEX_ARTIFACT_TYPE,
            "schema_version": PREGRASP_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        'Return full index content digest.'

        return stable_digest(self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        'Return complete document with index digest.'

        return {**self.payload_dict(), "index_digest": self.digest}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PregraspIndex:
        'Restore index strictly and recheck its digest.'

        if payload.get("artifact_type") != PREGRASP_INDEX_ARTIFACT_TYPE:
            raise PregraspIndexError("unsupported pregrasp index artifact_type")
        if payload.get("schema_version") != PREGRASP_SCHEMA_VERSION:
            raise PregraspIndexError("unsupported pregrasp index schema_version")
        index = cls(entries=tuple(PregraspIndexEntry.from_dict(item) for item in payload.get("entries", ())))
        if payload.get("index_digest") != index.digest:
            raise PregraspIndexError("pregrasp index digest mismatch")
        return index


class AtomicPregraspCache:
    (
        'Maintain the cache with file locks, content-addressed payloads, and atomic '
        'index publication. This guarantees consistency for multi-process writers on '
        'one machine. Publish payload before index; providers ignore orphan payloads '
        'after interruption. Publish the index via same-directory temporary file, '
        'flush, fsync, os.replace, then parent-directory fsync.'
    )

    def __init__(self, root: Path | str) -> None:
        'Initialize cache paths without trusting or repairing an existing index.'

        self.root = Path(root).expanduser().resolve()  # Use an absolute cache root so cwd changes cannot redirect it.
        self.records_dir = self.root / "records"  # immutable content-addressed payloads
        self.index_path = self.root / "index.json"  # Provider-visible commit marker.
        self.lock_path = self.root / ".lock"  # Linux advisory writer lock
        self.records_dir.mkdir(parents=True, exist_ok=True)  # Payload directory may contain uncommitted orphan files.

    def payload_path(self, entry: PregraspIndexEntry) -> Path:
        "Resolve and validate an entry's absolute payload path."

        path = (self.root / entry.payload_relpath).resolve()
        if self.root not in path.parents or path.parent != self.records_dir:
            raise PregraspIndexError("pregrasp payload path escapes cache records directory")
        return path

    def load_index(self) -> PregraspIndex:
        'Read the complete index; return empty when nothing has been published.'

        if not self.index_path.exists():
            return PregraspIndex()  # An empty cache is valid; provider lookup returns a typed miss.
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PregraspIndexError(f"cannot read pregrasp index: {exc}") from exc
        if not isinstance(payload, dict):
            raise PregraspIndexError("pregrasp index root must be a JSON object")
        return PregraspIndex.from_dict(payload)

    def publish(self, record: PregraspRecord) -> PregraspIndexEntry:
        (
            'Atomically publish one strict record and return its index entry. Identical '
            'records are idempotent. A different payload with an overlapping interval for '
            'the same lookup domain raises PregraspConflictError; never use '
            'last-writer-wins.'
        )

        validated = PregraspRecord.from_dict(record.to_dict())  # Recheck all scientific invariants at the writer boundary.
        entry = PregraspIndexEntry.from_record(validated)  # Index stores only query-sufficient fields.
        payload = canonical_json_bytes(validated.to_dict()) + b"\n"  # Trailing newline in human-readable text is excluded from record digest.
        self.root.mkdir(parents=True, exist_ok=True)  # Create lock and index parent directories on first publication.
        with self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)  # Index read-modify-write critical section.
            index = self.load_index()
            for existing in index.entries:
                if existing.record_digest == entry.record_digest:
                    return existing  # Repeated identical content does not touch the index.
                if existing.overlaps(entry):
                    raise PregraspConflictError("overlapping scale interval has a different pregrasp payload")
            payload_path = self.payload_path(entry)
            if payload_path.exists():
                if payload_path.read_bytes() != payload:
                    raise PregraspConflictError("content-addressed pregrasp payload bytes disagree with digest path")
            else:
                self._atomic_write(payload_path, payload)  # Persist payload before index.
            updated = PregraspIndex(entries=(*index.entries, entry))
            self._atomic_write(self.index_path, canonical_json_bytes(updated.to_dict()) + b"\n")
            return entry

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        'Flush/fsync and atomically replace the file from a same-directory temporary.'

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")  # Atomic replace requires the same filesystem.
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)  # Write canonical bytes once.
                stream.flush()  # Commit the Python buffer to the kernel.
                os.fsync(stream.fileno())  # payload/index bytes durable
            os.replace(temporary, path)  # Atomically switch the visible version.
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)  # directory entry durable
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)  # Remove this process's temporary file after failure.


__all__ = [
    "AtomicPregraspCache",
    "PregraspCacheError",
    "PregraspConflictError",
    "PregraspIndex",
    "PregraspIndexEntry",
    "PregraspIndexError",
]
