(
    'Palm-supported good-pregrasp Top-K catalog with exact runtime lookup. Each '
    'hand/object/exact-scale key stores eight coupled reset candidates (q0, u0, '
    'T_ho, active mask). Joint state/target use canonical 16-slot radians; MVP '
    'requires u0=q0 and upright object pose with wxyz quaternion. Contact values '
    'are quality metadata only, not a tier or admission gate. One exact physical '
    'key maps to one immutable Top-8 payload. Publish payload and index '
    'atomically on one filesystem; resolve rechecks both key and payload digest. '
    'SHA-256 identifies content but does not rank candidates.'
)

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GOOD_PREGRASP_SCHEMA_VERSION = "3.0.0"
'Good-pregrasp catalog version, separate from the older contact-tier schema.'

GOOD_PREGRASP_ENTRY_TYPE = "anymani.good_pregrasp.entry"
GOOD_PREGRASP_INDEX_TYPE = "anymani.good_pregrasp.index"
GOOD_PREGRASP_TOP_K = 8
CANONICAL_JOINT_COUNT = 16
CANONICAL_OWNER_COUNT = 21
UPRIGHT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GoodPregraspCatalogError(RuntimeError):
    'IO, conflict, or content-integrity error for a good-pregrasp catalog.'


class GoodPregraspMissError(GoodPregraspCatalogError):
    'No published Top-8 entry matches this exact hand/object/scale key.'


class GoodPregraspConflictError(GoodPregraspCatalogError):
    'Attempt to publish a different Top-8 payload for the same exact key.'


def _sha256(value: str, field_name: str) -> str:
    'Validate a lowercase 64-character SHA-256 identity.'

    parsed = str(value)
    if _SHA256.fullmatch(parsed) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return parsed


def _finite_tuple(values: Sequence[float], *, length: int, name: str) -> tuple[float, ...]:
    'Normalize a fixed-width finite float sequence.'

    parsed = tuple(float(value) for value in values)
    if len(parsed) != length or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must contain {length} finite values")
    return parsed


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    'Create stable, field-sorted JSON bytes with no NaN.'

    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, Any]) -> str:
    'Return the persistent document SHA-256.'

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class GoodPregraspKey:
    'Exact hand/object/scale query coordinate.'

    asset_id: str  # Human-readable asset label in formal selection.
    source_content_hash: str  # Source bundle content identity.
    physical_geometry_hash: str  # Canonical active-physical-mapping identity.
    canonical_schema_digest: str  # Canonical storage/routing schema identity.
    routing_digest: str  # Active-joint-mask identity.
    object_asset_id: str  # Currently DexCube.
    object_asset_sha256: str  # Identity of actual USD bytes.
    object_scale: float  # Exact dimensionless scale; MVP uses 1.1.
    physics_identity_digest: str  # Generated-physics identity: dt/material/solver/mass, etc.
    generation_identity_digest: str  # Proposal/settle/cold-reset protocol identity.

    def __post_init__(self) -> None:
        'Reject empty labels, invalid hashes, and non-positive scale.'

        if not self.asset_id.strip() or not self.object_asset_id.strip():
            raise ValueError("good-pregrasp asset/object IDs must be non-empty")
        for field_name in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "routing_digest",
            "object_asset_sha256",
            "physics_identity_digest",
            "generation_identity_digest",
        ):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        if not math.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("good-pregrasp object_scale must be finite and positive")

    @property
    def digest(self) -> str:
        'Return the content identity of the exact query key.'

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        'Return a JSON-safe key.'

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspKey:
        'Restore and revalidate a key from a persistent mapping.'

        return cls(**dict(payload))


@dataclass(frozen=True)
class GoodPregraspCandidate:
    'Initial hand/object state ready for training reset.'

    q_state_rad: tuple[float, ...]  # Canonical actual joint state [16], rad.
    q_target_rad: tuple[float, ...]  # Canonical PD target [16]; MVP requires target=q0.
    active_joint_mask: tuple[bool, ...]  # Canonical active subspace [16].
    object_position_h_m: tuple[float, float, float]  # Object origin in hand frame, meters.
    object_orientation_h_wxyz: tuple[float, float, float, float] = UPRIGHT_QUATERNION_WXYZ

    def __post_init__(self) -> None:
        'Validate q0=u0, zero ghosts, and strict upright initial orientation.'

        q_state = _finite_tuple(self.q_state_rad, length=CANONICAL_JOINT_COUNT, name="q_state_rad")
        q_target = _finite_tuple(self.q_target_rad, length=CANONICAL_JOINT_COUNT, name="q_target_rad")
        mask = tuple(bool(value) for value in self.active_joint_mask)
        if len(mask) != CANONICAL_JOINT_COUNT or not any(mask):
            raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
        if q_state != q_target:
            raise ValueError("good-pregrasp MVP requires q_target_rad to equal q_state_rad exactly")
        if any(value != 0.0 for value, active in zip(q_state, mask, strict=True) if not active):
            raise ValueError("inactive canonical joint states/targets must be exactly zero")
        position = _finite_tuple(self.object_position_h_m, length=3, name="object_position_h_m")
        orientation = _finite_tuple(
            self.object_orientation_h_wxyz,
            length=4,
            name="object_orientation_h_wxyz",
        )
        if orientation != UPRIGHT_QUATERNION_WXYZ:
            raise ValueError("good-pregrasp MVP object orientation must be exact hand-frame upright")
        object.__setattr__(self, "q_state_rad", q_state)
        object.__setattr__(self, "q_target_rad", q_target)
        object.__setattr__(self, "active_joint_mask", mask)
        object.__setattr__(self, "object_position_h_m", position)
        object.__setattr__(self, "object_orientation_h_wxyz", UPRIGHT_QUATERNION_WXYZ)

    def to_dict(self) -> dict[str, Any]:
        'Return a JSON-safe candidate.'

        return {
            "q_state_rad": list(self.q_state_rad),
            "q_target_rad": list(self.q_target_rad),
            "active_joint_mask": list(self.active_joint_mask),
            "object_position_h_m": list(self.object_position_h_m),
            "object_orientation_h_wxyz": list(self.object_orientation_h_wxyz),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspCandidate:
        'Restore a candidate from JSON-safe mapping.'

        return cls(
            q_state_rad=tuple(payload["q_state_rad"]),
            q_target_rad=tuple(payload["q_target_rad"]),
            active_joint_mask=tuple(payload["active_joint_mask"]),
            object_position_h_m=tuple(payload["object_position_h_m"]),
            object_orientation_h_wxyz=tuple(payload["object_orientation_h_wxyz"]),
        )


@dataclass(frozen=True)
class GoodPregraspMetrics:
    'Compact sufficient statistics for geometry and 1-second cold-reset acceptance.'

    joint_limit_margin_fraction: float  # Minimum normalized margin from active joints to limits.
    envelope_fingers: tuple[str, str, str]  # Thumb plus two non-thumb roles.
    envelope_sector_min_deg: float  # Minimum in-plane sector separation for three fingers, degrees.
    envelope_tip_center_distance_m: tuple[float, float, float]  # Distance from each of three TIPs to object center, meters.
    penetration_depth_max_m: float  # Maximum illegal penetration at initialization/replay, meters.
    object_displacement_max_m: float  # Maximum displacement from initial state over 1 s, meters.
    object_tilt_max_deg: float  # Maximum angle between object z and hand z over 1 s, degrees.
    peak_linear_velocity_m_s: float  # Peak linear speed during first 0.2 s of cold reset, m/s.
    peak_off_axis_angular_velocity_rad_s: float  # Peak non-target-axis angular speed during first 0.2 s, rad/s.
    palm_contact_fraction: float  # PALM support fraction during final 0.5 s.
    owner_contact_fraction: tuple[float, ...]  # Contact fraction for PALM+JOINT16+TIP4, shape [21].
    peak_angular_velocity_rad_s: float | None = None  # Optional peak total angular speed; strict v5 requires it, rad/s.

    def __post_init__(self) -> None:
        'Validate unit intervals, three-finger envelope, and finite physical statistics.'

        if len(self.envelope_fingers) != 3 or self.envelope_fingers[0] != "thumb":
            raise ValueError("envelope_fingers must be thumb followed by two non-thumb roles")
        if len(set(self.envelope_fingers)) != 3 or any(
            finger not in {"thumb", "index", "middle", "ring"} for finger in self.envelope_fingers
        ):
            raise ValueError("envelope_fingers must contain three distinct canonical finger roles")
        distances = _finite_tuple(
            self.envelope_tip_center_distance_m,
            length=3,
            name="envelope_tip_center_distance_m",
        )
        owner_contact = _finite_tuple(
            self.owner_contact_fraction,
            length=CANONICAL_OWNER_COUNT,
            name="owner_contact_fraction",
        )
        scalars = (
            self.joint_limit_margin_fraction,
            self.envelope_sector_min_deg,
            self.penetration_depth_max_m,
            self.object_displacement_max_m,
            self.object_tilt_max_deg,
            self.peak_linear_velocity_m_s,
            self.peak_off_axis_angular_velocity_rad_s,
            self.palm_contact_fraction,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("good-pregrasp metrics must be finite and non-negative")
        if not 0.0 <= self.joint_limit_margin_fraction <= 0.5:
            raise ValueError("joint_limit_margin_fraction must lie in [0,0.5]")
        if not 0.0 <= self.palm_contact_fraction <= 1.0 or any(not 0.0 <= value <= 1.0 for value in owner_contact):
            raise ValueError("contact fractions must lie in [0,1]")
        if any(value < 0.0 for value in distances):
            raise ValueError("envelope distances must be non-negative")
        if self.peak_angular_velocity_rad_s is not None and (
            not math.isfinite(self.peak_angular_velocity_rad_s) or self.peak_angular_velocity_rad_s < 0.0
        ):
            raise ValueError("peak total angular velocity must be finite and non-negative when provided")
        object.__setattr__(self, "envelope_tip_center_distance_m", distances)
        object.__setattr__(self, "owner_contact_fraction", owner_contact)

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe physical metrics.'

        payload = asdict(self)
        payload["envelope_fingers"] = list(self.envelope_fingers)
        payload["envelope_tip_center_distance_m"] = list(self.envelope_tip_center_distance_m)
        payload["owner_contact_fraction"] = list(self.owner_contact_fraction)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspMetrics:
        'Restore metrics from a persistent mapping.'

        values = dict(payload)
        values["envelope_fingers"] = tuple(values["envelope_fingers"])
        values["envelope_tip_center_distance_m"] = tuple(values["envelope_tip_center_distance_m"])
        values["owner_contact_fraction"] = tuple(values["owner_contact_fraction"])
        return cls(**values)


@dataclass(frozen=True)
class GoodPregraspMember:
    'One ordered Top-8 candidate with complete acceptance metrics.'

    rank: int  # MVP runtime consumes rank 0.
    candidate: GoodPregraspCandidate
    metrics: GoodPregraspMetrics
    selection_score: tuple[float, ...]  # Lexicographic quality vector from the generator.

    def __post_init__(self) -> None:
        'Validate rank and finite, non-empty selection score.'

        if self.rank < 0:
            raise ValueError("good-pregrasp rank must be non-negative")
        score = tuple(float(value) for value in self.selection_score)
        if not score or not all(math.isfinite(value) for value in score):
            raise ValueError("selection_score must be a non-empty finite tuple")
        object.__setattr__(self, "selection_score", score)

    def to_dict(self) -> dict[str, Any]:
        'Return a JSON-safe ranked member.'

        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "selection_score": list(self.selection_score),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspMember:
        'Restore a ranked member from persistent mapping.'

        return cls(
            rank=int(payload["rank"]),
            candidate=GoodPregraspCandidate.from_dict(payload["candidate"]),
            metrics=GoodPregraspMetrics.from_dict(payload["metrics"]),
            selection_score=tuple(payload["selection_score"]),
        )


@dataclass(frozen=True)
class GoodPregraspEntry:
    'Complete Top-8 reset set for one exact hand/object/scale key.'

    key: GoodPregraspKey
    members: tuple[GoodPregraspMember, ...]

    def __post_init__(self) -> None:
        'Require exactly 8 members, contiguous ranks, and unique candidates.'

        if len(self.members) != GOOD_PREGRASP_TOP_K:
            raise ValueError(f"published good-pregrasp entry requires exactly {GOOD_PREGRASP_TOP_K} members")
        if tuple(member.rank for member in self.members) != tuple(range(GOOD_PREGRASP_TOP_K)):
            raise ValueError("good-pregrasp member ranks must be contiguous 0..7")
        candidate_digests = [_digest(member.candidate.to_dict()) for member in self.members]
        if len(set(candidate_digests)) != GOOD_PREGRASP_TOP_K:
            raise ValueError("good-pregrasp Top-8 candidates must be unique")

    @property
    def digest(self) -> str:
        'Return entry payload content identity.'

        return _digest(self.to_dict())

    @property
    def primary(self) -> GoodPregraspMember:
        'Return rank-0 member, the only rank consumed by MVP.'

        return self.members[0]

    def to_dict(self) -> dict[str, Any]:
        'Return schema-3 JSON-safe entry.'

        return {
            "artifact_type": GOOD_PREGRASP_ENTRY_TYPE,
            "schema_version": GOOD_PREGRASP_SCHEMA_VERSION,
            "key": self.key.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspEntry:
        'Restore and validate entry from persistent document.'

        if payload.get("artifact_type") != GOOD_PREGRASP_ENTRY_TYPE:
            raise ValueError("unexpected good-pregrasp entry artifact_type")
        if payload.get("schema_version") != GOOD_PREGRASP_SCHEMA_VERSION:
            raise ValueError("unsupported good-pregrasp schema_version")
        return cls(
            key=GoodPregraspKey.from_dict(payload["key"]),
            members=tuple(GoodPregraspMember.from_dict(member) for member in payload["members"]),
        )


@dataclass(frozen=True)
class GoodPregraspIndexEntry:
    'Map exact index keys to content-addressed payloads.'

    key_digest: str
    entry_digest: str
    payload_relpath: str

    def __post_init__(self) -> None:
        _sha256(self.key_digest, "key_digest")
        _sha256(self.entry_digest, "entry_digest")
        expected = f"records/{self.entry_digest}.json"
        if self.payload_relpath != expected:
            raise ValueError(f"good-pregrasp payload path must be {expected!r}")


class GoodPregraspCatalog:
    'Atomic schema-3 Top-8 catalog publisher and fail-closed resolver.'

    def __init__(self, root: str | Path) -> None:
        'Bind catalog root; create directories only during publish.'

        self.root = Path(root).expanduser()
        self.index_path = self.root / "index.json"
        self.records_dir = self.root / "records"
        self.lock_path = self.root / ".publish.lock"  # Cross-process index read-modify-write critical section.

    def _load_index(self) -> tuple[GoodPregraspIndexEntry, ...]:
        'Read and validate ordered index; absent index means empty catalog.'

        if not self.index_path.is_file():
            return ()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GoodPregraspCatalogError(f"cannot read good-pregrasp index: {error}") from error
        if payload.get("artifact_type") != GOOD_PREGRASP_INDEX_TYPE:
            raise GoodPregraspCatalogError("unexpected good-pregrasp index artifact_type")
        if payload.get("schema_version") != GOOD_PREGRASP_SCHEMA_VERSION:
            raise GoodPregraspCatalogError("unsupported good-pregrasp index schema_version")
        try:
            entries = tuple(GoodPregraspIndexEntry(**entry) for entry in payload["entries"])
        except (KeyError, TypeError, ValueError) as error:
            raise GoodPregraspCatalogError(f"invalid good-pregrasp index entry: {error}") from error
        if tuple(entry.key_digest for entry in entries) != tuple(sorted(entry.key_digest for entry in entries)):
            raise GoodPregraspCatalogError("good-pregrasp index entries must be sorted by key_digest")
        if len({entry.key_digest for entry in entries}) != len(entries):
            raise GoodPregraspCatalogError("good-pregrasp index contains duplicate exact keys")
        return entries

    @staticmethod
    def _index_document(entries: Sequence[GoodPregraspIndexEntry]) -> dict[str, Any]:
        'Build a compact, stably ordered index document.'

        return {
            "artifact_type": GOOD_PREGRASP_INDEX_TYPE,
            "schema_version": GOOD_PREGRASP_SCHEMA_VERSION,
            "entries": [asdict(entry) for entry in sorted(entries, key=lambda item: item.key_digest)],
        }

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        (
            'Write durably beside the target, then atomically replace; fsync file and '
            'directory. Same-directory temp avoids cross-filesystem rename, random names '
            'isolate publishers, file fsync persists content before rename, and directory '
            'fsync persists the rename.'
        )

        path.parent.mkdir(parents=True, exist_ok=True)  # Temporary file and target must share a filesystem.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )  # Unique temp path prevents publishers overwriting uncommitted bytes.
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)  # Canonical JSON bytes; payload/index both end with newline.
                handle.flush()
                os.fsync(handle.fileno())  # Persist data before exposing index or payload filename.
            os.replace(temporary, path)  # Atomic same-filesystem switch; never expose a partial file.
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)  # Persist the renamed directory entry.
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)  # After replace the temporary path is gone; on error remove any orphan.

    @contextmanager
    def _publication_lock(self) -> Iterator[None]:
        (
            'Serialize cross-process publishers with a POSIX advisory lock. Resolvers '
            'need no lock: payload precedes atomic index replace, so readers see a '
            'complete old or new version. The lock protects publisher '
            'load/validate/replace transactions.'
        )

        self.root.mkdir(parents=True, exist_ok=True)  # Lock file belongs to catalog root, not scientific identity.
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # Wait for other processes to finish the entire index transaction.
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # Release publisher lock on success or exception.

    def publish(self, entry: GoodPregraspEntry) -> GoodPregraspIndexEntry:
        'Idempotently publish one Top-8 entry; conflict on a different payload for the same key.'

        return self.publish_many((entry,))[0]

    def publish_many(self, entries: Sequence[GoodPregraspEntry]) -> tuple[GoodPregraspIndexEntry, ...]:
        (
            'Batch-publish Top-8 entries with one index commit, so resolvers see either '
            'the full old set or full new set. Validate all keys/conflicts first, write '
            'content-addressed payloads, then atomically replace the index once. '
            'Interrupted payload writes may leave harmless unreferenced files, never '
            'partial cohort membership.'
        )

        requested = tuple(entries)  # Freeze caller sequence so generator state cannot change inside lock.
        if not requested:
            return ()

        # Allow each exact key once per input; reject duplicates before locking to avoid ambiguous order/overwrite.
        requested_key_digests = tuple(entry.key.digest for entry in requested)
        if len(set(requested_key_digests)) != len(requested_key_digests):
            raise GoodPregraspConflictError("batch publication contains duplicate exact keys")

        with self._publication_lock():
            current = list(self._load_index())  # Read under lock so publishers cannot commit from the same stale index.
            current_by_key = {item.key_digest: item for item in current}
            output: list[GoodPregraspIndexEntry] = []  # Preserve requested order exactly.
            pending_payloads: list[tuple[GoodPregraspIndexEntry, bytes]] = []

            # Compute all content digests and check conflicts before visible writes.
            for entry in requested:
                key_digest = entry.key.digest  # SHA-256 identity of exact hand/object/scale/protocol.
                payload_bytes = _canonical_bytes(entry.to_dict())
                entry_digest = hashlib.sha256(payload_bytes).hexdigest()  # Content identity of the complete Top-8.
                index_entry = GoodPregraspIndexEntry(
                    key_digest=key_digest,
                    entry_digest=entry_digest,
                    payload_relpath=f"records/{entry_digest}.json",
                )
                existing = current_by_key.get(key_digest)
                if existing is not None:
                    if existing.entry_digest != entry_digest:
                        raise GoodPregraspConflictError("exact good-pregrasp key already maps to another Top-8 payload")
                    output.append(existing)  # Same key/same payload is idempotent; do not rewrite files/index.
                    continue
                output.append(index_entry)
                pending_payloads.append((index_entry, payload_bytes))
                current.append(index_entry)  # Publish all new keys in one final index.
                current_by_key[key_digest] = index_entry

            # Write payloads before index so every newly visible reference resolves immediately.
            for index_entry, payload_bytes in pending_payloads:
                self._atomic_write(self.root / index_entry.payload_relpath, payload_bytes + b"\n")
            if pending_payloads:
                # Old checkpoints refer to the published directory version; preserve it for equivalence checks of used entries.
                if self.index_path.is_file():
                    previous_bytes = self.index_path.read_bytes()
                    previous_digest = hashlib.sha256(previous_bytes).hexdigest()
                    self._atomic_write(self.root / "index_history" / f"{previous_digest}.json", previous_bytes)
                index_bytes = _canonical_bytes(self._index_document(current)) + b"\n"
                self._atomic_write(self.index_path, index_bytes)  # Single visibility/commit point for the batch.
            return tuple(output)

    def resolve(self, key: GoodPregraspKey) -> GoodPregraspEntry:
        'Read Top-8 by exact key and recheck payload digest and embedded key.'

        return self.resolve_many((key,))[0]

    def _read_index_entry(self, match: GoodPregraspIndexEntry) -> GoodPregraspEntry:
        'Restore an index reference and validate complete content plus embedded exact key.'

        path = self.root / match.payload_relpath  # Index validation restricts paths to content-addressed records directory.
        try:
            payload = json.loads(path.read_bytes())
            entry = GoodPregraspEntry.from_dict(payload)  # Schema validates Top-8 completeness, rank, and candidate states.
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise GoodPregraspCatalogError(f"cannot restore good-pregrasp payload: {error}") from error
        if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != match.entry_digest:
            raise GoodPregraspCatalogError("good-pregrasp payload content digest mismatch")
        if entry.key.digest != match.key_digest:
            raise GoodPregraspCatalogError("good-pregrasp payload embedded key mismatch")
        return entry

    def read_entries(self) -> tuple[GoodPregraspEntry, ...]:
        (
            'Read the index once and restore a full catalog snapshot for copying/role '
            'snapshots. Read each payload once and reuse resolver schema/content/key '
            'checks. Preserve index key order; scanning does not generate candidates or '
            'change physics/generation identity.'
        )

        return tuple(self._read_index_entry(match) for match in self._load_index())

    def resolve_many(self, keys: Sequence[GoodPregraspKey]) -> tuple[GoodPregraspEntry, ...]:
        (
            'Read index once and resolve many exact keys in caller order. A ManagerBased '
            'full reset may resolve 80 assets; share the immutable index read, but still '
            'validate each payload digest and embedded key.'
        )

        requested = tuple(keys)
        if not requested:
            return ()
        index_by_digest = {entry.key_digest: entry for entry in self._load_index()}  # Read index once.
        resolved_by_digest: dict[str, GoodPregraspEntry] = {}
        output: list[GoodPregraspEntry] = []
        for key in requested:
            match = index_by_digest.get(key.digest)
            if match is None:
                raise GoodPregraspMissError(
                    f"no good-pregrasp entry for asset={key.asset_id} object={key.object_asset_id} scale={key.object_scale}"
                )
            entry = resolved_by_digest.get(match.entry_digest)
            if entry is None:
                entry = self._read_index_entry(match)  # Restore repeated identical payloads once.
                resolved_by_digest[match.entry_digest] = entry
            if entry.key != key:
                raise GoodPregraspCatalogError("good-pregrasp payload embedded key mismatch")
            output.append(entry)
        return tuple(output)


__all__ = [
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_OWNER_COUNT",
    "GOOD_PREGRASP_SCHEMA_VERSION",
    "GOOD_PREGRASP_TOP_K",
    "UPRIGHT_QUATERNION_WXYZ",
    "GoodPregraspCandidate",
    "GoodPregraspCatalog",
    "GoodPregraspCatalogError",
    "GoodPregraspConflictError",
    "GoodPregraspEntry",
    "GoodPregraspKey",
    "GoodPregraspMember",
    "GoodPregraspMetrics",
    "GoodPregraspMissError",
]
