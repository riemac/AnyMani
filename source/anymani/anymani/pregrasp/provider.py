'Read-only typed fail-closed file provider for pregrasp schema 2.'

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .cache import AtomicPregraspCache, PregraspIndexEntry, PregraspIndexError
from .schema import PregraspCoverage, PregraspLookupKey, PregraspRecord, PregraspTier, tier_satisfies


class PregraspProviderError(RuntimeError):
    'Base class for provider query failures.'


class PregraspMissError(PregraspProviderError):
    'No index entry covers this identity or requested scale.'


class PregraspInsufficientTierError(PregraspProviderError):
    'Matching record is below the requested minimum tier.'


class PregraspPointOnlyError(PregraspProviderError):
    'Query requires a basin but cache has only nominal-point evidence.'


class PregraspCorruptError(PregraspProviderError):
    'Index, payload, schema, digest, or lookup binding is corrupt.'


@dataclass(frozen=True)
class PregraspQuery:
    'Complete provider query with no implicit tier downgrade or nearest-scale fallback.'

    lookup_key: PregraspLookupKey  # Current runtime physical/cube/physics/search identity.
    requested_scale: float  # Actual absolute cube scale in this scene.
    min_tier: PregraspTier = PregraspTier.CONTACT_BASIN  # Training requires at least contact by default.
    require_basin: bool = True  # Reject point-only records by default.

    def __post_init__(self) -> None:
        'Validate a finite positive scale and normalize tier enum.'

        if not math.isfinite(self.requested_scale) or self.requested_scale <= 0.0:
            raise ValueError("requested_scale must be finite and positive")
        object.__setattr__(self, "min_tier", PregraspTier(self.min_tier))


@dataclass(frozen=True)
class PregraspResolution:
    'Strict record returned by provider and its index provenance.'

    record: PregraspRecord  # Complete q/T_ho/metrics/tier/certificate.
    index_entry: PregraspIndexEntry  # Provider selection reason and payload identity.


class FilePregraspProvider:
    (
        'Read-only provider that validates index and payload on every query. Never '
        'cache failures, fall back to q-home, or guess by nearest scale/asset row. A '
        'verified-record memo may be added only if an index-digest change invalidates '
        'it completely.'
    )

    def __init__(self, root: Path | str) -> None:
        'Bind a cache root; validate the current index on every resolve.'

        self.cache = AtomicPregraspCache(root)  # Reuse path/index validation; callers cannot publish.

    def resolve(self, query: PregraspQuery) -> PregraspResolution:
        (
            'Resolve by exact identity and closed scale interval, requiring the requested '
            'tier/coverage. Raise typed miss, insufficient-tier, point-only, or corrupt '
            'errors on failure.'
        )

        try:
            index = self.cache.load_index()
        except PregraspIndexError as exc:
            raise PregraspCorruptError(str(exc)) from exc
        lookup_digest = query.lookup_key.digest  # Exclude asset_id; physical identity must match exactly.
        candidates = [
            entry
            for entry in index.entries
            if entry.lookup_digest == lookup_digest and entry.scale_min <= query.requested_scale <= entry.scale_max
        ]
        if not candidates:
            raise PregraspMissError(
                f"no pregrasp covers lookup={lookup_digest} scale={query.requested_scale:.8g}"
            )
        if len(candidates) != 1:
            raise PregraspCorruptError("pregrasp index resolved multiple overlapping entries")
        entry = candidates[0]
        if not tier_satisfies(entry.tier, query.min_tier):
            raise PregraspInsufficientTierError(
                f"pregrasp tier {entry.tier.value} is below required {query.min_tier.value}"
            )
        if query.require_basin and entry.coverage != PregraspCoverage.BASIN:
            raise PregraspPointOnlyError("pregrasp query requires basin coverage but index contains point only")
        payload_path = self.cache.payload_path(entry)
        if not payload_path.is_file():
            raise PregraspCorruptError(f"pregrasp index references missing payload {entry.payload_relpath}")
        try:
            document = json.loads(payload_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("payload root is not a JSON object")
            record = PregraspRecord.from_dict(document)
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            raise PregraspCorruptError(f"cannot validate pregrasp payload: {exc}") from exc
        if record.digest != entry.record_digest:
            raise PregraspCorruptError("pregrasp payload digest disagrees with index")
        if record.lookup_key.digest != lookup_digest:
            raise PregraspCorruptError("pregrasp payload lookup identity disagrees with query/index")
        certificate = record.scale_certificate
        if entry.coverage == PregraspCoverage.BASIN and (
            certificate is None or not certificate.contains(query.requested_scale)
        ):
            raise PregraspCorruptError("pregrasp payload certificate does not cover requested scale")
        return PregraspResolution(record=record, index_entry=entry)


__all__ = [
    "FilePregraspProvider",
    "PregraspCorruptError",
    "PregraspInsufficientTierError",
    "PregraspMissError",
    "PregraspPointOnlyError",
    "PregraspProviderError",
    "PregraspQuery",
    "PregraspResolution",
]
