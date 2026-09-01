"""Pregrasp schema-2 provider的typed fail-closed查询合同。"""

from __future__ import annotations

import json

import pytest

from anymani.pregrasp.cache import AtomicPregraspCache
from anymani.pregrasp.provider import (
    FilePregraspProvider,
    PregraspCorruptError,
    PregraspInsufficientTierError,
    PregraspMissError,
    PregraspPointOnlyError,
    PregraspQuery,
)
from anymani.pregrasp.schema import PregraspCoverage, PregraspTier

from .test_schema import _metrics, _record


def test_provider_resolves_closed_interval_boundaries(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    record = _record()
    cache.publish(record)
    provider = FilePregraspProvider(tmp_path)
    for scale in (1.15, 1.2, 1.22):
        resolved = provider.resolve(
            PregraspQuery(
                lookup_key=record.lookup_key,
                requested_scale=scale,
                min_tier=PregraspTier.CONTACT_BASIN,
                require_basin=True,
            )
        )
        assert resolved.record == record


def test_provider_miss_and_tier_failure_never_fallback(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    support = _record(metrics=_metrics(tip_ge_2_fraction=0.0, tip_active_count_mean=0.0))
    cache.publish(support)
    provider = FilePregraspProvider(tmp_path)
    with pytest.raises(PregraspMissError):
        provider.resolve(PregraspQuery(support.lookup_key, 1.24, PregraspTier.SUPPORT_BASIN, True))
    with pytest.raises(PregraspInsufficientTierError):
        provider.resolve(PregraspQuery(support.lookup_key, 1.2, PregraspTier.CONTACT_BASIN, True))


def test_provider_rejects_point_when_basin_is_required(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    point = _record(coverage=PregraspCoverage.POINT)
    cache.publish(point)
    provider = FilePregraspProvider(tmp_path)
    with pytest.raises(PregraspPointOnlyError):
        provider.resolve(PregraspQuery(point.lookup_key, 1.2, PregraspTier.CONTACT_BASIN, True))


def test_provider_detects_payload_corruption(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    entry = cache.publish(_record())
    payload = cache.payload_path(entry)
    document = json.loads(payload.read_text())
    document["candidate"]["q_target_rad"][0] = 0.9
    payload.write_text(json.dumps(document))
    provider = FilePregraspProvider(tmp_path)
    with pytest.raises(PregraspCorruptError):
        provider.resolve(PregraspQuery(_record().lookup_key, 1.2, PregraspTier.CONTACT_BASIN, True))
