"""Pregrasp schema-2原子cache与冲突合同。"""

from __future__ import annotations

import pytest

from anymani.pregrasp.cache import AtomicPregraspCache, PregraspConflictError
from anymani.pregrasp.schema import PregraspCoverage

from .test_schema import _candidate, _certificate, _record


def test_publish_is_idempotent_and_index_is_commit_marker(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    record = _record()
    first = cache.publish(record)
    second = cache.publish(record)
    assert second == first
    assert cache.index_path.is_file()
    assert cache.payload_path(first).is_file()
    index = cache.load_index()
    assert index.entries == (first,)


def test_same_lookup_overlapping_interval_with_different_payload_conflicts(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    cache.publish(_record())
    conflicting = _record(candidate=_candidate(q0=0.2), certificate=_certificate(scale_min=1.18, scale_max=1.24))
    with pytest.raises(PregraspConflictError, match="overlap"):
        cache.publish(conflicting)


def test_same_lookup_non_overlapping_scale_intervals_can_coexist(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    first = _record(candidate=_candidate(scale=1.1), certificate=_certificate(anchor="1.1", scale_min=1.10, scale_max=1.15))
    second_certificate = _certificate(anchor="1.25", scale_min=1.20, scale_max=1.25)
    second = _record(candidate=_candidate(scale=1.25, q0=0.2), certificate=second_certificate)
    cache.publish(first)
    cache.publish(second)
    assert len(cache.load_index().entries) == 2


def test_point_record_can_be_stored_but_is_explicitly_marked(tmp_path) -> None:
    cache = AtomicPregraspCache(tmp_path)
    entry = cache.publish(_record(coverage=PregraspCoverage.POINT))
    assert entry.coverage == PregraspCoverage.POINT
    assert entry.scale_min == entry.scale_max == 1.2
