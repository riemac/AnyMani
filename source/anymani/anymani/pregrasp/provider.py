r"""Pregrasp schema-2的只读、typed fail-closed file provider。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .cache import AtomicPregraspCache, PregraspIndexEntry, PregraspIndexError
from .schema import PregraspCoverage, PregraspLookupKey, PregraspRecord, PregraspTier, tier_satisfies


class PregraspProviderError(RuntimeError):
    r"""所有provider查询失败的共同基类。"""


class PregraspMissError(PregraspProviderError):
    r"""Identity或requested scale没有任何index覆盖。"""


class PregraspInsufficientTierError(PregraspProviderError):
    r"""命中record的tier低于query minimum tier。"""


class PregraspPointOnlyError(PregraspProviderError):
    r"""Query要求basin但cache只包含nominal point证据。"""


class PregraspCorruptError(PregraspProviderError):
    r"""Index/payload/schema/content digest或lookup binding损坏。"""


@dataclass(frozen=True)
class PregraspQuery:
    r"""一次完整provider查询，不允许隐式tier降级或scale近邻。"""

    lookup_key: PregraspLookupKey  # 当前runtime physical/cube/physics/search identity
    requested_scale: float  # 当前scene的实际absolute cube scale
    min_tier: PregraspTier = PregraspTier.CONTACT_BASIN  # 当前训练默认至少contact
    require_basin: bool = True  # 默认拒绝point-only记录

    def __post_init__(self) -> None:
        r"""验证finite positive scale并规约tier enum。"""

        if not math.isfinite(self.requested_scale) or self.requested_scale <= 0.0:
            raise ValueError("requested_scale must be finite and positive")
        object.__setattr__(self, "min_tier", PregraspTier(self.min_tier))


@dataclass(frozen=True)
class PregraspResolution:
    r"""Provider命中的严格record与index provenance。"""

    record: PregraspRecord  # 完整q/T_ho/metrics/tier/certificate
    index_entry: PregraspIndexEntry  # provider选择依据与payload identity


class FilePregraspProvider:
    r"""每次查询都验证index与payload的只读file provider。

    Provider不缓存失败、不回退q-home，也不按最近scale或asset row猜测。长期训练若需要host cache，可在不改变
    本接口语义的前提下增加已验证record memoization；index digest变化时必须整体失效。
    """

    def __init__(self, root: Path | str) -> None:
        r"""绑定一个cache root；实际index在每次resolve时验证。"""

        self.cache = AtomicPregraspCache(root)  # 复用同一path/index校验，不开放publish给调用方

    def resolve(self, query: PregraspQuery) -> PregraspResolution:
        r"""按exact identity与闭scale interval返回满足tier/coverage的record。

        Raises:
            PregraspMissError: identity或scale没有覆盖。
            PregraspInsufficientTierError: 命中tier低于minimum tier。
            PregraspPointOnlyError: query要求basin但只有point。
            PregraspCorruptError: index或payload不能严格验证。
        """

        try:
            index = self.cache.load_index()
        except PregraspIndexError as exc:
            raise PregraspCorruptError(str(exc)) from exc
        lookup_digest = query.lookup_key.digest  # asset_id被排除，physical identity必须exact
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
