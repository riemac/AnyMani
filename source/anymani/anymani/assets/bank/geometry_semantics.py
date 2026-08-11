r"""HandBank 对版本化整手几何语义的按需解析与 legacy 迁移。

该模块是静态 schema 与资产集合交付之间的适配层：

- 新 generated/official sidecar：严格解析 ``geometry_semantics`` 并复核内容哈希；
- 旧 generated sidecar：从完整 ``hand_cfg`` 确定性派生，不改写原目录；
- official sidecar 缺字段：严格拒绝，不使用 generated 拓扑启发式。

``tasks`` 可不启用本路径；``distill`` 通过 ``HandBankCfg.require_geometry_semantics=True`` 明确要求。

该适配层不写回旧目录。迁移结果只存在当前 ``HandContainer`` 和后续 cache materialization 中，
并携带 schema/migration/content hash；这样读取旧 generated asset 不会产生大量文件改动，也不会
把一次迁移误认为 exporter 已经重新产出了新 sidecar。未来 exporter 写入新字段后，bank 直接读取
并通过哈希复核，避免同一资产同时存在“sidecar 语义”和“运行时猜测语义”两套真源。

`source_kind` 是安全边界而非 family 标签。generated 表示可由 generator truth 恢复；official
表示必须有人工核验的 owner、collision coverage、frame calibration 和 kinematic chain。LEAP/Allegro
即使包含完整 hand_cfg，也不能因为结构看起来像 generated 就绕过 official fail-closed。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from ..asset_schema_geometry import (
    HandGeometrySemanticsCfg,
    derive_generated_geometry_semantics,
    geometry_semantics_from_dict,
)
from ..asset_sidecar import restore_hand_cfg_snapshot

GEOMETRY_SEMANTICS_SIDECAR_KEY = "geometry_semantics"
"""``hand.yaml`` 中版本化静态几何语义的固定顶层键。"""

HandAssetSourceKind: TypeAlias = Literal["generated", "official"]
"""决定 legacy 迁移权限的资产来源类型。"""


def resolve_hand_geometry_semantics(
    sidecar: Mapping[str, Any],
    *,
    source_kind: HandAssetSourceKind,
    asset_id: str,
    topology_key: str | None = None,
) -> HandGeometrySemanticsCfg:
    r"""按来源策略解析一个 container 的类型化几何语义。

    Args:
        sidecar (Mapping[str, Any]): 已由 ``HandContainer`` 读取的完整 ``hand.yaml``。
        source_kind (HandAssetSourceKind): generated 允许旧 sidecar 迁移；official 只接受人工字段。
        asset_id (str): 当前 container 的稳定 ID，用于拒绝 sidecar 串包。
        topology_key (str | None): 调用方显式提供的 morphology 身份；缺失时读取 sidecar
            ``topology_name``，再退回 ``None``。

    Returns:
        HandGeometrySemanticsCfg: 下游可直接交给 ``robots`` lower 的静态资产事实。

    Raises:
        ValueError: 新字段与 container 身份不一致、official 缺字段或 legacy hand_cfg 缺失时抛出。
    """

    raw_semantics = sidecar.get(GEOMETRY_SEMANTICS_SIDECAR_KEY)
    if raw_semantics is not None:
        if not isinstance(raw_semantics, Mapping):
            raise TypeError(f"{GEOMETRY_SEMANTICS_SIDECAR_KEY} must be a mapping")
        semantics = geometry_semantics_from_dict(raw_semantics)
        if semantics.asset_id != asset_id:
            raise ValueError(
                f"geometry semantics asset_id={semantics.asset_id!r} does not match container asset_id={asset_id!r}"
            )
        if semantics.source_kind != source_kind:
            raise ValueError(
                f"geometry semantics source_kind={semantics.source_kind!r} does not match "
                f"container source_kind={source_kind!r}"
            )
        return semantics

    if source_kind == "official":
        raise ValueError(
            "official hand assets require an explicit, manually verified geometry_semantics sidecar field"
        )

    hand_cfg_raw = sidecar.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(
            "legacy generated hand sidecar is missing top-level 'hand_cfg'; "
            "geometry semantics cannot be migrated"
        )
    hand = restore_hand_cfg_snapshot(hand_cfg_raw)
    resolved_topology_key = topology_key
    if resolved_topology_key is None and sidecar.get("topology_name") is not None:
        resolved_topology_key = str(sidecar["topology_name"])
    return derive_generated_geometry_semantics(
        hand,
        asset_id=asset_id,
        topology_key=resolved_topology_key,
    )


__all__ = [
    "GEOMETRY_SEMANTICS_SIDECAR_KEY",
    "HandAssetSourceKind",
    "resolve_hand_geometry_semantics",
]
