r"""资产 sidecar 的共享编解码辅助。

``generator``、``exporter`` 与 ``bank`` 都需要恢复 ``hand.yaml.hand_cfg``，因此该逻辑位于资产
根层而不归属于任一流水线。它只把 dataclass 快照恢复为规范 ``HandCfg``；路径发现、资产选择、
几何语义迁移和文件写入仍由各自模块负责。

恢复顺序是“递归补最小 geometry type -> HandCfg schema 实例化 -> 下游几何语义校验”。补 type
只依据互斥字段：mesh path、elliptic radius_x/radius_z/length、box size、cylinder radius+length
或 sphere radius。它不能推断 owner、TIP 或运动学 frame；这些必须来自显式 generator truth/sidecar。
因此本编解码器可以被 generator runtime 和 bank 共用，却不会把旧 sidecar 的启发式扩大成跨官方
资产的语义规则。
"""

from __future__ import annotations

from typing import Any

from .asset_schema_embodiment import HandCfg


def restore_hand_cfg_snapshot(hand_cfg_raw: dict[str, Any]) -> HandCfg:
    r"""把 sidecar 中的 ``hand_cfg`` 快照恢复成真正的 ``HandCfg``。

    ``AssetCfgBase.to_dict()`` 会递归展开 dataclass，但几何子类的 ``geometry_type`` 是
    ``ClassVar``，旧 sidecar 不一定保存 ``type``。本函数只按互斥的几何参数签名补齐该字段，
    随后立即交回规范 schema 验证，不修改尺寸、位姿或碰撞列表。

    Args:
        hand_cfg_raw (dict[str, Any]): ``hand.yaml.hand_cfg`` 的原生映射。

    Returns:
        HandCfg: 已完成全部 schema 校验的规范手资产描述。
    """

    if not isinstance(hand_cfg_raw, dict):
        raise TypeError(f"'hand_cfg' must be a mapping, got {type(hand_cfg_raw).__name__}")
    normalized = _rehydrate_geometry_mappings(hand_cfg_raw)  # 先递归补 geometry type，再正式实例化
    return HandCfg(**normalized)


def _rehydrate_geometry_mappings(value: Any) -> Any:
    r"""递归补全旧 sidecar 中缺失的 ``geometry.type``。"""

    if isinstance(value, list):
        return [_rehydrate_geometry_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_geometry_mappings(item) for item in value)
    if not isinstance(value, dict):
        return value

    normalized = {key: _rehydrate_geometry_mappings(item) for key, item in value.items()}  # 先递归到底层节点
    if "geometry" in normalized and isinstance(normalized["geometry"], dict):
        normalized["geometry"] = _inject_geometry_type(normalized["geometry"])  # 最常见的缺类型位置
    return _inject_geometry_type(normalized)


def _inject_geometry_type(geometry_doc: dict[str, Any]) -> dict[str, Any]:
    r"""按互斥参数签名为旧几何映射补上类型分发字段。"""

    if "type" in geometry_doc or "kind" in geometry_doc:
        return geometry_doc  # 新 sidecar 已显式声明时不得篡改

    normalized = dict(geometry_doc)
    if any(key in normalized for key in ("file_path", "path", "mesh")):
        normalized["type"] = "mesh"
    elif {"radius_x", "radius_z", "length"} <= normalized.keys():
        normalized["type"] = "elliptic_cylinder"
    elif "size" in normalized:
        normalized["type"] = "box"
    elif "radius" in normalized and "length" in normalized:
        normalized["type"] = "cylinder"
    elif "radius" in normalized:
        normalized["type"] = "sphere"
    return normalized


__all__ = ["restore_hand_cfg_snapshot"]
