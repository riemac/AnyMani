r"""显式机械设计修订：扩大两关节 Allegro 非拇指的屈曲范围。

删去远端 PIP/DIP 后，Allegro 短指只剩 MCP1 侧摆与 MCP2 屈曲；继续采用
完整手的 MCP2 上限可能限制其向掌内推动的可达范围。本模块实现确定性、
可审计的设计变体，不推断学习困难的原因，也不覆盖既有 generated bundle。

默认上限 2.23 rad（127.77°）取自本项目 LEAP 近掌屈曲 physical preset；
Allegro 母体原上限是 1.61 rad（92.25°）。这是用户授权的机械设计候选，
不是 Allegro 硬件规格更正。只改上限，不改下限、力矩、速度、几何或 DOF。
新 bundle 的导出、身份冻结、碰撞及初态验证由调用方分别完成。
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace

from .asset_schema_core import JointLimitCfg
from .asset_schema_embodiment import HandCfg


@dataclass(frozen=True)
class JointUpperLimitRevision:
    """一次上限修改的最小物理差分；所有角度单位为 rad。"""

    finger_name: str  # semantic finger slot，不能从资产目录名反推
    joint_name: str  # 现有活动关节名称，不新增关节或重排 canonical slots
    old_upper_rad: float  # 原始 bundle 的真实上限，包含既有 limit_tweak
    new_upper_rad: float  # 候选设计上限，用于独立 provenance 与验证


def widen_two_joint_allegro_flexion(
    hand: HandCfg, *, upper_rad: float = 2.23
) -> tuple[HandCfg, tuple[JointUpperLimitRevision, ...]]:
    r"""复制整手，仅扩大显式 Allegro 两关节非拇指 MCP2 的上限。

    选择条件同时要求 non-thumb、finger family 为 Allegro、恰有两个 revolute
    joints、其名称与 child anatomy 为 j0/MCP1 和 j1/MCP2。仅以 DOF=2 判断
    会把另一种删关节拓扑误当作本次设计，因此歧义 anatomy 直接拒绝。

    更新采用 $q_{max,new}=\max(q_{max,old},q_{requested})$，不会缩窄已更大的
    活动范围。fixed TIP 不计入活动关节数。输入对象及其全部 metadata 保持。

    Args:
        hand: 从规范 sidecar 恢复的 HandCfg；finger family 使用生成器显式映射。
        upper_rad: 有限正弧度上限，默认 2.23 rad；不是增量角度。

    Returns:
        新 HandCfg 与实际修改的关节差分；无匹配项时仍返回独立副本及空元组。
    """

    if not math.isfinite(upper_rad) or upper_rad <= 0.0:
        raise ValueError("requested upper limit must be finite and positive (rad)")
    revised = deepcopy(hand)  # 原资产作为对照保留；禁止原地改写及共享 nested limits
    edits: list[JointUpperLimitRevision] = []  # 只登记实际变化，不把 no-op 伪装成新设计

    # family_composition 与 palm family 不能代替每根手指的真实来源。
    # 新旧 generated metadata 可能同时保留 topology/connectivity，两者必须一致。
    maps = [
        entry["slot_family_map"]
        for key in ("premade_topology", "premade_connectivity")
        if isinstance((entry := hand.metadata.get(key)), dict) and "slot_family_map" in entry
    ]  # generator truth 中的显式 slot→family 映射
    if not maps:
        raise ValueError("joint-limit revision requires explicit generated slot_family_map")
    if any(mapping != maps[0] for mapping in maps[1:]):
        raise ValueError("generated slot_family_map declarations disagree")
    family_by_slot = maps[0]  # 通过一致性检查后才允许作为物理 selector

    for finger in revised.fingers:
        if finger.name == "thumb":
            continue  # 用户明确要求拇指不参与本次短指修订
        if finger.name not in family_by_slot:
            raise ValueError(f"missing explicit finger family for slot {finger.name!r}")
        if str(family_by_slot[finger.name]).lower() != "allegro":
            continue  # 即使 palm 来自 Allegro，也不能改变 LEAP 指
        moving = [joint for joint in finger.joints if joint.joint_type == "revolute"]
        if len(moving) != 2:
            continue  # 完整指与三关节指保持逐字段不变；fixed TIP 不进入 DOF 计数
        first, flexion = moving  # 链顺序来自规范 FingerCfg，而非 XML 展示顺序
        if (first.name, first.child, flexion.name, flexion.child) != (
            f"{finger.name}_j0",
            f"{finger.name}_mcp1",
            f"{finger.name}_j1",
            f"{finger.name}_mcp2",
        ):
            raise ValueError(f"two-joint finger {finger.name!r} must explicitly retain MCP1 and MCP2")
        limit = flexion.limit  # JointCfg 已负责把合法 sidecar limits 恢复为 typed schema
        if not isinstance(limit, JointLimitCfg):
            raise ValueError(f"flexion joint {flexion.name!r} lacks a typed finite limit")
        if upper_rad <= limit.upper:
            continue  # 只允许扩展：已有更大上限不被默认设计值压低
        edits.append(JointUpperLimitRevision(finger.name, flexion.name, limit.upper, float(upper_rad)))
        flexion.limit = replace(limit, upper=float(upper_rad))  # lower/effort/velocity 原值完整保留
    return revised, tuple(edits)  # bundle identity 与 provenance 由 producer 使用差分生成


__all__ = ["JointUpperLimitRevision", "widen_two_joint_allegro_flexion"]
