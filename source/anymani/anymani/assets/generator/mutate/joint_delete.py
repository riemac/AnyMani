r"""关节删除工具：从 finger 运动学链中裁剪 joint，并做合理重连。

这是前后序里拓扑改变最大的操作，也是"从已有手派生新手"的核心路径之一，属于重合区，两个阶段都能调用。
在图 `资产生产概略.png` 中对应 `pre-made` 的 `joint delete + regroup` 分支；
在图 `前后序.png` 中明确归属后序，理由是"拓扑裁剪，在 HandCfg 上操作"。

分类说明
--------

- **结构裁剪**：从连续 finger 链中删除 1 个或多个 joint，并保持链的语义连续性
- **Regroup**：裁剪后把相邻 link 的 collision / visual 几何做合并或保留策略

设计说明
--------

### 职责边界

`JointDeleteMutator` 只接受已有的 `HandCfg`，不负责新建骨架。它的输出必须
仍是合法的 `HandCfg`（能通过 validator 的全局检查）。

### Regroup 策略

删除一个 joint 之后，其 parent link 和 child link 之间不再有关节相隔，
需要把 child link 的几何（collision / visual）并入 parent link，并更新 origin。
当前提供三种策略：

- ``"merge"``：把 child 几何写入 parent，origin 复合叠加
- ``"drop"``：直接丢弃被删 joint 的 child 几何（适合只关心运动学）
- ``"keep"``：保留 child 几何挂在新父 link 下，不做合并（最保守）

### 与 preset 的关系

每个 finger preset 应当能声明"最小关节数"或"不可删关节列表"。
`JointDeleteMutator` 在执行前先对照该约束做过滤，超出约束的删除请求会被拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import CollisionGeometryCfg, InertialCfg, PoseCfg, VisualGeometryCfg
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class JointDeleteCfg(AssetCfgBase):
    r"""关节删除工具配置。

    这个配置只描述"删哪些 joint、怎么重连、允许删到什么程度"，不描述
    具体链式重连实现。
    """

    class_type: type["JointDeleteMutator"] | None = None
    """关联的运行时类。"""

    target_finger: str | None = None
    """目标手指名称；若为 ``None``，则由上层策略（如流水线随机选择）决定。"""

    deleted_joints: tuple[str, ...] = ()
    """显式指定要删除的关节名称集合；顺序通常从近端到远端。空元组表示由运行时自动选取。"""

    regroup_strategy: Literal["merge", "drop", "keep"] = "merge"
    """删除后的几何重组策略。``merge`` 把子 link 几何并入父 link，``drop`` 直接丢弃，
    ``keep`` 保留子几何挂在新父 link 下。"""

    respect_preset: bool = True
    """是否遵守 finger preset 的保留关节约束（最小关节数、不可删列表）。默认开启。"""

    keep_terminal_joint: bool = True
    """是否默认保留末端关节的语义（即 ``is_tip=True`` 的 joint 不得删除）。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointDeleteMutator


# ============================================================================
#  运行时壳
# ============================================================================


class JointDeleteMutator(MutatorBase):
    r"""关节删除运行时壳。

    负责对已构建好的 `HandCfg` 执行 joint 删除 + 重连，并按 `regroup_strategy`
    处理被删关节的 child link 几何。
    """

    cfg: JointDeleteCfg

    def __init__(self, cfg: JointDeleteCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对一个已构建的 `HandCfg` 执行关节删除 + 链式重连。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 变异后的整手配置；若删除请求违反约束则返回 ``None``。
        """

        if self.cfg.regroup_strategy == "keep":
            raise NotImplementedError(
                "regroup_strategy='keep' requires orphan sub-link support, which current schema does not provide."
            )

        mutated = target.copy()  # joint delete 会重写 finger 链，因此必须在深拷贝上操作
        if not mutated.fingers:
            return None

        if self.cfg.target_finger is None:
            finger_index = random.randrange(len(mutated.fingers))  # 未显式指定时才随机选 finger
        else:
            finger_index = next((index for index, finger in enumerate(mutated.fingers) if finger.name == self.cfg.target_finger), -1)
            if finger_index < 0:
                return None

        finger = mutated.fingers[finger_index]
        deletable = [
            joint.name
            for joint in finger.joints
            if not (self.cfg.keep_terminal_joint and joint.is_tip)
        ]
        if not deletable:
            return None

        requested = list(self.cfg.deleted_joints) or [random.choice(deletable)]  # 空列表语义：运行时随机删一段
        delete_set = {name for name in requested if name in deletable}
        if not delete_set:
            return None

        # `respect_preset=True` 当前只能先落实到“至少保留一个 revolute joint”这条最小保护。
        # 这是因为更细的 preset-level 不可删列表还未沉到 metadata / schema。
        remaining_revolute = sum(
            1 for joint in finger.joints if joint.name not in delete_set and joint.joint_type == "revolute"
        )
        if self.cfg.respect_preset and remaining_revolute < 1:
            return None

        rebuilt = self._delete_from_finger(mutated, finger, delete_set)
        if rebuilt is None:
            return None

        mutated.fingers[finger_index] = rebuilt
        try:
            return mutated.replace(fingers=mutated.fingers)
        except Exception:
            return None

        # TODO:算法之一（joint delete + relink + regroup）
        # ────────────────────────────────────────
        # 输入
        #   target: 已经构建好的 `HandCfg`
        #   cfg.target_finger: 目标手指名，None 时由运行时从所有 finger 中随机选一个
        #   cfg.deleted_joints: 需要删除的关节名称序列，空元组时由运行时从链中随机选
        #   cfg.regroup_strategy: 几何重组策略 { "merge" | "drop" | "keep" }
        #   cfg.respect_preset: 是否遵守 preset 的保留约束
        #   cfg.keep_terminal_joint: 是否强制保留 is_tip=True 的末端关节
        #
        # 输出：HandCfg | None
        #
        # ── 选取目标 ──
        #   1. 若 target_finger 为 None，从 hand.fingers 中随机选一个
        #   2. 若 deleted_joints 为空，从该 finger 的可删 joint 子集中随机选 1~N 个
        #      可删子集 = 全关节列表 - 末端关节（若 keep_terminal_joint=True）
        #                            - preset 保留集（若 respect_preset=True）
        #
        # ── 约束过滤 ──
        #   3. 若删除后剩余关节数 < preset 规定的最小关节数，拒绝并返回 None
        #   4. 若删除集合包含不可删关节，从集合中剔除（或拒绝，取决于 strict 参数，
        #      当前草案默认剔除非法项后继续）
        #
        # ── 链式重连 ──
        #   5. 对于每个被删 joint j（child link = L_j, parent link = L_p）：
        #      a. 找到 j 的后继 joint j_next（parent 原为 L_j）
        #      b. 将 j_next.parent 改写为 L_p
        #      c. 将 j_next.origin 更新为：origin_new = origin_p ∘ origin_j_next
        #         （即在 L_p 坐标系下叠加原有 j_next 的局部变换）
        #
        # ── Regroup（几何重组） ──
        #   6. 按 regroup_strategy 处理被删 joint 的 child link 几何：
        #      - "merge"：将 L_j 的 collisions/visuals（变换到 L_p 坐标系后）并入 L_p
        #      - "drop" ：丢弃 L_j 的几何
        #      - "keep" ：将 L_j 的几何作为额外 sub-link 保留（挂在 j_next 下）
        #
        # ── 重建 HandCfg ──
        #   7. 用更新后的 joint 列表重建 FingerCfg（过滤掉被删 joint）
        #   8. 用新 FingerCfg 重建 HandCfg（深拷贝其余 finger 不变）
        #   9. 返回新 HandCfg
        #
        # ── 与 preset 的交叉验证 ──
        #   被删关节集合必须是 preset 规定的"可删子集"的子集；
        #   若删后违反最小关节数约束，应当明确拒绝（返回 None）而不是静默修复。
        #
        # IDEA：joint delete 是后序工具里拓扑改变最大的操作；其输出必须
        # 能通过 HandValidator 的全局链式一致性检查，建议在此处嵌入轻量预检。

    def _delete_from_finger(self, hand: HandCfg, finger, delete_set: set[str]):
        r"""在单根 finger 上执行 joint 删除 + 重连。"""

        new_joints = []
        last_kept_parent = finger.parent_link  # 当前保留下来的链尾 link 名
        last_kept_container = hand.palm  # 用于 `merge` 时接收被删 joint 的几何
        pending_origin = PoseCfg()  # 从 `last_kept_parent` 到“下一个原始 joint parent”的累计位姿

        for joint in finger.joints:
            composed_origin = _compose_pose(pending_origin, joint.origin)  # 当前 joint 相对最近保留 link 的位姿

            if joint.name in delete_set:
                if self.cfg.regroup_strategy == "merge":
                    _merge_deleted_joint_into_container(last_kept_container, joint, composed_origin)
                pending_origin = composed_origin  # 下一关节需要继续把这一步的位姿吃进去
                continue

            kept_joint = joint.replace(parent=last_kept_parent, origin=composed_origin)
            new_joints.append(kept_joint)
            last_kept_parent = kept_joint.child
            last_kept_container = kept_joint
            pending_origin = PoseCfg()  # 一旦保留了当前 joint，累计位姿就重新归零

        if not new_joints:
            return None
        return finger.replace(joints=new_joints)


def _compose_pose(lhs: PoseCfg, rhs: PoseCfg) -> PoseCfg:
    r"""用当前项目一贯的“小角度/声明式叠加”语义组合两个位姿。"""

    return PoseCfg(
        pos=(lhs.pos[0] + rhs.pos[0], lhs.pos[1] + rhs.pos[1], lhs.pos[2] + rhs.pos[2]),
        rpy=(lhs.rpy[0] + rhs.rpy[0], lhs.rpy[1] + rhs.rpy[1], lhs.rpy[2] + rhs.rpy[2]),
    )


def _merge_deleted_joint_into_container(container, joint, joint_pose_in_container: PoseCfg) -> None:
    r"""把被删 joint 的几何与惯量近似并入保留容器。

    当前容器可能是：

    - `PalmCfg`
    - 前一个保留下来的 `JointCfg`

    由于当前 schema 是 joint-centric，`merge` 的可实现语义是：

    1. 把被删 joint 的 collision / visual 变换到容器 frame；
    2. 用显式近似把 inertial 也并入容器，而不是只并几何不并质量。
    """

    container.collisions.extend(
        [
            CollisionGeometryCfg(
                name=collision.name,
                geometry=collision.geometry.copy(),
                origin=_compose_pose(joint_pose_in_container, collision.origin),
            )
            for collision in joint.collisions
        ]
    )
    container.visuals.extend(
        [
            VisualGeometryCfg(
                name=visual.name,
                geometry=visual.geometry.copy(),
                origin=_compose_pose(joint_pose_in_container, visual.origin),
            )
            for visual in joint.visuals
        ]
    )

    if getattr(container, "inertial", None) is not None and joint.inertial is not None:
        container.inertial = _merge_inertials(
            container.inertial,
            joint.inertial.replace(origin=_compose_pose(joint_pose_in_container, joint.inertial.origin)),
        )


def _merge_inertials(lhs: InertialCfg, rhs: InertialCfg) -> InertialCfg:
    r"""把两个 link 级惯量近似并成一个新的 `InertialCfg`。

    这里采用保守的刚体合并近似：

    1. 质心按质量加权平均；
    2. 对角惯量按平行轴定理搬到新质心；
    3. 非对角项简单相加。

    这不是几何真值重积分，但比“只加质量、不改惯量”更接近物理语义。
    """

    m1 = lhs.mass
    m2 = rhs.mass
    total_mass = m1 + m2
    com = (
        (m1 * lhs.origin.pos[0] + m2 * rhs.origin.pos[0]) / total_mass,
        (m1 * lhs.origin.pos[1] + m2 * rhs.origin.pos[1]) / total_mass,
        (m1 * lhs.origin.pos[2] + m2 * rhs.origin.pos[2]) / total_mass,
    )

    dx1 = lhs.origin.pos[0] - com[0]
    dy1 = lhs.origin.pos[1] - com[1]
    dz1 = lhs.origin.pos[2] - com[2]
    dx2 = rhs.origin.pos[0] - com[0]
    dy2 = rhs.origin.pos[1] - com[1]
    dz2 = rhs.origin.pos[2] - com[2]

    inertia = {
        "ixx": lhs.inertia.ixx + m1 * (dy1 * dy1 + dz1 * dz1) + rhs.inertia.ixx + m2 * (dy2 * dy2 + dz2 * dz2),
        "iyy": lhs.inertia.iyy + m1 * (dx1 * dx1 + dz1 * dz1) + rhs.inertia.iyy + m2 * (dx2 * dx2 + dz2 * dz2),
        "izz": lhs.inertia.izz + m1 * (dx1 * dx1 + dy1 * dy1) + rhs.inertia.izz + m2 * (dx2 * dx2 + dy2 * dy2),
        "ixy": lhs.inertia.ixy + rhs.inertia.ixy,
        "ixz": lhs.inertia.ixz + rhs.inertia.ixz,
        "iyz": lhs.inertia.iyz + rhs.inertia.iyz,
    }
    return InertialCfg(mass=total_mass, origin=PoseCfg(pos=com), inertia=inertia)


__all__ = ["JointDeleteCfg", "JointDeleteMutator"]
