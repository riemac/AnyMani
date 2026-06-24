r"""Reset and domain-randomization terms for `tasks.gm`.

本文件只保留 AnyMani 自己无法直接复用 IsaacLab 官方 MDP 的 event 语义。
能由 `isaaclab.envs.mdp` 表达的 reset，例如 hand joint offset reset、
object root pose reset、刚体材质或质量随机化，都应在 env cfg 中直接挂官方
`EventTerm`，避免把多个物理动作包成一个臃肿函数。

当前 reset 拆分原则：
    - hand joint state reset：优先使用 IsaacLab 官方 `reset_joints_by_offset`；
    - object root pose reset：优先使用 IsaacLab 官方 `reset_root_state_uniform`；
    - AnyMani 专属状态：只在这里实现，例如记录 object reset anchor，供
      `object_out_of_hand` 判断物体是否离开本次 episode 初始接触盆地。

hand orientation reset scaffold：
    任意手朝向训练的核心被控量不是 raw asset frame `{a}` 本身，而是 hand semantic
    frame `{h}` 在 env frame `{e}` 中的位姿 $T_{eh}$。`hand_spawn.HandFrameCfg`
    负责记录静态校准 $T_{ha}$ 与默认 anchor $T_{eh}^{anchor}$；reset event 后续
    负责采样 episode 级 $T_{eh}$，再 lower 成 raw root pose：
    $$
    T_{ea}=T_{eh}T_{ha}.
    $$
    第一版 scaffold 只定义 orientation 分布，不实现写 sim。默认 reference mode
    是 `anchor`：每次 reset 从 $R_{eh}^{anchor}$ 右乘 hand-frame 扰动
    $\Delta R_h$，即 $R_{eh}'=R_{eh}^{anchor}\Delta R_h$。`current` 随机游走模式
    仅作预留，避免训练初期把 i.i.d. 初态分布误写成 episode 间漂移。

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

GmHandOrientationMode = Literal["disabled", "roll", "pitch", "yaw", "so3"]
GmHandOrientationReferenceMode = Literal["anchor", "current"]


@dataclass(frozen=True)
class HandOrientationResetCfg:
    r"""Hand semantic frame orientation reset 的 scaffold 配置。

    该 dataclass 只表达 reset 分布语义，不是 IsaacLab `EventTermCfg`，也不实现
    采样 / 写 sim。未来实现时应在 reset event 中消费它：采样 $T_{eh}$，再结合
    `hand_spawn.HandFrameCfg` 的 $T_{ha}$ 写入 robot raw root pose $T_{ea}=T_{eh}T_{ha}$。

    设计要点：

    - `roll/pitch/yaw` 均解释为 hand semantic frame `{h}` 的 body/right 轴扰动；
    - `so3` 表示 Haar-uniform 全 $SO(3)$，实现时可在采样边界使用 quaternion 算法；
    - 默认 `reference_mode="anchor"`，即每次 reset 从 $R_{eh}^{anchor}$ 右乘扰动；
    - `reference_mode="current"` 只作为未来 continual perturbation / curriculum 预留。
    """

    mode: GmHandOrientationMode = "disabled"
    """orientation reset 模式；`disabled` 表示不改变 hand orientation。"""

    reference_mode: GmHandOrientationReferenceMode = "anchor"
    """扰动参考：`anchor` 为 i.i.d. reset 分布，`current` 为 episode 间累积随机游走预留。"""

    angle_range: tuple[float, float] = (0.0, 0.0)
    r"""`roll/pitch/yaw` 模式的角度范围，单位 rad。

    对 `so3` 模式，该字段应被忽略；`so3` 的科研语义是全 $SO(3)$ 均匀采样，
    不是随机轴 + 受限角度。
    """

    perturbation_frame: Literal["h"] = "h"
    """扰动轴所在 frame。第一版固定为 hand semantic frame `{h}` 的 body/right 语义。"""

    robot_asset_name: str = "robot"
    """scene 中 hand articulation 的名字；未来 event 实现用它读取 / 写入 root pose。"""


DEFAULT_HAND_ORIENTATION_RESET_CFG = HandOrientationResetCfg()


def generated_structural_collision_filter_pairs(
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
    *,
    filter_palm_finger: bool = True,
    filter_same_finger: bool = True,
) -> tuple[tuple[str, str], ...]:
    r"""构造 generated hand 的结构性 collision filter pair 集合。

    该 helper 只表达 pair-level 科研语义，不触碰 USD stage，便于用纯单元测试锁住
    “哪些 link 不应参与 PhysX 自碰撞求解”。当前 single-asset 训练采用的规则是：

    $$
    \mathcal{F}
      = \{(\text{palm}, l)\mid l \in \cup_f F_f\}
        \cup
        \bigcup_f \{(a,b)\mid a,b\in F_f,\ a\ne b\}.
    $$

    其中 $F_f$ 是同一根 finger 的 link 链。注意没有加入
    $F_i\times F_j,\ i\ne j$，所以不同 fingers 之间仍保留碰撞；这正是用户确认的
    训练物理规则：同 finger 内部忽略、finger-palm 忽略、finger-finger 保留。

    Args:
        palm_link_name (str): palm link 名，例如 `"palm"`。
        finger_link_chains (Sequence[Sequence[str]]): 每根 finger 的 link 名链。
        filter_palm_finger (bool): 是否过滤 palm 与所有 finger links 的 collision pair。
        filter_same_finger (bool): 是否过滤同一 finger 内部任意 link-link collision pair。

    Returns:
        tuple[tuple[str, str], ...]: 排序后的无向 link pair；每个 pair 内部也按字典序排序。
    """

    finger_link_chains = tuple(tuple(str(link_name) for link_name in chain) for chain in finger_link_chains)
    filtered_pairs: set[tuple[str, str]] = set()  # 无向 collision filter pair 集合 $\mathcal{F}$

    # palm-finger 结构过滤：palm 与每根 finger 的每个 link 都不让 PhysX 解自碰撞穿插。
    if filter_palm_finger:
        for finger_link_chain in finger_link_chains:
            for link_name in finger_link_chain:
                filtered_pairs.add(tuple(sorted((palm_link_name, str(link_name)))))  # $(palm,l)$

    # same-finger 结构过滤：单根 finger 内部由关节约束定义运动，不让 mesh 轻微穿插驱动解算抖动。
    if filter_same_finger:
        for finger_link_chain in finger_link_chains:
            chain = tuple(str(link_name) for link_name in finger_link_chain)  # 当前 finger 的 link 链 $F_f$
            for index_a, link_a in enumerate(chain):
                for link_b in chain[index_a + 1 :]:
                    filtered_pairs.add(tuple(sorted((link_a, link_b))))  # $(a,b), a,b\in F_f$

    return tuple(sorted(filtered_pairs))  # 排序只服务 stage diff / 测试可复现，不改变物理语义


def apply_generated_structural_collision_filter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    robot_prim_path: str,
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
    filter_palm_finger: bool = True,
    filter_same_finger: bool = True,
    collision_group_root: str = "/World/anymani_gm_generated_structural_collision_filters",
) -> None:
    r"""在 PhysX 初始化前 author generated-hand 结构性 collision group 过滤。

    该 event 必须以 `mode="prestartup"` 挂到 IsaacLab `EventTerm`。此时 scene prim
    已经 spawn 完成，但 `sim.reset()` 尚未启动 PhysX handles；把 USD
    `PhysicsCollisionGroup` 写在这个阶段，PhysX solver 才能在初始化时读取 pair filter。

    当前规则来自 2026-06-24 单资产标定台消融：generated hand 的 palm/finger root /
    same-finger link mesh 可能存在结构性穿插或过近距离，若全自碰撞开启，会导致 slider
    调关节时非目标手指和 object 被 solver 连带抖动。训练中保留这种假接触会污染接触盆地，
    因此在 single-asset MDP probe 中显式应用：

    - palm 与任意 finger link 不碰；
    - 同一根 finger 内部 links 不碰；
    - 不同 fingers 之间仍然碰，保留真实 finger-finger 接触约束。

    Args:
        env (ManagerBasedRLEnv): IsaacLab manager-based env；需提供 `scene.stage` 与 `scene.env_prim_paths`。
        env_ids (Sequence[int] | None): `prestartup` 模式下由 EventManager 传入，当前忽略。
        robot_prim_path (str): robot prim path 模板，通常为 `"{ENV_REGEX_NS}/Robot"`。
        palm_link_name (str): palm link 名。
        finger_link_chains (Sequence[Sequence[str]]): 每根 finger 的 link 链。
        filter_palm_finger (bool): 是否过滤 palm-finger collision。
        filter_same_finger (bool): 是否过滤 same-finger internal collision。
        collision_group_root (str): stage 中 external collision group scope 路径。

    Raises:
        RuntimeError: 当前 USD build 缺少 `UsdPhysics.CollisionGroup` 或 group scope 写入失败。
        ValueError: 未提供 finger link chains，或 `robot_prim_path` 不能映射 cloned env prim。
    """

    _ = env_ids  # prestartup 是 stage 级操作，不按 env subset 随 episode reset 重复执行
    finger_link_chains = tuple(tuple(str(link_name) for link_name in chain) for chain in finger_link_chains)
    if len(finger_link_chains) == 0:
        raise ValueError("finger_link_chains must be non-empty for generated structural collision filtering.")

    # 懒加载 pxr：纯 contract / tensor tests 不需要 USD runtime，只有 Isaac Sim prestartup 才需要。
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    if not hasattr(UsdPhysics, "CollisionGroup"):
        raise RuntimeError("Current USD build does not expose UsdPhysics.CollisionGroup")

    stage = env.scene.stage  # 当前 USD stage；scene 已 spawn，但 PhysX 尚未 reset
    root_layer = stage.GetRootLayer()  # 过滤 schema 写在 root layer，保证 PhysX 初始化可见
    env_prim_paths = tuple(str(env_prim_path) for env_prim_path in env.scene.env_prim_paths)  # cloned env roots
    filtered_pairs = generated_structural_collision_filter_pairs(
        palm_link_name=palm_link_name,
        finger_link_chains=finger_link_chains,
        filter_palm_finger=filter_palm_finger,
        filter_same_finger=filter_same_finger,
    )  # $\mathcal{F}$，无向 link-level pair 集合
    link_names = _structural_collision_filter_link_names(
        palm_link_name=palm_link_name,
        finger_link_chains=finger_link_chains,
        filter_palm_finger=filter_palm_finger,
        filter_same_finger=filter_same_finger,
    )  # 需要建 CollisionGroup 的 link 名集合

    # 先创建 external scope，然后在其中为每个 link 建 `PhysicsCollisionGroup`。
    with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
        UsdGeom.Scope.Define(stage, collision_group_root)
    collision_group_root_spec = root_layer.GetPrimAtPath(collision_group_root)
    if collision_group_root_spec is None:
        raise RuntimeError(f"Failed to define collision group scope at {collision_group_root}")

    link_group_paths: dict[str, str] = {}  # link name -> `/World/.../<link>` collision group prim path
    missing_link_names: set[str] = set()  # schema/link-name 不匹配时显式 warning，避免 silent wrong physics

    # 用 Sdf author collection，collection include 指向 link prim，并用 expandPrims 纳入其下 collision descendants。
    with Sdf.ChangeBlock():
        for link_name in link_names:
            first_env_link_path = _structural_collision_link_prim_path(env_prim_paths[0], robot_prim_path, link_name)
            if not stage.GetPrimAtPath(first_env_link_path).IsValid():
                missing_link_names.add(link_name)
                continue

            collision_group = Sdf.PrimSpec(
                collision_group_root_spec,
                link_name,
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )  # external collision group，不修改原 URDF/USD asset
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )  # collection expansion rule attribute
            expansion_rule.default = "expandPrims"  # link prim 下的 collision mesh descendants 全部纳入 group

            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            for env_prim_path in env_prim_paths:
                includes_rel.targetPathList.Append(
                    _structural_collision_link_prim_path(env_prim_path, robot_prim_path, link_name)
                )  # 每个 cloned env 的同名 link 都进入同一个 link-level group

            link_group_paths[link_name] = f"{collision_group_root}/{link_name}"  # 供 filteredGroups relationship 使用

    authored_group_edges = 0  # directed `physics:filteredGroups` edge 数；无向 pair 双向写入
    for link_a, link_b in filtered_pairs:
        group_a_path = link_group_paths.get(link_a)
        group_b_path = link_group_paths.get(link_b)
        if group_a_path is None or group_b_path is None:
            missing_link_names.update(
                link_name for link_name, group_path in ((link_a, group_a_path), (link_b, group_b_path)) if group_path is None
            )
            continue
        authored_group_edges += _author_structural_filtered_group_edge(stage, group_a_path, group_b_path)
        authored_group_edges += _author_structural_filtered_group_edge(stage, group_b_path, group_a_path)

    if missing_link_names:
        print(f"[WARN]: GM structural collision filter skipped missing hand links: {sorted(missing_link_names)}")
    print(
        "[INFO]: GM structural collision filter authored "
        f"groups={len(link_group_paths)}, link_pairs={len(filtered_pairs)}, directed_edges={authored_group_edges}"
    )
    env._gm_structural_collision_filter_stats = {
        "groups": len(link_group_paths),
        "link_pairs": len(filtered_pairs),
        "directed_edges": authored_group_edges,
        "missing_link_names": tuple(sorted(missing_link_names)),
    }  # debug-only metadata，便于 smoke / 日志排查 stage 过滤是否生效


def _structural_collision_filter_link_names(
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
    *,
    filter_palm_finger: bool,
    filter_same_finger: bool,
) -> tuple[str, ...]:
    r"""返回需要创建 `PhysicsCollisionGroup` 的 link 名集合。"""

    link_names: set[str] = set()  # group collection 只为参与 filter 的 links 创建
    if filter_palm_finger:
        link_names.add(palm_link_name)  # palm 是 palm-finger pair 的固定端点
    if filter_palm_finger or filter_same_finger:
        for finger_link_chain in finger_link_chains:
            link_names.update(str(link_name) for link_name in finger_link_chain)  # 所有参与 pair 的 finger links
    return tuple(sorted(link_names))


def _structural_collision_link_prim_path(env_prim_path: str, robot_prim_path: str, link_name: str) -> str:
    r"""构造某个 cloned env 中 robot link prim 的 USD path。"""

    if "{ENV_REGEX_NS}" not in robot_prim_path:
        raise ValueError("robot_prim_path must contain '{ENV_REGEX_NS}' so it can be resolved for each cloned env.")
    robot_path = robot_prim_path.replace("{ENV_REGEX_NS}", env_prim_path)  # `/World/envs/env_i/Robot`
    return f"{robot_path}/{link_name}"  # link prim path，collection expandPrims 会纳入其 collision descendants


def _author_structural_filtered_group_edge(stage, source_group_path: str, target_group_path: str) -> int:
    r"""写入一条 directed `physics:filteredGroups` relationship edge。"""

    from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        source_group = UsdPhysics.CollisionGroup.Get(stage, source_group_path)
        if not source_group:
            raise RuntimeError(f"Missing structural collision group at {source_group_path}")

        filtered_groups_rel = source_group.GetFilteredGroupsRel()
        if not filtered_groups_rel:
            filtered_groups_rel = source_group.CreateFilteredGroupsRel()

        target_path = Sdf.Path(target_group_path)  # relationship target 指向另一个 collision group prim
        if target_path in set(filtered_groups_rel.GetTargets()):
            return 0
        filtered_groups_rel.AddTarget(target_path)
        return 1


def record_object_reset_anchor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    r"""记录本次 reset 后 object 的 world-frame anchor。

    该 event 不改变仿真状态，只把当前 object root position 记为
    `object_out_of_hand` 的 episode anchor。这样 object 位姿 reset 可以复用
    IsaacLab 官方 `reset_root_state_uniform`，而离手判据仍保留 AnyMani 的
    “相对本次初态偏移”语义。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        env_ids (torch.Tensor): 需要记录 anchor 的环境 id。
        object_cfg (SceneEntityCfg): object rigid body 配置。
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    anchor_w = object_asset.data.root_pos_w[env_ids].clone()  # `[K,3]`，本次 reset 后 object world position
    if not isinstance(getattr(env, "_gm_object_reset_anchor_w", None), torch.Tensor):
        env._gm_object_reset_anchor_w = object_asset.data.root_pos_w.clone()  # `[B,3]`，初始化全量 anchor buffer
    env._gm_object_reset_anchor_w[env_ids] = anchor_w  # 只更新 reset 的 env


__all__ = [
    "DEFAULT_HAND_ORIENTATION_RESET_CFG",
    "GmHandOrientationMode",
    "GmHandOrientationReferenceMode",
    "HandOrientationResetCfg",
    "apply_generated_structural_collision_filter",
    "generated_structural_collision_filter_pairs",
    "record_object_reset_anchor",
]
