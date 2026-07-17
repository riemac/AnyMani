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
    frame `{h}` 在 env frame `{e}` 中的位姿 $T_{eh}$。`robots.hand_spawn.HandFrameCfg`
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
from typing import Literal, cast

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .adr_state import get_gm_adr_state
from .tactile_contact_state import reset_tactile_contact_state

GmHandOrientationMode = Literal["disabled", "roll", "pitch", "yaw", "so3"]
GmHandOrientationReferenceMode = Literal["anchor", "current"]


@dataclass(frozen=True)
class HandOrientationResetCfg:
    r"""Hand semantic frame orientation reset 的 scaffold 配置。

    该 dataclass 只表达 reset 分布语义，不是 IsaacLab `EventTermCfg`，也不实现
    采样 / 写 sim。未来实现时应在 reset event 中消费它：采样 $T_{eh}$，再结合
    `robots.hand_spawn.HandFrameCfg` 的 $T_{ha}$ 写入 robot raw root pose $T_{ea}=T_{eh}T_{ha}$。

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
                filtered_pairs.add(_ordered_link_pair(palm_link_name, str(link_name)))  # $(palm,l)$

    # same-finger 结构过滤：单根 finger 内部由关节约束定义运动，不让 mesh 轻微穿插驱动解算抖动。
    if filter_same_finger:
        for finger_link_chain in finger_link_chains:
            chain = tuple(str(link_name) for link_name in finger_link_chain)  # 当前 finger 的 link 链 $F_f$
            for index_a, link_a in enumerate(chain):
                for link_b in chain[index_a + 1 :]:
                    filtered_pairs.add(_ordered_link_pair(link_a, link_b))  # $(a,b), a,b\in F_f$

    return tuple(sorted(filtered_pairs))  # 排序只服务 stage diff / 测试可复现，不改变物理语义


def apply_generated_structural_collision_filter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    robot_prim_path: str,
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
    filter_palm_finger: bool = True,
    filter_same_finger: bool = True,
) -> None:
    r"""在 PhysX 初始化前 author generated-hand 结构性 pairwise collision 过滤。

    该 event 必须以 `mode="prestartup"` 挂到 IsaacLab `EventTerm`。此时 scene prim
    已经 spawn 完成，但 `sim.reset()` 尚未启动 PhysX handles；把 USD
    `FilteredPairsAPI` 写在这个阶段，PhysX solver 才能在初始化时读取 pair filter。

    当前规则来自 2026-06-24 单资产标定台消融：generated hand 的 palm/finger root /
    same-finger link mesh 可能存在结构性穿插或过近距离，若全自碰撞开启，会导致 slider
    调关节时非目标手指和 object 被 solver 连带抖动。训练中保留这种假接触会污染接触盆地，
    因此在 single-asset MDP probe 中显式应用：

    - palm 与任意 finger link 不碰；
    - 同一根 finger 内部 links 不碰；
    - 不同 fingers 之间仍然碰，保留真实 finger-finger 接触约束。

    NOTE(2026-06-25): 旧实现使用 external `PhysicsCollisionGroup`，会和 IsaacLab /
    IsaacSim Cloner 的 env-level collision filtering 竞争同一个 collider 的 group 归属。
    PhysX 日志会出现 `Collisions are supported currently only in one collision group`，
    并可能导致部分过滤规则被忽略。`FilteredPairsAPI` 是 IsaacSim 对 pairwise 过滤的
    更细粒度接口，且 schema 注释明确其优先级高于 `CollisionGroup`，因此这里改为
    对每个 link pair 显式写双向 `physics:filteredPairs`。

    Args:
        env (ManagerBasedRLEnv): IsaacLab manager-based env；需提供 `scene.stage` 与 `scene.env_prim_paths`。
        env_ids (Sequence[int] | None): `prestartup` 模式下由 EventManager 传入，当前忽略。
        robot_prim_path (str): robot prim path 模板，通常为 `"{ENV_REGEX_NS}/Robot"`。
        palm_link_name (str): palm link 名。
        finger_link_chains (Sequence[Sequence[str]]): 每根 finger 的 link 链。
        filter_palm_finger (bool): 是否过滤 palm-finger collision。
        filter_same_finger (bool): 是否过滤 same-finger internal collision。

    Raises:
        RuntimeError: 当前 USD build 缺少 `UsdPhysics.FilteredPairsAPI` 或 API authoring 失败。
        ValueError: 未提供 finger link chains，或 `robot_prim_path` 不能映射 cloned env prim。
    """

    _ = env_ids  # prestartup 是 stage 级操作，不按 env subset 随 episode reset 重复执行
    finger_link_chains = tuple(tuple(str(link_name) for link_name in chain) for chain in finger_link_chains)
    if len(finger_link_chains) == 0:
        raise ValueError("finger_link_chains must be non-empty for generated structural collision filtering.")

    # 懒加载 pxr：纯 contract / tensor tests 不需要 USD runtime，只有 Isaac Sim prestartup 才需要。
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    if not hasattr(UsdPhysics, "FilteredPairsAPI"):
        raise RuntimeError("Current USD build does not expose UsdPhysics.FilteredPairsAPI")

    stage = env.scene.stage  # 当前 USD stage；scene 已 spawn，但 PhysX 尚未 reset
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
    )  # 需要解析成 prim path 的 link 名集合

    link_paths_by_env: dict[str, dict[str, str]] = {}  # env path -> link name -> link prim path，便于逐 env 写 pair
    missing_link_names: set[str] = set()  # schema/link-name 不匹配时显式 warning，避免 silent wrong physics
    for env_prim_path in env_prim_paths:
        link_paths: dict[str, str] = {}  # 当前 cloned env 内的 link prim path 表
        for link_name in link_names:
            link_path = _structural_collision_link_prim_path(env_prim_path, robot_prim_path, link_name)
            if not stage.GetPrimAtPath(link_path).IsValid():
                missing_link_names.add(link_name)
                continue
            link_paths[link_name] = link_path  # `$l \mapsto /World/envs/env_i/Robot/l$`
        link_paths_by_env[env_prim_path] = link_paths

    authored_pair_edges = 0  # directed `physics:filteredPairs` edge 数；无向 pair 双向写入
    # 4096-env 训练会写约 $4096\times78\times2$ 条 directed edges，因此 edit target 外提，
    # 避免每条 edge 都进入一次 `Usd.EditContext` 造成启动时间膨胀。这里不使用 `Sdf.ChangeBlock`，
    # 因为 `UsdPrim.ApplyAPI(...)` 这类高层 API 在 ChangeBlock 中可能不可靠。
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        for link_paths in link_paths_by_env.values():
            for link_a, link_b in filtered_pairs:
                link_a_path = link_paths.get(link_a)
                link_b_path = link_paths.get(link_b)
                if link_a_path is None or link_b_path is None:
                    missing_link_names.update(
                        link_name
                        for link_name, link_path in ((link_a, link_a_path), (link_b, link_b_path))
                        if link_path is None
                    )
                    continue
                authored_pair_edges += _author_structural_filtered_pair_edge(stage, link_a_path, link_b_path)
                authored_pair_edges += _author_structural_filtered_pair_edge(stage, link_b_path, link_a_path)

    if missing_link_names:
        print(f"[WARN]: GM structural collision filter skipped missing hand links: {sorted(missing_link_names)}")
    print(
        "[INFO]: GM structural collision filter authored "
        f"link_pairs={len(filtered_pairs)}, directed_pair_edges={authored_pair_edges}"
    )
    setattr(env, "_gm_structural_collision_filter_stats", {
        "api": "FilteredPairsAPI",
        "link_pairs": len(filtered_pairs),
        "directed_edges": authored_pair_edges,
        "missing_link_names": tuple(sorted(missing_link_names)),
    })  # debug-only metadata，便于 smoke / 日志排查 pairwise 过滤是否生效


def _structural_collision_filter_link_names(
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
    *,
    filter_palm_finger: bool,
    filter_same_finger: bool,
) -> tuple[str, ...]:
    r"""返回需要写入 `FilteredPairsAPI` 的 link 名集合。"""

    link_names: set[str] = set()  # 只为参与结构过滤集合 $\mathcal{F}$ 的 links 写 pairwise API
    if filter_palm_finger:
        link_names.add(palm_link_name)  # palm 是 palm-finger pair 的固定端点
    if filter_palm_finger or filter_same_finger:
        for finger_link_chain in finger_link_chains:
            link_names.update(str(link_name) for link_name in finger_link_chain)  # 所有参与 pair 的 finger links
    return tuple(sorted(link_names))


def _ordered_link_pair(link_a: str, link_b: str) -> tuple[str, str]:
    r"""把无向 link pair 规范成静态可推断的二元 tuple。"""

    return (link_a, link_b) if link_a <= link_b else (link_b, link_a)


def _structural_collision_link_prim_path(env_prim_path: str, robot_prim_path: str, link_name: str) -> str:
    r"""构造某个 cloned env 中 robot link prim 的 USD path。"""

    if "{ENV_REGEX_NS}" not in robot_prim_path:
        raise ValueError("robot_prim_path must contain '{ENV_REGEX_NS}' so it can be resolved for each cloned env.")
    robot_path = robot_prim_path.replace("{ENV_REGEX_NS}", env_prim_path)  # `/World/envs/env_i/Robot`
    return f"{robot_path}/{link_name}"  # link prim path，FilteredPairsAPI 以 link prim 作为 pairwise source/target


def _author_structural_filtered_pair_edge(stage, source_link_path: str, target_link_path: str) -> int:
    r"""写入一条 directed `physics:filteredPairs` relationship edge。

    `FilteredPairsAPI` 的语义是“source prim 不与 relationship target prim 碰撞”。IsaacSim
    `RobotAssembler.mask_collisions(...)` 也按 prim-to-prim target 写法使用该 API。这里
    对无向结构 pair 写两条 directed edge，避免不同 PhysX/USD 版本对单向关系是否对称的解释
    影响训练物理。

    Args:
        stage: 当前 IsaacSim USD stage。
        source_link_path (str): 被应用 `PhysicsFilteredPairsAPI` 的 link prim path。
        target_link_path (str): 需要过滤碰撞的另一个 link prim path。

    Returns:
        int: 新增 relationship target 时返回 1，目标已存在时返回 0。

    Raises:
        RuntimeError: 当 source/target link prim 不存在，或 `FilteredPairsAPI.Apply(...)` 失败。
    """

    from pxr import Sdf, UsdPhysics  # noqa: PLC0415

    source_prim = stage.GetPrimAtPath(source_link_path)
    target_prim = stage.GetPrimAtPath(target_link_path)
    if not source_prim.IsValid() or not target_prim.IsValid():
        raise RuntimeError(
            "Cannot author structural filtered pair for invalid link prims: "
            f"source={source_link_path!r}, target={target_link_path!r}."
        )

    filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
    if not filtered_pairs_api:
        raise RuntimeError(f"Failed to apply UsdPhysics.FilteredPairsAPI to {source_link_path}")

    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    if not filtered_pairs_rel:
        filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()

    target_path = Sdf.Path(target_link_path)  # relationship target 指向另一个 link prim
    if target_path in set(filtered_pairs_rel.GetTargets()):
        return 0
    filtered_pairs_rel.AddTarget(target_path)
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
        setattr(env, "_gm_object_reset_anchor_w", object_asset.data.root_pos_w.clone())  # `[B,3]`，全量 anchor
    reset_anchor = getattr(env, "_gm_object_reset_anchor_w")
    reset_anchor[env_ids] = anchor_w  # 只更新 reset 的 env


def record_robot_reset_joint_anchor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    r"""记录本 episode reset 后的真实抓取关节姿态，供 pose stability penalty 使用。"""

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    anchor = getattr(env, "_gm_robot_reset_joint_anchor", None)
    if not isinstance(anchor, torch.Tensor) or anchor.shape != joint_pos.shape:
        anchor = joint_pos.clone()  # `[B,16]`，canonical articulation/sidecar order
        setattr(env, "_gm_robot_reset_joint_anchor", anchor)
    anchor[env_ids] = joint_pos[env_ids].clone()


def reset_adr_episode_length(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_episode_length_s: float = 20.0,
) -> None:
    r"""为每个 reset env 采样 20--120 s full horizon，并保留 policy-step 单位 buffer。"""

    ids = _resolve_event_env_ids(env, env_ids)
    episode_lengths = getattr(env, "leap_adr_episode_lengths", None)
    if not isinstance(episode_lengths, torch.Tensor):
        episode_lengths = torch.full(
            (env.num_envs,), env.max_episode_length, dtype=torch.long, device=env.device
        )
        setattr(env, "leap_adr_episode_lengths", episode_lengths)
    min_steps = max(1, int(float(min_episode_length_s) / float(env.step_dt)))
    episode_lengths[ids] = torch.randint(
        min_steps,
        env.max_episode_length + 1,
        (ids.numel(),),
        dtype=torch.long,
        device=env.device,
    )


def reset_adr_object_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""用当前 GM ADR width 从 default object root state 采样位置与 roll/pitch reset。"""

    ids = _resolve_event_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    object_asset: RigidObject = env.scene[asset_cfg.name]
    root_state = object_asset.data.default_root_state[ids].clone()
    root_state[:, :3] += env.scene.env_origins[ids]
    root_state[:, 7:] = 0.0
    widths_xy = torch.tensor(
        [float(getattr(env, "leap_adr_object_x_width", 0.0)), float(getattr(env, "leap_adr_object_y_width", 0.0))],
        device=env.device,
    )
    root_state[:, :2] += math_utils.sample_uniform(-1.0, 1.0, (ids.numel(), 2), env.device) * widths_xy
    widths_rpy = torch.tensor(
        [
            float(getattr(env, "leap_adr_object_x_rot", 0.0)),
            float(getattr(env, "leap_adr_object_y_rot", 0.0)),
            float(getattr(env, "leap_adr_object_z_rot", 0.0)),
        ],
        device=env.device,
    )
    rpy = math_utils.sample_uniform(-1.0, 1.0, (ids.numel(), 3), env.device) * widths_rpy
    noise_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    root_state[:, 3:7] = math_utils.quat_mul(noise_quat, root_state[:, 3:7])
    # Isaac Lab 5.1 stub 仍标作 `Sequence[int]`，但 PhysX torch frontend 实际要求 tensor 并调用 `.to()`。
    physx_ids = cast(Sequence[int], ids)  # 静态类型适配；runtime 对象仍是 contiguous CUDA tensor `[K]`
    object_asset.write_root_pose_to_sim(root_state[:, :7], physx_ids)  # 写入 $p_{wo},q_{wo}$
    object_asset.write_root_velocity_to_sim(root_state[:, 7:], physx_ids)  # `[K,6]`，同步清零线/角速度


def reset_adr_robot_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
) -> None:
    r"""从 default pre-grasp 加当前 ADR joint noise，写 sim 并缓存 action target/pose anchor。"""

    ids = _resolve_event_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = robot.data.default_joint_pos[ids][:, joint_ids].clone()
    joint_vel = robot.data.default_joint_vel[ids][:, joint_ids].clone()
    pos_width = float(getattr(env, "leap_adr_joint_pos_noise", 0.0))
    vel_width = float(getattr(env, "leap_adr_joint_vel_noise", 0.0))
    joint_pos += math_utils.sample_uniform(-pos_width, pos_width, joint_pos.shape, env.device)
    limits = robot.data.soft_joint_pos_limits[ids][:, joint_ids, :]
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
    joint_vel += math_utils.sample_uniform(-vel_width, vel_width, joint_vel.shape, env.device)

    reset_joint_pos = getattr(env, "leap_official_reset_joint_pos", None)
    if not isinstance(reset_joint_pos, torch.Tensor) or reset_joint_pos.shape != robot.data.default_joint_pos[:, joint_ids].shape:
        reset_joint_pos = robot.data.default_joint_pos[:, joint_ids].clone()
        setattr(env, "leap_official_reset_joint_pos", reset_joint_pos)
    reset_joint_pos[ids] = joint_pos
    # 与 rigid-object API 相同，torch backend 消费 tensor，type stub 则保守声明为 `Sequence[int]`。
    physx_ids = cast(Sequence[int], ids)  # 静态类型适配；不把 ids 转成会破坏 PhysX `.to()` 的 Python list
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=physx_ids)  # `[K,16]` reset target
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=physx_ids)  # 写实际状态

    anchor = getattr(env, "_gm_robot_reset_joint_anchor", None)
    if not isinstance(anchor, torch.Tensor) or anchor.shape != reset_joint_pos.shape:
        anchor = reset_joint_pos.clone()
        setattr(env, "_gm_robot_reset_joint_anchor", anchor)
    anchor[ids] = joint_pos  # actual reset pose，不是未加噪 pre-grasp preset


def randomize_object_com_from_default_and_record(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""相对首次捕获的 default COM 采样每轴 offset，禁止 episode 间随机游走。

    当前 half-width 由 `env.gm_adr_com_half_width` 从 0 线性扩到 0.01 m。实际 offset
    直接写入 48D critic state；每次 reset 不从 current COM 加噪。
    """

    ids = _resolve_event_env_ids(env, env_ids)
    object_asset: RigidObject = env.scene[asset_cfg.name]
    ids_cpu = ids.cpu()
    default_coms = getattr(env, "_gm_default_object_coms_cpu", None)
    if not isinstance(default_coms, torch.Tensor):
        default_coms = object_asset.root_physx_view.get_coms().clone().cpu()
        setattr(env, "_gm_default_object_coms_cpu", default_coms)
    coms = object_asset.root_physx_view.get_coms().clone()
    half_width = float(getattr(env, "gm_adr_com_half_width", 0.0))
    offsets = math_utils.sample_uniform(-half_width, half_width, (ids.numel(), 3), device="cpu")
    if coms.ndim == 2:  # RigidObject view: `[N,7]`，每个 env 单一 body COM pose
        coms[ids_cpu, :3] = default_coms[ids_cpu, :3] + offsets
    elif coms.ndim == 3:  # Articulation-compatible fallback: `[N,B,7]`
        coms[ids_cpu, :, :3] = default_coms[ids_cpu, :, :3] + offsets[:, None, :]
    else:
        raise RuntimeError(f"Unsupported PhysX COM tensor shape: {tuple(coms.shape)}")
    object_asset.root_physx_view.set_coms(coms, ids_cpu)
    get_gm_adr_state(env).set(env, "com", offsets.to(env.device), ids)


def randomize_object_scale_and_record(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""调用 Isaac Lab prestartup scale randomization，并从 USD 读取实际 isotropic multiplier。"""

    isaac_mdp.randomize_rigid_body_scale(env, env_ids, scale_range, asset_cfg)
    ids = _resolve_event_env_ids(env, env_ids)
    ids_cpu = ids.cpu()
    from isaaclab.sim import find_matching_prim_paths  # noqa: PLC0415 - 只在 prestartup/Kit runtime 需要
    from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415

    object_asset: RigidObject = env.scene[asset_cfg.name]
    prim_paths = find_matching_prim_paths(object_asset.cfg.prim_path)
    stage = get_current_stage()
    actual_scale = []
    for env_id in ids_cpu.tolist():
        scale_value = stage.GetPrimAtPath(prim_paths[env_id]).GetAttribute("xformOp:scale").Get()
        actual_scale.append(float(scale_value[0]))  # isotropic U(1.1,1.25)，三轴相同
    get_gm_adr_state(env).set(env, "scale", torch.tensor(actual_scale, device=env.device), ids)


class RandomizeRigidBodyMassAndRecord(isaac_mdp.randomize_rigid_body_mass):
    r"""Isaac Lab mass randomizer + actual mass snapshot writer。"""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1.0e-6,
    ) -> None:
        r"""应用 mass randomization 后一次读取实际 kg 值并写 critic state。"""

        super().__call__(
            env,
            env_ids,
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution,
            recompute_inertia,
            min_mass,
        )
        ids = _resolve_event_env_ids(env, env_ids)
        masses = self.asset.root_physx_view.get_masses()[ids.cpu()].float().mean(dim=-1).to(env.device)
        get_gm_adr_state(env).set(env, "mass", masses, ids)


class RandomizeActuatorGainsAndRecord(isaac_mdp.randomize_actuator_gains):
    r"""Isaac Lab actuator gain randomizer + canonical 16D actual Kp/Kd writer。"""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        r"""随机化后按 resolved `joint_ids` 记录 actual gains。"""

        super().__call__(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params,
            damping_distribution_params,
            operation,
            distribution,
        )
        ids = _resolve_event_env_ids(env, env_ids)
        if not isinstance(self.asset, Articulation):
            raise TypeError("RandomizeActuatorGainsAndRecord requires an Articulation asset.")
        joint_ids = self.asset_cfg.joint_ids
        stiffness = self.asset.data.joint_stiffness[ids][:, joint_ids]
        damping = self.asset.data.joint_damping[ids][:, joint_ids]
        state = get_gm_adr_state(env)
        state.set(env, "stiffness", stiffness, ids)
        state.set(env, "damping", damping, ids)


class RandomizeRigidBodyMaterialAndRecord(isaac_mdp.randomize_rigid_body_material):
    r"""Material bucket assignment + selected body/contact-shape actual mean writer。"""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
        adr_state_field: Literal["object_material", "hand_contact_material"] = "object_material",
    ) -> None:
        r"""分配 bucket 后记录所选 shapes 的 `(mu_s,mu_d,restitution)` 均值。"""

        super().__call__(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )
        ids = _resolve_event_env_ids(env, env_ids)
        materials = self.asset.root_physx_view.get_material_properties()[ids.cpu()]
        selected_body_ids = self.asset_cfg.body_ids
        if self.num_shapes_per_body is not None and isinstance(selected_body_ids, list):
            shape_ids: list[int] = []
            for body_id in selected_body_ids:
                start = sum(self.num_shapes_per_body[:body_id])
                shape_ids.extend(range(start, start + self.num_shapes_per_body[body_id]))
            materials = materials[:, shape_ids]
        material_mean = materials.float().mean(dim=1).to(env.device)
        get_gm_adr_state(env).set(env, adr_state_field, material_mean, ids)


def resample_adr_material_buckets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str,
    range_attr: str,
) -> None:
    r"""ADR range 改变时重建有限 material buckets，同档仅重新分配并记录实际值。"""

    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except ValueError:
        return
    term = term_cfg.func
    if not hasattr(term, "material_buckets"):
        return
    ranges_dict = getattr(env, range_attr)
    ranges = torch.tensor(
        [ranges_dict["static"], ranges_dict["dynamic"], ranges_dict["restitution"]], dtype=torch.float32
    )
    num_buckets = int(term_cfg.params.get("num_buckets", term.material_buckets.shape[0]))
    signature = (
        tuple(float(value) for value in ranges[:, 0]),
        tuple(float(value) for value in ranges[:, 1]),
        num_buckets,
        bool(term_cfg.params.get("make_consistent", False)),
    )
    if getattr(term, "_gm_adr_bucket_signature", None) != signature:
        buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        if term_cfg.params.get("make_consistent", False):
            buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
        term.material_buckets = buckets
        term._gm_adr_bucket_signature = signature
    term(env, env_ids, **term_cfg.params)


def reset_adr_wrench_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    probability: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""采样 episode-level Bernoulli wrench gate，并记录 gate/max acceleration actual state。"""

    ids = _resolve_event_env_ids(env, env_ids)
    gate = getattr(env, "leap_adr_apply_wrench", None)
    if not isinstance(gate, torch.Tensor):
        gate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, "leap_adr_apply_wrench", gate)
    gate[ids] = torch.rand(ids.numel(), device=env.device) <= float(probability)
    state = get_gm_adr_state(env)
    state.set(env, "wrench_gate", gate[ids], ids)
    state.set(env, "max_acceleration", float(getattr(env, "leap_adr_max_linear_accel", 0.5)), ids)
    apply_adr_object_wrench(env, ids, asset_cfg)


def apply_adr_object_wrench(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    torsional_radius: float = 0.0,
) -> None:
    r"""按 actual mass 与 ADR max acceleration 采样分段常值 object force/torque。"""

    ids = _resolve_event_env_ids(env, env_ids)
    object_asset: RigidObject = env.scene[asset_cfg.name]
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else object_asset.num_bodies
    max_acceleration = float(getattr(env, "leap_adr_max_linear_accel", 0.5))
    masses = object_asset.root_physx_view.get_masses().to(env.device)[ids]
    max_force = (masses * max_acceleration).unsqueeze(-1)
    max_torque = (masses * max_acceleration * float(torsional_radius)).unsqueeze(-1)
    forces = max_force * math_utils.sample_uniform(-1.0, 1.0, (ids.numel(), num_bodies, 3), env.device)
    torques = max_torque * math_utils.sample_uniform(-1.0, 1.0, (ids.numel(), num_bodies, 3), env.device)
    gate = getattr(env, "leap_adr_apply_wrench", None)
    if isinstance(gate, torch.Tensor):
        gate_subset = gate[ids].view(-1, 1, 1)
        forces = torch.where(gate_subset, forces, torch.zeros_like(forces))
        torques = torch.where(gate_subset, torques, torch.zeros_like(torques))
    body_ids = asset_cfg.body_ids
    resolved_body_ids = torch.tensor(body_ids, dtype=torch.long, device=env.device) if isinstance(body_ids, list) else None
    object_asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=resolved_body_ids,
        env_ids=ids,
    )
    get_gm_adr_state(env).set(env, "max_acceleration", max_acceleration, ids)


def _resolve_event_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    r"""把 prestartup/reset/interval event env ids 统一为 env-device LongTensor。"""

    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


__all__ = [
    "DEFAULT_HAND_ORIENTATION_RESET_CFG",
    "GmHandOrientationMode",
    "GmHandOrientationReferenceMode",
    "HandOrientationResetCfg",
    "RandomizeActuatorGainsAndRecord",
    "RandomizeRigidBodyMassAndRecord",
    "RandomizeRigidBodyMaterialAndRecord",
    "apply_adr_object_wrench",
    "apply_generated_structural_collision_filter",
    "generated_structural_collision_filter_pairs",
    "record_object_reset_anchor",
    "record_robot_reset_joint_anchor",
    "randomize_object_com_from_default_and_record",
    "randomize_object_scale_and_record",
    "resample_adr_material_buckets",
    "reset_adr_episode_length",
    "reset_adr_object_state",
    "reset_adr_robot_joints",
    "reset_adr_wrench_state",
    "reset_tactile_contact_state",
]
