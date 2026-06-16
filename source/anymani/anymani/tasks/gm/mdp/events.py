r"""Reset and domain-randomization semantics for `tasks.gm`.

本文件当前只落 **event MDP 语义契约**，不实现 Isaac Lab event function。
这样做是为了避免在 cache tensor 格式、multi-asset batching、per-env scale
bucket 尚未验证前，把一个会被误认为可训练的 reset pipeline 接进环境。

主线 reset 分布：grasp cache。
    在线 reset 应从经过物理验证的经验分布采样：
    $$
    (q, T^h_o) \sim
    \mathcal{D}_{\text{grasp}}(q, T^h_o \mid a, o, s, \rho),
    $$
    其中 $a$ 是 hand `asset_id`，$o$ 是 object id，$s$ 是 object scale bucket，
    $\rho$ 是 cache generation 时采用的 pose distribution。reset 写入 hand
    joint position $q$ 与 object 相对 hand semantic frame `{h}` 的位姿 $T^h_o$；
    hand / object velocity 第一版置零，并同步 action target / previous target，避免
    PD controller 在 reset 后立刻把手拉离 cache 稳定姿态。

object pose DR 的互斥关系：
    cache reset 启用时，不再叠加普通 `reset_root_state_uniform` 的 `x/y/z/yaw`
    扰动。`<=1 cm` 平移扰动与 yaw-uniform 属于 cache generation / validation
    分布的一部分，而不是 reset 后置扰动。无 cache 消融才允许走 random
    object pose + random hand joint reset。

object scale DR 的阶段语义：
    Isaac Lab 的 `randomize_rigid_body_scale` 会修改 USD `xformOp:scale`，上游
    明确要求只在 simulation playing 前执行。因此第一版采用 startup / usd-time
    离散 scale bucket：每个 env 在启动时绑定一个 object scale bucket，episode
    reset 只根据该 bucket 查 cache，不在 reset 中改变 mesh scale。

hand 相关 event 的分解：
    - hand topology / articulation scale 不属于 event；它们来自 `assets` 生成的
      hand bundle 或 multi-asset spawner，不能在 reset 中偷偷变化；
    - hand joint **state** 在 cache reset 下由 cache entry 控制，无 cache 消融
      才使用随机 joint reset；
    - hand dynamics DR 与 cache 基本正交，推荐 startup 采样：link material / mass
      / CoM、actuator stiffness / damping、joint friction / armature；
    - 第一版暂缓 joint limit DR、collider offset DR、fixed tendon DR、interval 外力，
      因为它们会更直接改变可达域、接触 margin 或扰动课程，容易和 grasp cache
      稳定性验证语义混淆。

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

TODO:
    后续实现 `reset_from_grasp_cache(...)` 时，应只消费
    `tasks/gm/grasp_cache` 的 store / sampler 契约；不要在这里扫描 asset bank、
    决定 train/heldout split，或在线运行 grasp generator。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

GmResetMode = Literal["grasp_cache", "random_joint_object"]
GmObjectScaleMode = Literal["startup_discrete_bucket", "nominal_only"]
GmPhysicsDrPhase = Literal["startup", "reset_light_ablation", "disabled"]
GmHandOrientationMode = Literal["disabled", "roll", "pitch", "yaw", "so3"]
GmHandOrientationReferenceMode = Literal["anchor", "current"]


@dataclass(frozen=True)
class GmEventDesign:
    r"""Lightweight contract for the first `gm` event MDP design.

    这个 dataclass 不是 Isaac Lab runtime cfg；它是科研语义锚点，用于让
    `events.py`、`inhand_env_cfg.py`、grasp-cache generator 脚本和训练 manifest
    对同一套 reset / DR 决策达成一致。

    Args:
        reset_mode (GmResetMode): 主线 reset 模式。`grasp_cache` 表示采样
            $(q,T^h_o)$；`random_joint_object` 只用于 no-cache ablation。
        object_scale_mode (GmObjectScaleMode): object scale 的阶段语义。第一版
            `startup_discrete_bucket` 表示启动时离散分桶，reset 时只查对应 cache。
        physics_dr_phase (GmPhysicsDrPhase): 物理 DR 默认阶段。第一版以 startup
            为主，避免 episode reset 中频繁 CPU setter 造成吞吐与稳定性风险。
        cache_excludes_object_pose_dr (bool): cache reset 是否排斥独立 object pose DR。
        cache_controls_hand_joint_state (bool): cache reset 是否接管 hand joint state。
    """

    reset_mode: GmResetMode = "grasp_cache"  # 主线：经验稳定 grasp 分布，而非随机初态
    object_scale_mode: GmObjectScaleMode = "startup_discrete_bucket"  # scale 是几何分桶，不是 reset 噪声
    physics_dr_phase: GmPhysicsDrPhase = "startup"  # mass/friction/PD 等默认启动时采样
    cache_excludes_object_pose_dr: bool = True  # cache entry 已包含 object pose 分布
    cache_controls_hand_joint_state: bool = True  # cache entry 同时写 hand $q$ 与 object $T^h_o$


DEFAULT_GM_EVENT_DESIGN = GmEventDesign()


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


def simple_no_cache_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_position_range: tuple[float, float] = (-0.05, 0.05),
    robot_velocity_range: tuple[float, float] = (0.0, 0.0),
    object_pose_range: dict[str, tuple[float, float]] | None = None,
    object_velocity_range: dict[str, tuple[float, float]] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    r"""无 Grasp Cache 的最小 reset，用于第一阶段 smoke / 短训练。

    该函数不是主线长期 reset 分布。它只在 Grasp Cache 暂后时给 GM teacher
    一个可运行初态，使我们能先验证：generated hand 能加载、action/obs/reward
    维度闭合、rl_games rollout 能完成。

    reset 分布：

    $$
    q_0 = q_{home} + \epsilon_q,
    \qquad \epsilon_q \sim \mathcal{U}(q_{min}, q_{max}),
    $$

    object 则在默认掌心附近做厘米级平移扰动与小角度姿态扰动。后续接入
    Grasp Cache 后，应禁用该函数，改由 cache 写入稳定的 $(q,T_o^h)$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        env_ids (torch.Tensor): 需要 reset 的环境 id。
        robot_position_range (tuple[float, float]): hand joint home pose 附近的随机偏移范围，单位 rad。
        robot_velocity_range (tuple[float, float]): hand joint velocity 随机范围，单位 rad/s。
        object_pose_range (dict[str, tuple[float, float]] | None): object root pose 扰动范围。
        object_velocity_range (dict[str, tuple[float, float]] | None): object root velocity 扰动范围。
        robot_cfg (SceneEntityCfg): robot articulation 配置。
        object_cfg (SceneEntityCfg): object rigid body 配置。
    """

    if object_pose_range is None:
        object_pose_range = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    if object_velocity_range is None:
        object_velocity_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}

    # 手关节 reset 使用 IsaacLab 通用实现：围绕 default_joint_pos 做小扰动并自动 clamp 到 soft limit。
    isaac_mdp.reset_joints_by_offset(
        env,
        env_ids,
        position_range=robot_position_range,
        velocity_range=robot_velocity_range,
        asset_cfg=robot_cfg,
    )

    # 物体 reset 使用 IsaacLab 通用实现：相对 default_root_state 做轻扰动，保证第一版初态仍在掌心附近。
    isaac_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range=object_pose_range,
        velocity_range=object_velocity_range,
        asset_cfg=object_cfg,
    )

    # 记录 reset anchor，供 `object_out_of_hand` 判断 object 是否离开初始手内区域。
    object_asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    anchor_w = object_asset.data.root_pos_w[env_ids].clone()  # `[K,3]`，刚写入 sim 的 object world position
    if not isinstance(getattr(env, "_gm_object_reset_anchor_w", None), torch.Tensor):
        env._gm_object_reset_anchor_w = object_asset.data.root_pos_w.clone()  # `[B,3]`，初始化全量 anchor buffer
    env._gm_object_reset_anchor_w[env_ids] = anchor_w  # 只更新 reset 的 env

    # 同步 command 内部目标：EventManager reset 早于 CommandManager reset 时此处无影响；若顺序改变，
    # command reset 也会重新采样 goal，因此这里不直接操作 command term，避免 event/command 双写。
    _ = math_utils  # 保留数学依赖锚点；后续如改为直接写 $T_o^h$ 会在本函数使用 SE(3) 工具。


__all__ = [
    "DEFAULT_GM_EVENT_DESIGN",
    "DEFAULT_HAND_ORIENTATION_RESET_CFG",
    "GmEventDesign",
    "GmHandOrientationMode",
    "GmHandOrientationReferenceMode",
    "GmObjectScaleMode",
    "GmPhysicsDrPhase",
    "GmResetMode",
    "HandOrientationResetCfg",
    "simple_no_cache_reset",
]
