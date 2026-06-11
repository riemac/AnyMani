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

TODO:
    后续实现 `reset_from_grasp_cache(...)` 时，应只消费
    `tasks/gm/grasp_cache` 的 store / sampler 契约；不要在这里扫描 asset bank、
    决定 train/heldout split，或在线运行 grasp generator。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GmResetMode = Literal["grasp_cache", "random_joint_object"]
GmObjectScaleMode = Literal["startup_discrete_bucket", "nominal_only"]
GmPhysicsDrPhase = Literal["startup", "reset_light_ablation", "disabled"]


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


__all__ = [
    "DEFAULT_GM_EVENT_DESIGN",
    "GmEventDesign",
    "GmObjectScaleMode",
    "GmPhysicsDrPhase",
    "GmResetMode",
]
