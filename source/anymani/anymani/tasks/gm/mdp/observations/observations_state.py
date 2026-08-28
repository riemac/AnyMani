r"""GM robot/action 的时变 state observation terms。

本模块只负责从 articulation 与 ActionManager 读取当前动态状态，不决定 policy 的最终数值尺度。
例如 tactile rotation 在 config 中对 $q_t$ 与 target $u_t$ 施加 $1/\pi$，而其他对照环境可直接
消费 raw rad。当前公开语义为：

- `joint_pos_raw`: $q_t$，单位 rad；
- `joint_pos_limit_normalized`: soft-limit 归一化位置，范围 $[-1,1]$；
- `joint_vel_raw`: $\dot q_t$，单位 rad/s；
- `joint_target`: action term 当前 recurrent target $u_t$，单位 rad；
- `last_action`: 上一帧 raw policy output，无量纲；
- `last_processed_action`: action term scale/clip 后的 processed action，单位由具体 action law 定义。

所有 joint tensors 都继承 `SceneEntityCfg.joint_ids` 的 resolved order。静态 joint limits 的独立
geometry 表示仍由 `observations_geometry.py` 所有，避免在 temporal history 中重复静态形态量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_raw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""读取 hand articulation 的 raw joint position $q$。

    数学语义为：
    $$
    \mathbf{q}_t = [q_{1,t},\dots,q_{n,t}] \in \mathbb{R}^{n},
    $$
    单位为 rad，索引顺序由 `SceneEntityCfg.joint_ids` 与 action joint schema 决定。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: raw joint position，形状 `[num_envs, num_joints]`，单位 rad。
    """

    # `asset_cfg` 在 ObservationManager 初始化时会解析 joint_names → joint_ids；preserve_order=True 要求顺序一致。
    asset: Articulation = env.scene[asset_cfg.name]
    values = asset.data.joint_pos[:, asset_cfg.joint_ids]  # $q_i$，raw rad，不随 limit 归一化
    return _mask_canonical_joint_values(env, values, asset_cfg.joint_ids)  # ghost $q_i=0$


def joint_pos_limit_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""返回 IsaacLab soft-limit normalized joint position。

    对每个关节 $i$，soft limit 归一化定义为：
    $$
    q_i^{\mathrm{norm}}
      = 2\,\frac{q_i-q_i^{\min}}{q_i^{\max}-q_i^{\min}} - 1.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: limit-normalized joint position，形状 `[num_envs, num_joints]`，无量纲。
    """

    # 读取与 IsaacLab 官方 term 完全相同的 runtime soft limits；这些 limit 也是 action clamp 的物理边界。
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # $q_i$，raw joint position，单位 rad
    q_min = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]  # $q_i^{\min}$，soft lower bound，单位 rad
    q_max = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]  # $q_i^{\max}$，soft upper bound，单位 rad

    # 线性映射 $[q^{\min},q^{\max}]\mapsto[-1,1]$，公式等价于 IsaacLab `scale_transform`。
    span = (q_max - q_min).clamp_min(1.0e-8)  # ghost [0,0] 的退化区间只作为数值占位
    values = 2.0 * (q - q_min) / span - 1.0  # $q_i^{\mathrm{norm}}$，无量纲
    return _mask_canonical_joint_values(env, values, asset_cfg.joint_ids)  # inactive token 恒为 0


def joint_vel_raw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""读取 hand articulation 的 raw joint velocity $\dot q$。

    速度与位置同属 state obs 的动态本体感受项：
    $$
    \dot{\mathbf{q}}_t = [\dot q_{1,t},\dots,\dot q_{n,t}] \in \mathbb{R}^{n}.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: raw joint velocity，形状 `[num_envs, num_joints]`，单位 rad/s。
    """

    # 不减 default_joint_vel；手内操作关心当前真实角速度，而不是相对默认速度。
    asset: Articulation = env.scene[asset_cfg.name]
    values = asset.data.joint_vel[:, asset_cfg.joint_ids]  # $\dot q_i$，raw rad/s
    return _mask_canonical_joint_values(env, values, asset_cfg.joint_ids)  # ghost $\dot q_i=0$


def joint_target(
    env: ManagerBasedRLEnv,
    action_name: str = "hand_joint_pos",
) -> torch.Tensor:
    r"""读取 action term 当前 recurrent joint target $u_t$。

    对 policy-step target action，$u_t$ 是跨 physics decimation hold 的关节位置目标，单位 rad。
    它不是 raw policy action，也不是本 step 的 target increment。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        action_name (str): 暴露 `current_targets` 的 action term 名称。

    Returns:
        torch.Tensor: 当前 target，形状 `[B,N_j]`，单位 rad。

    Raises:
        RuntimeError: action term 不暴露 tensor `current_targets`。
    """

    action_term = env.action_manager.get_term(action_name)  # policy-step target state owner
    current_targets = getattr(action_term, "current_targets", None)  # $u_t$，`[B,N_j]`，rad
    if not isinstance(current_targets, torch.Tensor):
        raise RuntimeError(f"Action term '{action_name}' must expose tensor current_targets for joint_target obs.")
    return _mask_canonical_joint_values(env, current_targets, getattr(action_term, "_joint_ids", None))


def last_processed_action(
    env: ManagerBasedRLEnv,
    action_name: str = "hand_joint_pos",
) -> torch.Tensor:
    r"""读取 action term scale/clip 后的 processed action。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        action_name (str): `ActionManager` 中的 action term 名称，默认 `hand_joint_pos`。

    Returns:
        torch.Tensor: processed action，形状 `[B,A_{term}]`；单位由 action term 定义。

    Raises:
        RuntimeError: 若 action term 不暴露 `processed_actions`，说明 obs/action 合同不匹配。
    """

    # 与 `ClampedRelativeJointPositionAction` 直接耦合：processed_actions 正是 target 更新使用的 $\Delta_t$。
    action_term = env.action_manager.get_term(action_name)
    processed_actions = getattr(action_term, "processed_actions", None)  # `[B,N_j]`，rad delta
    if not isinstance(processed_actions, torch.Tensor):
        raise RuntimeError(f"Action term '{action_name}' must expose processed_actions for gm raw-rad last_action obs.")
    return _mask_canonical_joint_values(env, processed_actions, getattr(action_term, "_joint_ids", None))


def last_action(
    env: ManagerBasedRLEnv,
    action_name: str | None = None,
) -> torch.Tensor:
    r"""读取上一帧 raw policy action。

    该 term 与 `last_processed_action` 分别表达两种实验语义：

    - `last_action`: policy 网络上一帧输出的 raw action，无量纲，复刻 IsaacLab 官方；
    - `last_processed_action`: action term 经 scale / clip 后实际下发的 $\Delta q$，单位 rad。

    因此 LEAP official-subset 对照可以等价替换 `isaac_mdp.last_action`，
    而 raw-rad 主线仍可显式选择 `gm_mdp.last_processed_action`。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        action_name (str | None): action term 名称；为 `None` 时返回完整 action tensor。

    Returns:
        torch.Tensor: raw action tensor，形状由 action manager / term 决定，无量纲。
    """

    # `action_name=None` 时保持 IsaacLab 官方语义：返回 manager 拼接后的整条 raw action。
    if action_name is None:
        return _mask_canonical_joint_values(env, env.action_manager.action, None)  # `[B,A]`，ghost raw action 为 0

    # 指定 term 时读取该 term 的 `raw_actions`，而不是 `processed_actions`，以保证官方等价替换。
    action_term = env.action_manager.get_term(action_name)
    return _mask_canonical_joint_values(env, action_term.raw_actions, getattr(action_term, "_joint_ids", None))


def canonical_asset_row(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""读取 canonical evidence-bank 的离散 asset row，形状 `[B,1]`。

    该字段只作为 routing identifier；训练入口必须关闭 rl_games 的全局 input RMS，不能把
    row number 误读成有物理顺序的连续 morphology feature。
    """

    _ensure_canonical_state_from_cfg(env)
    rows = getattr(env, "_anymani_canonical_asset_row", None)
    if not isinstance(rows, torch.Tensor) or rows.shape != (env.num_envs,):
        raise RuntimeError("canonical asset row state is not initialized")
    return rows.to(dtype=torch.float32).unsqueeze(-1)


def canonical_active_joint_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""读取 canonical `[B,16]` active mask 并转成 flat observation 的 float 0/1。"""

    _ensure_canonical_state_from_cfg(env)
    mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.shape != (env.num_envs, 16):
        raise RuntimeError("canonical active joint mask is not initialized")
    return mask.to(dtype=torch.float32)


def _mask_canonical_joint_values(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    joint_ids,
) -> torch.Tensor:
    r"""按 env-level active routing mask 置零 joint tensor 的 inactive columns。"""

    _ensure_canonical_state_from_cfg(env)
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if not isinstance(active_mask, torch.Tensor):
        return values
    if joint_ids is None:
        if active_mask.shape != values.shape:
            return values
        subset_mask = active_mask
    else:
        subset_mask = active_mask[:, joint_ids]
    if subset_mask.shape != values.shape:
        return values
    return values * subset_mask.to(dtype=values.dtype)


def _ensure_canonical_state_from_cfg(env: ManagerBasedRLEnv) -> None:
    r"""在 ObservationManager probe 阶段从 startup EventTerm 参数预装 routing state。

    IsaacLab 会先调用 observation term 推断 shape，再执行 startup event；因此这里允许
    manager preparation 读取同一份 JSON-safe static routing，而不是创建第二套 mask 真源。
    """

    if isinstance(getattr(env, "_anymani_canonical_active_joint_mask", None), torch.Tensor):
        return
    events_cfg = getattr(getattr(env, "cfg", None), "events", None)
    initialize_cfg = getattr(events_cfg, "initialize_runtime", None)
    params = getattr(initialize_cfg, "params", None)
    if not isinstance(params, dict) or "active_joint_mask" not in params:
        return
    if params.get("routing_mode", "explicit") == "round_robin":
        from ..canonical_runtime import expand_round_robin_routing

        mask, rows = expand_round_robin_routing(
            params["active_joint_mask"],
            params.get("asset_rows", ()),
            num_envs=env.num_envs,
            device=env.device,
        )
        setattr(env, "_anymani_canonical_active_joint_mask", mask)
        setattr(env, "_anymani_canonical_asset_row", rows)
        return
    mask = torch.as_tensor(params["active_joint_mask"], dtype=torch.bool, device=env.device)
    if mask.shape != (env.num_envs, 16):
        raise RuntimeError(f"canonical startup mask must have shape {(env.num_envs, 16)}, got {tuple(mask.shape)}")
    setattr(env, "_anymani_canonical_active_joint_mask", mask)
    rows = torch.as_tensor(params.get("asset_rows", range(env.num_envs)), dtype=torch.long, device=env.device)
    if rows.shape != (env.num_envs,):
        raise RuntimeError(f"canonical startup asset rows must have shape {(env.num_envs,)}, got {tuple(rows.shape)}")
    setattr(env, "_anymani_canonical_asset_row", rows)


__all__ = [
    "joint_pos_limit_normalized",
    "joint_pos_raw",
    "joint_target",
    "joint_vel_raw",
    "canonical_active_joint_mask",
    "canonical_asset_row",
    "last_action",
    "last_processed_action",
]
