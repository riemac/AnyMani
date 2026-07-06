# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的事件函数"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    r"""将 EventManager 传入的 env ids 统一为 GPU long tensor。"""

    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def reset_adr_episode_length(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_episode_length_s: float = 20.0,
) -> None:
    r"""按 LEAP 官方随机 horizon 重采样每个 env 的 timeout。

    官方 LEAP 不是让所有并行环境都用固定 120s episode，而是在每次 reset 时采样：
    $$
    T_i\sim U[T_{\min}, T_{\max}],\qquad T_{\min}=20s,\ T_{\max}=120s.
    $$
    在 ManagerBasedRLEnv 中 ``env.max_episode_length`` 已由 ``episode_length_s`` 转成 policy steps，
    因此这里维护一个同单位的 per-env buffer，供 ``adr_randomized_time_out`` 使用。
    """

    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "leap_adr_episode_lengths"):
        env.leap_adr_episode_lengths = torch.full(
            (env.num_envs,), env.max_episode_length, device=env.device, dtype=torch.long
        )

    min_steps = max(1, int(min_episode_length_s / env.step_dt))
    env.leap_adr_episode_lengths[env_ids] = torch.randint(
        min_steps,
        env.max_episode_length + 1,
        (env_ids.numel(),),
        device=env.device,
        dtype=torch.long,
    )


def reset_adr_object_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""使用 LEAP ADR 当前档位重置 object pose。

    当前档位 $k$ 给出位置/姿态扰动半宽：
    $$
    p_x'=p_{x,0}+\epsilon_x w_x(k),\quad p_y'=p_{y,0}+\epsilon_y w_y(k),\quad
    \epsilon_x,\epsilon_y\sim U[-1,1].
    $$
    姿态只扰动 roll/pitch，保留官方 ``z_rotation=0``，避免 reset 时随机 yaw 改变固定 z 轴
    continuous-rotation 任务的相位结构。
    """

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    obj: RigidObject = env.scene[asset_cfg.name]
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0

    x_width = float(getattr(env, "leap_adr_object_x_width", 0.0))
    y_width = float(getattr(env, "leap_adr_object_y_width", 0.0))
    if x_width > 0.0 or y_width > 0.0:
        pos_noise = sample_uniform(-1.0, 1.0, (env_ids.numel(), 2), device=env.device)
        root_state[:, 0] += pos_noise[:, 0] * x_width
        root_state[:, 1] += pos_noise[:, 1] * y_width

    roll_width = float(getattr(env, "leap_adr_object_x_rot", 0.0))
    pitch_width = float(getattr(env, "leap_adr_object_y_rot", 0.0))
    yaw_width = float(getattr(env, "leap_adr_object_z_rot", 0.0))
    if roll_width > 0.0 or pitch_width > 0.0 or yaw_width > 0.0:
        rpy_noise = sample_uniform(-1.0, 1.0, (env_ids.numel(), 3), device=env.device)
        roll = rpy_noise[:, 0] * roll_width
        pitch = rpy_noise[:, 1] * pitch_width
        yaw = rpy_noise[:, 2] * yaw_width
        noise_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 3:7] = math_utils.quat_mul(noise_quat, root_state[:, 3:7])

    obj.write_root_pose_to_sim(root_state[:, :7], env_ids)
    obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids)


def reset_adr_robot_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="a_.*"),
) -> None:
    r"""使用 LEAP ADR 当前档位重置手关节。

    该项替代 AnyMani baseline 中一开始就很宽的 ``(-0.2,0.2)`` reset 噪声，改为官方课程：
    $$
    q_0'=q_{\mathrm{pregrasp}}+\epsilon_q\sigma_q(k),\quad \sigma_q:0\to0.05.
    $$
    """

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = robot.data.default_joint_pos[env_ids][:, joint_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids][:, joint_ids].clone()

    # 为 official target-buffer action 缓存“本次 reset 真正写入仿真的 joint target 初值”。
    # 这样动作项 reset 时可以把 `prev_targets = cur_targets = q_0'`，而不是退回默认 pregrasp。
    if not hasattr(env, "leap_official_reset_joint_pos"):
        env.leap_official_reset_joint_pos = robot.data.default_joint_pos[:, joint_ids].clone()

    pos_width = float(getattr(env, "leap_adr_joint_pos_noise", 0.0))
    vel_width = float(getattr(env, "leap_adr_joint_vel_noise", 0.0))
    if pos_width > 0.0:
        joint_pos += sample_uniform(-1.0, 1.0, joint_pos.shape, device=env.device) * pos_width
        limits = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids, :]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
    if vel_width > 0.0:
        joint_vel += sample_uniform(-1.0, 1.0, joint_vel.shape, device=env.device) * vel_width

    env.leap_official_reset_joint_pos[env_ids] = joint_pos
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def reset_adr_wrench_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    probability: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    r"""为每个新 episode 采样是否启用 object wrench，并立即刷新一次外力。

    LEAP 官方 object wrench 不是每个 episode 都开启，而是先为每个环境采样一个 episode-level
    Bernoulli gate：
    $$
    b_i\sim\mathrm{Bernoulli}(p),\qquad p=0.5.
    $$
    其中 $b_i=1$ 表示环境 $i$ 的当前 episode 会受到分段常值外力扰动；$b_i=0$ 表示该
    episode 不施加 object wrench。这个 gate 在 episode 内保持不变，具体 force/torque
    数值再由 ``apply_adr_object_wrench`` 按 interval event 周期重采样。

    Args:
        env: ManagerBasedRLEnv 运行时对象，承载 ``leap_adr_apply_wrench`` episode gate buffer。
        env_ids: 当前 reset 的环境集合，形状 $[N_r]$。
        probability: Bernoulli 概率 $p$；官方默认 $p=0.5$。
        asset_cfg: 被施加 wrench 的刚体对象，默认是任务中的 cube/object。
    """

    # 把 EventManager 传入的 env ids 规范成 GPU long tensor，记为 reset 集合 $\mathcal{E}_t$。
    env_ids = _resolve_env_ids(env, env_ids)

    # 首次 reset 时创建 episode-level gate buffer，形状 $[N_{env}]$，每个元素对应一个并行环境。
    if not hasattr(env, "leap_adr_apply_wrench"):
        env.leap_adr_apply_wrench = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 采样 $b_i\sim\mathrm{Bernoulli}(p)$；这里用 uniform 阈值实现：$b_i=\mathbf{1}[u_i\le p]$。
    env.leap_adr_apply_wrench[env_ids] = torch.rand(env_ids.numel(), device=env.device) <= probability

    # reset 当下立即刷新一次 wrench，避免 episode 前 90 policy steps 完全没有外力样本。
    apply_adr_object_wrench(env, env_ids=env_ids, asset_cfg=asset_cfg)


def apply_adr_object_wrench(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    torsional_radius: float = 0.0,
) -> None:
    r"""按 LEAP 官方 object wrench 语义施加分段常值外力。

    当前 ADR 档位给出最大线加速度 $a_{\max}(k)$，官方范围为：
    $$
    a_{\max}(k):0.5\rightarrow5.0\ \mathrm{m/s^2}.
    $$
    对环境 $i$ 中质量为 $m_i$ 的 object，先把加速度上界换算成力上界：
    $$
    F_{\max,i}=m_i a_{\max}(k).
    $$
    随后逐轴采样分段常值外力：
    $$
    \mathbf f_i=F_{\max,i}\boldsymbol\epsilon_i,
    \qquad
    \boldsymbol\epsilon_i\sim U([-1,1]^3).
    $$
    若设置 torsional radius $\rho$，对应 torque 上界为：
    $$
    \tau_{\max,i}=m_i a_{\max}(k)\rho.
    $$
    当前 AnyMani ADR 复刻官方默认 ``torsional_radius=0``，因此 $\tau_{\max,i}=0$，wrench
    只注入平移外力，不直接注入随机扭矩。

    Args:
        env: ManagerBasedRLEnv 运行时对象，读取 ``leap_adr_max_linear_accel`` 和 episode gate。
        env_ids: 需要刷新 wrench 的环境集合；``None`` 表示全部环境。
        asset_cfg: 被施加 wrench 的刚体对象，默认是任务中的 cube/object。
        torsional_radius: 力臂半径 $\rho$，用于把线加速度上界换成 torque 上界。
    """

    # 规范化目标环境集合；interval event 可能传入 None，此时刷新全部并行环境。
    env_ids = _resolve_env_ids(env, env_ids)

    # 读取被扰动物体及 body 数；RigidObject 通常只有一个 body，但这里保持 SceneEntityCfg 兼容。
    obj: RigidObject = env.scene[asset_cfg.name]
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else obj.num_bodies

    # 读取当前 ADR 档位插值得到的 $a_{\max}(k)$，单位 m/s^2；第 0 档为 0.5。
    max_accel = float(getattr(env, "leap_adr_max_linear_accel", 0.5))

    # PhysX mass buffer 给出 $m_i$；形状通常为 $[N,1]$，后续广播到三轴 force。
    masses = obj.root_physx_view.get_masses().to(device=env.device)[env_ids]

    # $F_{\max}=m a_{\max}$，形状扩展为 $[N,B,1]$ 以逐 body/axis 采样。
    max_force = (masses * max_accel).unsqueeze(-1)

    # $\tau_{\max}=m a_{\max}\rho$；当前 $\rho=0$，因此 torque 样本会严格为零。
    max_torque = (masses * max_accel * torsional_radius).unsqueeze(-1)

    # 逐轴 uniform 采样 $\boldsymbol\epsilon\sim U([-1,1]^3)$ 并缩放到力上界。
    forces = max_force * sample_uniform(-1.0, 1.0, (env_ids.numel(), num_bodies, 3), device=env.device)

    # 逐轴 uniform 采样 torque；默认 torsional_radius=0 时该张量数值全为零。
    torques = max_torque * sample_uniform(-1.0, 1.0, (env_ids.numel(), num_bodies, 3), device=env.device)

    # 如果 reset 阶段已采样 episode-level gate $b_i$，则用 $b_i=0$ 的环境清零 wrench。
    if hasattr(env, "leap_adr_apply_wrench"):
        gate = env.leap_adr_apply_wrench[env_ids].view(-1, 1, 1)  # $b_i$，形状 $[N,1,1]$，广播到 body/axis。
        forces = torch.where(gate, forces, torch.zeros_like(forces))  # $b_i\mathbf f_i$。
        torques = torch.where(gate, torques, torch.zeros_like(torques))  # $b_i\boldsymbol\tau_i$。

    # 写入永久 wrench composer；该外力会持续到下一次 interval/reset event 刷新。
    obj.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def resample_adr_material_buckets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str,
    range_attr: str,
) -> None:
    r"""显式重采 material buckets，并立即把材料分配到 reset env。

    IsaacLab 的 ``randomize_rigid_body_material`` 会在 term 初始化时采样 ``material_buckets``。
    如果 ADR 只改 cfg 里的 friction/restitution range，bucket 本身不会自动更新。这里在**范围签名变化**
    时刷新 ``term.func.material_buckets``，让材料 ADR 真实生效，同时避免每个 reset 都生成新的
    PhysX material。PhysX 的 unique material 上限是 64K，连续重采样会把 scene 推爆；按档位缓存后，
    同一 ADR 档位内只重新分配 bucket id，不再累积新材质。
    """

    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except ValueError:
        return
    term = term_cfg.func
    if not hasattr(term, "material_buckets"):
        return

    ranges_dict = getattr(env, range_attr)
    ranges = torch.tensor(
        [ranges_dict["static"], ranges_dict["dynamic"], ranges_dict["restitution"]],
        device="cpu",
        dtype=torch.float32,
    )
    num_buckets = int(term_cfg.params.get("num_buckets", term.material_buckets.shape[0]))
    signature = (
        tuple(float(v) for v in ranges[:, 0].tolist()),
        tuple(float(v) for v in ranges[:, 1].tolist()),
        num_buckets,
        bool(term_cfg.params.get("make_consistent", False)),
    )

    if getattr(term, "_adr_bucket_signature", None) != signature:
        buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        if term_cfg.params.get("make_consistent", False):
            buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
        term.material_buckets = buckets
        term._adr_bucket_signature = signature
    term(env, env_ids, **term_cfg.params)
