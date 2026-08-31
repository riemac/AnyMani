r"""异构逐JOINT策略的低维时序observation terms。

每个JOINT frame只含部署期可用的当前$q$、target $u$、上一policy action与owner fingertip contact：

$$
y_{t,j}=[q_{t,j}/\pi,u_{t,j}/\pi,a_{t-1,j}^{policy},c_{t,f(j)}]\in\mathbb R^4.
$$

函数返回`[B,16,4]`当前帧；IsaacLab ObservationTerm history负责形成oldest-to-latest
`[B,30,16,4]`，从而partial reset与前缀填充沿用manager统一生命周期。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .observations_state import joint_pos_raw, joint_target, last_action
from .observations_tactile import tip_contact_bits_ema

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def per_joint_policy_frame(
    env: ManagerBasedRLEnv,
    *,
    asset_cfg: SceneEntityCfg,
    action_name: str,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""构造canonical depth-major `[B,16,4]`逐JOINT当前帧。

    Contact输入顺序固定为`index,middle,ring,thumb`，canonical joint轴按
    `[depth0×4 fingers, ..., depth3×4 fingers]`排列，因此同一finger的TIP bit复制到它的四个
    depth slots。Inactive slots已由state/action terms置零，返回前再用runtime mask保持contact为零。

    Args:
        env (ManagerBasedRLEnv): heterogeneous GM环境。
        asset_cfg (SceneEntityCfg): canonical 16-JOINT顺序配置。
        action_name (str): policy-step target action term名称。
        fingertip_sensor_names (tuple[str, ...]): 四个TIP sensors，canonical PhysX finger order。
        finger_non_tip_sensor_names (tuple[str, ...]): shared EMA contact state的19个finger non-tip sensors。
        palm_sensor_name (str): neutral palm support sensor。
        ema_alpha (float): policy-rate force EMA系数。
        force_threshold (float): binary contact threshold，单位N。

    Returns:
        torch.Tensor: `[B,16,4]`，前三通道无量纲，末通道为0/1。
    """

    q_unit = joint_pos_raw(env, asset_cfg) / torch.pi  # `[B,16]`，physical rad→无量纲
    target_unit = joint_target(env, action_name) / torch.pi  # `[B,16]` recurrent PD target
    previous_action = last_action(env, action_name)  # `[B,16]`，wrapper-clamped policy action
    tip_bits = tip_contact_bits_ema(
        env,
        fingertip_sensor_names,
        finger_non_tip_sensor_names,
        palm_sensor_name,
        ema_alpha,
        force_threshold,
    )  # `[B,4]` index/middle/ring/thumb
    if q_unit.shape[1] != 16 or tip_bits.shape[1] != 4:
        raise RuntimeError(
            f"per-joint policy frame requires 16 joints and 4 tips, got q={tuple(q_unit.shape)} tips={tuple(tip_bits.shape)}"
        )

    # `[B,1,4]→[B,4 depths,4 fingers]→[B,16]`严格匹配canonical depth-major JOINT axis。
    joint_contact = tip_bits.unsqueeze(1).expand(-1, 4, -1).reshape(env.num_envs, 16)
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if isinstance(active_mask, torch.Tensor) and active_mask.shape == joint_contact.shape:
        joint_contact = joint_contact * active_mask.to(dtype=joint_contact.dtype)  # ghost history无contact泄漏
    return torch.stack((q_unit, target_unit, previous_action, joint_contact), dim=-1)  # `[B,16,4]`


__all__ = ["per_joint_policy_frame"]
