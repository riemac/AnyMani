r"""canonical runtime 的 mask、reset 与正则化纯 tensor 合同。

canonical PhysX articulation 的数组固定为 J=16。对第 b 个环境定义 m[b,j] in {0,1}，
其中 m[b,j]=0 表示 ghost slot。action、observation、joint-limit reset、机械功率、
torque/action regularization 与 PPO 概率统计都必须消费同一个 [env,joint] mask。

本文件不导入 IsaacLab，因此可用 fake env / Torch contract test 验证第一性原理；依赖
PhysX handle 的 writer 位于 events.py。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

CANONICAL_RUNTIME_DOF = 16
"""v1 canonical action/joint tensor 的固定最后一维。"""


def normalize_active_joint_mask(
    active_joint_mask: torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]],
    *,
    batch_size: int,
    dof: int = CANONICAL_RUNTIME_DOF,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""将 active mask 规范化为 [B,J] bool tensor。

    一维 mask 代表整个 batch 共享 routing；二维 mask 代表每个 env 的 asset row。
    J 不允许通过 silent padding 改变，避免 action/observation 维度与 PhysX schema 不一致。
    """

    mask = torch.as_tensor(active_joint_mask, dtype=torch.bool, device=device)
    if mask.ndim == 1:
        if mask.shape[0] != dof:
            raise ValueError(f"active_joint_mask must have {dof} joints, got {tuple(mask.shape)}")
        mask = mask.unsqueeze(0).expand(batch_size, -1).clone()
    elif mask.ndim == 2:
        if mask.shape != (batch_size, dof):
            raise ValueError(f"active_joint_mask must have shape {(batch_size, dof)}, got {tuple(mask.shape)}")
    else:
        raise ValueError(f"active_joint_mask must be rank 1 or 2, got rank {mask.ndim}")
    return mask


def mask_action(action: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    r"""把 policy action 投影到 active joint 子空间，ghost raw/processed action 恒为 0。"""

    if action.ndim != 2 or active_joint_mask.shape != action.shape:
        raise ValueError(
            f"action and active_joint_mask must share [B,J], got {tuple(action.shape)} and {tuple(active_joint_mask.shape)}"
        )
    return action * active_joint_mask.to(dtype=action.dtype)  # a_env[b,j] = m[b,j] * a_policy[b,j]


def canonical_reset_pose(
    joint_limits: torch.Tensor,
    active_joint_mask: torch.Tensor,
    q_home: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""生成 per-env reset pose：active joint 取 q_home/limit midpoint，ghost 为 0。

    limit midpoint 为 q_reset = (q_min + q_max) / 2。它只作为没有显式 q_home 时的
    保守物理初值；无论 source limit 如何，inactive slot 都直接写 q=0。
    """

    if joint_limits.ndim != 3 or joint_limits.shape[-1] != 2:
        raise ValueError(f"joint_limits must have shape [B,J,2], got {tuple(joint_limits.shape)}")
    if active_joint_mask.shape != joint_limits.shape[:2]:
        raise ValueError("active_joint_mask must match joint_limits [B,J]")
    midpoint = 0.5 * (joint_limits[..., 0] + joint_limits[..., 1])  # q_mid[b,j]，单位 rad
    if q_home is not None:
        if q_home.shape != midpoint.shape:
            raise ValueError("q_home must match joint_limits [B,J]")
        reset = torch.maximum(torch.minimum(q_home, joint_limits[..., 1]), joint_limits[..., 0])
    else:
        reset = midpoint
    return torch.where(active_joint_mask, reset, torch.zeros_like(reset))  # ghost q_reset = 0


def masked_joint_limits(joint_limits: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    r"""将 inactive joint 的 runtime limits 替换为精确 [0,0]。"""

    if joint_limits.ndim != 3 or active_joint_mask.shape != joint_limits.shape[:2]:
        raise ValueError("joint_limits must be [B,J,2] and mask must be [B,J]")
    return torch.where(
        active_joint_mask.unsqueeze(-1),
        joint_limits,
        torch.zeros_like(joint_limits),
    )  # inactive joint limit is exactly [0,0]


def masked_mean(values: torch.Tensor, active_joint_mask: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    r"""对每个 env 只按 active joints 求均值，避免不同 DOF asset 的正则量纲漂移。"""

    if values.ndim != 2 or values.shape != active_joint_mask.shape:
        raise ValueError("values and active_joint_mask must share [B,J]")
    weights = active_joint_mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(eps)


def install_canonical_runtime_state(
    env: object,
    active_joint_mask: torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]],
    *,
    asset_rows: torch.Tensor | Sequence[int] | None = None,
    dof: int = CANONICAL_RUNTIME_DOF,
) -> torch.Tensor:
    r"""在 env 上安装 canonical routing state，并返回 [B,J] mask。

    asset_rows 是离散 evidence-bank row；它与 mask 一样是环境状态，而不是 policy
    网络内部的可学习 embedding。
    """

    batch_size = int(getattr(env, "num_envs"))
    device = getattr(env, "device", None)
    mask = normalize_active_joint_mask(
        active_joint_mask,
        batch_size=batch_size,
        dof=dof,
        device=device,
    )
    setattr(env, "_anymani_canonical_active_joint_mask", mask)
    if asset_rows is None:
        rows = torch.arange(batch_size, device=mask.device, dtype=torch.long)
    else:
        rows = torch.as_tensor(asset_rows, device=mask.device, dtype=torch.long)
        if rows.shape != (batch_size,):
            raise ValueError(f"asset_rows must have shape {(batch_size,)}, got {tuple(rows.shape)}")
    setattr(env, "_anymani_canonical_asset_row", rows)
    return mask


def expand_round_robin_routing(
    active_joint_mask_rows: torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]],
    asset_rows: torch.Tensor | Sequence[int],
    *,
    num_envs: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""按 IsaacLab ``MultiAssetSpawnerCfg(random_choice=False)`` 展开环境路由。

    该 spawner 对第 ``b`` 个环境选择 ``b % num_assets`` 号 prototype，因此 mask 与
    evidence row 必须使用同一模运算。不能把每个 mother 连续重复若干次，否则会把真实
    活动关节误判成 ghost，或让 policy 从错误的 evidence row 取静态几何。
    """

    masks = torch.as_tensor(active_joint_mask_rows, dtype=torch.bool, device=device)
    rows = torch.as_tensor(asset_rows, dtype=torch.long, device=device)
    if masks.ndim != 2 or masks.shape[1] != CANONICAL_RUNTIME_DOF:
        raise ValueError(f"active_joint_mask_rows must have shape [R,{CANONICAL_RUNTIME_DOF}]")
    if rows.shape != (masks.shape[0],):
        raise ValueError("asset_rows must contain one evidence row per active mask row")
    if masks.shape[0] == 0 or num_envs < 1:
        raise ValueError("round-robin routing requires at least one asset and one environment")
    selectors = torch.arange(num_envs, device=masks.device) % masks.shape[0]
    return masks[selectors], rows[selectors]


__all__ = [
    "CANONICAL_RUNTIME_DOF",
    "canonical_reset_pose",
    "expand_round_robin_routing",
    "install_canonical_runtime_state",
    "mask_action",
    "masked_joint_limits",
    "masked_mean",
    "normalize_active_joint_mask",
]
