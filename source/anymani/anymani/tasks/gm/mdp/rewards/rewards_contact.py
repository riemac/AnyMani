r"""Contact reward terms for GM in-hand manipulation.

DONE(contact reward): good fingertip contact 使用二值 $n_{tip}\ge k$，默认 `min_contacts=2`；
bad non-tip contact 通过 cfg 显式传入 sensor names，reward 不猜 asset schema。sensor names
由 `gm.contact_sensors` 从 hand sidecar 自动生成。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from ...contact_sensors import sensor_contact_indicator
from .rewards_common import curriculum_gain


def good_fingertip_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    min_contacts: int = 2,
    force_threshold: float = 1.0,
    use_curriculum: bool = True,
    lambda_floor: float = 0.05,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Good Contact reward：鼓励至少 `min_contacts` 个指尖与物体接触。

    对齐 AnyRotate Appendix B 的二值接触项：
    $$
    r_{gc}=\begin{cases}1,& n_{tip-contact}\ge k\\0,&\text{otherwise}\end{cases}.
    $$

    默认 `lambda_floor=0.05`，含义是训练一开始也给弱多指接触提示，但不会像 full contact
    reward 那样压过重定向主任务。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (Sequence[str]): 指尖 ContactSensor 名称列表。
        min_contacts (int): 至少接触的指尖数，默认 2。
        force_threshold (float): 判断接触的力阈值，单位 N。
        use_curriculum (bool): 是否乘以 adaptive reward curriculum 系数。
        lambda_floor (float): curriculum 早期下限，默认 0.05。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: reward，形状 `[num_envs]`。
    """

    # 逐个显式 sensor 统计是否接触；reward 不依赖 hand topology / link metadata 自动推断。
    contact_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)  # `[B]`，接触指尖数
    for sensor_name in sensor_names:
        contact_count += sensor_contact_indicator(env, sensor_name, force_threshold).int()  # 每个 sensor 贡献 0/1

    reward = (contact_count >= int(min_contacts)).float()  # `[B]`，$r_{gc}$
    if use_curriculum:
        reward = reward * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)  # $\lambda_{gc}r_{gc}$
    return reward


def bad_non_tip_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    force_threshold: float = 0.5,
    use_curriculum: bool = True,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Bad Contact penalty indicator：检测任意非指尖部位是否接触物体。

    对齐 AnyRotate Appendix B 的非指尖接触惩罚项：
    $$
    r_{bc}=\begin{cases}1,& n_{non-tip-contact}>0\\0,&\text{otherwise}\end{cases}.
    $$

    本函数返回正值 indicator，实际惩罚由 `RewardsCfg` 中的负权重实现。默认
    `lambda_floor=0`，即早期不惩罚 palm / link 辅助接触，避免策略尚未学会重定向前
    被过早约束到狭窄行为流形。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (Sequence[str]): 非指尖 ContactSensor 名称列表。
        force_threshold (float): 判断接触的力阈值，单位 N。
        use_curriculum (bool): 是否乘以 adaptive reward curriculum 系数。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: penalty indicator，形状 `[num_envs]`。
    """

    # 统计是否存在任何非指尖部位接触；只要有一个 sensor 超阈值，就触发 bad contact。
    any_bad_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)  # `[B]`，是否有非指尖接触
    for sensor_name in sensor_names:
        any_bad_contact |= sensor_contact_indicator(env, sensor_name, force_threshold)  # OR 聚合所有 non-tip sensors

    penalty = any_bad_contact.float()  # `[B]`，$r_{bc}$，外部配置负 weight 后成为惩罚
    if use_curriculum:
        penalty = penalty * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)  # $\lambda_{bc}r_{bc}$
    return penalty


__all__ = ["bad_non_tip_contact", "good_fingertip_contact"]
