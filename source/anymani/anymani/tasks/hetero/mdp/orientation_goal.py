r"""姿态目标主导任务的可复用配置装配，训练与固定评价消费同一数学定义。

奖励采用实际每策略步单位：倒数核1/(theta+0.1)或指数核1/(exp(4*theta)+0.1)，合格事件+250。
目标只按角度推进，2.5cm控制奖金资格，7cm/45度仍定义物理失败。
训练开启逐环境位置ADR，固定评价使用相同任务但关闭初态扰动。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab.managers import RewardTermCfg

from . import rewards
from .adr import HeterogeneousAdrCfg, ObjectPositionAdrCfg


@dataclass(frozen=True)
class OrientationGoalCfg:
    r"""已确认的冷启动任务配置；数值分别带角度、长度、时间或单步奖励语义。"""

    weight: float = 1.0  # 实际单步orientation reward系数。
    epsilon_rad: float = 0.1  # 倒数核平滑参数，不是成功阈值。
    kernel: str = "inverse"  # inverse或exponential，写入身份并由固定评价读取。
    exponential_slope_rad_inv: float = 4.0  # 指数核k，rad^{-1}；theta按rad输入。
    exponential_denominator_epsilon: float = 0.1  # 指数分母的无量纲偏移，与epsilon_rad分别定义。
    angle_tolerance_rad: float = 0.2  # 对齐用户旧LEAP合格角度容差。
    position_tolerance_m: float = 0.025  # 只控制合格奖金，目标推进不受它阻塞。
    goal_bonus: float = 250.0  # 每个合格目标仅一次。
    failure_penalty: float = -20.0  # 旋转诱导阶段可承担的失败代价。
    non_tip_penalty: float = 0.0  # 不把任意指腹接触预设为坏接触。
    reference_seconds: float = 30.0  # 主任务固定评价窗口。
    target_turns_min: float = 1.0
    target_turns_max: float = 2.0  # 调整软速度带；不硬截正常瞬时换接触速度。

    def to_dict(self) -> dict[str, Any]:
        r"""将实际数学配置写入训练身份。"""
        return asdict(self)


def configure_orientation_goal(env_cfg: Any, cfg: OrientationGoalCfg, *, training: bool, adr: HeterogeneousAdrCfg) -> None:
    r"""显式替换KD、设置姿态序列与位置ADR，保持其他已配置采样/控制参数。"""
    env_cfg.rewards.pose_keypoint = None  # 不在旧KD名称下偷换公式。
    # 两个核消费同一theta和同一单步单位；其余任务事件、位置与终止配置共享。
    if cfg.kernel == "inverse":
        env_cfg.rewards.orientation_tracking = RewardTermCfg(
            func=rewards.track_orientation_inv_l2, weight=cfg.weight,
            params={'command_name': 'goal_pose', 'rot_eps': cfg.epsilon_rad},
        )  # 实际单步w/(theta+epsilon_rad)。
    elif cfg.kernel == "exponential":
        env_cfg.rewards.orientation_tracking = RewardTermCfg(
            func=rewards.track_orientation_exponential, weight=cfg.weight,
            params={'command_name': 'goal_pose', 'slope_rad_inv': cfg.exponential_slope_rad_inv,
                    'denominator_epsilon': cfg.exponential_denominator_epsilon},
        )  # 实际单步w/(exp(k*theta)+epsilon)，峰值不重标度。
    else:
        raise ValueError(f"Unknown orientation kernel: {cfg.kernel}")  # 未声明的数学形式不参与训练。
    env_cfg.rewards.goal_success.weight = cfg.goal_bonus
    env_cfg.rewards.failure.weight = cfg.failure_penalty
    env_cfg.rewards.bad_finger_non_tip_contact.weight = cfg.non_tip_penalty
    env_cfg.rewards.joint_pose_anchor.weight = 0.0
    env_cfg.rewards.speed_band.params.update(
        speed_min_rad_s=2 * math.pi * cfg.target_turns_min / cfg.reference_seconds,
        speed_max_rad_s=2 * math.pi * cfg.target_turns_max / cfg.reference_seconds,
    )  # 与1–2圈/30秒目标一致，取代旧0.6–0.833rad/s参考。
    command = env_cfg.commands.goal_pose
    command.orientation_only_advance = True
    command.orientation_success_threshold_rad = cfg.angle_tolerance_rad
    command.position_success_threshold_m = cfg.position_tolerance_m
    command.goal_reference = 'previous_goal'
    command.adr_reference_seconds = cfg.reference_seconds
    env_cfg.adr = adr if training else HeterogeneousAdrCfg(object_position=ObjectPositionAdrCfg(enabled=False))
