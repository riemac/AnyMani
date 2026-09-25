(
    'Shared pose-goal task config for training and fixed evaluation. Per-step '
    'reward is 1/(theta+0.1) or 1/(exp(4*theta)+0.1), with a +250 qualified '
    'bonus. Advance goals by angle only; 2.5 cm gates the bonus, while 7 cm/45 '
    'deg remain physical failure limits. Train with position ADR and disable it '
    'for fixed evaluation.'
)

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab.managers import RewardTermCfg

from . import rewards
from .adr import HeterogeneousAdrCfg, ObjectPositionAdrCfg


@dataclass(frozen=True)
class OrientationGoalCfg:
    'Confirmed cold-start task config; numeric values retain angle, length, time, and per-step reward units.'

    weight: float = 1.0  # Actual per-step orientation reward weight.
    epsilon_rad: float = 0.1  # Inverse-kernel smoothing parameter, not a success threshold.
    kernel: str = "inverse"  # Select inverse or exponential and record it in run identity/evaluation.
    exponential_slope_rad_inv: float = 4.0  # Exponential-kernel k in rad^-1; theta is in rad.
    exponential_denominator_epsilon: float = 0.1  # Dimensionless exponential denominator offset; separate from epsilon_rad.
    angle_tolerance_rad: float = 0.2  # Match the legacy LEAP qualification angle tolerance.
    position_tolerance_m: float = 0.025  # Controls bonus qualification only; it does not block goal advancement.
    goal_bonus: float = 250.0  # Award each qualified goal once.
    failure_penalty: float = -20.0  # Failure cost affordable during rotation learning.
    non_tip_penalty: float = 0.0  # Do not treat arbitrary fingertip contact as bad contact.
    reference_seconds: float = 30.0  # Fixed main-task evaluation window.
    target_turns_min: float = 1.0
    target_turns_max: float = 2.0  # Tune the soft speed band; do not clip normal transient recontact speed.

    def to_dict(self) -> dict[str, Any]:
        'Record the effective mathematical configuration in run identity.'
        return asdict(self)


def configure_orientation_goal(env_cfg: Any, cfg: OrientationGoalCfg, *, training: bool, adr: HeterogeneousAdrCfg) -> None:
    (
        'Explicitly replace KD and set the pose sequence and position ADR while '
        'preserving other configured sampling/control parameters.'
    )
    env_cfg.rewards.pose_keypoint = None  # Do not change the formula under the legacy KD name.
    # Both kernels use the same theta and per-step units; task events, position, and terminations are shared.
    if cfg.kernel == "inverse":
        env_cfg.rewards.orientation_tracking = RewardTermCfg(
            func=rewards.track_orientation_inv_l2, weight=cfg.weight,
            params={'command_name': 'goal_pose', 'rot_eps': cfg.epsilon_rad},
        )  # Actual per-step reward w/(theta + epsilon_rad).
    elif cfg.kernel == "exponential":
        env_cfg.rewards.orientation_tracking = RewardTermCfg(
            func=rewards.track_orientation_exponential, weight=cfg.weight,
            params={'command_name': 'goal_pose', 'slope_rad_inv': cfg.exponential_slope_rad_inv,
                    'denominator_epsilon': cfg.exponential_denominator_epsilon},
        )  # Actual per-step reward w/(exp(k*theta) + epsilon); do not rescale the peak.
    else:
        raise ValueError(f"Unknown orientation kernel: {cfg.kernel}")  # Undeclared mathematical forms are excluded from training.
    env_cfg.rewards.goal_success.weight = cfg.goal_bonus
    env_cfg.rewards.failure.weight = cfg.failure_penalty
    env_cfg.rewards.bad_finger_non_tip_contact.weight = cfg.non_tip_penalty
    env_cfg.rewards.joint_pose_anchor.weight = 0.0
    env_cfg.rewards.speed_band.params.update(
        speed_min_rad_s=2 * math.pi * cfg.target_turns_min / cfg.reference_seconds,
        speed_max_rad_s=2 * math.pi * cfg.target_turns_max / cfg.reference_seconds,
    )  # Match the 1-2 turns per 30 seconds target; replace the old 0.6-0.833 rad/s reference.
    command = env_cfg.commands.goal_pose
    command.orientation_only_advance = True
    command.orientation_success_threshold_rad = cfg.angle_tolerance_rad
    command.position_success_threshold_m = cfg.position_tolerance_m
    command.goal_reference = 'previous_goal'
    command.adr_reference_seconds = cfg.reference_seconds
    env_cfg.adr = adr if training else HeterogeneousAdrCfg(object_position=ObjectPositionAdrCfg(enabled=False))
