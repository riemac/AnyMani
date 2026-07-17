r"""GM tactile/contact sensor observation terms。

本模块按传感模态而非具体任务组织 observation：

- `fingertip_contact_force` 与 `fingertip_contact_binary` 直接读取当前 ContactSensor snapshot；
- `*_ema` terms 读取 `GmTactileContactState` 的 policy-rate shared snapshot，使 actor、critic、reward
  和 diagnostics 在同一 policy step 使用完全相同的滤波后接触证据。

瞬时 contact 与 EMA contact 是两种明确不同的测量语义，因此函数名显式携带 `ema`。力向量可表达在
hand semantic frame `{h}` 或 env/world 轴 `{e}`；EMA terms 当前输出 force magnitude 或 threshold bits，
不携带方向。传感器顺序继承 hand sidecar，不在 observation 内按名称重新排序。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ...contact_sensors import sensor_total_force_w
from ..tactile_contact_state import GmTactileContactState, get_tactile_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FrameName = Literal["h", "e"]
r"""Contact force vector 的输出坐标轴：hand semantic `{h}` 或 env/world `{e}`。"""


def _semantic_R_ha_tensor(env: ManagerBasedRLEnv, semantic_R_ha: tuple[float, ...]) -> torch.Tensor:
    r"""把 row-major $R_{ha}$ 配置转换为 env-device `[3,3]` tensor。"""

    return torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)  # $v^h=R_{ha}v^a$


def fingertip_contact_force(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    frame: FrameName = "h",
) -> torch.Tensor:
    r"""读取各 fingertip 对 object 的瞬时总接触力。

    `frame="h"` 时执行：
    $$
    F_k^h=R_{ha}R_{aw}F_k^w,
    $$
    得到固定在整只手上的语义轴，而不是随单根 fingertip link 快速旋转的 sensor-local 轴。
    `frame="e"` 保留 world/env 轴向，用于坐标表征消融。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (tuple[str, ...]): fingertip sensors 的 canonical sidecar order。
        robot_cfg (SceneEntityCfg): hand articulation root pose source。
        semantic_R_ha (tuple[float, ...]): raw asset `{a}` 到 hand semantic `{h}` 的旋转矩阵。
        frame (FrameName): 输出轴，`"h"` 或 `"e"`。

    Returns:
        torch.Tensor: 形状 `[B,3K]` 的 force vectors，单位 N。
    """

    robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root orientation $R_{wa}$
    R_ha = _semantic_R_ha_tensor(env, semantic_R_ha)  # `{a}->{h}` 静态语义标定
    forces = []  # 每个 sensor 一项 `[B,3]`，最终严格按 `sensor_names` 拼接
    for sensor_name in sensor_names:
        force_w = sensor_total_force_w(env, sensor_name)  # 瞬时 world-frame total force，`[B,3]`，N
        if frame == "e":
            forces.append(force_w)  # world/env 轴不做旋转
            continue
        if frame != "h":
            raise ValueError(f"Unsupported frame for fingertip_contact_force: {frame}.")
        force_a = math_utils.quat_apply_inverse(robot_asset.data.root_quat_w, force_w)  # $R_{aw}F^w$
        forces.append(force_a @ R_ha.T)  # row-vector form of $F^h=R_{ha}F^a$
    return torch.cat(forces, dim=-1)  # `[B,3K]`，单位 N


def fingertip_contact_binary(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float = 0.2,
) -> torch.Tensor:
    r"""对瞬时 fingertip force magnitude 做 threshold，返回 `[B,K]` binary channels。

    该函数不使用 EMA；需要 policy-rate filtered bits 时应使用 `tip_contact_bits_ema`。
    """

    contact_bits = []  # 每个 fingertip 一列 `[B]`
    for sensor_name in sensor_names:
        force_w = sensor_total_force_w(env, sensor_name)  # `[B,3]`，N
        contact_bits.append((torch.linalg.norm(force_w, dim=-1) > float(force_threshold)).float())
    return torch.stack(contact_bits, dim=-1)  # `[B,K]`，无量纲 0/1


def _shared_contact_state(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float,
    force_threshold: float,
) -> GmTactileContactState:
    r"""取得当前 policy step 唯一的 shared EMA contact snapshot。"""

    return get_tactile_contact_state(
        env,
        fingertip_sensor_names,
        finger_non_tip_sensor_names,
        palm_sensor_name,
        ema_alpha,
        force_threshold,
    )


def tip_contact_bits_ema(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""返回 EMA force magnitude threshold 后的 fingertip bits，形状 `[B,K_{tip}]`。"""

    state = _shared_contact_state(
        env, fingertip_sensor_names, finger_non_tip_sensor_names, palm_sensor_name, ema_alpha, force_threshold
    )
    return state.tip_bits.float()  # actor-facing 0/1 channels，canonical fingertip order


def tip_force_magnitude_ema(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""返回各 fingertip 的 EMA force magnitude，形状 `[B,K_{tip}]`，单位 N。"""

    state = _shared_contact_state(
        env, fingertip_sensor_names, finger_non_tip_sensor_names, palm_sensor_name, ema_alpha, force_threshold
    )
    return state.tip_force_ema  # magnitude only，不包含 vector direction


def palm_force_magnitude_ema(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""返回 neutral palm support 的 EMA force magnitude，形状 `[B,1]`，单位 N。"""

    state = _shared_contact_state(
        env, fingertip_sensor_names, finger_non_tip_sensor_names, palm_sensor_name, ema_alpha, force_threshold
    )
    return state.palm_force_ema  # palm 是合法支撑 role，不进入 finger non-tip bad-contact bits


def finger_non_tip_contact_bits_ema(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: tuple[str, ...],
    finger_non_tip_sensor_names: tuple[str, ...],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""返回 finger non-tip EMA contact bits，形状 `[B,K_{non-tip}]`，显式排除 palm。"""

    state = _shared_contact_state(
        env, fingertip_sensor_names, finger_non_tip_sensor_names, palm_sensor_name, ema_alpha, force_threshold
    )
    return state.finger_non_tip_bits.float()  # 0/1 attribution channels，canonical non-tip order


__all__ = [
    "finger_non_tip_contact_bits_ema",
    "fingertip_contact_binary",
    "fingertip_contact_force",
    "palm_force_magnitude_ema",
    "tip_contact_bits_ema",
    "tip_force_magnitude_ema",
]
