r"""Contact observation terms for GM in-hand manipulation.

TODO(contact obs): 指尖触觉观测应服务 embodiment-centric 表征，而非 world-centric 表征。
本项目主线是手型泛化（embodiment generalization），不是 gravity-invariant multi-axis
rotation。接触信息未来应表达在指尖/传感器局部坐标系下，使接触语义绑定到 embodiment
自身，避免世界系手部姿态变化污染 contact 表征。

若未来引入任意手姿态的 multi-axis rotation（AnyRotate 路线），届时追加 gravity direction
in hand frame 作为额外输入即可，不在当前阶段引入该复杂度。

当前 first runnable slice 先提供两类 cheap contact obs：

- `fingertip_contact_force_w`: 每个 fingertip 对 object 的总接触力，世界系 `{w}`，形状 `[B,3K]`；
- `fingertip_contact_binary`: 每个 fingertip 是否超过力阈值，形状 `[B,K]`。

NOTE: 传感器 prim_path 不再硬编码四指名称，而是由 `gm.contact_sensors` 从 hand sidecar
自动推导 per-link `ContactSensorCfg`。filtered contact 只过滤到 `"{ENV_REGEX_NS}/object"`，
避免手指间自碰污染 teacher contact obs。

未来局部触觉规格仍保留：每个指尖 $k$ 可输出
$$
\mathbf{c}^{\text{contact}}_k =
\big[\,c_x,c_y,c_z,F_x,F_y,F_z,\|F\|\,\big]^{\{S_k\}} \in \mathbb{R}^7,
$$
其中接触点来自 IsaacLab 平均 `contact_pos_w`，无接触时 NaN 填零；力矢量从
`net_forces_w / force_matrix_w` 经 sensor pose 转到局部系。该完整局部 obs 暂后，避免在
当前 refactor 中改变张量语义。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...contact_sensors import sensor_total_force_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingertip_contact_force_w(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
) -> torch.Tensor:
    r"""读取所有 fingertip ContactSensor 的世界系接触力。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (tuple[str, ...]): 指尖传感器名称，顺序应与 hand/finger 语义顺序一致。

    Returns:
        torch.Tensor: 拼接后的接触力，形状 `[num_envs, 3 * num_sensors]`，单位 N。
    """

    forces = [sensor_total_force_w(env, sensor_name) for sensor_name in sensor_names]  # 每项 `[B,3]`
    return torch.cat(forces, dim=-1)  # `[B,3K]`，teacher critic 可直接使用的 force obs


def fingertip_contact_binary(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float = 0.2,
) -> torch.Tensor:
    r"""读取 fingertip 是否有效接触 object 的二值观测。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (tuple[str, ...]): 指尖传感器名称。
        force_threshold (float): 接触判定阈值，单位 N。

    Returns:
        torch.Tensor: 二值接触观测，形状 `[num_envs, num_sensors]`。
    """

    contact_bits = []  # 每个 fingertip 一列 0/1
    for sensor_name in sensor_names:
        force_w = sensor_total_force_w(env, sensor_name)  # `[B,3]`，该 fingertip 总接触力
        contact_bits.append((torch.linalg.norm(force_w, dim=-1) > float(force_threshold)).float())  # `[B]`
    return torch.stack(contact_bits, dim=-1)  # `[B,K]`，K 个指尖接触位


__all__ = ["fingertip_contact_binary", "fingertip_contact_force_w"]
