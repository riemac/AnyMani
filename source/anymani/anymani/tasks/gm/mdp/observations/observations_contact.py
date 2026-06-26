r"""Contact observation terms for GM in-hand manipulation.

DONE(contact obs): 指尖触觉观测应服务 hand-centric 表征，而非 world-centric 表征。
本项目主线是手型泛化（embodiment generalization），不是 gravity-invariant multi-axis
rotation。接触力向量默认表达在 hand semantic frame `{h}` 下，使语义绑定到整只手
的操作坐标系，而不是绑定到 world frame，也不是绑定到随关节快速旋转的 fingertip
sensor frame。

若未来引入任意手姿态的 multi-axis rotation（AnyRotate 路线），届时追加 gravity direction
in hand frame 作为额外输入即可，不在当前阶段引入该复杂度。

当前 first runnable slice 先提供两类 cheap contact obs：

- `fingertip_contact_force`: 每个 fingertip 对 object 的总接触力，可选择 `{h}` / `{e}` 轴，形状 `[B,3K]`；
- `fingertip_contact_binary`: 每个 fingertip 是否超过力阈值，形状 `[B,K]`。

NOTE: 传感器 prim_path 不再硬编码四指名称，而是由 `gm.contact_sensors` 从 hand sidecar
自动推导 per-link `ContactSensorCfg`。filtered contact 只过滤到 `"{ENV_REGEX_NS}/object"`，
避免手指间自碰污染 teacher contact obs。

为何不采用 fingertip sensor local frame `{S_k}`：
    `{S_k}` 会随对应 finger link 的关节姿态持续旋转。同一个 hand-level 接触趋势在
    不同 $q$ 下会被编码成不同局部分量，使 RL policy 额外学习一个随状态变化的反变换。
    即便理论上 $q$ 进入 observation 后仍满足 Markov，表征也明显更非平稳。相比之下，
    `{h}` 固定在整只手的语义坐标系上，既去掉 hand global pose，又不随单根手指局部
    frame 跳变，是当前 teacher obs 更稳的接触力表达。

未来若需要接触点，可输出 hand-frame 版本：
$$
\mathbf{c}^{\text{contact}}_k =
\big[\,c_x,c_y,c_z,F_x,F_y,F_z,\|F\|\,\big]^{\{h\}} \in \mathbb{R}^7,
$$
其中接触点来自 IsaacLab 平均 `contact_pos_w`，先减 hand root position 再转到 `{h}`；
力矢量从 `net_forces_w / force_matrix_w` 转到 `{h}`。该完整点-力 obs 暂后，当前
只落总接触力向量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ...contact_sensors import sensor_total_force_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _semantic_R_ha_tensor(env: ManagerBasedRLEnv, semantic_R_ha: tuple[float, ...]) -> torch.Tensor:
    r"""把配置层 row-major $R_{ha}$ 转成 runtime tensor。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env，用于确定 device。
        semantic_R_ha (tuple[float, ...]): row-major 9 元组，语义为列向量约定下
            $v^h = R_{ha} v^a$。

    Returns:
        torch.Tensor: 旋转矩阵 $R_{ha}$，形状 `[3, 3]`，位于 env device。
    """

    return torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)  # $R_{ha}$


FrameName = Literal["h", "e"]


def fingertip_contact_force(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    frame: FrameName = "h",
) -> torch.Tensor:
    r"""读取所有 fingertip ContactSensor 的接触力，并按配置选择输出坐标轴。

    IsaacLab contact sensor 原始力向量在 world frame `{w}` 下。`frame="h"` 时把每个
    fingertip 的总接触力转到 hand semantic frame `{h}`：
    $$
    F_k^h = R_{ha} R_{aw} F_k^w.
    $$
    `frame="e"` 时直接返回 env/world 轴向的力，用于诊断 hand-centric force 是否比
    env-centric force 更适合 RL。

    力是自由向量，不受 hand root position 平移影响；只需用 hand root orientation
    和静态 $R_{ha}$ 做旋转变换。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (tuple[str, ...]): 指尖传感器名称，顺序应与 hand/finger 语义顺序一致。
        robot_cfg (SceneEntityCfg): hand articulation 配置，用于读取 root orientation。
        semantic_R_ha (tuple[float, ...]): $R_{ha}$，row-major 9 元组。
        frame (FrameName): 输出坐标轴，`"h"` 为 hand semantic 轴，`"e"` 为 env/world 轴。

    Returns:
        torch.Tensor: 拼接后的接触力，形状 `[num_envs, 3 * num_sensors]`，单位 N。
    """

    robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root orientation 来源
    R_ha = _semantic_R_ha_tensor(env, semantic_R_ha)  # `{a}->{h}` 语义对齐矩阵
    forces_h = []  # 每个 fingertip 一项 `[B,3]`，最后按 sensor 顺序拼接
    for sensor_name in sensor_names:
        force_w = sensor_total_force_w(env, sensor_name)  # `[B,3]`，该 fingertip world-frame 总接触力
        if frame == "e":
            forces_h.append(force_w)  # `[B,3]`，env/world 轴向，不做 hand-frame 旋转
            continue
        if frame != "h":
            raise ValueError(f"Unsupported frame for fingertip_contact_force: {frame}.")
        force_a = math_utils.quat_apply_inverse(robot_asset.data.root_quat_w, force_w)  # $R_{aw}F^w$
        forces_h.append(force_a @ R_ha.T)  # row-vector 写法：$F^h=F^aR_{ha}^T$
    return torch.cat(forces_h, dim=-1)  # `[B,3K]`，teacher policy/critic 的 hand-frame force obs


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


__all__ = ["fingertip_contact_binary", "fingertip_contact_force"]
