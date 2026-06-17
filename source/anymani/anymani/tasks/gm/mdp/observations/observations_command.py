r"""Command observation terms for GM in-hand manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reorient_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    r"""读取 `ReorientCommand` 生成的 policy-facing command。

    `ReorientCommand` 的 `command` property 已固定为：
    $$
    \mathbf{c}_t =
    \left[\hat\omega^{\{h\}},\ \phi_e^{\{h\}}\right]
    \in \mathbb{R}^{6}
    $$

    其中：

    - $\hat\omega^{\{h\}}$：hand semantic frame `{h}` 下的有向单位旋转轴；
    - $\phi_e^{\{h\}}$：space error $\log(R_{goal}R_{current}^{-1})$ 表达到 `{h}` 后的 so(3) 向量。

    DONE(与 command/reward 合同对齐): 默认 `axis_resample_mode="subgoal"`，所以 axis 不再
    承诺整个 episode 固定；policy 只看到 `{h}` 表达，保持 hand-centric 任务语义。reward /
    termination / curriculum 若需要 `{e}` 轴或 goal quaternion，应从 command term 内部 buffer
    读取 `axis_e`、`error_so3_e`、`goal_quat_w` 等，不从 obs 反推。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的重定向 command 名称。

    Returns:
        torch.Tensor: command tensor，形状 `[num_envs, 6]`。
    """

    return env.command_manager.get_command(command_name)  # `[B,6]`，即 `[axis_h, error_so3_h]`


__all__ = ["reorient_command"]
