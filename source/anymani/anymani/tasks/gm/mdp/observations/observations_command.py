r"""Command observation terms for GM in-hand manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reorient_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    output_spec: dict[str, Any] | None = None,
) -> torch.Tensor:
    r"""读取 `ReorientCommand` 生成的 policy-facing command 表示。

    默认情况下，本 observation term 直接返回 `ReorientCommandCfg.command_output` 指定的
    `command` property。若传入 `output_spec`，则通过 command term 的
    `format_command(output_spec)` 临时读取另一种表示，但不改变 command 内部目标、metrics
    或 resampling 时序。

    历史默认可以复现：
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
        output_spec (dict[str, Any] | None): 可选 override spec；`None` 表示使用 command cfg 默认输出。

    Returns:
        torch.Tensor: command tensor，形状由 command output spec 决定。
    """

    if output_spec is None:
        return env.command_manager.get_command(command_name)  # cfg 默认 command 表示，通常来自 `command_output`

    command_term = env.command_manager.get_term(command_name)  # ReorientCommand term，提供 canonical buffers 与 formatter
    return command_term.format_command(output_spec)  # 无副作用地覆盖输出格式，用于局部 obs 消融


__all__ = ["reorient_command"]
