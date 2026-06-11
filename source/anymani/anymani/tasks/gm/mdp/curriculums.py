r"""Curriculum terms for `tasks.gm`.

本模块只承载 `gm` 任务语义内部的 curriculum 状态更新，不处理 asset-bank
采样策略，也不处理训练算法超参。当前唯一落子的 curriculum 是 AnyRotate 风格的
adaptive reward curriculum：先让策略学会基本重定向，再逐步释放 contact / stable
正则项。

核心思想来自 AnyRotate Appendix B.3：

$$
\lambda_{rew} = \operatorname{clip}\left(
\frac{g_{eval}-g_{min}}{g_{max}-g_{min}},\ 0,\ 1
\right)
$$

但本项目把 $g_{eval}$ 明确命名为 `goal_success_count` 的全局 EMA，避免沿用
IsaacLab 官方 inhand 中语义偏含糊的 `consecutive_success`。这里的
`goal_success_count` 指“一个 episode 内完成了多少个重定向子目标”，不是在
阈值内停留了多少帧。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


class RewardCurriculumByGoalSuccess(ManagerTermBase):
    r"""根据平均子目标成功数释放 reward curriculum 系数。

    该 term 维护一个全局标量 $\lambda_{global}\in[0,1]$，并写入 env 属性，
    供 `rewards.py` 中的 contact / stable / action 正则项读取。之所以采用
    全局标量，而不是 per-env 系数，是因为它更接近 AnyRotate 的实验语义：
    “策略整体已经学会基本重定向后，再要求接触质量和动作稳定性”。

    计算流程：

    1. command term 在每个 env 中维护 `metrics[metric_key]`，默认
       `metric_key="goal_success_count"`；该值表示当前 episode 已完成的
       重定向子目标数量。
    2. 本 curriculum term 对当前参与更新的 env ids 求平均：
       $g_{batch}=\operatorname{mean}(g_i)$。
    3. 用 EMA 得到平滑全局指标：
       $g_{ema}\leftarrow (1-\alpha)g_{ema}+\alpha g_{batch}$。
    4. 线性映射为 release 系数：
       $\lambda=\operatorname{clip}((g_{ema}-g_{min})/(g_{max}-g_{min}),0,1)$。

    preset:
        - `g_min=1.0, g_max=2.0`：对齐 AnyRotate 的释放区间直觉，即平均每
          个 episode 至少完成约 1 个子目标后才开始释放 contact/stable；到约
          2 个子目标时完全释放。
        - `ema_alpha=0.05`：约 $1/\alpha=20$ 次 curriculum update 的平滑窗口，
          作为第一版保守默认；后续可按 rollout 频率调参。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        r"""初始化 EMA 状态。

        Args:
            cfg (CurriculumTermCfg): Isaac Lab curriculum term 配置。
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        """

        # 父类保存 cfg/env，保持与 Isaac Lab manager term 生命周期一致
        super().__init__(cfg, env)

        # EMA 从 0 开始，意味着训练最初默认处在 curriculum 未释放状态
        self._ema_goal_success = torch.tensor(0.0, device=env.device)  # 标量，$g_{ema}$
        self._lambda_global = torch.tensor(0.0, device=env.device)  # 标量，$\lambda_{global}$

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""Reset hook：不清空全局 curriculum 进度。

        Curriculum 表示训练整体阶段，不是单个 env episode 状态；因此 env reset
        时不应把 EMA 清零。若用户要重新开始训练，应重新创建 env / manager。

        Args:
            env_ids (Sequence[int] | None): Isaac Lab manager 传入的 reset env ids。
        """

        _ = env_ids  # curriculum 是全局训练状态，单个 env reset 不影响它

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        metric_key: str = "goal_success_count",
        g_min: float = 1.0,
        g_max: float = 2.0,
        ema_alpha: float = 0.05,
        lambda_attr_name: str = "_gm_reward_curriculum_lambda",
        progress_attr_name: str = "_gm_reward_curriculum_goal_success_ema",
    ) -> dict[str, torch.Tensor]:
        r"""更新并写出全局 reward curriculum 系数。

        Args:
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
            env_ids (Sequence[int] | slice): 本次 curriculum update 涉及的 env ids。
            command_name (str): command manager 中的重定向 command term 名称。
            metric_key (str): command term `metrics` 中的 per-env 成功计数字段名。
            g_min (float): release 起点；$g_{ema}\le g_{min}$ 时 $\lambda=0$。
            g_max (float): release 终点；$g_{ema}\ge g_{max}$ 时 $\lambda=1$。
            ema_alpha (float): EMA 更新率 $\alpha$。
            lambda_attr_name (str): 写入 env 的全局 release 系数属性名。
            progress_attr_name (str): 写入 env 的 EMA progress 属性名。

        Returns:
            dict[str, torch.Tensor]: 供 Isaac Lab logging 使用的 curriculum 状态。

        Raises:
            RuntimeError: 当 command term 没有暴露 `metric_key` 时抛出。
            ValueError: 当 `g_max <= g_min` 或 `ema_alpha` 越界时抛出。
        """

        # 检查 release 区间，避免除以 0 或负斜率导致 curriculum 语义反转
        if float(g_max) <= float(g_min):
            raise ValueError(f"g_max must be larger than g_min, got g_min={g_min}, g_max={g_max}.")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}.")

        # 从 command term 读取 per-env 子目标成功数；该 metric 是 ReorientCommand 必须兑现的契约
        command_term = env.command_manager.get_term(command_name)  # 重定向 command term
        goal_success_count = command_term.metrics.get(metric_key, None)  # `[B]`，每个 env 当前 episode 完成的子目标数
        if goal_success_count is None:
            raise RuntimeError(
                f"Command term '{command_name}' must expose metrics['{metric_key}'] for reward curriculum. "
                "Use `goal_success_count` to count completed subgoals, not threshold-satisfied frames."
            )

        # 按 Isaac Lab 传入的 env_ids 取子集；slice(None) 表示全部 env
        if isinstance(env_ids, slice):
            batch_success = goal_success_count[env_ids].float()  # `[B_update]`，本次更新子集
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=goal_success_count.device, dtype=torch.long)  # env id 索引
            batch_success = goal_success_count[env_ids_tensor].float()  # `[B_update]`，本次更新子集

        # 计算 batch 平均成功数，并做 EMA 平滑，避免单个 rollout 抖动导致 reward 突然释放/关闭
        batch_mean = batch_success.mean()  # 标量，$g_{batch}$
        alpha = float(ema_alpha)  # EMA 更新率 $\alpha$
        self._ema_goal_success = (1.0 - alpha) * self._ema_goal_success + alpha * batch_mean.detach()  # $g_{ema}$

        # 线性 release：$g_{min}$ 前为 0，$g_{max}$ 后为 1，中间线性过渡
        raw_lambda = (self._ema_goal_success - float(g_min)) / (float(g_max) - float(g_min))  # 未裁剪 release
        self._lambda_global = torch.clamp(raw_lambda, 0.0, 1.0)  # $\lambda_{global}\in[0,1]$

        # 写入 env 属性，供 rewards.py 的 `_curriculum_gain` 读取；detach 表示这是训练调度状态而非梯度图
        setattr(env, lambda_attr_name, self._lambda_global.detach())  # 全局 reward release 系数
        setattr(env, progress_attr_name, self._ema_goal_success.detach())  # 平滑后的平均子目标成功数

        # 返回 dict 让 Isaac Lab 的 curriculum logging 能记录关键状态
        return {
            "lambda": self._lambda_global.detach(),  # 当前 release 系数
            "goal_success_ema": self._ema_goal_success.detach(),  # 平滑 progress
            "goal_success_batch": batch_mean.detach(),  # 当前 batch 原始平均成功数
        }


__all__ = ["RewardCurriculumByGoalSuccess"]
