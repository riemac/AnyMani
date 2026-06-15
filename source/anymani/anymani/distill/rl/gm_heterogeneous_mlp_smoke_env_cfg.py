r"""Distill-owned env cfg for heterogeneous-hand MLP training feasibility.

这个文件不重新定义 MDP；MDP 仍由 `tasks/gm/heterogeneous_test_env_cfg.py`
负责。这里位于 `distill/rl`，只表达训练管线选择：用 3 个 same-schema hand
variants 做最小 MLP PPO 可行性测试，默认并行规模为 $3\times100=300$ envs。

该环境的 reward 仍是 alive reward，因此训练曲线没有科研指标意义；它只回答
一个工程-科研边界问题：异构 URDF articulation batch 能否被 rl_games 的最小
MLP policy 完成 rollout、反传、优化器更新与 checkpoint/log 写出。
"""

from __future__ import annotations

from anymani.tasks.gm.heterogeneous_test_env_cfg import HeterogeneousHandTestEnvCfg
from isaaclab.utils import configclass

HETEROGENEOUS_MLP_SMOKE_VARIANT_COUNT = 3
r"""当前 MVP 固定的 same-schema hand variant 数。"""

HETEROGENEOUS_MLP_SMOKE_ENVS_PER_VARIANT = 100
r"""每个 hand variant 分到的并行 env 数。"""

HETEROGENEOUS_MLP_SMOKE_NUM_ENVS = HETEROGENEOUS_MLP_SMOKE_VARIANT_COUNT * HETEROGENEOUS_MLP_SMOKE_ENVS_PER_VARIANT
r"""MLP smoke 默认总并行规模，即 $3\times100=300$。"""

HETEROGENEOUS_MLP_SMOKE_EPISODE_LENGTH_S = 0.25
r"""训练 smoke 的短 episode 长度（秒）。

`HeterogeneousHandTestEnvCfg` 的 policy step 为 $\Delta t=4/120=1/30$ 秒，
MLP YAML 的 `horizon_length=8` 对应约 $0.267$ 秒 rollout。把 episode 设为
$0.25$ 秒可以保证一个最短训练 epoch 内至少出现 timeout reset，从而让 rl_games
写出有限 reward 统计，而不是停在“尚未有 episode 结束”的 `-inf` 占位值。
"""


@configclass
class HeterogeneousMlpSmokeEnvCfg(HeterogeneousHandTestEnvCfg):
    r"""异构 generated hand 的最小 MLP PPO 可行性训练环境。

    继承关系刻意保持单向：`tasks/gm` 给出环境语义，本类只在 `distill` 内调整
    训练规模。这样后续正式 teacher specialist policy 可以复用同一个 env contract，
    而不会把 rl_games / checkpoint / 网络选择反向塞进 `tasks/gm`。
    """

    def __post_init__(self) -> None:
        r"""把 GUI 默认 9 envs 提升到 300，并缩短 episode 以得到有限统计。"""

        super().__post_init__()
        self.scene.num_envs = HETEROGENEOUS_MLP_SMOKE_NUM_ENVS
        self.episode_length_s = HETEROGENEOUS_MLP_SMOKE_EPISODE_LENGTH_S


__all__ = [
    "HETEROGENEOUS_MLP_SMOKE_ENVS_PER_VARIANT",
    "HETEROGENEOUS_MLP_SMOKE_EPISODE_LENGTH_S",
    "HETEROGENEOUS_MLP_SMOKE_NUM_ENVS",
    "HETEROGENEOUS_MLP_SMOKE_VARIANT_COUNT",
    "HeterogeneousMlpSmokeEnvCfg",
]
