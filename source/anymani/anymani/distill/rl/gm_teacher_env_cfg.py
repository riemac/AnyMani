r"""GM teacher RL 环境配置，由 `distill` 消费 `tasks/gm`。

`tasks/gm` 只定义 object-in-hand MDP；它不决定训练算法、网络结构、checkpoint、
rollout dataset 或实验 manifest。这里位于 `distill/rl`，因此可以把 `gm` 的默认
in-hand 环境作为 teacher debug route 暴露给 rl_games 配置。

当前 debug teacher 不再维护旧的单资产 `asset_binding`。资产选择的声明式 contract 已经
落在 `GmInHandEnvCfg.scene.robot` 内部：固定 post-mutate run、`sample_count=128`、
`sample_seed=42`、round-robin routing。`distill` 后续若需要 train/heldout split，
应通过替换 `HandSpawnCfg.bank` 或 manifest wrapper 实现，而不是恢复旧单资产接口。
"""

from __future__ import annotations

from anymani.tasks.gm.inhand_env_cfg import GM_DEFAULT_ENVS_PER_HAND, GM_DEFAULT_NUM_ENVS, GmInHandEnvCfg
from isaaclab.utils import configclass

GM_TEACHER_DEBUG_NUM_ENVS = GM_DEFAULT_NUM_ENVS
r"""debug teacher 默认并行环境数，沿用 `gm` 的默认 hand routing 合同。"""

GM_TEACHER_DEBUG_ENVS_PER_HAND = GM_DEFAULT_ENVS_PER_HAND
r"""每个 selected hand asset 的 env 数，保持 $32$ 作为 round-robin routing 的阅读锚点。"""

GM_TEACHER_DEBUG_EPISODE_LENGTH_S = 10.0
r"""debug teacher 的较短 episode 长度，便于 first runnable slice 快速暴露 reset/reward 问题。"""


@configclass
class GmTeacherDebugEnvCfg(GmInHandEnvCfg):
    r"""GM teacher debug 环境。

    该 cfg 是 distill 训练管线消费 `tasks/gm` 的最小例子：

    $$
    \texttt{GmInHandEnvCfg}
    \rightarrow \texttt{rl\_games\_cfg\_entry\_point}
    \rightarrow \texttt{teacher rollout}.
    $$

    它不覆写 `scene.robot`，因此默认消费 `gm` 已装配好的 generated hand asset
    selection。这里仅调整训练入口语义上的 episode 长度，避免把资产选择重新复制到
    `distill`。
    """

    def __post_init__(self):
        r"""设置 teacher debug 的训练时长参数。"""

        super().__post_init__()
        self.scene.num_envs = GM_TEACHER_DEBUG_NUM_ENVS  # 与 gm 默认 hand routing 对齐
        self.scene.replicate_physics = False  # 多 URDF prototype 的 batched scene 必须保持非 replicate physics
        self.episode_length_s = GM_TEACHER_DEBUG_EPISODE_LENGTH_S  # 短 episode，优先发现 reset/reward contract 问题


@configclass
class GmTeacherDebugEnvCfg_PLAY(GmTeacherDebugEnvCfg):
    r"""视觉检查 / smoke 用小规模 GM teacher 环境。"""

    def __post_init__(self):
        r"""进一步缩小 env 数，并关闭 policy corruption。"""

        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


__all__ = [
    "GM_TEACHER_DEBUG_ENVS_PER_HAND",
    "GM_TEACHER_DEBUG_EPISODE_LENGTH_S",
    "GM_TEACHER_DEBUG_NUM_ENVS",
    "GmTeacherDebugEnvCfg",
    "GmTeacherDebugEnvCfg_PLAY",
]
