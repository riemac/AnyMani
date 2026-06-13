r"""GM teacher RL 环境配置，由 `distill` 选择资产并消费 `tasks/gm`。

`tasks/gm` 只定义 object-in-hand MDP；它不决定训练用哪只手、哪批资产、如何
划分 train/heldout。这里位于 `distill/rl`，因此可以把一个 generated hand bundle
绑定进 GM env，形成第一阶段 debug teacher task。

当前第一版只选择单个 same-topology post-mutate asset。多资产并行时，本文件会
演化为 manifest-driven 配置：由 manifest 给出一组 `GmHandAssetRef`，再按 env id
分配 asset。Grasp Cache 暂后，不在本文件实现。
"""

from __future__ import annotations

from pathlib import Path

from anymani.tasks.gm.asset_binding import GmHandAssetRef, build_hand_articulation_cfg
from anymani.tasks.gm.inhand_env_cfg import GmInHandEnvCfg
from isaaclab.utils import configclass

DEBUG_GM_HAND_ROOT = Path(
    "/home/hac/isaac/AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
    "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/46e6ea57"
)
r"""第一阶段 GM teacher debug asset。

该路径指向一个已生成的 `hand.urdf` / `hand.yaml` bundle，topology 为
`right_t4_i4_m4_r4`，DOF=16。它只用于先打通 vertical slice；正式同拓扑并行训练
应由 manifest 传入一组同 topology asset，而不是继续硬编码单路径。
"""


DEBUG_GM_HAND_REF = GmHandAssetRef(
    root_dir=DEBUG_GM_HAND_ROOT,
    topology_name="right_t4_i4_m4_r4",
    asset_id="46e6ea57",
)
"""传给 `tasks/gm` 的单资产引用；它是已选资产，不是 asset bank。"""


@configclass
class GmTeacherDebugEnvCfg(GmInHandEnvCfg):
    r"""单资产 GM teacher debug 环境。

    该 cfg 是 distill 训练管线消费 tasks/gm 的最小例子：

    $$
    \texttt{DEBUG\_GM\_HAND\_REF}
    \rightarrow \texttt{build\_hand\_articulation\_cfg}
    \rightarrow \texttt{scene.robot}.
    $$

    它不负责 asset split、不负责网络结构、不负责 rl_games runner；这些继续留在
    `distill/rl` 的训练入口与模型 adapter 中。
    """

    def __post_init__(self):
        r"""绑定 debug hand asset 并降低默认并行规模，便于 smoke。"""

        super().__post_init__()
        self.scene.robot = build_hand_articulation_cfg(DEBUG_GM_HAND_REF, prim_path="{ENV_REGEX_NS}/robot")
        self.scene.num_envs = 256  # debug teacher 默认小规模；命令行可覆盖
        self.scene.replicate_physics = False  # 后续多资产混合需要 False；单资产先保持一致语义
        self.episode_length_s = 10.0  # 第一阶段短 episode，便于快速发现 reset/reward 问题


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
    "DEBUG_GM_HAND_REF",
    "DEBUG_GM_HAND_ROOT",
    "GmTeacherDebugEnvCfg",
    "GmTeacherDebugEnvCfg_PLAY",
]
