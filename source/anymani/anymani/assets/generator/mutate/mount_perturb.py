r"""挂载点扰动工具：在已有 HandCfg 上对 finger 挂载位姿做小范围局部微调。

对应 `资产生产概略.png` 中 `挂载点扰动 三维位置 + rpy 可选`，以及
`前后序.png` 中归属后序的 `挂载点微扰`，理由是"纯位姿修改"。

设计说明
--------

### 职责边界

`MountPerturbMutator` 只修改 `FingerCfg.mount.pos`（以及可选的 `.rpy`）。
它不改变 finger 内部的 joint 链，不改变几何，不改变拓扑。

### 三维位置扰动

平移扰动在 palm 坐标系下施加，以正态分布随机偏移各分量：

$$
\mathbf{p}_{\text{new}} = \mathbf{p}_0 + \boldsymbol{\delta}, \quad
\delta_k \sim \mathcal{N}(0, \sigma_{\text{trans}})
$$

### 旋转扰动（可选）

若 `perturb_rotation=True`，对 rpy 各分量独立叠加正态偏移：

$$
\mathbf{r}_{\text{new}} = \mathbf{r}_0 + \boldsymbol{\epsilon}, \quad
\epsilon_k \sim \mathcal{N}(0, \sigma_{\text{rot}})
$$

### 裁剪保护

分别对平移和旋转扰动幅度做 clip，防止极端值破坏手部整体结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...asset_base import AssetCfgBase, HandCfg
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class MountPerturbCfg(AssetCfgBase):
    r"""挂载点扰动工具配置。

    用于在已构建好的 HandCfg 上做 finger 级挂载位姿的小范围微调。
    """

    class_type: type["MountPerturbMutator"] | None = None
    """关联的运行时类。"""

    target_fingers: tuple[str, ...] = ()
    """需要扰动的手指名称集合；空元组表示作用于全部手指。"""

    translation_sigma: float = 0.003
    """平移扰动强度（meter）；对应每个分量的正态标准差。
    0.003 m 约为 3 mm，适合在已有设计基础上做小范围局部搜索。"""

    perturb_rotation: bool = False
    """是否同时对旋转（rpy）施加扰动。默认关闭；打开后需配合 rotation_sigma。"""

    rotation_sigma: float = 0.05
    """旋转扰动强度（rad）；仅 ``perturb_rotation=True`` 时有效。
    0.05 rad ≈ 2.9°，对应小范围姿态搜索。"""

    clip_translation: float | None = 0.02
    """平移扰动的最大绝对幅度（meter）；为 ``None`` 时不额外裁剪。"""

    clip_rotation: float | None = 0.2
    """旋转扰动的最大绝对幅度（rad）；为 ``None`` 时不额外裁剪。"""

    per_finger_translation_sigma: dict[str, float] = field(default_factory=dict)
    """可选的每 finger 单独 translation_sigma 覆盖；键为 finger 名。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = MountPerturbMutator
        if self.translation_sigma < 0:
            raise ValueError(f"translation_sigma must be >= 0, got {self.translation_sigma}")
        if self.rotation_sigma < 0:
            raise ValueError(f"rotation_sigma must be >= 0, got {self.rotation_sigma}")


# ============================================================================
#  运行时壳
# ============================================================================


class MountPerturbMutator(MutatorBase):
    r"""挂载点扰动运行时壳。

    在已构建好的 `HandCfg` 上对目标 finger 的挂载位姿做小范围局部微调，
    不改变拓扑和内部 joint 链。
    """

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行挂载点位姿扰动。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 挂载点微调后的整手配置。
        """

        pass

        # TODO:算法之一（mount perturb）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.target_fingers: 目标 finger 名集合（空 = 全部）
        #   cfg.translation_sigma: 平移扰动标准差（m）
        #   cfg.perturb_rotation: 是否同时扰动旋转
        #   cfg.rotation_sigma: 旋转扰动标准差（rad）
        #   cfg.clip_translation / cfg.clip_rotation: 单步幅度裁剪
        #   cfg.per_finger_translation_sigma: 每 finger 单独 sigma 覆盖
        #
        # 输出：HandCfg（深拷贝 + 修改目标 finger 的 mount.pos / mount.rpy）
        #
        # ── 对每个目标 finger f ──
        #   σ_t = per_finger_translation_sigma.get(f.name, translation_sigma)
        #
        #   平移扰动（在 palm 坐标系下施加）：
        #     δ_pos[k] ~ N(0, σ_t)，k ∈ {0, 1, 2}
        #     若 clip_translation 不为 None：裁剪 δ_pos[k] 到 [-clip_t, clip_t]
        #     pos_new = f.mount.pos + δ_pos（逐分量相加）
        #
        #   旋转扰动（仅 perturb_rotation=True 时）：
        #     δ_rpy[k] ~ N(0, rotation_sigma)，k ∈ {0, 1, 2}
        #     若 clip_rotation 不为 None：裁剪 δ_rpy[k] 到 [-clip_r, clip_r]
        #     rpy_new = f.mount.rpy + δ_rpy（逐分量相加）
        #
        # ── 重建 HandCfg ──
        #   深拷贝 target，修改目标 finger 的 mount.pos（和可选的 mount.rpy），
        #   返回新对象
        #
        # ── 与 preset 的交叉验证 ──
        #   挂载点的初始值应与 preset 保持同一语义基线；
        #   扰动完成后，若相邻 finger 的挂载点发生碰撞或超出 palm 边界，
        #   应当标记为警告（但当前草案不强制拒绝，由 validator 处理）。
        #
        # IDEA：这个工具适合接入可视化闭环；用户先看树状结构，再决定是否接受
        # 这次扰动结果，或者再次采样。


__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
