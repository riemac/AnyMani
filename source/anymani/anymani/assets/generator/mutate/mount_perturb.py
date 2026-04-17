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
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import PoseCfg
from ._base import MutatorBase
from ._distribution import ScalarDistributionCfg


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

    translation_distribution: ScalarDistributionCfg = field(
        default_factory=lambda: ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.003)
    )
    """挂载平移扰动分布；默认是 $\sigma=3\text{ mm}$ 的零均值高斯。"""

    perturb_rotation: bool = False
    """是否同时对旋转（rpy）施加扰动。默认关闭；打开后需配合 rotation_sigma。"""

    rotation_distribution: ScalarDistributionCfg = field(
        default_factory=lambda: ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.05)
    )
    """挂载姿态扰动分布；仅 `perturb_rotation=True` 时参与联合采样。"""

    clip_translation: float | None = 0.02
    """平移扰动的最大绝对幅度（meter）；为 ``None`` 时不额外裁剪。"""

    clip_rotation: float | None = 0.2
    """旋转扰动的最大绝对幅度（rad）；为 ``None`` 时不额外裁剪。"""

    per_finger_translation_distribution: dict[str, ScalarDistributionCfg] = field(default_factory=dict)
    """可选的每 finger 单独平移分布覆盖；键为 finger 名。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = MountPerturbMutator
class MountPerturbMutator(MutatorBase):
    r"""挂载点扰动运行时壳。

    在已构建好的 `HandCfg` 上对目标 finger 的挂载位姿做小范围局部微调，
    不改变拓扑和内部 joint 链。
    """

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, ScalarDistributionCfg]:
        r"""描述当前 hand 上每根目标 finger 的挂载位姿扰动分布。"""

        target_names = set(self.cfg.target_fingers)
        distribution_plan: dict[str, ScalarDistributionCfg] = {}
        for finger in target.fingers:
            if target_names and finger.name not in target_names:
                continue

            translation_cfg = self.cfg.per_finger_translation_distribution.get(
                finger.name,
                self.cfg.translation_distribution,
            ).copy()
            if self.cfg.clip_translation is not None:
                clip_t = float(self.cfg.clip_translation)
                translation_cfg.clip_min = -clip_t if translation_cfg.clip_min is None else translation_cfg.clip_min
                translation_cfg.clip_max = clip_t if translation_cfg.clip_max is None else translation_cfg.clip_max
            for axis_name in ("tx", "ty", "tz"):
                distribution_plan[f"{finger.name}::{axis_name}"] = translation_cfg.copy()

            if self.cfg.perturb_rotation:
                rotation_cfg = self.cfg.rotation_distribution.copy()
                if self.cfg.clip_rotation is not None:
                    clip_r = float(self.cfg.clip_rotation)
                    rotation_cfg.clip_min = -clip_r if rotation_cfg.clip_min is None else rotation_cfg.clip_min
                    rotation_cfg.clip_max = clip_r if rotation_cfg.clip_max is None else rotation_cfg.clip_max
                for axis_name in ("rr", "rp", "ry"):
                    distribution_plan[f"{finger.name}::{axis_name}"] = rotation_cfg.copy()

        return distribution_plan

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行挂载点位姿扰动。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 挂载点微调后的整手配置。
        """

        mutated = target.copy()
        target_names = set(self.cfg.target_fingers)
        sampled = sampled_params or {}

        for finger in mutated.fingers:
            if target_names and finger.name not in target_names:
                continue

            delta_pos = [
                float(sampled.get(f"{finger.name}::tx", 0.0)),
                float(sampled.get(f"{finger.name}::ty", 0.0)),
                float(sampled.get(f"{finger.name}::tz", 0.0)),
            ]

            # 旋转默认关闭；这符合“先稳住位置扰动，再按需放开姿态搜索”的保守策略。
            delta_rpy = [0.0, 0.0, 0.0]
            if self.cfg.perturb_rotation:
                delta_rpy = [
                    float(sampled.get(f"{finger.name}::rr", 0.0)),
                    float(sampled.get(f"{finger.name}::rp", 0.0)),
                    float(sampled.get(f"{finger.name}::ry", 0.0)),
                ]

            finger.mount = PoseCfg(
                pos=(
                    finger.mount.pos[0] + delta_pos[0],
                    finger.mount.pos[1] + delta_pos[1],
                    finger.mount.pos[2] + delta_pos[2],
                ),
                rpy=(
                    finger.mount.rpy[0] + delta_rpy[0],
                    finger.mount.rpy[1] + delta_rpy[1],
                    finger.mount.rpy[2] + delta_rpy[2],
                ),
            )

        return mutated

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
