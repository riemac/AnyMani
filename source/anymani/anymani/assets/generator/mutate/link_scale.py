r"""连杆长度缩放工具：在已有 HandCfg 上对 link 两岸距离做 ± 扰动。

对应 `资产生产概略.png` 中 `连杆长度 已有长度 ± l% 扰动` 项，以及
`前后序.png` 中归属后序的 `link 长度微调 (±Δ)`，理由是"已构建好的 HandCfg 上做 scale"。

有效长度公式
------------

$$
l_{\text{valid},\,i} = l_i + d_i, \quad d_i \sim \mathcal{N}(0,\; \sigma_i)
$$

其中 $l_i$ 是 joint $i$ 的 `origin.pos` 的欧氏模长，$d_i$ 是正态采样的绝对偏移量。
若配置 `scale_mode = "relative"`，则扰动以比例形式施加：

$$
l_{\text{valid},\,i} = l_i \cdot (1 + \varepsilon_i), \quad \varepsilon_i \sim \mathcal{N}(0,\; \sigma_\%/100)
$$

设计说明
--------

### 缩放语义

实际修改的是 joint 的 `origin.pos`，即把 pos 向量在其方向上做等比缩放。
不改变旋转（`origin.rpy`），不改变拓扑。

### 选择性支持

可以通过 `target_joints` 指定只作用于部分关节，默认作用于全部非固定关节。

### 裁剪保护

`clip_ratio` 限制最终比例落在 `[1 - clip, 1 + clip]` 范围内，确保不产生
负长度或极端形变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...asset_base import AssetCfgBase, HandCfg
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class LinkScaleCfg(AssetCfgBase):
    r"""连杆长度缩放工具配置。"""

    class_type: type["LinkScaleMutator"] | None = None
    """关联的运行时类。"""

    target_joints: tuple[str, ...] = ()
    """需要缩放的关节名称集合；空元组表示作用于全部非固定关节。"""

    scale_mode: Literal["absolute", "relative"] = "relative"
    """扰动模式。``absolute`` 以绝对长度（meter）施加偏移 $d_i$；
    ``relative`` 以百分比形式施加比例扰动 $\varepsilon_i$。"""

    sigma: float = 0.05
    """扰动强度。``absolute`` 模式下单位为 meter；``relative`` 模式下为比例（如 0.05 表示 ±5%）。"""

    clip_ratio: float = 0.3
    """缩放比例裁剪上限；最终比例限定在 ``[1 - clip_ratio, 1 + clip_ratio]`` 内。
    仅在 ``relative`` 模式下有意义。"""

    clip_absolute: float | None = None
    """绝对偏移裁剪上限（meter）；为 ``None`` 时不额外裁剪。仅在 ``absolute`` 模式下有意义。"""

    per_joint_sigma: dict[str, float] = field(default_factory=dict)
    """可选的每 joint 单独 sigma 覆盖；键为 joint 名，值为对应 sigma。
    未在此 dict 中出现的 joint 使用全局 ``sigma``。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = LinkScaleMutator
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if not 0.0 < self.clip_ratio <= 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1], got {self.clip_ratio}")


# ============================================================================
#  运行时壳
# ============================================================================


class LinkScaleMutator(MutatorBase):
    r"""连杆长度缩放运行时壳。

    在已构建好的 `HandCfg` 上对指定（或全部）关节的 `origin.pos` 做
    方向保持的等比缩放，不改变拓扑与旋转。
    """

    cfg: LinkScaleCfg

    def __init__(self, cfg: LinkScaleCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行连杆长度缩放。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 缩放后的整手配置；若所有关节长度为零则返回 ``None``。
        """

        pass

        # TODO:算法之一（link length perturbation）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.target_joints: 需要缩放的关节名集合（空 = 全部非固定关节）
        #   cfg.scale_mode: "absolute" | "relative"
        #   cfg.sigma: 全局扰动强度
        #   cfg.per_joint_sigma: 每关节单独 sigma 覆盖字典
        #   cfg.clip_ratio / cfg.clip_absolute: 裁剪边界
        #
        # 输出：HandCfg（深拷贝 + 修改 origin.pos）
        #
        # ── 按 scale_mode 分支 ──
        #
        #   [relative 模式]
        #   对每个目标 joint j：
        #     σ_j = per_joint_sigma.get(j.name, sigma)
        #     ε_j ~ N(0, σ_j)，裁剪到 [-clip_ratio, clip_ratio]
        #     pos_new = pos_j × (1 + ε_j)
        #     注意：pos_j 是向量，缩放后保持方向，只改变模长
        #     l_j = ||pos_j||，若 l_j < ε（1e-6）则跳过该关节（长度为零无法缩放）
        #
        #   [absolute 模式]
        #   对每个目标 joint j：
        #     σ_j = per_joint_sigma.get(j.name, sigma)
        #     d_j ~ N(0, σ_j)，若 clip_absolute 不为 None 则裁剪到 [-clip, clip]
        #     l_j = ||pos_j||，方向 n_j = pos_j / l_j
        #     l_new = max(l_j + d_j, ε)  # 保证非负
        #     pos_new = n_j × l_new
        #
        # ── 重建 HandCfg ──
        #   深拷贝 target，对每个目标 joint 修改 origin.pos，返回新对象
        #
        # ── 与 preset 的交叉验证 ──
        #   有效长度公式：l_valid = l_i + d_i（absolute）或 l_i × (1+ε_i)（relative）
        #   若某个 finger preset 规定了关节长度的参考范围，缩放后应在该范围内；
        #   当前草案不强制校验，留给 validator 阶段检查。
        #
        # IDEA：sigma 初始值 0.05 对应 ±5% 扰动，适合在已有手上做小范围局部搜索。


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
