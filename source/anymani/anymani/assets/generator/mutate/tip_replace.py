r"""指尖替换工具：对 finger 末端做几何级别的替换或扰动。

对应 `资产生产概略.png` 中两个指尖相关操作：

- **拓扑扰动**：`box / cylinder`（含形状互转：Cylinder ←长轴:宽/短轴:高→ box）
- **mesh 偏移**：已有偏移 ± d% 扰动

以及 `前后序.png` 中归属后序的 `替换指尖`，理由是"末端几何替换"。

分类说明
--------

- **GeometrySwap**：把末端 link 的 collision/visual 基元替换为另一种（box ↔ cylinder）
- **MeshPerturb**：在已有 mesh offset 或 scale 上叠加小范围随机扰动（± d%）

设计说明
--------

### 职责边界

指尖替换只作用于 `FingerCfg.tip_joint`（即 `is_tip=True` 的末端关节对应的 child link）。
它不改变运动学拓扑（不增删关节），只修改末端 link 的 collision/visual 几何描述。

### 形状互转规则（Box ↔ Cylinder）

根据图中注记：

- **Cylinder → Box**：长轴（半径 r）→ 宽；短轴（高 h）→ 高
  即 box.size = (2r, 2r, h)
- **Box → Cylinder**：宽 → 直径（2r）；高 → 高度 h
  即 radius = size[0]/2，height = size[2]

实际实现中，若原几何参数不足以完全还原，可优先保证"体积接近"。

### Mesh 偏移扰动

若末端 link 使用 mesh 几何，`MeshPerturb` 模式在 mesh 的 origin 偏移上
叠加 ± d% 的比例扰动（不改变 mesh 文件路径和 scale 主值）。
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
class TipReplaceCfg(AssetCfgBase):
    r"""指尖替换工具配置。"""

    class_type: type["TipReplaceMutator"] | None = None
    """关联的运行时类。"""

    target_fingers: tuple[str, ...] = ()
    """需要替换指尖的手指名称集合；空元组表示作用于全部手指。"""

    mode: Literal["geometry_swap", "mesh_perturb"] = "geometry_swap"
    """替换模式。

    - ``geometry_swap``：把末端 link 几何从 box/cylinder 替换为另一种；
    - ``mesh_perturb``：在已有 mesh origin 上叠加随机偏移扰动。
    """

    target_geometry: Literal["box", "cylinder"] | None = None
    """目标几何类型（仅 ``geometry_swap`` 模式有效）。为 ``None`` 时自动选择
    与当前类型相反的几何（box → cylinder，cylinder → box）。"""

    size_sigma: float = 0.0
    """几何尺寸扰动强度（meter）；在形状替换基础上叠加小范围随机扰动。
    设为 0.0 表示纯粹做形状替换，不额外引入尺寸随机。"""

    mesh_perturb_ratio: float = 0.05
    """mesh origin 偏移比例扰动强度（仅 ``mesh_perturb`` 模式有效）；
    0.05 表示在原偏移值基础上叠加 ± 5% 的随机扰动。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TipReplaceMutator
        if self.size_sigma < 0:
            raise ValueError(f"size_sigma must be >= 0, got {self.size_sigma}")
        if not 0.0 <= self.mesh_perturb_ratio <= 1.0:
            raise ValueError(
                f"mesh_perturb_ratio must be in [0, 1], got {self.mesh_perturb_ratio}"
            )


# ============================================================================
#  运行时壳
# ============================================================================


class TipReplaceMutator(MutatorBase):
    r"""指尖替换运行时壳。

    支持两种模式：末端 collision/visual 几何的形状替换（box ↔ cylinder），
    以及 mesh origin 偏移的随机扰动。
    """

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行指尖几何替换。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 替换指尖几何后的整手配置。
        """

        pass

        # TODO:算法之一（tip geometry swap）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.target_fingers: 目标手指名集合（空 = 全部）
        #   cfg.mode: "geometry_swap" | "mesh_perturb"
        #   cfg.target_geometry: 目标几何类型（None 时自动取反）
        #   cfg.size_sigma: 几何尺寸扰动强度
        #   cfg.mesh_perturb_ratio: mesh origin 偏移扰动比例
        #
        # 输出：HandCfg（深拷贝 + 末端 link 几何修改）
        #
        # ── [geometry_swap 模式] ──
        #   对每个目标手指 f：
        #     1. 取 f.tip_joint.child 对应 link 的 collision 基元
        #     2. 检测当前几何类型（box / cylinder）
        #     3. 若 target_geometry 为 None，选取相反类型；否则取指定类型
        #     4. 按形状互转规则计算新尺寸参数：
        #        Cylinder（r, h）→ Box(2r, 2r, h)
        #        Box(w, d, h) → Cylinder(radius=w/2, height=h)
        #     5. 若 size_sigma > 0，在新尺寸上叠加 N(0, sigma) 扰动（per-dim）
        #     6. 用新几何对象替换 tip_joint.child 的 collisions/visuals
        #
        # ── [mesh_perturb 模式] ──
        #   对每个目标手指 f：
        #     1. 取 f.tip_joint.child 对应 link 的 mesh collision/visual
        #     2. 对该 mesh 的 origin.pos 各分量施加比例扰动：
        #        pos_new[i] = pos[i] × (1 + ε)，ε ~ U(-ratio, +ratio)
        #        （绝对值接近零的分量跳过，避免数值不稳定）
        #
        # ── 重建 HandCfg ──
        #   深拷贝 target，修改目标 finger 的 tip_joint 几何，返回新对象
        #
        # ── 与 preset 的交叉验证 ──
        #   替换后的几何类型和尺寸应与 preset 的"指尖拓扑"描述兼容；
        #   若 preset 限制了指尖几何类型，应在替换前检查并拒绝不兼容的切换。
        #
        # IDEA：几何替换是纯参数级操作，风险低；但 mesh_perturb 的偏移量若过大
        # 可能导致视觉 / 碰撞不一致，建议在 validator 里对偏移上限做检查。


__all__ = ["TipReplaceCfg", "TipReplaceMutator"]
