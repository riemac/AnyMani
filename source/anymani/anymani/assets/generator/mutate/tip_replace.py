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
import math
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import CollisionGeometryCfg, MeshGeometryCfg, PoseCfg, VisualGeometryCfg
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

    size_distribution: ScalarDistributionCfg = field(
        default_factory=lambda: ScalarDistributionCfg(kind="fixed", value=0.0)
    )
    """几何尺寸扰动分布；采样值直接叠加到 shape swap 后的线性尺寸上。"""

    mesh_offset_distribution: ScalarDistributionCfg = field(
        default_factory=lambda: ScalarDistributionCfg(kind="uniform", low=-0.05, high=0.05)
    )
    """mesh origin 偏移比例扰动分布；采样值 $\varepsilon$ 用在 $p'=p(1+\varepsilon)$。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TipReplaceMutator


class TipReplaceMutator(MutatorBase):
    r"""指尖替换运行时壳。

    """

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        self.cfg = cfg


__all__ = ["TipReplaceCfg", "TipReplaceMutator"]
