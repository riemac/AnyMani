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
from ._distribution import ScalarDistributionCfg


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

    支持两种模式：末端 collision/visual 几何的形状替换（box ↔ cylinder），
    以及 mesh origin 偏移的随机扰动。
    """

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, ScalarDistributionCfg]:
        r"""描述 tip replace 当前需要的独立采样维度。"""

        target_names = set(self.cfg.target_fingers)
        distribution_plan: dict[str, ScalarDistributionCfg] = {}
        for finger in target.fingers:
            if target_names and finger.name not in target_names:
                continue

            tip_joint = finger.tip_joint
            if self.cfg.mode == "geometry_swap":
                source_kind = _resolve_tip_body_kind(tip_joint)
                target_kind = self.cfg.target_geometry or ("box" if source_kind == "cylinder" else "cylinder")
                if source_kind == target_kind or source_kind not in {"box", "cylinder"}:
                    continue
                if source_kind == "cylinder":
                    for axis_name in ("width", "depth", "length"):
                        distribution_plan[f"{finger.name}::{axis_name}"] = self.cfg.size_distribution.copy()
                else:
                    for axis_name in ("radius", "length"):
                        distribution_plan[f"{finger.name}::{axis_name}"] = self.cfg.size_distribution.copy()
                continue

            for collection_name in ("collisions", "visuals"):
                collection = getattr(tip_joint, collection_name)
                for element_index, element in enumerate(collection):
                    if not isinstance(element.geometry, MeshGeometryCfg):
                        continue
                    for coord_index, value in enumerate(element.origin.pos):
                        if abs(value) <= 1e-12:
                            continue
                        distribution_plan[f"{finger.name}::{collection_name}::{element_index}::{coord_index}"] = (
                            self.cfg.mesh_offset_distribution.copy()
                        )

        return distribution_plan

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行指尖几何替换。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 替换指尖几何后的整手配置。
        """

        mutated = target.copy()
        target_names = set(self.cfg.target_fingers)
        sampled = sampled_params or {}

        for finger in mutated.fingers:
            if target_names and finger.name not in target_names:
                continue

            tip_joint = finger.tip_joint  # 当前约定下末端 joint 就是 tip joint
            if self.cfg.mode == "geometry_swap":
                self._swap_tip_geometry(finger.name, tip_joint, sampled)
            else:
                self._perturb_mesh_tip_origin(finger.name, tip_joint, sampled)

        return mutated

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

    def _swap_tip_geometry(self, finger_name: str, tip_joint, sampled_params: dict[str, Any]) -> None:
        r"""在 `box` 与 `cylinder` 之间替换 tip 主体几何。

        当前把“主体几何”定义为：tip joint 的 collision / visual 列表里，
        第一个非 `sphere` 的 primitive。这样可以兼容当前两种 pre-made tip：

        - `cs`：主体是 cylinder，球帽保持 sphere
        - `bs`：主体是 box，球帽保持 sphere
        """

        body_collision_index = _find_tip_body_index(tip_joint.collisions)
        body_visual_index = _find_tip_body_index(tip_joint.visuals)
        if body_collision_index is None or body_visual_index is None:
            return  # 当前 tip 没有可替换的 primitive 主体时，不做 silent fallback 以外的越权处理

        body_collision = tip_joint.collisions[body_collision_index]
        body_visual = tip_joint.visuals[body_visual_index]
        source_geometry = body_collision.geometry
        source_kind = source_geometry.kind
        target_kind = self.cfg.target_geometry or ("box" if source_kind == "cylinder" else "cylinder")

        if source_kind == target_kind:
            return  # 显式要求的目标几何和当前相同，则本次 mutate 为空操作
        if source_kind not in {"box", "cylinder"} or target_kind not in {"box", "cylinder"}:
            return  # mesh / sphere tip 当前不走 geometry swap

        geometry, origin = _swap_geometry_and_origin(
            finger_name,
            source_geometry,
            body_collision.origin,
            target_kind,
            sampled_params,
        )
        tip_joint.collisions[body_collision_index] = CollisionGeometryCfg(
            name=body_collision.name,
            geometry=geometry,
            origin=origin,
        )
        tip_joint.visuals[body_visual_index] = VisualGeometryCfg(
            name=body_visual.name,
            geometry=geometry,
            origin=origin,
        )

    def _perturb_mesh_tip_origin(self, finger_name: str, tip_joint, sampled_params: dict[str, Any]) -> None:
        r"""对 mesh tip 的局部原点做比例扰动。

        这里不改 mesh 文件路径，也不改 scale，只对 origin.pos 做：

        $$
        p_i' = p_i (1 + \varepsilon_i), \quad \varepsilon_i \sim U(-r, r).
        $$

        绝对值很接近零的分量保持不动，避免“本来就想贴在轴上”的量被噪声抬起来。
        """

        for collection_name in ("collisions", "visuals"):
            collection = getattr(tip_joint, collection_name)
            updated = []
            for element_index, element in enumerate(collection):
                if not isinstance(element.geometry, MeshGeometryCfg):
                    updated.append(element)
                    continue

                pos_new = []
                for coord_index, value in enumerate(element.origin.pos):
                    if abs(value) <= 1e-12:
                        pos_new.append(value)
                        continue
                    epsilon = float(
                        sampled_params.get(
                            f"{finger_name}::{collection_name}::{element_index}::{coord_index}",
                            0.0,
                        )
                    )
                    pos_new.append(value * (1.0 + epsilon))

                updated.append(
                    element.replace(
                        origin=PoseCfg(pos=tuple(pos_new), rpy=element.origin.rpy),
                    )
                )
            setattr(tip_joint, collection_name, updated)


def _find_tip_body_index(elements) -> int | None:
    r"""找到 tip 复合几何里“主体几何”的索引。"""

    for index, element in enumerate(elements):
        if element.geometry.kind != "sphere":
            return index
    return None


def _resolve_tip_body_kind(tip_joint) -> str | None:
    r"""解析当前 tip 主体 primitive 的几何类型。"""

    body_index = _find_tip_body_index(tip_joint.collisions)
    if body_index is None:
        return None
    return tip_joint.collisions[body_index].geometry.kind


def _swap_geometry_and_origin(
    finger_name: str,
    source_geometry,
    source_origin: PoseCfg,
    target_kind: str,
    sampled_params: dict[str, Any],
):
    r"""根据目标几何类型构造替换后的 `(geometry, origin)`。

    这里采用一个保守规则：

    1. `cylinder → box` 时，保留原有局部旋转 `rpy`，这样当前项目里沿 $+y$
       的圆柱轴语义会继续通过 `origin.rpy=(-\pi/2,0,0)` 体现在 box 上；
    2. `box → cylinder` 时，在原 box `rpy` 基础上额外叠加 `(-\pi/2,0,0)`，
       把 URDF 默认沿 $z$ 的圆柱轴转到当前 box 主长度方向；
    3. 尺寸若启用 `size_sigma`，则逐维叠加一个小的绝对扰动，但仍保证正值。
    """

    if source_geometry.kind == "cylinder":
        radius = float(source_geometry.radius)
        length = float(source_geometry.length)
        width = max(2.0 * radius + float(sampled_params.get(f"{finger_name}::width", 0.0)), 1e-9)
        depth = max(2.0 * radius + float(sampled_params.get(f"{finger_name}::depth", 0.0)), 1e-9)
        body_length = max(length + float(sampled_params.get(f"{finger_name}::length", 0.0)), 1e-9)
        return {"type": "box", "size": (width, depth, body_length)}, source_origin

    size = tuple(float(value) for value in source_geometry.size)
    radius = max(min(size[0], size[2]) / 2.0 + float(sampled_params.get(f"{finger_name}::radius", 0.0)), 1e-9)
    length = max(size[1] + float(sampled_params.get(f"{finger_name}::length", 0.0)), 1e-9)
    return (
        {"type": "cylinder", "radius": radius, "length": length},
        PoseCfg(
            pos=source_origin.pos,
            rpy=(source_origin.rpy[0] - math.pi / 2.0, source_origin.rpy[1], source_origin.rpy[2]),
        ),
    )


__all__ = ["TipReplaceCfg", "TipReplaceMutator"]
