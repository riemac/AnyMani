r"""指尖替换变异算子设计草稿。

该算子位于 post-mutate 阶段，替换每根 finger 末端 `tip_joint`
所连接的整个 tip child link embodiment：collision、visual、inertial
以及必要的 metadata / 材质语义应作为同一个 tip spec 一起更新。
它不改变 finger 拓扑、关节数量、关节轴、挂载点或 tip joint 本身在
运动链上的位置。

从科研上看，这个算子处理的是末端接触材料与局部刚体近似，而不是整根
finger 的运动学重建。因此 collision、visual、inertial、metadata 应当被
视为同一个 tip spec 的不同投影，不能只改其中一项就当作完成了 tip replacement。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import (
    BoxGeometryCfg,
    CollisionGeometryCfg,
    CylinderGeometryCfg,
    InertialCfg,
    PoseCfg,
    SphereGeometryCfg,
    Vector2,
    VisualGeometryCfg,
)
from ...builder.joint_builders_primitive import _box_inertia, _cylinder_inertia, _estimate_mass
from ._base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


_TIP_DEFAULT_DENSITY = 650.0
r"""tip replacement 后重算 inertial 时使用的默认密度 $\rho$ [kg/m^3]。

这里沿用 primitive/custom builders 的默认密度，保证 pre-made tip 与 post-mutate
tip 在质量量级上连续，而不是因为变异算子切换了一套隐藏密度假设。
"""


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class TipReplaceCfg(MutatorBaseCfg):
    r"""指尖替换工具配置。

    科研语义上，`tip_replace` 控制的是末端接触皮肤与其刚体物理属性，
    而不是整根手指的 kinematic embodiment。它和 `link_scale` /
    `mount_perturb` 的边界应保持清晰：

    - `link_scale`：改变运动链中 link 的有效长度 / 尺度；
    - `mount_perturb`：改变 finger root 相对 palm 的刚体位姿；
    - `tip_replace`：替换末端 tip child link 的完整物理描述，包括
      collision、visual、inertial、近似质量 / 惯量来源和相关 metadata。

    对手内操作来说，指尖几何、质量分布和接触皮肤近似是接触动力学中
    最敏感的局部变量之一。因此这里的 `self_mode` 不应被理解为普通工程
    开关，而是一次资产采样中全手层面的 morphology / physics coherence 假设。
    """

    class_type: type["TipReplaceMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类。"""

    target_fingers: tuple[str, ...] | None = None
    r"""目标 finger 名称；`None` 表示所有存在 finger 都参与 tip 变异。"""

    mode: Literal["geometry_swap", "mesh_perturb"] = "geometry_swap"
    r"""首版运行时模式：primitive 几何互换，或 mesh tip 局部 offset 微扰。"""

    target_geometry: Literal["box", "cylinder"] | None = None
    r"""`geometry_swap` 的目标主体类型；`None` 时在 box/cylinder 间切换。"""

    self_mode: Literal["general", "same"] | dict[str, float] | None = "same"
    r"""指尖替换的高层形态模式配置。

    该字段描述一次 post-mutate 中“全手的指尖皮肤是否共享同一种宏观假设”。
    它不直接指定某个具体 mesh、半径或缩放值，而是决定 `tip_range` 与 `scale`
    在 finger 之间如何耦合采样。

    支持三种输入语义：
    - `None`：不显式指定高层模式，由运行时使用默认模式，通常等价于 `"same"`。
    - `str`：固定使用某一种模式，例如始终使用 `"general"`。
    - `dict[str, float]`：混合模式采样。键为模式名，值为模式采样概率；
      概率应非负，且概率和应为 1。

    预设模式：
    - `"general"`：每根目标 finger 独立采样完整 tip spec。
      也就是说，thumb / index / middle / ring 的指尖类型可以不同，
      连续参数也可以不同：
      $$
      s_f \sim p(s),\quad \theta_f \sim p(\theta \mid s_f),\quad f\in\mathcal{F}.
      $$
      这里 $s_f$ 表示离散 tip 类型，$\theta_f$ 表示该类型下由 preset 与 `scale`
      共同确定的完整 tip spec。该模式最大化接触皮肤与局部物理属性 diversity，
      适合做 domain randomization，但会采到“不像真实同一只手”的组合。
    - `"same"`：全体目标 finger 共享同一个完整 tip spec。
      运行时应只采样一次
      $$
      s \sim p(s),\quad \theta \sim p(\theta \mid s),
      $$
      然后广播到所有目标 finger。这里的“一致”包括 tip 类型、缩放值，以及由
      preset/lowering 推导出的局部几何、质量、密度、惯量近似策略等参数；
      唯一不共享的是每根 finger 已有的运动链位置，例如
      `finger.tip_joint.origin` 和 parent / child link 名。
      该模式更像一种 hand-family coherent morphology：同一只手的所有指尖皮肤
      来自同一设计族。

    # NOTE:
    `same` 不意味着把所有 finger 的 tip joint 移到同一个空间位置；它同步的是
    tip child link 的完整物理 spec。finger chain 的长度和 tip joint frame 仍由
    pre-made / `link_scale` 决定。
    """

    tip_range: list[str] | dict[str, float] | None = None
    r"""指尖候选范围。

    该字段描述离散 tip 类型的候选集合，而不是 mesh 文件路径集合。合法名称包括：
    - primitive tip recipe：例如 `"cs"`；
    - `CustomTipBuilderCfg` 能通过 `tip_type` resolve 的 custom tip key：
      例如 `"round"`、`"leap_cube"`、`"wedge"`、`"thinner"`。

    输入语义：
    - `None`：默认包含当前 pre-made 手型原本的 tip 类型，以及所有已注册合法 custom tip。
    - `list[str]`：离散列出合法 tip 类型，默认每个 tip 被采样的概率相等。
    - `dict[str, float]`：显式给出不同 tip 类型的采样概率；概率应非负，且概率和为 1。

    # NOTE:
    custom tip 是否“合法”，不取决于它是否只是出现在 `assets/custom/tips` 文件夹中，
    而取决于它是否能由 `CustomTipBuilderCfg` 自动补齐 mesh_path、anchor_point、
    unit_scale、base_rpy、approx_size 等 lowering 所需语义。
    """

    scale: Vector2 | dict[str, Vector2] = (1.0, 1.0)
    r"""指尖尺寸缩放范围。

    该字段描述对 tip spec 的无量纲比例缩放，例如 `(0.9, 1.1)` 表示采样一个
    $s\in[0.9, 1.1]$ 的 scale。缩放示意图见
    `AnyMani/source/anymani/anymani/assets/doc/指尖scale示意.png`。

    输入语义：
    - `Vector2`：所有 tip 类型共享同一个缩放范围。
    - `dict[str, Vector2]`：不同 tip 类型使用各自的缩放范围；没有显式列出的 tip
      可回退到 `(1.0, 1.0)` 或运行时默认策略。

    几何语义：
    - scale 不移动 `tip_joint`，也不改变上一段 link 的末端位置；
    - 对 custom mesh tip，缩放应围绕 preset 中的语义锚点 `anchor_point` 重新 lowering。
      换句话说，指尖 mesh 底部锚点仍与上一 link 顶部 / tip joint frame 对齐，
      而不是简单地把 mesh origin 乘以 scale；
    - 对 primitive tip，运行时应同步缩放半径、宽度、深度、高度等几何参数，并重建
      collision、visual 与 inertial，保持底部接合处和上一 link 中心线在局部
      $z$-$x$ 平面内重合。

    默认 `(1.0, 1.0)` 表示只替换 tip 类型，不额外引入尺寸缩放。
    """

    _resolved_self_mode: str | None = field(init=False, default=None, repr=False)
    r"""内部字段：运行时解析后的单一 self mode。"""

    def __post_init__(self):
        r"""补齐运行时类并缓存归一化后的模式信息。

        这里允许少量内部解析缓存，但不把复杂分布对象重新引入成公开字段；
        研究者只需要看 `tip_range`、`scale`、`self_mode` 这些公开语义就够了。
        """

        if self.class_type is None:
            self.class_type = TipReplaceMutator
        if isinstance(self.target_fingers, list):
            self.target_fingers = tuple(self.target_fingers)
        self._resolved_self_mode = "same" if self.self_mode is None else str(self.self_mode)


class TipReplaceMutator(MutatorBase):
    r"""指尖替换运行时壳。

    按当前 post-mutator 的 Declare / Sample / Apply 设计实现 tip replacement。
    输入：
    - `HandCfg`：已经由 pre-made builder 或前序 post-mutator 生成的手资产；
    - `TipReplaceCfg`：描述目标 finger、tip 候选集合、连续参数范围与 self mode；
    - 可选 `sampled_params`：用于测试 / 可复现生成的外部采样结果，内容应是聚合
      `TipSpec` 或可 lowering 成 `TipSpec` 的最小参数，而不是零散物理字段。

    输出：
    - 新的 `HandCfg` 或原地可接受的 mutated cfg，具体跟随 pipeline 约定；
    - 每个目标 finger 的 `tip_joint.collisions` / `tip_joint.visuals` /
      `tip_joint.inertial` / `tip_joint.metadata` 被替换或更新为同一套 tip spec
      lowering 后的结果；
    - 不修改 `finger.joints[:-1]`、`finger.mount`、`tip_joint.origin`、`tip_joint.parent`
      和 `tip_joint.child`。

    Sample 层伪算法：
    1. 解析目标 finger 集合 `F`；默认取 `hand.fingers` 全部存在项。
    2. 解析 `self_mode`：
       - `general`：对每个 `f in F` 调用一次 `_sample_tip_spec()`；
       - `same`：调用一次 `_sample_tip_spec()`，并复制给所有 `f in F`。
    3. `_sample_tip_spec()` 先从 `tip_range` 采样离散候选 `s`，再从 `scale`
       采样缩放值，并结合 primitive recipe 或 custom tip preset 形成完整 `TipSpec`。
    4. 返回一组 planned edits，而不是立即随意改 `HandCfg`：
       `finger_name -> TipSpec`，或更底层的
       `attribute_path -> sampled_value`，具体跟随 `HandMutator` 最终 Apply 协议。

    Apply / lowering 层伪算法：
    1. 将 `TipSpec` lowering 为完整的 tip child link spec：
       - primitive `cs/bs` 可复用 `ComPrimJointBuilder` 的几何公式；
       - mesh preset 可复用 `CustomTipBuilderCfg` 的 anchor formula；
       - `InertialCfg` 必须随 tip spec 重建，包含质量、质心和惯量张量；
       - collision 与 visual 默认保持几何一致，避免训练时接触皮肤和可视皮肤语义分裂；
       - metadata 记录 `tip_type`、采样参数、inertia approximation 和 mesh anchor 信息，
         方便后续按资产 family 做 ablation。
    2. 写回每个 finger 的 `tip_joint`，同时保留 joint/link 命名和原有必要 metadata。

    验收条件：
    - `same` 下所有目标 finger 的 tip 类型和连续参数完全一致；
    - `general` 下每个目标 finger 可独立拥有不同 tip 类型与参数；
    - 替换后 `tip_joint.is_tip is True`，且 finger chain parent/child 关系不断裂；
    - primitive / mesh tip 都能导出 URDF，并通过 hand validator。
    """

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""把 tip 高层模式 lowering 成 pipeline 可批量采样的局部变量。

        `tip_replace` 不再暴露 `size_distribution` 或 `mesh_offset_distribution`。
        连续随机性统一来自用户笔记中的 `scale` 字段；离散 tip 类型选择留给
        `tip_range/self_mode` 的后续完整 lowering。
        """

        # 先按 finger 粒度分发，再按 mode 决定是共享还是独立采样。
        # 这样能保持 same/general 两类语义都能从同一个 lowering 路径落地。
        specs: dict[str, Any] = {}
        target_fingers = list(_iter_target_fingers(target, self.cfg.target_fingers))
        if _resolved_self_mode(self.cfg) == "same" and target_fingers:
            specs["shared::scale"] = _make_range_sampler(
                _scale_range_for_tip(self.cfg.scale, "shared"),
                distrib="uniform",
                boundary_policy="none",
            )
            return specs
        for _, finger in target_fingers:
            specs[f"{finger.name}::scale"] = _make_range_sampler(
                _scale_range_for_tip(self.cfg.scale, _current_tip_type(finger.tip_joint)),
                distrib="uniform",
                boundary_policy="none",
            )
        return specs

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""生成 tip child-link 的延迟 patch。

        该函数严格不修改 `tip_joint.origin`、`parent`、`child` 和 finger 链关系；
        只替换 tip child link 内部的 collision/visual/inertial 局部物理描述。
        """

        sampled_params = sampled_params or {}
        patch = HandPatch()
        shared_scale = float(sampled_params.get("shared::scale", 1.0))
        for finger_index, finger in _iter_target_fingers(target, self.cfg.target_fingers):
            if self.cfg.mode == "geometry_swap":
                scale_factor = shared_scale if "shared::scale" in sampled_params else float(sampled_params.get(f"{finger.name}::scale", 1.0))

                def apply_swap(hand: HandCfg, *, fi=finger_index, scale=scale_factor) -> None:
                    tip_joint = hand.fingers[fi].tip_joint
                    _swap_primitive_tip_body(tip_joint, target_geometry=self.cfg.target_geometry, scale_factor=scale)

                patch.add(("finger", finger_index, "tip", "geometry_swap"), apply_swap)
            elif self.cfg.mode == "mesh_perturb":
                scale_factor = shared_scale if "shared::scale" in sampled_params else float(sampled_params.get(f"{finger.name}::scale", 1.0))

                def apply_mesh_perturb(hand: HandCfg, *, fi=finger_index, scale=scale_factor) -> None:
                    _scale_mesh_tip_offsets(hand.fingers[fi].tip_joint, scale)

                patch.add(("finger", finger_index, "tip", "mesh_perturb"), apply_mesh_perturb)
        return patch


def _iter_target_fingers(hand: HandCfg, target_fingers: tuple[str, ...] | None):
    """按配置解析目标 finger 集合。"""

    target_set = set(target_fingers or ())
    for finger_index, finger in enumerate(hand.fingers):
        if target_set and finger.name not in target_set:
            continue
        yield finger_index, finger


def _swap_primitive_tip_body(tip_joint, *, target_geometry: str | None, scale_factor: float) -> None:
    r"""替换 primitive tip 主体，保留 sphere cap 与 tip joint 位姿。

    首版只处理当前 builder 产生的 `cs/bs` 风格：主体是 collision/visual
    列表第 0 个，cap 是第 1 个球体。这样能满足 quick 验收，同时不把完整
    tip preset lowering 过早做死。
    """

    # 如果 tip 还没有完整的 collision / visual 结构，就不要强行替换，
    # 否则会把一个不完整输入伪装成已完成的 tip spec。
    if not tip_joint.collisions or not tip_joint.visuals:
        return
    body = tip_joint.collisions[0].geometry
    if body.kind not in {"box", "cylinder"}:
        return
    new_kind = target_geometry or ("box" if body.kind == "cylinder" else "cylinder")
    radius, length, width, depth, cap_radius = _tip_body_dimensions(tip_joint, scale_factor=scale_factor)
    if new_kind == "box":
        geometry = BoxGeometryCfg(size=(width, length, depth))
        origin = PoseCfg(pos=(0.0, length / 2.0, 0.0))
    else:
        geometry = CylinderGeometryCfg(radius=radius, length=length)
        origin = PoseCfg(pos=(0.0, length / 2.0, 0.0), rpy=(-1.5707963267948966, 0.0, 0.0))
    tip_joint.collisions[0] = CollisionGeometryCfg(name=tip_joint.collisions[0].name, geometry=geometry, origin=origin)
    tip_joint.visuals[0] = VisualGeometryCfg(name=tip_joint.visuals[0].name, geometry=geometry, origin=origin)
    # 几何改变以后，惯性必须一起重建，否则质量和接触皮肤语义会拆开。
    tip_joint.inertial = _estimate_swapped_tip_inertial(
        new_kind=new_kind,
        radius=radius,
        length=length,
        width=width,
        depth=depth,
        cap_radius=cap_radius,
    )
    tip_joint.metadata = {**tip_joint.metadata, "post_mutate_tip_mode": "geometry_swap", "post_mutate_tip_body": new_kind}


def _tip_body_dimensions(tip_joint, *, scale_factor: float) -> tuple[float, float, float, float, float]:
    r"""从现有 tip 主体和 cap 估计互换后的保守尺寸。"""

    body = tip_joint.collisions[0].geometry
    cap_radius = 0.006
    if len(tip_joint.collisions) > 1 and tip_joint.collisions[1].geometry.kind == "sphere":
        cap_radius = float(tip_joint.collisions[1].geometry.radius)
    if body.kind == "box":
        width, length, depth = body.size
        radius = max(width, depth) / 2.0
    else:
        radius = float(body.radius)
        length = float(body.length)
        width = depth = 2.0 * radius
    scale = max(1e-4, float(scale_factor))
    length = max(1e-4, float(length) * scale)
    width = max(1e-4, float(width) * scale)
    depth = max(1e-4, float(depth) * scale)
    radius = max(1e-4, float(radius) * scale, cap_radius * 0.5)
    return radius, length, width, depth, cap_radius


def _estimate_swapped_tip_inertial(
    *,
    new_kind: str,
    radius: float,
    length: float,
    width: float,
    depth: float,
    cap_radius: float,
) -> InertialCfg:
    r"""为 geometry-swapped primitive tip 重算质量、质心与惯量。

    `tip_replace` 改变的是 tip child link 的几何皮肤，因此 inertial 也必须随
    geometry 一起更新。这里采用和 primitive builder 一致的首版近似：

    - 主体为 cylinder 时，用圆柱体积估质量；
    - 主体为 box 时，用长方体体积估质量；
    - cap 仍近似为完整 sphere，和 pre-made `cs/bs` tip 规则保持一致；
    - 整体惯量用等效 cylinder / box 表达，保证正定且量级合理。
    """

    # 球帽仍按完整 sphere 估体积，因为当前 primitive tip 的语义就是
    # “主体 + cap” 的二段式近似。
    cap_mass = _estimate_mass(
        volume=4.0 * math.pi * cap_radius**3 / 3.0,
        cfg_mass=None,
        density=_TIP_DEFAULT_DENSITY,
    )  # 球帽仍按完整球估体积，保持和 pre-made primitive tip 一致
    cap_com_y = length  # 球帽中心落在主体末端中心

    if new_kind == "box":
        body_mass = _estimate_mass(
            volume=width * length * depth,
            cfg_mass=None,
            density=_TIP_DEFAULT_DENSITY,
        )  # box 主体体积 $V=wld$
        total_mass = body_mass + cap_mass
        com_y = (body_mass * (length / 2.0) + cap_mass * cap_com_y) / total_mass
        return InertialCfg(
            mass=total_mass,
            origin=PoseCfg(pos=(0.0, com_y, 0.0)),
            inertia=_box_inertia((width, length + 2.0 * cap_radius, depth), total_mass),
        )

    body_mass = _estimate_mass(
        volume=math.pi * radius * radius * length,
        cfg_mass=None,
        density=_TIP_DEFAULT_DENSITY,
    )  # cylinder 主体体积 $V=\pi r^2l$
    total_mass = body_mass + cap_mass
    com_y = (body_mass * (length / 2.0) + cap_mass * cap_com_y) / total_mass
    return InertialCfg(
        mass=total_mass,
        origin=PoseCfg(pos=(0.0, com_y, 0.0)),
        inertia=_cylinder_inertia(radius, length + 2.0 * cap_radius, total_mass),
    )


def _scale_mesh_tip_offsets(tip_joint, scale_factor: float) -> None:
    r"""对 mesh tip 的局部 x/z origin 做比例缩放，保持 y 轴贴合语义不动。"""

    # mesh tip 的 y 轴一般是贴合 / 延伸方向，因此这里只缩放横向展开轴，
    # 避免把接合方向也一并扯大。
    for collection_name in ("collisions", "visuals"):
        collection = getattr(tip_joint, collection_name)
        for geom_index, element in enumerate(collection):
            pos = list(element.origin.pos)
            for axis_index in (0, 2):
                pos[axis_index] = pos[axis_index] * float(scale_factor)
            collection[geom_index] = element.replace(origin=PoseCfg(pos=tuple(pos), rpy=element.origin.rpy))
    tip_joint.metadata = {**tip_joint.metadata, "post_mutate_tip_mode": "mesh_perturb"}


def _resolved_self_mode(cfg: TipReplaceCfg) -> str:
    r"""解析 `self_mode`，首版 dict 混合模式默认落到 `"same"`。"""

    if isinstance(cfg.self_mode, str):
        return cfg.self_mode
    return "same"


def _scale_range_for_tip(scale: Vector2 | dict[str, Vector2], tip_type: str) -> Vector2:
    r"""按 tip 类型解析无量纲 scale 采样范围。"""

    if isinstance(scale, dict):
        return scale.get(tip_type, scale.get("shared", (1.0, 1.0)))
    return scale


def _current_tip_type(tip_joint) -> str:
    r"""从 metadata 中取当前 tip 类型，缺失时回退到 `shared`。"""

    return str(tip_joint.metadata.get("tip_type", tip_joint.metadata.get("post_mutate_tip_body", "shared")))


__all__ = ["TipReplaceCfg", "TipReplaceMutator"]
