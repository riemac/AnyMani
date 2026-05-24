r"""连杆长度缩放变异算子：在已有 `HandCfg` 上对 link 两岸距离做扰动。

科研语义上，这个算子对应的是“同一条运动学链上，某一段主体几何的轴向
长度发生轻微变化”，而不是重新生成一个新的 hand family。它必须结合
`AnyMani/source/anymani/anymani/assets/doc/长度和宽度变异示意.jpg` 和 `AnyMani/source/anymani/anymani/assets/doc/拇指连杆尺寸变异示意图.png` 理解：

- `link_scale` 只改变自身有效长度 $L_i$；
- 不缩放 mesh offset $d_i$，也不改视觉锚点的局部语义；
- 下游 joint / tip origin 使用新的 $L_i+d_i$ 重新放置；
- 不改变 topology、DOF、parent-child 串联关系或 joint axis。

因此该算子更接近“几何长度的局部再标定”，而不是“随便把 link 拉伸一遍”。
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
import math
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import EllipticCylinderGeometryCfg, PoseCfg, Vector2, Vector6
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


@dataclass
class LinkScaleCfg(MutatorBaseCfg):
    r"""连杆尺寸缩放算子配置类。

    主要是对 pre-made 产生后的某具体拓扑灵巧手 (HandCfg) 其手指
    joint/child link 的尺寸进行缩放变异，不包括手掌和指尖。缩放变异
    是在原有 HandCfg 基础上，对其尺寸进行扰动变异，是增量型的，而非
    重新在指定范围赋予新的尺寸值。

    科研上这里遵守 `doc/长度变异示意.jpg` 的第一性原理：

    - 改的是 link 的有效长度 $L_i$；
    - 不改 mesh 的贴图/视觉锚点偏移 $d_i$；
    - 下游 joint / tip origin 只根据新的 $L_i + d_i$ 重新放置。

    这样做的意义是把“几何主体大小变化”和“局部装配偏移”分开，
    避免把视觉语义和运动语义混成一团。
    """

    class_type: type["LinkScaleMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类；由配置层把 schema 绑定到真正的执行器。"""

    link_scale: Vector2 | Vector6 = field(default=MISSING)
    r"""连杆尺寸变异范围配置。

    这个字段的设计必须同时结合两张局部图理解：

    - `doc/长度和宽度变异示意.jpg`
    - `doc/拇指连杆尺寸变异示意图.png`

    当前已经实现的是 ``Vector2`` 的长度变异；``Vector6`` 是下一步要实现的
    长宽高联合变异设计锚点。

    - ``Vector2``：$l=(l_{\min}, l_{\max})$，只控制主长度方向；
    - ``Vector6``：$(l_{\min}, l_{\max}, w_{\min}, w_{\max}, h_{\min}, h_{\max})$。

    长度、宽度、高度的采样耦合关系并不相同：

    - 长度 $l_i$ 是 per-link 独立随机变量。即使所有 link 共享同一组
      ``(l_min, l_max)`` 配置，每个 link 仍各自 draw 一次。
    - 宽度 $w$ 与高度 $h$ 是全局共享随机变量。每次 mutate 只采样一个
      ``w_scale`` 和一个 ``h_scale``，再共同作用到所有可变 link mesh。
      这样得到的是“同一只手整体变粗/变薄/变高/变矮”的 morphology
      family variation，而不是每节 link 横截面各自乱跳。

    non-thumb 语义：

    - 长度变异只改变当前 link 自身有效长度 $L_i$；
    - mesh offset $d_i$ 不随长度缩放；
    - 下游 joint origin 按新的 $L_i+d_i$ 推进；
    - 宽度/高度变异只改 box/cylinder 横截面尺寸，不改变 joint frame，
      不改变 mesh frame origin。当前适用前提是 $x/z$ 方向没有有效
      mesh offset，即横截面本来就在 joint/mesh 语义中心附近对齐。

    thumb 语义：

    - CMC1 是特殊 link：零 offset 时 mesh frame 与 joint frame 完全重合，
      不能套普通 link 的 $L/2+d_y$ mesh origin 规则；
    - CMC1 -> CMC2 的 origin 由 CMC1 的长度、宽度、高度和 CMC2 横截面共同决定；
    - 因此 CMC1 的长度变异会影响 CMC2 的 $y$ 位置，宽度变异会影响
      CMC2 的 $x$ 对齐，高度变异会影响 CMC2 的 $z$ 对齐；
    - CMC2/MCP/DIP 之后的串联关系可以继续近似使用 non-thumb 的规则。

    # TODO(link-scale-vector6): 实现 ``Vector6`` 的 sampler lowering。
    # 输入：`link_scale` / `clip` 中的 l/w/h 三组范围。
    # 输出：per-link 的 length samples，加上 shared::width/shared::height。
    # 约束：长度独立采样；宽度和高度每个 batch sample 只采一次并全局共享。
    # 验收：metadata 中能看出每个 link 的 length scale 不同，但所有 link 的
    # width scale / height scale 相同。

    # TODO(link-scale-nonthumb-width-height): 实现 non-thumb 横截面缩放。
    # 输入：某 joint 的 collision/visual 主体几何与 shared w/h scale。
    # 输出：box.size.x / box.size.z 或 cylinder.radius 的对称缩放 patch。
    # 约束：不改 joint.origin，不改 element.origin，不改 mesh offset d_i。
    # 验收：URDF 中 link 横截面尺寸变化，但各 joint origin 与 visual/collision
    # origin 的 xyz/rpy 不因 w/h scale 漂移。

    # TODO(link-scale-thumb-cmc1): 实现 CMC1 的专用下游 origin 重解算。
    # 输入：CMC1 新长度/宽度/高度、CMC1 mesh offset、CMC2 当前横截面尺寸。
    # 输出：CMC2 origin = ((W_cmc1-W_cmc2)/2, d_y+L_cmc1/2, d_z-(H_cmc1-H_cmc2)/2)。
    # 约束：CMC1 自身 mesh origin 保持 center_on_joint 语义；CMC2 之后按普通链推进。
    # 验收：对照 `拇指连杆尺寸变异示意图.png`，CMC1 放大后 CMC2 仍贴在图示边界上。
    """

    link_type: str = "box"
    r"""Joint/child link mesh 的种类，默认是 URDF 中最常见的 ``"box"``。

    这里保留的是“几何主体的表示类型”，而不是最终物理量。也就是说，
    `link_type` 决定后续应当按 box/cylinder 的哪一套几何公式去理解
    `link_scale`，而不是直接决定质量或惯量模型。
    """

    scale_type: Literal["abs", "rel"] = "rel"
    r"""缩放语义：绝对长度扰动 ``abs``，或相对比例扰动 ``rel``。

    这两个模式的第一性原理不同：
    - ``rel`` 更像围绕原尺寸做局部 family variation，适合 pre-made /
      post-mutate 的轻扰动；
    - ``abs`` 更像把某个维度直接当作几何增量，适合精细实验中对绝对
      尺寸锚点的控制。
    """

    clip: Vector2 | Vector6 | None = None
    r"""扰动后尺寸的裁剪范围；`None` 表示只使用 `link_scale` 的采样边界。

    这里的裁剪不是为了“美化输出”，而是为了把异常采样截断在合理几何域
    内，避免极端样本把链长推到接近零或负值。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""分布类型。可选正态分布/均匀分布；不同 joint 每次采样互相独立。

    这意味着联采样时每个 joint 的尺度扰动都被视作独立随机变量，
    但它们共享同一套高层配置约束，例如同一 `boundary_policy` 和同一
    `scale_type`。这种设计是为了把“分布假设”与“几何解释”解耦。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""采样越界时的边界处理策略。

    这是采样语义的一部分，不是额外的几何规则。换言之，`distrib` 定义
    基础随机性，`boundary_policy` 定义样本超出允许域后如何解释。
    """

    _link_meshes: list[Any] = field(default_factory=list)
    r"""内部使用的 link mesh 列表，预留给后续更细粒度长宽高解析。

    之所以放在私有字段里，是因为它更像运行时缓存，而不是研究者需要
    直接调节的公开超参。
    """

    _distribution: Any = field(init=False, repr=False)
    r"""内部采样器占位；不作为 public distribution schema 暴露。

    这个字段只服务执行期的 sampler lowering，不应成为外部实验接口的一部分。
    """

    def __post_init__(self) -> None:
        r"""补齐运行时类并校验必填范围字段。

        这里的校验只关心最小可执行条件，不抢占 higher-level 决策：
        运行时类绑定、采样范围是否显式给出，以及必要的配置是否能被
        lower 成 patch 逻辑所需的数值形式。
        """

        self.class_type = LinkScaleMutator
        if self.link_scale is MISSING:
            raise ValueError("LinkScaleCfg.link_scale must be set explicitly")
        if isinstance(self.link_scale, dict):
            raise TypeError("LinkScaleCfg.link_scale no longer supports dict[str, ...]; use Vector2 or Vector6 only")
        if isinstance(self.clip, dict):
            raise TypeError("LinkScaleCfg.clip no longer supports dict[str, ...]; use Vector2, Vector6 or None only")


class LinkScaleMutator(MutatorBase):
    r"""连杆长度缩放运行时壳。

    在已构建好的 `HandCfg` 上对目标关节 child link 的有效长度做扰动，
    不改变拓扑与旋转。

    这里的“有效长度”并不是抽象数值，而是实际几何主体在运动学链上
    占据的那段轴向长度，因此它会直接影响下游 joint 起点位置。
    """

    cfg: LinkScaleCfg

    def __init__(self, cfg: LinkScaleCfg):
        r"""绑定一份 `LinkScaleCfg`。

        运行时壳本身不再承载科研语义，只负责把配置里已经定义好的
        采样语义与 patch 生成语义串起来。
        """

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""为每个目标 joint 声明长度扰动随机变量。

        返回的 key 使用 joint name 作为语义锚点，因为在 post-delete /
        regroup 之后，数字 index 可能会变，但 child link / joint name 仍然
        是我们跨版本追踪几何语义的稳定标识。

        # TODO(link-scale-vector6): 当 `link_scale` 为 Vector6 时，这里应声明：
        # - `{joint.name}::length`：每个 link 独立 draw；
        # - `shared::width`：整次 sample 共享 draw；
        # - `shared::height`：整次 sample 共享 draw。
        # 当前实现仍然只声明 `{joint.name}` 并只消费 Vector6 前两个元素。
        """

        # 先把所有可扰动 joint 的局部采样语义一次性声明出来。
        # 这样 pipeline 看到的是“联合参数表”，而不是互相串联的逐步修改。
        specs: dict[str, Any] = {}
        shared_ranges = _shared_cross_section_ranges(self.cfg.link_scale, self.cfg.clip, target=target)
        if shared_ranges is not None:
            width_range, height_range = shared_ranges
            specs["shared::width"] = _make_range_sampler(
                width_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )  # 全手共享的宽度缩放随机变量
            specs["shared::height"] = _make_range_sampler(
                height_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )  # 全手共享的高度缩放随机变量
        for _, _, joint in _iter_target_joints(target):
            value_range = _range_for_joint(self.cfg.link_scale, joint.child)
            clip_range = _range_for_joint(self.cfg.clip, joint.child) if self.cfg.clip is not None else None
            specs[f"{joint.name}::length"] = _make_range_sampler(
                _length_range(value_range, clip_range),
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        return specs

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""生成 link length 与下游 joint origin 的 deferred patch。

        这一步只生成 patch，不直接修改原对象；这样 joint 长度的修改和
        下游 origin 的推进可以作为一组原子语义来处理，而不是两次独立写入。

        # TODO(link-scale-vector6): 将 patch 拆成三个语义层：
        # 1. 写回当前 link 的 length/width/height 主体尺寸；
        # 2. 对 non-thumb 和 thumb 非 CMC1 段，按 `L_i+d_i` 推进下游 origin；
        # 3. 对 thumb CMC1，按 CMC1 专用公式同时重解算 CMC2 的 x/y/z origin。
        # 这三层必须仍然基于同一个原始 HandCfg 生成 deferred patch，不能让
        # 宽高 patch 先污染长度 patch 的几何读取。
        """

        sampled_params = sampled_params or {}
        patch = HandPatch()
        shared_width = float(sampled_params["shared::width"]) if "shared::width" in sampled_params else None
        shared_height = float(sampled_params["shared::height"]) if "shared::height" in sampled_params else None
        patch.metadata.setdefault("post_mutate_link_scale", {})
        patch.metadata["post_mutate_link_scale"]["width_scale"] = shared_width  # sidecar 中记录当前全手共享的 semantic 宽度缩放
        patch.metadata["post_mutate_link_scale"]["height_scale"] = shared_height  # sidecar 中记录当前全手共享的 semantic 高度缩放
        patch.metadata["post_mutate_link_scale"]["length_scale"] = {}  # 每个 link 的长度缩放单独记录

        # 先遍历每个目标 joint，再为当前 joint 自己和下游 joint 生成
        # 一组最小 patch。这里的核心原则是：同一个源长度变动，只产生
        # 一次语义上闭合的几何更新，而不是让后续算子再去猜测这个变化。
        for finger_index, joint_index, joint in _iter_target_joints(target):
            delta_or_ratio = float(sampled_params.get(f"{joint.name}::length", sampled_params.get(joint.name, 0.0)))
            old_length = _joint_primary_length(joint)
            if old_length is None:
                continue
            new_length = _mutated_length(old_length, delta_or_ratio, self.cfg)
            if new_length <= 1e-6:
                continue
            patch.metadata["post_mutate_link_scale"]["length_scale"][joint.child] = delta_or_ratio  # 每个 child link 记录自身长度缩放

            old_cross_section = _joint_cross_section(joint)  # 读取当前 link 的 local $(x,z)$ 横截面，用于后续 geometry patch
            new_cross_section = _mutated_cross_section(
                old_cross_section,
                width_scale=_semantic_width_scale_for_joint(joint, semantic_width_scale=shared_width, semantic_height_scale=shared_height),
                height_scale=_semantic_height_scale_for_joint(joint, semantic_width_scale=shared_width, semantic_height_scale=shared_height),
            )  # shared 宽高先按 semantic 语义采样，再按当前 joint 是否为 thumb 映射到 local $(x,z)$
            is_cmc1 = str(joint.child).endswith("_cmc1")

            def apply_link(
                hand: HandCfg,
                *,
                fi=finger_index,
                ji=joint_index,
                old=old_length,
                new=new_length,
                old_cross=old_cross_section,
                new_cross=new_cross_section,
                cmc1=is_cmc1,
            ) -> None:
                r"""写回当前 joint child link 的新有效长度。"""

                mutated_joint = hand.fingers[fi].joints[ji]
                _set_joint_primary_geometry(
                    mutated_joint,
                    old_length=old,
                    new_length=new,
                    old_cross_section=old_cross,
                    new_cross_section=new_cross,
                    keep_center=cmc1,
                )

            patch.add(("finger", finger_index, "joint", joint_index, "link_geometry"), apply_link)

            next_index = joint_index + 1
            if next_index < len(target.fingers[finger_index].joints):
                next_joint = target.fingers[finger_index].joints[next_index]
                next_old_cross_section = _joint_cross_section(next_joint)  # CMC1 -> CMC2 对齐时，下游段自己的横截面也必须使用同一次 mutate 后的新语义
                next_new_cross_section = _mutated_cross_section(
                    next_old_cross_section,
                    width_scale=_semantic_width_scale_for_joint(
                        next_joint,
                        semantic_width_scale=shared_width,
                        semantic_height_scale=shared_height,
                    ),
                    height_scale=_semantic_height_scale_for_joint(
                        next_joint,
                        semantic_width_scale=shared_width,
                        semantic_height_scale=shared_height,
                    ),
                )  # 这里即便下游段长度不变，也要先把它的 semantic 宽高映射到 local 截面，供 thumb 边界对齐使用
                next_origin = _next_origin_from_link_scale(
                    current_joint=joint,
                    next_joint=next_joint,
                    old_length=old_length,
                    new_length=new_length,
                    old_cross_section=old_cross_section,
                    new_cross_section=new_cross_section,
                    next_old_cross_section=next_old_cross_section,
                    next_new_cross_section=next_new_cross_section,
                )  # thumb CMC1 与普通串联链在这里分流

                def apply_next_origin(hand: HandCfg, *, fi=finger_index, ni=next_index, origin=next_origin) -> None:
                    r"""按新的几何边界重解算下游 joint origin。"""

                    hand.fingers[fi].joints[ni].origin = origin

                patch.add(("finger", finger_index, "joint", next_index, "origin_from_link_scale", joint.name), apply_next_origin)

        return patch


def _iter_target_joints(hand: HandCfg):
    r"""遍历所有可被 link_scale 处理的 revolute 非 tip joint。"""

    # 这里严格跳过 tip，因为 tip 的几何和接触语义不应该和中间链段
    # 共用同一套长度缩放逻辑。
    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.is_tip:
                continue
            if _joint_primary_length(joint) is None:
                continue
            yield finger_index, joint_index, joint


def _range_for_joint(config: Vector2 | Vector6 | None, child_name: str) -> Vector2 | Vector6 | None:
    r"""返回某个 joint 的范围配置。

    当前 `link_scale` 与 `clip` 已经显式出清 `dict[str, ...]` 形式，因此这里的
    “按 joint 解析”只保留一个很薄的兼容入口：对所有 joint 直接返回同一份全局范围。
    保留这个 helper 的原因，是后续若重新引入更细粒度 façade，不需要再改动
    `describe_sampling()` / `plan_patch()` 的调用形状。
    """

    _ = child_name  # 当前全局范围对所有 child link 一视同仁；保留参数位只是为了接口稳定
    return config


def _length_range(value_range: Vector2 | Vector6, clip_range: Vector2 | Vector6 | None) -> Vector2:
    r"""返回当前 link 主长度方向的最终采样区间。"""

    low = float(value_range[0])  # 主长度扰动下界
    high = float(value_range[1])  # 主长度扰动上界
    if clip_range is not None:
        clip_low = float(clip_range[0])  # 长度 clip 下界
        clip_high = float(clip_range[1])  # 长度 clip 上界
        low = max(low, clip_low)
        high = min(high, clip_high)
    return (low, high)


def _shared_cross_section_ranges(
    value_range: Vector2 | Vector6,
    clip_range: Vector2 | Vector6 | None,
    *,
    target: HandCfg,
) -> tuple[Vector2, Vector2] | None:
    r"""返回当前 mutate 批次共享的宽度/高度采样区间。

    这里故意接收 `target`，是为了把“只有存在可变 joint 时才声明 shared sample”
    的语义放在 sampling 层，而不是在 config 层偷做假设。
    """

    if len(value_range) == 2:
        return None  # Vector2 只控制长度，不声明 shared width/height 变量
    if not any(True for _ in _iter_target_joints(target)):
        return None  # 当前 hand 没有可变 joint 时，无需声明 shared 横截面随机变量

    width_range = (float(value_range[2]), float(value_range[3]))  # 全手共享宽度扰动区间
    height_range = (float(value_range[4]), float(value_range[5]))  # 全手共享高度扰动区间
    if clip_range is not None and len(clip_range) == 6:
        width_range = (
            max(width_range[0], float(clip_range[2])),
            min(width_range[1], float(clip_range[3])),
        )  # 宽度 clip 只裁到共享随机变量自己的合法域
        height_range = (
            max(height_range[0], float(clip_range[4])),
            min(height_range[1], float(clip_range[5])),
        )  # 高度 clip 同理
    return width_range, height_range


def _joint_primary_length(joint) -> float | None:
    r"""从 joint 的 collision / visual 主体几何中读取有效长度 $L_i$。"""

    # 首先看 collision，因为它更接近物理接触和碰撞语义；
    # 如果没有 collision，再退回 visual，保证在更稀疏的资产上也能工作。
    geometry = None
    if joint.collisions:
        geometry = joint.collisions[0].geometry
    elif joint.visuals:
        geometry = joint.visuals[0].geometry
    if geometry is None:
        return None
    if geometry.kind == "box":
        return float(geometry.size[1])
    if geometry.kind == "cylinder":
        return float(geometry.length)
    if geometry.kind == "elliptic_cylinder":
        return float(geometry.length)
    return None


def _joint_cross_section(joint) -> tuple[float, float] | None:
    r"""返回 joint 主体几何在局部 $(x, z)$ 平面上的横截面全尺寸。

    这里统一返回“全尺寸”而不是半轴，是为了和 `Vector6=(l,w,h)` 的公开语义对齐。
    对圆柱/椭圆柱而言，内部几何仍然可以用半径/半轴存储，但 mutate 层只关心
    当前横截面的可见宽度与高度。
    """

    geometry = None
    if joint.collisions:
        geometry = joint.collisions[0].geometry
    elif joint.visuals:
        geometry = joint.visuals[0].geometry
    if geometry is None:
        return None
    if geometry.kind == "box":
        return float(geometry.size[0]), float(geometry.size[2])
    if geometry.kind == "cylinder":
        diameter = 2.0 * float(geometry.radius)
        return diameter, diameter
    if geometry.kind == "elliptic_cylinder":
        return 2.0 * float(geometry.radius_x), 2.0 * float(geometry.radius_z)
    return None


def _is_thumb_joint(joint) -> bool:
    r"""判断当前 joint 是否属于 thumb 链。

    当前最稳定的语义锚点仍是 child link 名。regular thumb 的 child link 会被
    lower 成 `thumb_cmc1/cmc2/mcp/dip` 这类前缀，因此这里直接按 `thumb_`
    前缀识别，而不去猜测 joint index 或 mount 方向。
    """

    child_name = str(getattr(joint, "child", ""))  # child link 名在 pre-made / mutate / exporter 间都保持稳定
    return child_name.startswith("thumb_")  # thumb 全链统一使用 `thumb_` 语义前缀


def _semantic_width_scale_for_joint(
    joint,
    *,
    semantic_width_scale: float | None,
    semantic_height_scale: float | None,
) -> float | None:
    r"""把 semantic 宽度缩放解释成当前 joint local $x$ 方向应采用的比例。

    本轮 contract 不改建模坐标约定：

    - 所有 finger builder 仍按 `x=宽, y=长, z=高` 建模；
    - 这里只在 post-mutate 的 morphology 语义层重新解释 shared sample。

    因而：

    - non-thumb：local $x$ 对应 semantic 宽度；
    - thumb：local $x$ 对应 semantic 高度。
    """

    if not _is_thumb_joint(joint):
        return semantic_width_scale  # non-thumb: local $x \leftarrow$ semantic width
    return semantic_height_scale  # thumb: local $x \leftarrow$ semantic height


def _semantic_height_scale_for_joint(
    joint,
    *,
    semantic_width_scale: float | None,
    semantic_height_scale: float | None,
) -> float | None:
    r"""把 semantic 高度缩放解释成当前 joint local $z$ 方向应采用的比例。

    与 `_semantic_width_scale_for_joint(...)` 配套：

    - non-thumb：local $z$ 对应 semantic 高度；
    - thumb：local $z$ 对应 semantic 宽度。
    """

    if not _is_thumb_joint(joint):
        return semantic_height_scale  # non-thumb: local $z \leftarrow$ semantic height
    return semantic_width_scale  # thumb: local $z \leftarrow$ semantic width


def _mutated_length(old_length: float, delta_or_ratio: float, cfg: LinkScaleCfg) -> float:
    r"""根据 `scale_type` 把采样值解释为相对比例或绝对长度扰动。

    数学上可写成：
    $$
    L_i' =
    \begin{cases}
        L_i(1+\delta), & \text{rel}\\
        L_i+\delta, & \text{abs}
    \end{cases}
    $$
    这里的 $\delta$ 就是采样到的局部扰动量。
    """

    # 这里不是“做了一个普通乘法”这么简单，而是把采样变量解释成
    # 不同语义下的局部扰动：比例或绝对增量。
    if cfg.scale_type == "rel":
        return old_length * float(delta_or_ratio)
    return old_length + float(delta_or_ratio)


def _mutated_cross_section(
    old_cross_section: tuple[float, float] | None,
    *,
    width_scale: float | None,
    height_scale: float | None,
) -> tuple[float, float] | None:
    r"""根据当前 joint 已完成轴语义映射后的 local 宽度/高度扰动生成新横截面尺寸。

    传进来的 `width_scale` / `height_scale` 已经不再是“原始 semantic 宽高”，
    而是针对当前 joint local $(x,z)$ 的最终缩放比例：

    - non-thumb：`(width_scale, height_scale) = (semantic width, semantic height)`
    - thumb：`(width_scale, height_scale) = (semantic height, semantic width)`
    """

    if old_cross_section is None:
        return None
    width, height = old_cross_section
    new_width = width if width_scale is None else width * float(width_scale)  # 宽度使用全手共享随机变量
    new_height = height if height_scale is None else height * float(height_scale)  # 高度使用全手共享随机变量
    if new_width <= 1e-6 or new_height <= 1e-6:
        return old_cross_section  # 极端非法尺度直接回退原尺寸，避免几何退化成负值或零
    return new_width, new_height


def _set_joint_primary_geometry(
    joint,
    *,
    old_length: float,
    new_length: float,
    old_cross_section: tuple[float, float] | None,
    new_cross_section: tuple[float, float] | None,
    keep_center: bool,
) -> None:
    r"""写回 joint child link 的主体几何尺寸，并保持 mesh offset $d_i$ 不被缩放。

    若 `keep_center=False`，则会按旧中心偏移量重新计算 origin；
    若 `keep_center=True`，则保留几何中心，仅调整长度。这一分支主要
    服务于 CMC1 这类需要特殊中心保持语义的关节。

    宽度/高度缩放遵守 `长度和宽度变异示意.jpg` 的语义：只改横截面尺寸，
    不改 `element.origin` 与 `joint.origin`。这意味着 non-thumb 的 box/cylinder
    截面只是在局部 $(x,z)$ 平面上做对称放缩，而不会把整段 link 横向平移走。
    """

    # collision 和 visual 要同步更新，否则视觉和接触皮肤会在局部产生
    # 不一致的长度语义。
    for collection_name in ("collisions", "visuals"):
        collection = getattr(joint, collection_name)
        for index, element in enumerate(collection):
            geometry = element.geometry
            if geometry.kind == "box":
                size = geometry.size
                width = float(size[0]) if new_cross_section is None else float(new_cross_section[0])  # box 局部 $x$ 全尺寸
                height = float(size[2]) if new_cross_section is None else float(new_cross_section[1])  # box 局部 $z$ 全尺寸
                geometry = geometry.replace(size=(width, new_length, height))
            elif geometry.kind == "cylinder":
                if new_cross_section is None:
                    geometry = geometry.replace(length=new_length)
                else:
                    radius_x = 0.5 * float(new_cross_section[0])  # 从公开的全尺寸宽度恢复成局部 $x$ 半轴
                    radius_z = 0.5 * float(new_cross_section[1])  # 从公开的全尺寸高度恢复成局部 $z$ 半轴
                    if math.isclose(radius_x, radius_z, rel_tol=0.0, abs_tol=1e-12):
                        geometry = geometry.replace(radius=radius_x, length=new_length)  # 等径时保持标准圆柱
                    else:
                        geometry = EllipticCylinderGeometryCfg(radius_x=radius_x, radius_z=radius_z, length=new_length)
            elif geometry.kind == "elliptic_cylinder":
                if new_cross_section is None:
                    geometry = geometry.replace(length=new_length)
                else:
                    radius_x = 0.5 * float(new_cross_section[0])  # 椭圆柱局部 $x$ 半轴
                    radius_z = 0.5 * float(new_cross_section[1])  # 椭圆柱局部 $z$ 半轴
                    if math.isclose(radius_x, radius_z, rel_tol=0.0, abs_tol=1e-12):
                        from ...asset_schema_core import CylinderGeometryCfg

                        geometry = CylinderGeometryCfg(radius=radius_x, length=new_length)  # 再次收敛到等径时允许规范化回标准圆柱
                    else:
                        geometry = geometry.replace(radius_x=radius_x, radius_z=radius_z, length=new_length)
            else:
                continue

            origin = element.origin
            if keep_center:
                new_origin = origin.copy()
            else:
                offset_y = origin.pos[1] - old_length / 2.0
                new_origin = PoseCfg(
                    pos=(origin.pos[0], new_length / 2.0 + offset_y, origin.pos[2]),
                    rpy=origin.rpy,
                )
            collection[index] = element.replace(geometry=geometry, origin=new_origin)
            # 若当前 joint 已携带 inertial，则只把惯性参考原点同步到新的主体几何中心。
            # 真正的质量 / 惯量重建已统一上收至 `asset_physics.py` 的 physics closure，
            # 避免同一套几何语义在 mutator 内部和 generator 主链里重复实现。
            if joint.inertial is not None and index == 0:
                joint.inertial = joint.inertial.replace(origin=new_origin)


def _next_origin_from_link_scale(
    *,
    current_joint,
    next_joint,
    old_length: float,
    new_length: float,
    old_cross_section: tuple[float, float] | None,
    new_cross_section: tuple[float, float] | None,
    next_old_cross_section: tuple[float, float] | None,
    next_new_cross_section: tuple[float, float] | None,
) -> PoseCfg:
    r"""根据当前 link 的新几何边界重解算下游 joint origin。

    普通 non-thumb 以及 thumb 非 CMC1 段仍然遵守：
    $$
    y_{next}' = y_{next} + (L_i' - L_i),
    $$
    因为它们的 joint 串联逻辑本质上仍是“上一段有效长度沿局部 $+y$ 推进”。

    对 CMC1，这个规则会漏掉宽度/高度对 CMC2 挂接边界的影响，因此必须改用
    thumb builder 里同源的专用公式。
    """

    if not str(current_joint.child).endswith("_cmc1"):
        delta_y = new_length - old_length  # 普通串联段只沿局部 $+y$ 方向推进长度差
        return PoseCfg(
            pos=(next_joint.origin.pos[0], next_joint.origin.pos[1] + delta_y, next_joint.origin.pos[2]),
            rpy=next_joint.origin.rpy,
        )

    old_width = old_cross_section[0] if old_cross_section is not None else 0.0  # 旧 CMC1 在局部 $x$ 的全宽
    old_height = old_cross_section[1] if old_cross_section is not None else 0.0  # 旧 CMC1 在局部 $z$ 的全高
    new_width = new_cross_section[0] if new_cross_section is not None else old_width  # 新 CMC1 在局部 $x$ 的全宽
    new_height = new_cross_section[1] if new_cross_section is not None else old_height  # 新 CMC1 在局部 $z$ 的全高
    next_width = (
        next_new_cross_section[0] if next_new_cross_section is not None else (
            next_old_cross_section[0] if next_old_cross_section is not None else old_width
        )
    )  # CMC2 的边界对齐必须读取“同一次 mutate 后、按 thumb 语义 swap 过的 local $x$ 截面”
    next_height = (
        next_new_cross_section[1] if next_new_cross_section is not None else (
            next_old_cross_section[1] if next_old_cross_section is not None else old_height
        )
    )  # CMC2 的局部 $z$ 边界同理，不能再退回 mutate 前的横截面
    current_origin = current_joint.collisions[0].origin if current_joint.collisions else current_joint.visuals[0].origin
    return PoseCfg(
        pos=(
            (new_width - next_width) / 2.0,  # CMC1 宽度变化后，CMC2 需要重新贴到新的侧边界
            current_origin.pos[1] + new_length / 2.0,  # CMC1 长度变化后，CMC2 仍从新的远端边界长出
            current_origin.pos[2] - (new_height - next_height) / 2.0,  # CMC1 高度变化后，CMC2 在局部 $z$ 上重新对齐
        ),
        rpy=next_joint.origin.rpy,
    )


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
