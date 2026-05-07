r"""连杆长度缩放变异算子：在已有 `HandCfg` 上对 link 两岸距离做扰动。

科研语义上，这个算子对应的是“同一条运动学链上，某一段主体几何的轴向
长度发生轻微变化”，而不是重新生成一个新的 hand family。它必须结合
`AnyMani/source/anymani/anymani/assets/doc/长度变异示意.jpg` 理解：

- `link_scale` 只改变自身有效长度 $L_i$；
- 不缩放 mesh offset $d_i$，也不改视觉锚点的局部语义；
- 下游 joint / tip origin 使用新的 $L_i+d_i$ 重新放置；
- 不改变 topology、DOF、parent-child 串联关系或 joint axis。

因此该算子更接近“几何长度的局部再标定”，而不是“随便把 link 拉伸一遍”。
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector6
from ._base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


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

    link_scale: Vector2 | Vector6 | dict[str, Vector2 | Vector6] = field(default=MISSING)
    r"""连杆尺寸变异范围配置。

    - ``Vector2``：$l=(l_{\min},l_{\max})$，只控制主长度方向；
    - ``Vector6``：$(l_{\min},l_{\max},w_{\min},w_{\max},h_{\min},h_{\max})$；
    - ``dict``：按 child link 语义名或 joint child suffix 细粒度配置。

    数值锚点：
    - quick post-mutate 默认采用 `0.97~1.03` 的相对缩放；
    - 这对应约 $\pm3\%$ 的轻扰动量级，目的是在不破坏家族语义的前提下
      生成足够的长度多样性，而不是做大尺度 morphology search。
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

    clip: Vector2 | Vector6 | dict[str, Vector2 | Vector6] | None = None
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
        """

        # 先把所有可扰动 joint 的局部采样语义一次性声明出来。
        # 这样 pipeline 看到的是“联合参数表”，而不是互相串联的逐步修改。
        specs: dict[str, Any] = {}
        for _, _, joint in _iter_target_joints(target):
            value_range = _range_for_joint(self.cfg.link_scale, joint.child)
            specs[joint.name] = _make_range_sampler(
                _primary_range(value_range),
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        return specs

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""生成 link length 与下游 joint origin 的 deferred patch。

        这一步只生成 patch，不直接修改原对象；这样 joint 长度的修改和
        下游 origin 的推进可以作为一组原子语义来处理，而不是两次独立写入。
        """

        sampled_params = sampled_params or {}
        patch = HandPatch()

        # 先遍历每个目标 joint，再为当前 joint 自己和下游 joint 生成
        # 一组最小 patch。这里的核心原则是：同一个源长度变动，只产生
        # 一次语义上闭合的几何更新，而不是让后续算子再去猜测这个变化。
        for finger_index, joint_index, joint in _iter_target_joints(target):
            delta_or_ratio = float(sampled_params.get(joint.name, 0.0))
            old_length = _joint_primary_length(joint)
            if old_length is None:
                continue
            new_length = _mutated_length(old_length, delta_or_ratio, self.cfg)
            if new_length <= 1e-6:
                continue
            length_delta = new_length - old_length
            is_cmc1 = str(joint.child).endswith("_cmc1")

            def apply_link(hand: HandCfg, *, fi=finger_index, ji=joint_index, old=old_length, new=new_length, cmc1=is_cmc1) -> None:
                r"""写回当前 joint child link 的新有效长度。"""

                mutated_joint = hand.fingers[fi].joints[ji]
                _set_joint_primary_length(mutated_joint, old_length=old, new_length=new, keep_center=cmc1)

            patch.add(("finger", finger_index, "joint", joint_index, "link_length"), apply_link)

            next_index = joint_index + 1
            if next_index < len(target.fingers[finger_index].joints):
                advance_delta = length_delta * 0.5 if is_cmc1 else length_delta

                def apply_next_origin(hand: HandCfg, *, fi=finger_index, ni=next_index, dy=advance_delta) -> None:
                    r"""按新的 $L_i+d_i$ 推进下游 joint origin。"""

                    next_joint = hand.fingers[fi].joints[ni]
                    pos = next_joint.origin.pos
                    next_joint.origin = PoseCfg(pos=(pos[0], pos[1] + dy, pos[2]), rpy=next_joint.origin.rpy)

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


def _range_for_joint(config: Vector2 | Vector6 | dict[str, Vector2 | Vector6], child_name: str) -> Vector2 | Vector6:
    r"""按 child link 语义名解析某个 joint 的配置范围。

    之所以优先按 child 名称而不是 joint 数字编号匹配，
    是因为 pre-made / post-mutate 中 joint 编号可能因为 delete / regroup
    改写，但 child link 语义名通常更接近实际机械结构。
    """

    # 允许全局单一范围，也允许按 child link 名字做精细分配。
    if not isinstance(config, dict):
        return config
    child = str(child_name)
    suffix = child.rsplit("_", 1)[-1]
    if child in config:
        return config[child]
    if suffix in config:
        return config[suffix]
    raise KeyError(f"no link_scale range configured for child link {child!r}")


def _primary_range(value_range: Vector2 | Vector6) -> Vector2:
    r"""取主长度方向的采样范围。"""

    return (float(value_range[0]), float(value_range[1]))


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
    return None


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


def _set_joint_primary_length(joint, *, old_length: float, new_length: float, keep_center: bool) -> None:
    r"""写回 joint child link 的主体几何长度，并保持 mesh offset $d_i$ 不被缩放。

    若 `keep_center=False`，则会按旧中心偏移量重新计算 origin；
    若 `keep_center=True`，则保留几何中心，仅调整长度。这一分支主要
    服务于 CMC1 这类需要特殊中心保持语义的关节。
    """

    # collision 和 visual 要同步更新，否则视觉和接触皮肤会在局部产生
    # 不一致的长度语义。
    for collection_name in ("collisions", "visuals"):
        collection = getattr(joint, collection_name)
        for index, element in enumerate(collection):
            geometry = element.geometry
            if geometry.kind == "box":
                size = geometry.size
                geometry = geometry.replace(size=(size[0], new_length, size[2]))
            elif geometry.kind == "cylinder":
                geometry = geometry.replace(length=new_length)
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
            # 如果 joint 带 inertial，就把惯性参考原点也一起跟着移动，
            # 这样长度变化不会把质心锚点留在旧的位置。
            if joint.inertial is not None and index == 0:
                joint.inertial = joint.inertial.replace(origin=new_origin)


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
