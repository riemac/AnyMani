r"""指尖替换变异算子：在 post-mutate 阶段重采末端 tip embodiment。

该算子位于 post-mutate 阶段，替换每根 finger 末端 `tip_joint`
所连接的整个 tip child link embodiment：collision、visual
以及必要的 metadata / 材质语义应作为同一个 tip spec 一起更新。
它不改变 finger 拓扑、关节数量、关节轴、挂载点或 tip joint 本身在
运动链上的位置。

从科研上看，这个算子处理的是末端接触材料与局部刚体近似，而不是整根
finger 的运动学重建。因此 collision、visual、metadata 应当被视为同一个
tip spec 的不同投影，不能只改其中一项就当作完成了 tip replacement。

# NOTE:
custom mesh tip 的最终 `mass / inertial` 属于 generator 主链里的 physics closure。
本算子只改 tip spec 的几何与接触语义，不在局部 mutator 内写最终动力学属性。

本轮 contract 明确区分两层概率：

1. `self_mode` 是样本级 accepted/output 分布，和 `mount_perturb` /
   `limit_tweak` 一样由 generator quota 保障；
2. `tip_range` 是 tip_type proposal 分布。若 validator 因 tip 几何偏置拒绝
   某些 proposal，最终 accepted tip_type 分布可能轻微漂移，因此必须在 summary
   中显式记录 proposed / accepted 计数，而不是把它伪装成后验 quota。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg, JointCfg
from ...asset_schema_core import PoseCfg, Vector2
from ...builder.joint_builders_custom import CustomTipBuilderCfg, apply_thumb_functional_tip_phase
from ...builder.joint_builders_primitive import PrimJointBuilderCfg
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler

_MODE_IDENTITY = "identity"
_MODE_SAME = "same"
_MODE_GENERAL = "general"

_ALL_SELF_MODES = (
    _MODE_IDENTITY,
    _MODE_SAME,
    _MODE_GENERAL,
)
r"""`tip_replace.self_mode` 当前支持的全部高层 mode。"""

_CUSTOM_TIP_TYPES = (
    "leap_cube",
    "round",
    "wedge",
    "thinner",
)
r"""由 `CustomTipBuilderCfg` preset 表支持的 custom mesh tip 类型。"""

_PRIMITIVE_TIP_TYPES = ("cs",)
r"""本轮 tip replacement 支持的 primitive tip 类型；`bs` 暂不纳入。"""

_DEFAULT_TIP_TYPES = _PRIMITIVE_TIP_TYPES + _CUSTOM_TIP_TYPES
r"""`tip_range=None` 时使用的默认 proposal 候选集合。"""

_MODE_TOLERANCE = 1e-9
r"""mode / tip_type 概率求和与正概率判定的数值容差。"""

_MIN_POSITIVE = 1e-6
r"""几何长度、半径和 scale 的最小正值，防止导出退化刚体。"""


@dataclass
class TipReplaceCfg(MutatorBaseCfg):
    r"""指尖替换工具配置。

    科研语义上，`tip_replace` 控制的是末端接触皮肤与其刚体物理属性，
    而不是整根手指的 kinematic embodiment。它和 `link_scale` /
    `mount_perturb` 的边界应保持清晰：

    - `link_scale`：改变运动链中 link 的有效长度 / 尺度；
    - `mount_perturb`：改变 finger root 相对 palm 的刚体位姿；
    - `tip_replace`：替换末端 tip child link 的几何与接触皮肤描述，包括
      collision、visual 与相关 metadata；最终 inertial 由 physics closure 闭包。

    对手内操作来说，指尖几何、质量分布和接触皮肤近似是接触动力学中
    最敏感的局部变量之一。因此这里的 `self_mode` 不应被理解为普通工程
    开关，而是一次资产采样中全手层面的 morphology / physics coherence 假设。
    """

    class_type: type[TipReplaceMutator] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类。"""

    target_fingers: tuple[str, ...] | None = None
    r"""目标 finger 名称；`None` 表示当前 hand 中所有 finger 都参与 tip replacement。"""

    self_mode: Literal["identity", "same", "general"] | dict[str, float] | None = _MODE_SAME
    r"""指尖替换的高层形态模式配置。

    该字段描述一次 post-mutate 中“全手的指尖皮肤是否共享同一种宏观假设”。
    它不直接指定某个具体 mesh、半径或缩放值，而是决定 `tip_range` 与 `scale`
    在 finger 之间如何耦合采样。

    支持三种输入语义：

    - `None`：未显式指定时默认落到 `"same"`；
    - `str`：固定使用某一个 mode；
    - `dict[str, float]`：按概率混合采样 mode，且该概率由 generator 解释为
      accepted/output quota，而不是 proposal prior。

    预设 mode：

    - `"identity"`：显式 no-op，不改任何 tip，只记录 provenance；
    - `"same"`：全体目标 finger 共享同一个完整 tip spec；
    - `"general"`：每根目标 finger 独立采样完整 tip spec。
    """

    tip_range: list[str] | dict[str, float] | None = None
    r"""指尖候选类型 proposal 分布。

    该字段描述离散 `tip_type` 的候选集合，而不是 hand family，也不是 mesh 文件路径集合。
    合法名称包括：

    - primitive tip recipe：`"cs"`；
    - `CustomTipBuilderCfg` preset：`"leap_cube"`、`"round"`、`"wedge"`、`"thinner"`。

    输入语义：

    - `None`：默认使用 `("cs", "leap_cube", "round", "wedge", "thinner")`；
    - `list[str]`：离散列出合法 tip_type，默认每个 tip_type 等概率 proposal；
    - `dict[str, float]`：显式 proposal 概率，概率应非负且和为 1。

    # NOTE:
    这里刻意叫 `tip_type`，不叫 `tip_family`。`hand family` 保留给 `leap/allegro`
    这类 pre-made 家族；指尖候选只是同一只 hand 上的末端接触几何变体。

    # NOTE:
    `tip_range` 是 proposal 分布。若某类 tip_type 更容易被 validator 拒绝，最终
    accepted 分布可能偏离 proposal，因此运行时会记录 proposed / accepted 计数。
    """

    scale: Vector2 | dict[str, Vector2] = (1.0, 1.0)
    r"""指尖尺寸缩放范围。

    该字段描述对 tip spec 的无量纲比例缩放，例如 `(0.9, 1.1)` 表示采样一个
    $s\in[0.9, 1.1]$ 的 scale。缩放示意图见
    `AnyMani/source/anymani/anymani/assets/doc/指尖scale示意.png`。

    输入语义：

    - `Vector2`：所有 tip_type 共享同一个缩放范围；
    - `dict[str, Vector2]`：不同 tip_type 使用各自的缩放范围；没有显式列出的
      tip_type 回退到 `shared`，再回退到 `(1.0, 1.0)`。

    几何语义：

    - scale 不移动 `tip_joint`，也不改变上一段 link 的末端位置；
    - 对 custom mesh tip，缩放围绕 preset 中的 `anchor_point` 重新 lowering；
    - 对 `cs`，scale 同时缩放半径 $r$ 和高度 $h$，再由 `cs_ratio` 可选改写
      $h/r$。
    """

    cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2] | None = None
    r"""`cs` 类型指尖的高度与半径比 $\lambda=h/r$ 变异范围。

    `cs` 的建模约定是 cylinder + sphere：

    - 圆柱半径为 $r$；
    - 圆柱高度为 $h=\lambda r$；
    - 球帽半径仍为 $r$；
    - 球心落在圆柱顶面中心，使球最大截面和圆柱顶面重合，形成平滑过渡。

    输入语义：

    - `None`：保持当前 tip 或默认 `cs` 的原始 $\lambda$；
    - `Vector2`：等价于 `{"abs": Vector2}`，直接采绝对比例；
    - `{"abs": (a,b)}`：从绝对比例区间采样 $\lambda$；
    - `{"add": (a,b)}`：在当前 tip 的 $\lambda_0$ 基础上叠加增量；若当前 tip
      不是 `cs`，则 $\lambda_0=1$。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""连续变量采样分布，当前作用于 `scale` 与 `cs_ratio`。

    - `uniform`：在配置区间中均匀采样；
    - `normal`：先采中心正态，再交给 `boundary_policy` 处理越界。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""连续变量边界处理策略。

    该字段只规定采样结果超出 `scale` / `cs_ratio` 区间时如何处理，不改变
    `self_mode` 和 `tip_range` 的概率语义。
    """

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    r"""当前 cfg 真正会被采样到的 mode 集合；dict 输入时只保留正概率项。"""

    def __post_init__(self) -> None:
        r"""校验 mode / tip_type 契约，并补齐运行时类。"""

        self.class_type = TipReplaceMutator
        if isinstance(self.target_fingers, list):
            self.target_fingers = tuple(str(name) for name in self.target_fingers)
        self._active_modes = _resolve_active_modes(self.self_mode)
        _resolve_tip_distribution(self.tip_range)  # fail-fast 校验 tip_type 概率和合法性
        _validate_scale_ranges(self.scale)
        _validate_cs_ratio(self.cs_ratio)


class TipReplaceMutator(MutatorBase):
    r"""把高层 tip replacement contract lowering 成 tip child link patch。

    该 runtime 使用结构化 sample payload，而不是把每根 finger 的 tip_type / scale
    平铺成一堆局部 sampler。原因和 `mount_perturb` / `limit_tweak` 相同：
    mode 一旦进入 accepted quota，generator 必须能强制某个 mode 并重新生成
    该 mode 所需的完整低层随机量，避免“只改 mode 名但缺少 tip specs”的伪样本。
    """

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        r"""绑定一份 `TipReplaceCfg`。"""

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""返回一个结构化样本生成器。

        Returns:
            dict[str, Any]: 单键 `"sample"`，其 value 是生成完整 tip replacement
            payload 的 callable。
        """

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于结构化 sample payload 生成 tip child-link 的延迟 patch。

        该函数严格不修改 `finger.joints[:-1]`、`finger.mount` 以及 tip joint 的
        运动学壳体：`origin`、`parent`、`child`、`axis` 和 `limit`。它只替换
        末端 child link 的 collision / visual / metadata，以及必要时的过渡性
        inertial 占位。

        # NOTE:
        patch 规划仍然基于同一份原始 `HandCfg`，但真正 apply 时可能已有
        `link_scale` 等前序 patch 把 `tip_joint.origin` 推到新的远端边界。这里
        必须把 apply-time 的当前 tip joint 视为 kinematic shell，而不是把
        `_build_replacement_tip_joint(...)` 生成的原始 origin 整体写回。
        """

        sample = _normalize_sample_payload(sampled_params, self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["tip_replace"] = sample
        patch.metadata["post_mutate_tip_replace"] = sample

        if resolved_mode == _MODE_IDENTITY:
            return patch  # `identity` 是显式 no-op，只保留 provenance

        finger_specs = dict(sample.get("finger_specs", {}))
        for finger_index, finger in _iter_target_fingers(target, self.cfg.target_fingers):
            spec = dict(finger_specs.get(finger.name, {}))
            if not spec:
                continue
            replacement = _build_replacement_tip_joint(finger.tip_joint, spec)
            patch.add(
                ("finger", finger_index, "tip"),
                _tip_joint_replacer(finger_index=finger_index, replacement=replacement),
            )
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:
        r"""为当前 hand 样本生成一份已经解析好 mode 的结构化随机量。"""

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        r"""为 accepted-quota 路径生成指定 mode 的结构化随机量。

        Args:
            target (HandCfg): 当前原始 hand schema。
            resolved_mode (str): generator 强制或本算子自行采样得到的 mode。

        Returns:
            dict[str, Any]: 包含 `resolved_self_mode` 与 per-finger tip spec 的结构化
            payload。
        """

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace resolved mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY, "finger_specs": {}}

        target_fingers = list(_iter_target_fingers(target, self.cfg.target_fingers))
        if resolved_mode == _MODE_SAME:
            shared_spec = _sample_tip_spec(self.cfg, target_fingers[0][1].tip_joint if target_fingers else None)
            return {
                "resolved_self_mode": _MODE_SAME,
                "finger_specs": {finger.name: dict(shared_spec) for _, finger in target_fingers},
            }

        return {
            "resolved_self_mode": _MODE_GENERAL,
            "finger_specs": {
                finger.name: _sample_tip_spec(self.cfg, finger.tip_joint)
                for _, finger in target_fingers
            },
        }


def _iter_target_fingers(hand: HandCfg, target_fingers: tuple[str, ...] | None):
    r"""按配置解析目标 finger 集合。"""

    target_set = set(target_fingers or ())
    for finger_index, finger in enumerate(hand.fingers):
        if target_set and finger.name not in target_set:
            continue
        yield finger_index, finger


def _tip_joint_replacer(*, finger_index: int, replacement: JointCfg):
    r"""构造一个只替换 finger 末端 contact embodiment 的 patch callable。

    `replacement` 来自原始 tip joint 与新 tip spec 的 builder lowering，它携带了
    新的 collision / visual / inertial / metadata；但 apply 阶段的当前 tip joint
    可能已经被 `link_scale` 更新了运动链边界。科研语义上：

    $$
    \text{tip\_replace}:\quad \mathcal{G}_{tip}\mapsto\mathcal{G}_{tip}',
    \qquad
    \text{link\_scale}:\quad y_{tip}\mapsto y_{tip}'.
    $$

    两个算子作用在不同变量上，因此这里保留当前 `JointCfg` 的 kinematic fields，
    只写入 replacement 的末端接触几何与 provenance。
    """

    def _apply(hand: HandCfg) -> None:
        joints = list(hand.fingers[finger_index].joints)
        current_tip = joints[-1]  # apply-time 运动学壳体，可能已经包含 `link_scale` 更新后的 $y_{tip}'$
        joints[-1] = current_tip.replace(
            inertial=replacement.inertial.copy() if replacement.inertial is not None else None,  # tip child link 的惯性占位
            collisions=[collision.copy() for collision in replacement.collisions],  # 新 tip 接触几何，定义局部接触边界
            visuals=[visual.copy() for visual in replacement.visuals],  # 新 tip 可视几何，与 collision 同属 tip spec 投影
            is_tip=replacement.is_tip,  # 继续显式标记 fixed tip child link
            metadata=dict(replacement.metadata),  # 记录 tip_type / scale / provenance，供 summary 与 sidecar 消费
        )
        hand.fingers[finger_index] = hand.fingers[finger_index].replace(joints=joints)

    return _apply


def _build_replacement_tip_joint(original: JointCfg, spec: dict[str, Any]) -> JointCfg:
    r"""把一份 tip spec lowering 成完整 `JointCfg`。

    这里复用 builder 层的 primitive / custom tip 公式，避免在 mutate 层复制
    anchor 对齐、圆柱球帽等几何细节。mutate 层只负责决定“采样到了
    哪个 tip spec”，具体 child link embodiment 仍交给 builder。

    对 custom mesh 路线，builder 现在只负责几何 lowering；真正的最终 inertial
    会在 generator 主链中由 physics closure 根据最终 collision 几何统一补齐。
    """

    tip_type = str(spec["tip_type"])
    common_kwargs = {
        "name": original.name,
        "parent": original.parent,
        "child": original.child,
        "joint_type": original.joint_type,
        "origin": original.origin,
        "axis": (0.0, 0.0, 0.0) if original.joint_type == "fixed" else original.axis,
        "limit": original.limit,
        "is_tip": True,
        "metadata": _replacement_metadata(original, spec),
    }

    if tip_type == "cs":
        radius = max(_MIN_POSITIVE, float(spec["radius"]))
        height = max(_MIN_POSITIVE, float(spec["height"]))
        builder_cfg = PrimJointBuilderCfg(
            mesh={"type": "cs", "radius": radius, "height": height, "offset": _tip_offset_from_original(original)},
            **common_kwargs,
        )
    else:
        mesh_offset = _tip_offset_from_original(original)
        if _is_thumb_tip_joint(original):
            mesh_offset = apply_thumb_functional_tip_phase(mesh_offset)
        builder_cfg = CustomTipBuilderCfg(
            tip_type=tip_type,
            mesh_offset=mesh_offset,
            scale=float(spec.get("scale", 1.0)),
            **common_kwargs,
        )
    builder = builder_cfg.class_type(builder_cfg)
    return builder.build()


def _replacement_metadata(original: JointCfg, spec: dict[str, Any]) -> dict[str, Any]:
    r"""生成替换后 tip joint 的 metadata。"""

    metadata = {
        **dict(original.metadata),
        "post_mutate_tip_mode": "tip_replace",
        "tip_type": str(spec["tip_type"]),
        "post_mutate_tip_type": str(spec["tip_type"]),
        "post_mutate_tip_scale": float(spec.get("scale", 1.0)),
        "post_mutate_tip_spec": dict(spec),
    }
    if str(spec["tip_type"]) != "cs" and _is_thumb_tip_joint(original):
        metadata["thumb_functional_tip_phase_rpy"] = apply_thumb_functional_tip_phase(PoseCfg()).rpy
    return metadata


def _is_thumb_tip_joint(joint: JointCfg) -> bool:
    r"""判断当前末端 joint 是否属于 thumb。

    这里优先读 builder 写入的 `finger_name`，再回退到 child/parent/joint 名。
    post-mutate 可能来自历史 YAML 或不同 builder 版本，因此不能只押一个字段。
    """

    finger_name = str(joint.metadata.get("finger_name", "")).lower()
    if finger_name == "thumb":
        return True
    name_fields = (joint.name, joint.parent, joint.child)
    return any(str(value).lower().startswith("thumb_") or str(value).lower() == "thumb" for value in name_fields)


def _tip_offset_from_original(original: JointCfg) -> PoseCfg:
    r"""从原 tip joint 读取局部 tip geometry 的锚点 offset。

    对 primitive `cs`，builder 写入的是几何中心：

    $$
    y_{\mathrm{cyl}} = d_y + h/2,\qquad
    y_{\mathrm{sph}} = d_y + h.
    $$

    因此这里不能把 `collision.origin` 直接当成 offset，否则会在重新 lowering 时
    把 $h/2$ 或 $h$ 重复加一次。对于 custom mesh，当前 metadata 尚未显式记录
    原始 `mesh_offset`，首版保守回退到零位姿，让 preset anchor 贴回 tip joint。
    """

    if original.collisions:
        first = original.collisions[0]
        if first.geometry.kind == "cylinder":
            length = float(first.geometry.length)
            cap_rpy = original.collisions[1].origin.rpy if len(original.collisions) > 1 else (0.0, 0.0, 0.0)
            return PoseCfg(
                pos=(first.origin.pos[0], first.origin.pos[1] - length / 2.0, first.origin.pos[2]),
                rpy=cap_rpy,
            )
        if first.geometry.kind == "box":
            size_y = float(first.geometry.size[1])
            cap_rpy = original.collisions[1].origin.rpy if len(original.collisions) > 1 else first.origin.rpy
            return PoseCfg(
                pos=(first.origin.pos[0], first.origin.pos[1] - size_y / 2.0, first.origin.pos[2]),
                rpy=cap_rpy,
            )
    return PoseCfg()


def _sample_tip_spec(cfg: TipReplaceCfg, current_tip: JointCfg | None) -> dict[str, Any]:
    r"""采样一个完整 tip spec。

    离散变量 `tip_type` 来自 `tip_range` proposal 分布；连续变量来自 `scale` 与
    可选 `cs_ratio`。该函数只输出最小语义参数，collision / visual / metadata
    会在 `_build_replacement_tip_joint()` 中由 builder lowering。
    """

    tip_type = _draw_tip_type(cfg.tip_range)
    scale = _sample_scale(cfg, tip_type=tip_type)
    spec: dict[str, Any] = {"tip_type": tip_type, "scale": scale}
    if tip_type == "cs":
        radius, base_ratio = _current_cs_radius_and_ratio(current_tip)
        scaled_radius = max(_MIN_POSITIVE, radius * scale)
        ratio = _sample_cs_ratio(cfg, current_ratio=base_ratio)
        spec.update(
            {
                "radius": scaled_radius,
                "height": max(_MIN_POSITIVE, scaled_radius * ratio),
                "cs_ratio": ratio,
            }
        )
    return spec


def _current_cs_radius_and_ratio(current_tip: JointCfg | None) -> tuple[float, float]:
    r"""从当前 tip 中估计 `cs` 的半径 $r$ 与高半比 $\lambda=h/r$。"""

    if current_tip is None:
        return 0.012, 1.0

    radius: float | None = None
    height: float | None = None
    for element in current_tip.collisions:
        geometry = element.geometry
        if geometry.kind == "cylinder":
            radius = float(geometry.radius)
            height = float(geometry.length)
            break
    if radius is None:
        for element in current_tip.collisions:
            geometry = element.geometry
            if geometry.kind == "sphere":
                radius = float(geometry.radius)
                break
    radius = max(_MIN_POSITIVE, float(radius if radius is not None else 0.012))
    ratio = max(_MIN_POSITIVE, float(height) / radius) if height is not None else 1.0
    return radius, ratio


def _sample_scale(cfg: TipReplaceCfg, *, tip_type: str) -> float:
    r"""按 tip_type 解析并采样无量纲 scale。"""

    low, high = _scale_range_for_tip(cfg.scale, tip_type)
    sampler = _make_range_sampler(
        (low, high),
        distrib=cfg.distrib,
        boundary_policy=cfg.boundary_policy,
    )
    return max(_MIN_POSITIVE, float(sampler()))


def _sample_cs_ratio(cfg: TipReplaceCfg, *, current_ratio: float) -> float:
    r"""采样 `cs` 高半比 $\lambda=h/r$。"""

    if cfg.cs_ratio is None:
        return max(_MIN_POSITIVE, float(current_ratio))

    mode, ratio_range = _resolve_cs_ratio_range(cfg.cs_ratio)
    sampler = _make_range_sampler(
        ratio_range,
        distrib=cfg.distrib,
        boundary_policy=cfg.boundary_policy,
    )
    sampled = float(sampler())
    if mode == "add":
        return max(_MIN_POSITIVE, float(current_ratio) + sampled)
    return max(_MIN_POSITIVE, sampled)


def _resolve_active_modes(self_mode: Any) -> tuple[str, ...]:
    r"""把 `self_mode` lowering 成当前 cfg 可能采样到的 mode 集合。"""

    if self_mode is None:
        return (_MODE_SAME,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError(f"tip_replace.self_mode must be str | dict[str, float] | None, got {type(self_mode).__name__}")

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"tip_replace.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("tip_replace.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"tip_replace.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _draw_resolved_mode(cfg: TipReplaceCfg) -> str:
    r"""按 `self_mode` 为当前样本解析最终 mode。"""

    if cfg.self_mode is None:
        return _MODE_SAME
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_SAME
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _resolve_tip_distribution(tip_range: list[str] | dict[str, float] | None) -> dict[str, float]:
    r"""把 `tip_range` 解析为 tip_type proposal 概率表。"""

    if tip_range is None:
        probability = 1.0 / len(_DEFAULT_TIP_TYPES)
        return {tip_type: probability for tip_type in _DEFAULT_TIP_TYPES}
    if isinstance(tip_range, list):
        if not tip_range:
            raise ValueError("tip_replace.tip_range list must not be empty")
        normalized = tuple(_normalize_tip_type(tip_type) for tip_type in tip_range)
        probability = 1.0 / len(normalized)
        return {tip_type: probability for tip_type in normalized}
    if not isinstance(tip_range, dict):
        raise TypeError(f"tip_replace.tip_range must be list[str] | dict[str, float] | None, got {type(tip_range).__name__}")

    distribution: dict[str, float] = {}
    total = 0.0
    for tip_type, probability in tip_range.items():
        normalized = _normalize_tip_type(tip_type)
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"tip_replace.tip_range probability must be non-negative, got {tip_type!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            distribution[normalized] = prob
    if not distribution:
        raise ValueError("tip_replace.tip_range dict must contain at least one positive-probability tip_type")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"tip_replace.tip_range probabilities must sum to 1, got {total!r}")
    return distribution


def _draw_tip_type(tip_range: list[str] | dict[str, float] | None) -> str:
    r"""按 proposal 分布采样一个 tip_type。"""

    distribution = _resolve_tip_distribution(tip_range)
    threshold = random.random()
    cumulative = 0.0
    last_tip_type = next(iter(distribution))
    for tip_type, probability in distribution.items():
        cumulative += float(probability)
        last_tip_type = tip_type
        if threshold <= cumulative + _MODE_TOLERANCE:
            return tip_type
    return last_tip_type


def _normalize_tip_type(tip_type: Any) -> str:
    r"""规范化并校验 tip_type。"""

    normalized = str(tip_type).lower()
    if normalized not in set(_DEFAULT_TIP_TYPES):
        raise ValueError(f"unsupported tip_replace tip_type: {tip_type!r}")
    return normalized


def _scale_range_for_tip(scale: Vector2 | dict[str, Vector2], tip_type: str) -> Vector2:
    r"""按 tip_type 解析无量纲 scale 采样范围。"""

    if isinstance(scale, dict):
        return scale.get(tip_type, scale.get("shared", (1.0, 1.0)))
    return scale


def _validate_scale_ranges(scale: Vector2 | dict[str, Vector2]) -> None:
    r"""校验所有 scale 区间都为正数。"""

    ranges = scale.values() if isinstance(scale, dict) else (scale,)
    for value_range in ranges:
        low, high = float(value_range[0]), float(value_range[1])
        if low <= 0.0 or high <= 0.0:
            raise ValueError(f"tip_replace.scale range must be positive, got {value_range!r}")


def _validate_cs_ratio(cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2] | None) -> None:
    r"""校验 `cs_ratio` 输入结构。"""

    if cs_ratio is None:
        return
    _resolve_cs_ratio_range(cs_ratio)


def _resolve_cs_ratio_range(cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2]) -> tuple[str, Vector2]:
    r"""解析 `cs_ratio` 为 `(mode, range)`。"""

    if isinstance(cs_ratio, dict):
        if set(cs_ratio) == {"add"}:
            return "add", _checked_ratio_range(cs_ratio["add"], allow_negative=True)
        if set(cs_ratio) == {"abs"}:
            return "abs", _checked_ratio_range(cs_ratio["abs"], allow_negative=False)
        raise ValueError("tip_replace.cs_ratio dict must contain exactly one key: 'add' or 'abs'")
    return "abs", _checked_ratio_range(cs_ratio, allow_negative=False)


def _checked_ratio_range(value_range: Vector2, *, allow_negative: bool) -> Vector2:
    r"""校验并返回高半比区间。"""

    low, high = float(value_range[0]), float(value_range[1])
    if not allow_negative and (low <= 0.0 or high <= 0.0):
        raise ValueError(f"tip_replace.cs_ratio abs range must be positive, got {value_range!r}")
    return (low, high)


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    cfg: TipReplaceCfg,
    *,
    target: HandCfg,
) -> dict[str, Any]:
    r"""把外部传入的 sampled params 统一规约成结构化 `sample`。

    支持两种入口：

    1. 新 contract：`{"sample": {...}}`；
    2. 少量测试便捷入口：直接传入 `{"resolved_self_mode": ..., "finger_specs": ...}`。
    """

    sampled = dict(sampled_params or {})
    sample = sampled.get("sample")
    if isinstance(sample, dict):
        return dict(sample)
    if "resolved_self_mode" in sampled:
        return sampled
    if cfg.self_mode == _MODE_IDENTITY:
        return {"resolved_self_mode": _MODE_IDENTITY, "finger_specs": {}}
    return TipReplaceMutator(cfg).sample_one_for_mode(target, resolved_mode=_draw_resolved_mode(cfg))


def iter_tip_types_from_sample(sample: dict[str, Any] | None) -> list[str]:
    r"""从 `tip_replace` sample payload 中取出所有 per-finger tip_type。

    这个函数供 generator summary 统计复用。`tip_range` 是 proposal 分布，因此
    summary 需要在 result 之外也能从 sampled_terms 中读取 proposal tip_type。
    """

    if not isinstance(sample, dict):
        return []
    payload = sample.get("sample") if "sample" in sample else sample
    if not isinstance(payload, dict):
        return []
    finger_specs = payload.get("finger_specs")
    if not isinstance(finger_specs, dict):
        return []
    tip_types: list[str] = []
    for spec in finger_specs.values():
        if isinstance(spec, dict) and "tip_type" in spec:
            tip_types.append(str(spec["tip_type"]))
    return tip_types


__all__ = ["TipReplaceCfg", "TipReplaceMutator", "iter_tip_types_from_sample"]
