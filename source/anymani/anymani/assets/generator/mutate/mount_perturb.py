r"""挂载点扰动变异：在已有 `HandCfg` 上对 finger 挂载位姿做结构化 family variation。

科研语义：
`finger.mount` 表示 finger root frame $M$ 相对 palm frame $P$ 的刚体位姿
$T_{PM}=(R_{PM},\mathbf{p}_{PM})$。这一轮重构明确区分两类语义：

1. `general`：对每根 finger 的 root frame 做局部椭球小扰动；
2. `index_ring_*`：对 index/ring 施加带镜像先验的 family-level cube 采样；
3. `identity`：显式保留 pre-made 原始 mount，给正样本 / 锚点样本保留概率质量。

这里最关键的数学约定是：位姿增量采用 **右乘** 语义，
即局部扰动 $\Delta_M$ 定义在当前 mount frame 上，并写成：

$$
T'_{PM} = T_{PM}\Delta_M.
$$

因此：

$$
\mathbf{p}'_{PM} = \mathbf{p}_{PM} + R_{PM}\,\delta\mathbf{p}_M,
\qquad
R'_{PM} = R_{PM}\,\exp([\delta\boldsymbol{\omega}_M]_\times).
$$

这正对应你强调的“坐标系绕自身而不是绕 fixed palm frame 变化”的直觉。
若后续需要更完整的 screw / SE(3) 实验，这里仍可继续外推，而不会和当前
`mount_perturb` 的科研语义冲突。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector3, _ensure_tuple
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


_MODE_GENERAL = "general"
_MODE_IDENTITY = "identity"
_MODE_INDEX_RING_YAW = "index_ring_yaw_rot"
_MODE_INDEX_RING_X = "index_ring_x_pos"
_MODE_INDEX_RING_BOTH = "index_ring"

_ALL_SELF_MODES = {
    _MODE_GENERAL,
    _MODE_IDENTITY,
    _MODE_INDEX_RING_YAW,
    _MODE_INDEX_RING_X,
    _MODE_INDEX_RING_BOTH,
}

_INDEX_RING_MODES = {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}

_MODE_TOLERANCE = 1e-9


@dataclass
class MountPerturbCfg(MutatorBaseCfg):
    r"""finger mount 小扰动配置。

    这份 cfg 不再把 `general` 与 index/ring 模式硬塞进同一组 range 字段里，
    而是显式区分两类数学对象：

    - `general`：局部椭球半径 $\mathbf{r}_p,\mathbf{r}_\omega$
    - index/ring family variation：镜像 cube 区间

    这样做的原因不是接口洁癖，而是为了避免同一个字段一会儿表示上下界、
    一会儿表示椭球半轴，最终让研究者只能靠读实现猜语义。
    """

    class_type: type["MountPerturbMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类；配置层负责声明 contract，运行时负责采样与 patch lowering。"""

    self_mode: Literal[
        "identity",
        "general",
        "index_ring_yaw_rot",
        "index_ring_x_pos",
        "index_ring",
    ] | dict[str, float] | None = _MODE_GENERAL
    r"""挂载点扰动的高层 mode 选择器。

    支持三种输入语义：

    - `None`：未显式指定，默认落到 `general`
    - `str`：固定使用某一个 mode
    - `dict[str, float]`：按概率混合采样一个 mode；概率和必须严格为 1

    预设 mode：

    - `identity`
      不做 mount patch，只记录 provenance，供正样本 / 锚点样本保留权重。
    - `general`
      对所有现存 finger 的 mount frame 做局部椭球增量采样。
    - `index_ring_yaw_rot`
      只对 index/ring 的 yaw 做镜像 cube 采样；middle 保持不动；thumb 独立。
    - `index_ring_x_pos`
      只对 index/ring 的 palm-frame 横向间距做镜像 cube 采样；middle 保持不动；thumb 独立。
    - `index_ring`
      同时启用 `index_ring_x_pos` 与 `index_ring_yaw_rot`。
    """

    pos_radius: float | Vector3 | None = None
    r"""`general` 模式的位置椭球半径，单位为 meter。

    - `float`：各向同性半径 $r_x=r_y=r_z$
    - `Vector3`：各轴半径 $(r_x,r_y,r_z)$

    `general` 下的局部位置扰动采样满足：
    $$
    \delta\mathbf{p}_M = \operatorname{diag}(r_x,r_y,r_z)\mathbf{u},
    \qquad
    \|\mathbf{u}\|_2 \le 1.
    $$
    """

    rot_radius: float | Vector3 | None = None
    r"""`general` 模式的局部旋转椭球半径，单位为 rad。

    这里采样的是 mount frame 局部切空间中的小旋转向量
    $\delta\boldsymbol{\omega}_M \in \mathbb{R}^3$，而不是固定 palm frame 的逐轴角度加法。
    """

    mirror_yaw_range: Vector2 | None = None
    r"""index/ring 模式的镜像 yaw cube 区间，单位为 rad。

    这是一个一维上下界区间 $(\psi_{\min}, \psi_{\max})$。采样得到共享标量 $\delta\psi$ 后：

    - index 写回 $-\delta\psi$
    - ring 写回 $+\delta\psi$

    从而让正值更接近“左右边界指整体外展”，负值更接近“整体收拢”。
    """

    mirror_x_range: Vector2 | None = None
    r"""index/ring 模式的镜像横向间距 cube 区间，单位为 meter。

    该字段在 **palm frame 的横向 $x$ 方向** 上定义共享标量 $\delta x$：

    - index 写回 $+\delta x$
    - ring 写回 $-\delta x$

    因而正值表示两侧边界指同时远离 middle，负值表示同时靠近 middle。
    """

    thumb_pos_radius: float | Vector3 | None = None
    r"""index/ring 模式下 thumb 独立位置扰动的局部椭球半径，单位为 meter。"""

    thumb_rot_radius: float | Vector3 | None = None
    r"""index/ring 模式下 thumb 独立旋转扰动的局部椭球半径，单位为 rad。"""

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""扰动幅度分布。

    - `uniform`：在合法球/区间中均匀采样
    - `normal`：先采零均值高斯，再交给 `boundary_policy` 处理越界

    这里保留全局字段，是因为当前 `general` 椭球采样、thumb 独立扰动、
    index/ring cube 共享标量都仍可共享同一套分布假设。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""边界处理策略。

    - `none`：允许高斯样本越界
    - `clip` / `truncate`：越界后投影回合法边界
    - `resample`：越界则重采，超过预算后退化为投影
    """

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    r"""当前 cfg 真正会被采样到的 mode 集合；dict 输入时只保留正概率项。"""

    def __post_init__(self) -> None:
        r"""校验 mode 与字段契约，并补齐运行时类。

        本轮选择的是“一次性出清”而不是兼容旧接口，因此这里宁可早报错，
        也不允许旧 `sample_space + pos_range/rot_range` 语义静默混入新实现。
        """

        self.class_type = MountPerturbMutator
        self._active_modes = _resolve_active_modes(self.self_mode)
        _validate_mode_fields(self)


class MountPerturbMutator(MutatorBase):
    r"""把结构化 mount 扰动 lowering 成一次性写回 `finger.mount` 的 patch。

    与旧实现最大的差别有两点：

    1. 采样先解析 `resolved_self_mode`，再只生成当前样本真正需要的随机量；
    2. patch 写回按局部 frame 右乘增量解释，而不是 palm-frame 分量直加。
    """

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        r"""绑定一份 `MountPerturbCfg`。"""

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""返回一个结构化样本生成器。

        这里不再把 `mount_perturb` lower 成一堆轴级 sampler，例如
        `index::tx/index::ry/...`。原因是 mixed mode 下那样会把未消费的备用随机量
        也写进 sidecar，破坏科研可读性。
        """

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于结构化样本 payload 生成 mount patch。

        Args:
            target (HandCfg): 当前原始 hand schema。
            sampled_params (dict[str, Any] | None): 当前 term 的结构化样本。

        Returns:
            HandPatch: 延迟写回 patch，同时携带 sidecar/summary 所需 metadata。
        """

        sample = _normalize_sample_payload(sampled_params, self.cfg)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["mount_perturb"] = sample
        patch.metadata["post_mutate_mount_perturb"] = sample

        # `identity` 是显式 no-op mode：不改任何 mount，只保留 provenance。
        if resolved_mode == _MODE_IDENTITY:
            return patch

        if resolved_mode == _MODE_GENERAL:
            self._plan_general_patch(target, sample=sample, patch=patch)
            return patch

        self._plan_index_ring_patch(target, sample=sample, patch=patch)
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:
        r"""为当前 hand 样本生成一份已经解析好 mode 的结构化随机量。"""

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        r"""为显式 mode 的局部测试或诊断生成结构化随机量。

        generator 层需要把 `self_mode` dict 解释成 accepted/output 分布。
        若只是在 proposal 样本上事后改写 `resolved_self_mode`，就会出现
        “forced general 但没有 `finger_deltas`”这类伪样本。因此指定 mode
        的重新采样必须留在 mutator 内部完成，让每个 mode 拿到自己完整的
        低层随机变量。
        """

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb resolved mode: {resolved_mode!r}")

        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY}

        if resolved_mode == _MODE_GENERAL:
            return {
                "resolved_self_mode": _MODE_GENERAL,
                "finger_deltas": {
                    finger.name: {
                        "delta_pos_local": _sample_ball_vector(
                            self.cfg.pos_radius,
                            distrib=self.cfg.distrib,
                            boundary_policy=self.cfg.boundary_policy,
                        ),
                        "delta_rotvec_local": _sample_ball_vector(
                            self.cfg.rot_radius,
                            distrib=self.cfg.distrib,
                            boundary_policy=self.cfg.boundary_policy,
                        ),
                    }
                    for finger in target.fingers
                },
            }

        sample: dict[str, Any] = {"resolved_self_mode": resolved_mode}
        if resolved_mode in {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH}:
            sample["mirror_yaw"] = _sample_scalar(
                self.cfg.mirror_yaw_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        if resolved_mode in {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}:
            sample["mirror_x"] = _sample_scalar(
                self.cfg.mirror_x_range,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        if _hand_has_finger(target, "thumb"):
            sample["thumb_delta_pos_local"] = _sample_ball_vector(
                self.cfg.thumb_pos_radius,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
            sample["thumb_delta_rotvec_local"] = _sample_ball_vector(
                self.cfg.thumb_rot_radius,
                distrib=self.cfg.distrib,
                boundary_policy=self.cfg.boundary_policy,
            )
        return sample

    def _plan_general_patch(self, target: HandCfg, *, sample: dict[str, Any], patch: HandPatch) -> None:
        r"""把 `general` 模式的 per-finger 局部椭球增量写成 patch。"""

        finger_deltas = dict(sample.get("finger_deltas", {}))
        for finger_index, finger in enumerate(target.fingers):
            finger_payload = dict(finger_deltas.get(finger.name, {}))
            delta_pos_local = _as_vector3(finger_payload.get("delta_pos_local", (0.0, 0.0, 0.0)))
            delta_rotvec_local = _as_vector3(finger_payload.get("delta_rotvec_local", (0.0, 0.0, 0.0)))
            new_mount = _apply_local_mount_delta(
                finger.mount,
                delta_pos_local=delta_pos_local,
                delta_rotvec_local=delta_rotvec_local,
            )
            patch.add(
                ("finger", finger_index, "mount"),
                _mount_replacer(finger_index=finger_index, new_mount=new_mount),
            )

    def _plan_index_ring_patch(self, target: HandCfg, *, sample: dict[str, Any], patch: HandPatch) -> None:
        r"""把 index/ring family variation 写成 patch。"""

        finger_index_map = {finger.name: index for index, finger in enumerate(target.fingers)}
        mirror_yaw = float(sample.get("mirror_yaw", 0.0))
        mirror_x = float(sample.get("mirror_x", 0.0))

        # `mirror_pair_applied` 显式记录这一轮是否真的同时存在 index/ring。
        # 这样 sidecar 看到共享标量时，能区分“有镜像对但采样接近 0”和“根本没有成对目标”。
        sample["mirror_pair_applied"] = "index" in finger_index_map and "ring" in finger_index_map

        if sample["mirror_pair_applied"]:
            if "index" in finger_index_map:
                index_mount = target.fingers[finger_index_map["index"]].mount
                index_new_mount = _apply_mount_delta_with_palm_shift(
                    index_mount,
                    delta_pos_palm=(mirror_x, 0.0, 0.0),
                    delta_rotvec_local=(0.0, 0.0, -mirror_yaw),
                )
                patch.add(
                    ("finger", finger_index_map["index"], "mount"),
                    _mount_replacer(finger_index=finger_index_map["index"], new_mount=index_new_mount),
                )
            if "ring" in finger_index_map:
                ring_mount = target.fingers[finger_index_map["ring"]].mount
                ring_new_mount = _apply_mount_delta_with_palm_shift(
                    ring_mount,
                    delta_pos_palm=(-mirror_x, 0.0, 0.0),
                    delta_rotvec_local=(0.0, 0.0, mirror_yaw),
                )
                patch.add(
                    ("finger", finger_index_map["ring"], "mount"),
                    _mount_replacer(finger_index=finger_index_map["ring"], new_mount=ring_new_mount),
                )

        # middle 是 index/ring family variation 的几何锚点，不参与 patch。
        if "thumb" in finger_index_map:
            thumb_delta_pos_local = _as_vector3(sample.get("thumb_delta_pos_local", (0.0, 0.0, 0.0)))
            thumb_delta_rotvec_local = _as_vector3(sample.get("thumb_delta_rotvec_local", (0.0, 0.0, 0.0)))
            thumb_mount = target.fingers[finger_index_map["thumb"]].mount
            thumb_new_mount = _apply_local_mount_delta(
                thumb_mount,
                delta_pos_local=thumb_delta_pos_local,
                delta_rotvec_local=thumb_delta_rotvec_local,
            )
            patch.add(
                ("finger", finger_index_map["thumb"], "mount"),
                _mount_replacer(finger_index=finger_index_map["thumb"], new_mount=thumb_new_mount),
            )


def _resolve_active_modes(self_mode: Any) -> tuple[str, ...]:
    r"""把 `self_mode` 解析成当前 cfg 可能采样到的有效 mode 集合。"""

    if self_mode is None:
        return (_MODE_GENERAL,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError(f"mount_perturb.self_mode must be str | dict[str, float] | None, got {type(self_mode).__name__}")

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported mount_perturb self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"mount_perturb.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("mount_perturb.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"mount_perturb.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _validate_mode_fields(cfg: MountPerturbCfg) -> None:
    r"""按 active mode 集合校验字段组合是否合法。"""

    active_modes = set(cfg._active_modes)

    # 先把所有半径字段规约成合法输入，避免后面错误信息被“tuple 长度不对”掩盖。
    _normalize_radius(cfg.pos_radius, field_name="mount_perturb.pos_radius")
    _normalize_radius(cfg.rot_radius, field_name="mount_perturb.rot_radius")
    _normalize_radius(cfg.thumb_pos_radius, field_name="mount_perturb.thumb_pos_radius")
    _normalize_radius(cfg.thumb_rot_radius, field_name="mount_perturb.thumb_rot_radius")
    _normalize_range(cfg.mirror_yaw_range, field_name="mount_perturb.mirror_yaw_range")
    _normalize_range(cfg.mirror_x_range, field_name="mount_perturb.mirror_x_range")

    if _MODE_GENERAL in active_modes and cfg.pos_radius is None and cfg.rot_radius is None:
        raise ValueError("mount_perturb general mode requires at least one of pos_radius or rot_radius")

    if active_modes & {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH} and cfg.mirror_yaw_range is None:
        raise ValueError("mount_perturb index_ring_yaw_rot/index_ring requires mirror_yaw_range")
    if active_modes & {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH} and cfg.mirror_x_range is None:
        raise ValueError("mount_perturb index_ring_x_pos/index_ring requires mirror_x_range")

    if _MODE_GENERAL not in active_modes and (cfg.pos_radius is not None or cfg.rot_radius is not None):
        raise ValueError("mount_perturb pos_radius/rot_radius are only valid when general mode is active")
    if not (active_modes & _INDEX_RING_MODES) and (cfg.thumb_pos_radius is not None or cfg.thumb_rot_radius is not None):
        raise ValueError("mount_perturb thumb_pos_radius/thumb_rot_radius are only valid for index/ring modes")
    if not (active_modes & {_MODE_INDEX_RING_YAW, _MODE_INDEX_RING_BOTH}) and cfg.mirror_yaw_range is not None:
        raise ValueError("mount_perturb mirror_yaw_range is only valid for index_ring_yaw_rot/index_ring")
    if not (active_modes & {_MODE_INDEX_RING_X, _MODE_INDEX_RING_BOTH}) and cfg.mirror_x_range is not None:
        raise ValueError("mount_perturb mirror_x_range is only valid for index_ring_x_pos/index_ring")

    if active_modes == {_MODE_IDENTITY}:
        any_payload = any(
            value is not None
            for value in (
                cfg.pos_radius,
                cfg.rot_radius,
                cfg.mirror_yaw_range,
                cfg.mirror_x_range,
                cfg.thumb_pos_radius,
                cfg.thumb_rot_radius,
            )
        )
        if any_payload:
            raise ValueError("mount_perturb identity mode must not carry perturbation payload fields")


def _draw_resolved_mode(cfg: MountPerturbCfg) -> str:
    r"""按 `self_mode` 为当前样本解析最终 mode。"""

    if cfg.self_mode is None:
        return _MODE_GENERAL
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_GENERAL
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _normalize_sample_payload(sampled_params: dict[str, Any] | None, cfg: MountPerturbCfg) -> dict[str, Any]:
    r"""把 direct-call 与 batch sampling 两种输入统一规约成结构化 payload。"""

    if not sampled_params:
        return {"resolved_self_mode": _draw_resolved_mode(cfg)}
    if "sample" in sampled_params and isinstance(sampled_params["sample"], dict):
        return dict(sampled_params["sample"])
    if "resolved_self_mode" in sampled_params:
        return dict(sampled_params)
    raise ValueError("mount_perturb sampled_params must provide either {'sample': {...}} or a structured payload")


def _normalize_radius(value: float | Vector3 | None, *, field_name: str) -> Vector3 | None:
    r"""把 isotropic / anisotropic 半径统一规约成 `Vector3`。"""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        radius = float(value)
        if radius < 0.0:
            raise ValueError(f"{field_name} must be non-negative, got {value!r}")
        return (radius, radius, radius)
    radius_vec = _ensure_tuple(value, length=3, field_name=field_name)
    if any(component < 0.0 for component in radius_vec):
        raise ValueError(f"{field_name} components must be non-negative, got {radius_vec!r}")
    return radius_vec


def _normalize_range(value: Vector2 | None, *, field_name: str) -> Vector2 | None:
    r"""校验 cube 区间字段。"""

    if value is None:
        return None
    return _ensure_tuple(value, length=2, field_name=field_name)


def _as_vector3(value: Any) -> Vector3:
    r"""把结构化样本中的向量字段规约成标准 `Vector3`。"""

    return _ensure_tuple(value, length=3, field_name="mount_perturb.sample_vector")


def _hand_has_finger(hand: HandCfg, finger_name: str) -> bool:
    r"""判断当前 hand schema 是否包含指定 finger。"""

    return any(finger.name == finger_name for finger in hand.fingers)


def _sample_scalar(
    value_range: Vector2 | None,
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> float:
    r"""从一维 cube 区间采样标量；空区间退化为 0。"""

    if value_range is None:
        return 0.0
    return float(_make_range_sampler(value_range, distrib=distrib, boundary_policy=boundary_policy)())


def _sample_ball_vector(
    radius: float | Vector3 | None,
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> Vector3:
    r"""从局部椭球内采样一个向量；空半径退化为零向量。"""

    radius_vec = _normalize_radius(radius, field_name="mount_perturb.sample_radius")
    if radius_vec is None:
        return (0.0, 0.0, 0.0)

    distrib_type = distrib.get("type", "uniform") if isinstance(distrib, dict) else distrib
    distrib_type = str(distrib_type).lower()
    if distrib_type == "uniform":
        unit_vector = _sample_uniform_unit_ball()
    elif distrib_type == "normal":
        unit_vector = _sample_normalized_gaussian_ball(distrib=distrib, boundary_policy=boundary_policy)
    else:
        raise ValueError(f"unsupported mount_perturb vector distribution type: {distrib_type!r}")
    return (
        radius_vec[0] * unit_vector[0],
        radius_vec[1] * unit_vector[1],
        radius_vec[2] * unit_vector[2],
    )


def _sample_uniform_unit_ball() -> Vector3:
    r"""在单位球 $\|\mathbf{u}\|_2\le 1$ 内按体积均匀采样。"""

    direction = _sample_unit_sphere_direction()
    radius = random.random() ** (1.0 / 3.0)  # 球体体积均匀分布对应 $r \sim U^{1/3}$
    return (radius * direction[0], radius * direction[1], radius * direction[2])


def _sample_normalized_gaussian_ball(
    *,
    distrib: str | dict[str, Any],
    boundary_policy: str | None,
) -> Vector3:
    r"""采样一个归一化高斯向量，并按边界策略规约到单位球。"""

    sigma_rule = float(distrib.get("sigma_rule", 3.0)) if isinstance(distrib, dict) else 3.0
    sigma = 1.0 / max(abs(sigma_rule), 1e-12)  # 默认令约 $3\sigma$ 落在单位半径内
    if isinstance(distrib, dict) and "sigma" in distrib:
        sigma = float(distrib["sigma"])

    policy = boundary_policy or "clip"
    for _ in range(32):
        sample = (
            random.gauss(0.0, sigma),
            random.gauss(0.0, sigma),
            random.gauss(0.0, sigma),
        )
        norm = _vector_norm(sample)
        if norm <= 1.0 + _MODE_TOLERANCE or policy == "none":
            return sample
        if policy in {"resample"}:
            continue
        return _project_to_unit_ball(sample)
    return _project_to_unit_ball(sample)


def _sample_unit_sphere_direction() -> Vector3:
    r"""均匀采样单位球面方向。"""

    while True:
        candidate = (
            random.gauss(0.0, 1.0),
            random.gauss(0.0, 1.0),
            random.gauss(0.0, 1.0),
        )
        norm = _vector_norm(candidate)
        if norm > _MODE_TOLERANCE:
            return (candidate[0] / norm, candidate[1] / norm, candidate[2] / norm)


def _project_to_unit_ball(vector: Vector3) -> Vector3:
    r"""把越界向量沿原方向投影回单位球面。"""

    norm = _vector_norm(vector)
    if norm <= 1.0 + _MODE_TOLERANCE:
        return vector
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _vector_norm(vector: Vector3) -> float:
    r"""返回 3D 向量的欧氏范数。"""

    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _mount_replacer(*, finger_index: int, new_mount: PoseCfg):
    r"""构造一个把指定 finger mount 直接替换成新位姿的 patch op。"""

    def apply_mount(hand: HandCfg, *, fi=finger_index, mount=new_mount) -> None:
        hand.fingers[fi].mount = mount

    return apply_mount


def _apply_local_mount_delta(
    mount: PoseCfg,
    *,
    delta_pos_local: Vector3,
    delta_rotvec_local: Vector3,
) -> PoseCfg:
    r"""把局部 frame 增量右乘到当前 mount 上。

    数学上这里实现的是：
    $$
    T'_{PM} = T_{PM}\Delta_M.
    $$
    """

    base_rotation = _rpy_rotation_matrix(mount.rpy)
    delta_pos_palm = _apply_rotation(base_rotation, delta_pos_local)
    delta_rotation = _rotvec_to_matrix(delta_rotvec_local)
    new_rotation = _matrix_multiply(base_rotation, delta_rotation)
    return PoseCfg(
        pos=(
            mount.pos[0] + delta_pos_palm[0],
            mount.pos[1] + delta_pos_palm[1],
            mount.pos[2] + delta_pos_palm[2],
        ),
        rpy=_matrix_to_rpy(new_rotation),
    )


def _apply_mount_delta_with_palm_shift(
    mount: PoseCfg,
    *,
    delta_pos_palm: Vector3,
    delta_rotvec_local: Vector3,
) -> PoseCfg:
    r"""对 index/ring 模式应用“palm 横向位移 + 局部 yaw”组合扰动。"""

    rotated_mount = _apply_local_mount_delta(
        mount,
        delta_pos_local=(0.0, 0.0, 0.0),
        delta_rotvec_local=delta_rotvec_local,
    )
    return PoseCfg(
        pos=(
            rotated_mount.pos[0] + delta_pos_palm[0],
            rotated_mount.pos[1] + delta_pos_palm[1],
            rotated_mount.pos[2] + delta_pos_palm[2],
        ),
        rpy=rotated_mount.rpy,
    )


def _rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""构造 URDF 风格 `rpy` 旋转矩阵。

    采用与 URDF 一致的固定轴解释：
    $$
    R(\phi,\theta,\psi) = R_z(\psi) R_y(\theta) R_x(\phi).
    $$
    """

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    r"""计算 $R\mathbf{p}$。"""

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def _rotvec_to_matrix(rotvec: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""把局部旋转向量 $\delta\boldsymbol{\omega}$ 映射成旋转矩阵。"""

    theta = _vector_norm(rotvec)
    if theta <= _MODE_TOLERANCE:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ux, uy, uz = rotvec[0] / theta, rotvec[1] / theta, rotvec[2] / theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_minus_c = 1.0 - c
    return (
        (
            c + ux * ux * one_minus_c,
            ux * uy * one_minus_c - uz * s,
            ux * uz * one_minus_c + uy * s,
        ),
        (
            uy * ux * one_minus_c + uz * s,
            c + uy * uy * one_minus_c,
            uy * uz * one_minus_c - ux * s,
        ),
        (
            uz * ux * one_minus_c - uy * s,
            uz * uy * one_minus_c + ux * s,
            c + uz * uz * one_minus_c,
        ),
    )


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    r"""计算 $R = R_1R_2$。"""

    rhs_cols = (
        (rhs[0][0], rhs[1][0], rhs[2][0]),
        (rhs[0][1], rhs[1][1], rhs[2][1]),
        (rhs[0][2], rhs[1][2], rhs[2][2]),
    )
    rows: list[Vector3] = []
    for row in lhs:
        rows.append(
            (
                row[0] * rhs_cols[0][0] + row[1] * rhs_cols[0][1] + row[2] * rhs_cols[0][2],
                row[0] * rhs_cols[1][0] + row[1] * rhs_cols[1][1] + row[2] * rhs_cols[1][2],
                row[0] * rhs_cols[2][0] + row[1] * rhs_cols[2][1] + row[2] * rhs_cols[2][2],
            )
        )
    return (rows[0], rows[1], rows[2])


def _matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:
    r"""把旋转矩阵反解回 URDF 风格 `rpy`。

    这里保持与 `_rpy_rotation_matrix()` 一致的固定轴约定：
    $$
    R = R_z(\psi)R_y(\theta)R_x(\phi).
    $$
    """

    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
        return (roll, pitch, yaw)

    # gimbal lock 附近仍返回一组连续的等价 `rpy`，优先固定 yaw=0。
    roll = math.atan2(-matrix[0][1], matrix[1][1])
    yaw = 0.0
    return (roll, pitch, yaw)


__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
