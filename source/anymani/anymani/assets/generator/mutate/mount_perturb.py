r"""挂载点扰动变异：以 deferred patch 方式微调 finger root 位姿。

科研语义：
`finger.mount` 表示 finger root frame $M$ 相对 palm frame $P$ 的刚体位姿
$T_{PM}=(R_{PM},\mathbf{p}_{PM})$。本首版实现采用小扰动近似：

$$
\mathbf{p}'_{PM}=\mathbf{p}_{PM}+\delta\mathbf{p},\quad
\mathbf{r}'_{PM}=\mathbf{r}_{PM}+\delta\mathbf{r}.
$$

这里直接在 RPY 分量上加小角度，是因为 quick post-mutate 的默认扰动量级
约为 $0.03\text{rad}$，处于小角度近似可接受范围；若后续需要大角度姿态
扰动，再把这里替换成真正的 $SO(3)$ 指数映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import PoseCfg
from ._base import HandPatch, MutatorBase
from ._distribution import ScalarDistributionCfg, normalize_distribution


@dataclass
class MountPerturbCfg(AssetCfgBase):
    r"""finger mount 小扰动配置。

    该配置保留用户草稿中的高层语义字段，同时兼容 quick/test 里已经使用的
    `target_fingers`、`translation_distribution` 和 `rotation_distribution`。
    pipeline 不直接理解这些字段，而是由 `MountPerturbMutator` lowering 成
    每个 finger 的 $t_x,t_y,t_z,r_r,r_p,r_y$ 六个局部随机变量。
    """

    class_type: type["MountPerturbMutator"] | None = None
    target_fingers: tuple[str, ...] | None = None
    perturb_rotation: bool = False
    translation_distribution: Any = None
    rotation_distribution: Any = None
    clip_translation: float | None = None
    clip_rotation: float | None = None

    disturb_unit: Literal["deg", "rad"] = "rad"
    sample_space: dict[Literal["pos", "rot"], Literal["cube", "ellipsoid"]] = field(
        default_factory=lambda: {"pos": "ellipsoid", "rot": "ellipsoid"}
    )
    self_mode: Literal["general", "index_ring_yaw_rot", "index_ring_x_pos", "index_ring"] | dict[str, float] | None = "general"
    pos_range: tuple[float, float] | tuple[float, float, float, float, float, float] | None = None
    rot_range: tuple[float, float] | tuple[float, float, float, float, float, float] | None = None
    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None

    def __post_init__(self) -> None:
        self.class_type = MountPerturbMutator
        if isinstance(self.target_fingers, list):
            self.target_fingers = tuple(self.target_fingers)
        self.translation_distribution = normalize_distribution(
            self.translation_distribution,
            default=ScalarDistributionCfg(kind="fixed", value=0.0),
        )
        self.rotation_distribution = normalize_distribution(
            self.rotation_distribution,
            default=ScalarDistributionCfg(kind="fixed", value=0.0),
        )


class MountPerturbMutator(MutatorBase):
    r"""将挂载点扰动 lowering 成一次性写入 `finger.mount` 的 patch。"""

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""为每个目标 finger 声明平移和可选旋转扰动变量。"""

        specs: dict[str, Any] = {}
        for _, finger in _iter_target_fingers(target, self.cfg.target_fingers):
            for axis_name in ("tx", "ty", "tz"):
                specs[f"{finger.name}::{axis_name}"] = self.cfg.translation_distribution
            if self.cfg.perturb_rotation:
                for axis_name in ("rr", "rp", "ry"):
                    specs[f"{finger.name}::{axis_name}"] = self.cfg.rotation_distribution
        return specs

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""生成 finger-level mount patch，不立即修改原始 `HandCfg`。"""

        sampled_params = sampled_params or {}
        patch = HandPatch()
        for finger_index, finger in _iter_target_fingers(target, self.cfg.target_fingers):
            tx = _clip(float(sampled_params.get(f"{finger.name}::tx", 0.0)), self.cfg.clip_translation)
            ty = _clip(float(sampled_params.get(f"{finger.name}::ty", 0.0)), self.cfg.clip_translation)
            tz = _clip(float(sampled_params.get(f"{finger.name}::tz", 0.0)), self.cfg.clip_translation)
            rr = _clip(float(sampled_params.get(f"{finger.name}::rr", 0.0)), self.cfg.clip_rotation)
            rp = _clip(float(sampled_params.get(f"{finger.name}::rp", 0.0)), self.cfg.clip_rotation)
            ry = _clip(float(sampled_params.get(f"{finger.name}::ry", 0.0)), self.cfg.clip_rotation)

            def apply_mount(hand: HandCfg, *, fi=finger_index, dp=(tx, ty, tz), dr=(rr, rp, ry)) -> None:
                mount = hand.fingers[fi].mount
                hand.fingers[fi].mount = PoseCfg(
                    pos=(mount.pos[0] + dp[0], mount.pos[1] + dp[1], mount.pos[2] + dp[2]),
                    rpy=(mount.rpy[0] + dr[0], mount.rpy[1] + dr[1], mount.rpy[2] + dr[2]),
                )

            patch.add(("finger", finger_index, "mount"), apply_mount)
        return patch


def _iter_target_fingers(hand: HandCfg, target_fingers: tuple[str, ...] | None):
    target_set = set(target_fingers or ())
    for finger_index, finger in enumerate(hand.fingers):
        if target_set and finger.name not in target_set:
            continue
        yield finger_index, finger


def _clip(value: float, bound: float | None) -> float:
    if bound is None:
        return value
    abs_bound = abs(float(bound))
    return max(-abs_bound, min(abs_bound, value))


__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
