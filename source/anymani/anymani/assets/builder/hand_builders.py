r"""整手级构建器：负责把 palm 与 fingers 装配成 `HandCfg`。

这里的职责边界与你在 `前后序.png`、`资产生产概略.png` 中的设计是一致的：

- 前序 `HandBuilder` 负责“造骨架”
- 后序 mutate 负责“在已有骨架上做派生”

也就是说，本文件只处理：

1. 选用哪种 palm builder
2. 选用哪种 finger builder
3. 非拇指与拇指如何挂载到 palm 上
4. 如何把这些子结构装配成一个合法的 `HandCfg`

而不会在这里做：

- link 长度扰动
- joint 删除重连
- tip 替换
- limit 微调
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import HandCfg
from ..asset_builders import FingerBuilderCfg, HandBuilder, HandBuilderCfg
from ..asset_schema_core import PoseCfg
from .finger_buiders import RegularFingerBuilderCfg
from .palm_builders import SinglePalmBuilderCfg


NON_THUMB_FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")


def _to_pose_dict(values: dict[str, PoseCfg]) -> dict[str, PoseCfg]:
    r"""把宽松挂载点输入统一规范为 `PoseCfg` 字典。"""
    return {name: PoseCfg.from_value(value) for name, value in values.items()}


@dataclass
class HumanLikeHandBuilderCfg(HandBuilderCfg):
    r"""类人手构建器配置。

    当前首轮实现聚焦于 Allegro / LEAP 这类“明显区分拇指与非拇指”的 hand。
    因此该 cfg 的关键字段不是一般装配参数，而是：

    - `handedness`
    - `finger_cfg`
    - `thumb_cfg`
    - `mounts`
    """

    class_type: type["HumanLikeHandBuilder"] | None = None
    handedness: Literal["left", "right"] = "right"
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    thumb_cfg: FingerBuilderCfg | None = None
    num_non_thumb: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        if isinstance(self.finger_cfg, dict):
            invalid = set(self.finger_cfg) - set(NON_THUMB_FINGER_NAMES)
            if invalid:
                raise ValueError(f"finger_cfg dict keys must be drawn from {NON_THUMB_FINGER_NAMES}, got {invalid}")
            self.num_non_thumb = len(self.finger_cfg)
        elif self.finger_cfg is not None and not 1 <= self.num_non_thumb <= len(NON_THUMB_FINGER_NAMES):
            raise ValueError(f"num_non_thumb must be in [1, {len(NON_THUMB_FINGER_NAMES)}]")
        self.class_type = HumanLikeHandBuilder


@dataclass
class GripperLikeHandBuilderCfg(HandBuilderCfg):
    r"""夹爪手构建器配置占位。

    本轮先不实现，但保留这个 cfg，是为了不破坏你原先“HumanLike /
    GripperLike 两分”的总体框架。
    """

    class_type: type["GripperLikeHandBuilder"] | None = None
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    num_fingers: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        self.class_type = GripperLikeHandBuilder


class HumanLikeHandBuilder(HandBuilder):
    r"""类人手装配器。

    当前装配顺序采用非常明确的优先级：

    1. 显式 `cfg.mounts`
    2. palm preset 自带的 `finger_mounts`
    3. 参数化 fallback 挂载点

    之所以这样排序，是为了兼顾两条研究路径：

    - 真实 hand 锚点：直接复用 preset mount
    - 参数化 hand 枚举：在没有 preset 时也能自动产出结构合理的初始手型
    """

    cfg: HumanLikeHandBuilderCfg

    def __init__(self, cfg: HumanLikeHandBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> HandCfg:
        r"""构建类人手的 canonical `HandCfg`。

        Returns:
            HandCfg: 已装配完成的整手描述。

        Raises:
            ValueError: 当 palm 或 finger 配置缺失时抛出。
        """

        if self.cfg.palm_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires palm_cfg")
        if self.cfg.finger_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires finger_cfg")

        palm_builder = self.cfg.palm_cfg.class_type(self.cfg.palm_cfg)
        palm = palm_builder.build()

        # 先读取 palm preset 中记录的基准挂载点。
        # 对 Allegro / LEAP 来说，这些值直接来自真实 URDF 的 palm frame。
        preset_mounts = {
            name: PoseCfg.from_value(value)
            for name, value in palm.metadata.get("finger_mounts", {}).items()
        }
        # mount 优先级：fallback < preset < 显式 cfg.mounts
        mounts = {**self._fallback_mounts(palm), **preset_mounts, **self.cfg.mounts}

        fingers = []
        if isinstance(self.cfg.finger_cfg, dict):
            items = list(self.cfg.finger_cfg.items())
        else:
            items = [(name, self.cfg.finger_cfg) for name in NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]]

        for finger_name, finger_cfg in items:
            built = self._build_named_finger(finger_cfg, finger_name, mounts.get(finger_name, PoseCfg()))
            fingers.append(built)

        if self.cfg.thumb_cfg is not None:
            thumb_mount = mounts.get("thumb", PoseCfg())
            fingers.append(self._build_named_finger(self.cfg.thumb_cfg, "thumb", thumb_mount))

        metadata = {"builder": "HumanLikeHandBuilder"}
        if self.cfg.palm_cfg.wrist_joints:
            # Question:
            # 这里保留你在 PalmBuilderCfg 中定义的“前溯腕关节”接口，但当前
            # `HandCfg` 还没有 wrist chain 的标准槽位，所以首轮只能把它们
            # 原样挂在 metadata 中，留待后续真正 lower 成链式 joint/link。
            metadata["wrist_joints"] = [joint.to_dict() for joint in self.cfg.palm_cfg.wrist_joints]

        return HandCfg(
            name=self.cfg.name,
            family=self.cfg.family,
            handedness=self.cfg.handedness,
            palm=palm,
            fingers=fingers,
            metadata=metadata,
        )

    def _build_named_finger(self, finger_cfg: FingerBuilderCfg, finger_name: str, mount: PoseCfg):
        r"""把一个 finger cfg 变成具名 finger，并赋予 mount。"""

        if not hasattr(finger_cfg, "replace"):
            raise TypeError(f"Finger cfg {finger_cfg!r} is not a dataclass-backed config")

        updates = {"name": finger_name}
        if isinstance(finger_cfg, RegularFingerBuilderCfg):
            updates["parent_link"] = "palm"
        built_cfg = finger_cfg.replace(**updates)
        finger_builder = built_cfg.class_type(built_cfg)
        finger = finger_builder.build()
        return finger.replace(name=finger_name, mount=mount, parent_link="palm")

    def _fallback_mounts(self, palm) -> dict[str, PoseCfg]:
        r"""在没有显式 mount 也没有 preset mount 时，生成参数化挂载点。

        这里只追求“结构合理的初始解”，不是追求一步到位地拟合真实手。
        对 box palm，我们采用一个极简比例模型：

        - 非拇指沿 palm 顶缘展开
        - 拇指位于 palm 侧前方，并给一个固定 yaw

        这样做的意义在于：

        1. 参数化枚举时不至于完全没有 mount 初值；
        2. 后续可以把人工调参、mutate 扰动、真实 preset 替换叠加上去。
        """

        if isinstance(self.cfg.palm_cfg, SinglePalmBuilderCfg) and self.cfg.palm_cfg.shape == "box":
            width = float(self.cfg.palm_cfg.width)
            length = float(self.cfg.palm_cfg.length)
            height = float(self.cfg.palm_cfg.height)
            names = NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]
            if len(names) == 1:
                xs = [0.0]
            else:
                # 非拇指在顶缘横向铺开，当前取约 $0.35W$ 的半展宽。
                half_span = width * 0.35
                step = 2.0 * half_span / max(len(names) - 1, 1)
                xs = [half_span - idx * step for idx in range(len(names))]
            mounts = {
                name: PoseCfg(pos=(x, length, height / 2.0))
                for name, x in zip(names, xs)
            }
            # 拇指用一个简化比例模型近似落在 palm 侧前方。
            thumb_x = width * 0.22 if self.cfg.handedness == "right" else -width * 0.22
            mounts["thumb"] = PoseCfg(
                pos=(thumb_x, length * 0.33, -height * 0.15),
                rpy=(0.0, 0.0, -1.5707963267948966 if self.cfg.handedness == "right" else 1.5707963267948966),
            )
            return mounts
        return {name: PoseCfg() for name in (*NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb], "thumb")}


class GripperLikeHandBuilder(HandBuilder):
    r"""夹爪手构建器占位。"""

    cfg: GripperLikeHandBuilderCfg

    def __init__(self, cfg: GripperLikeHandBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> HandCfg:
        raise NotImplementedError("GripperLikeHandBuilder is intentionally out of scope for the first pre-made slice.")


__all__ = [
    "NON_THUMB_FINGER_NAMES",
    "HumanLikeHandBuilderCfg",
    "GripperLikeHandBuilderCfg",
    "HumanLikeHandBuilder",
    "GripperLikeHandBuilder",
]
