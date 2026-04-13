r"""TODO:整手级构建器：负责把 palm 与 fingers 装配成 `HandCfg`。

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

换句话说，本文件负责的是“前序造骨架”，不是“后序变体扰动”。
这也是为什么它要高度尊重 palm/finger builder 已经给出的局部语义，而不该
在 hand 级偷偷篡改下层设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import HandCfg
from ..asset_builders import FingerBuilderCfg, HandBuilder, HandBuilderCfg
from ..asset_schema_core import PoseCfg
from .finger_buiders import RegularFingerBuilderCfg
from .palm_builders import ComPalmBuilderCfg, SinglePalmBuilderCfg


NON_THUMB_FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")  # 统一非拇指命名顺序


def _to_pose_dict(values: dict[str, PoseCfg]) -> dict[str, PoseCfg]:
    r"""把宽松挂载点输入统一规范为 `PoseCfg` 字典。"""
    return {name: PoseCfg.from_value(value) for name, value in values.items()}  # 兼容 tuple / dict / PoseCfg


def _ensure_resolved_finger_cfg(slot_name: str, cfg: FingerBuilderCfg | str | None) -> FingerBuilderCfg | None:
    r"""确保 hand builder 只接收到已解析好的 finger cfg，而不是 preset 字符串。"""

    if cfg is None:
        return None
    if isinstance(cfg, str):
        raise TypeError(
            f"{slot_name} must be a resolved FingerBuilderCfg, got preset string {cfg!r}. "
            "Resolve preset names in `assets.presets` or `RecipeLoader` before constructing HumanLikeHandBuilderCfg."
        )
    return cfg


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
    handedness: Literal["left", "right"] = "right"  # 当前先显式区分左右手
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None  # 非拇指配置：builder 层只接受已解析 cfg
    thumb_cfg: FingerBuilderCfg | None = None  # 拇指配置：preset 解析必须在更上层完成
    num_non_thumb: int = 3  # 默认 index/middle/ring 三根非拇指
    mounts: dict[str, PoseCfg] = field(default_factory=dict)  # 显式挂载点覆盖；preset 解析应已在 preset/recipe 层完成

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)  # 显式 mount 一律先规约到标准 PoseCfg
        if isinstance(self.finger_cfg, dict):
            self.finger_cfg = {
                name: _ensure_resolved_finger_cfg(f"finger_cfg[{name!r}]", cfg)
                for name, cfg in self.finger_cfg.items()
            }
            invalid = set(self.finger_cfg) - set(NON_THUMB_FINGER_NAMES)
            if invalid:
                raise ValueError(f"finger_cfg dict keys must be drawn from {NON_THUMB_FINGER_NAMES}, got {invalid}")
            self.num_non_thumb = len(self.finger_cfg)  # 字典模式下非拇指数由键数决定
        elif self.finger_cfg is not None and not 1 <= self.num_non_thumb <= len(NON_THUMB_FINGER_NAMES):
            raise ValueError(f"num_non_thumb must be in [1, {len(NON_THUMB_FINGER_NAMES)}]")
        else:
            self.finger_cfg = _ensure_resolved_finger_cfg("finger_cfg", self.finger_cfg)
        self.thumb_cfg = _ensure_resolved_finger_cfg("thumb_cfg", self.thumb_cfg)
        self.class_type = HumanLikeHandBuilder  # human-like hand 统一走这个装配器


@dataclass
class GripperLikeHandBuilderCfg(HandBuilderCfg):
    r"""夹爪手构建器配置占位。

    本轮先不实现，但保留这个 cfg，是为了不破坏你原先“HumanLike /
    GripperLike 两分”的总体框架。
    """

    class_type: type["GripperLikeHandBuilder"] | None = None
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None  # 夹爪手未来可按指分配 cfg
    num_fingers: int = 3  # 夹爪数量
    mounts: dict[str, PoseCfg] = field(default_factory=dict)  # 指根挂载点

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)  # 先把 mounts 规约好，虽然本轮尚未实现
        self.class_type = GripperLikeHandBuilder


class HumanLikeHandBuilder(HandBuilder):
    r"""类人手装配器。

    当前装配顺序采用非常明确的优先级：

        1. 显式 `cfg.mounts`
        2. palm metadata 自带的 `finger_mounts`
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

        palm_builder = self.cfg.palm_cfg.class_type(self.cfg.palm_cfg)  # palm 的具体几何由 palm builder 自己负责
        palm = palm_builder.build()  # hand-level 这里只消费已经构建好的 PalmCfg

        metadata_mounts = self._metadata_mounts(palm)
        # mount 优先级：fallback < palm metadata < 显式 cfg.mounts
        mounts = {**self._fallback_mounts(palm), **metadata_mounts, **self.cfg.mounts}

        fingers = []
        if isinstance(self.cfg.finger_cfg, dict):
            items = list(self.cfg.finger_cfg.items())  # 字典模式：允许每根非拇指独立配置
        else:
            items = [(name, self.cfg.finger_cfg) for name in NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]]  # 共享非拇指模板

        for finger_name, finger_cfg in items:
            built = self._build_named_finger(finger_cfg, finger_name, mounts.get(finger_name, PoseCfg()))  # 给每根 finger 注入名字与挂载点
            fingers.append(built)

        if self.cfg.thumb_cfg is not None:
            thumb_mount = mounts.get("thumb", PoseCfg())  # 拇指挂点允许由 preset 或显式覆盖
            fingers.append(self._build_named_finger(self.cfg.thumb_cfg, "thumb", thumb_mount))

        metadata = {"builder": "HumanLikeHandBuilder"}  # sidecar / 调试时可回溯 hand builder 来源
        if self.cfg.palm_cfg.wrist_joints:
            # Question:
            # 这里保留你在 PalmBuilderCfg 中定义的“前溯腕关节”接口，但当前
            # `HandCfg` 还没有 wrist chain 的标准槽位，所以首轮只能把它们
            # 原样挂在 metadata 中，留待后续真正 lower 成链式 joint/link。
            metadata["wrist_joints"] = [joint.to_dict() for joint in self.cfg.palm_cfg.wrist_joints]

        return HandCfg(
            name=self.cfg.name,  # hand 名
            family=self.cfg.family,  # hand family 标签
            handedness=self.cfg.handedness,  # 左右手属性
            palm=palm,  # 已构建好的 palm
            fingers=fingers,  # 已装配好的 fingers
            metadata=metadata,  # 附加 provenance / wrist 占位信息
        )

    def _build_named_finger(self, finger_cfg: FingerBuilderCfg, finger_name: str, mount: PoseCfg):
        r"""把一个 finger cfg 变成具名 finger，并赋予 mount。"""

        if not hasattr(finger_cfg, "replace"):
            raise TypeError(f"Finger cfg {finger_cfg!r} is not a dataclass-backed config")

        updates = {"name": finger_name}  # 先注入 finger 逻辑名，保证 joint/link 命名稳定
        if isinstance(finger_cfg, RegularFingerBuilderCfg):
            updates["parent_link"] = "palm"  # human-like hand 中 finger 根默认挂到 palm
        built_cfg = finger_cfg.replace(**updates)  # 不原地改 preset，对调用方保持纯净
        finger_builder = built_cfg.class_type(built_cfg)
        finger = finger_builder.build()
        return finger.replace(name=finger_name, mount=mount, parent_link="palm")  # 最终再把 mount 写进 FingerCfg

    def _metadata_mounts(self, palm) -> dict[str, PoseCfg]:
        r"""从 palm metadata 中读取显式挂载点。

        hand builder 不再负责解析任何 preset 名称；若 palm builder 或 preset 层
        已经提供了 `finger_mounts`，这里就直接消费。
        """

        if not isinstance(palm.metadata, dict):
            return {}
        finger_mounts = palm.metadata.get("finger_mounts")
        if not isinstance(finger_mounts, dict):
            return {}
        return _to_pose_dict(finger_mounts)

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
            width = float(self.cfg.palm_cfg.width)  # palm 宽度 $W$
            length = float(self.cfg.palm_cfg.length)  # palm 长度 $L$
            height = float(self.cfg.palm_cfg.height)  # palm 厚度 $H$
            names = NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]
            if len(names) == 1:
                xs = [0.0]  # 只有一根非拇指时直接放在中线
            else:
                # 非拇指在顶缘横向铺开，当前取约 $0.35W$ 的半展宽。
                half_span = width * 0.35  # 非拇指横向半展宽
                step = 2.0 * half_span / max(len(names) - 1, 1)  # 相邻 finger 的横向步长
                xs = [half_span - idx * step for idx in range(len(names))]  # 从 radial 侧到 ulnar 侧铺开
            mounts = {
                name: PoseCfg(pos=(x, length, height / 2.0))  # 非拇指统一挂在 palm 顶缘
                for name, x in zip(names, xs)
            }
            # 拇指用一个简化比例模型近似落在 palm 侧前方。
            thumb_x = width * 0.22 if self.cfg.handedness == "right" else -width * 0.22
            mounts["thumb"] = PoseCfg(
                pos=(thumb_x, length * 0.33, -height * 0.15),
                rpy=(0.0, 0.0, -1.5707963267948966 if self.cfg.handedness == "right" else 1.5707963267948966),
            )
            return mounts
        return {name: PoseCfg() for name in (*NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb], "thumb")}  # 若无更好先验，就全部回退到零挂载点


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
