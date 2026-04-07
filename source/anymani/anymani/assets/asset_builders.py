"""手部资产生成的 Builder 层空骨架。

本文件只保留你主导的运行时框架定义，不再内置任何默认构建算法。
也就是说，这里当前只回答“将来准备在哪些层级放置构建算法”，而不回答
“算法现在具体怎么实现”。

当前预设四个层级：

- `JointBuilder`: 关节级构建器
- `FingerBuilder`: 手指级构建器
- `PalmBuilder`: 掌级构建器
- `HandBuilder`: 整手级构建器

每个层级都保留两部分：

- 声明式配置对象 `*BuilderCfg`
- 运行时对象 `*Builder`

后续 joint-level / finger-level / palm-level / hand-level 的真实生成算法，
将由研究推进时逐步填入这些类中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .asset_base import AssetCfgBase, FingerCfg, HandCfg, JointCfg, PalmCfg

if TYPE_CHECKING:
    from .asset_schema_core import WristJointSpec


@dataclass
class BuilderCfg(AssetCfgBase):
    r"""构建器配置基类。

    这个类只负责表达“某个构建器需要哪些参数”，并通过 `class_type`
    指向对应的运行时类。当前阶段不在这里预埋任何算法默认值。
    """

    class_type: type["Builder"] | None = None
    """关联的构建器运行时类。"""


@dataclass
class JointBuilderCfg(BuilderCfg):
    r"""关节级构建器配置。"""

    class_type: type["Builder"] | None = None
    """关联的关节级构建器类。"""

    is_customized: bool = None
    """mesh是否自定义，即非URDF默认的box/cylinder/sphere。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointBuilder


@dataclass
class FingerBuilderCfg(BuilderCfg):
    r"""手指级构建器配置。"""

    class_type: type["Builder"] | None = None
    """关联的手指级构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerBuilder


@dataclass
class PalmBuilderCfg(BuilderCfg):
    r"""掌级构建器配置。"""

    class_type: type["Builder"] | None = None
    """关联的掌级构建器类。"""

    wrist_joints: list["WristJointSpec"] | None = None
    r"""可选的前溯腕关节声明列表。

    从 palm frame 出发、向 parent 方向声明旋转自由度。
    所有空间量在 palm frame 下表达，builder 负责反推 URDF 链式 origin。
    列表顺序为 **从 palm 到 base** 的方向（索引 0 最近 palm）。
    若为 ``None`` 或空列表，则 palm 直接作为基座的 child。
    """

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmBuilder


@dataclass
class HandBuilderCfg(BuilderCfg):
    r"""手级构建器配置。

    这里当前只保留最薄的手级入口。后续若你需要把“自定义人为形状”
    “allegro-like / leap-like / mixed-like”这类参数显式写进来，就在
    这个配置对象上继续扩展，而不是让我先替你预设一套默认语义。
    """

    class_type: type["Builder"] | None = None
    """关联的手级构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandBuilder


class Builder:
    r"""构建器基类。

    这里的 `build()` 只定义接口，不提供默认实现。这样做是为了避免
    运行时层偷偷长出“已经决定好的算法”，从而反过来污染你的研究设计。
    """

    cfg: BuilderCfg

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg

    def build(self) -> AssetCfgBase:
        r"""构建一个资产配置对象。

        Returns:
            AssetCfgBase: 构建结果。

        Raises:
            NotImplementedError: 当前只是算法入口骨架，尚未填入真实实现。
        """

        raise NotImplementedError("Builder 骨架已保留，但具体构建算法需后续实现。")


class JointBuilder(Builder):  # TODO:预计打算Primitive / Custom子类，然后再对每种子类用工厂模式，避免过度类膨胀
    r"""关节级构建器。

    预期职责是：根据 joint-level 的显式参数与规则，生成一个 `JointCfg`。
    例如后续你若把“box 底面贴 joint frame”这类建模约定正式写进代码，
    入口就应该位于这里。
    """

    def __init__(self, cfg: JointBuilderCfg):
        super().__init__(cfg)

    def build(self) -> JointCfg:
        r"""构建一个 `JointCfg`。

        Returns:
            JointCfg: 关节级资产描述。

        Raises:
            NotImplementedError: joint-level 构建算法尚未实现。
        """

        raise NotImplementedError("JointBuilder 目前只保留骨架，等待 joint-level 算法实现。")


class FingerBuilder(Builder):
    r"""手指级构建器。

    预期职责是：组合若干 `JointCfg`，并加入 finger-level 的结构规则，
    最终构成 `FingerCfg`。
    """

    def __init__(self, cfg: FingerBuilderCfg):
        super().__init__(cfg)

    def build(self) -> FingerCfg:
        r"""构建一个 `FingerCfg`。

        Returns:
            FingerCfg: 手指级资产描述。

        Raises:
            NotImplementedError: finger-level 构建算法尚未实现。
        """

        raise NotImplementedError("FingerBuilder 目前只保留骨架，等待 finger-level 算法实现。")


class PalmBuilder(Builder):
    r"""掌级构建器。

    预期职责是：描述掌部几何、掌部惯性、掌部与手指挂载关系等掌级信息，
    最终构成 `PalmCfg`。
    """

    def __init__(self, cfg: PalmBuilderCfg):
        super().__init__(cfg)

    def build(self) -> PalmCfg:
        r"""构建一个 `PalmCfg`。

        Returns:
            PalmCfg: 掌级资产描述。

        Raises:
            NotImplementedError: palm-level 构建算法尚未实现。
        """

        raise NotImplementedError("PalmBuilder 目前只保留骨架，等待 palm-level 算法实现。")


class HandBuilder(Builder):
    r"""手级构建器。

    预期职责是：在更高层组合 palm 与 fingers，形成整手 `HandCfg`。
    当前不再默认补 inertial、不再默认拼装一只可运行的 hand，也不再
    预设任何 allegro / leap / mixed 的具体生成逻辑。
    """

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)

    def build(self) -> HandCfg:
        r"""构建一个 `HandCfg`。

        Returns:
            HandCfg: 整手资产描述。

        Raises:
            NotImplementedError: hand-level 构建算法尚未实现。
        """

        raise NotImplementedError("HandBuilder 目前只保留骨架，等待 hand-level 算法实现。")


__all__ = [
    "BuilderCfg",
    "JointBuilderCfg",
    "FingerBuilderCfg",
    "PalmBuilderCfg",
    "HandBuilderCfg",
    "Builder",
    "JointBuilder",
    "FingerBuilder",
    "PalmBuilder",
    "HandBuilder",
]
