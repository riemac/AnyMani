r"""TODO:手部资产生成的 Builder 层骨架。

这里不是具体的几何构建算法文件，而是“算法应该被放在哪一层”的总框架文件。
对这个项目而言，Builder 分层本身就是研究设计的一部分，因此这里不应该只剩
一个空壳接口，而应该把各层的职责边界写清楚，避免后续 coding 时再被实现
细节反过来绑架设计。

当前预设四个层级：

- `JointBuilder`：关节级构建器
- `FingerBuilder`：手指级构建器
- `PalmBuilder`：掌级构建器
- `HandBuilder`：整手级构建器

每个层级都拆成两部分：

- 声明式配置对象 `*BuilderCfg`
- 运行时对象 `*Builder`

这样做的根本原因是把“我要什么”与“怎么构造”分开：

1. `*BuilderCfg` 用来表达研究者可直接理解、可直接控制的参数；
2. `*Builder` 用来把这些参数解释成 canonical asset schema。

因此，本文件默认不替你私自预设某一版 hand/finger/palm 的算法，只保留
框架与职责说明，真正的算法放在各自的 builder 子文件里实现。
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
    """关联的构建器运行时类。

    这里采用“配置类反指运行时类”的显式方式，而不是更重的全局 registry。
    原因很简单：当前仍处在科研探索期，分层和字段都可能继续演化；显式关系
    最利于逐段阅读与审查，不会把控制流藏进框架魔法里。
    """


@dataclass
class JointBuilderCfg(BuilderCfg):
    r"""关节级构建器配置。

    关节级 builder 的主要任务，是描述“本 joint 之后这段 child link 的局部
    几何与惯量”。它一般不决定整根 finger 的串联关系；那部分属于 finger
    级 builder 的职责。
    """

    class_type: type["Builder"] | None = None
    """关联的关节级构建器类。"""

    is_customized: bool = None
    """mesh 是否自定义。

    - `False`：通常表示可以用 URDF 默认 primitive，如 box/cylinder/sphere
    - `True`：通常表示需要 custom mesh、复合 mesh 或额外导出逻辑
    """

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointBuilder  # 默认回落到最薄的 joint 壳子


@dataclass
class FingerBuilderCfg(BuilderCfg):
    r"""手指级构建器配置。

    手指级 builder 是整个资产生成链里最关键的一层之一，因为它同时处理两套
    关系：

    1. `joint frame -> mesh frame`
    2. `parent link frame -> child joint frame`

    因此，thumb / non-thumb、全驱 / 欠驱、球关节 / 非球关节这类差异，
    往往都会首先在这一层被结构化表达。
    """

    class_type: type["Builder"] | None = None
    """关联的手指级构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerBuilder  # 默认只保留 finger 层接口，不注入算法


@dataclass
class PalmBuilderCfg(BuilderCfg):
    r"""掌级构建器配置。

    Palm 不只是一个“大块碰撞体”，也是所有 finger mount 的基座参考系。
    因此 palm frame 的原点、朝向、尺寸语义，都会直接影响 hand 级装配。
    """

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
            self.class_type = PalmBuilder  # palm 层默认入口


@dataclass
class HandBuilderCfg(BuilderCfg):
    r"""手级构建器配置。

    HandBuilder 的核心职责是**装配**：接收一个 palm builder cfg 和若干
    finger builder cfgs，分别调用对应的 builder 生成 PalmCfg / FingerCfg，
    再组装成 HandCfg。

    本类只承载"装配所需的声明式参数"，不承载具体的几何算法——
    那些由 PalmBuilderCfg / FingerBuilderCfg 的子类各自负责。
    """

    name: str = "hand"
    """手资产名称。"""

    family: str = "generic"
    """手族标签，例如 ``"leap"``、``"allegro"`` 或 ``"generic"``。

    这里的 `family` 更偏向 provenance / 研究分组标签，而不是严格的运行时
    分发键。真正的构建逻辑仍由 palm/finger 子 cfg 自己决定。
    """

    palm_cfg: "PalmBuilderCfg | None" = None
    """掌部构建器配置。

    子类可使用：

    - `SinglePalmBuilderCfg`：连续参数化 palm
    - `ComPalmBuilderCfg`：复合 preset palm
    """

    class_type: type["Builder"] | None = None
    """关联的手级构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandBuilder  # 默认只保留整手装配入口


class Builder:
    r"""构建器基类。

    这里的 `build()` 只定义接口，不提供默认实现。这样做是为了避免
    运行时层偷偷长出“已经决定好的算法”，从而反过来污染你的研究设计。
    """

    cfg: BuilderCfg

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg  # 运行时 builder 显式持有声明式 cfg，便于调试与追溯

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
        super().__init__(cfg)  # 当前这里只保留 joint 级统一入口

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
        super().__init__(cfg)  # finger 级真实算法由更细的子类承担

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
        super().__init__(cfg)  # palm 级真实算法由子类承担

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

    职责是**装配**：分别调用 PalmBuilder 和 FingerBuilder 产出配件，
    再组合成 HandCfg。不负责自己生成几何——那是子 builder 的事。
    """

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)  # hand 级只做装配，不抢下层几何职责

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
