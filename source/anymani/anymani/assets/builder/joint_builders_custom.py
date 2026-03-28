r"""TODO:自定义关节构建器配置类 `JointBuilderCfg` 和运行时类 `JointBuilder`
- 这里构建关节类型的mesh是自定义stl/obj等文件
- 主要是来自指尖的 mesh 需要精细调整
"""
from __future__ import annotations

from assets.asset_builders import JointBuilderCfg, JointBuilder
from assets.asset_base import JointCfg
from assets.asset_schema_core import Vector6, Vector3

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CustomJointBuilderCfg(JointBuilderCfg):
    r"""自定义关节构建器配置类。

    该声明式配置类包含的字段为构建类算法所需，而非单纯照搬 `JointCfg` 的所有字段

    核心思想是 “算法里人易理解和显式控制的参数” 映射到 `JointCfg` 的字段上
    """

    class_type: type["CustomJointBuilder"] | None = None
    """关联的自定义关节构建器类。"""

    mesh: dict[str, Any] = field(default_factory=dict)  # default_factory 是 @dataclass 专属的“补丁”，用来模拟普通类 __init__ 里每次创建新对象的行为。普通类里直接在 __init__ 里赋值就行了，不需要 default_facto
    """自定义 mesh 的参数字典。包括 mesh 路径，偏移和缩放参数"""

    origin: Vector6 | dict[str, Vector3] = field(default_factory=lambda: Vector6(0, 0, 0, 0, 0, 0))
    """joint frame 相对于父 link 坐标系的位姿，包含位置和平移两部分。

    需要注意的是，关节级构建器并不涉对该字段的处理，而是手指级/手掌级构建器中的算法处理该字段。
    """

    is_customized: bool = True
    """mesh是否自定义，即非URDF默认的box/cylinder/sphere。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = CustomJointBuilder


class CustomJointBuilder(JointBuilder):
    r"""自定义关节构建器。

    这里预期承担的职责是：根据 `CustomJointBuilderCfg` 里的显式参数，构建出
    对应的自定义 mesh。当前阶段，这些算法细节仍由你主导，因此这里只
    保留运行时壳子。
    """

    cfg: CustomJointBuilderCfg

    def __init__(self, cfg: CustomJointBuilderCfg):
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据 `CustomJointBuilderCfg` 构建对应的自定义 mesh。

        这里的返回值类型暂时用 `Any` 占位，具体类型取决于你选择的 mesh 表示和构建库。
        """
    # NOTE:这里的算法具有相当的 “定制性” ，不同的指尖mesh,因为是从CAD等处导出来的，mesh origin位置和 joint frame 的关系可能都不一样，因此需要每个指尖mesh单独设计构建算法，来正确处理 `origin` 字段，并把 mesh 放到正确的位置。