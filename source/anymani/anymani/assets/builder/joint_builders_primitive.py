r"""TODO:基础几何构建器配置类 `JointBuilderCfg` 和运行时类 `JointBuilder`。构造时默认 visual 和 collision 一致。包含的类型有
- box
- cylinder
- sphere
与 URDF 默认支持的一致
"""
from __future__ import annotations

from assets.asset_builders import JointBuilderCfg, JointBuilder
from assets.asset_base import JointCfg
from assets.asset_base import Vector6, Vector3

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PrimJointBuilderCfg(JointBuilderCfg):
    r"""基础几何构建器配置类。

    该声明式配置类包含的字段为构建类算法所需，而非单纯照搬 `JointCfg` 的所有字段

    核心思想是 “算法里人易理解和显式控制的参数” 映射到 `JointCfg` 的字段上
    """

    class_type: type["PrimJointBuilder"] | type["ComPrimJointBuilder"] |None = None
    """关联的基础几何构建器类。"""

    mesh: dict[str, Any] = field(default_factory=dict)  # default_factory 是 @dataclass 专属的“补丁”，用来模拟普通类 __init__ 里每次创建新对象的行为。普通类里直接在 __init__ 里赋值就行了，不需要 default_facto
    
    joint_type: Literal["revolute", "fixed"] = "revolute"
    """"""

    origin: Vector6 | dict[str, Vector3] = field(default_factory=lambda: Vector6(0, 0, 0, 0, 0, 0))
    """joint frame 相对于父 link 坐标系的位姿，包含位置和平移两部分。

    需要注意的是，关节级构建器并不涉对该字段的处理，而是手指级/手掌级构建器中的算法处理该字段。FIXME：所以我认为该字段可取消，如无必要，勿增实体
    """

    axis: Vector3 = None
    """旋转轴，仅在 joint_type 为 revolute 时需要。
    
    该值的赋予也处在手指级/手掌级构建器中的算法中，而非关节级构建器中。
    """

    is_customized: bool = False
    """mesh是否自定义，即非URDF默认的box/cylinder/sphere。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PrimJointBuilder


class PrimJointBuilder(JointBuilder):
    r"""基础几何构建器。

    这里预期承担的职责是：根据 `PrimJointBuilderCfg` 里的显式参数，构建出
    对应的基础几何 mesh。当前阶段，这些算法细节仍由你主导，因此这里只
    保留运行时壳子。
    """

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据 `PrimJointBuilderCfg` 构建对应的基础几何 mesh。主要是 mesh size 和 frame

        这里的返回值类型暂时用 `Any` 占位，具体类型取决于你选择的 mesh
        表示和构建库。
        """
        pass
    # 这里看是采用工厂方法模式，还是采用子类分化
    # 我们知道 urdf 中偏移量默认为0时，即 mesh 的 origin 全为0时，基础形状的中心及坐标系和 joint frame是完全重合的，我称之为 “旧约”
    # 但以下算法则修改来偏移量为0时的mesh origin行为，我称之为 “新约”，更符合我对手指构建的想象 
    # --- TODO:算法之一: Box（最常用，一般用作手指link/palm的构成）
    # 用 box mesh 来构造 joint/child link 的骨肉，这里最主要关注的是 box 的尺寸与相对于 joint frame的偏移
    # 输入: 偏移量 $d=(d_x, d_y, d_z)\in \mathbb{R}^3$, box 尺寸 $s=(s_x, s_y, s_z)\in \mathbb{R}^3$，以及 joint frame 的定义（旋转轴和坐标系语义）
    # 输出: joint frame 下的 box mesh frame，即 mesh frame 相对于 joint frame 的位置, $x_{mesh} = d_x, y_{mesh} = s_y/2 + d_y, z_{mesh} = d_z$
    # 补充: 这个语义可结合 `AnyMani/source/anymani/anymani/assets/平面示意.png` 来理解。我这里默认用 x-y 轴来建立手指的运动学树的平面图
    # 就是假设 $d = 0$，那么 box 的底面就和 joint frame 的 x-z 平面重合，box 的中心在 joint frame 的 y 轴延伸上，距离为 $s_y/2$
    # 如果 $d$ 不为零，那么就是在这个基础上进行平移，例如 $d_y > 0$ 就是把 box 往 joint frame 的 y 轴正方向平移，$d_y < 0$ 就是往 y 轴负方向平移
    # --- NOTE:再补上 rpy 偏移量，由于 finger_buiders.py 的关系；
    # 再补上 CMC1 joint mesh 特例，和 RegularThumbBuilderCfg 对应
    # 再补上 

    # --- TODO:算法之二: Cyliner（Box下的替代，一般也用作手指link的构成）---
    # 用 cylinder mesh 来构造 joint/child link 的骨肉，这里最主要关注的是 cylinder 的半径与高度，以及相对于 joint frame的偏移
    # 输入: 偏移量 $d=(d_x, d_y, d_z)\in \mathbb{R}^3$, cylinder 尺寸 $s=(r, h)\in \mathbb{R}^2$，以及 joint frame 的定义（旋转轴和坐标系语义）
    # 输出: joint frame 下的 cylinder mesh frame，即 mesh frame 相对于 joint frame 的位置, $x_{mesh} = d_x, y_{mesh} = h/2 + d_y, z_{mesh} = d_z$
    # 就是假设 $d = 0$，那么 cylinder 的底面就和 joint frame 的 x-z 平面重合，cylinder 的中心在 joint frame 的 y 轴延伸上，距离为 $h/2$
    # 如果 $d$ 不为零，那么就是在这个基础上进行平移，例如 $d_y > 0$ 就是把 cylinder 往 joint frame 的 y 轴正方向平移，$d_y < 0$ 就是往 y 轴负方向平移

    # --- TODO:算法之三: Sphere（特殊情况，我没想到它怎么用于组成手指或手背。情况很少，一种是球型关节 mesh,但这个也要复合 Box 或 Cyliner，且我目前关于手型泛化的idea暂不涉及球型关节的手，这里预留一个接口，未来再实现）---

class ComPrimJointBuilder(JointBuilder):
    r"""基础几何复合构建器

    非单个基础图形构建 joint mesh,而是由至少2个及以上 mesh 构建复合 mesh，一般用作指尖的构建，如 cylinder + sphere 的组合，或者 box + sphere 的组合等
    """

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据 `PrimJointBuilderCfg` 构建对应的复合 mesh。

        这里的返回值类型暂时用 `Any` 占位，具体类型取决于你选择的 mesh 表示和构建库。
        """
        pass

    # --- TODO:算法之一 ---：cylinder + sphere 构造指尖的复合 mesh（最常用）
    # 输入: 半径 $r$，高度 $h$，偏移 $d \in \mathbb{R}^3$，表示 cylinder 尺寸为 $(r, h)$，sphere 尺寸为 $r$
    # 输出: joint frame 下的复合 mesh frame，即 cylinder mesh frame 和 sphere mesh frame 相对于 joint frame 的位置
    # <续> $x_c = d_x, y_c = d_y + h/2, z_c = d_z$，而 $x_s = d_x, y_s = d_y + h, z_s = d_z$
    # 补充: 这个语义同样可结合 `AnyMani/source/anymani/anymani/assets/平面示意.png` 来理解。我这里默认用 x-y 轴来建立手指的运动学树的平面图
    # <续> 就是假设 $d = 0$，那么 cylinder 的底面就和 joint frame 的 x-z 平面重合，cylinder 的中心在 joint frame 的 y 轴延伸上，距离为 $h/2$
    # <续> sphere 的中心在 joint frame 的 y 轴延伸上，距离为 $h$，也就是 cylinder 的顶面。这样保证球面最大截面和圆柱顶面重合，从而形成比较自然的指尖形状

    # --- TODO:算法之二 ---：box + sphere 构造指尖的复合 mesh
    # 输入: 半径 $r$，高度 $h$，宽度 $w$，表示 box 尺寸为 $(r, w, h)$，sphere 尺寸为 $r$，偏移 $d \in \mathbb{R}^3$
    # 输出: joint frame 下的复合 mesh frame，即 box mesh frame 和 sphere mesh frame 相对于 joint frame 的位置
    # <续> $x_b = d_x, y_b = d_y + h/2, z_b = d_z$，而 $x_s = d_x, y_s = d_y + h, z_s = d_z$
    # 补充: 这个语义同样可结合 `AnyMani/source/anymani/anymani/assets/平面示意.png` 来理解。我这里默认用 x-y 轴来建立手指的运动学树的平面图
    # <续> 就是假设 $d = 0$，那么 box 的底面就和 joint frame 的 x-z 平面重合，box 的中心在 joint frame 的 y 轴延伸上，距离为 $h/2$
    # <续> sphere 的中心在 joint frame 的 y 轴延伸上，距离为 $h$，也就是 box 的顶面。这样保证球面最大截面和盒子顶面重合，从而形成比较自然的指尖形状