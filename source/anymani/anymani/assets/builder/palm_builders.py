r"""掌心级构建器相关的配置类和构建类。

"""

from __future__ import annotations

from assets.asset_builders import PalmBuilder, PalmBuilderCfg
from assets.asset_base import PalmCfg
from assets.asset_schema_core import Vector6, Vector3, Vector2

from dataclasses import dataclass, field
from typing import Any, Literal

# --- 手掌的声明式配置类 --- #
@dataclass
class SinglePalmBuilderCfg(PalmBuilderCfg):
    """基础几何形状构成的手掌配置类。
    """
    shape: Literal["box", "cylinder", "sphere", "ellipse"] = "box"
    """手掌的基础几何形状类型，目前支持 "box"（最常用）、"cylinder" 、 "sphere"（很少用） 和 "ellipse"，与 URDF 默认支持的一致。
    
    其中 "ellipse" 的类型也是 "sphere"，但在构建算法里会根据 `a` 和 `b` 的值调整为椭球体，这一点通过 urdf scale 字段来决定。
    """

    length: float = None
    """手掌长度，仅在 shape 为 "box" 时需要。沿 y轴 表示"""

    width: float = None
    """手掌宽度，仅在 shape 为 "box" 时需要。沿 x轴 表示"""

    height: float = None
    """手掌高度。所有形状都必填的字段。沿 z轴 表示"""

    radius: float = None
    """手掌半径，仅在 shape 为 "cylinder" 和 "sphere" 时需要"""

    a: float = None
    """椭圆体长轴，仅在 shape 为 "ellipse" 时需要。沿 x轴 表示"""

    b: float = None
    """椭圆体短轴，仅在 shape 为 "ellipse" 时需要。沿 y轴 表示"""

    def __post_init__(self):
        return super().__post_init__()

@dataclass
class ComPalmBuilderCfg(PalmBuilderCfg):
    """多基础几何形状复合而成的手掌配置类。
    """


@dataclass
class CustomPalmBuilderCfg(PalmBuilderCfg):
    """自定义手掌 mesh的配置类。

    主要从 CAD/Blender 等软件导出
    """


# --- 手掌的构建运行时类 --- #
class SinglePalmBuilder(PalmBuilder):
    """基础几何形状构成的手掌构建类。
    """

    def __init__(self, cfg: SinglePalmBuilderCfg):
        super().__init__(cfg)

    def build(self) -> PalmCfg:
        """构建手掌配置。

        Returns:
            PalmCfg: 构建完成的手掌配置。
        """
    
    # ================================================================
    #  Palm 设计帧约定（方案 C，与 PrimJointBuilder 同构）
    # ================================================================
    # 原点：形体底部中心，z 方向居中于厚度 (z = h/2)
    #   x → 右（宽度方向）
    #   y → 上（朝指方向，即 palm 的"生长方向"）
    #   z → 右手定则（朝外）
    #
    # 与 PrimJointBuilder 的 box 算法数学同构：
    #   底面 ≡ palm frame 的 x‑z 平面，几何沿 +y 生长。
    #
    # 注：腕关节前溯（wrist_joints）是 PalmBuilderCfg 基类的职责，
    #     由外层或 HandBuilder 统一消费，本 build() 不处理。
    # 可参考 `AnyMani/source/anymani/anymani/assets/doc/Single-Palm.jpg``
    # ================================================================

    # --- TODO:算法之一：box（像 leap, allegro 这类手比较常用）
    # 输入：宽 $w$ (x), 长 $l$ (y), 高 $h$ (z), 质量 $m$
    #
    # 几何中心（collision / visual origin）：
    #   $\mathbf{c} = (0,\; l/2,\; 0)$
    #
    # URDF 输出：<box size="w l h"/>，origin = $\mathbf{c}$
    #
    # 惯量（均质长方体，COM = $\mathbf{c}$）：

    #   $I_{xx} = \frac{m}{12}(l^2 + h^2)$

    #   $I_{yy} = \frac{m}{12}(w^2 + h^2)$

    #   $I_{zz} = \frac{m}{12}(w^2 + l^2)$

    # 如有可复用的惯量计算方法则直接调用，否则待后续 coding 实现。

    # --- TODO:算法之二：cylinder（比较适合夹爪手）
    # 输入：半径 $r$, 高 $h$ (z), 质量 $m$
    #
    # 圆柱轴沿 z（与 URDF <cylinder> 默认一致），x‑y 截面为圆。
    # 因径向对称且 z 已居中，palm frame 直接落在几何中心。
    #
    # 几何中心：$\mathbf{c} = (0,\; 0,\; 0)$
    #
    # URDF 输出：<cylinder radius="r" length="h"/>，origin = $\mathbf{c}$
    # 手指在圆周面上的 $(r\cos\theta,\; r\sin\theta)$ 处挂载。
    #
    # 惯量（均质圆柱，COM = $\mathbf{c}$）：

    #   $I_{xx} = I_{yy} = \frac{m}{12}(3r^2 + h^2)$

    #   $I_{zz} = \frac{m}{2} r^2$

    # 如有可复用的惯量计算方法则直接调用，否则待后续 coding 实现。

    # --- TODO:算法之三：ellipse（夹爪手和类人手都可以用）
    # 输入：x 半轴 $a$, y 半轴 $b$, 高 $h$ (z), 质量 $m$
    #
    # 在 URDF 中用 <sphere radius="1.0"> + scale $(a, b, c)$ 模拟，$c = h/2$。
    # 原点在椭球底部（y 最低点），沿 +y 从 0 到 2b。
    #
    # 几何中心：$\mathbf{c} = (0,\; b,\; 0)$
    #
    # URDF 输出：sphere + scale $(a, b, h/2)$；origin = $\mathbf{c}$
    #
    # 惯量（均质椭球体，半轴 $a, b, c = h/2$，COM = $\mathbf{c}$）：

    #   $I_{xx} = \frac{m}{5}(b^2 + c^2)$

    #   $I_{yy} = \frac{m}{5}(a^2 + c^2)$

    #   $I_{zz} = \frac{m}{5}(a^2 + b^2)$

    # 如有可复用的惯量计算方法则直接调用，否则待后续 coding 实现。

    # --- TODO:算法之四：sphere（仅保留接口，这个很少用）
    # 输入：半径 $r$, 质量 $m$
    #
    # 原点在球体底极点，球沿 +y 延伸。
    #
    # 几何中心：$\mathbf{c} = (0,\; r,\; 0)$
    #
    # URDF 输出：<sphere radius="r"/>，origin = $\mathbf{c}$
    #
    # 惯量（均质球体，COM = $\mathbf{c}$）：

    #   $I_{xx} = I_{yy} = I_{zz} = \frac{2m}{5} r^2$

    # 如有可复用的惯量计算方法则直接调用，否则待后续 coding 实现。


class ComPalmBuilder(PalmBuilder):
    """多基础几何形状复合而成的手掌构建类。
    """

    def __init__(self, cfg: ComPalmBuilderCfg):
        super().__init__(cfg)

    def build(self) -> PalmCfg:
        """构建手掌配置。

        Returns:
            PalmCfg: 构建完成的手掌配置。
        """