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

    origin: Vector6 | dict[str, Vector3] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """joint frame 相对于父 link 坐标系的位姿，包含位置和平移两部分。

    需要注意的是，关节级构建器并不涉对该字段的处理，而是手指级/手掌级构建器中的算法处理该字段。
    """

    mesh_offset: float | Vector3 | Vector6 | dict[str, Vector3] = 0
    """mesh的偏移参数，相对于其 joint frame，类型可以是单个float（均匀缩放），Vector3（xyz轴缩放）或者 Vector6（包含位置和姿态的全位姿偏移）。

    这里的偏移并不是指 mesh frame 相对于 joint frame 的固定变换，而是一个方便配置的参数，构建算法会根据这个参数和类型参数计算出最终的 mesh frame 位姿。
    由于是自定义 mesh，即使 mesh frame 为0,视觉上也不代表它的底部就和 joint frame 的 z-x 平面重合。mesh_offset 设为0,语义上即代表我们期盼 mesh 底部应和 joint frame 的 z-x 平面重合。
    剩余的就交予构建算法去解决。
    
    渐进式精度设计，支持三种精度的输入，内部统一解析为 list[Vector6]：
    - list[float]   : 仅沿手指伸长方向(y轴)的偏移，最常用。默认情况下
    - list[Vector3]  : xyz 位置偏移
    - list[Vector6]  : 完整 6D 位姿 (x, y, z, roll, pitch, yaw)
    """

    _mesh_offset_6d: Vector6 = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """内部解析后的 mesh 偏移，统一为 Vector6 形式，方便构建算法使用。"""

    is_customized: bool = True
    """mesh是否自定义，即非URDF默认的box/cylinder/sphere。"""

    def __post_init__(self):
        # 解析 _mesh_offset_6d
        raise NotImplementedError()


class CustomTipBuilderCfg(CustomJointBuilderCfg):
    r"""指尖专用的自定义关节构建器配置类。

    该声明式配置类包含的字段为构建类算法所需，而非单纯照搬 `JointCfg` 的所有字段

    核心思想是 “算法里人易理解和显式控制的参数” 映射到 `JointCfg` 的字段上
    """

    scale: float | Vector3 = 1
    """缩放参数，默认为1。对应 urdf 中的 mesh scale 字段。
    
    为float时表示沿 xyz 轴的均匀缩放；为 Vector3 时表示沿 xyz 轴的各自比例缩放。
    """

    tip_type: str = "round"
    """指尖类型，用于区分不同的指尖构建算法。比如 "leap_cube"、"round"、"wedge" 等。"""


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
        raise NotImplementedError()
    # NOTE:这里的算法具有相当的 “定制性” ，不同的指尖mesh,因为是从CAD等处导出来的，mesh origin位置和 joint frame 的关系可能都不一样，因此需要每个指尖mesh单独设计构建算法，来正确处理 `origin` 字段，并把 mesh 放到正确的位置。
    # 因此对于 mesh_offset，它并不是等同于 mesh frame 相对于 joint frame 的偏移，而是方便配置声明。

    # --- TODO:算法之一 ---：`AnyMani/source/anymani/anymani/assets/custom/tips/finger_tip_soft.stl` 定制
    # 输入：tip_type: leap_cube / ...，配置层 scale_cfg（默认 $(1,1,1)$），内部实际缩放 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$（mm→m），偏移 $d\in\mathbb{R}$
    # 输出：使 leap-cube 指尖的底层平面与 joint frame 的 x-z 平面平行；当 $d=0$ 时，底层中心与 joint 原点重合；当 $d\neq 0$ 时，沿 y 轴方向额外平移 $d$，但底层仍保持与 x-z 平面平行；
    #       同时保证 y 轴从底部指向指尖，x 轴映射到 +z，剩下的 z 轴由右手系确定
    # 处理流程：
    # - 第一步：读取 `finger_tip_soft.stl`；该 STL 以 mm 级网格坐标存储，因此 builder 内部先乘 0.001 做 mm→m 转换
    # - 第二步：确定语义锚点 $p^*$ 为最底层的中心；对当前 leap-cube，原始网格里底层中心约为 $(9.485707,0,-16.5)$（STL 单位）
    # - 第三步：构造旋转矩阵 $R=R_y(-\pi/2)$，保持 y 轴为底→tip，同时把原始 x 轴映射到 joint 的 +z 方向
    # - 第四步：构造缩放矩阵 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$；若配置层 scale_cfg=(1,1,1)，则 $S=0.001I$，这就是当前 leap-cube 的 canonical configuration
    # - 第五步：求平移 $t$，使锚点在变换后落到 joint 原点；若没有额外偏移，则 $t=-R\,S\,p^*$，若有额外偏移，则 $t=d\,\mathbf{e}_y-R\,S\,p^*$
    # - 第六步：将同一套 $R,t$ 同时写入 visual / collision；底部的 6 个圆孔作为 STL 几何的一部分保留，不把它额外解释成“必须 2mm”的工艺孔
    # 公式：
    # $$p_{joint}=R\,S\,p_{mesh}+t$$
    # $$t=d\,\mathbf{e}_y-R\,S\,p^*$$
    # - $R$: 旋转矩阵，对应 `rpy`
    # - $t$: 平移向量，对应 `xyz`
    # - $p^*$: leap-cube mesh 局部坐标里的语义锚点，不是几何中心本身
    # preset: 当配置层 scale_cfg=(1,1,1),\ d=0 时（mm→m canonical configuration）
    # - $\prescript{m}{}{x}=-1.6499999959cm,\ \prescript{m}{}{y}=0cm,\ \prescript{m}{}{z}=-0.9485706925cm$
    # - $\prescript{m}{}{roll}=0,\ \prescript{m}{}{pitch}=-\pi/2,\ \prescript{m}{}{yaw}=0$
    # 当配置层 scale_cfg\neq(1,1,1) 时
    # - 若为 uniform scale $s_u$，则实际缩放是 $S=0.001\,s_u I$，`R` 不变，`t` 按同一公式重算
    # - 若为 non-uniform scale $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$，则按 $t(S)=d\,\mathbf{e}_y-R\,S\,p^*$ 重算平移，`rpy` 不变
    # 当 $d\neq 0$ 时
    # - 在 tip 的 $y$ 轴方向额外加偏移 $d$，即 $p_{tip}$ 在 joint frame 下整体平移 $d$
    # - 先按 scale 算出基准位姿，再叠加 $d$ 的轴向平移；视觉和碰撞几何保持同一套位姿

    # --- TODO:算法之二 ---：`AnyMani/source/anymani/anymani/assets/custom/tips/wedge_finger_tip_soft.stl` 定制
    # 输入：tip_type: wedge / ...，配置层 scale_cfg（默认 $(1,1,1)$），内部实际缩放 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$（mm→m），偏移 $d\in\mathbb{R}$
    # 输出：使 wedge tip 的底面与 joint frame 的 x-z 平面平行；当 $d=0$ 时，底部中心与 joint 原点重合；当 $d\neq 0$ 时，沿 y 轴方向额外平移 $d$，但底面仍保持与 x-z 平面平行；
    #       同时保证 y 轴从底部指向指尖，斜面朝向约定为 +z，x 轴由右手系确定
    # 处理流程：
    # - 第一步：读取 wedge STL；注意该 STL 以 mm 级网格坐标存储，因此 builder 内部先乘 0.001 做 mm→m 转换
    # - 第二步：确定语义锚点 $p^*$ 为平底面的中心；对当前 wedge，原始网格里平底中心约为 $(9.5,0,-16.5)$（STL 单位）
    # - 第三步：构造旋转矩阵 $R=R_y(-\pi/2)$，把原始 +x 方向的斜坡朝向映射到 joint 的 +z 方向，同时保持 y 轴为底→tip
    # - 第四步：构造缩放矩阵 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$；若配置层 scale=(1,1,1)，则 $S=0.001I$，这就是当前 wedge 的 canonical configuration
    # - 第五步：求平移 $t$，使锚点在变换后落到 joint 原点；若没有额外偏移，则 $t=-R\,S\,p^*$，若有额外偏移，则 $t=d\,\mathbf{e}_y-R\,S\,p^*$
    # - 第六步：将同一套 $R,t$ 写入 visual / collision；孔洞默认保留 STL 几何，不把它额外解释成“必须 2mm”的工艺孔
    # 公式：
    # $$p_{joint}=R\,S\,p_{mesh}+t$$
    # $$t=d\,\mathbf{e}_y-R\,S\,p^*$$
    # - $R$: 旋转矩阵，对应 `rpy`
    # - $t$: 平移向量，对应 `xyz`
    # - $p^*$: wedge mesh 局部坐标里的语义锚点，不是几何中心本身
    # preset: 当配置层 scale_cfg=(1,1,1),\ d=0 时（mm→m canonical configuration）
    # - $\prescript{m}{}{x}=-1.65cm,\ \prescript{m}{}{y}=0cm,\ \prescript{m}{}{z}=-0.95cm$
    # - $\prescript{m}{}{roll}=0,\ \prescript{m}{}{pitch}=-\pi/2,\ \prescript{m}{}{yaw}=0$
    # 当配置层 scale_cfg\neq(1,1,1) 时
    # - 若为 uniform scale $s_u$，则实际缩放是 $S=0.001\,s_u I$，`R` 不变，`t` 按同一公式重算
    # - 若为 non-uniform scale $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$，则按 $t(S)=d\,\mathbf{e}_y-R\,S\,p^*$ 重算平移，`rpy` 不变
    # 当 $d\neq 0$ 时
    # - 在 tip 的 $y$ 轴方向额外加偏移 $d$，即 $p_{tip}$ 在 joint frame 下整体平移 $d$
    # - 先按 scale 算出基准位姿，再叠加 $d$ 的轴向平移；视觉和碰撞几何保持同一套位姿

    # --- TODO:算法之三 ---：`AnyMani/source/anymani/anymani/assets/custom/tips/round_finger_tip_soft.stl` 定制
    # 这是刚才 `test_round.urdf` 对应的具体展开版；用于把 round tip 接到 URDF / Isaac 的 m 制世界，并保持底部中心与 joint 原点重合。
    # 输入：tip_type: round / ...，配置层 scale_cfg（默认 $(1,1,1)$），内部实际缩放 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$，偏移 $d\in\mathbb{R}$
    # 输出：使 round tip 的底面与 joint frame 的 x-z 平面平行；当 $d=0$ 时，底部中心与 joint 原点重合；当 $d\neq 0$ 时，沿 y 轴方向额外平移 $d$，但底面仍保持与 x-z 平面平行；
    #       同时保证 y 轴从底部指向指尖，x/z 侧向按右手系确定
    # 处理流程：
    # - 第一步：读取 round_finger_tip_soft.stl；该 STL 仍按网格单位存储，builder 内部通过固定 scale 把它放到 URDF/Isaac 的 m 制世界
    # - 第二步：确定语义锚点 $p^*$ 为平底面的中心；对当前 round tip，原始网格里平底中心约为 $(9.509864,0,-16.491319)$（STL 单位）
    # - 第三步：构造旋转矩阵 $R=R_y(-\pi/2)$，让底→tip 的方向落到 joint 的 +y 轴，同时保留 x/z 的右手系约定
    # - 第四步：构造缩放矩阵 $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$；若配置层 scale_cfg=(1,1,1)，则 $S=0.001I$，这就是当前 round tip 的 canonical configuration
    # - 第五步：求平移 $t$，使锚点在变换后落到 joint 原点；若没有额外偏移，则 $t=-R\,S\,p^*$，若有额外偏移，则 $t=d\,\mathbf{e}_y-R\,S\,p^*$
    # - 第六步：将同一套 $R,t$ 写入 visual / collision；几何本体保持 STL 原始形状，scale 只承担单位与尺寸标定
    # 公式：
    # $$p_{joint}=R\,S\,p_{mesh}+t$$
    # $$t=d\,\mathbf{e}_y-R\,S\,p^*$$
    # - $R$: 旋转矩阵，对应 `rpy`
    # - $t$: 平移向量，对应 `xyz`
    # - $p^*$: round mesh 局部坐标里的语义锚点，不是几何中心本身
    # preset: 当配置层 scale_cfg=(1,1,1),\ d=0 时（即 $S=0.001I$）
    # - $\prescript{m}{}{x}\approx-0.01649132\,m,\ \prescript{m}{}{y}=0,\ \prescript{m}{}{z}\approx-0.00950986\,m$
    # - $\prescript{m}{}{roll}=0,\ \prescript{m}{}{pitch}=-\pi/2,\ \prescript{m}{}{yaw}=0$
    # 当配置层 scale_cfg\neq(1,1,1) 时
    # - 若为 uniform scale $s_u$，则实际缩放是 $S=0.001\,s_u I$，`R` 不变，`t` 按同一公式重算
    # - 若为 non-uniform scale $S=0.001\,\mathrm{diag}(s_x,s_y,s_z)$，则按 $t(S)=d\,\mathbf{e}_y-R\,S\,p^*$ 重算平移，`rpy` 不变
    # 当 $d\neq 0$ 时
    # - 在 tip 的 $y$ 轴方向额外加偏移 $d$，即 $p_{tip}$ 在 joint frame 下整体平移 $d$
    # - 先按 scale 算出基准位姿，再叠加 $d$ 的轴向平移；视觉和碰撞几何保持同一套位姿