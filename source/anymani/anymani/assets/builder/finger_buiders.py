r"""TODO:自定义手指构建器配置类 `FingerBuilderCfg` 和运行时类 `FingerBuilderCfg`

补充：以真实人手为例，此处我分为拇指型和非拇指型，用于和机器手做对照。以下注释内容在 agent（你）cdoing 时最好放在相应地方做注释背景补充，而不是完全除去
- 非拇指型：即 index, middle, ring, little 这四个手指，具有相似的结构，尺寸有微小的差异，以下为从离手掌从近到远开始介绍
    - MCP关节（Metacarpophalangeal Joint）：连接掌骨（Metacarpal）与近节指骨（Proximal Phalanx），有2自由度，1是屈曲/伸展（Flexion/Extension），另一个是外展/内收（Abduction/Adduction）。
    > 严格来说MCP还有极其微小的轴向旋转（Axial Rotation），但这几乎是被动的、非主动控制的，通常不计入功能性自由度。所以工程上一般记为 2 DOF
    - PIP关节（Proximal Interphalangeal Joint）：连接近节指骨与中节指骨（Middle Phalanx），只有屈曲/伸展这1个自由度,结构上是典型的铰链关节（Hinge Joint），侧副韧带非常强，几乎完全限制了侧向运动
    - DIP关节（Distal Interphalangeal Joint）：连接中节指骨与远节指骨（Distal Phalanx），同样只有屈曲/伸展 1 DOF，与PIP存在肌腱耦合，欠驱动特性明显。DIP可以单独主动弯曲（FDP发力），但幅度和力量有限
- 拇指型：即 thumb 这个手指，结构上与其他四个手指有较大差异，是人手中最特殊、最复杂的手指，其灵活性是人类能够精细操作的核心，以下为从离手掌从近到远开始介绍
    - CMC关节（Carpometacarpal Joint）：腕掌关节，连接腕骨（Carpal）与掌骨（Metacarpal），结构上是鞍状关节（Saddle Joint），有 2 DOF
        - 屈曲/伸展（Flexion/Extension）：拇指在手掌平面内的摆动。拇指的 CMC 关节在胚胎发育过程中旋转了约 90°，所以和非拇指型的运动平面不一样
        - 内收/外展（Adduction/Abduction）：拇指离开/靠近手掌平面
        - 这两个自由度组合，产生了拇指最重要的功能动作：对掌（Opposition）：拇指指尖转向其他四指，是人类抓握的核心能力
    > 鞍状关节的两个轴并非完全正交，加上软组织约束，实际运动轨迹是一段圆锥形曲面，这也是为什么拇指运动难以用简单铰链模拟
    - MCP关节：屈曲/伸展 1个自由度，有少量内收/外展，但远比其他四指MCP小，通常不计为独立功能自由度
    - IP关节（Interphalangeal Joint）：与其他四指的DIP类似，纯铰链关节，但拇指IP没有欠驱动耦合问题，1个自由度：屈曲/伸展
    
目前对手指的拆分与否与如何拆分有着犹豫，主要是尺寸外观的影响。

既有像 allegro/leaphand 这样的全驱中型手指（其中又细分thumb和其他）
- allegro手指：分非拇指型和非拇指型两种。这里以 `AnyMani/source/anymani/anymani/assets/hands/allegro_hand/allegro_hand_right.urdf` 的真实坐标系情况做介绍，可结合`AnyMani/source/anymani/anymani/assets/allegro.png` 来理解
    - 非拇指型：index, middle, ring 这三个手指，有14~15cm长，2cm左右高，2.7cm左右宽，比真实对应人手指尺寸大一级。从离掌心距离由近及远介绍各个关节（以index为例子）
        - 0号关节：1 DOF，绕y轴正方向旋转。大约1.8cm*2cm*2.7cm
        - 1号关节：1 DOF，绕x轴正方向旋转。大约5.4cm*2cm*2.7cm
        > 0-1号关节实质上构成了人手的 MCP 关节，但不完全一样，1号关节是屈曲/伸展，而0号关节不是外展/内收。但0号关节绕y轴旋转，在手指伸长方向上引入一个侧向偏转自由度——这并非严格意义上的外展/内收，而是对整根手指施加一个绕长轴的侧摆
        - 2号关节：1 DOF，绕x轴正方向旋转，对应人手的 PIP 关节。大约3.8cm*2cm*2.7cm
        - 3号关节：1 DOF，绕x轴正方向旋转，对应人手的 DIP 关节。包含一个小块。大约2.2cm*2cm*2.7cm，但底部和2号关节mesh有一定重合，自然伸展状态下长度重合了0.6cm
        - 虚拟关节joint_3.0_tip：Fixed，构成了指尖。形状类似圆柱 + 半球，球的最大截面与圆柱顶面重合。圆柱大约长1cm,半径约1.2cm
    - 拇指型：thumb，有17cm左右长，但不是匀称的手指
        - 12号关节：1 DOF，绕x轴负方向旋转。这里不是base/env坐标系，而是 thumb joint frame。非拇指那里的介绍同理。该关节/子连杆是一个较大的块体，大部分嵌入了手掌内。大约4.5cm*3.5cm*3.4cm
        - 13号关节：1 DOF，绕z轴正方向旋转。大约1.7cm*1.9cm*2.7cm。底部和12号关节mesh有一定间隙，约0.2cm，且未处于mesh方形中部
        > 12-13号关节实质上构成了人手的 CMC 关节，但仍有差异。12号类似内收/外展，而13号不是屈曲/伸展，而是绕z轴的旋转，在手指伸长方向上引入一个侧向偏转自由度
        - 14号关节：1 DOF，绕y轴正方向旋转。对应人手拇指的MCP关节。大约4.3cm*1.9cm*2.7cm
        - 15号关节：1 DOF，绕y轴正方向旋转。对应人手拇指的IP关节。大约4.0cm*1.9cm*2.7cm。底部和14号关节mesh有一定重合，约0.9cm
        - 虚拟关节joint_15.0_tip：Fixed，构成了指尖。形状类似圆柱 + 半球，球的最大截面与圆柱顶面重合。圆柱大约长1cm,半径约1.2cm

- leap手指：亦分非拇指型和非拇指型两种。这里以 `AnyMani/source/anymani/anymani/assets/hands/leap_hand/leap_hand_right.urdf` 的真实坐标系情况做介绍，可结合`AnyMani/source/anymani/anymani/assets/leap.png` 来理解
    - 非拇指型：index, middle, ring 这三个手指，有14~15cm长，3.4cm左右高，3.5cm左右宽，比真实对应人手指尺寸大一级。从离掌心距离由近及远介绍各个关节（以index为例子）
    > 需注意的是，leap的urdf分为2版。官方原版所采用的collision mesh和visual mesh基本一致，非常精细。而这里所采用的是简化过后的urdf,把joint/child link mesh都简化了，但也是由多个基础形状如box复合而成的，比allegro要复杂和难以描述的多，以下讲述的是个大概轮廓的尺寸（单个box近似），而不是真实的mesh
        - 1号关节：1 DOF，绕z轴负方向旋转。约3.2cm*3.5cm*3.4cm。这里没包括0号关节mesh后侧的1号关节msh（在手掌侧），它负责连接了 mcp_joint 和 pip link
        - 0号关节：1 DOF，绕z轴负方向旋转。约3.3cm*3.5cm*3.4cm。离1号关节mesh有一空隙（在手心侧），约1cm
        > 0-1号关节实质上构成了人手的 MCP 关节，1号关节是屈曲/伸展，0号关节是外展/内收。这是和 allegro 在手指设计上的一大区别
        - 2号关节：1 DOF，绕z轴负方向旋转。对应人手的 PIP 关节。约4cm*3.5cm*3.4cm。底部和0号关节mesh有一定重合，约1.4cm
        - 3号关节：1 DOF，绕z轴负方向旋转。对应人手的 DIP 关节。约5cm*3.5cm*3.4cm。底部和2号关节mesh有一定重合，约2.0cm
        > 3号关节这里以后没再设虚拟固定关节来构成指尖了。而是将连杆和指尖复合构成了一个整体mesh。指尖部分采用 leaphand 官方设计的 `white_tip.obj`
    - 拇指型：thumb，约15.6cm长，4~5cm宽，3~4cm高，比真实拇指大很多。这里也是简化后（比官方urdf）的，但每个 joint 仍很复杂，由多个基础形状复合而成。以下讲述的是个大概轮廓的尺寸（单个box近似），而不是真实的mesh
        - 12号关节：1 DOF，绕z轴负方向旋转。约2.6cm*3.8cm*3.5cm。该关节/子连杆是一个较大的块体，基本嵌入了手掌内
        - 13号关节：1 DOF，绕z轴负方向旋转。约2.3cm*1.8cm*3.4cm。12号类似内收/外展，而13号不是屈曲/伸展，而是绕z轴的旋转，在手指伸长方向上引入一个侧向偏转自由度。
        > 0-1号关节实质上构成了人手的 CMC 关节，但仍有差异。12号类似内收/外展，而13号不是屈曲/伸展，而是绕z轴的旋转，在手指伸长方向上引入一个侧向偏转自由度。从这个角度来说，leap 和 allegro 的thumb设计是一致的，除了几何mesh有差异
        - 14号关节：1 DOF，绕z轴负方向旋转。对应拇指的MCP关节。约9.1cm*2.6cm*3.4cm。底部和13号关节mesh有一定重合，约1.5cm
        - 15号关节：1 DOF，绕z轴负方向旋转。对应拇指的IP关节。约7.0cm*2.6cm*3.4cm。底部和14号关节mesh有一定重合，约1.5cm
        > 15号关节这里以后没再设虚拟固定关节来构成指尖了。而是将连杆和指尖复合构成了一个整体mesh。指尖部分采用 leaphand 官方设计的 `white_tip.obj`

这里再对 leap 和 allegro 再做对比小结。两者有两大主要差异
    - mesh差异：allegro 的整体mesh很规整，每个joint基本都采用单一的基础几何体（如box）来构成；而 leap 的整体mesh比较复杂。简化后版本的每个joint基本都由多个基础几何体复合而成
    - 非拇指型手指差异：都是离手掌近的两个关节构成MCP。allegro 0号提供绕手指伸长的方向的自旋运动，而1号提供屈曲/伸展；leap 1号提供屈曲/伸展（leap这里命名规则1反倒比0更靠近手掌），0号提供外展/内收，相对更接近真是人手MCP
拇指型手指则设计风格一致，除了mesh差异
    
除此之外也有像 shadow 这样的全驱小型手指（也细分为thumb和其他），有10cm左右，和真实的人类手指更为接近，具有 “球型关节”。
以及 schunk 这样的欠驱小型手指（也细分为thumb和其他），dip关节模仿了人类手指的dip关节的欠驱特性（由pip关节驱动，但dip本身也能主动操控）。
甚至还有 dclaw_gripper, bhand 这样的“夹爪手”（大于2根），通常是圆形底盘，但手指分布相对均匀或对称，和人手的形态外貌不太一致。

但本次科研idea-跨手型泛化的手内操作，仍先以 leap/allegro family 为主要切入点，后期再考虑球型关节、欠驱特性、夹爪手等泛化研究（形态差异很大的手的泛化是否值得研究也是一个值得商榷的课题）
"""
from __future__ import annotations

from assets.asset_builders import FingerBuilderCfg, FingerBuilder
from assets.asset_base import FingerCfg
from assets.asset_schema_core import Vector6, Vector3, Vector2

from dataclasses import dataclass, field
from typing import Any, Literal


# --- 手指的声明式配置类 --- #
# TODO: 经过大量手指的调研，决定划分为含球类关节手指、不含球类关节手指两大 FingerBuilderCfg 子类。
# 对独不含球类关节手指，thumb/non-thumb，全驱/欠驱，夹爪/类人，很多时候只是 mesh 怎么摆、旋转轴怎么放、关节链长度怎么配，算法差异不是很大，不值得继续细分子类。可以preset/方法工厂来实现
# 但对含球类关节手指，如 shadow/schunk，joint mesh 的构造方式会明显变化，不是简单 box/cylinder 的局部变体，关节外形、轴心语义、碰撞几何组织方式都会更特殊
# 目前对 RegularFingerBuilderCfg，采用多个并列式 Cfg → 一个 Builder，使得每个 Cfg 的字段集合精确的、自洽，builder 也易于处理
@dataclass
class RegularFingerBuilderCfg(FingerBuilderCfg):
    r"""常规/非球类关节的手指构建器配置类。

    包括 leap/allegro/夹爪手指等大多数常见手指类型，包含全驱/欠驱，thumb/non-thumb等差异，但不包含球类关节的手指。

    NOTE:该类的 mesh_shape 和 mesh_offsets 的不同 preset 理论上是可以涵盖 leap/allegro thumb/non-thumb 的一切情况的。但为了让配置更显式、易于理解和使用，
    我仍细分为 AllegroFingerBuilderCfg、LeapFingerBuilderCfg、RegularThumbBuilderCfg 等多个子类，为其额外配置一些字段。
    这些字段具有针对性，属于构建对应手指类型最常见的思路参数（所谓 Canonical Configuration），但经过解析仍能统一到 mesh_shape 和 mesh_offsets 里，因此 builder 只需要处理 mesh_shape 和 mesh_offsets 就行了，职责边界清晰。
    同时子类继承了 mesh_shape 和 mesh_offsets 的字段，在非常规构建算法需要更细粒度控制 mesh 形状时也能直接使用，例如4个 joint mesh 里同时有 box/cylinder，而不需要额外字段，保持了足够的灵活性。
    """

    class_type: type["RegularFingerBuilder"] | None = None
    """关联的自定义手指构建器类。"""

    num_joints: int = 4
    """关节数。allegro的非拇指型手指有4个关节。这里并不包含最后的指尖虚拟关节。"""

    mesh_shape: list[dict[str, Any]] = field(default_factory=list)
    """每个关节mesh的几何形状。list数要和 num_joints 一致。

        这里包含 box,cylinder 两种：
        - "box":(length, width, height)，其中length是沿着手指伸长方向的尺寸，width是侧向尺寸，height是垂直于手指伸长方向的尺寸
        - "cylinder":(length, radius)，其中length是沿着手指伸长方向的尺寸，radius是圆柱的半径，沿着手指伸长方向
    """

    mesh_offsets: list | list[Vector3] | list[Vector6] = field(default_factory=list)
    """每个关节mesh相对于joint frame的偏移。
    
    这里的偏移主要指 box/cylinder/sphere mesh 等相对于 joint frame 的偏移，而且和 PrimJointBuilder 中的行为一致。

        渐进式精度设计，支持三种精度的输入，内部统一解析为 list[Vector6]：
        - list[float]   : 仅沿手指伸长方向(y轴)的偏移，最常用。默认情况下
        - list[Vector3]  : xyz 位置偏移
        - list[Vector6]  : 完整 6D 位姿 (x, y, z, roll, pitch, yaw)

        示例::
    
            mesh_offsets = [0.0, -0.6, 0.0, 0.0]          # 仅 y 轴
            mesh_offsets = [(0,0,0), (0,-0.6,0.1), ...]    # xyz
            mesh_offsets = [(0,0,0,0,0,0), ...]            # 完整 6D
    """

    _mesh_offsets_6d: list[Vector6] = field(init=False, default_factory=list)
    """mesh_offsets 解析后的结果，统一为 list[Vector6] 格式，供构建算法使用。"""

    tip: dict[str, Any] = field(default_factory=dict)
    """指尖参数字典。包括指尖类型（基础几何复合或自定义），以及相应的参数。
    
        基础几何复合有：(和 :class:`PrimJointBuilderCfg` 中的指尖类型方法一致)
        - "cs" (cylinder + sphere): (radius, height)
        - "bs" (box + sphere): (radius, height, width)

        自定义保留接口，待后续实现
    """

    tip_offset: float | Vector3 | Vector6 = None
    """指尖相对于最后一个关节mesh的偏移。

        同样支持三种精度的输入，内部统一解析为 Vector6：
        - float          : 仅沿手指伸长方向(y轴)的偏移，最常用
        - Vector3        : xyz 位置偏移
        - Vector6        : 完整 6D 位姿 (x, y, z, roll, pitch, yaw)
    """

    _tip_offset_6d: Vector6 = field(init=False, default_factory=lambda: Vector6(0, 0, 0, 0, 0, 0))
    """tip_offset 解析后的结果，统一为 Vector6 格式，供构建算法使用。"""

    axes: list[Vector3] = field(default_factory=list)
    """每个关节的旋转轴。list数要和 num_joints 一致。3个维度分别代表 $x, y, z$ 分量"""

    def __post_init__(self):
        super().__post_init__()
        # TODO:解析 mesh_offsets 和 tip_offset，统一转换为 _mesh_offsets_6d 和 _tip_offset_6d，供构建算法使用；
        # 校验 num_joints 和 部分字段长度是否匹配；
        # axes 归一化；

# 这里以手指伸长的方向为y轴，侧向为x轴，手心朝外为z轴，分别对应长、宽、高，和 `AnyMani/source/anymani/anymani/assets/平面示意.png` 一致，在关节级构建器里也是怎么约定的坐标系


@dataclass
class AllegroFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""Allegro 非拇指类型手指的几何构建器配置类。

    TODO: 封装 :class:`AllegroFingerBuilder` 构建一根 Allegro 非拇指手指所需的全部参数。多精度输入字段统一解析为内部标准格式，:class:`AllegroFingerBuilder` 直接读取解析结果，职责边界清晰。
    """

    width: float = None
    """关节mesh的宽度。该字段对除 tip 外的所有 joint mesh 都设置了相同的宽度"""

    height: float = None
    """关节mesh的高度。该字段对除 tip 外的所有 joint mesh 都设置了相同的高度"""

    radius: float = None
    """关节mesh的半径。该字段对除 tip 外的所有 joint mesh 都设置了相同的半径"""

    length: list[float] = field(default_factory=lambda: [1.8, 5.4, 3.8, 2.2])
    """每个关节mesh沿手指伸长方向的长度。allegro的非拇指型手指的四个关节mesh沿手指伸长方向的长度分别约为1.8cm、5.4cm、3.8cm、2.2cm。"""


    def __post_init__(self):
        # radius（对应 cylinder 关节）和 height-width（对应 box 关节）只能择其一，若两者都有，则优先使用 height-width。对应 PrimJointBuilder 中的 box/cyliner 关节算法
        # 把 radius/height-width 和 length 解析到 mesh_shape，这个解析在 super().__post_init__() 之前
        super().__post_init__()
        pass


@dataclass
class LeapFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""Leap 非拇指类型手指的几何构建器配置类。
    """

    width: float = None

    height: float = None

    radius: float = None

    length: list[float] = field(default_factory=lambda: [1.8, 5.4, 3.8, 2.2])

    fixed_part: float = None 
    """固定部分长度。

    leap 手指的第一个关节轴到手根底部不是同平面，而是有一块较短长度的固定mesh 连接着。对此虚设一虚拟关节 mesh来代表此部分，长度自定义，宽度、高度和手指其他关节mesh一致。
    该虚拟关节不参与运动学计算，仅作为构建时的一个参考点，来帮助正确组织后续关节mesh的构建和连接。
    """

@dataclass
class RegularThumbBuilderCfg(RegularFingerBuilderCfg):
    r"""基础几何构建器配置类。
    """

    cmc1_width: float = None
    """拇指 CMC 关节的第一个关节（对应 allegro/leap 的 12 号关节）的宽度"""

    cmc1_height: float = None
    """拇指 CMC 关节的第一个关节（对应 allegro/leap 的 12 号关节）的高度"""

    width: float = None
    """拇指其他关节mesh的宽度。该字段对除 tip 外的所有非CMC1 joint mesh 都设置了相同的宽度"""

    height: float = None
    """拇指其他关节mesh的高度。该字段对除 tip 外的所有非CMC1 joint mesh 都设置了相同的高度"""

    lengths: list[float] = field(default_factory=lambda: [3.2, 3.3, 4.0, 5.0])

    cmc1_offset: float | Vector2 | Vector3 = Vector2(0, 0)
    r"""拇指 CMC 关节的第一个关节（对应 allegro/leap 的 12 号关节）mesh 相对于本 joint frame 的位置偏移 (y), (y, z)，或 (x, y, z)。
    
    一般使用 (y, z)，默认 x 为0，因为对于 CMC1 一般是绕 x 轴旋转，与具体的 x 点值关系不大。就算需要调整 CMC1 mesh 在手中位置，也是先定下 joint frame。
    
    这里的 joint mesh frame 和 joint frame 的关系约定和其他类型不一样。这里的惯例和 urdf 中保持一致，即偏移为0时，两者重合。

    否则 $\prescript{m}{}{x}=d_{x}, \prescript{m}{}{y}=d_{y}, \prescript{m}{}{z}=d_{z}$
    """

    non_cmc1_offset: list | list[Vector3] | list[Vector6] = field(default_factory=list)
    """拇指除 CMC1 的其他 joint mesh 相对于各自 joint frame 的偏移。
    
        这里的偏移主要指 box/cylinder/sphere mesh 等相对于 joint frame 的偏移，而且和 PrimJointBuilder 中的行为一致。

        渐进式精度设计，支持三种精度的输入，内部统一解析为 list[Vector6]：
        - list[float]   : 仅沿手指伸长方向(y轴)的偏移，最常用。默认情况下
        - list[Vector3]  : xyz 位置偏移
        - list[Vector6]  : 完整 6D 位姿 (x, y, z, roll, pitch, yaw)

        示例::
    
            mesh_offsets = [0.0, -0.6, 0.0, 0.0]          # 仅 y 轴
            mesh_offsets = [(0,0,0), (0,-0.6,0.1), ...]    # xyz
            mesh_offsets = [(0,0,0,0,0,0), ...]            # 完整 6D
    """

    def __post_init__(self):
        # 解析参数后再调用
        super().__post_init__()
        pass



@dataclass
class SphericalFingerBuilderCfg(FingerBuilderCfg):
    r"""自定义手指构建器配置类。

    该声明式配置类包含的字段为构建类算法所需，而非单纯照搬 `FingerCfg` 的所有字段

    核心思想是 “算法里人易理解和显式控制的参数” 映射到 `FingerCfg` 的字段上
    """

    class_type: type["CustomThumbBuilder"] | None = None
    """关联的自定义手指构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = CustomThumbBuilder

    pass


# --- 手指的运行时构建类 --- #
# TODO：该文件开头详细介绍了不同手指的情况。它们各自的坐标系约定相异。在这里，我们按照 `AnyMani/source/anymani/anymani/assets/平面示意.png` 统一约定（Coding时读取该图片理解我的意图）
# 具体来说，手指伸长的方向为joint y轴，面向手心朝外的一侧为joint z轴，最后由右手法则确定侧向为joint x轴。无论thumb/non-thumb等最受此约定。由其带来的旋转轴语义性问题后续交由网络架构层考虑解决
# 手指构建算法的核心在与正确组织串联好每个关节 joint frame（child link 与 parent link之间的相对位姿）和 joint frame 与 joint mesh 的位置关系
# 不同的构建思路可对比图示来理解
class RegularFingerBuilder(FingerBuilder):
    r"""常规/非球类关节的手指构建器。

    包括 leap/allegro/夹爪手指等大多数常见手指类型，包含全驱/欠驱，thumb/non-thumb等差异，但不包含球类关节的手指。
    """

    cfg: RegularFingerBuilderCfg

    def __init__(self, cfg: RegularFingerBuilderCfg):
        self.cfg = cfg
        """IDEA：这里有一个想法就是
        """

        pass

    def build(self) -> FingerCfg:
        r"""根据 `RegularFingerBuilderCfg` 构建对应的常规/非球类关节的手指。

        这里的返回值类型暂时用 `Any` 占位，具体类型取决于配置类的构造细
        """
        pass

    # NOTE:以下算法的输入输出并不和构建器的输入输出完全一致，也不完整，而是更贴近于我个人的心流过程。以下算法目前不完全偏向工厂方法，而是根据不同 Cfg 来统一构建？但部分参数用分发逻辑可能更清晰
    # joint mesh frame 是相对于本 joint frame 的，joint frame 是child link相对于 parent link的。所谓手指构建算法主要围绕这两层关系展开
    # joint mesh frame 可复用 PrimJointBuilder 的构建逻辑，joint frame 则在这里处理
    # --- TODO:算法之一 ---：allegro 非拇指型手指构造
    # 输入：手指关节数 $N$（2~4)，joint mesh 长度 $l\in \mathbb{R}^N$，mesh 偏移 $d\in\mathbb{R}^{N}$，指尖类型及其参数
    # coding 前需读示意图 `AnyMani/source/anymani/anymani/assets/doc/Allegro-Non-Thumb.png`。算法核心在于不同 link 间 joint frame 的组织解算
    # > 关于数量，采用和 allegro 非拇指型手指一致对齐的路子。如果是 2, 则为 mcp joints。如果是 3, 则为 mcp joints + pip joint。如果是 4，则为 mcp joints + pip joint + dip joint
    # > 实际上除了 mcp joints 的第一个关节旋转轴为 $y$, 其余的都为 $x$，这些在文件开头说明过。因而后3个关节类型是完全同构的
    # > 因此，至少需要2个关节，如果只保留一个关节，即 mcp joints 的第一个，意义不大，它和物体的交互能力非常有限
    # > 指尖类型分 joint_builders_primitive.py 中的 PrimJointBuilderCfg 由基础形状 cylinder/box + sphere 复合而成的指尖，以及 joint_builders_custom.py 中的自定义指尖
    # > 第一版先实现 cylinder/box + sphere 复合指尖，自定义指尖留作接口，待整体实现、基础功能得以完善后补充
    # > 对 cylinder + sphere，需提供指尖参数 $r, h, d_{tip}$，对 box + sphere，需提供 $r, h, w, d_{tip}$
    # preset：根据前面的手指分析，allegro 的非拇指型手指官方 preset 为：$N=4$，$r=(1.8, 5.4, 3.8, 2.2)$cm，$d=(0, 0, -0.6, 0)$cm，指尖类型为 cylinder + sphere，指尖参数为 $r=1.2$cm, $h=1$cm, $d_{tip}=0$cm
    # 输出：
    # - 构建时调用配置好 JointBuilderCfg 的 JointBuilder 方法，来构建每个 JointCfg。这一块负责构建每个 joint mesh 相对于 joint frame 的关系等
    # - 并把它们按照正确的 kinematic chain 组织起来，最后输出 FingerCfg。这一块构建 joint frame 相对上一 joint frame,或者说 child link 相对 parent link frame 的关系等


    # --- TODO:算法之二 ---：leap 非拇指型手指构造
    # 输入：手指关节数 $N$(1~4)，joint mesh 长度 $l\in \mathbb{R}^N$，宽度 $w$，高度 $h$，或半径 $r$，mesh 偏移 $d\in\mathbb{R}^{N}$，指尖类型及其参数，固定部分长度 $l_f$
    # 输出：
    # - $y_0=l_f,\ z_0=0,\ x_0=0$
    # - $y_i=l_{i-1}+d_{i-1},\ z_i=0,\ x_i=0,\ 1\le i\le N-1$，$\prescript{m}{}{y}_i=l_i/2+d_i,\ \prescript{m}{}{z}_i=0,\ \prescript{m}{}{x}_i=0,\ 0\le i\le N-1$，这里可以复用 PrimJointBuilder 的 joint mesh 构造逻辑
    # - $y_{tip}=y_N=l_{N-1}+d_{N-1},\ z_{tip}=0,\ x_{tip}=0$，指尖 joint frame
    # - $\prescript{m}{}{y}_{tip}=d_{N}=d_{tip},\ \prescript{m}{}{z}_{tip}=0,\ \prescript{m}{}{x}_{tip}=0$，指尖 mesh frame
    # - N=4 时, axis0 为 x轴，axis1 为 z轴， axis2 为 x轴，axis3 为 x轴；N=1 时，axis0 为 x轴；N=2 时，axis0 为 x轴， axis1 允许为 x轴/z轴；
    # - N=3 时，axis0 为 x轴， axis1 为 z轴， axis2 为 x轴
    # preset：本算法对原始 leap 构造形状进行大大简化。$N=4,\ l=(3.9,1.5,3.6,2.0)cm,\ w=3.4cm,\ h=2.05cm,\ d=(0,0,0,0), l_f=1.3cm$。
    # 指尖类型为 leap_cube，自定义类型，scale为1,偏置为0。此处可调用 CustomJointBuilder 处理指尖mesh 的构造，同时本层处理指尖 joint frame 的构造


    # --- TODO:算法之三与算法之四 ---：allegro 和 leap 拇指型手指构造（preset区分）这里暂时用 box 作为 mesh 形状，先不考虑 cylinder
    # 输入：手指关节数 $N$（3~4），CMC1 宽度和高度 $w_{cmc1}, h_{cmc1}$，CMC1 mesh 偏移 $d_{cmc1,y}, d_{cmc1,z}$，其他手指宽度和高度 $w, h$，其他 joint mesh 偏移 $d\in\mathbb{R}^{N-1}$（编号从1开始，0号视作 $d_{CMC}$），长度 $l\in\mathbb{R}^N$，指尖类型及其参数
    # 输出：coding前对照图示示例 `AnyMani/source/anymani/anymani/assets/doc/Thumb.png`理解实现
    # - $\prescript{m}{}{y}_0=d_{cmc1,y}=d_{0y},\prescript{m}{}{z}_0=d_{cmc1,z}=d_{0z}, \prescript{m}{}{x}_0=0=d_{0x}$，0关节/CMC1旋转轴为x轴
    # - $y_1=d_{cmc1,y} + l_0/2,\ z_1=d_{cmc1,z} - (h_{cmc1}-h)/2,\ x_1=(w_{cmc1}-w)/2$，1关节/CMC2旋转轴为y轴
    # - $\prescript{m}{}{y}_i=l_i/2+d_i,\ \prescript{m}{}{z}_i=0,\ \prescript{m}{}{x}_i=0,\ 1\le i\le N-1$，这里除 CMC1 joint mesh的构造逻辑和 allegro 非拇指型手指其实是一致的，可以复用/调用 PrimJointBuilder 
    # - $y_i=l_{i-1}+d_{i-1,y},\ z_i=0,\ x_i=0,\ 2\le i\le N-1$，其他关节旋转轴为z轴
    # - $y_{tip}=y_N=l_{N-1}+d_{N-1},\ z_{tip}=0,\ x_{tip}=0$，指尖 joint frame
    # - $\prescript{m}{}{y}_{tip}=d_{N}=d_{tip},\ \prescript{m}{}{z}_{tip}=0,\ \prescript{m}{}{x}_{tip}=0$，指尖 mesh frame
    # preset：
    # - allegro：$N=4,\ w_{cmc1}=3.5cm,\ h_{cmc1}=3.4cm,\ d_{cm1,y}=0.9cm,\ d_{cm1,z}=1.45cm,\ w=1.9cm,\ h=2.7cm,\ d=(-0.2,0,-0.9)cm,\ l=(4.5,1.7,4.3,4.0)$，指尖类型为 cylinder + sphere, $d_{tip}=0,\ r_{tip}=1.2cm,\ h_{tip}=1cm$
    # - leap: $N=4,\ h_{cmc1}=2.67cm,\ w_{cmc1}=2.30cm,\ d_{cm1,z}=-0.33cm,\ d_{cm1,y}=0,\ w=2.3cm,\ h=3.47cm,\ d=(0,0,0),\ l=(2.8,1.7,4.7,2.3)$，指尖类型为 custom 的 leap_cube，scale默认为1,偏移为0