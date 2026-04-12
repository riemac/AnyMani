r"""TODO:掌部构建器：把掌级几何参数落为 `PalmCfg`。

本文件对应你在 `assets/doc/Single-Palm.jpg`、`allegro-palm.png`、
`leap-palm.png` 中画出的掌部建模约定。当前首轮实现只做 pre-made，
因此关注点不是“如何在运行时完美复刻真实 mesh”，而是先给出一套
跨 hand family 可复用、并且足够清晰的掌级中间表示（IR）。

核心约定有三条：

1. `SinglePalmBuilder` 负责“单一基础几何”的掌部。
   这条路径服务于跨 family 的参数化枚举，强调描述简单、便于采样。
2. `ComPalmBuilder` 负责“复合基础几何”的真实 preset。
   这条路径服务于 Allegro / LEAP 这类已知锚点，强调和真实碰撞体的
   空间组织保持一致。
3. palm frame 统一采用你图里的“底边中心原点”语义：
   - $x$：掌宽方向
   - $y$：朝指方向的掌长方向
   - $z$：掌厚方向

这样做的原因是：非拇指手指通常都从 palm 顶缘沿 $+y$ 生长，若 palm
与 finger 共用这一几何直觉，则后续 hand-level mount 组织会非常直接。

这里同样保留高密度注释，因为 palm 不只是一个大块碰撞体，它还是：

1. hand 的几何主躯干；
2. finger mounts 的参考系；
3. human-like / gripper-like 进一步分化时的结构分水岭。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..asset_base import PalmCfg
from ..asset_builders import PalmBuilder, PalmBuilderCfg
from ..asset_schema_core import CollisionGeometryCfg, InertialCfg, PoseCfg, VisualGeometryCfg


_DEFAULT_PALM_DENSITY = 700.0
"""掌部默认密度 $\\rho$ [kg/m^3]。

这里不是材料学上的精确设定，而是首轮纵向打通时的工程默认值。
它的目标只是让自动合成的 primitive palm 拥有正定且量级合理的惯量，
避免出现极小质量导致的数值不稳定。
"""


_SINGLE_PALM_BOX_PRESETS = {
    "allegro": {"width": 0.112, "length": 0.0944, "height": 0.042},
    "leap": {"width": 0.12, "length": 0.08, "height": 0.046},
}
"""单一 box palm 的参考 preset 尺寸 [m]。

这组数值对应你原始 TODO 里写的“box preset”，它们目前不是
`SinglePalmBuilderCfg` 的显式字段，而是保留为“参数化手掌的参考锚点”：

- `allegro`: $w=11.2\\text{cm}, l=9.44\\text{cm}, h=4.2\\text{cm}$
- `leap`: $w=12\\text{cm}, l=8\\text{cm}, h=4.6\\text{cm}$

之所以这次实现里没有把它们做成 `preset=` 字段，是因为 `SinglePalmBuilderCfg`
当前强调的是“连续参数化空间”，而不是离散 preset 选择。
但这些参考值本身是重要的建模上下文，所以应该明确保留下来，方便后续：

1. 做 box palm 的参数初始化；
2. 做 human-like fallback mount 的比例拟合；
3. 对照真实 hand family 的量级。
"""


def _box_inertia(width: float, length: float, height: float, mass: float) -> dict[str, float]:
    r"""计算均质长方体在质心处的惯量张量对角项。

    这里采用标准公式：

    $$
    I_{xx} = \frac{m}{12}(l^2 + h^2),\quad
    I_{yy} = \frac{m}{12}(w^2 + h^2),\quad
    I_{zz} = \frac{m}{12}(w^2 + l^2).
    $$
    """
    return {
        "ixx": mass * (length * length + height * height) / 12.0,
        "iyy": mass * (width * width + height * height) / 12.0,
        "izz": mass * (width * width + length * length) / 12.0,
    }


def _cylinder_inertia(radius: float, height: float, mass: float) -> dict[str, float]:
    r"""计算均质圆柱体在质心处的惯量张量对角项。

    当前掌部圆柱采用轴向沿 $z$ 的语义，因此使用：

    $$
    I_{xx} = I_{yy} = \frac{m}{12}(3r^2 + h^2),\quad
    I_{zz} = \frac{m}{2}r^2.
    $$
    """
    return {
        "ixx": mass * (3.0 * radius * radius + height * height) / 12.0,
        "iyy": mass * (3.0 * radius * radius + height * height) / 12.0,
        "izz": mass * radius * radius / 2.0,
    }


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    r"""计算均质球体在质心处的惯量张量对角项。

    $$
    I_{xx} = I_{yy} = I_{zz} = \frac{2mr^2}{5}.
    $$
    """
    moment = 2.0 * mass * radius * radius / 5.0
    return {"ixx": moment, "iyy": moment, "izz": moment}


def _estimate_mass(volume: float) -> float:
    r"""由体积 $V$ 和默认密度 $\\rho$ 估算质量。

    这是一个工程近似，而不是来源于真实材料的精确建模：

    $$
    m = \rho V.
    $$

    之所以这里要自动估质量，是因为参数化 palm 的主要关注点是几何，
    用户未必会在每次枚举时都手工指定质量。
    """
    return max(volume * _DEFAULT_PALM_DENSITY, 1e-5)


@dataclass
class SinglePalmBuilderCfg(PalmBuilderCfg):
    r"""单一基础几何掌部配置。

    该 cfg 对应“方案 C”的掌部设计帧：原点在掌底边中心，几何沿 $+y$
    生长，与 finger builder 中“从 joint frame 的 $x-z$ 平面向 $+y$ 长出”
    的新约定保持同构。
    """

    shape: Literal["box", "cylinder", "sphere", "ellipse"] = "box"
    """单一 palm 的基础几何类型。"""

    length: float | None = None
    """掌长，沿 ``+y`` 方向，仅 box / ellipse 使用。"""

    width: float | None = None
    """掌宽，沿 ``+x`` 方向，主要用于 box palm。"""

    height: float | None = None
    """掌厚，沿 ``+z`` 方向。所有形状都需要。"""

    radius: float | None = None
    """圆柱/球 palm 的半径。"""

    a: float | None = None
    """椭球在 ``+x`` 方向的半轴。"""

    b: float | None = None
    """椭球在 ``+y`` 方向的半轴。"""

    def __post_init__(self):
        super().__post_init__()
        if self.shape == "box":
            for field_name in ("width", "length", "height"):
                value = getattr(self, field_name)  # box 需要三条边都合法
                if value is None or float(value) <= 0.0:
                    raise ValueError(f"{field_name} must be positive for box palms")
        elif self.shape in {"cylinder", "sphere"}:
            if self.radius is None or float(self.radius) <= 0.0:
                raise ValueError(f"radius must be positive for {self.shape} palms")
            if self.height is None or float(self.height) <= 0.0:
                raise ValueError(f"height must be positive for {self.shape} palms")
        elif self.shape == "ellipse":
            if self.a is None or float(self.a) <= 0.0 or self.b is None or float(self.b) <= 0.0:
                raise ValueError("ellipse palms require positive a and b")
            if self.height is None or float(self.height) <= 0.0:
                raise ValueError("ellipse palms require positive height")
        else:
            raise ValueError(f"unsupported palm shape: {self.shape}")
        self.class_type = SinglePalmBuilder  # 单一 primitive palm 统一走 SinglePalmBuilder


@dataclass
class ComPalmBuilderCfg(PalmBuilderCfg):
    r"""复合基础几何掌部配置。

    这条路径不做连续参数化，而是直接锚定到真实 hand family 的碰撞体布置。
    当前支持：

    - `allegro`
    - `leap`
    """

    preset: Literal["leap", "allegro"] = "allegro"
    """复合 palm 的 hand family preset 名。"""

    def __post_init__(self):
        super().__post_init__()
        self.class_type = ComPalmBuilder  # 复合 palm preset 统一走 ComPalmBuilder


@dataclass
class CustomPalmBuilderCfg(PalmBuilderCfg):
    r"""自定义掌部 mesh 配置占位。

    这条路径留给未来从 CAD / Blender / mesh 工具链导出的掌部。
    本轮 pre-made 纵向切片暂不展开。
    """


class SinglePalmBuilder(PalmBuilder):
    r"""单一基础几何掌部构建器。"""

    cfg: SinglePalmBuilderCfg

    def __init__(self, cfg: SinglePalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        r"""根据单一 primitive 参数构建 `PalmCfg`。

        这里真正要表达的是“掌部的几何与惯量如何落到当前 schema 上”，
        而不是简单拼几个字段。

        当前支持四类 shape：

        1. `box`
        2. `cylinder`
        3. `sphere`
        4. `ellipse`

        其中 `ellipse` 需要额外说明：

        理论上，你原始 TODO 的意图是“把椭球体视作球体经各向异性 scale 后的结果”，
        也就是底层语义上接近：

        $$
        \text{sphere}(1.0) \xrightarrow{\text{scale}(a,b,c)} \text{ellipsoid}(a,b,c),
        \quad c = h/2.
        $$

        这条思路本身没有问题，我之前把它误写成“schema 不能表达，所以只能球近似”，
        这是我实现时的错误表达。更准确地说是：

        - 你的 TODO 说的是一种**期望的几何表达方式**
        - 但当前这版 `canonical schema + 标准 URDF writer` 还没有真正打通
          “scaled primitive” 这条通道
        - 标准 URDF 1.0 里 primitive (`box/cylinder/sphere`) 本身没有通用
          `scale` 属性，`scale` 主要属于 `<mesh>` 语义

        所以本轮实现先采取工程折中：

        - 惯量仍按**均质椭球体**公式计算
        - 几何导出先落为一个外包球 `sphere-envelope`

        这样做的原因不是 TODO 不清楚，而是我这次先优先保证
        `schema -> exporter -> test` 纵向闭环稳定。若后续你决定椭圆 palm
        是正式训练对象，推荐的升级方向是二选一：

        1. 在 schema / exporter 里显式支持“scaled primitive”；
        2. 使用单位球 mesh + scale 来表达椭球。

        Returns:
            PalmCfg: 掌部的 canonical 描述。
        """

        # ================================================================
        # Palm 设计帧约定（方案 C，与 PrimJointBuilder 同构）
        # ================================================================
        # 原点：形体底边中心
        #   x -> 掌宽方向
        #   y -> 朝指方向，即 palm 的“生长方向”
        #   z -> 掌厚方向
        #
        # --- TODO:算法之一：box（像 LEAP、Allegro 这类手较常用）
        # 输入：宽 $w$ (x), 长 $l$ (y), 高 $h$ (z), 质量 $m$
        # 几何中心：$\mathbf{c} = (0,\; l/2,\; 0)$
        #
        # --- TODO:算法之二：cylinder（比较适合夹爪手）
        # 输入：半径 $r$, 高 $h$ (z), 质量 $m$
        # 几何中心：$\mathbf{c} = (0,\; 0,\; 0)$
        #
        # --- TODO:算法之三：ellipse（夹爪手和类人手都可以用）
        # 输入：x 半轴 $a$, y 半轴 $b$, 高 $h$ (z), 质量 $m$
        # 期望表达：`sphere + scale(a, b, c)`，其中 $c = h/2$
        #
        # --- TODO:算法之四：sphere（很少用，先保留接口）
        # 输入：半径 $r$, 质量 $m$
        # 几何中心：$\mathbf{c} = (0,\; r,\; 0)$
        # ================================================================
        if self.cfg.shape == "box":
            # box palm 对应你原始 TODO 的“算法之一”。
            width = float(self.cfg.width)
            length = float(self.cfg.length)
            height = float(self.cfg.height)
            origin = PoseCfg(pos=(0.0, length / 2.0, 0.0))  # 几何中心 $\mathbf{c}=(0,l/2,0)$
            mass = _estimate_mass(width * length * height)  # 体积 $V = wlh$
            inertia = _box_inertia(width, length, height, mass)  # 长方体理论惯量
            geometry = {"type": "box", "size": (width, length, height)}
        elif self.cfg.shape == "cylinder":
            # cylinder palm 对应你原始 TODO 的“算法之二”。
            radius = float(self.cfg.radius)
            height = float(self.cfg.height)
            origin = PoseCfg()  # 圆柱体质心直接与 palm frame 重合
            mass = _estimate_mass(math.pi * radius * radius * height)  # 体积 $V=\pi r^2 h$
            inertia = _cylinder_inertia(radius, height, mass)  # 圆柱理论惯量
            geometry = {"type": "cylinder", "radius": radius, "length": height}
        elif self.cfg.shape == "sphere":
            # sphere palm 对应你原始 TODO 的“算法之四”。
            radius = float(self.cfg.radius)
            origin = PoseCfg(pos=(0.0, radius, 0.0))  # 球心位于底极点上方 $r$
            mass = _estimate_mass(4.0 * math.pi * radius**3 / 3.0)  # 体积 $V=4\pi r^3/3$
            inertia = _sphere_inertia(radius, mass)  # 球理论惯量
            geometry = {"type": "sphere", "radius": radius}
        else:
            # ellipse palm 对应你原始 TODO 的“算法之三”。
            a = float(self.cfg.a)
            b = float(self.cfg.b)
            c = float(self.cfg.height) / 2.0  # 半轴 $c=h/2$
            radius = max(a, b, c)  # 当前外包球近似半径
            origin = PoseCfg(pos=(0.0, b, 0.0))  # 质心 $\mathbf{c}=(0,b,0)$
            mass = _estimate_mass(4.0 * math.pi * a * b * c / 3.0)  # 椭球体积 $V=4\pi abc/3$
            inertia = {
                "ixx": mass * (b * b + c * c) / 5.0,
                "iyy": mass * (a * a + c * c) / 5.0,
                "izz": mass * (a * a + b * b) / 5.0,
            }
            geometry = {"type": "sphere", "radius": radius}  # 工程近似：先写外包球

        collision = CollisionGeometryCfg(name="palm_collision", geometry=geometry, origin=origin)  # collision 与 visual 首轮保持一致
        visual = VisualGeometryCfg(name="palm_visual", geometry=geometry, origin=origin)
        metadata = {"shape": self.cfg.shape}  # 保留 palm shape 供 hand-level / exporter 做 provenance
        if self.cfg.shape == "ellipse":
            metadata["ellipse_axes"] = {
                "a": float(self.cfg.a),
                "b": float(self.cfg.b),
                "c": float(self.cfg.height) / 2.0,
            }  # 记录真实椭球半轴，便于后续 exporter 升级
            metadata["approximation"] = "sphere_envelope"  # 当前几何仍是外包球近似
        return PalmCfg(
            name="palm",
            inertial=InertialCfg(mass=mass, origin=origin, inertia=inertia),
            collisions=[collision],
            visuals=[visual],
            metadata=metadata,
        )


_COM_PALM_PRESETS: dict[str, dict[str, object]] = {
    "allegro": {
        "collisions": [
            {"size": (0.0414, 0.1120, 0.0448), "origin": (-0.0090, 0.0000, -0.0230)},  # palm 主体
            {"size": (0.0414, 0.0538, 0.0428), "origin": (-0.0090, -0.0253, -0.0667)},  # 掌根后侧加厚块
            {"size": (0.0414, 0.0720, 0.0130), "origin": (-0.0093, -0.00557, -0.08874)},  # 更薄的下层条带
        ],
        "inertial": {
            "mass": 0.4154,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {"ixx": 1.0e-4, "iyy": 1.0e-4, "izz": 1.0e-4},
        },
        "finger_mounts": {
            "index": {"pos": (0.0, 0.0435, -0.001542), "rpy": (-0.0873, 0.0, 0.0)},
            "middle": {"pos": (0.0, 0.0, 0.0007), "rpy": (0.0, 0.0, 0.0)},
            "ring": {"pos": (0.0, -0.0435, -0.001542), "rpy": (0.0873, 0.0, 0.0)},
            "thumb": {"pos": (-0.0182, 0.019333, -0.045987), "rpy": (0.0, -1.6581, -1.5708)},
        },
    },
    "leap": {
        "collisions": [
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, 0.008, -0.011)},  # 上部块
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, -0.037, -0.011)},  # 下部块
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, -0.082, -0.011)},
            {"size": (0.058, 0.020, 0.046), "origin": (-0.066, -0.078, -0.0115), "rpy": (0.0, 0.0, -0.2967)},
            {"size": (0.020, 0.120, 0.030), "origin": (-0.030, -0.035, -0.003)},
            {"size": (0.010, 0.120, 0.020), "origin": (-0.032, -0.035, -0.024), "rpy": (0.0, 0.785, 0.0)},
            {"size": (0.024, 0.116, 0.046), "origin": (-0.048, -0.033, -0.0115)},
            {"size": (0.044, 0.052, 0.046), "origin": (-0.078, -0.053, -0.0115)},
            {"size": (0.004, 0.036, 0.034), "origin": (-0.098, -0.009, -0.006)},
            {"size": (0.044, 0.056, 0.004), "origin": (-0.078, -0.003, 0.010)},
        ],
        "inertial": {
            "mass": 0.237,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {
                "ixx": 3.54094e-4,
                "ixy": -1.193e-6,
                "ixz": -2.445e-6,
                "iyy": 2.60915e-4,
                "iyz": -2.905e-6,
                "izz": 5.29257e-4,
            },
        },
        "finger_mounts": {
            "index": {"pos": (-0.0070, 0.0230, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "middle": {"pos": (-0.0071, -0.0224, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "ring": {"pos": (-0.00709, -0.0678, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "thumb": {"pos": (-0.0693, -0.0012, -0.0216), "rpy": (0.0, 1.5708, 0.0)},
        },
    },
}


def get_single_palm_box_preset(name: str) -> SinglePalmBuilderCfg:
    r"""按名字返回一份单一 box palm preset。

    这组 preset 主要服务于“参数化 palm 的参考锚点”，不是为了取代
    `ComPalmBuilder` 的真实碰撞体 preset。
    """

    try:
        dims = _SINGLE_PALM_BOX_PRESETS[name]  # 先取参考锚点尺寸
    except KeyError as exc:
        raise KeyError(f"Unknown single palm box preset: {name!r}") from exc
    return SinglePalmBuilderCfg(shape="box", **dims)  # 返回一份新的 cfg，避免外部污染模块常量


def get_com_palm_preset(name: str) -> ComPalmBuilderCfg:
    r"""按名字返回一份复合 palm preset cfg。"""

    if name not in _COM_PALM_PRESETS:
        raise KeyError(f"Unknown composite palm preset: {name!r}")
    return ComPalmBuilderCfg(preset=name)  # 复合 palm preset 直接按名字回填


PALM_PRESET_REGISTRY = {
    "single_box_allegro": lambda: get_single_palm_box_preset("allegro"),
    "single_box_leap": lambda: get_single_palm_box_preset("leap"),
    "com_allegro": lambda: get_com_palm_preset("allegro"),
    "com_leap": lambda: get_com_palm_preset("leap"),
}
"""掌部 preset 的轻量注册表。

finger / palm 两侧都保留这种显式字典，是为了让 asset generator 后续在
“离散 preset 空间”和“连续参数空间”之间切换时有统一入口。
"""


class ComPalmBuilder(PalmBuilder):
    r"""复合基础几何掌部构建器。

    与 `SinglePalmBuilder` 不同，这里不是从连续参数空间采样，而是直接把
    真实手掌碰撞体的 box 组合搬进 `PalmCfg`。这样做的目的，是让：

    - Allegro / LEAP 作为 hand family 锚点时，空间语义尽量贴近真实 hand；
    - hand-level mount 直接复用真实 URDF 中的基准位姿。
    """

    cfg: ComPalmBuilderCfg

    def __init__(self, cfg: ComPalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        r"""根据 preset 查表构建复合掌部。

        Returns:
            PalmCfg: 含多组 collision / visual box 的掌部描述。
        """

        preset = _COM_PALM_PRESETS[self.cfg.preset]  # 真实 hand family 的复合碰撞体查表
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.preset}_col_{index}",  # collision box 命名稳定带上 preset 前缀
                geometry={"type": "box", "size": entry["size"]},  # 复合 palm 当前统一用 box 近似
                origin=PoseCfg(pos=entry["origin"], rpy=entry.get("rpy", (0.0, 0.0, 0.0))),  # 每块 box 都保留独立位姿
            )
            for index, entry in enumerate(preset["collisions"])
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.preset}_vis_{index}",  # visual 先与 collision 保持一致
                geometry={"type": "box", "size": entry["size"]},
                origin=PoseCfg(pos=entry["origin"], rpy=entry.get("rpy", (0.0, 0.0, 0.0))),
            )
            for index, entry in enumerate(preset["collisions"])
        ]
        mounts = {name: PoseCfg.from_value(value) for name, value in preset["finger_mounts"].items()}  # palm 自带 finger mounts
        metadata = {"preset": self.cfg.preset, "finger_mounts": mounts}  # hand builder 会继续读取这些 mount
        return PalmCfg(
            name="palm",
            inertial=InertialCfg(**preset["inertial"]),
            collisions=collisions,
            visuals=visuals,
            metadata=metadata,
        )


__all__ = [
    "SinglePalmBuilderCfg",
    "ComPalmBuilderCfg",
    "CustomPalmBuilderCfg",
    "SinglePalmBuilder",
    "ComPalmBuilder",
    "PALM_PRESET_REGISTRY",
    "get_single_palm_box_preset",
    "get_com_palm_preset",
]
