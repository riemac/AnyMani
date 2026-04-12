r"""TODO:自定义手指构建器配置类 `FingerBuilderCfg` 和运行时类 `FingerBuilder`。

补充背景：以真实人手为例，此处我分为拇指型和非拇指型，用于和机器手做对照。
以下说明不是“可有可无的背景文字”，而是后续 finger-level 算法设计的语义依据。

- 非拇指型：即 index、middle、ring、little 四指，结构较为相似。
  从靠近手掌到远端，典型由 MCP / PIP / DIP 构成。
  - MCP（Metacarpophalangeal Joint）：工程上通常看作 2 DOF，
    主要负责屈曲/伸展与外展/内收。
  - PIP（Proximal Interphalangeal Joint）：典型铰链关节，1 DOF。
  - DIP（Distal Interphalangeal Joint）：同样近似铰链关节，1 DOF，
    并且与 PIP 存在肌腱耦合、欠驱动意味更浓。
- 拇指型：thumb 是整个人手里最特殊的手指。
  - CMC（Carpometacarpal Joint）：鞍状关节，承担拇指最核心的对掌能力。
  - MCP：以屈曲/伸展为主。
  - IP：远端铰链关节。

当前资产建模上，又主要面对几类机器手：

- `allegro` / `leap`：全驱中型手，是本轮跨手型泛化的主要切入点；
- `shadow` / `schunk`：球关节或欠驱更明显，但暂不作为首轮重点；
- `dclaw_gripper` / `bhand`：夹爪手，形态差异更大，后续再议。

本文件当前首轮实现只覆盖 regular family：

- Allegro 非拇指
- LEAP 非拇指
- regular thumb（含 `CMC1` 特例）

阅读本文件时，建议始终把两层关系分开看：

1. `joint frame -> mesh frame`
2. `parent link frame -> child joint frame`

真正的建模难点不是 box / cylinder 本身，而是不同 hand family 下，
这两层关系如何保持坐标语义一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..asset_base import FingerCfg
from ..asset_builders import FingerBuilder, FingerBuilderCfg
from ..asset_schema_core import PoseCfg, Vector2, Vector3, Vector6, _ensure_tuple, _normalize_axis
from ._utils import (
    _build_box_mesh,
    _build_cylinder_mesh,
    _mesh_cross_section,
    _mesh_length,
    _normalize_joint_limits,
    _normalize_pose_list,
    _normalize_pose_value,
    _to_si,
)
from .joint_builders_primitive import PrimJointBuilderCfg


def _normalize_tip_dict(tip: dict[str, Any] | None) -> dict[str, Any]:
    r"""规范化指尖 recipe，并把长度统一成米制。

    当前首轮只支持两类 primitive 复合指尖：

    - `cs`: cylinder + sphere
    - `bs`: box + sphere

    这和你原始 TODO 里的“第一版先实现 cylinder/box + sphere，自定义指尖留接口”
    是一致的。
    """

    tip = dict(tip or {"type": "cs", "radius": 0.012, "height": 0.01})  # 默认走最常见的圆柱+半球指尖
    tip_type = str(tip.get("type", tip.get("kind", "cs"))).lower()  # 兼容历史 `kind=` 写法
    normalized: dict[str, Any] = {"type": tip_type}
    if tip_type == "cs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
    elif tip_type == "bs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
        normalized["width"] = _to_si(tip.get("width", tip.get("depth", 0.02)))
        normalized["depth"] = _to_si(tip.get("depth", tip.get("width", 0.02)))
    else:
        raise ValueError(f"Only cs/bs tip recipes are supported in v1, got {tip_type!r}")
    return normalized



# --- 手指的声明式配置类 --- #
# TODO: 经过大量手指调研，决定先划分为：
# 1. 含球类关节的手指
# 2. 不含球类关节的手指
#
# 对不含球类关节手指而言，thumb/non-thumb、全驱/欠驱、夹爪/类人，很多时候
# 主要差异都体现在：
# - mesh 怎么摆
# - 旋转轴怎么放
# - 关节链长度怎么配
# 它们的算法骨干并没有大到必须彻底分裂成很多 builder 子类。
#
# 因此当前采用“多个并列 cfg -> 一个 RegularFingerBuilder”的方式：
# - 每个 cfg 的字段集合更精确、更接近研究者心流
# - builder 端仍只需要消费 canonical 字段，职责边界清晰
@dataclass
class RegularFingerBuilderCfg(FingerBuilderCfg):
    r"""常规非球关节手指配置。

    这个 cfg 对应你原始注释里的“RegularFingerBuilderCfg 大类”。核心思想是：

    - 把 Allegro / LEAP / thumb / non-thumb 先压到同一套 canonical 字段；
    - 真正让 builder 消费的只保留 `mesh_shape`、`mesh_offsets`、`axes`、
      `tip` 这几组高信息量输入；
    - 其余子类字段，本质上都是为了更方便地构造这几组 canonical 输入。
    """

    class_type: type["RegularFingerBuilder"] | None = None
    """关联的 regular 手指运行时构建器。"""

    name: str = "finger"
    """手指逻辑名。

    该字段不是纯展示字符串，而会直接参与 joint/link 命名，因此需要保持稳定。
    """

    parent_link: str = "palm"
    """finger 根部挂载到哪个 parent link。默认挂到 palm。"""

    num_joints: int = 4
    """运动关节数，不包含最后额外补出的 fixed tip joint。"""

    mesh_shape: list[dict[str, Any]] = field(default_factory=list)
    """每个 joint mesh 的 canonical primitive recipe。

    这组字段是 builder 真正消费的核心输入之一。对于 Allegro/LEAP/thumb 的差异，
    最终都会被规约到这里。
    """

    mesh_offsets: list[Any] = field(default_factory=list)
    """每个 joint mesh 相对于本 joint frame 的偏移。

    支持三种渐进式输入精度：

    - `list[float]`：只写 $y$ 偏移
    - `list[xyz]`：写平移
    - `list[xyzrpy]`：写完整位姿
    """

    _mesh_offsets_6d: list[Vector6] = field(init=False, default_factory=list)
    """解析后的标准 6D mesh offsets，供构建算法直接使用。"""

    tip: dict[str, Any] = field(default_factory=dict)
    """指尖 recipe。

    首轮只实现：

    - `cs`：cylinder + sphere
    - `bs`：box + sphere

    自定义 tip 仍留作接口，不在本轮抢跑。
    """

    tip_offset: Any = None
    """指尖 mesh 相对于 fixed tip joint frame 的偏移。"""

    _tip_offset_6d: Vector6 = field(init=False, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """解析后的标准 6D tip offset。"""

    axes: list[Vector3] = field(default_factory=list)
    """每个关节的旋转轴，在统一 finger frame 下表达。"""

    joint_limits: list[Any] = field(default_factory=list)
    """逐关节限位覆盖。为空时使用宽松默认限位。"""

    def __post_init__(self):
        super().__post_init__()
        if self.num_joints < 1:
            raise ValueError("num_joints must be >= 1")
        # 先把多精度偏移统一压到 6D 表达，后续算法只读取 canonical 结果。
        self._mesh_offsets_6d = _normalize_pose_list(self.mesh_offsets, count=self.num_joints, field_name="mesh_offsets")
        self._tip_offset_6d = _normalize_pose_value(self.tip_offset, field_name="tip_offset")
        self.tip = _normalize_tip_dict(self.tip)  # 统一成 builder/exporter 都可理解的 tip recipe
        if not self.axes:
            self.axes = [(0.0, 0.0, 1.0) for _ in range(self.num_joints)]  # 默认给一个占位轴，子类通常会覆盖
        if len(self.axes) != self.num_joints:
            raise ValueError(f"axes length must equal num_joints={self.num_joints}")
        self.axes = [_normalize_axis(_ensure_tuple(axis, length=3, field_name="axes")) for axis in self.axes]  # 旋转轴统一归一化
        self.joint_limits = _normalize_joint_limits(self.joint_limits, count=self.num_joints)  # 限位也压成统一对象
        if len(self.mesh_shape) != self.num_joints:
            raise ValueError(f"mesh_shape length must equal num_joints={self.num_joints}")
        self.class_type = RegularFingerBuilder  # 所有 regular family 先统一走一个 builder 骨干

@dataclass
class AllegroFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""Allegro 非拇指配置。

    这里直接保留了你 TODO 里的第一版 preset 直觉：

    $$
    l=(1.8, 5.4, 3.8, 2.2)\text{cm},\quad
    d=(0, 0, -0.6, 0)\text{cm}.
    $$

    默认仍以 box primitive 为主，因为这更贴近 Allegro“规整块体”的碰撞体直觉。
    """

    width: float | None = None
    """关节 mesh 的统一宽度。

    这里的宽度对应统一 finger frame 下的 $x$ 方向尺寸。
    对 Allegro 非拇指而言，若走 box 路线，则除 tip 外的各段通常都共用这一个宽度。
    默认值按你原始 TODO 里的经验值取约 $2.7$cm。
    """

    height: float | None = None
    """关节 mesh 的统一高度。

    这里的高度对应统一 finger frame 下的 $z$ 方向尺寸。
    对 Allegro 非拇指而言，若走 box 路线，则除 tip 外的各段通常都共用这一个高度。
    默认值按你原始 TODO 里的经验值取约 $2.0$cm。
    """

    radius: float | None = None
    """关节 mesh 的统一半径。

    这是给 cylinder 路线保留的替代参数。也就是说：

    - 若显式给出 `radius` 且未给 `width/height`，则当前实现可切到 cylinder primitive；
    - 若给了 `width/height`，则仍优先按 box primitive 处理。

    这样做是为了保留“规整块体”和“圆柱近似”两条 regular finger 路线。
    """

    length: list[float] = field(default_factory=lambda: [1.8, 5.4, 3.8, 2.2])
    """各段 joint mesh 沿 finger 生长方向的长度列表。

    默认值对应你原始 TODO 里的 Allegro 非拇指四段经验长度：

    $$
    l=(1.8, 5.4, 3.8, 2.2)\text{cm}.
    $$

    它表达的是几何段长，而不是 joint-to-joint 纯运动学长度；
    真正用于推进下一 joint 的量，会在 builder 中进一步结合 `mesh_offsets`
    变成有效长度。
    """

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]  # 长度列表 $l_i$
        width = _to_si(self.width or 2.7)  # Allegro 非拇指默认宽度 $w=2.7$cm
        height = _to_si(self.height or 2.0)  # Allegro 非拇指默认高度 $h=2.0$cm
        radius = _to_si(self.radius) if self.radius is not None else None  # 若显式给半径，则允许切到 cylinder 路线
        defaults = _normalize_pose_list([0.0, 0.0, -0.6, 0.0][: self.num_joints], count=self.num_joints, field_name="allegro_default_offsets")
        merged_offsets = self.mesh_offsets or defaults  # 默认只保留第三段负向 $y$ 偏移，对应原始 TODO 中的末段重叠近似
        self.mesh_offsets = merged_offsets
        if not self.axes:
            self.axes = [(0.0, 1.0, 0.0)] + [(1.0, 0.0, 0.0)] * max(self.num_joints - 1, 0)  # 0 号绕 $y$，其余近似绕 $x$
        if not self.tip:
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}  # 默认圆柱+半球指尖
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset  # 解析后的 canonical offset 回填到 mesh recipe


@dataclass
class LeapFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""LEAP 非拇指配置。

    与 Allegro 非拇指相比，LEAP 在首轮实现里最重要的差异不是段数，而是：

    1. palm 侧存在固定段 `fixed_part`
    2. MCP 两关节的轴语义不同
    """

    width: float | None = None
    """关节 mesh 的统一宽度。

    这里对应简化后 LEAP 非拇指 box 近似下的 $x$ 向尺寸。
    默认值按当前第一版近似取约 $3.4$cm。
    """

    height: float | None = None
    """关节 mesh 的统一高度。

    这里对应简化后 LEAP 非拇指 box 近似下的 $z$ 向尺寸。
    默认值按当前第一版近似取约 $2.05$cm。
    """

    radius: float | None = None
    """关节 mesh 的统一半径。

    和 `AllegroFingerBuilderCfg.radius` 一样，这是给 cylinder 路线保留的替代参数。
    当前 LEAP 默认仍走 box 简化近似，但保留 cylinder 路线是为了不把 regular
    family 的参数空间过早锁死。
    """

    length: list[float] = field(default_factory=lambda: [3.9, 1.5, 3.6, 2.0])
    """各段 joint mesh 沿 finger 生长方向的长度列表。

    默认值对应你原始 TODO 里 LEAP 非拇指的第一版简化近似：

    $$
    l=(3.9, 1.5, 3.6, 2.0)\text{cm}.
    $$

    这些量描述的是“每段 mesh 的主长度”，而不是完整 fingertip 合并后的真实 mesh。
    """
    fixed_part: float | None = None
    """固定部分长度 $l_f$。

    LEAP 非拇指中，第一个运动关节轴到 palm 边缘之间通常还有一小段固定部分。
    当前实现不把它单独建成一个 fixed joint，而是把它直接折算为
    “从 palm 到第一个 joint frame 的前置 $y$ 向长度”。
    """

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]  # 长度列表 $l_i$
        width = _to_si(self.width or 3.4)  # LEAP 简化 box 宽度
        height = _to_si(self.height or 2.05)  # LEAP 简化 box 高度
        radius = _to_si(self.radius) if self.radius is not None else None
        self.fixed_part = _to_si(self.fixed_part or 1.3)  # palm 侧固定段长度 $l_f$
        if not self.axes:
            defaults = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
            self.axes = defaults[: self.num_joints]  # 保留你 TODO 里的 LEAP 非拇指轴语义
        if not self.tip:
            # User confirmed that the first testing path may use cylinder+sphere.
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset  # 解析后的标准位姿反写回 recipe


@dataclass
class RegularThumbBuilderCfg(RegularFingerBuilderCfg):
    r"""regular thumb 配置。

    这里最关键的不是长度数值，而是 `CMC1` 的坐标约定和普通 finger joint 不同：

    - 普通 joint：偏移为 0 时，mesh 默认从 joint frame 的 $x-z$ 平面向 $+y$ 生长
    - `CMC1`：偏移为 0 时，mesh frame 与 joint frame 完全重合

    这正是你在 `Thumb.png` 里强调的 thumb 特例。
    """

    cmc1_width: float | None = None
    """CMC1 这一特例块体的宽度。

    它对应 thumb 第一段、也就是你原始说明里的 12 号关节/CMC1 那个“大块体”。
    该段通常明显不同于后续各段，因此单独给宽度字段，而不和其他段共用 `width`。
    """

    cmc1_height: float | None = None
    """CMC1 这一特例块体的高度。

    和 `cmc1_width` 一样，这里单独给出是因为 CMC1 几何外形通常更厚、更嵌入 palm。
    """

    width: float | None = None
    """除 CMC1 外，其余 thumb joint mesh 的统一宽度。

    该字段主要作用于 CMC2/MCP/IP 等后续段，使它们共享一套更细长的截面尺寸。
    """

    height: float | None = None
    """除 CMC1 外，其余 thumb joint mesh 的统一高度。"""

    lengths: list[float] = field(default_factory=lambda: [4.5, 1.7, 4.3, 4.0])
    """thumb 各段长度列表。

    默认值当前更偏向 Allegro 风格的 thumb 近似：

    $$
    l=(4.5, 1.7, 4.3, 4.0)\text{cm}.
    $$

    这里用复数 `lengths` 而不是单数 `length`，是为了和 non-thumb 的 cfg 在阅读上
    做出一点区分，提醒读者：thumb 的第一段 `CMC1` 在几何语义上并不是普通串联段。
    """

    cmc1_offset: float | Vector2 | Vector3 = (0.9, 1.45)
    r"""CMC1 mesh 相对于本 joint frame 的位置偏移。

    这是 thumb 最关键的特例字段之一。它支持三种精度：

    - `float`：只写 $y$ 偏移
    - `Vector2`：写 $(y, z)$ 偏移
    - `Vector3`：写 $(x, y, z)$ 偏移

    之所以单独保留这个字段，而不是把它塞进通用 `mesh_offsets`，是因为：

    1. `CMC1` 的零偏移语义与普通 joint 不同；
    2. 研究者在调 thumb 时，通常会首先思考 CMC1 在 palm 内部的嵌入位置。
    """

    non_cmc1_offset: list[Any] = field(default_factory=lambda: [-0.2, 0.0, -0.9])
    """除 CMC1 外，其余 thumb joint mesh 的偏移列表。

    该字段只负责 CMC2 及之后的各段，因此长度应为 `num_joints - 1`。
    它支持和 `mesh_offsets` 相同的渐进式输入精度，但把 CMC1 单独拆出来后，
    thumb 的建模语义会更清晰，不容易把“CMC1 特例”和“普通串联段”混在一起。
    """

    def __post_init__(self):
        self.num_joints = len(self.lengths)  # thumb 段数直接由 lengths 决定
        lengths = [_to_si(value) for value in self.lengths]  # 长度列表 $l_i$
        cmc1_width = _to_si(self.cmc1_width or 3.5)  # CMC1 特例块体宽度
        cmc1_height = _to_si(self.cmc1_height or 3.4)  # CMC1 特例块体高度
        width = _to_si(self.width or 1.9)  # 其余关节统一宽度
        height = _to_si(self.height or 2.7)  # 其余关节统一高度

        cmc1_pose = _normalize_pose_value(self.cmc1_offset, field_name="cmc1_offset")  # CMC1 偏移单独解析
        other_offsets = _normalize_pose_list(self.non_cmc1_offset, count=self.num_joints - 1, field_name="non_cmc1_offset")
        self.mesh_offsets = [cmc1_pose] + other_offsets  # thumb 先拼成一份完整 offsets，再交给 super 统一规范化
        if not self.axes:
            # Question:
            # thumb 各关节的轴语义在 Allegro / LEAP 原始 URDF 中并不完全同构。
            # 当前先选一套稳定 canonical 约定，后续若网络结构希望显式编码
            # “轴语义差异”，这里再继续细分。
            self.axes = [
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        if not self.tip:
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}  # thumb 首轮也统一走 primitive tip
        if not self.mesh_shape:
            self.mesh_shape = [
                _build_box_mesh(length=lengths[0], width=cmc1_width, height=cmc1_height, offset=cmc1_pose, center_on_joint=True),  # CMC1 零偏移时 mesh frame 与 joint frame 重合
                *[
                    _build_box_mesh(length=lengths[idx], width=width, height=height, offset=other_offsets[idx - 1])
                    for idx in range(1, self.num_joints)
                ],
            ]
        super().__post_init__()
        self.mesh_shape[0]["center_on_joint"] = True  # 再次显式标记 CMC1 特例，避免下游丢语义
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset  # 把标准位姿写回 recipe


@dataclass
class SphericalFingerBuilderCfg(FingerBuilderCfg):
    r"""球关节手指配置占位。

    这里对应你原始设想里的 shadow / schunk 一类手指。
    由于它们的 joint mesh 组织方式、轴心语义、碰撞体组合方式都明显不同于
    regular finger，本轮先不与 `RegularFingerBuilder` 混在一起实现。
    """


class RegularFingerBuilder(FingerBuilder):
    r"""常规/非球类关节手指构建器。

    NOTE: 以下算法说明的输入输出并不完全等同于构建器的函数签名，
    它更贴近研究设计时的心流表达。真正的代码实现则把这些思路压到
    `mesh_shape`、`mesh_offsets`、`axes`、`tip` 等 canonical 字段里。
    """

    cfg: RegularFingerBuilderCfg

    def __init__(self, cfg: RegularFingerBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg  # builder 端只消费已经规范化好的 regular cfg

    def build(self) -> FingerCfg:
        r"""构建一根 regular finger。

        Returns:
            FingerCfg: 由若干 revolute joint 与一个 fixed tip joint 组成的手指描述。
        """

        # 构建分流：
        # - thumb：走 CMC1 特例链
        # - 非 thumb：走普通串联链
        if isinstance(self.cfg, RegularThumbBuilderCfg):
            joints = self._build_thumb_chain()  # thumb 不能完全套普通 serial chain
        else:
            root_fixed_length = self.cfg.fixed_part if isinstance(self.cfg, LeapFingerBuilderCfg) else 0.0
            joints = self._build_serial_chain(root_fixed_length=root_fixed_length)  # LEAP 用 `l_f` 表达 palm 到第一个 joint frame 的前置长度
        return FingerCfg(
            name=self.cfg.name,  # finger 名会继续传播到 exporter / validator / sidecar
            parent_link=self.cfg.parent_link,  # hand-level 装配前的默认 parent
            mount=PoseCfg(),  # 具体 mount 在 hand builder 中再覆盖
            joints=joints,  # joint 串联链是 finger builder 的核心输出
            metadata={"builder": self.cfg.__class__.__name__},  # 保留 provenance 便于追溯
        )

    def _build_serial_chain(self, *, root_fixed_length: float) -> list[Any]:
        r"""构建普通串联链。

        这里对应 Allegro / LEAP 非拇指的共同主干。推进下一关节时，真正使用的
        不是单纯 `mesh length`，而是你原始 TODO 里强调的“有效长度”：

        $$
        c_{\text{valid}, i} = l_i + d_i.
        $$
        """
        # --- TODO:算法之一 ---：allegro 非拇指型手指构造
        # 输入：手指关节数 $N$（2~4），joint mesh 长度 $l\in \mathbb{R}^N$，
        # mesh 偏移 $d\in\mathbb{R}^{N}$，指尖类型及其参数。
        #
        # --- TODO:算法之二 ---：leap 非拇指型手指构造
        # 输入：手指关节数 $N$(1~4)，joint mesh 长度 $l\in \mathbb{R}^N$，
        # 宽度 $w$，高度 $h$，或半径 $r$，mesh 偏移 $d\in\mathbb{R}^{N}$，
        # 指尖类型及其参数，固定部分长度 $l_f$。
        #
        # 当前实现让 Allegro / LEAP 共用同一条 serial-chain 骨干：
        # - Allegro：从 palm 到第一个 joint frame 的前置长度为 $0$
        # - LEAP：从 palm 到第一个 joint frame 的前置长度为 $l_f$
        joints = []
        parent_link = self.cfg.parent_link  # 第一个 joint 默认挂在 palm 上
        previous_valid_length = root_fixed_length  # 先吃掉 palm 侧固定部分长度；Allegro 情况下该值为 0
        for index in range(self.cfg.num_joints):
            origin = PoseCfg(pos=(0.0, previous_valid_length, 0.0)) if index > 0 or root_fixed_length > 0.0 else PoseCfg()
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)  # 本 joint frame 相对上一 child link 的位姿
            joints.append(joint)
            parent_link = joint.child  # 下一关节继续接在当前 child link 后
            previous_valid_length = _mesh_length(self.cfg.mesh_shape[index]) + self.cfg._mesh_offsets_6d[index][1]  # 有效推进长度 $c_{\text{valid},i}$

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=previous_valid_length))  # 最后一段后面补 fixed tip joint
        return joints

    def _build_thumb_chain(self) -> list[Any]:
        r"""构建拇指链。

        这里把你图里的两套量分开处理：

        - $P_i$：第 $i$ 个 joint 相对于上一 joint 的位置
        - $mP_i$：第 $i$ 个 mesh 相对于本 joint 的位置

        因此 `CMC1 -> CMC2` 这一步不能直接套普通 serial chain。
        """
        # --- TODO:算法之三与算法之四 ---：allegro 和 leap 拇指型手指构造
        # 真正的关键不是简单串联，而是把：
        # - $P_i$：joint frame 串联位置
        # - $mP_i$：mesh 相对于本 joint frame 的位置
        # 分开表达。尤其 `CMC1 -> CMC2` 这一步是 thumb 的真正特例。
        cfg = self.cfg
        assert isinstance(cfg, RegularThumbBuilderCfg)

        joints = [self._build_joint(index=0, parent_link=cfg.parent_link, origin=PoseCfg())]  # CMC1 从 finger 根直接长出
        parent_link = joints[0].child  # 后续 joint 都挂在 CMC1 的 child link 之后

        cmc1_offset = cfg._mesh_offsets_6d[0]  # CMC1 的 mesh 偏移 $mP_0$
        cmc1_length = _mesh_length(cfg.mesh_shape[0])  # CMC1 长度 $l_0$
        cmc1_width, cmc1_height = _mesh_cross_section(cfg.mesh_shape[0])  # CMC1 横截面尺寸
        next_width, next_height = _mesh_cross_section(cfg.mesh_shape[1])  # CMC2 横截面尺寸
        origin_1 = PoseCfg(
            pos=(
                (cmc1_width - next_width) / 2.0,  # 通过宽度差把 CMC2 对齐到 thumb 侧边
                cmc1_offset[1] + cmc1_length / 2.0,  # 沿生长方向推进到 CMC1 中上部
                cmc1_offset[2] - (cmc1_height - next_height) / 2.0,  # 在 $z$ 上补偿 CMC1 与后续段高度差
            )
        )
        joint_1 = self._build_joint(index=1, parent_link=parent_link, origin=origin_1)  # 单独处理 CMC1 -> CMC2 这一步
        joints.append(joint_1)
        parent_link = joint_1.child

        previous_valid_length = _mesh_length(cfg.mesh_shape[1]) + cfg._mesh_offsets_6d[1][1]  # CMC2 之后重新回到普通串联推进
        for index in range(2, cfg.num_joints):
            origin = PoseCfg(pos=(0.0, previous_valid_length, 0.0))  # 其余 MCP / IP 段近似同构
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)
            joints.append(joint)
            parent_link = joint.child
            previous_valid_length = _mesh_length(cfg.mesh_shape[index]) + cfg._mesh_offsets_6d[index][1]  # 继续按有效长度推进

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=previous_valid_length))  # 末端补一个 fixed tip
        return joints

    def _build_joint(self, *, index: int, parent_link: str, origin: PoseCfg):
        r"""把第 `index` 个运动关节落成 `JointCfg`。"""
        mesh = dict(self.cfg.mesh_shape[index])  # 复制一份 recipe，避免修改 cfg 常量
        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_j{index}",  # joint 名稳定由 finger 名和序号组成
            parent=parent_link,  # parent 指向上一段 child link
            child=f"{self.cfg.name}_link_{index}",  # 当前 child link 名
            joint_type="revolute",  # 运动关节统一为 revolute
            origin=origin,  # 当前 joint frame 相对 parent link frame 的位姿
            axis=self.cfg.axes[index],  # 当前关节旋转轴
            limit=self.cfg.joint_limits[index],  # 当前关节限位
            mesh=mesh,
            metadata={
                "finger_name": self.cfg.name,
                "joint_index": index,
                "allow_zero_origin": index == 0 and origin.pos == (0.0, 0.0, 0.0),  # 根 joint 允许零位 origin
            },
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()

    def _build_tip_joint(self, *, parent_link: str, tip_origin_y: float):
        r"""构建固定指尖关节。

        当前 LEAP 首轮测试路径也统一走 `fixed tip joint + primitive tip`，
        这是为了先把 pre-made 闭环跑通；若后续要恢复“tip 并入最后一段 link”
        的 LEAP 语义，可以从这里继续分叉。
        """
        tip_recipe = dict(self.cfg.tip)  # 指尖 recipe 先复制，避免污染 cfg
        tip_recipe["offset"] = self.cfg._tip_offset_6d  # 指尖 mesh 相对 tip joint frame 的位姿
        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_tip",  # tip joint 命名稳定，便于 exporter / validator 识别
            parent=parent_link,  # tip 接在最后一个运动关节之后
            child=f"{self.cfg.name}_tip_link",  # tip link 也独立命名
            joint_type="fixed",  # 指尖关节为 fixed
            origin=PoseCfg(pos=(0.0, tip_origin_y, 0.0)),  # tip joint frame 落在最后一段有效长度末端
            axis=(0.0, 0.0, 0.0),  # fixed joint 不需要有效转轴
            limit=None,  # fixed joint 不需要限位
            mesh=tip_recipe,
            is_tip=True,
            metadata={"finger_name": self.cfg.name, "joint_index": "tip"},
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()


def get_finger_builder_preset(name: str) -> RegularFingerBuilderCfg:
    r"""按名字返回一份 finger builder preset 副本。

    Args:
        name (str): preset 名。

    Returns:
        RegularFingerBuilderCfg: 对应的构建器配置副本。

    Raises:
        KeyError: 当 preset 名不存在时抛出。
    """

    try:
        return FINGER_PRESET_REGISTRY[name].copy()
    except KeyError as exc:
        raise KeyError(f"Unknown finger builder preset: {name!r}") from exc


# ============================================================================
#  预定义 preset
# ============================================================================

# 这些 preset 不是为了取代参数化 cfg，而是为了给 generator / hand builder
# 一个稳定的离散锚点库。对于跨 hand family 的枚举实验，通常会把：
# - palm 类型
# - 非拇指 finger 类型
# - thumb 类型
# 组合起来形成 pre-made 生成空间。
#
# 它们的存在意义不只是“方便起手的默认参数”，更是：
# 1. 真实 hand family 的离散锚点；
# 2. 批量枚举时的稳定节点；
# 3. sidecar / provenance 可回溯的语义标签。

ALLEGRO_FINGER_PRESET = AllegroFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=2.7,
    height=2.0,
    length=[1.8, 5.4, 3.8, 2.2],
    mesh_offsets=[0.0, 0.0, -0.6, 0.0],
    axes=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""Allegro 非拇指执行型 preset。

这里刻意把 preset 的几何参数完整展开写在本地，而不是只写一个抽象名字，
这样读代码时一眼就能看到：

- 长度
- 截面尺寸
- 偏移
- 旋转轴
- 指尖类型

注册表名 `allegro_non_thumb_v1` 只保留在 `FINGER_PRESET_REGISTRY` 里，作为
外部查找键，而不是反向塞回 cfg 内部。
"""

LEAP_FINGER_PRESET = LeapFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=3.4,
    height=2.05,
    length=[3.9, 1.5, 3.6, 2.0],
    mesh_offsets=[0.0, 0.0, 0.0, 0.0],
    fixed_part=1.3,
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""LEAP 非拇指执行型 preset。

# Question:
原始设计里，LEAP 的 tip 更贴近 custom `white_tip.obj` 语义；当前 v1 为了
先打通 pre-made 闭环，执行路径仍采用 `cylinder + sphere` primitive tip。
"""

ALLEGRO_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[4.5, 1.7, 4.3, 4.0],
    cmc1_width=3.5,
    cmc1_height=3.4,
    width=1.9,
    height=2.7,
    cmc1_offset=(0.9, 1.45),
    non_cmc1_offset=[-0.2, 0.0, -0.9],
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""Allegro 拇指执行型 preset。"""

LEAP_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[2.8, 1.7, 4.7, 2.3],
    cmc1_width=2.30,
    cmc1_height=2.67,
    width=2.3,
    height=3.47,
    cmc1_offset=(0.0, -0.33),
    non_cmc1_offset=[0.0, 0.0, 0.0],
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""LEAP 拇指执行型 preset。"""

FINGER_PRESET_REGISTRY: dict[str, RegularFingerBuilderCfg] = {
    "allegro_non_thumb_v1": ALLEGRO_FINGER_PRESET,
    "leap_non_thumb_v1": LEAP_FINGER_PRESET,
    "allegro_thumb_v1": ALLEGRO_THUMB_PRESET,
    "leap_thumb_v1": LEAP_THUMB_PRESET,
}
"""finger builder 的轻量注册表。

这里刻意不用 gym-style 全局重注册机制，而是先保持一个模块内显式字典：

- 类型安全
- 易读
- 便于后续扩展到 YAML / recipe loader
"""


__all__ = [
    "RegularFingerBuilderCfg",
    "AllegroFingerBuilderCfg",
    "LeapFingerBuilderCfg",
    "RegularThumbBuilderCfg",
    "SphericalFingerBuilderCfg",
    "RegularFingerBuilder",
    "ALLEGRO_FINGER_PRESET",
    "LEAP_FINGER_PRESET",
    "ALLEGRO_THUMB_PRESET",
    "LEAP_THUMB_PRESET",
    "FINGER_PRESET_REGISTRY",
    "get_finger_builder_preset",
]
