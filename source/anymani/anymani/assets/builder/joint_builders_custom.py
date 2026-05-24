r"""自定义指尖关节构建器：把 mesh tip recipe 落为 `JointCfg`。

这一层对应你原始草稿里“custom fingertip v1”的入口，但当前实现刻意只做
**指尖**，不抢跑到一般 joint-level custom mesh。原因有三条：

1. 当前 pre-made 主链已经用 primitive regular link 跑稳，真正需要提高表达力的
   首先是 tip，而不是整根 finger 的每一段；
2. 指尖是主要接触部位，自定义 mesh 在这里最有物理意义；
3. 你在草稿里和后续可视化校准中已经收敛出具体的 mesh tip 锚点算法：
   - `leap_cube`
   - `wedge`
   - `round`
   - `thinner`

因此本文件的职责非常窄：

- 读取一个“tip 类型 + mesh 锚点 + scale + offset”的声明式配置；
- 把 visual / collision 都写成 `mesh` 几何；
- 只表达几何、锚点与安装相位；最终动力学由 `asset_physics.py` 统一闭包。

动力学闭包尤其重要。你在测试 URDF 注释里已经明确提醒：
“不写 inertial 不代表就没事，只是把问题交给 importer/PhysX 兜底。”
对于接触敏感的 tip，我们不应把这个语义空着。

# NOTE:
custom mesh builder 的 contract 是几何 lowering。真正写入最终 sidecar / URDF 的
`mass / inertial`，应由 generator 主链中的 physics closure 根据最终 collision
几何统一闭包；若有人绕过 generator 直接导出，`UrdfWriter` 只能写入默认占位
inertial，因此科研主链不要绕过 closure。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..asset_base import JointCfg
from ..asset_builders import JointBuilder, JointBuilderCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    JointLimitCfg,
    PoseCfg,
    Vector3,
    VisualGeometryCfg,
    _ensure_tuple,
)
from .joint_builders_primitive import _add_rpy


_CUSTOM_TIP_DIR = Path(__file__).resolve().parents[1] / "custom" / "tips"
"""当前项目内 custom tip mesh 的默认目录。"""


_DEFAULT_BASE_RPY = (0.0, -math.pi / 2.0, 0.0)
r"""custom tip 的默认 canonical 朝向。

你在 `test_round.urdf`、`test_wedge.urdf` 和 `test_leap_cube.urdf` 里都把
底→tip 的主方向规约到 joint 的 $+y$ 轴，且都采用了：

$$
R = R_y(-\pi/2).
$$

这意味着：

- 原始 mesh 的“前向/厚度语义”会被旋到当前 finger builder 采用的 $+y$ 生长方向；
- x/z 侧向则由右手系自动闭合。
"""


THUMB_FUNCTIONAL_TIP_PHASE_RPY = (0.0, -math.pi / 2.0, 0.0)
r"""thumb custom tip 的功能相位补偿。

custom tip mesh 不是轴对称几何，除了“底面锚点贴到 tip joint”之外，还必须
保留掌侧/背侧相位。non-thumb 的 canonical 相位直接采用 `_DEFAULT_BASE_RPY`；
thumb 的 MCP/DIP 弯曲平面要在 CMC2 绕局部 $y$ 约 $-\pi/2$ 后才进入
“朝掌心弯”的功能姿态，因此 thumb custom tip 在静态安装时额外叠加这个相位。

这个补偿只应作用于 custom mesh tip。`cs` 这类轴对称 primitive tip 不需要，
也不应该因此改变圆柱主轴。
"""


_CUSTOM_TIP_PRESETS: dict[str, dict[str, object]] = {
    "leap_cube": {
        "file_name": "finger_tip_soft.stl",
        "anchor_point": (9.48570692492, 0.0, -16.4999999586),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "round": {
        "file_name": "round_finger_tip_soft.stl",
        "anchor_point": (9.50986387389, 0.0, -16.4913187022),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "wedge": {
        "file_name": "wedge_finger_tip_soft.stl",
        "anchor_point": (9.5, 0.0, -16.5),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
    "thinner": {
        "file_name": "thinner_finger_tip_soft.stl",
        "anchor_point": (9.5, 0.0, -16.5),
        "unit_scale": 0.001,
        "base_rpy": _DEFAULT_BASE_RPY,
    },
}
r"""custom tip 预定义锚点库。

字段语义：

- `file_name`：默认 mesh 文件名
- `anchor_point`：mesh 局部坐标系中的语义锚点 $p^\*$
- `unit_scale`：从 mesh 文件单位到米制世界的基准换算
- `base_rpy`：canonical 朝向

# Question:
`wedge` 的测试 URDF 里出现过一个“按孔径 2mm 反推”的特殊统一缩放
`0.000689655...`；而草稿算法注释里又把 canonical configuration 写成
`0.001 I`。当前实现默认采用草稿算法里的 canonical `0.001`，并允许用户
通过 `scale` 显式覆写，不在这里偷偷替你选边站。
"""


def _pose_from_value(value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
    r"""把宽松位姿输入统一规范为 `PoseCfg`。"""

    return PoseCfg.from_value(value)  # 兼容 tuple / dict / PoseCfg，和 primitive builder 保持一致


def apply_thumb_functional_tip_phase(offset: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
    r"""给 thumb custom tip 的局部 mesh offset 叠加功能相位。

    Args:
        offset: 原始 tip mesh offset。它描述的是 anchor 在 tip joint frame 下的目标位姿。

    Returns:
        PoseCfg: 位置不变、`rpy` 额外叠加 `THUMB_FUNCTIONAL_TIP_PHASE_RPY` 的位姿。
    """

    pose = _pose_from_value(offset)
    return PoseCfg(pos=pose.pos, rpy=_add_rpy(pose.rpy, THUMB_FUNCTIONAL_TIP_PHASE_RPY))


def _scale_to_vector(value: float | Sequence[float]) -> Vector3:
    r"""把用户给的 scale 规约为三轴缩放向量。

    这里的 `scale` 是**无量纲**的用户级缩放，而不是最终写进 URDF 的 mesh scale。
    真正进入 URDF 的量是：

    $$
    \mathbf{s}_{\text{urdf}} = s_u \cdot \mathbf{s}_{\text{unit}}
    $$

    或逐轴形式：

    $$
    \mathbf{s}_{\text{urdf}} =
    (s_x, s_y, s_z) \odot \mathbf{s}_{\text{unit}}.
    $$
    """

    if isinstance(value, (int, float)):
        scale = float(value)  # uniform scale $s_u$
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {value}")
        return (scale, scale, scale)
    scale = _ensure_tuple(value, length=3, field_name="custom_tip.scale")
    if any(component <= 0.0 for component in scale):
        raise ValueError(f"custom tip scale must be positive, got {scale}")
    return scale


def _rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""构造 URDF 风格 `rpy` 的旋转矩阵。

    这里采用和 URDF 一致的固定轴旋转解释：

    $$
    R(\phi,\theta,\psi) = R_z(\psi) R_y(\theta) R_x(\phi).
    $$

    返回值按行存储，后续只需要做 `R p` 这种向量旋转，因此不引入额外矩阵类。
    """

    roll, pitch, yaw = rpy  # $(\phi,\theta,\psi)$
    cr, sr = math.cos(roll), math.sin(roll)  # $\cos\phi,\sin\phi$
    cp, sp = math.cos(pitch), math.sin(pitch)  # $\cos\theta,\sin\theta$
    cy, sy = math.cos(yaw), math.sin(yaw)  # $\cos\psi,\sin\psi$

    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    r"""把旋转矩阵作用到一个 3D 点上。"""

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def _resolve_tip_preset(tip_type: str) -> dict[str, object]:
    r"""按 tip 名返回预定义 custom tip 锚点。"""

    try:
        return dict(_CUSTOM_TIP_PRESETS[tip_type])
    except KeyError as exc:
        raise KeyError(f"Unknown custom tip preset: {tip_type!r}") from exc


@dataclass
class CustomJointBuilderCfg(JointBuilderCfg):
    r"""自定义 mesh 关节构建器配置基类。

    当前虽然只真正执行到 tip，但字段仍然故意做成 joint-centric，保持和
    `PrimJointBuilderCfg` 一致的接口肌理。这样 finger builder 在“primitive tip”
    与 “custom mesh tip” 之间切换时，不需要重写整套 joint-level 装配逻辑。
    """

    class_type: type["CustomJointBuilder"] | None = None
    """关联的自定义 mesh 关节构建器类。"""

    name: str = "joint"
    """输出到 `JointCfg` 中的 joint 名。"""

    parent: str = "palm"
    """输出到 `JointCfg` 中的 parent link 名。"""

    child: str | None = None
    """可选显式 child link 名。"""

    joint_type: str = "fixed"
    """当前 custom tip 首轮统一采用 `fixed` 关节。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """tip joint frame 相对 parent link frame 的位姿。"""

    axis: Vector3 = (0.0, 0.0, 0.0)
    """fixed joint 默认允许零轴输入。"""

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = None
    """fixed joint 默认没有限位。"""

    is_tip: bool = True
    """该 joint/link 是否应被标记为指尖相关。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加 metadata，会原样转发到结果 `JointCfg`。"""

    is_customized: bool = True
    """明确标记该 joint 使用的是 custom mesh，而不是 URDF primitive。"""

    def __post_init__(self):
        super().__post_init__()
        self.origin = _pose_from_value(self.origin)  # joint frame 先规约到标准 `PoseCfg`
        self.axis = _ensure_tuple(self.axis, length=3, field_name="custom_joint.axis")  # fixed joint 允许零轴
        if self.class_type in {None, JointBuilder}:
            self.class_type = CustomJointBuilder  # custom mesh 路线统一走 `CustomJointBuilder`


@dataclass
class CustomTipBuilderCfg(CustomJointBuilderCfg):
    r"""指尖专用的自定义 mesh 构建配置。

    当前首轮只实现四类显式锚点：

    - `round`
    - `wedge`
    - `leap_cube`
    - `thinner`

    它们共享同一个核心公式：

    $$
    p_{\text{joint}} = R\,S\,p_{\text{mesh}} + t
    $$

    其中：

    - $R$：canonical 朝向与用户附加 `rpy` 的合成旋转
    - $S$：`unit_scale` 与用户 `scale` 合成后的缩放
    - $p^\*$：mesh 局部坐标里的“底面中心锚点”
    - $t$：把锚点对齐到目标 joint frame 的平移

    更准确地说，平移是通过“让锚点落到 `mesh_offset.pos` 指定的位置”来求得：

    $$
    t = p_{\text{target}} - R\,S\,p^\*.
    $$
    """

    tip_type: str = "round"
    """指尖类型；当前支持 `round` / `wedge` / `leap_cube` / `thinner`。"""

    mesh_path: str | Path | None = None
    """可选显式 mesh 路径。

    若为 `None`，则按 `tip_type` 从当前项目内的 preset 表查默认文件。
    """

    mesh_offset: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """mesh 锚点目标位姿。

    这里不是“mesh 原点相对 joint frame 的直接位姿”，而是：
    `anchor_point` 在 joint frame 下希望落到哪里。

    当 `mesh_offset.pos=(0,0,0)` 时，语义是“底面中心锚点对齐到 tip joint 原点”。
    """

    scale: float | Sequence[float] = 1.0
    """用户级无量纲缩放。

    - 标量：uniform scale
    - 三元组：逐轴 non-uniform scale
    """

    unit_scale: float | None = None
    """mesh 文件单位到米制世界的基准缩放。

    例如 STL 以 mm 存储时，通常取 `0.001`。
    """

    anchor_point: Vector3 | Sequence[float] | None = None
    r"""mesh 局部坐标里的语义锚点 $p^\*$。"""

    base_rpy: Vector3 | Sequence[float] | None = None
    """canonical 朝向；默认由 tip preset 给出。"""

    _mesh_scale_xyz: Vector3 = field(init=False, default=(1.0, 1.0, 1.0))
    """最终写入 URDF `<mesh scale>` 的三轴缩放。"""

    def __post_init__(self):
        super().__post_init__()

        preset = _resolve_tip_preset(str(self.tip_type).lower())  # 先拿当前 tip 类型的默认锚点表
        self.tip_type = str(self.tip_type).lower()
        self.mesh_offset = _pose_from_value(self.mesh_offset)  # `p_target` 的位姿入口
        user_scale = _scale_to_vector(self.scale)  # 用户级无量纲 scale

        default_mesh_path = _CUSTOM_TIP_DIR / str(preset["file_name"])
        self.mesh_path = Path(self.mesh_path) if self.mesh_path is not None else default_mesh_path
        self.unit_scale = float(self.unit_scale if self.unit_scale is not None else preset["unit_scale"])
        if self.unit_scale <= 0.0:
            raise ValueError("unit_scale must be positive")

        self.anchor_point = _ensure_tuple(
            self.anchor_point if self.anchor_point is not None else preset["anchor_point"],
            length=3,
            field_name="custom_tip.anchor_point",
        )
        self.base_rpy = _ensure_tuple(
            self.base_rpy if self.base_rpy is not None else preset["base_rpy"],
            length=3,
            field_name="custom_tip.base_rpy",
        )

        # 真正进 URDF 的 mesh scale = 单位换算 × 用户级缩放。
        self._mesh_scale_xyz = tuple(self.unit_scale * component for component in user_scale)


class CustomJointBuilder(JointBuilder):
    r"""自定义 mesh 关节构建器。

    当前运行时只真正服务于 `CustomTipBuilderCfg`。之所以仍保留更一般的
    `CustomJointBuilder` 命名，是为了不把未来“custom palm / custom finger link”
    的扩展空间提前堵死。
    """

    cfg: CustomJointBuilderCfg

    def __init__(self, cfg: CustomJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""根据 `CustomTipBuilderCfg` 构建对应的 custom mesh tip。

        Returns:
            JointCfg: 以 joint-centric 形式表达的 custom mesh tip。

        Raises:
            NotImplementedError: 当前若传入的不是 `CustomTipBuilderCfg`，说明调用方
                试图把一般 custom joint 路线提前接进来；这不在本轮范围内。
        """

        if not isinstance(self.cfg, CustomTipBuilderCfg):
            raise NotImplementedError("CustomJointBuilder v1 currently only supports CustomTipBuilderCfg.")

        mesh_origin = self._build_mesh_origin()  # 先解出真正写入 visual/collision 的 mesh frame
        geometry = {"type": "mesh", "file_path": str(self.cfg.mesh_path), "scale": self.cfg._mesh_scale_xyz}
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_mesh_col",
                geometry=geometry,
                origin=mesh_origin,
            )
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_mesh_vis",
                geometry=geometry,
                origin=mesh_origin,
            )
        ]

        metadata = {
            **self.cfg.metadata,
            "custom_tip_type": self.cfg.tip_type,
            "mesh_path": str(self.cfg.mesh_path),
            "anchor_point": self.cfg.anchor_point,
            "mesh_scale": self.cfg._mesh_scale_xyz,
            "mesh_origin_rpy": mesh_origin.rpy,
        }
        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=None,  # custom mesh 的最终 `mass / inertial` 统一交给 physics closure
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=metadata,
        )

    def _build_mesh_origin(self) -> PoseCfg:
        r"""计算 mesh geometry 相对 tip joint frame 的最终位姿。

        设：

        - $p^\*$：mesh 局部坐标中的语义锚点
        - $S$：最终三轴缩放
        - $R$：canonical 朝向与附加 `rpy` 的组合旋转
        - $p_{\text{target}}$：锚点在 joint frame 下希望落到的位置

        则：

        $$
        t = p_{\text{target}} - R\,S\,p^\*.
        $$

        这正是你在 custom tip 草稿里一直强调的“对齐的是底面中心锚点，而不是
        mesh 原点本身”的语义。
        """

        assert isinstance(self.cfg, CustomTipBuilderCfg)
        total_rpy = _add_rpy(self.cfg.base_rpy, self.cfg.mesh_offset.rpy)  # 当前仍采用小角度/声明式的分量叠加
        scaled_anchor = (
            self.cfg.anchor_point[0] * self.cfg._mesh_scale_xyz[0],
            self.cfg.anchor_point[1] * self.cfg._mesh_scale_xyz[1],
            self.cfg.anchor_point[2] * self.cfg._mesh_scale_xyz[2],
        )
        rotated_anchor = _apply_rotation(_rpy_rotation_matrix(total_rpy), scaled_anchor)
        return PoseCfg(
            pos=(
                self.cfg.mesh_offset.pos[0] - rotated_anchor[0],
                self.cfg.mesh_offset.pos[1] - rotated_anchor[1],
                self.cfg.mesh_offset.pos[2] - rotated_anchor[2],
            ),
            rpy=total_rpy,
        )


__all__ = [
    "THUMB_FUNCTIONAL_TIP_PHASE_RPY",
    "CustomJointBuilderCfg",
    "CustomTipBuilderCfg",
    "CustomJointBuilder",
    "apply_thumb_functional_tip_phase",
]
