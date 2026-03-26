"""手部资产生成的 builder 侧运行时对象。

本模块是纯 schema 之上的第一层运行时。当前阶段最重要的设计决定是
把公开的 runtime 层保持得尽量浅：

- schema 文件回答“手资产是什么”；
- builder 文件回答“一个具体的 `HandCfg` 怎么被组装出来”。

我们刻意没有立刻展开成完整的 ``JointBuilder`` / ``FingerBuilder`` /
``PalmBuilder`` 类族。原因是当前研究仍在迭代 joint-level 显式参数的
数学含义，代码结构必须保留足够的可塑性，避免过早抽象把后续修改
成本抬高。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Literal, cast

from .asset_schema_core import (
    AssetCfgBase,
    BoxGeometryCfg,
    CollisionGeometryCfg,
    CylinderGeometryCfg,
    InertiaTensorCfg,
    Handedness,
    InertialCfg,
    PoseCfg,
    SphereGeometryCfg,
)
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

HandRule = Callable[[HandCfg], None]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> Mat3:
    r"""计算 URDF RPY 角对应的旋转矩阵。

    Args:
        rpy (tuple[float, float, float]): URDF 的 roll-pitch-yaw 角。

    Returns:
        Mat3: 按 URDF 外禀 RPY 约定得到的旋转矩阵。

    Notes:
        当 primitive collision 相对 link frame 存在局部旋转时，builder
        会使用这个 helper。像 box / cylinder 这类惯量公式，通常先在
        primitive 自身主轴系里写成 $I_c$，再旋转到 link frame：
        $$
        I = R I_c R^\top .
        $$
    """

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _transpose(matrix: Mat3) -> Mat3:
    r"""返回一个 $3 \\times 3$ 矩阵的转置。"""

    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(left: Mat3, right: Mat3) -> Mat3:
    r"""计算两个 $3 \\times 3$ 矩阵的乘积。"""

    return (
        (
            sum(left[0][k] * right[k][0] for k in range(3)),
            sum(left[0][k] * right[k][1] for k in range(3)),
            sum(left[0][k] * right[k][2] for k in range(3)),
        ),
        (
            sum(left[1][k] * right[k][0] for k in range(3)),
            sum(left[1][k] * right[k][1] for k in range(3)),
            sum(left[1][k] * right[k][2] for k in range(3)),
        ),
        (
            sum(left[2][k] * right[k][0] for k in range(3)),
            sum(left[2][k] * right[k][1] for k in range(3)),
            sum(left[2][k] * right[k][2] for k in range(3)),
        ),
    )


def _matrix_add(left: Mat3, right: Mat3) -> Mat3:
    r"""逐元素相加两个 $3 \\times 3$ 矩阵。"""

    return (
        (left[0][0] + right[0][0], left[0][1] + right[0][1], left[0][2] + right[0][2]),
        (left[1][0] + right[1][0], left[1][1] + right[1][1], left[1][2] + right[1][2]),
        (left[2][0] + right[2][0], left[2][1] + right[2][1], left[2][2] + right[2][2]),
    )


def _matrix_scale(matrix: Mat3, scale: float) -> Mat3:
    r"""用标量缩放一个 $3 \\times 3$ 矩阵。"""

    return (
        (scale * matrix[0][0], scale * matrix[0][1], scale * matrix[0][2]),
        (scale * matrix[1][0], scale * matrix[1][1], scale * matrix[1][2]),
        (scale * matrix[2][0], scale * matrix[2][1], scale * matrix[2][2]),
    )


def _outer(vec: tuple[float, float, float]) -> Mat3:
    r"""计算三维向量的外积 $v v^\top$。"""

    return (
        (vec[0] * vec[0], vec[0] * vec[1], vec[0] * vec[2]),
        (vec[1] * vec[0], vec[1] * vec[1], vec[1] * vec[2]),
        (vec[2] * vec[0], vec[2] * vec[1], vec[2] * vec[2]),
    )


def _identity3() -> Mat3:
    r"""返回 $3 \\times 3$ 单位矩阵。"""

    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _diag(ixx: float, iyy: float, izz: float) -> Mat3:
    r"""构造一个对角惯量矩阵。"""

    return ((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz))


def _inertia_tensor_from_matrix(matrix: Mat3) -> InertiaTensorCfg:
    r"""把稠密对称惯量矩阵转换成 URDF 风格的 6 参数表示。"""

    return InertiaTensorCfg(
        ixx=matrix[0][0],
        iyy=matrix[1][1],
        izz=matrix[2][2],
        ixy=matrix[0][1],
        ixz=matrix[0][2],
        iyz=matrix[1][2],
    )


def _primitive_mass_and_inertia_at_centroid(
    collision: CollisionGeometryCfg,
    density: float,
) -> tuple[float, Mat3]:
    r"""在 primitive 自身局部坐标系中计算质量与质心惯量。

    Args:
        collision (CollisionGeometryCfg): primitive collision 描述。
        density (float): 假定使用的均匀密度。

    Returns:
        tuple[float, Mat3]: primitive 质量与质心处的惯量矩阵。

    Raises:
        TypeError: 当 collision 几何不是当前支持的 primitive 时抛出。

    Notes:
        builder v1 只从 primitive collision 反推惯性描述。这是刻意的：
        对 ``box`` / ``cylinder`` / ``sphere`` 我们有解析公式，
        而 mesh 体积分会显著增加复杂度，不适合放在这个仍在快速迭代的
        研究阶段。
    """

    geometry = collision.geometry
    if isinstance(geometry, BoxGeometryCfg):
        inertial = InertialCfg.from_box(geometry.size, density=density, inertia_padding=0.0)
    elif isinstance(geometry, CylinderGeometryCfg):
        inertial = InertialCfg.from_cylinder(
            geometry.radius,
            geometry.length,
            density=density,
            principal_axis="z",
            inertia_padding=0.0,
        )
    elif isinstance(geometry, SphereGeometryCfg):
        inertial = InertialCfg.from_sphere(geometry.radius, density=density, inertia_padding=0.0)
    else:
        raise TypeError("aggregate_primitive_inertial 在 v1 只支持 primitive collision 几何")

    tensor = cast(InertiaTensorCfg, inertial.inertia)
    return inertial.mass, _diag(tensor.ixx, tensor.iyy, tensor.izz)


def aggregate_primitive_inertial(
    collisions: Sequence[CollisionGeometryCfg],
    *,
    density: float,
    inertia_padding: float = 1e-8,
) -> InertialCfg:
    r"""把多个 primitive collision 聚合成一个 link 级惯性描述。

    这一层就是前面 joint-level / link-level 显式参数方案在 builder 侧的
    物理对应物：多个 primitive collision 最终会被折叠为一个 URDF
    inertial block，通过合成质心和惯量张量来描述整个 link。

    Args:
        collisions (Sequence[CollisionGeometryCfg]): primitive collision 元素。
        density (float): 用于 primitive 质量近似的共享密度。
        inertia_padding (float): 数值稳定性所需的对角项 padding。

    Returns:
        InertialCfg: 聚合后的惯性描述。

    Raises:
        ValueError: 当没有 collision 输入时抛出。
        TypeError: 当某个 collision 使用了非 primitive 几何时抛出。
    """

    if not collisions:
        raise ValueError("aggregate_primitive_inertial requires at least one collision element")
    if density <= 0.0:
        raise ValueError("density must be positive")

    # 每个 primitive 会贡献三项：
    # 1. 自身质量；
    # 2. 自身质心在 link frame 下的位置；
    # 3. 已经旋转到 link frame 的质心惯量张量。
    #
    # 我们先把这些项收集起来，再求复合质心，最后用平行轴定理把每个
    # 各个 primitive 的惯量统一平移到同一个参考点上。
    primitive_terms: list[tuple[float, tuple[float, float, float], Mat3]] = []
    total_mass = 0.0
    weighted_pos = [0.0, 0.0, 0.0]
    for collision in collisions:
        mass_i, inertia_local = _primitive_mass_and_inertia_at_centroid(collision, density)
        origin = PoseCfg.from_value(collision.origin)
        rotation = _rotation_matrix_from_rpy(origin.rpy)
        # 将 primitive 主轴系下的质心惯量旋转到父 link 坐标系：
        #
        # $$
        # $I_{link, c} = R I_c R^\top$
        # $$
        inertia_rotated = _matmul(_matmul(rotation, inertia_local), _transpose(rotation))
        pos = origin.pos
        primitive_terms.append((mass_i, pos, inertia_rotated))
        total_mass += mass_i
        weighted_pos[0] += mass_i * pos[0]
        weighted_pos[1] += mass_i * pos[1]
        weighted_pos[2] += mass_i * pos[2]

    # 复合质心：
    #
    # $$
    # $c = \frac{\sum_i m_i p_i}{\sum_i m_i}$。
    # $$
    com = (weighted_pos[0] / total_mass, weighted_pos[1] / total_mass, weighted_pos[2] / total_mass)
    inertia_about_com = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    identity = _identity3()
    for mass_i, pos, inertia_rotated in primitive_terms:
        rel = (pos[0] - com[0], pos[1] - com[1], pos[2] - com[2])
        rel_sq = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]
        # 平行轴定理：
        #
        # $$
        # $I_{com} = I_{link, c}
        # + m \left( \|r\|^2 \mathbf{I} - r r^\top \right)$。
        # $$
        #
        # 其中 $r$ 是 primitive 质心到复合质心的偏移。
        parallel_axis = _matrix_scale(
            _matrix_add(_matrix_scale(identity, rel_sq), _matrix_scale(_outer(rel), -1.0)),
            mass_i,
        )
        inertia_about_com = _matrix_add(inertia_about_com, _matrix_add(inertia_rotated, parallel_axis))

    return InertialCfg(
        mass=total_mass,
        origin=PoseCfg(pos=com),
        inertia=_inertia_tensor_from_matrix(inertia_about_com),
        inertia_padding=inertia_padding,
    )


@dataclass
class BuilderCfg(AssetCfgBase):
    r"""构建器运行时对象的基础配置。

    schema 层回答的是“什么是合法的资产描述”；builder 配置回答的是
    “在这一次 pipeline 调用里，运行时 builder 应该怎样组装这个描述”。
    """

    class_type: type["Builder"] | None = None
    """关联的 builder 运行时类。"""


class Builder:
    r"""资产构建器的基础运行时对象。

    子类应该保持为轻量的运行时协调器，不要重复定义 schema。它们只负责
    消费 schema 对象，并在生成时补齐那些更适合在运行时计算的部分。
    """

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg

    def build(self) -> HandCfg:
        r"""组装并返回一个 :class:`HandCfg` 实例。"""

        raise NotImplementedError


@dataclass
class HandBuilderCfg(BuilderCfg):
    r"""生成器 v1 中的整手组装顶层配置。

    这个配置把当前真正重要的手组装粒度固定住：

    - 一个 palm 输入；
    - 一组 finger 输入；
    - 可选的惯性反推；
    - 轻量 metadata 透传。

    我们刻意止步于此，而不是提前引入更深的运行时类层级。原因是用户
    仍在探索 joint-level 显式参数到 URDF 几何描述的映射，运行时边界
    必须保持足够容易重构。
    """

    class_type: type["Builder"] | None = None
    """关联的 hand builder 运行时类。"""

    hand_name: str = "generated_hand"
    """生成的 `HandCfg` 名称。"""

    family: str = "generic"
    """生成的 `HandCfg` family 标签。"""

    handedness: Handedness = "unknown"
    """生成的 `HandCfg` handedness 标签。"""

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    """手掌配置输入或其映射形式。"""

    fingers: list[FingerCfg | Mapping[str, Any]] = field(default_factory=list)
    """手指配置输入或其映射形式。"""

    auto_compute_missing_inertial: bool = False
    """是否根据 primitive collision 自动补全缺失的惯性项。"""

    default_density: float = 500.0
    """在反推惯性时使用的默认密度。"""

    inertia_padding: float = 1e-8
    """反推惯性时使用的对角项 padding。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留给 builder 的 metadata。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandBuilder


class HandBuilder(Builder):
    r"""整手 builder 的顶层运行时对象。

    v1 故意把 runtime 层做得很浅：
    只有 `HandBuilder` 是公开对象，而 joint / finger / palm 的处理都保留
    为内部组装步骤。这样做的目的，是让 joint-level 算法在迭代时不被
    过深的类层级拖慢。
    """

    cfg: HandBuilderCfg

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)

    def _coerce_palm(self) -> PalmCfg:
        r"""把 palm 输入规范化成 canonical :class:`PalmCfg`。

        Returns:
            PalmCfg: 规范化后的 palm schema 对象。
        """

        return self.cfg.palm if isinstance(self.cfg.palm, PalmCfg) else PalmCfg(**self.cfg.palm)

    def _coerce_fingers(self) -> list[FingerCfg]:
        r"""把每个 finger 输入规范化成 canonical :class:`FingerCfg`。

        Returns:
            list[FingerCfg]: 规范化后的 finger schema 列表。
        """

        return [finger if isinstance(finger, FingerCfg) else FingerCfg(**finger) for finger in self.cfg.fingers]

    def _fill_missing_inertial(self, hand: HandCfg) -> None:
        r"""在需要时根据 primitive collision 补全缺失的惯性项。

        Args:
            hand (HandCfg): 需要原地修改的 hand schema。

        Notes:
            builder v1 只补“缺失”的惯性项。如果调用者已经手工提供了
            惯性描述，我们把它视为有意设计，不会覆盖。
        """

        palm = cast(PalmCfg, hand.palm)
        if palm.inertial is None and palm.collisions:
            # 掌部在这里被视作一个刚性整体，因此其 primitive collision
            # 会被聚合成一个单独的 URDF inertial block。
            palm.inertial = aggregate_primitive_inertial(
                palm.collisions,
                density=self.cfg.default_density,
                inertia_padding=self.cfg.inertia_padding,
            )

        for finger in hand.fingers:
            for joint in finger.joints:
                if joint.inertial is None and joint.collisions:
                    # 在 v1 中，每个 joint 都拥有其后续 child link 的
                    # 子 link 的 collision 布局已经给定，因此这里直接由这些元素
                    # 反推出对应的子 link inertial。
                    joint.inertial = aggregate_primitive_inertial(
                        joint.collisions,
                        density=self.cfg.default_density,
                        inertia_padding=self.cfg.inertia_padding,
                    )

    def build(self) -> HandCfg:
        r"""由配置好的 palm / finger 输入组装出一只手。

        Returns:
            HandCfg: canonical hand schema 对象，可选地补充惯性描述。
        """

        palm = self._coerce_palm()
        fingers = self._coerce_fingers()
        # 构建器的职责不是再发明一套 runtime-only representation。
        # 它要做的是把最终 schema 对象实例化出来，供后续 validator 和
        # 由导出器阶段直接消费。
        hand = HandCfg(
            name=self.cfg.hand_name,
            palm=palm,
            fingers=fingers,
            family=self.cfg.family,
            handedness=self.cfg.handedness,
            metadata=self.cfg.metadata.copy(),
        )
        if self.cfg.auto_compute_missing_inertial:
            # 惯性反推是显式 opt-in 的，因为在科研迭代阶段，用户往往
            # 需要对比“手工惯性”与“builder 近似惯性”的差异。
            self._fill_missing_inertial(hand)
        return hand


__all__ = [
    "BuilderCfg",
    "Builder",
    "HandBuilderCfg",
    "HandBuilder",
    "HandRule",
    "aggregate_primitive_inertial",
]
