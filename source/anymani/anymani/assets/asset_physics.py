"""手资产物理闭包：由最终 collision 几何重建 link 级质量与惯量。

本模块服务的是 AnyMani 资产生产链里一个非常具体、但又不能继续拖延的科研问题：

1. pre-made 与 post-mutate 会改变 link 几何、tip mesh、甚至 joint-child 的组织方式；
2. 这些变化若只改 visual / collision 而不改 `mass / inertial`，URDF 在动力学层面就是失配的；
3. `mass / inertial` 必须以**最终 collision 几何**为准，在导出前做一次闭包。

当前 v1 的闭包策略是：

- primitive collision：直接用解析体积与解析惯量；
- custom mesh collision：默认用 `trimesh` 的真实 polyhedral mass properties；
- 多几何 link：在 link frame 下做质量加权质心 + 平行轴定理合并；
- non-uniform mesh scale：当前显式 fail-hard，不偷做不透明近似。

这层故意放在 `assets/` 顶层，而不是塞进 builder / mutator / exporter 的任意一边，
原因是它既不属于“造几何”，也不属于“写文件”，而是一个横跨 pre-made 与 post-mutate 的
统一物理规范钩子。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .asset_base import AssetCfgBase, HandCfg, InertialCfg, JointCfg, PalmCfg, PoseCfg
from .asset_schema_core import CollisionGeometryCfg, InertiaTensorCfg, MeshGeometryCfg

_FLOAT_TOLERANCE = 1e-12
"""物理闭包里统一使用的近零容差。"""


_ASSETS_ROOT = Path(__file__).resolve().parent
"""assets/ 子项目根目录。

若 sidecar 中保存的是相对 mesh 路径，则这里作为第二层解析兜底。
"""


@dataclass
class DensityProfileCfg(AssetCfgBase):
    r"""按手部部位划分的均匀密度配置。

    当前先采用最朴素、也最容易解释的科研假设：

    - palm、finger regular link、fingertip、custom tip 各自是均匀材料；
    - 若某一层未显式给值，则回退到 `default`；
    - 不在这一层引入更细的 slot / family / mesh-path 特化，避免过早把策略写死。

    单位统一为 kg/m^3。
    """

    default: float = 650.0
    """全局默认密度 $\rho$ [kg/m^3]。"""

    palm: float | None = None
    """掌部专用密度；`None` 表示回退到 `default`。"""

    finger_link: float | None = None
    """普通 finger link 的专用密度；`None` 表示回退到 `default`。"""

    fingertip: float | None = None
    """primitive fingertip 的专用密度；`None` 表示回退到 `default`。"""

    custom_tip: float | None = None
    """custom mesh fingertip 的专用密度；`None` 表示回退到 `fingertip/default`。"""

    def __post_init__(self) -> None:
        self.default = _coerce_positive_density(self.default, field_name="density.default")
        self.palm = _coerce_optional_density(self.palm, field_name="density.palm")
        self.finger_link = _coerce_optional_density(self.finger_link, field_name="density.finger_link")
        self.fingertip = _coerce_optional_density(self.fingertip, field_name="density.fingertip")
        self.custom_tip = _coerce_optional_density(self.custom_tip, field_name="density.custom_tip")

    def for_palm(self) -> float:
        r"""返回 palm 闭包应使用的密度。"""

        return self.palm if self.palm is not None else self.default

    def for_joint(self, joint: JointCfg) -> float:
        r"""返回某个 joint child-link 闭包应使用的密度。"""

        has_mesh_collision = any(collision.geometry.kind == "mesh" for collision in joint.collisions)
        if joint.is_tip and has_mesh_collision and _is_procedural_cs_tip_joint(joint):
            return self.fingertip if self.fingertip is not None else self.default
        if joint.is_tip and has_mesh_collision:
            if self.custom_tip is not None:
                return self.custom_tip
            if self.fingertip is not None:
                return self.fingertip
            return self.default
        if joint.is_tip:
            return self.fingertip if self.fingertip is not None else self.default
        return self.finger_link if self.finger_link is not None else self.default


@dataclass
class AssetPhysicsCfg(AssetCfgBase):
    r"""手资产物理闭包配置。

    这层配置只关心“最终 link 刚体参数怎么闭合”，不接管 builder / mutate 的几何语义。
    因此字段刻意保持克制，只保留：

    - 是否启用；
    - 均匀密度假设；
    - 极小几何的数值稳定下限；
    - mesh 质量属性后端；
    - non-uniform mesh scale 的处理策略。
    """

    class_type: type[AssetPhysicsClosure] | None = None
    """关联的运行时类。"""

    enabled: bool = True
    """是否启用物理闭包。"""

    density: DensityProfileCfg | dict[str, Any] = field(default_factory=DensityProfileCfg)
    """分部位密度配置。"""

    min_mass: float = 1e-6
    """单个 collision contribution 的质量下限，用于极小几何数值稳定。"""

    inertia_padding: float = 0.0
    """最终 `InertialCfg` 对角项工程性 padding。"""

    mesh_backend: Literal["trimesh"] = "trimesh"
    """mesh collision 质量属性后端。

    当前只落地 `trimesh`，因为 custom tip 与 materialized `cs` 都需要真实
    polyhedral volume / center-of-mass / inertia，而 validator 已经共享 mesh volume
    相关前置条件。
    """

    nonuniform_mesh_scale_policy: Literal["fail"] = "fail"
    """non-uniform mesh scale 的处理策略。当前只允许显式失败。"""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = AssetPhysicsClosure
        if not isinstance(self.density, DensityProfileCfg):
            self.density = DensityProfileCfg(**dict(self.density))
        self.min_mass = float(self.min_mass)
        if self.min_mass <= 0.0:
            raise ValueError("AssetPhysicsCfg.min_mass must be positive")
        self.inertia_padding = float(self.inertia_padding)
        if self.inertia_padding < 0.0:
            raise ValueError("AssetPhysicsCfg.inertia_padding must be >= 0")
        if self.mesh_backend != "trimesh":
            raise ValueError(f"Unsupported mesh_backend: {self.mesh_backend!r}")
        if self.nonuniform_mesh_scale_policy != "fail":
            raise ValueError(
                "Unsupported nonuniform_mesh_scale_policy: "
                f"{self.nonuniform_mesh_scale_policy!r}"
            )


@dataclass
class _MassContribution:
    r"""单个 collision geometry 对 link 刚体的质量贡献。

    所有量都已经被表达在 **link frame** 下：

    - `center_of_mass`：该几何自身的质心位置；
    - `inertia_about_com`：绕该几何自身质心、但轴已经旋到 link frame 的惯量张量；
    - `backend`：当前贡献来自解析 primitive 还是 `trimesh`。
    """

    mass: float
    """该几何贡献的刚体质量 $m_i$。"""

    center_of_mass: tuple[float, float, float]
    """该几何质心在 link frame 下的位置。"""

    inertia_about_com: np.ndarray
    """绕该几何质心、表达在 link frame 下的 $3\times3$ 惯量张量。"""

    backend: Literal["analytic", "trimesh"]
    """该质量属性的求解后端。"""


@dataclass
class _CanonicalMeshMassProperties:
    r"""缓存中的 canonical mesh 质量属性。

    这里的“canonical”特指：

    - 使用 mesh 文件原始坐标；
    - 不施加 geometry origin；
    - 不施加任何 user scale；
    - 取单位密度 $\rho=1$。

    因此后续 uniform scale $s$ 与目标密度 $\rho$ 的样本，只需套：

    $$
    m = \rho V_0 s^3,\qquad
    \mathbf{c} = s\mathbf{c}_0,\qquad
    \mathbf{I}_C = \rho s^5 \mathbf{I}_{0,C}.
    $$
    """

    volume: float
    """原始 mesh 在文件坐标系下的体积 $V_0$。"""

    center_of_mass: tuple[float, float, float]
    r"""原始 mesh 的体积质心 $\mathbf{c}_0$。"""

    inertia_about_com: np.ndarray
    r"""单位密度下、绕原始 mesh 质心的惯量张量 $\mathbf{I}_{0,C}$。"""


class AssetPhysicsClosure:
    r"""物理闭包运行时壳。

    运行时职责很单纯：对一份最终 `HandCfg` 做纯函数式闭包，输出一份新的 `HandCfg`，
    其 `PalmCfg.inertial` 与所有 `JointCfg.inertial` 都与最终 collision 几何一致。
    """

    cfg: AssetPhysicsCfg

    def __init__(self, cfg: AssetPhysicsCfg):
        self.cfg = cfg

    def close(self, target: HandCfg, *, stage: str | None = None) -> HandCfg:
        r"""基于最终 collision 几何闭合整手的刚体参数。

        Args:
            target (HandCfg): 待闭包的手资产。
            stage (str | None): 当前闭包发生在哪个阶段，例如 `pre_made` /
                `post_mutate`。该字段只写 metadata，不影响数值结果。

        Returns:
            HandCfg: 闭包后的新 `HandCfg` 副本。
        """

        if not self.cfg.enabled:
            return target.copy()

        closed_hand = target.copy()

        # palm 是单独的根 link；它没有 joint 包裹，因此直接在 `PalmCfg` 层闭包。
        closed_hand.palm = self._close_palm(closed_hand.palm, stage=stage)

        # 每个 `JointCfg` 在当前 schema 里都携带一个 child link embodiment，
        # 所以真正要闭包的是 `joint.collisions -> joint.inertial` 这一对。
        closed_fingers = []
        for finger in closed_hand.fingers:
            closed_joints = [self._close_joint_child_link(joint, stage=stage) for joint in finger.joints]
            closed_fingers.append(finger.replace(joints=closed_joints))
        closed_hand.fingers = closed_fingers

        # 顶层 metadata 只记录“这只手已经被 physics closure 处理过”，不复制每个 link 的细节。
        hand_metadata = dict(closed_hand.metadata)
        hand_metadata["physics_closure"] = {
            "enabled": True,
            "stage": stage or "unspecified",
            "mesh_backend": self.cfg.mesh_backend,
            "nonuniform_mesh_scale_policy": self.cfg.nonuniform_mesh_scale_policy,
        }
        closed_hand.metadata = hand_metadata
        return closed_hand

    def _close_palm(self, target: PalmCfg, *, stage: str | None) -> PalmCfg:
        r"""闭合 palm root-link 的惯性参数。"""

        closed_inertial, backend = _aggregate_collision_inertial(
            target.collisions,
            density=self.cfg.density.for_palm(),
            min_mass=self.cfg.min_mass,
            inertia_padding=self.cfg.inertia_padding,
            mesh_backend=self.cfg.mesh_backend,
            nonuniform_mesh_scale_policy=self.cfg.nonuniform_mesh_scale_policy,
        )
        if closed_inertial is None:
            return target

        metadata = dict(target.metadata)
        metadata["inertial_source"] = "collision_closure_v1"
        metadata["inertial_backend"] = backend
        metadata["inertial_stage"] = stage or "unspecified"
        return target.replace(inertial=closed_inertial, metadata=metadata)

    def _close_joint_child_link(self, target: JointCfg, *, stage: str | None) -> JointCfg:
        r"""闭合某个 joint 所携带 child link 的惯性参数。"""

        closed_inertial, backend = _aggregate_collision_inertial(
            target.collisions,
            density=self.cfg.density.for_joint(target),
            min_mass=self.cfg.min_mass,
            inertia_padding=self.cfg.inertia_padding,
            mesh_backend=self.cfg.mesh_backend,
            nonuniform_mesh_scale_policy=self.cfg.nonuniform_mesh_scale_policy,
        )
        if closed_inertial is None:
            return target

        metadata = dict(target.metadata)
        metadata["inertial_source"] = "collision_closure_v1"
        metadata["inertial_backend"] = backend
        metadata["inertial_stage"] = stage or "unspecified"
        return target.replace(inertial=closed_inertial, metadata=metadata)


def close_hand_physics(target: HandCfg, cfg: AssetPhysicsCfg | None, *, stage: str | None = None) -> HandCfg:
    r"""对外暴露的轻量 helper：按配置闭合整手物理参数。"""

    if cfg is None:
        return target.copy()
    return AssetPhysicsClosure(cfg).close(target, stage=stage)


def _is_procedural_cs_tip_joint(joint: JointCfg) -> bool:
    r"""判断一个 mesh fingertip 是否仍属于 procedural `cs` 密度通道。

    `cs` 从 two-primitive schema 改成 single mesh schema 后，几何后端确实变成
    `trimesh`，但材料语义并没有变成 custom fingertip：它仍是由半径 $r$ 与高度
    $h$ 参数化的默认 fingertip primitive。因此密度应走 `DensityProfileCfg.fingertip`，
    而不是 `custom_tip`。
    """

    metadata = dict(joint.metadata)  # joint metadata 是 builder / materializer 传递 tip provenance 的最小证书
    return (
        metadata.get("tip_type") == "cs"
        or metadata.get("procedural_tip_type") == "cs"
        or metadata.get("procedural_mesh_kind") == "cs_tip"
    )


def _aggregate_collision_inertial(
    collisions: list[CollisionGeometryCfg],
    *,
    density: float,
    min_mass: float,
    inertia_padding: float,
    mesh_backend: Literal["trimesh"],
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> tuple[InertialCfg | None, str]:
    r"""把一组 collision geometry 合并成一个 link 级 `InertialCfg`。

    Returns:
        tuple[InertialCfg | None, str]:
            1. 闭包后的 `InertialCfg`；若当前 link 没有 collision，则返回 `None`；
            2. backend 证书：`analytic` / `trimesh` / `mixed` / `none`。
    """

    if not collisions:
        return None, "none"

    contributions = [
        _mass_contribution_from_collision(
            collision,
            density=density,
            min_mass=min_mass,
            mesh_backend=mesh_backend,
            nonuniform_mesh_scale_policy=nonuniform_mesh_scale_policy,
        )
        for collision in collisions
    ]

    total_mass = sum(item.mass for item in contributions)
    if total_mass <= 0.0:
        raise ValueError("physics closure got non-positive total mass from collision geometry")

    # 先在 link frame 下求合成质心：
    # $$
    # \mathbf{c} = \frac{1}{M}\sum_i m_i \mathbf{c}_i.
    # $$
    total_com = np.zeros(3, dtype=np.float64)
    for item in contributions:
        total_com += item.mass * np.asarray(item.center_of_mass, dtype=np.float64)
    total_com /= total_mass

    # 再把每个子刚体的惯量搬到合成质心：
    # $$
    # \mathbf{I}_C = \sum_i \left(\mathbf{I}_{i,C_i} + m_i[(\mathbf{d}_i^\top \mathbf{d}_i)\mathbf{E}-\mathbf{d}_i\mathbf{d}_i^\top]\right).
    # $$
    total_inertia = np.zeros((3, 3), dtype=np.float64)
    for item in contributions:
        delta = np.asarray(item.center_of_mass, dtype=np.float64) - total_com
        total_inertia += item.inertia_about_com + item.mass * _parallel_axis_matrix(delta)

    backend_names = {item.backend for item in contributions}
    backend = "mixed" if len(backend_names) > 1 else next(iter(backend_names))

    inertial = InertialCfg(
        mass=total_mass,
        origin=PoseCfg(pos=(float(total_com[0]), float(total_com[1]), float(total_com[2]))),
        inertia=InertiaTensorCfg(
            ixx=float(total_inertia[0, 0]),
            iyy=float(total_inertia[1, 1]),
            izz=float(total_inertia[2, 2]),
            ixy=float(total_inertia[0, 1]),
            ixz=float(total_inertia[0, 2]),
            iyz=float(total_inertia[1, 2]),
        ),
        inertia_padding=inertia_padding,
    )
    return inertial, backend


def _mass_contribution_from_collision(
    collision: CollisionGeometryCfg,
    *,
    density: float,
    min_mass: float,
    mesh_backend: Literal["trimesh"],
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> _MassContribution:
    r"""把单个 collision geometry lower 成 link frame 下的质量贡献。"""

    geometry = collision.geometry
    rotation = _rotation_matrix(collision.origin.rpy)

    if geometry.kind == "box":
        sx, sy, sz = (float(geometry.size[0]), float(geometry.size[1]), float(geometry.size[2]))
        volume = sx * sy * sz
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_box((sx, sy, sz), density=mass / volume, min_mass=mass).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "cylinder":
        radius = float(geometry.radius)
        length = float(geometry.length)
        volume = math.pi * radius * radius * length
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_cylinder(
            radius,
            length,
            density=mass / volume,
            principal_axis="z",
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "elliptic_cylinder":
        radius_x = float(geometry.radius_x)
        radius_z = float(geometry.radius_z)
        length = float(geometry.length)
        volume = math.pi * radius_x * radius_z * length
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_elliptic_cylinder(
            radius_x,
            radius_z,
            length,
            density=mass / volume,
            principal_axis="y",
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "sphere":
        radius = float(geometry.radius)
        volume = 4.0 * math.pi * radius**3 / 3.0
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_sphere(
            radius,
            density=mass / volume,
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind != "mesh":
        raise ValueError(f"Unsupported collision geometry for physics closure: {geometry.kind!r}")

    if mesh_backend != "trimesh":
        raise ValueError(f"Unsupported mesh backend for physics closure: {mesh_backend!r}")

    uniform_scale = _extract_uniform_scale(
        geometry,
        nonuniform_mesh_scale_policy=nonuniform_mesh_scale_policy,
    )
    canonical = _canonical_mesh_mass_properties(_mesh_cache_key(geometry.file_path))

    # 对 uniform scale 的 mesh，不需要每次重跑体积分；直接套缩放律即可。
    scaled_volume = canonical.volume * uniform_scale**3
    mass = max(density * scaled_volume, min_mass)
    scaled_center = tuple(component * uniform_scale for component in canonical.center_of_mass)
    rotated_center = _apply_rotation(rotation, scaled_center)
    center_of_mass = (
        collision.origin.pos[0] + rotated_center[0],
        collision.origin.pos[1] + rotated_center[1],
        collision.origin.pos[2] + rotated_center[2],
    )
    scaled_inertia = canonical.inertia_about_com * (density * uniform_scale**5)
    inertia_about_com = _rotate_inertia(scaled_inertia, rotation)
    return _MassContribution(
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_about_com=inertia_about_com,
        backend="trimesh",
    )


def _coerce_positive_density(value: Any, *, field_name: str) -> float:
    r"""把密度字段规约为正浮点数。"""

    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def _coerce_optional_density(value: Any, *, field_name: str) -> float | None:
    r"""把可选密度字段规约为 `float | None`。"""

    if value is None:
        return None
    return _coerce_positive_density(value, field_name=field_name)


def _rotation_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    r"""返回 URDF 固定轴 RPY 对应的旋转矩阵 $R_z R_y R_x$。"""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _apply_rotation(rotation: np.ndarray, point: tuple[float, float, float]) -> tuple[float, float, float]:
    r"""计算 $R\mathbf{x}$，并返回 Python tuple。"""

    rotated = rotation @ np.asarray(point, dtype=np.float64)
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _rotate_inertia(inertia: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    r"""把局部惯量张量旋到 link frame：$\mathbf{I}'=R\mathbf{I}R^\top$。"""

    return rotation @ inertia @ rotation.T


def _parallel_axis_matrix(delta: np.ndarray) -> np.ndarray:
    r"""返回平行轴定理里的几何矩阵项。

    对位移 $\mathbf{d}$，平行轴矩阵为：
    $$
    (\mathbf{d}^\top\mathbf{d})\mathbf{E}-\mathbf{d}\mathbf{d}^\top.
    $$
    """

    distance_sq = float(delta @ delta)
    return distance_sq * np.eye(3, dtype=np.float64) - np.outer(delta, delta)


def _inertia_cfg_to_matrix(inertia: InertiaTensorCfg) -> np.ndarray:
    r"""把 `InertiaTensorCfg` 展开成对称 $3\times3$ 矩阵。"""

    return np.asarray(
        (
            (inertia.ixx, inertia.ixy, inertia.ixz),
            (inertia.ixy, inertia.iyy, inertia.iyz),
            (inertia.ixz, inertia.iyz, inertia.izz),
        ),
        dtype=np.float64,
    )


def _extract_uniform_scale(
    geometry: MeshGeometryCfg,
    *,
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> float:
    r"""从 mesh scale 中抽出 uniform scale。

    当前 contract 已经明确：physics closure v1 不处理 non-uniform mesh scale。
    若三轴不等，则按用户约定 fail-hard，避免把未验证近似混进动力学真值。
    """

    sx, sy, sz = (float(geometry.scale[0]), float(geometry.scale[1]), float(geometry.scale[2]))
    if math.isclose(sx, sy, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE) and math.isclose(
        sx,
        sz,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        return sx
    if nonuniform_mesh_scale_policy == "fail":
        raise ValueError(
            "physics closure only supports uniform mesh scale in v1, "
            f"got {geometry.scale!r} for {geometry.file_path!r}"
        )
    raise ValueError(f"Unsupported nonuniform_mesh_scale_policy: {nonuniform_mesh_scale_policy!r}")


def _mesh_cache_key(file_path: str) -> tuple[str, int, int]:
    r"""构造 mesh canonical mass properties 的缓存 key。

    key 使用：

    - 解析后的绝对路径；
    - 文件大小；
    - mtime_ns。

    这样既能让同一轮批量生成复用缓存，也能在 mesh 文件被替换后自动失效。
    """

    resolved = _resolve_mesh_path(file_path)
    stat = resolved.stat()
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


@lru_cache(maxsize=128)
def _canonical_mesh_mass_properties(cache_key: tuple[str, int, int]) -> _CanonicalMeshMassProperties:
    r"""加载并缓存 canonical mesh 的体积质心与惯量。"""

    # `cache_key` 的后两项不会直接参与加载，但它们决定了 LRU 缓存何时失效：
    # mesh 文件内容一旦变，size / mtime 也会变，旧缓存因此自动被绕开。
    path = Path(cache_key[0])
    mesh = _load_checked_trimesh(path)

    # `trimesh.mass_properties` 默认假设单位密度；这正好符合我们缓存 canonical 真值的需求。
    mass_properties = mesh.mass_properties
    return _CanonicalMeshMassProperties(
        volume=float(mass_properties.volume),
        center_of_mass=(
            float(mass_properties.center_mass[0]),
            float(mass_properties.center_mass[1]),
            float(mass_properties.center_mass[2]),
        ),
        inertia_about_com=np.asarray(mass_properties.inertia, dtype=np.float64),
    )


def _resolve_mesh_path(file_path: str) -> Path:
    r"""解析 custom mesh 的本地路径。

    预期主路径是 builder 写入的绝对路径；若不是，则依次尝试：

    1. 当前工作目录；
    2. `assets/` 子项目根目录。
    """

    if file_path.startswith("package://"):
        raise ValueError(
            "physics closure expects local mesh paths before export, "
            f"got package path {file_path!r}"
        )
    raw_path = Path(os.path.expanduser(file_path))
    if raw_path.is_absolute():
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        return raw_path
    for candidate in (Path.cwd() / raw_path, _ASSETS_ROOT / raw_path):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve mesh path for physics closure: {file_path!r}")


def _load_checked_trimesh(path: Path):
    r"""加载并验证可用于真实体积分的 mesh。

    这里和 mesh SDF validator 保持同一科学标准：

    - 不是 triangle mesh：拒绝；
    - 空 mesh：拒绝；
    - `is_volume=False`：拒绝。
    """

    import trimesh

    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"physics closure expects triangle mesh, got {type(mesh).__name__}: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"physics closure got empty mesh: {path}")
    if not mesh.is_volume:
        raise ValueError(f"physics closure requires watertight positive-volume mesh: {path}")
    return mesh


__all__ = [
    "DensityProfileCfg",
    "AssetPhysicsCfg",
    "AssetPhysicsClosure",
    "close_hand_physics",
]
