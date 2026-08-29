r"""固定物质点相对 PALM anchors 的坐标无关一阶运动学真值。

本模块把 hand-frame 三维 material-point Jacobian 投影到由 physical anchor constellation 与有向
palm normal 定义的关系坐标。对 owner $g$ 上固定 local identity 的物质点 $\bar p_{g,m}$：

$$
p_{g,m}^h(q)=T_{hg}(q)\bar p_{g,m},
\qquad
J_{g,mi}^h(q)=\frac{\partial p_{g,m}^h(q)}{\partial q_i}.
$$

第 $k$ 个固定 PALM anchor 为 $a_k^h$，anchor center 为 $\bar a$。定义：

$$
r_k=p_{g,m}^h-a_k^h,
\qquad
b_k=a_k^h-\bar a,
$$

并用有向 palm normal $n_p$ 分解面内分量 $r_{k,\parallel}$、$b_{k,\parallel}$。长度尺度
$L=0.1\,\mathrm m$ 只做全数据集统一的 SI 数值缩放，不按 morphology 单独归一化。四个关系坐标为：

$$
\phi_k(p)
=
\begin{bmatrix}
n_p^Tr_k/L\\
\|r_{k,\parallel}\|/L\\
r_{k,\parallel}^Tb_{k,\parallel}/L^2\\
n_p^T(r_{k,\parallel}\times b_{k,\parallel})/L^2
\end{bmatrix}.
$$

对应的 relation Jacobian 是：

$$
\Gamma_{gmki}
=
\frac{\partial\phi_k}{\partial q_i}
=
\begin{bmatrix}
n_p^TJ_i/L\\
\hat r_{k,\parallel}^TJ_{i,\parallel}/L\\
b_{k,\parallel}^TJ_{i,\parallel}/L^2\\
n_p^T(J_{i,\parallel}\times b_{k,\parallel})/L^2
\end{bmatrix}
\in\mathbb R^4/\mathrm{rad}.
$$

这些通道对共同的 hand-frame $SE(3)$ 变换为标量不变量，对 joint-coordinate rewrite
$(q_i,\mathcal S_i)\mapsto(-q_i,-\mathcal S_i)$ 为奇。前三个通道对 physical reflection 为偶，
最后一个 chirality 通道为奇，因此该 target 不会像纯欧氏距离一样把 left/right mirror 完全等同。

物质点身份必须固定在 owner-local frame；禁止在 $q+\epsilon$ 与 $q-\epsilon$ 重新查询 closest point。
这样一阶真值只来自刚体运动学，不含 nearest-face switch 或 barycentric projection drift。Anchors 当前固定在
PALM 支持域，因此 $\partial a_k/\partial q_i=0$；若未来引入 moving anchors，必须显式改为相对 Jacobian
$J_p-J_a$，不能沿用本模块公式。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms_and_spatial_screws,
    selected_point_jacobian,
    transform_owner_points,
)

RELATION_CHANNELS = ("height", "radius", "dot", "chirality")
"""Relation target 最后一维的稳定物理顺序；不是可任意重排的匿名 channel。"""


@dataclass(frozen=True)
class MaterialPointRelationJacobianCfg:
    r"""固定物质点关系 Jacobian 的 SI 数值尺度与奇点 mask。

    $L=0.1\,\mathrm m$ 与 retained point-anchor frontend 使用同一全手尺度。它不消除手型大小信息：
    所有资产共同除以同一个米制常数，跨 morphology 的绝对尺度差异仍被保留。

    `distance_epsilon_m` 只屏蔽 $p=a_k$ 时未定义的欧氏径向方向；`plane_radius_epsilon_m` 只屏蔽
    $r_{k,\parallel}=0$ 时未定义的面内 radial direction。Height、dot 与 chirality 在该点仍有定义，
    因而共享 relation mask 只能专门标记 radius channel，不能删除其余三个通道。
    """

    length_scale_m: float = 0.1  # 全数据集统一长度尺度 $L$，单位 m
    distance_epsilon_m: float = 1.0e-9  # $\|p-a_k\|$ 径向导数奇点阈值，单位 m
    plane_radius_epsilon_m: float = 1.0e-9  # $\|r_{k,\parallel}\|$ 面内径向奇点阈值，单位 m

    def __post_init__(self) -> None:
        r"""拒绝非正尺度，避免无量纲关系值与一阶量纲被静默破坏。"""

        if self.length_scale_m <= 0.0:  # $L$ 出现在一阶与二阶关系分母中，必须严格为正
            raise ValueError("length_scale_m must be strictly positive")
        if self.distance_epsilon_m <= 0.0 or self.plane_radius_epsilon_m <= 0.0:
            raise ValueError("distance and plane-radius epsilons must be strictly positive")


@dataclass(frozen=True)
class MaterialPointAnchorJacobianMeasurements:
    r"""已知当前物质点位置与三维 Jacobian 后的 per-anchor 标量测量。

    设 batch size 为 $B$、sampled material-point/joint edge 数为 $E$、当前资产 anchor 数为 $K$。
    Anchor 轴保持无序集合语义：输入 anchors 置换后，所有 `[B,E,K,...]` 输出只同步置换 $K$ 轴。

    `relation_values` 是无量纲的 `[height,radius,dot,chirality]`；其 Jacobian 单位统一为
    rad$^{-1}$。`distance_sensitivity_m_per_rad` 保留 m/rad，作为 radial-only 物理对照。
    """

    distance_m: torch.Tensor  # `[B,E,K]`，$\|p-a_k\|$，m
    distance_sensitivity_m_per_rad: torch.Tensor  # `[B,E,K]`，$\hat r_k^TJ_i$，m/rad
    relation_values: torch.Tensor  # `[B,E,K,4]`，无量纲 $\phi_k(p)$
    relation_sensitivity_per_rad: torch.Tensor  # `[B,E,K,4]`，$\partial\phi_k/\partial q_i$，rad$^{-1}$
    distance_valid_mask: torch.Tensor  # `[B,E,K]`，False 表示 $p\approx a_k$
    radius_valid_mask: torch.Tensor  # `[B,E,K]`，False 表示 $r_{k,\parallel}\approx0$

    def __post_init__(self) -> None:
        r"""验证统一 `[B,E,K]` 轴、四通道顺序和 mask dtype。"""

        if self.distance_m.ndim != 3:  # 物理测量必须保留 batch/edge/anchor 三轴
            raise ValueError("distance_m must have shape [B,E,K]")
        measurement_shape = self.distance_m.shape  # `[B,E,K]`
        for name, value in (
            ("distance_sensitivity_m_per_rad", self.distance_sensitivity_m_per_rad),
            ("distance_valid_mask", self.distance_valid_mask),
            ("radius_valid_mask", self.radius_valid_mask),
        ):
            if value.shape != measurement_shape:
                raise ValueError(f"{name} must have shape [B,E,K]={tuple(measurement_shape)}")
        relation_shape = (*measurement_shape, len(RELATION_CHANNELS))  # `[B,E,K,4]`
        if self.relation_values.shape != relation_shape or self.relation_sensitivity_per_rad.shape != relation_shape:
            raise ValueError(f"relation tensors must have shape {relation_shape}")
        if self.distance_valid_mask.dtype != torch.bool or self.radius_valid_mask.dtype != torch.bool:
            raise TypeError("distance/radius validity masks must use torch.bool")


@dataclass(frozen=True)
class MaterialPointRelationJacobianTarget(MaterialPointAnchorJacobianMeasurements):
    r"""POE/FK、fixed material identity 与 anchor measurements 组成的完整 privileged target。

    `owner_index/joint_index` 可以由 method 在 source-local 阶段保存为共享 `[E]` selector，也可以在
    跨结构 padding 后保存为 `[B,E]`。本对象统一保存展开后的 `[B,E]` selector，避免 target consumer
    再猜测广播语义。非祖先 edge 的三维 Jacobian、distance sensitivity 与四通道 relation sensitivity
    必须逐元素精确为零。
    """

    material_points_h_m: torch.Tensor  # `[B,E,3]`，当前固定物质点，`{h}`，m
    point_jacobian_h_m_per_rad: torch.Tensor  # `[B,E,3]`，raw audit truth，m/rad
    owner_index: torch.Tensor  # `[B,E]`，surface owner selector
    joint_index: torch.Tensor  # `[B,E]`，selected Jacobian column
    ancestor_mask: torch.Tensor  # `[B,E]`，True=active descendant，False=structural zero
    provenance: Mapping[str, str] = field(default_factory=dict)  # 公式、frame、单位与 material identity

    def __post_init__(self) -> None:
        r"""验证 target 轴与精确 structural-zero 物理合同。"""

        super().__post_init__()  # 先验证所有 per-anchor measurement 形状
        batch_size, edge_count = self.distance_m.shape[:2]  # target 的稳定 `[B,E]` 前缀
        edge_shape = (batch_size, edge_count)  # selector 与 ancestor mask 的共同形状
        if self.material_points_h_m.shape != (*edge_shape, 3):
            raise ValueError("material_points_h_m must have shape [B,E,3]")
        if self.point_jacobian_h_m_per_rad.shape != (*edge_shape, 3):
            raise ValueError("point_jacobian_h_m_per_rad must have shape [B,E,3]")
        for name, selector in (
            ("owner_index", self.owner_index),
            ("joint_index", self.joint_index),
            ("ancestor_mask", self.ancestor_mask),
        ):
            if selector.shape != edge_shape:
                raise ValueError(f"{name} must have shape [B,E]={edge_shape}")
        if self.owner_index.dtype != torch.long or self.joint_index.dtype != torch.long:
            raise TypeError("owner_index and joint_index must use torch.long")
        if self.ancestor_mask.dtype != torch.bool:
            raise TypeError("ancestor_mask must use torch.bool")

        # 非祖先 edge 是运动学严格零；这里使用精确比较，不用 tolerance 掩盖拓扑错误。
        structural_zero = ~self.ancestor_mask  # `[B,E]`，跨指/PALM 等 non-ancestor edges
        if (
            torch.any(self.point_jacobian_h_m_per_rad[structural_zero] != 0)
            or torch.any(self.distance_sensitivity_m_per_rad[structural_zero] != 0)
            or torch.any(self.relation_sensitivity_per_rad[structural_zero] != 0)
        ):
            raise ValueError("non-ancestor material-point Jacobian targets must be exactly zero")
        if self.provenance and (
            self.provenance.get("frame") != "h"
            or self.provenance.get("distance_unit") != "m"
            or self.provenance.get("joint_unit") != "rad"
            or self.provenance.get("material_identity") != "fixed_owner_local_home_surface_point"
        ):
            raise ValueError("material-point target provenance does not match the fixed owner-local contract")


@dataclass(frozen=True)
class _RelationGeometry:
    r"""一次 measurement 内复用的 point-anchor 几何中间量。"""

    relation: torch.Tensor  # `[B,E,K,3]`，$r_k=p-a_k$，m
    relation_height: torch.Tensor  # `[B,E,K]`，$n_p^Tr_k$，m
    relation_plane: torch.Tensor  # `[B,E,K,3]`，$r_{k,\parallel}$，m
    relation_radius: torch.Tensor  # `[B,E,K]`，$\|r_{k,\parallel}\|$，m
    anchor_plane: torch.Tensor  # `[K,3]`，$b_{k,\parallel}$，m


def _validate_measurement_inputs(
    material_points_h_m: torch.Tensor,
    point_jacobian_h_m_per_rad: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> None:
    r"""拒绝不一致 frame/device/dtype 与非单位 palm normal。"""

    if material_points_h_m.ndim != 3 or material_points_h_m.shape[-1] != 3:
        raise ValueError("material_points_h_m must have shape [B,E,3]")
    if point_jacobian_h_m_per_rad.shape != material_points_h_m.shape:
        raise ValueError("point_jacobian_h_m_per_rad must share [B,E,3] with material points")
    if anchors_h_m.ndim != 2 or anchors_h_m.shape[-1] != 3 or anchors_h_m.shape[0] < 1:
        raise ValueError("anchors_h_m must have non-empty shape [K,3]")
    if palm_normal_h.shape != (3,):
        raise ValueError("palm_normal_h must have shape [3]")
    tensors = (material_points_h_m, point_jacobian_h_m_per_rad, anchors_h_m, palm_normal_h)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("material points, Jacobians, anchors and palm normal must be floating-point")
    if any(tensor.device != material_points_h_m.device for tensor in tensors[1:]):
        raise ValueError("material points, Jacobians, anchors and palm normal must share device")
    if any(tensor.dtype != material_points_h_m.dtype for tensor in tensors[1:]):
        raise ValueError("material points, Jacobians, anchors and palm normal must share dtype")
    normal_norm = torch.linalg.vector_norm(palm_normal_h)  # 有向 palm normal 的无量纲长度
    if not torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("palm_normal_h must be a unit vector")


def _relation_geometry(
    material_points_h_m: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> _RelationGeometry:
    r"""构造 point-anchor relation 与 palm-plane 分解，供 value/Jacobian 共享。"""

    anchor_center = anchors_h_m.mean(dim=0)  # $\bar a$，完整等地位 anchor set 的几何中心，m
    anchor_centered = anchors_h_m - anchor_center  # $b_k=a_k-\bar a$，形状 `[K,3]`，m
    relation = material_points_h_m.unsqueeze(-2) - anchors_h_m  # $r_k=p-a_k$，`[B,E,K,3]`，m
    relation_height = torch.sum(relation * palm_normal_h, dim=-1)  # $n_p^Tr_k$，`[B,E,K]`，m
    anchor_height = torch.sum(anchor_centered * palm_normal_h, dim=-1)  # $n_p^Tb_k$，`[K]`，m
    relation_plane = relation - relation_height.unsqueeze(-1) * palm_normal_h  # $r_{k,\parallel}$，m
    anchor_plane = anchor_centered - anchor_height.unsqueeze(-1) * palm_normal_h  # $b_{k,\parallel}$，m
    relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1)  # $\|r_{k,\parallel}\|$，m
    return _RelationGeometry(relation, relation_height, relation_plane, relation_radius, anchor_plane)


def measure_material_point_anchor_jacobian(
    material_points_h_m: torch.Tensor,
    point_jacobian_h_m_per_rad: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
    config: MaterialPointRelationJacobianCfg = MaterialPointRelationJacobianCfg(),
) -> MaterialPointAnchorJacobianMeasurements:
    r"""把 raw 三维 material-point Jacobian 写成 per-anchor invariant scalar measurements。

    Args:
        material_points_h_m (torch.Tensor): 当前固定物质点，形状 `[B,E,3]`，`{h}`，m。
        point_jacobian_h_m_per_rad (torch.Tensor): 对应 selected JOINT columns，`[B,E,3]`，m/rad。
        anchors_h_m (torch.Tensor): 当前资产固定 PALM anchors，`[K,3]`，`{h}`，m。
        palm_normal_h (torch.Tensor): 有向单位 palm normal，`[3]`，`{h}`。
        config (MaterialPointRelationJacobianCfg): 全数据集统一长度尺度与奇点阈值。

    Returns:
        MaterialPointAnchorJacobianMeasurements: `[B,E,K]` distance 与 `[B,E,K,4]` relation 真值。
    """

    _validate_measurement_inputs(material_points_h_m, point_jacobian_h_m_per_rad, anchors_h_m, palm_normal_h)
    geometry = _relation_geometry(material_points_h_m, anchors_h_m, palm_normal_h)  # 共享关系几何
    length_scale = float(config.length_scale_m)  # $L$，m；所有资产共享
    quadratic_scale = length_scale * length_scale  # $L^2$，m$^2$

    # 欧氏距离与 radial projection；$p=a_k$ 时方向未定义，只通过 mask 声明，不伪造有效 target。
    distance_m = torch.linalg.vector_norm(geometry.relation, dim=-1)  # `[B,E,K]`，m
    distance_valid = distance_m > config.distance_epsilon_m  # 欧氏 radial direction 有效域
    distance_direction = geometry.relation / distance_m.clamp_min(config.distance_epsilon_m).unsqueeze(-1)
    distance_sensitivity = torch.sum(
        distance_direction * point_jacobian_h_m_per_rad.unsqueeze(-2),
        dim=-1,
    )  # $\hat r_k^TJ_i$，`[B,E,K]`，m/rad

    # Relation values 与 retained point-anchor frontend 使用同一物理坐标，只保留 q-dependent 四通道。
    anchor_plane = geometry.anchor_plane  # `[K,3]`，m
    anchor_plane_batched = anchor_plane.view(1, 1, anchor_plane.shape[0], 3)  # `[1,1,K,3]`
    dot = torch.sum(geometry.relation_plane * anchor_plane_batched, dim=-1)  # m$^2$
    chirality = torch.sum(
        torch.cross(geometry.relation_plane, anchor_plane_batched.expand_as(geometry.relation_plane), dim=-1)
        * palm_normal_h,
        dim=-1,
    )  # $n_p^T(r_{\parallel}\times b_{\parallel})$，m$^2$
    relation_values = torch.stack(
        (
            geometry.relation_height / length_scale,
            geometry.relation_radius / length_scale,
            dot / quadratic_scale,
            chirality / quadratic_scale,
        ),
        dim=-1,
    )  # `[B,E,K,4]`，无量纲

    # 把 $J_i$ 分解成 palm-normal 与 palm-plane 两部分；anchors 固定，因此没有 $J_a$ 项。
    point_height_velocity = torch.sum(
        point_jacobian_h_m_per_rad * palm_normal_h,
        dim=-1,
    )  # $n_p^TJ_i$，`[B,E]`，m/rad
    point_plane_velocity = (
        point_jacobian_h_m_per_rad - point_height_velocity.unsqueeze(-1) * palm_normal_h
    )  # $J_{i,\parallel}$，`[B,E,3]`，m/rad
    radius_valid = geometry.relation_radius > config.plane_radius_epsilon_m  # 面内 radial direction 有效域
    radial_direction = geometry.relation_plane / geometry.relation_radius.clamp_min(
        config.plane_radius_epsilon_m
    ).unsqueeze(-1)  # $\hat r_{k,\parallel}$，无量纲
    height_sensitivity = point_height_velocity.unsqueeze(-1).expand_as(distance_m) / length_scale  # rad$^{-1}$
    radius_sensitivity = torch.sum(
        radial_direction * point_plane_velocity.unsqueeze(-2),
        dim=-1,
    ) / length_scale  # $\hat r_{k,\parallel}^TJ_{i,\parallel}/L$，rad$^{-1}$
    dot_sensitivity = torch.sum(
        point_plane_velocity.unsqueeze(-2) * anchor_plane_batched,
        dim=-1,
    ) / quadratic_scale  # $b_{k,\parallel}^TJ_{i,\parallel}/L^2$，rad$^{-1}$
    chirality_sensitivity = torch.sum(
        torch.cross(
            point_plane_velocity.unsqueeze(-2).expand_as(geometry.relation_plane),
            anchor_plane_batched.expand_as(geometry.relation_plane),
            dim=-1,
        )
        * palm_normal_h,
        dim=-1,
    ) / quadratic_scale  # $n_p^T(J_{i,\parallel}\times b_{k,\parallel})/L^2$，rad$^{-1}$
    relation_sensitivity = torch.stack(
        (height_sensitivity, radius_sensitivity, dot_sensitivity, chirality_sensitivity),
        dim=-1,
    )  # `[B,E,K,4]`，最后一维顺序由 `RELATION_CHANNELS` 固定

    return MaterialPointAnchorJacobianMeasurements(
        distance_m=distance_m,
        distance_sensitivity_m_per_rad=distance_sensitivity,
        relation_values=relation_values,
        relation_sensitivity_per_rad=relation_sensitivity,
        distance_valid_mask=distance_valid,
        radius_valid_mask=radius_valid,
    )


def _expanded_selector(selector: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
    r"""把共享 `[E]` selector 显式广播成 target 保存的 `[B,E]`。"""

    if selector.ndim == 1:  # 同一资产 q-block 共享 owner/joint edge 轴
        return selector.unsqueeze(0).expand(batch_size, -1)
    if selector.ndim == 2 and selector.shape[0] == batch_size:  # 每个 q row 独立 selector
        return selector
    raise ValueError(f"{name} must have shape [E] or [B,E] sharing q batch size")


def generate_material_point_relation_jacobian_targets(
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_material_points_m: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
    config: MaterialPointRelationJacobianCfg = MaterialPointRelationJacobianCfg(),
    *,
    owner_transforms: torch.Tensor | None = None,
    current_spatial_screws: torch.Tensor | None = None,
) -> MaterialPointRelationJacobianTarget:
    r"""由固定 owner-local material identities 生成完整 anchor-relational Jacobian target。

    `local_material_points_m` 是 target 的身份锚点；它在全部 $q$ realization 中保持同一 owner-local
    坐标，不可替换为每次重新查询的 closest point。函数可复用上游已计算的 owner transforms/current
    screws，使 Material-point Jacobian 单目标 teacher 不重复 POE。

    Args:
        spec (EmbodimentGeometrySpec): 当前资产 POE、home transforms 与 ancestor truth。
        q (torch.Tensor): 当前构型 `[B,N_J]`，rad。
        owner_index (torch.Tensor): `[E]` 或 `[B,E]` owner selectors。
        joint_index (torch.Tensor): 与 owner selector 同形状的 selected JOINT columns。
        local_material_points_m (torch.Tensor): `[E,3]` 或 `[B,E,3]` 固定 owner-local 点，m。
        anchors_h_m (torch.Tensor): `[K,3]` 固定 PALM anchors，`{h}`，m。
        palm_normal_h (torch.Tensor): `[3]` 有向单位 palm normal，`{h}`。
        config (MaterialPointRelationJacobianCfg): 全数据集统一尺度与奇点 mask。
        owner_transforms (torch.Tensor | None): 可复用 `[B,G,4,4]` 当前 owner poses。
        current_spatial_screws (torch.Tensor | None): 可复用 `[B,N_J,6]` 当前 screws。

    Returns:
        MaterialPointRelationJacobianTarget: 固定物质点位置、raw Jacobian 与全部 per-anchor scalar truth。
    """

    if owner_index.shape != joint_index.shape or owner_index.ndim not in {1, 2}:
        raise ValueError("owner_index and joint_index must have identical [E] or [B,E] shape")
    batch_size = q.shape[0]  # 当前同资产 q-block 大小 $B$
    owner_selector = _expanded_selector(owner_index, batch_size=batch_size, name="owner_index")  # `[B,E]`
    joint_selector = _expanded_selector(joint_index, batch_size=batch_size, name="joint_index")  # `[B,E]`

    # 任一复用张量缺失时共同计算 POE 与 current screws，避免两个独立路径重复 twist exponential。
    if owner_transforms is None or current_spatial_screws is None:
        computed_transforms, computed_screws = forward_owner_transforms_and_spatial_screws(spec, q)
        owner_transforms = computed_transforms if owner_transforms is None else owner_transforms
        current_spatial_screws = computed_screws if current_spatial_screws is None else current_spatial_screws

    # 同一 local identity 随 owner rigid transform 运动；这一步定义 persistent material trajectory。
    material_points_h_m = transform_owner_points(
        owner_transforms,
        owner_selector,
        local_material_points_m,
    )  # `[B,E,3]`，`{h}`，m
    point_jacobian_h_m_per_rad = selected_point_jacobian(
        spec,
        q,
        owner_selector,
        joint_selector,
        local_material_points_m,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
    )  # `[B,E,3]`，m/rad；non-ancestor 精确为 0
    measurements = measure_material_point_anchor_jacobian(
        material_points_h_m,
        point_jacobian_h_m_per_rad,
        anchors_h_m,
        palm_normal_h,
        config,
    )  # per-anchor distance/relation values 与 sensitivities
    ancestor_mask = spec.owner_ancestor_mask[owner_selector, joint_selector]  # `[B,E]` 运动学真值

    return MaterialPointRelationJacobianTarget(
        distance_m=measurements.distance_m,
        distance_sensitivity_m_per_rad=measurements.distance_sensitivity_m_per_rad,
        relation_values=measurements.relation_values,
        relation_sensitivity_per_rad=measurements.relation_sensitivity_per_rad,
        distance_valid_mask=measurements.distance_valid_mask,
        radius_valid_mask=measurements.radius_valid_mask,
        material_points_h_m=material_points_h_m,
        point_jacobian_h_m_per_rad=point_jacobian_h_m_per_rad,
        owner_index=owner_selector,
        joint_index=joint_selector,
        ancestor_mask=ancestor_mask,
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "relation_unit": "dimensionless",
            "relation_sensitivity_unit": "rad^-1",
            "relation_channels": ",".join(RELATION_CHANNELS),
            "material_identity": "fixed_owner_local_home_surface_point",
            "anchor_motion": "fixed_palm_support",
        },
    )


__all__ = [
    "RELATION_CHANNELS",
    "MaterialPointAnchorJacobianMeasurements",
    "MaterialPointRelationJacobianCfg",
    "MaterialPointRelationJacobianTarget",
    "generate_material_point_relation_jacobian_targets",
    "measure_material_point_anchor_jacobian",
]
