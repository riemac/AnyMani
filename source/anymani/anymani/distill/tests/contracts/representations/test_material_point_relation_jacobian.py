r"""固定物质点的 anchor-relational Jacobian 物理合同。

测试只验证 representation truth，不涉及 reader、loss 或训练结论。核心对象是 owner-local 固定
material point 随 POE/FK 运动后，相对 PALM anchor constellation 的四通道关系导数：

$$
\Gamma_{gmki}
=
\frac{\partial}{\partial q_i}
\begin{bmatrix}
n_p^Tr_k/L\\
\|r_{k,\parallel}\|/L\\
r_{k,\parallel}^Tb_{k,\parallel}/L^2\\
n_p^T(r_{k,\parallel}\times b_{k,\parallel})/L^2
\end{bmatrix}.
$$

其中 $r_k=p_{g,m}(q)-a_k$，$b_k=a_k-\bar a$，$L=0.1\,\mathrm m$。前三个
通道对共同 reflection 为偶，chirality 通道为奇；全部 sensitivity 对 joint-coordinate sign 为奇。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
)
from anymani.distill.representations.targets.material_point_jacobian import (
    MaterialPointRelationJacobianCfg,
    generate_material_point_relation_jacobian_targets,
    measure_material_point_anchor_jacobian,
)

pytestmark = pytest.mark.contract


def _two_joint_spec() -> EmbodimentGeometrySpec:
    r"""构造两关节分支；PALM 对任意 joint 是结构零，distal owner 受两关节影响。"""

    space_screws = torch.tensor(  # $[N_J,6]$，顺序为 $[\omega,v]$，单位轴 + m
        (
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.8),
        ),
        dtype=torch.float64,
    )
    owner_home = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)  # PALM/proximal/distal 三个 owner
    owner_home[1, :3, 3] = torch.tensor((0.8, 0.0, 0.1), dtype=torch.float64)  # proximal origin，m
    owner_home[2, :3, 3] = torch.tensor((1.4, 0.2, 0.2), dtype=torch.float64)  # distal origin，m
    owner_ancestor = torch.tensor(  # owner $g$ 是否受 JOINT $i$ 运动
        ((False, False), (True, False), (True, True)),
        dtype=torch.bool,
    )
    joint_ancestor = torch.tensor(  # 第二关节的当前 screw 受第一关节变换
        ((False, False), (True, False)),
        dtype=torch.bool,
    )
    return EmbodimentGeometrySpec(
        space_screws=space_screws,
        q_home=torch.tensor((0.1, -0.2), dtype=torch.float64),  # 非零 home 验证 $q-q_{home}$ 语义
        owner_home_transforms=owner_home,
        owner_ancestor_mask=owner_ancestor,
        joint_ancestor_mask=joint_ancestor,
        joint_limits=torch.tensor(((-1.0, 1.2), (-0.9, 1.1)), dtype=torch.float64),
    )


def _anchors() -> torch.Tensor:
    r"""返回不共面且具有有向 palm-plane 布局的五个固定 anchors，单位 m。"""

    return torch.tensor(
        (
            (-0.25, -0.20, -0.03),
            (0.30, -0.15, 0.02),
            (0.28, 0.24, 0.01),
            (-0.22, 0.27, -0.02),
            (0.03, 0.01, 0.06),
        ),
        dtype=torch.float64,
    )


def _edge_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""返回两条 active edge 与一条 PALM structural-zero edge。"""

    owner_index = torch.tensor((2, 2, 0), dtype=torch.long)  # distal/distal/PALM
    joint_index = torch.tensor((0, 1, 1), dtype=torch.long)  # active/active/non-ancestor
    local_points = torch.tensor(  # 每条 edge 的固定 owner-local material identity，m
        ((0.12, -0.04, 0.03), (-0.08, 0.06, -0.02), (0.02, 0.01, 0.0)),
        dtype=torch.float64,
    )
    return owner_index, joint_index, local_points


def _rewrite_spec(spec: EmbodimentGeometrySpec, sign: torch.Tensor) -> EmbodimentGeometrySpec:
    r"""执行完整 joint-coordinate rewrite，保持同一物理 POE。"""

    assert spec.joint_limits is not None  # 测试规格显式提供 limits
    rewritten_limits = torch.where(
        sign[:, None] > 0.0,
        spec.joint_limits,
        torch.stack((-spec.joint_limits[:, 1], -spec.joint_limits[:, 0]), dim=-1),
    )  # $[q'_{min},q'_{max}]=[-q_{max},-q_{min}]$
    return replace(
        spec,
        space_screws=spec.space_screws * sign[:, None],  # $\mathcal S'_i=s_i\mathcal S_i$
        q_home=spec.q_home * sign,  # $q'_{home,i}=s_iq_{home,i}$
        joint_limits=rewritten_limits,
    )


def test_fixed_material_relation_jacobian_matches_central_difference() -> None:
    r"""解析 distance/relation sensitivities 必须匹配同一 material identity 的 $q\pm\epsilon$。"""

    spec = _two_joint_spec()  # float64 物理规格
    q = torch.tensor(((0.37, -0.51),), dtype=torch.float64)  # 当前合法构型，形状 [1,2]
    owner_index, joint_index, local_points = _edge_inputs()  # 三条固定物质点 edge
    anchors = _anchors()  # $K=5$ 固定 PALM anchors
    normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)  # 有向 palm normal
    target = generate_material_point_relation_jacobian_targets(
        spec,
        q,
        owner_index,
        joint_index,
        local_points,
        anchors,
        normal,
    )

    # 每条 edge 独立扰动自己的 joint；plus/minus 始终变换同一个 owner-local material point。
    epsilon = 1.0e-6  # rad；float64 中心差分数值锚点
    for edge in range(owner_index.numel()):
        q_plus = q.clone()  # $q+\epsilon e_i$
        q_minus = q.clone()  # $q-\epsilon e_i$
        q_plus[:, joint_index[edge]] += epsilon
        q_minus[:, joint_index[edge]] -= epsilon
        plus = generate_material_point_relation_jacobian_targets(
            spec,
            q_plus,
            owner_index[edge : edge + 1],
            joint_index[edge : edge + 1],
            local_points[edge : edge + 1],
            anchors,
            normal,
        )
        minus = generate_material_point_relation_jacobian_targets(
            spec,
            q_minus,
            owner_index[edge : edge + 1],
            joint_index[edge : edge + 1],
            local_points[edge : edge + 1],
            anchors,
            normal,
        )
        distance_fd = (plus.distance_m - minus.distance_m) / (2.0 * epsilon)  # $\partial d_k/\partial q_i$
        relation_fd = (plus.relation_values - minus.relation_values) / (2.0 * epsilon)  # $\partial\phi_k/\partial q_i$
        torch.testing.assert_close(
            target.distance_sensitivity_m_per_rad[:, edge : edge + 1],
            distance_fd,
            atol=2.0e-8,
            rtol=2.0e-8,
        )
        torch.testing.assert_close(
            target.relation_sensitivity_per_rad[:, edge : edge + 1],
            relation_fd,
            atol=2.0e-7,
            rtol=2.0e-7,
        )


def test_joint_coordinate_rewrite_preserves_points_and_flips_selected_columns() -> None:
    r"""同一物理构型下，material position/relations 为偶，一阶 selected column 按 $s_i$ 变号。"""

    spec = _two_joint_spec()
    q = torch.tensor(((0.41, -0.63),), dtype=torch.float64)
    owner_index, joint_index, local_points = _edge_inputs()
    anchors = _anchors()
    normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    sign = torch.tensor((-1.0, 1.0), dtype=torch.float64)  # 只改写第一根 joint 的 coordinate gauge
    baseline = generate_material_point_relation_jacobian_targets(
        spec, q, owner_index, joint_index, local_points, anchors, normal
    )
    rewritten = generate_material_point_relation_jacobian_targets(
        _rewrite_spec(spec, sign),
        q * sign,
        owner_index,
        joint_index,
        local_points,
        anchors,
        normal,
    )
    selected_sign = sign[joint_index].view(1, -1)  # 每条 edge 所选 Jacobian column 的 parity

    torch.testing.assert_close(rewritten.material_points_h_m, baseline.material_points_h_m, atol=0.0, rtol=0.0)
    torch.testing.assert_close(rewritten.distance_m, baseline.distance_m, atol=0.0, rtol=0.0)
    torch.testing.assert_close(rewritten.relation_values, baseline.relation_values, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        rewritten.point_jacobian_h_m_per_rad,
        baseline.point_jacobian_h_m_per_rad * selected_sign[..., None],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rewritten.distance_sensitivity_m_per_rad,
        baseline.distance_sensitivity_m_per_rad * selected_sign[..., None],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rewritten.relation_sensitivity_per_rad,
        baseline.relation_sensitivity_per_rad * selected_sign[..., None, None],
        atol=0.0,
        rtol=0.0,
    )


def test_measurements_are_se3_invariant_and_chirality_is_reflection_odd() -> None:
    r"""共同 rigid frame change 不改 scalar；physical mirror 只翻 oriented chirality channel。"""

    spec = _two_joint_spec()
    q = torch.tensor(((0.29, -0.44),), dtype=torch.float64)
    owner_index, joint_index, local_points = _edge_inputs()
    anchors = _anchors()
    normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    baseline = generate_material_point_relation_jacobian_targets(
        spec, q, owner_index, joint_index, local_points, anchors, normal
    )

    # 取 det(R)=+1 的轴角旋转与任意平移，验证共同 SE(3) frame rewrite。
    axis = torch.tensor((0.31, -0.72, 0.62), dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    theta = torch.tensor(0.83, dtype=torch.float64)
    skew = torch.tensor(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=torch.float64,
    )
    rotation = torch.eye(3, dtype=torch.float64) + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)
    translation = torch.tensor((0.02, -0.03, 0.01), dtype=torch.float64)
    transformed = measure_material_point_anchor_jacobian(
        baseline.material_points_h_m @ rotation.T + translation,
        baseline.point_jacobian_h_m_per_rad @ rotation.T,
        anchors @ rotation.T + translation,
        normal @ rotation.T,
    )
    torch.testing.assert_close(transformed.distance_m, baseline.distance_m, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(transformed.relation_values, baseline.relation_values, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(
        transformed.relation_sensitivity_per_rad,
        baseline.relation_sensitivity_per_rad,
        atol=1.0e-14,
        rtol=1.0e-14,
    )

    # Reflection 不属于 SE(3)：纯距离与前三个 relation channels 为偶，oriented chirality 为奇。
    reflection = torch.diag(torch.tensor((-1.0, 1.0, 1.0), dtype=torch.float64))
    mirrored = measure_material_point_anchor_jacobian(
        baseline.material_points_h_m @ reflection.T,
        baseline.point_jacobian_h_m_per_rad @ reflection.T,
        anchors @ reflection.T,
        normal @ reflection.T,
    )
    parity = torch.tensor((1.0, 1.0, 1.0, -1.0), dtype=torch.float64)
    torch.testing.assert_close(mirrored.distance_m, baseline.distance_m, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mirrored.distance_sensitivity_m_per_rad, baseline.distance_sensitivity_m_per_rad)
    torch.testing.assert_close(mirrored.relation_values, baseline.relation_values * parity)
    torch.testing.assert_close(
        mirrored.relation_sensitivity_per_rad,
        baseline.relation_sensitivity_per_rad * parity,
    )


def test_anchor_axis_is_permutation_equivariant_and_structural_zero_is_exact() -> None:
    r"""Anchor 存储顺序只能同步重排 $K$ 轴，非祖先 edge 的全部一阶输出必须逐元素为零。"""

    spec = _two_joint_spec()
    q = torch.tensor(((0.33, -0.39),), dtype=torch.float64)
    owner_index, joint_index, local_points = _edge_inputs()
    anchors = _anchors()
    normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    baseline = generate_material_point_relation_jacobian_targets(
        spec, q, owner_index, joint_index, local_points, anchors, normal
    )
    permutation = torch.tensor((3, 0, 4, 1, 2), dtype=torch.long)
    permuted = generate_material_point_relation_jacobian_targets(
        spec, q, owner_index, joint_index, local_points, anchors[permutation], normal
    )

    torch.testing.assert_close(permuted.distance_m, baseline.distance_m[..., permutation])
    torch.testing.assert_close(
        permuted.distance_sensitivity_m_per_rad,
        baseline.distance_sensitivity_m_per_rad[..., permutation],
    )
    torch.testing.assert_close(permuted.relation_values, baseline.relation_values[..., permutation, :])
    torch.testing.assert_close(
        permuted.relation_sensitivity_per_rad,
        baseline.relation_sensitivity_per_rad[..., permutation, :],
    )
    assert baseline.ancestor_mask.tolist() == [[True, True, False]]
    assert torch.count_nonzero(baseline.point_jacobian_h_m_per_rad[:, 2]) == 0
    assert torch.count_nonzero(baseline.distance_sensitivity_m_per_rad[:, 2]) == 0
    assert torch.count_nonzero(baseline.relation_sensitivity_per_rad[:, 2]) == 0


def test_target_rejects_non_unit_palm_normal_and_degenerate_scale() -> None:
    r"""物理边界拒绝非单位 palm normal 与非正长度尺度，避免静默改变 target 量纲。"""

    points = torch.zeros(1, 1, 3, dtype=torch.float64)
    jacobian = torch.zeros_like(points)
    anchors = _anchors()
    with pytest.raises(ValueError, match="unit"):
        measure_material_point_anchor_jacobian(
            points,
            jacobian,
            anchors,
            torch.tensor((0.0, 0.0, 2.0), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="length_scale_m"):
        MaterialPointRelationJacobianCfg(length_scale_m=0.0)
