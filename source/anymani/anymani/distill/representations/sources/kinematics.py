r"""Current posed geometry 所需的纯 PyTorch 运动学实现。

设 PALM/JOINT/TIP surface owner $g$ 受 $n_g$ 个有序活动关节影响。$q_i$ 是第 $i$ 个
ancestor JOINT 的当前角度，$q_{\mathfrak m,i}^{0}$ 是资产基准构型角，二者单位均为 rad；
$\Delta q_i=q_i-q_{\mathfrak m,i}^{0}$。$\mathcal S_i\in\mathbb R^6$ 是基准构型空间旋量，
角分量无量纲、线分量单位 m；$T_{hg}(q_{\mathfrak m}^{0})\in SE(3)$ 是 owner local
coordinates 到 ``{h}`` 的基准刚体变换。当前 pose 满足 Product of Exponentials：

$$
T_{hg}(q)
=
e^{[\mathcal S_1]\Delta q_1}
\cdots
e^{[\mathcal S_{n_g}]\Delta q_{n_g}}
T_{hg}(q_{\mathfrak m}^{0}).
$$

资产基准构型不强制为全零；若实现直接把当前 $q$ 当作从零 home 开始的指数坐标，会在存在
joint zero offset 或非零 $q_{\mathfrak m}^{0}$ 时产生错误物理表面。

输入必须保留 ancestor membership、严格 chain order、parent/child topology、group owner
与 valid mask。纯 current-geometry target 不需要 $\dot q$、action/history、command、
contact、object state、mass/inertia 或 joint limits；这些量可以服务 policy、采样与控制
边界，但不决定给定 $q$ 下的刚体 collision geometry。

Gauge 边界：同一 physical joint 可同时重写 screw sign 与 coordinate sign，joint zero
offset 也会联动改变 $q$ 与 home geometry。source 必须允许 paired re-gauged evidence，
不能假设不同 URDF 的 axis、origin 与 zero convention 天然一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KinematicTreeSpec:
    r"""同一结构模式下的空间旋量、基准位姿与祖先关系。

    空间旋量使用顺序 `[omega_x, omega_y, omega_z, v_x, v_y, v_z]`，满足
    $\dot y=\omega\times y+v$。`owner_ancestor_mask[g,i]` 指示 JOINT $i$
    是否影响 owner $g$；`joint_ancestor_mask[i,j]` 指示 JOINT $j$ 是否是 JOINT
    $i$ 的严格祖先。后者防止 canonical joint order 中更早出现的其他手指关节错误地
    变换当前轴线。

    Attributes:
        space_screws (torch.Tensor): 基准 `{h}` 中的空间旋量，形状 `[N_J,6]`。
        q_home (torch.Tensor): 资产显式基准构型，形状 `[N_J]`，单位 rad。
        owner_home_transforms (torch.Tensor): owner local 到 `{h}` 的基准变换，形状 `[G,4,4]`。
        owner_ancestor_mask (torch.Tensor): 形状 `[G,N_J]` 的祖先布尔矩阵。
        joint_ancestor_mask (torch.Tensor): 形状 `[N_J,N_J]` 的严格祖先布尔矩阵。
    """

    space_screws: torch.Tensor
    q_home: torch.Tensor
    owner_home_transforms: torch.Tensor
    owner_ancestor_mask: torch.Tensor
    joint_ancestor_mask: torch.Tensor

    def __post_init__(self) -> None:
        r"""在 source 边界验证形状、dtype 与旋量单位轴。"""

        if self.space_screws.ndim != 2 or self.space_screws.shape[-1] != 6:
            raise ValueError(f"space_screws must have shape [N_J,6], got {tuple(self.space_screws.shape)}")
        joint_count = self.space_screws.shape[0]  # 当前结构模式活动 JOINT 数 $N_J$
        if self.q_home.shape != (joint_count,):
            raise ValueError(f"q_home must have shape [{joint_count}], got {tuple(self.q_home.shape)}")
        if self.owner_home_transforms.ndim != 3 or self.owner_home_transforms.shape[-2:] != (4, 4):
            raise ValueError(
                "owner_home_transforms must have shape [G,4,4], "
                f"got {tuple(self.owner_home_transforms.shape)}"
            )
        owner_count = self.owner_home_transforms.shape[0]  # 归属体/实体数 $G=N_E$
        if self.owner_ancestor_mask.shape != (owner_count, joint_count):
            raise ValueError(
                f"owner_ancestor_mask must have shape [{owner_count},{joint_count}], "
                f"got {tuple(self.owner_ancestor_mask.shape)}"
            )
        if self.joint_ancestor_mask.shape != (joint_count, joint_count):
            raise ValueError(
                f"joint_ancestor_mask must have shape [{joint_count},{joint_count}], "
                f"got {tuple(self.joint_ancestor_mask.shape)}"
            )
        if self.owner_ancestor_mask.dtype != torch.bool or self.joint_ancestor_mask.dtype != torch.bool:
            raise TypeError("ancestor masks must use torch.bool")
        if not self.space_screws.is_floating_point() or not self.q_home.is_floating_point():
            raise TypeError("space_screws and q_home must be floating-point tensors")

        # 当前 AnyMani 合同只含 revolute JOINT，因此旋量角分量必须是单位轴。
        angular_norm = torch.linalg.vector_norm(self.space_screws[:, :3], dim=-1)  # $\|\omega_i\|_2$
        if not torch.allclose(angular_norm, torch.ones_like(angular_norm), atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("each revolute space screw must have a unit angular axis")

    def to(self, *, device: torch.device | str, dtype: torch.dtype | None = None) -> KinematicTreeSpec:
        r"""把同一结构模式的静态张量整体迁移到目标设备与浮点 dtype。"""

        target_dtype = dtype or self.space_screws.dtype  # 浮点精度默认保持 source 规格
        return KinematicTreeSpec(
            space_screws=self.space_screws.to(device=device, dtype=target_dtype),
            q_home=self.q_home.to(device=device, dtype=target_dtype),
            owner_home_transforms=self.owner_home_transforms.to(device=device, dtype=target_dtype),
            owner_ancestor_mask=self.owner_ancestor_mask.to(device=device),
            joint_ancestor_mask=self.joint_ancestor_mask.to(device=device),
        )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    r"""构造叉乘矩阵 $[v]_\times$，输入形状 `[...,3]`。"""

    x, y, z = vector.unbind(dim=-1)  # 三个分量均保持前导 batch 轴
    zero = torch.zeros_like(x)  # 叉乘矩阵对角线严格为零
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _revolute_twist_exp(space_screw: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""计算单位转动空间旋量的 $SE(3)$ 指数映射。

    对 $\mathcal S=(\omega,v)$ 且 $\|\omega\|=1$：

    $$
    e^{[\mathcal S]\theta}
    =
    \begin{bmatrix}
    e^{[\omega]\theta} &
    (I\theta+(1-\cos\theta)[\omega]+(\theta-\sin\theta)[\omega]^2)v\\
    0 & 1
    \end{bmatrix}.
    $$
    """

    omega = space_screw[:3]  # 单位角轴 $\omega$，无量纲
    linear = space_screw[3:]  # 空间旋量线分量 $v$，单位 m
    omega_hat = _skew(omega)  # $[\omega]_\times$，形状 `[3,3]`
    omega_hat_squared = omega_hat @ omega_hat  # Rodrigues 二次项 $[\omega]^2$
    identity3 = torch.eye(3, device=theta.device, dtype=theta.dtype)  # 与 q 同 device/dtype 的 $I_3$

    sin_theta = torch.sin(theta)[..., None, None]  # $\sin\theta$，形状 `[...,1,1]`
    cos_theta = torch.cos(theta)[..., None, None]  # $\cos\theta$，形状 `[...,1,1]`
    rotation = identity3 + sin_theta * omega_hat + (1.0 - cos_theta) * omega_hat_squared  # Rodrigues $R$

    theta_matrix = theta[..., None, None]  # $\theta$，为左 Jacobian 三项提供广播轴
    translation_operator = (
        theta_matrix * identity3
        + (1.0 - cos_theta) * omega_hat
        + (theta_matrix - sin_theta) * omega_hat_squared
    )  # $V(\theta)$，将空间旋量线分量变成平移
    translation = torch.matmul(translation_operator, linear[..., None]).squeeze(-1)  # $p=V(\theta)v$，单位 m

    transform = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)  # `[...,4,4]`
    transform[..., :3, :3] = rotation  # 当前旋转 $R\in SO(3)$
    transform[..., :3, 3] = translation  # 当前平移 $p\in\mathbb R^3$，单位 m
    transform[..., 3, 3] = 1.0  # 齐次坐标底行
    return transform


def forward_owner_transforms(spec: KinematicTreeSpec, q: torch.Tensor) -> torch.Tensor:
    r"""按 owner ancestor mask 计算 batched POE 刚体位姿。

    Args:
        spec (KinematicTreeSpec): 当前同构结构模式的静态运动学事实。
        q (torch.Tensor): 当前物理关节构型，形状 `[B,N_J]`，单位 rad。

    Returns:
        torch.Tensor: owner local 到 `{h}` 的变换，形状 `[B,G,4,4]`。
    """

    joint_count = spec.space_screws.shape[0]  # 活动关节数 $N_J$
    owner_count = spec.owner_home_transforms.shape[0]  # 归属体数 $G$
    if q.ndim != 2 or q.shape[1] != joint_count:
        raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
    if q.device != spec.space_screws.device:
        raise ValueError("q and KinematicTreeSpec tensors must be on the same device")

    delta_q = q - spec.q_home  # $\Delta q=q-q_{home}$，形状 `[B,N_J]`，单位 rad
    batch_size = q.shape[0]  # 同结构模式 microbatch 大小 $B$
    transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(batch_size, owner_count, 4, 4).clone()

    # 每个 canonical JOINT 依次左乘；非祖先通过 theta=0 产生严格单位变换。
    for joint_index in range(joint_count):
        owner_theta = delta_q[:, joint_index : joint_index + 1] * spec.owner_ancestor_mask[
            :, joint_index
        ].to(q.dtype).unsqueeze(0)  # `[B,G]`，非祖先严格为 0
        joint_transform = _revolute_twist_exp(spec.space_screws[joint_index], owner_theta)  # `[B,G,4,4]`
        transform = transform @ joint_transform  # 按 canonical chain order 复合 POE

    return transform @ spec.owner_home_transforms.unsqueeze(0)  # $T_{hg}(q)=\prod e^{S_i\Delta q_i}M_g$


def transform_owner_points(
    owner_transforms: torch.Tensor,
    owner_index: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    r"""把 selected owner-local material points 变换到 `{h}`。

    Args:
        owner_transforms (torch.Tensor): 形状 `[B,G,4,4]` 的当前 owner 位姿。
        owner_index (torch.Tensor): 形状 `[E]` 的 owner selector。
        local_points (torch.Tensor): 形状 `[E,3]` 的 owner-local 点，单位 m。

    Returns:
        torch.Tensor: 形状 `[B,E,3]` 的 `{h}` 点，单位 m。
    """

    if owner_index.ndim != 1 or local_points.shape != (owner_index.numel(), 3):
        raise ValueError("owner_index/local_points must have shapes [E] and [E,3]")
    selected = owner_transforms.index_select(1, owner_index)  # `[B,E,4,4]`，selected owner poses
    rotation = selected[..., :3, :3]  # `[B,E,3,3]`，owner local -> `{h}`
    translation = selected[..., :3, 3]  # `[B,E,3]`，owner origin in `{h}`，单位 m
    return torch.matmul(rotation, local_points[None, ..., None]).squeeze(-1) + translation  # $Rp^{local}+t$


def _current_spatial_screws(spec: KinematicTreeSpec, q: torch.Tensor) -> torch.Tensor:
    r"""把每个基准空间旋量经其严格祖先变换到当前 `{h}`。"""

    batch_size, joint_count = q.shape  # 同结构模式 batch 与 JOINT 轴
    delta_q = q - spec.q_home  # 当前相对基准的物理角度，单位 rad
    current = torch.empty(batch_size, joint_count, 6, device=q.device, dtype=q.dtype)  # `[B,N_J,6]`

    # 每个 JOINT 只复合自身严格祖先；其他手指即使在 canonical order 更早也不参与。
    for target_joint in range(joint_count):
        prefix = torch.eye(4, device=q.device, dtype=q.dtype).expand(batch_size, 4, 4).clone()  # $T_{prefix}$
        for source_joint in range(joint_count):
            if not bool(spec.joint_ancestor_mask[target_joint, source_joint]):
                continue
            prefix = prefix @ _revolute_twist_exp(
                spec.space_screws[source_joint], delta_q[:, source_joint]
            )  # 只复合 target JOINT 的严格祖先

        rotation = prefix[:, :3, :3]  # $R_{prefix}$，形状 `[B,3,3]`
        translation = prefix[:, :3, 3]  # $p_{prefix}$，形状 `[B,3]`，单位 m
        omega_home = spec.space_screws[target_joint, :3]  # 基准角轴 $\omega_i$
        linear_home = spec.space_screws[target_joint, 3:]  # 基准线分量 $v_i$
        omega_current = torch.matmul(rotation, omega_home[:, None]).squeeze(-1)  # $R\omega_i$
        linear_current = (
            torch.cross(translation, omega_current, dim=-1)
            + torch.matmul(rotation, linear_home[:, None]).squeeze(-1)
        )  # $v_i'=p\times R\omega_i+Rv_i$
        current[:, target_joint] = torch.cat((omega_current, linear_current), dim=-1)  # 当前空间旋量
    return current


def selected_point_jacobian(
    spec: KinematicTreeSpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    r"""计算 sampled owner–JOINT edges 上的解析 material-point Jacobian。

    对当前 `{h}` 点 $y$ 与当前空间旋量 $(\omega_i,v_i)$：

    $$
    J_{g,i}^{h}(y)
    =
    \omega_i^h\times y+v_i^h.
    $$

    非祖先 edge 由拓扑 mask 乘成精确零，不用“未采样”替代结构零监督。
    """

    if joint_index.shape != owner_index.shape:
        raise ValueError("owner_index and joint_index must have identical [E] shape")
    owner_transforms = forward_owner_transforms(spec, q)  # `[B,G,4,4]` 当前 owner poses
    hand_points = transform_owner_points(owner_transforms, owner_index, local_points)  # `[B,E,3]`，单位 m
    current_screws = _current_spatial_screws(spec, q).index_select(1, joint_index)  # `[B,E,6]`
    omega = current_screws[..., :3]  # `[B,E,3]` 当前单位关节轴
    linear = current_screws[..., 3:]  # `[B,E,3]` 当前空间旋量线分量，单位 m
    jacobian = torch.cross(omega, hand_points, dim=-1) + linear  # $\partial y/\partial q_i$，单位 m/rad
    ancestor = spec.owner_ancestor_mask[owner_index, joint_index].to(q.dtype)  # `[E]` 结构祖先指示量
    return jacobian * ancestor[None, :, None]  # 非祖先列严格为零


__all__ = [
    "KinematicTreeSpec",
    "forward_owner_transforms",
    "selected_point_jacobian",
    "transform_owner_points",
]
