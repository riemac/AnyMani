r"""资产静态语义到当前构型物理几何的纯 PyTorch 运动学实现。

设 PALM/JOINT/TIP 表面归属体 $g$ 受 $n_g$ 个有序活动关节影响。$q_i$ 是第 $i$ 个
祖先 JOINT 的当前角度，$q_{\mathfrak m,i}^{0}$ 是资产基准构型角，二者单位均为 rad；
$\Delta q_i=q_i-q_{\mathfrak m,i}^{0}$。$\mathcal S_i\in\mathbb R^6$ 是基准构型空间旋量，
角分量无量纲、线分量单位 m；$T_{hg}(q_{\mathfrak m}^{0})\in SE(3)$ 是归属体局部坐标
到 ``{h}`` 的基准刚体变换。当前位姿满足指数积公式：

$$
T_{hg}(q)
=
e^{[\mathcal S_1]\Delta q_1}
\cdots
e^{[\mathcal S_{n_g}]\Delta q_{n_g}}
T_{hg}(q_{\mathfrak m}^{0}).
$$

资产基准构型不强制为全零；若实现直接把当前 $q$ 当作从零基准开始的指数坐标，会在存在
关节零偏或非零 $q_{\mathfrak m}^{0}$ 时产生错误物理表面。

输入必须保留祖先关系、严格链顺序、父子拓扑、表面归属与有效掩码。纯当前几何监督不需要
$\dot q$、动作、历史、指令、接触、物体状态、质量/惯量或关节限位；这些量可以服务策略、采样与控制
边界，但不决定给定 $q$ 下的刚体 collision geometry。

规范边界：同一物理关节可同时重写旋量符号与坐标符号，关节零偏也会联动改变 $q$ 与基准几何。
运动学来源必须允许成对规范改写，不能假设不同 URDF 的轴向、原点与零位约定天然一致。

本模块使用空间旋量约定：

$$
\mathcal S_i
=
\begin{bmatrix}\omega_i\\v_i\end{bmatrix},
\qquad
\dot y
=
\omega_i\times y+v_i.
$$

对穿过轴上一点 $p_i$ 的单位转轴，$v_i=-\omega_i\times p_i$。空间旋量表达于基准手部语义
坐标系 `{h}`，不随归属体局部坐标系重参数化改变。owner home transform 负责把每个碰撞载体
的局部坐标接到同一 `{h}` 中。

手型可能包含多根独立手指。全局 JOINT 顺序只规定张量索引，不表示所有关节是一条串联链。
`owner_ancestor_mask[g,i]` 决定 JOINT $i$ 是否进入归属体 $g$ 的指数积；
`joint_ancestor_mask[i,j]` 决定 JOINT $j$ 是否变换 JOINT $i$ 的当前空间旋量。其他手指即使在
规范顺序中更早，也不能污染当前轴线。

一阶教师只对抽样边计算物质点 Jacobian。对当前物质点 $y_g^*$：

$$
J_{g,i}^{h}
=
\omega_i^h(q)\times y_g^*+v_i^h(q).
$$

若 JOINT $i$ 不是归属体 $g$ 的祖先，拓扑掩码把该列精确置零。这是模型必须学习或被结构
约束的物理零，不能用“没有抽到该边”代替。

本模块是 ``representations.sources`` 的动态 embodiment 真源；它只消费 assets 交付的类型化
sidecar 语义，不重新解析 URDF。全部函数只依赖 PyTorch，不导入 Isaac Sim；普通 SSL 训练和默认
pytest 不启动 Kit、USD 或 PhysX。

静态 lowering 数据流：

```text
HandGeometrySemanticsCfg
  palm origin + T_ha
  complete fixed/revolute joint tree
  q_home + limits
  owner reference links + parent graph
  collision carrier links
        |
        v
lower_hand_geometry_semantics
  space_screws                 [N_J, 6]
  owner_home_transforms        [G, 4, 4]
  owner_ancestor_mask          [G, N_J]
  joint_ancestor_mask          [N_J, N_J]
  owner graph distances        [G, G]
  component_owner_transforms   [C, 4, 4]
```

`lower_hand_geometry_semantics` 只在 asset/cache materialization 时执行一次。结果可通过 ``to()`` 整体
迁移 GPU，并在同结构 minibatch 的所有训练 step 复用。当前 q 的 FK 才是 batch-dependent；limits、
sidecar 字符串和 mesh 几何都不进入每步 POE。

树遍历从 palm link 开始。每个 joint 先复合固定 origin；revolute joint 在 origin frame 中定义局部
单位轴。轴线在 home 构型的 `{h}` 表达为：

$$
\omega_i^h=R_{hj}(q_{home})a_i,
\qquad
p_i^h=t_{hj}(q_{home}),
\qquad
v_i^h=-\omega_i^h\times p_i^h.
$$

自身 q_home rotation 影响 child link home transform，但不改变自身轴线；祖先 q_home rotation 同时
改变轴方向和轴上一点。由此得到的 space screw 与 owner home transform 共同定义以
$\Delta q=q-q_{home}$ 为指数坐标的 POE，既支持零 home，也支持任意显式非零 home。

branched hand 不能按全局 joint 列表机械做 prefix product。`joint_ancestor_mask[i,j]` 只让目标 JOINT
i 的严格祖先 j 参与当前空间旋量变换；另一根 finger 即使全局 index 更小，也不能进入 prefix。
`owner_ancestor_mask[g,i]` 包含会移动 owner reference link 的全部活动关节，其中 JOINT owner 包含
自身，PALM 全零，TIP 包含整根所属 finger 的活动链。

owner graph 与运动学 ancestor 是不同关系。owner parent graph服务 Transformer soft bias 和 adjacent
query：PALM 为根，每根 finger 的 JOINT 串联，TIP 接最后 JOINT。它不替代 kinematic masks，也不把
不同 finger 变成串联机构。shortest distance 是无向路径；parent/child direction 只在祖先/后代可达
时记录距离，其他关系进入末桶。

collision component 可以附着在 owner reference link 之外的 fixed descendant。为避免 distill 或
geometry cache 重走 link tree，lowering 预计算：

$$
T_{g c}(q_{home})
=
T_{h g}(q_{home})^{-1}
T_{h,carrier}(q_{home})
T_{carrier,c}.
$$

该 component-to-owner 变换只用于静态 owner union。当前 q 下整个 owner union 由单一
$T_{hg}(q)$ 刚体移动；若一个拟议 owner 内的 components 并非刚性共动，就说明 owner sidecar 语义错误，
不能靠每 component 动态变换掩盖。

`selected_point_jacobian` 只物化 sampled edge。输入 local point 可以是 `[E,3]`，表示 batch 共享
物质点，也可以是 `[B,E,3]`，表示最近点后端逐样本选择的物质点。两种情况输出统一为 `[B,E,3]`
且单位 m/rad。非祖先 edge 最后再乘拓扑 mask，保证浮点运算不会留下小残差。

`joint_limits` 保存在规格中供 scrambled Sobol/合法 q sampler 使用，但 ``forward_owner_transforms``
完全不读取它。固定 q 和静态几何时只改 limits，owner transforms、轴线、Jacobian 与 encoder evidence
必须逐元素不变。q_home 允许在 limits 外，因为它是坐标参考，不是 rollout 合法状态承诺。

数值实现不使用小角度分支。Rodrigues/SE(3) 指数中的 sin/cos 在 theta=0 具有解析稳定值，translation
operator 直接使用单位转轴闭式；contract test 以 float64 中心有限差分验证，训练使用 float32。若未来
引入 prismatic joint，必须扩展 screw exponential 和 schema 类型，不能把零角轴塞进 revolute 路径。

Isaac importer parity 是本模块的外部证据而非实现依赖。显式 runtime smoke 应比较给定 q 的 link/owner
pose，并记录 importer 是否 merge fixed joints、轴/RPY 约定和 `{a}->{h}` calibration；普通测试只验证
纯张量公式，保证 SSL 不启动 simulator。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg


@dataclass(frozen=True)
class EmbodimentGeometrySpec:
    r"""同一结构模式下的空间旋量、基准位姿与祖先关系。

    空间旋量使用顺序 `[omega_x, omega_y, omega_z, v_x, v_y, v_z]`，满足
    $\dot y=\omega\times y+v$。`owner_ancestor_mask[g,i]` 指示 JOINT $i$
    是否影响归属体 $g$；`joint_ancestor_mask[i,j]` 指示 JOINT $j$ 是否是 JOINT
    $i$ 的严格祖先。后者防止 canonical joint order 中更早出现的其他手指关节错误地
    变换当前轴线。

    Attributes:
        space_screws (torch.Tensor): 基准 `{h}` 中的空间旋量，形状 `[N_J,6]`。
        q_home (torch.Tensor): 资产显式基准构型，形状 `[N_J]`，单位 rad。
        owner_home_transforms (torch.Tensor): 归属体局部坐标到 `{h}` 的基准变换，形状 `[G,4,4]`。
        owner_ancestor_mask (torch.Tensor): 形状 `[G,N_J]` 的祖先布尔矩阵。
        joint_ancestor_mask (torch.Tensor): 形状 `[N_J,N_J]` 的严格祖先布尔矩阵。
    """

    space_screws: torch.Tensor
    q_home: torch.Tensor
    owner_home_transforms: torch.Tensor
    owner_ancestor_mask: torch.Tensor
    joint_ancestor_mask: torch.Tensor
    joint_limits: torch.Tensor | None = None
    owner_parent_indices: torch.Tensor | None = None
    owner_graph_shortest: torch.Tensor | None = None
    owner_graph_parent: torch.Tensor | None = None
    owner_graph_child: torch.Tensor | None = None
    component_owner_indices: torch.Tensor | None = None
    component_owner_local_transforms: torch.Tensor | None = None
    owner_ids: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        r"""在运动学来源边界验证形状、数值类型与旋量单位轴。

        当前 AnyMani 手资产只把可驱动转动关节纳入 $q$，因此每个空间旋量的角分量必须是单位轴。
        固定关节已经在资产 lowering 时吸收到归属体基准变换，不作为零旋量混入活动关节轴。
        """

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
        if self.joint_limits is not None and self.joint_limits.shape != (joint_count, 2):
            raise ValueError(f"joint_limits must have shape [{joint_count},2]")
        if self.owner_ids and len(self.owner_ids) != owner_count:
            raise ValueError("owner_ids must align with owner_home_transforms")
        if self.joint_names and len(self.joint_names) != joint_count:
            raise ValueError("joint_names must align with space_screws")
        _validate_optional_graph_tensor(self.owner_parent_indices, (owner_count,), "owner_parent_indices")
        _validate_optional_graph_tensor(
            self.owner_graph_shortest, (owner_count, owner_count), "owner_graph_shortest"
        )
        _validate_optional_graph_tensor(self.owner_graph_parent, (owner_count, owner_count), "owner_graph_parent")
        _validate_optional_graph_tensor(self.owner_graph_child, (owner_count, owner_count), "owner_graph_child")
        if self.component_owner_indices is not None and (
            self.component_owner_indices.ndim != 1
            or torch.any(self.component_owner_indices < 0)
            or torch.any(self.component_owner_indices >= owner_count)
        ):
            raise ValueError("component_owner_indices must be a valid owner index vector")
        if self.component_owner_local_transforms is not None:
            component_count = (
                self.component_owner_indices.numel() if self.component_owner_indices is not None else None
            )
            if component_count is None or self.component_owner_local_transforms.shape != (component_count, 4, 4):
                raise ValueError("component_owner_local_transforms must have shape [C,4,4]")

        # 当前 AnyMani 合同只含 revolute JOINT，因此旋量角分量必须是单位轴。
        angular_norm = torch.linalg.vector_norm(self.space_screws[:, :3], dim=-1)  # $\|\omega_i\|_2$
        if not torch.allclose(angular_norm, torch.ones_like(angular_norm), atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("each revolute space screw must have a unit angular axis")

    def to(self, *, device: torch.device | str, dtype: torch.dtype | None = None) -> EmbodimentGeometrySpec:
        r"""把同一结构模式的静态张量整体迁移到目标设备与浮点 dtype。"""

        target_dtype = dtype or self.space_screws.dtype  # 浮点精度默认保持 source 规格
        return EmbodimentGeometrySpec(
            space_screws=self.space_screws.to(device=device, dtype=target_dtype),
            q_home=self.q_home.to(device=device, dtype=target_dtype),
            owner_home_transforms=self.owner_home_transforms.to(device=device, dtype=target_dtype),
            owner_ancestor_mask=self.owner_ancestor_mask.to(device=device),
            joint_ancestor_mask=self.joint_ancestor_mask.to(device=device),
            joint_limits=None
            if self.joint_limits is None
            else self.joint_limits.to(device=device, dtype=target_dtype),
            owner_parent_indices=None
            if self.owner_parent_indices is None
            else self.owner_parent_indices.to(device=device),
            owner_graph_shortest=None
            if self.owner_graph_shortest is None
            else self.owner_graph_shortest.to(device=device),
            owner_graph_parent=None
            if self.owner_graph_parent is None
            else self.owner_graph_parent.to(device=device),
            owner_graph_child=None
            if self.owner_graph_child is None
            else self.owner_graph_child.to(device=device),
            component_owner_indices=None
            if self.component_owner_indices is None
            else self.component_owner_indices.to(device=device),
            component_owner_local_transforms=None
            if self.component_owner_local_transforms is None
            else self.component_owner_local_transforms.to(device=device, dtype=target_dtype),
            owner_ids=self.owner_ids,
            joint_names=self.joint_names,
        )


def _validate_optional_graph_tensor(
    value: torch.Tensor | None,
    shape: tuple[int, ...],
    name: str,
) -> None:
    """验证可选图 lower 张量；旧纯数学 contract 可省略这些 metadata。"""

    if value is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def lower_hand_geometry_semantics(
    semantics: HandGeometrySemanticsCfg,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_graph_distance: int = 8,
) -> EmbodimentGeometrySpec:
    r"""把 bank 交付的静态资产语义 lower 成 GPU 可驻留的动态运动学规格。

    Args:
        semantics (HandGeometrySemanticsCfg): 已校验/迁移的资产静态事实。
        device (torch.device | str): 目标张量设备；默认 CPU 便于离线 cache 与合同测试。
        dtype (torch.dtype): 运动学浮点精度；训练通常 float32，数值 oracle 使用 float64。
        max_graph_distance (int): 图关系桶的最大整数值，远距离统一截断。

    Returns:
        EmbodimentGeometrySpec: 空间旋量、home 位姿、祖先掩码与实体图关系。

    计算先在显式 home 构型做一次树遍历。对 joint frame 原点 $p_i^h$ 与局部单位轴
    $a_i$：

    $$
    \omega_i^h=R_{hj}a_i,
    \qquad
    v_i^h=-\omega_i^h\times p_i^h.
    $$

    child link 的 home 位姿包含自身 $q_{home,i}$ 旋转；当前批量 FK 此后只使用
    $\Delta q_i=q_i-q_{home,i}$，因此任意非零 home 不会被重复施加。
    """

    if not dtype.is_floating_point:
        raise TypeError(f"kinematic dtype must be floating point, got {dtype}")
    if max_graph_distance < 1:
        raise ValueError("max_graph_distance must be at least one")
    target_device = torch.device(device)
    joint_count = len(semantics.active_joint_names)  # 活动关节数 $N_J$
    owner_count = len(semantics.owners)  # 归属体/实体数 $G=N_E$

    transform_ha = _rigid_transform(
        semantics.asset_to_hand_rotation,
        semantics.asset_to_hand_translation_m,
        device=target_device,
        dtype=dtype,
    )  # `{a}` -> `{h}`
    transform_ap = _rigid_transform_from_rpy(
        semantics.palm_origin_rpy_rad,
        semantics.palm_origin_pos_m,
        device=target_device,
        dtype=dtype,
    )  # palm link -> `{a}`
    link_home: dict[str, torch.Tensor] = {
        semantics.palm_link: transform_ha @ transform_ap
    }  # 每个 link 局部坐标到 `{h}` 的 home 变换
    link_ancestors: dict[str, tuple[int, ...]] = {semantics.palm_link: ()}

    q_home = torch.tensor(semantics.q_home_rad, device=target_device, dtype=dtype)
    joint_limits = torch.tensor(semantics.joint_limits_rad, device=target_device, dtype=dtype)
    space_screws = torch.empty(joint_count, 6, device=target_device, dtype=dtype)
    joint_ancestor_mask = torch.zeros(joint_count, joint_count, device=target_device, dtype=torch.bool)
    for joint in semantics.kinematic_joints:
        parent_home = link_home[joint.parent_link]  # 父 link home 位姿
        origin_transform = _rigid_transform_from_rpy(
            joint.origin_rpy_rad,
            joint.origin_pos_m,
            device=target_device,
            dtype=dtype,
        )
        joint_frame_home = parent_home @ origin_transform  # joint frame 在 `{h}` 中的 home 位姿
        parent_ancestors = link_ancestors[joint.parent_link]

        if joint.joint_type == "revolute":
            active_index = joint.active_joint_index
            if active_index is None:
                raise ValueError(f"revolute joint '{joint.joint_name}' is missing active_joint_index")
            joint_ancestor_mask[active_index, list(parent_ancestors)] = True
            axis_local = torch.tensor(joint.axis_local, device=target_device, dtype=dtype)
            omega_home = joint_frame_home[:3, :3] @ axis_local  # 当前 home 空间角轴
            axis_point_home = joint_frame_home[:3, 3]  # joint frame 原点是轴上一点，单位 m
            linear_home = -torch.cross(omega_home, axis_point_home, dim=-1)  # $v=-\omega\times p$
            space_screws[active_index] = torch.cat((omega_home, linear_home), dim=-1)
            home_rotation = _axis_rotation(axis_local, q_home[active_index])
            child_home = joint_frame_home @ home_rotation  # 自身非零 home 只进入基准 child 位姿
            child_ancestors = (*parent_ancestors, active_index)
        else:
            child_home = joint_frame_home  # fixed joint 不增加广义坐标轴
            child_ancestors = parent_ancestors

        link_home[joint.child_link] = child_home
        link_ancestors[joint.child_link] = child_ancestors

    owner_home_transforms = torch.stack(
        tuple(link_home[owner.reference_link] for owner in semantics.owners),
        dim=0,
    )  # `[G,4,4]`
    owner_ancestor_mask = torch.zeros(owner_count, joint_count, device=target_device, dtype=torch.bool)
    for owner in semantics.owners:
        owner_ancestor_mask[owner.owner_index, list(link_ancestors[owner.reference_link])] = True

    owner_parent_indices, graph_shortest, graph_parent, graph_child = _lower_owner_graph(
        semantics,
        max_graph_distance=max_graph_distance,
        device=target_device,
    )
    owner_index_by_id = {owner.owner_id: owner.owner_index for owner in semantics.owners}
    component_owner_indices = torch.tensor(
        [owner_index_by_id[component.owner_id] for component in semantics.components],
        device=target_device,
        dtype=torch.long,
    )
    component_owner_local_transforms = torch.stack(
        tuple(
            torch.linalg.inv(owner_home_transforms[owner_index_by_id[component.owner_id]])
            @ link_home[component.carrier_link]
            @ _rigid_transform_from_rpy(
                component.origin_rpy_rad,
                component.origin_pos_m,
                device=target_device,
                dtype=dtype,
            )
            for component in semantics.components
        ),
        dim=0,
    )  # collision local frame -> owner reference link
    return EmbodimentGeometrySpec(
        space_screws=space_screws,
        q_home=q_home,
        owner_home_transforms=owner_home_transforms,
        owner_ancestor_mask=owner_ancestor_mask,
        joint_ancestor_mask=joint_ancestor_mask,
        joint_limits=joint_limits,
        owner_parent_indices=owner_parent_indices,
        owner_graph_shortest=graph_shortest,
        owner_graph_parent=graph_parent,
        owner_graph_child=graph_child,
        component_owner_indices=component_owner_indices,
        component_owner_local_transforms=component_owner_local_transforms,
        owner_ids=tuple(owner.owner_id for owner in semantics.owners),
        joint_names=semantics.active_joint_names,
    )


def _rigid_transform(
    rotation_flat: tuple[float, ...],
    translation: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """由按行展开的旋转和平移构造齐次刚体变换。"""

    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = torch.tensor(rotation_flat, device=device, dtype=dtype).reshape(3, 3)
    transform[:3, 3] = torch.tensor(translation, device=device, dtype=dtype)
    return transform


def _rigid_transform_from_rpy(
    rpy: tuple[float, float, float],
    translation: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""按 URDF 固定轴约定构造 $R_z(yaw)R_y(pitch)R_x(roll)$ 齐次变换。"""

    roll, pitch, yaw = (torch.tensor(value, device=device, dtype=dtype) for value in rpy)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    rotation_x = torch.stack(
        (one, zero, zero, zero, torch.cos(roll), -torch.sin(roll), zero, torch.sin(roll), torch.cos(roll))
    ).reshape(3, 3)
    rotation_y = torch.stack(
        (torch.cos(pitch), zero, torch.sin(pitch), zero, one, zero, -torch.sin(pitch), zero, torch.cos(pitch))
    ).reshape(3, 3)
    rotation_z = torch.stack(
        (torch.cos(yaw), -torch.sin(yaw), zero, torch.sin(yaw), torch.cos(yaw), zero, zero, zero, one)
    ).reshape(3, 3)
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = torch.tensor(translation, device=device, dtype=dtype)
    return transform


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    r"""构造绕 joint-local 单位轴的纯旋转齐次变换。"""

    axis_hat = _skew(axis)
    rotation = (
        torch.eye(3, device=axis.device, dtype=axis.dtype)
        + torch.sin(angle) * axis_hat
        + (1.0 - torch.cos(angle)) * (axis_hat @ axis_hat)
    )
    transform = torch.eye(4, device=axis.device, dtype=axis.dtype)
    transform[:3, :3] = rotation
    return transform


def _lower_owner_graph(
    semantics: HandGeometrySemanticsCfg,
    *,
    max_graph_distance: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""把 owner parent 树 lower 成最短、父向和子向图距离桶。"""

    owner_count = len(semantics.owners)
    index_by_id = {owner.owner_id: owner.owner_index for owner in semantics.owners}
    parent_indices = [-1] * owner_count  # PALM 根使用 -1
    adjacency: list[list[int]] = [[] for _ in range(owner_count)]
    for owner in semantics.owners:
        if owner.parent_owner_id is None:
            continue
        parent_index = index_by_id[owner.parent_owner_id]
        parent_indices[owner.owner_index] = parent_index
        adjacency[owner.owner_index].append(parent_index)
        adjacency[parent_index].append(owner.owner_index)

    shortest = torch.full((owner_count, owner_count), max_graph_distance, device=device, dtype=torch.long)
    parent_direction = torch.full_like(shortest, max_graph_distance)
    child_direction = torch.full_like(shortest, max_graph_distance)
    for source in range(owner_count):
        shortest[source, source] = 0
        frontier = [source]
        visited = {source}
        distance = 0
        while frontier:
            next_frontier: list[int] = []
            for node in frontier:
                shortest[source, node] = min(distance, max_graph_distance)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            distance += 1

        parent_direction[source, source] = 0
        ancestor = parent_indices[source]
        ancestor_distance = 1
        while ancestor >= 0:
            parent_direction[source, ancestor] = min(ancestor_distance, max_graph_distance)
            ancestor = parent_indices[ancestor]
            ancestor_distance += 1

    child_direction.copy_(parent_direction.transpose(0, 1))  # 父向关系的转置即子向关系
    return (
        torch.tensor(parent_indices, device=device, dtype=torch.long),
        shortest,
        parent_direction,
        child_direction,
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


def forward_owner_transforms(spec: EmbodimentGeometrySpec, q: torch.Tensor) -> torch.Tensor:
    r"""按归属体祖先掩码计算批量指数积刚体位姿。

    Args:
        spec (EmbodimentGeometrySpec): 当前同构结构模式的静态运动学事实。
        q (torch.Tensor): 当前物理关节构型，形状 `[B,N_J]`，单位 rad。

    Returns:
        torch.Tensor: 归属体局部坐标到 `{h}` 的变换，形状 `[B,G,4,4]`。
    """

    joint_count = spec.space_screws.shape[0]  # 活动关节数 $N_J$
    owner_count = spec.owner_home_transforms.shape[0]  # 归属体数 $G$
    if q.ndim != 2 or q.shape[1] != joint_count:
        raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
    if q.device != spec.space_screws.device:
        raise ValueError("q and EmbodimentGeometrySpec tensors must be on the same device")

    delta_q = q - spec.q_home  # $\Delta q=q-q_{home}$，形状 `[B,N_J]`，单位 rad
    batch_size = q.shape[0]  # 同结构模式微批次大小 $B$
    transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(batch_size, owner_count, 4, 4).clone()

    # 每个规范 JOINT 依次左乘；非祖先通过 $\theta=0$ 产生严格单位变换。
    for joint_index in range(joint_count):
        owner_theta = delta_q[:, joint_index : joint_index + 1] * spec.owner_ancestor_mask[
            :, joint_index
        ].to(q.dtype).unsqueeze(0)  # `[B,G]`，非祖先严格为 0
        joint_transform = _revolute_twist_exp(spec.space_screws[joint_index], owner_theta)  # `[B,G,4,4]`
        transform = transform @ joint_transform  # 按规范链顺序复合指数积

    return transform @ spec.owner_home_transforms.unsqueeze(0)  # $T_{hg}(q)=\prod e^{S_i\Delta q_i}M_g$


def transform_owner_points(
    owner_transforms: torch.Tensor,
    owner_index: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    r"""把选中的归属体局部物质点变换到 `{h}`。

    Args:
        owner_transforms (torch.Tensor): 形状 `[B,G,4,4]` 的当前归属体位姿。
        owner_index (torch.Tensor): 形状 `[E]` 的归属体选择索引。
        local_points (torch.Tensor): 形状 `[E,3]` 或 `[B,E,3]` 的归属体局部点，单位 m。

    Returns:
        torch.Tensor: 形状 `[B,E,3]` 的 `{h}` 点，单位 m。
    """

    edge_count = owner_index.numel()
    if owner_index.ndim != 1 or local_points.shape not in {
        (edge_count, 3),
        (owner_transforms.shape[0], edge_count, 3),
    }:
        raise ValueError("owner_index/local_points must have shapes [E] and [E,3] or [B,E,3]")
    selected = owner_transforms.index_select(1, owner_index)  # `[B,E,4,4]`，选中归属体位姿
    rotation = selected[..., :3, :3]  # `[B,E,3,3]`，归属体局部坐标 -> `{h}`
    translation = selected[..., :3, 3]  # `[B,E,3]`，归属体原点在 `{h}` 中的位置，单位 m
    if local_points.ndim == 2:
        local_points = local_points.unsqueeze(0)  # 跨 batch 复用同一组物质点
    return torch.matmul(rotation, local_points[..., None]).squeeze(-1) + translation  # $Rp^{local}+t$


def _current_spatial_screws(spec: EmbodimentGeometrySpec, q: torch.Tensor) -> torch.Tensor:
    r"""把每个基准空间旋量经其严格祖先变换到当前 `{h}`。"""

    batch_size, joint_count = q.shape  # 同结构模式批次与 JOINT 轴
    delta_q = q - spec.q_home  # 当前相对基准的物理角度，单位 rad
    current = torch.empty(batch_size, joint_count, 6, device=q.device, dtype=q.dtype)  # `[B,N_J,6]`

    # 每个 JOINT 只复合自身严格祖先；其他手指即使在规范顺序中更早也不参与。
    for target_joint in range(joint_count):
        prefix = torch.eye(4, device=q.device, dtype=q.dtype).expand(batch_size, 4, 4).clone()  # $T_{prefix}$
        for source_joint in range(joint_count):
            if not bool(spec.joint_ancestor_mask[target_joint, source_joint]):
                continue
            prefix = prefix @ _revolute_twist_exp(
                spec.space_screws[source_joint], delta_q[:, source_joint]
            )  # 只复合目标 JOINT 的严格祖先

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
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    r"""计算抽样归属体—JOINT 边上的解析物质点 Jacobian。

    对当前 `{h}` 点 $y$ 与当前空间旋量 $(\omega_i,v_i)$：

    $$
    J_{g,i}^{h}(y)
    =
    \omega_i^h\times y+v_i^h.
    $$

    ``local_points`` 可为跨批次共享的 `[E,3]`，也可为最近点后端逐样本返回的 `[B,E,3]`。
    非祖先边由拓扑掩码乘成精确零，不用“未采样”替代结构零监督。
    """

    if joint_index.shape != owner_index.shape:
        raise ValueError("owner_index and joint_index must have identical [E] shape")
    owner_transforms = forward_owner_transforms(spec, q)  # `[B,G,4,4]` 当前归属体位姿
    hand_points = transform_owner_points(owner_transforms, owner_index, local_points)  # `[B,E,3]`，单位 m
    current_screws = _current_spatial_screws(spec, q).index_select(1, joint_index)  # `[B,E,6]`
    omega = current_screws[..., :3]  # `[B,E,3]` 当前单位关节轴
    linear = current_screws[..., 3:]  # `[B,E,3]` 当前空间旋量线分量，单位 m
    jacobian = torch.cross(omega, hand_points, dim=-1) + linear  # $\partial y/\partial q_i$，单位 m/rad
    ancestor = spec.owner_ancestor_mask[owner_index, joint_index].to(q.dtype)  # `[E]` 结构祖先指示量
    return jacobian * ancestor[None, :, None]  # 非祖先列严格为零


__all__ = [
    "EmbodimentGeometrySpec",
    "forward_owner_transforms",
    "lower_hand_geometry_semantics",
    "selected_point_jacobian",
    "transform_owner_points",
]
