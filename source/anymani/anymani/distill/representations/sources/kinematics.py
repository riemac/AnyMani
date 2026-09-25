"""Product-of-exponentials kinematics and spatial screws. Joint angles use radians, translation uses metres, and owner points are expressed in hand frame {h}."""


from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg


@dataclass(frozen=True)
class EmbodimentGeometrySpec:


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
    owner_roles: tuple[str, ...] = ()
    owner_finger_names: tuple[str | None, ...] = ()
    owner_joint_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:


        if self.space_screws.ndim != 2 or self.space_screws.shape[-1] != 6:
            raise ValueError(f"space_screws must have shape [N_J,6], got {tuple(self.space_screws.shape)}")
        joint_count = self.space_screws.shape[0]
        if self.q_home.shape != (joint_count,):
            raise ValueError(f"q_home must have shape [{joint_count}], got {tuple(self.q_home.shape)}")
        if self.owner_home_transforms.ndim != 3 or self.owner_home_transforms.shape[-2:] != (4, 4):
            raise ValueError(
                "owner_home_transforms must have shape [G,4,4], "
                f"got {tuple(self.owner_home_transforms.shape)}"
            )
        owner_count = self.owner_home_transforms.shape[0]
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
        if self.owner_roles and len(self.owner_roles) != owner_count:
            raise ValueError("owner_roles must align with owner_home_transforms")
        if self.owner_finger_names and len(self.owner_finger_names) != owner_count:
            raise ValueError("owner_finger_names must align with owner_home_transforms")
        if self.owner_joint_indices and len(self.owner_joint_indices) != owner_count:
            raise ValueError("owner_joint_indices must align with owner_home_transforms")
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


        angular_norm = torch.linalg.vector_norm(self.space_screws[:, :3], dim=-1)
        if not torch.allclose(angular_norm, torch.ones_like(angular_norm), atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("each revolute space screw must have a unit angular axis")

    def to(self, *, device: torch.device | str, dtype: torch.dtype | None = None) -> EmbodimentGeometrySpec:


        target_dtype = dtype or self.space_screws.dtype
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
            owner_roles=self.owner_roles,
            owner_finger_names=self.owner_finger_names,
            owner_joint_indices=self.owner_joint_indices,
        )


def _validate_optional_graph_tensor(
    value: torch.Tensor | None,
    shape: tuple[int, ...],
    name: str,
) -> None:


    if value is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def lower_hand_geometry_semantics(
    semantics: HandGeometrySemanticsCfg,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_graph_distance: int = 8,
) -> EmbodimentGeometrySpec:


    if not dtype.is_floating_point:
        raise TypeError(f"kinematic dtype must be floating point, got {dtype}")
    if max_graph_distance < 1:
        raise ValueError("max_graph_distance must be at least one")
    target_device = torch.device(device)
    joint_count = len(semantics.active_joint_names)
    owner_count = len(semantics.owners)

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
    }
    link_ancestors: dict[str, tuple[int, ...]] = {semantics.palm_link: ()}

    q_home = torch.tensor(semantics.q_home_rad, device=target_device, dtype=dtype)
    joint_limits = torch.tensor(semantics.joint_limits_rad, device=target_device, dtype=dtype)
    space_screws = torch.empty(joint_count, 6, device=target_device, dtype=dtype)
    joint_ancestor_mask = torch.zeros(joint_count, joint_count, device=target_device, dtype=torch.bool)
    for joint in semantics.kinematic_joints:
        parent_home = link_home[joint.parent_link]
        origin_transform = _rigid_transform_from_rpy(
            joint.origin_rpy_rad,
            joint.origin_pos_m,
            device=target_device,
            dtype=dtype,
        )
        joint_frame_home = parent_home @ origin_transform
        parent_ancestors = link_ancestors[joint.parent_link]

        if joint.joint_type == "revolute":
            active_index = joint.active_joint_index
            if active_index is None:
                raise ValueError(f"revolute joint '{joint.joint_name}' is missing active_joint_index")
            joint_ancestor_mask[active_index, list(parent_ancestors)] = True
            axis_local = torch.tensor(joint.axis_local, device=target_device, dtype=dtype)
            omega_home = joint_frame_home[:3, :3] @ axis_local
            axis_point_home = joint_frame_home[:3, 3]
            linear_home = -torch.cross(omega_home, axis_point_home, dim=-1)
            space_screws[active_index] = torch.cat((omega_home, linear_home), dim=-1)
            home_rotation = _axis_rotation(axis_local, q_home[active_index])
            child_home = joint_frame_home @ home_rotation
            child_ancestors = (*parent_ancestors, active_index)
        else:
            child_home = joint_frame_home
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
        owner_roles=tuple(str(owner.role) for owner in semantics.owners),
        owner_finger_names=tuple(owner.finger_name for owner in semantics.owners),
        owner_joint_indices=tuple(
            next(
                (
                    int(item.active_joint_index)
                    for item in semantics.kinematic_joints
                    if item.joint_name == owner.joint_name and item.active_joint_index is not None
                ),
                -1,
            )
            if owner.role == "joint"
            else -1
            for owner in semantics.owners
        ),
    )


def _rigid_transform(
    rotation_flat: tuple[float, ...],
    translation: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:


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


    owner_count = len(semantics.owners)
    index_by_id = {owner.owner_id: owner.owner_index for owner in semantics.owners}
    parent_indices = [-1] * owner_count
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

    child_direction.copy_(parent_direction.transpose(0, 1))
    return (
        torch.tensor(parent_indices, device=device, dtype=torch.long),
        shortest,
        parent_direction,
        child_direction,
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:


    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
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


    omega = space_screw[:3]
    linear = space_screw[3:]
    omega_hat = _skew(omega)
    omega_hat_squared = omega_hat @ omega_hat
    identity3 = torch.eye(3, device=theta.device, dtype=theta.dtype)

    sin_theta = torch.sin(theta)[..., None, None]
    cos_theta = torch.cos(theta)[..., None, None]
    rotation = identity3 + sin_theta * omega_hat + (1.0 - cos_theta) * omega_hat_squared

    theta_matrix = theta[..., None, None]
    translation_operator = (
        theta_matrix * identity3
        + (1.0 - cos_theta) * omega_hat
        + (theta_matrix - sin_theta) * omega_hat_squared
    )
    translation = torch.matmul(translation_operator, linear[..., None]).squeeze(-1)

    transform = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)  # `[...,4,4]`
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def forward_owner_transforms_and_spatial_screws(
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:


    joint_count = spec.space_screws.shape[0]
    owner_count = spec.owner_home_transforms.shape[0]
    if q.ndim != 2 or q.shape[1] != joint_count:
        raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
    if q.device != spec.space_screws.device:
        raise ValueError("q and EmbodimentGeometrySpec tensors must be on the same device")

    delta_q = q - spec.q_home
    batch_size = q.shape[0]
    identity = torch.eye(4, device=q.device, dtype=q.dtype).view(1, 1, 4, 4)
    transform = identity.expand(batch_size, owner_count, 4, 4).clone()

    joint_exponentials = tuple(
        _revolute_twist_exp(spec.space_screws[joint_index], delta_q[:, joint_index])
        for joint_index in range(joint_count)
    )


    for joint_index in range(joint_count):
        is_ancestor = spec.owner_ancestor_mask[:, joint_index].view(1, owner_count, 1, 1)
        joint_transform = torch.where(
            is_ancestor,
            joint_exponentials[joint_index].unsqueeze(1),
            identity,
        )
        transform = transform @ joint_transform

    current_screws = _current_spatial_screws_from_exponentials(spec, joint_exponentials, q)
    owner_transforms = transform @ spec.owner_home_transforms.unsqueeze(0)
    return owner_transforms, current_screws


def forward_owner_transforms(spec: EmbodimentGeometrySpec, q: torch.Tensor) -> torch.Tensor:


    joint_count = spec.space_screws.shape[0]
    owner_count = spec.owner_home_transforms.shape[0]
    if q.ndim != 2 or q.shape[1] != joint_count:
        raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
    if q.device != spec.space_screws.device:
        raise ValueError("q and EmbodimentGeometrySpec tensors must be on the same device")
    delta_q = q - spec.q_home
    joint_exponentials = tuple(
        _revolute_twist_exp(spec.space_screws[joint_index], delta_q[:, joint_index])
        for joint_index in range(joint_count)
    )
    identity = torch.eye(4, device=q.device, dtype=q.dtype).view(1, 1, 4, 4)
    transform = identity.expand(q.shape[0], owner_count, 4, 4).clone()
    for joint_index in range(joint_count):
        is_ancestor = spec.owner_ancestor_mask[:, joint_index].view(1, owner_count, 1, 1)
        transform = transform @ torch.where(
            is_ancestor,
            joint_exponentials[joint_index].unsqueeze(1),
            identity,
        )
    return transform @ spec.owner_home_transforms.unsqueeze(0)


def transform_owner_points(
    owner_transforms: torch.Tensor,
    owner_index: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:


    if owner_index.ndim not in {1, 2}:
        raise ValueError("owner_index must have shape [E] or [B,E]")
    edge_count = owner_index.shape[-1]
    if local_points.shape not in {
        (edge_count, 3),
        (owner_transforms.shape[0], edge_count, 3),
    }:
        raise ValueError("local_points must have shape [E,3] or [B,E,3]")
    if owner_index.ndim == 1:
        selected = owner_transforms.index_select(1, owner_index)  # `[B,E,4,4]`
    else:
        if owner_index.shape[0] != owner_transforms.shape[0]:
            raise ValueError("batched owner_index must share B with owner_transforms")
        batch_index = torch.arange(owner_transforms.shape[0], device=owner_transforms.device).unsqueeze(1)
        selected = owner_transforms[batch_index, owner_index]
    rotation = selected[..., :3, :3]
    translation = selected[..., :3, 3]
    if local_points.ndim == 2:
        local_points = local_points.unsqueeze(0)
    return torch.matmul(rotation, local_points[..., None]).squeeze(-1) + translation


def _current_spatial_screws(spec: EmbodimentGeometrySpec, q: torch.Tensor) -> torch.Tensor:


    joint_count = spec.space_screws.shape[0]
    if q.ndim != 2 or q.shape[1] != joint_count:
        raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
    if q.device != spec.space_screws.device:
        raise ValueError("q and EmbodimentGeometrySpec tensors must be on the same device")
    delta_q = q - spec.q_home
    joint_exponentials = tuple(
        _revolute_twist_exp(spec.space_screws[joint_index], delta_q[:, joint_index])
        for joint_index in range(joint_count)
    )
    return _current_spatial_screws_from_exponentials(spec, joint_exponentials, q)


def _current_spatial_screws_from_exponentials(
    spec: EmbodimentGeometrySpec,
    joint_exponentials: tuple[torch.Tensor, ...],
    q: torch.Tensor,
) -> torch.Tensor:


    joint_count = spec.space_screws.shape[0]
    current = torch.empty(q.shape[0], joint_count, 6, device=q.device, dtype=q.dtype)
    for target_joint in range(joint_count):
        prefix = torch.eye(4, device=q.device, dtype=q.dtype).expand(q.shape[0], 4, 4).clone()
        for source_joint in range(joint_count):
            if bool(spec.joint_ancestor_mask[target_joint, source_joint]):
                prefix = prefix @ joint_exponentials[source_joint]
        rotation = prefix[:, :3, :3]
        translation = prefix[:, :3, 3]
        omega_home = spec.space_screws[target_joint, :3]
        linear_home = spec.space_screws[target_joint, 3:]
        omega_current = torch.matmul(rotation, omega_home[:, None]).squeeze(-1)
        linear_current = torch.cross(translation, omega_current, dim=-1) + torch.matmul(
            rotation, linear_home[:, None]
        ).squeeze(-1)
        current[:, target_joint] = torch.cat((omega_current, linear_current), dim=-1)
    return current


def selected_point_jacobian(
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_points: torch.Tensor,
    *,
    owner_transforms: torch.Tensor | None = None,
    current_spatial_screws: torch.Tensor | None = None,
) -> torch.Tensor:


    if joint_index.shape != owner_index.shape or owner_index.ndim not in {1, 2}:
        raise ValueError("owner_index and joint_index must have identical [E] or [B,E] shape")
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q)
    expected_shape = (q.shape[0], spec.owner_home_transforms.shape[0], 4, 4)
    if (
        owner_transforms.shape != expected_shape
        or owner_transforms.device != q.device
        or owner_transforms.dtype != q.dtype
        or owner_transforms.requires_grad
    ):
        raise ValueError("owner_transforms must be detached [B,G,4,4] matching q/spec")
    hand_points = transform_owner_points(owner_transforms, owner_index, local_points)
    if current_spatial_screws is None:
        current_all = _current_spatial_screws(spec, q)
    else:
        expected_screw_shape = (q.shape[0], spec.space_screws.shape[0], 6)
        if (
            current_spatial_screws.shape != expected_screw_shape
            or current_spatial_screws.device != q.device
            or current_spatial_screws.dtype != q.dtype
            or current_spatial_screws.requires_grad
        ):
            raise ValueError("current_spatial_screws must be detached [B,N_J,6] matching q/spec")
        current_all = current_spatial_screws
    if joint_index.ndim == 1:
        current_screws = current_all.index_select(1, joint_index)
        ancestor = spec.owner_ancestor_mask[owner_index, joint_index].to(q.dtype).unsqueeze(0)
    else:
        if joint_index.shape[0] != q.shape[0]:
            raise ValueError("batched joint_index must share B with q")
        batch_index = torch.arange(q.shape[0], device=q.device).unsqueeze(1)
        current_screws = current_all[batch_index, joint_index]
        ancestor = spec.owner_ancestor_mask[owner_index, joint_index].to(q.dtype)
    omega = current_screws[..., :3]
    linear = current_screws[..., 3:]
    jacobian = torch.cross(omega, hand_points, dim=-1) + linear
    return jacobian * ancestor.unsqueeze(-1)


__all__ = [
    "EmbodimentGeometrySpec",
    "forward_owner_transforms",
    "forward_owner_transforms_and_spatial_screws",
    "lower_hand_geometry_semantics",
    "selected_point_jacobian",
    "transform_owner_points",
]
