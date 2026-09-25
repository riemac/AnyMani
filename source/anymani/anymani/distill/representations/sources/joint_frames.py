"""Build static joint kinematics and single-point FK targets from typed geometry.

Each real joint uses 15 features: zero-angle translation divided by 0.1 m, a row-major rotation matrix, and a local unit axis. The zero-angle transform maps the nearest active parent's child frame to the joint frame; fixed roots and spacers are folded into it. A root uses the semantic hand frame ``{h}``.

FK uses physical joint angles in radians and returns joint-frame origins in metres, not collision centroids or off-axis fingertip points. The zero pose comes from the shared POE implementation, so nonzero ``q_home`` is not added twice. Inputs come from typed ``HandContainer`` geometry; routing is explicit and no task, contact, or object state is consumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg

from .kinematics import forward_owner_transforms, lower_hand_geometry_semantics

LENGTH_SCALE_M = 0.1  # Shared scale preserves between-hand size differences.
JOINT_KINEMATICS_WIDTH = 15  # Translation 3 + rotation 9 + local axis 3.


@dataclass(frozen=True)
class JointKinematicsBank:
    """Batched physical kinematics on a fixed canonical joint axis.

    ``features`` is ``[A,J,15]`` with translation scaled by 0.1 m, a rotation matrix, and a unit axis. ``parent_slot`` and ``depth`` are ``[A,J]``; roots use parent -1 and depth 0, while ghosts use depth -1. Finger chains remain independent.
    """

    features: torch.Tensor
    parent_slot: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        """Reject invalid rotations, axes, ghost parents, or topology at assembly time."""
        if self.features.ndim != 3 or self.features.shape[-1] != JOINT_KINEMATICS_WIDTH:
            raise ValueError("joint kinematics features must have shape [A,J,15]")
        shape = self.features.shape[:2]  # Asset rows by canonical joint slots.
        if any(x.shape != shape for x in (self.parent_slot, self.depth, self.valid)):
            raise ValueError("joint kinematics metadata shape disagrees with features")
        if not shape[0] or not shape[1] or self.valid.dtype != torch.bool:
            raise ValueError("joint kinematics needs nonempty axes and boolean valid mask")
        if self.parent_slot.dtype != torch.long or self.depth.dtype != torch.long:
            raise ValueError("joint parent and depth must be int64")
        if len({x.device for x in (self.features, self.parent_slot, self.depth, self.valid)}) != 1:
            raise ValueError("joint kinematics tensors must share a device")
        if not self.features.dtype.is_floating_point or not torch.isfinite(self.features).all():
            raise ValueError("joint kinematics features must be finite floating point")
        if torch.count_nonzero(self.features[~self.valid]) or (self.depth[~self.valid] != -1).any():
            raise ValueError("ghost joint features must be zero and depth -1")
        parent = self.parent_slot  # Static validation, outside the per-step path.
        if ((parent < -1) | (parent >= shape[1])).any():
            raise ValueError("joint parent slot lies outside the canonical axis")
        has_parent = self.valid & (parent >= 0)  # Only active joints participate in parent chains.
        safe_parent = parent.clamp_min(0)  # Roots map to slot 0 and are excluded by has_parent.
        if (has_parent & ~self.valid.gather(1, safe_parent)).any():
            raise ValueError("active joint parent must be another active joint")
        parent_depth = self.depth.gather(1, safe_parent)
        if (has_parent & (self.depth != parent_depth + 1)).any():
            raise ValueError("joint parent must be exactly one depth before its child")
        if (self.valid & ~has_parent & (self.depth != 0)).any():
            raise ValueError("root joint depth must be zero")
        active_features = self.features[self.valid]
        rotation = active_features[:, 3:12].reshape(-1, 3, 3).double()  # Validate independently of TF32 settings.
        eye = torch.eye(3, dtype=rotation.dtype, device=self.features.device)
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation), atol=2e-5, rtol=0):
            raise ValueError("joint rest rotation is not orthogonal")
        if not torch.allclose(
            torch.linalg.det(rotation),
            torch.ones(rotation.shape[0], device=rotation.device, dtype=rotation.dtype),
            atol=2e-5,
            rtol=0,
        ):
            raise ValueError("joint rest rotation must be proper SO(3)")
        norms = torch.linalg.vector_norm(active_features[:, 12:15], dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=0):
            raise ValueError("joint local axis must have unit length")

    def to(self, device: torch.device | str) -> JointKinematicsBank:
        """Move the fixed bank without changing values or dtypes."""
        return JointKinematicsBank(
            *(value.to(device) for value in (self.features, self.parent_slot, self.depth, self.valid))
        )

    def joint_origins(self, q_rad: torch.Tensor, asset_index: torch.Tensor) -> torch.Tensor:
        """Return hand-frame joint origins as ``[B,J,3]`` in metres; ghosts are zero.

        Rodrigues rotations use physical angles in radians and each joint's local unit axis. Joints at the same depth are evaluated together so separate finger chains cannot be joined by slot order. ``asset_index`` routes static physical data only.
        """
        if q_rad.ndim != 2 or q_rad.shape[1] != self.features.shape[1] or asset_index.shape != q_rad.shape[:1]:
            raise ValueError("joint FK expects q shape [B,J] and asset_index [B]")
        if (
            q_rad.device != self.features.device
            or asset_index.device != q_rad.device
            or asset_index.dtype != torch.long
        ):
            raise ValueError("joint FK inputs must share the bank device and int64 asset indices")
        if q_rad.dtype != self.features.dtype:
            raise ValueError("joint FK q and static features must share a floating dtype")
        feature = self.features[asset_index]  # [B,J,15]; one batch may mix parent topologies.
        valid, depth = self.valid[asset_index], self.depth[asset_index]
        parent = self.parent_slot[asset_index]  # Root slots are -1.
        axis = feature[..., 12:15]  # Local unit axes.
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)  # Match [B,J] without a fixed batch size.
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*q_rad.shape, 3, 3)
        eye = torch.eye(3, dtype=q_rad.dtype, device=q_rad.device)  # Root orientation.
        motion = eye + q_rad.sin()[..., None, None] * skew + (1 - q_rad.cos())[..., None, None] * (skew @ skew)
        local_rotation = feature[..., 3:12].reshape(*q_rad.shape, 3, 3) @ motion  # R0 Rot(a,q)。
        local_translation = feature[..., :3] * LENGTH_SCALE_M  # Restore metres.
        rotation = eye.expand(*q_rad.shape, 3, 3)
        position = torch.zeros(*q_rad.shape, 3, dtype=q_rad.dtype, device=q_rad.device)
        parent_index = parent.clamp_min(0)  # Roots are replaced below.
        max_depth = int(self.depth.max().item())  # Fixed by topology, independent of q.
        for level in range(max_depth + 1):
            parent_rotation = rotation.gather(1, parent_index[..., None, None].expand(-1, -1, 3, 3))
            parent_position = position.gather(1, parent_index[..., None].expand(-1, -1, 3))
            parent_rotation = torch.where((parent < 0)[..., None, None], eye, parent_rotation)  # Roots use {h}.
            parent_position = torch.where((parent < 0)[..., None], 0.0, parent_position)
            next_position = (parent_rotation @ local_translation.unsqueeze(-1)).squeeze(-1) + parent_position
            next_rotation = parent_rotation @ local_rotation
            at_level = valid & (depth == level)
            position = torch.where(at_level[..., None], next_position, position)
            rotation = torch.where(at_level[..., None, None], next_rotation, rotation)
        return position  # Ghost slots remain exactly zero.


def build_joint_kinematics_bank(
    semantics: Sequence[HandGeometrySemanticsCfg],
    mappings: Sequence[Mapping[str, int]],
    *,
    joint_count: int = 16,
    dtype: torch.dtype = torch.float32,
) -> JointKinematicsBank:
    """Build a bank from typed semantics and explicit source-name to canonical-slot mappings.

    The shared POE source folds fixed geometry between active joints into each relative transform. Zero-angle transforms are composed in float64, then features are cast to the requested dtype.
    """
    if not semantics or len(semantics) != len(mappings):
        raise ValueError("joint kinematics needs aligned nonempty semantics and routing mappings")
    count = len(semantics)  # Number of asset rows.
    feature = torch.zeros(count, joint_count, JOINT_KINEMATICS_WIDTH, dtype=torch.float64)
    parent = torch.full((count, joint_count), -1, dtype=torch.long)  # -1 marks roots and ghosts.
    depth = torch.full_like(parent, -1)  # Only active joints receive nonnegative depth.
    valid = torch.zeros(count, joint_count, dtype=torch.bool)
    for asset, (item, mapping) in enumerate(zip(semantics, mappings, strict=True)):
        if set(mapping) != set(item.active_joint_names) or len(set(mapping.values())) != len(mapping):
            raise ValueError("source joint names must map one-to-one to canonical slots")
        if any(not 0 <= slot < joint_count for slot in mapping.values()):
            raise ValueError("canonical joint slot lies outside target axis")
        spec = lower_hand_geometry_semantics(item, dtype=torch.float64)  # Shared physical source.
        zero_pose = forward_owner_transforms(spec, torch.zeros(1, len(item.active_joint_names), dtype=torch.float64))[0]
        owners = {owner.joint_name: owner.owner_index for owner in item.owners if owner.role == "joint"}
        joints = {joint.joint_name: joint for joint in item.kinematic_joints if joint.joint_type == "revolute"}
        if set(owners) != set(mapping):
            raise ValueError("FK origin target requires one named JOINT reference frame per active joint")
        for index, name in enumerate(item.active_joint_names):
            slot = mapping[name]  # Use explicit routing; do not infer order from names.
            ancestors = torch.nonzero(spec.joint_ancestor_mask[index], as_tuple=False).flatten()
            transform = zero_pose[owners[name]]  # Child frame at zero joint angle.
            level = int(ancestors.numel())  # Fixed joints do not increase active depth.
            if level:
                parent_source = int(ancestors[spec.joint_ancestor_mask[ancestors].sum(dim=1).argmax()])
                parent_name = item.active_joint_names[parent_source]  # Nearest active parent.
                parent[asset, slot] = mapping[parent_name]
                transform = torch.linalg.inv(zero_pose[owners[parent_name]]) @ transform  # Express child relative to parent.
            feature[asset, slot, :3] = transform[:3, 3] / LENGTH_SCALE_M
            feature[asset, slot, 3:12] = transform[:3, :3].reshape(-1)
            feature[asset, slot, 12:15] = torch.tensor(joints[name].axis_local, dtype=torch.float64)
            valid[asset, slot], depth[asset, slot] = True, level
    return JointKinematicsBank(feature.to(dtype=dtype), parent, depth, valid)
