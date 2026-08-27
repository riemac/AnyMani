r"""Static geometry evidence contract and cross-structure batch assembly。

本模块只拥有 retained encoder 的静态输入类型、mask/routing 合同、source-to-model lowering 与
padding。它不包含 ``torch.nn``、teacher target、loss 或 trainer state；可学习 encoder 位于
``encoder.py``。``StaticGeometryEvidence`` 明确排除 limits、current posed surface、distance、
closest point、Jacobian、contact、action、history 与 object state。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandRouting
from anymani.distill.representations.sources.anchor_sampling import AnchorSamples
from anymani.distill.representations.sources.collision_geometry import HomeSurfaceSamples
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec


@dataclass(frozen=True)
class GeometryPaddingCfg:
    r"""跨结构同批的保守 JOINT/TIP/entity 上限。"""

    max_joint_count: int = 20  # 默认 $5\times4$ 活动 JOINT 上限
    max_tip_count: int = 5  # 默认每根手指一个 TIP owner
    max_graph_distance: int = 8  # padding 图关系的末桶值

    def __post_init__(self) -> None:
        r"""拒绝无法容纳 PALM 或没有有效图桶的上限。"""

        if self.max_joint_count < 1 or self.max_tip_count < 1 or self.max_graph_distance < 1:
            raise ValueError("padding joint/tip counts and graph distance must be positive")

    @property
    def max_owner_count(self) -> int:
        r"""返回 PALM + JOINT + TIP 的最大 entity/owner 数。"""

        return 1 + self.max_joint_count + self.max_tip_count


@dataclass(frozen=True)
class StaticGeometryEvidence:
    r"""允许进入 retained encoder 的静态物理证据。

    ``q_home`` 是基准表面与 screw 的参考构型，不是 joint limit；所有 current target 和动态
    closest/Jacobian 信息都留在 representations.targets。
    """

    anchors: torch.Tensor  # `[K,3]` 或 `[B,K,3]`，`{h}`，m
    home_surface_points: torch.Tensor  # `[G,M,3]` 或 `[B,G,M,3]`，真实 owner boundary
    home_surface_mask: torch.Tensor  # `[G,M]` 或 `[B,G,M]`
    palm_normal: torch.Tensor  # `[3]`，有向 $n_p=z_h$
    space_screws: torch.Tensor  # `[N_J,6]` 或 `[B,N_J,6]`
    q_home: torch.Tensor  # `[N_J]` 或 `[B,N_J]`，rad
    entity_role: torch.Tensor  # `[G]`，0=PALM、1=JOINT、2=TIP
    entity_joint_index: torch.Tensor  # `[G]`，JOINT entity 对应坐标，其他为 -1
    joint_entity_index: torch.Tensor  # `[N_J]`，每个 JOINT 坐标对应实体索引
    shortest_path: torch.Tensor  # `[G,G]`，无向图距离
    parent_direction: torch.Tensor  # `[G,G]`，有向 parent 距离桶
    child_direction: torch.Tensor  # `[G,G]`，有向 child 距离桶
    entity_valid_mask: torch.Tensor | None = None  # `[G]` 或 `[B,G]`
    joint_valid_mask: torch.Tensor | None = None  # `[N_J]` 或 `[B,N_J]`
    anchor_valid_mask: torch.Tensor | None = None  # `[K]` 或 `[B,K]`

    def __post_init__(self) -> None:
        r"""验证 entity、joint、anchor 与 graph axes 严格闭合。"""

        if self.anchors.ndim not in {2, 3} or self.anchors.shape[-1] != 3 or self.anchors.shape[-2] == 0:
            raise ValueError("anchors must have non-empty shape [K,3] or [B,K,3]")
        batched = self.anchors.ndim == 3
        anchor_mask_shape = self.anchors.shape[:-1]
        anchor_valid = (
            self.anchor_valid_mask
            if self.anchor_valid_mask is not None
            else torch.ones(anchor_mask_shape, device=self.anchors.device, dtype=torch.bool)
        )
        if anchor_valid.shape != anchor_mask_shape or anchor_valid.dtype != torch.bool:
            raise ValueError("anchor_valid_mask must have bool shape [K] or [B,K]")
        if torch.any(anchor_valid.sum(dim=-1) == 0):
            raise ValueError("every asset must retain at least one valid physical anchor")
        if self.home_surface_points.ndim != (4 if batched else 3) or self.home_surface_points.shape[-1] != 3:
            raise ValueError("home_surface_points must have shape [G,M,3] or [B,G,M,3]")
        if batched:
            batch_size, owner_count, home_count = self.home_surface_points.shape[:3]
            if self.anchors.shape[0] != batch_size:
                raise ValueError("batched anchors/home_surface_points must share B")
            expected_mask_shape = (batch_size, owner_count, home_count)
        else:
            owner_count, home_count = self.home_surface_points.shape[:2]
            expected_mask_shape = (owner_count, home_count)
        if self.home_surface_mask.shape != expected_mask_shape:
            raise ValueError("home_surface_mask must align with home_surface_points")
        if self.home_surface_mask.dtype != torch.bool:
            raise TypeError("home_surface_mask must use torch.bool")
        allowed_normal_shapes = ({(3,), (batch_size, 3)} if batched else {(3,)})
        if self.palm_normal.shape not in allowed_normal_shapes:
            raise ValueError("palm_normal must have shape [3] or [B,3]")
        if not torch.allclose(
            torch.linalg.vector_norm(self.palm_normal, dim=-1),
            torch.ones(self.palm_normal.shape[:-1], dtype=self.palm_normal.dtype, device=self.palm_normal.device),
            atol=1.0e-6,
            rtol=1.0e-6,
        ):
            raise ValueError("palm_normal must be a unit vector")
        if self.space_screws.ndim != (3 if batched else 2) or self.space_screws.shape[-1] != 6:
            raise ValueError("space_screws must have shape [N_J,6] or [B,N_J,6]")
        joint_count = self.space_screws.shape[-2]
        expected_q_home_shape = (batch_size, joint_count) if batched else (joint_count,)
        if self.q_home.shape != expected_q_home_shape:
            raise ValueError("q_home must align with space_screws")
        if batched and self.space_screws.shape[0] != batch_size:
            raise ValueError("batched screws/home geometry must share B")
        entity_mask_shape = (batch_size, owner_count) if batched else (owner_count,)
        joint_mask_shape = (batch_size, joint_count) if batched else (joint_count,)
        entity_valid = (
            self.entity_valid_mask
            if self.entity_valid_mask is not None
            else torch.ones(entity_mask_shape, device=self.home_surface_mask.device, dtype=torch.bool)
        )
        joint_valid = (
            self.joint_valid_mask
            if self.joint_valid_mask is not None
            else torch.ones(joint_mask_shape, device=self.q_home.device, dtype=torch.bool)
        )
        if entity_valid.shape != entity_mask_shape or entity_valid.dtype != torch.bool:
            raise ValueError("entity_valid_mask must have bool shape [G] or [B,G]")
        if joint_valid.shape != joint_mask_shape or joint_valid.dtype != torch.bool:
            raise ValueError("joint_valid_mask must have bool shape [N_J] or [B,N_J]")
        if torch.any(self.home_surface_mask.sum(dim=-1)[entity_valid] == 0):
            raise ValueError("every valid owner must have at least one home surface point")

        allowed_entity_shapes = {(owner_count,), (batch_size, owner_count)} if batched else {(owner_count,)}
        if self.entity_role.shape not in allowed_entity_shapes or self.entity_joint_index.shape not in allowed_entity_shapes:
            raise ValueError("entity_role/entity_joint_index must have shape [G] or [B,G]")
        allowed_joint_shapes = {(joint_count,), (batch_size, joint_count)} if batched else {(joint_count,)}
        if self.joint_entity_index.shape not in allowed_joint_shapes:
            raise ValueError("joint_entity_index must have shape [N_J] or [B,N_J]")
        graph_shape = (owner_count, owner_count)
        batched_graph_shape = (batch_size, owner_count, owner_count) if batched else graph_shape
        if any(
            matrix.shape not in {graph_shape, batched_graph_shape}
            for matrix in (self.shortest_path, self.parent_direction, self.child_direction)
        ):
            raise ValueError("graph relation matrices must have shape [G,G] or [B,G,G]")

        check_batch = batch_size if batched else 1
        role_batch = self.entity_role if self.entity_role.ndim == 2 else self.entity_role.unsqueeze(0).expand(check_batch, -1)
        entity_joint_batch = (
            self.entity_joint_index
            if self.entity_joint_index.ndim == 2
            else self.entity_joint_index.unsqueeze(0).expand(check_batch, -1)
        )
        joint_entity_batch = (
            self.joint_entity_index
            if self.joint_entity_index.ndim == 2
            else self.joint_entity_index.unsqueeze(0).expand(check_batch, -1)
        )
        entity_valid_batch = entity_valid if entity_valid.ndim == 2 else entity_valid.unsqueeze(0)
        joint_valid_batch = joint_valid if joint_valid.ndim == 2 else joint_valid.unsqueeze(0)
        for batch_index in range(check_batch):
            expected_joint_entities = torch.where((role_batch[batch_index] == 1) & entity_valid_batch[batch_index])[0]
            valid_joint_slots = torch.where(joint_valid_batch[batch_index])[0]
            mapped_entities = joint_entity_batch[batch_index, valid_joint_slots]
            if not torch.equal(expected_joint_entities.sort().values, mapped_entities.sort().values):
                raise ValueError("valid joint_entity_index must bijectively cover all valid JOINT entities")
            if not torch.equal(entity_joint_batch[batch_index, mapped_entities], valid_joint_slots):
                raise ValueError("entity/joint routing must be exact inverses on valid slots")


def build_static_geometry_evidence(
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    home_surface: HomeSurfaceSamples,
    anchors: AnchorSamples,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> StaticGeometryEvidence:
    r"""把静态 source truth 转换为 retained encoder 输入 package。"""

    if len(semantics.owners) != spec.owner_home_transforms.shape[0]:
        raise ValueError("semantics/spec owner axes must match")
    if home_surface.points_owner_local_m.shape[0] != len(semantics.owners):
        raise ValueError("home surface owner axis must match semantics")
    if spec.owner_graph_shortest is None or spec.owner_graph_parent is None or spec.owner_graph_child is None:
        raise ValueError("robots spec must provide all owner graph relations")
    target_device = torch.device(device)
    owner_home = spec.owner_home_transforms.to(device=target_device, dtype=dtype)
    local_home = torch.as_tensor(home_surface.points_owner_local_m, device=target_device, dtype=dtype)
    home_points = (
        torch.einsum("gij,gmj->gmi", owner_home[:, :3, :3], local_home) + owner_home[:, None, :3, 3]
    )
    anchor_points = torch.as_tensor(anchors.anchors_hand_m, device=target_device, dtype=dtype)
    role_index = {"palm": 0, "joint": 1, "tip": 2}
    entity_role = torch.tensor(
        [role_index[owner.role] for owner in semantics.owners], device=target_device, dtype=torch.long
    )
    joint_index_by_name = {name: index for index, name in enumerate(spec.joint_names)}
    entity_joint_index = torch.full((len(semantics.owners),), -1, device=target_device, dtype=torch.long)
    for owner in semantics.owners:
        if owner.role == "joint":
            if owner.joint_name not in joint_index_by_name:
                raise ValueError(f"owner '{owner.owner_id}' joint is missing from spec joint axis")
            entity_joint_index[owner.owner_index] = joint_index_by_name[owner.joint_name]
    joint_entity_index = torch.tensor(
        [owner.owner_index for owner in semantics.owners if owner.role == "joint"],
        device=target_device,
        dtype=torch.long,
    )
    if tuple(entity_joint_index[joint_entity_index].tolist()) != tuple(range(len(spec.joint_names))):
        raise ValueError("owner JOINT axis is not the exact inverse of joint axis")
    return StaticGeometryEvidence(
        anchors=anchor_points,
        home_surface_points=home_points,
        home_surface_mask=torch.ones(home_points.shape[:2], device=target_device, dtype=torch.bool),
        palm_normal=torch.tensor((0.0, 0.0, 1.0), device=target_device, dtype=dtype),
        space_screws=spec.space_screws.to(device=target_device, dtype=dtype),
        q_home=spec.q_home.to(device=target_device, dtype=dtype),
        entity_role=entity_role,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=spec.owner_graph_shortest.to(device=target_device),
        parent_direction=spec.owner_graph_parent.to(device=target_device),
        child_direction=spec.owner_graph_child.to(device=target_device),
        anchor_valid_mask=torch.ones(anchor_points.shape[:-1], device=target_device, dtype=torch.bool),
    )


def canonicalize_static_geometry_evidence(
    evidence: StaticGeometryEvidence,
    semantics: HandGeometrySemanticsCfg,
    routing: CanonicalHandRouting,
    *,
    max_graph_distance: int = 8,
) -> StaticGeometryEvidence:
    r"""把 source PALM/JOINT/TIP 证据 scatter 到 canonical [21,16] axes。"""

    if evidence.anchors.ndim != 2 or evidence.space_screws.ndim != 2:
        raise ValueError("canonicalization expects one unbatched source evidence")
    schema = CANONICAL_HAND_SCHEMA_V1
    device = evidence.anchors.device
    dtype = evidence.anchors.dtype
    home_budget = evidence.home_surface_points.shape[1]
    owner_count = 1 + schema.dof_count + len(schema.physx_finger_order)
    joint_count = schema.dof_count
    owner_by_joint = {
        str(owner.joint_name): owner.owner_index
        for owner in semantics.owners
        if owner.role == "joint" and owner.joint_name is not None
    }
    tip_by_finger = {
        str(owner.finger_name): owner.owner_index
        for owner in semantics.owners
        if owner.role == "tip" and owner.finger_name is not None
    }
    palm_indices = [owner.owner_index for owner in semantics.owners if owner.role == "palm"]
    if len(palm_indices) != 1:
        raise ValueError("canonical evidence requires exactly one PALM owner")
    source_joint_index = {name: index for index, name in enumerate(semantics.active_joint_names)}
    canonical_to_source = {canonical: source for source, canonical in routing.source_to_canonical}
    owner_source_indices: list[int | None] = [palm_indices[0]]
    joint_source_indices: list[int | None] = []
    for canonical_name in schema.joint_names:
        source_name = canonical_to_source.get(canonical_name)
        owner_source_indices.append(owner_by_joint.get(source_name) if source_name is not None else None)
        joint_source_indices.append(source_joint_index.get(source_name) if source_name is not None else None)
    owner_source_indices.extend(tip_by_finger.get(finger) for finger in schema.physx_finger_order)

    home_points = torch.zeros(owner_count, home_budget, 3, device=device, dtype=dtype)
    home_mask = torch.zeros(owner_count, home_budget, device=device, dtype=torch.bool)
    screws = torch.zeros(joint_count, 6, device=device, dtype=dtype)
    q_home = torch.zeros(joint_count, device=device, dtype=dtype)
    owner_valid = torch.zeros(owner_count, device=device, dtype=torch.bool)
    joint_valid = torch.zeros(joint_count, device=device, dtype=torch.bool)
    graph_shape = (owner_count, owner_count)
    shortest = torch.full(graph_shape, max_graph_distance, device=device, dtype=torch.long)
    parent = torch.full_like(shortest, max_graph_distance)
    child = torch.full_like(shortest, max_graph_distance)
    valid_owner_destinations: list[int] = []
    valid_owner_sources: list[int] = []
    for destination, source in enumerate(owner_source_indices):
        if source is None:
            continue
        home_points[destination] = evidence.home_surface_points[source]
        home_mask[destination] = evidence.home_surface_mask[source]
        owner_valid[destination] = True
        valid_owner_destinations.append(destination)
        valid_owner_sources.append(source)
    for destination, source in enumerate(joint_source_indices):
        if source is None:
            continue
        screws[destination] = evidence.space_screws[source]
        q_home[destination] = evidence.q_home[source]
        joint_valid[destination] = True
    destination_index = torch.tensor(valid_owner_destinations, device=device, dtype=torch.long)
    source_index = torch.tensor(valid_owner_sources, device=device, dtype=torch.long)
    destination_grid = destination_index[:, None], destination_index[None, :]
    source_grid = source_index[:, None], source_index[None, :]
    shortest[destination_grid] = evidence.shortest_path[source_grid]
    parent[destination_grid] = evidence.parent_direction[source_grid]
    child[destination_grid] = evidence.child_direction[source_grid]
    if tuple(joint_valid.tolist()) != routing.active_joint_mask:
        raise ValueError("canonical geometry joint mask disagrees with artifact routing")
    tip_valid = tuple(owner_valid[1 + joint_count :].tolist())
    if tip_valid != routing.active_tip_mask:
        raise ValueError("canonical geometry TIP mask disagrees with artifact routing")
    return StaticGeometryEvidence(
        anchors=evidence.anchors,
        home_surface_points=home_points,
        home_surface_mask=home_mask,
        palm_normal=evidence.palm_normal,
        space_screws=screws,
        q_home=q_home,
        entity_role=torch.tensor([0, *([1] * joint_count), *([2] * 4)], device=device, dtype=torch.long),
        entity_joint_index=torch.tensor([-1, *range(joint_count), *([-1] * 4)], device=device, dtype=torch.long),
        joint_entity_index=torch.arange(1, 1 + joint_count, device=device, dtype=torch.long),
        shortest_path=shortest,
        parent_direction=parent,
        child_direction=child,
        entity_valid_mask=owner_valid,
        joint_valid_mask=joint_valid,
        anchor_valid_mask=evidence.anchor_valid_mask,
    )


def stack_static_geometry_evidence(evidences: Sequence[StaticGeometryEvidence]) -> StaticGeometryEvidence:
    r"""堆叠同一结构的多项 evidence，新增 B axis 而不做 entity padding。"""

    if not evidences:
        raise ValueError("at least one StaticGeometryEvidence is required")
    if any(evidence.anchors.ndim != 2 for evidence in evidences):
        raise ValueError("stack_static_geometry_evidence expects unbatched asset evidence")
    reference = evidences[0]
    shared_fields = (
        "entity_role",
        "entity_joint_index",
        "joint_entity_index",
        "shortest_path",
        "parent_direction",
        "child_direction",
    )
    for evidence_index, evidence in enumerate(evidences[1:], start=1):
        for field_name in shared_fields:
            if not torch.equal(getattr(reference, field_name), getattr(evidence, field_name)):
                raise ValueError(
                    f"evidence[{evidence_index}] field '{field_name}' differs; split into another structure minibatch"
                )
        if evidence.anchors.shape != reference.anchors.shape:
            raise ValueError("same minibatch requires one configured anchor count K")
        if evidence.home_surface_points.shape != reference.home_surface_points.shape:
            raise ValueError("same minibatch requires one configured home point budget M")
        if evidence.space_screws.shape != reference.space_screws.shape:
            raise ValueError("same minibatch requires one active JOINT axis length")
    return StaticGeometryEvidence(
        anchors=torch.stack([evidence.anchors for evidence in evidences], dim=0),
        home_surface_points=torch.stack([evidence.home_surface_points for evidence in evidences], dim=0),
        home_surface_mask=torch.stack([evidence.home_surface_mask for evidence in evidences], dim=0),
        palm_normal=torch.stack([evidence.palm_normal for evidence in evidences], dim=0),
        space_screws=torch.stack([evidence.space_screws for evidence in evidences], dim=0),
        q_home=torch.stack([evidence.q_home for evidence in evidences], dim=0),
        entity_role=reference.entity_role,
        entity_joint_index=reference.entity_joint_index,
        joint_entity_index=reference.joint_entity_index,
        shortest_path=reference.shortest_path,
        parent_direction=reference.parent_direction,
        child_direction=reference.child_direction,
        anchor_valid_mask=torch.stack(
            [
                evidence.anchor_valid_mask
                if evidence.anchor_valid_mask is not None
                else torch.ones(evidence.anchors.shape[0], device=evidence.anchors.device, dtype=torch.bool)
                for evidence in evidences
            ],
            dim=0,
        ),
    )


def pad_static_geometry_evidence(
    evidences: Sequence[StaticGeometryEvidence],
    *,
    config: GeometryPaddingCfg = GeometryPaddingCfg(),
) -> StaticGeometryEvidence:
    r"""把不同 owner/JOINT 长度的资产填充为统一稠密 batch。

    padding 区域的 home/screw/q 为零、routing 为 -1，attention/backbone 由显式 valid masks 屏蔽；
    真实 owner/joint 的原始顺序、图距离和物理 evidence 不变。
    """

    if not evidences:
        raise ValueError("at least one StaticGeometryEvidence is required")
    if any(evidence.anchors.ndim != 2 for evidence in evidences):
        raise ValueError("padding expects one unbatched StaticGeometryEvidence per asset")
    reference = evidences[0]
    max_anchor_count = max(evidence.anchors.shape[0] for evidence in evidences)
    home_budget = reference.home_surface_points.shape[1]
    for evidence in evidences:
        if evidence.home_surface_points.shape[1] != home_budget:
            raise ValueError("padded batch requires one configured home point budget M")
    batch_size = len(evidences)
    max_owner_count = config.max_owner_count
    max_joint_count = config.max_joint_count
    device = reference.anchors.device
    dtype = reference.anchors.dtype
    anchors = torch.zeros(batch_size, max_anchor_count, 3, device=device, dtype=dtype)
    anchor_valid = torch.zeros(batch_size, max_anchor_count, device=device, dtype=torch.bool)
    home_points = torch.zeros(batch_size, max_owner_count, home_budget, 3, device=device, dtype=dtype)
    home_mask = torch.zeros(batch_size, max_owner_count, home_budget, device=device, dtype=torch.bool)
    screws = torch.zeros(batch_size, max_joint_count, 6, device=device, dtype=dtype)
    q_home = torch.zeros(batch_size, max_joint_count, device=device, dtype=dtype)
    entity_role = torch.zeros(batch_size, max_owner_count, device=device, dtype=torch.long)
    entity_joint_index = torch.full((batch_size, max_owner_count), -1, device=device, dtype=torch.long)
    joint_entity_index = torch.full((batch_size, max_joint_count), -1, device=device, dtype=torch.long)
    entity_valid = torch.zeros(batch_size, max_owner_count, device=device, dtype=torch.bool)
    joint_valid = torch.zeros(batch_size, max_joint_count, device=device, dtype=torch.bool)
    graph_shape = (batch_size, max_owner_count, max_owner_count)
    shortest_path = torch.full(graph_shape, config.max_graph_distance, device=device, dtype=torch.long)
    parent_direction = torch.full_like(shortest_path, config.max_graph_distance)
    child_direction = torch.full_like(shortest_path, config.max_graph_distance)

    for batch_index, evidence in enumerate(evidences):
        owner_count = evidence.home_surface_points.shape[0]
        joint_count = evidence.space_screws.shape[0]
        anchor_count = evidence.anchors.shape[0]
        tip_count = int(torch.count_nonzero(evidence.entity_role == 2))
        if joint_count > max_joint_count:
            raise ValueError(f"asset[{batch_index}] has {joint_count} joints, exceeds {max_joint_count}")
        if tip_count > config.max_tip_count:
            raise ValueError(f"asset[{batch_index}] has {tip_count} TIP owners, exceeds {config.max_tip_count}")
        if owner_count > max_owner_count:
            raise ValueError(f"asset[{batch_index}] has {owner_count} owners, exceeds {max_owner_count}")
        anchors[batch_index, :anchor_count] = evidence.anchors
        source_anchor_mask = evidence.anchor_valid_mask
        anchor_valid[batch_index, :anchor_count] = (
            source_anchor_mask
            if source_anchor_mask is not None
            else torch.ones(anchor_count, device=device, dtype=torch.bool)
        )
        home_points[batch_index, :owner_count] = evidence.home_surface_points
        home_mask[batch_index, :owner_count] = evidence.home_surface_mask
        screws[batch_index, :joint_count] = evidence.space_screws
        q_home[batch_index, :joint_count] = evidence.q_home
        entity_role[batch_index, :owner_count] = evidence.entity_role
        entity_joint_index[batch_index, :owner_count] = evidence.entity_joint_index
        joint_entity_index[batch_index, :joint_count] = evidence.joint_entity_index
        entity_valid[batch_index, :owner_count] = (
            evidence.entity_valid_mask
            if evidence.entity_valid_mask is not None
            else torch.ones(owner_count, device=device, dtype=torch.bool)
        )
        joint_valid[batch_index, :joint_count] = (
            evidence.joint_valid_mask
            if evidence.joint_valid_mask is not None
            else torch.ones(joint_count, device=device, dtype=torch.bool)
        )
        shortest_path[batch_index, :owner_count, :owner_count] = evidence.shortest_path
        parent_direction[batch_index, :owner_count, :owner_count] = evidence.parent_direction
        child_direction[batch_index, :owner_count, :owner_count] = evidence.child_direction

    return StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_points,
        home_surface_mask=home_mask,
        palm_normal=torch.stack([evidence.palm_normal for evidence in evidences], dim=0),
        space_screws=screws,
        q_home=q_home,
        entity_role=entity_role,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=shortest_path,
        parent_direction=parent_direction,
        child_direction=child_direction,
        entity_valid_mask=entity_valid,
        joint_valid_mask=joint_valid,
        anchor_valid_mask=anchor_valid,
    )


__all__ = [
    "GeometryPaddingCfg",
    "StaticGeometryEvidence",
    "build_static_geometry_evidence",
    "canonicalize_static_geometry_evidence",
    "pad_static_geometry_evidence",
    "stack_static_geometry_evidence",
]
