(
    'Pure-Torch contract for pregrasp partial reset and action sync. Events run '
    'before ActionManager.reset, so carry actual q_s and PD target q_t through a '
    'full-size sidecar. Sync reset rows only; CPU tensors support checks for '
    'ghost, stale-row, and partial-index errors.'
)

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from anymani.pregrasp import GoodPregraspEntry, GoodPregraspKey, PregraspLookupKey, PregraspRecord

HETERO_PREGRASP_STATE_ATTR = "_anymani_hetero_pregrasp_reset_state"  # Single preload-sidecar key on the env.
CANONICAL_JOINT_COUNT = 16  # Canonical v1 transport width is fixed; active-mask sets logical cardinality.
CANONICAL_TIP_COUNT = 4  # index/middle/ring/thumb
CANONICAL_OWNER_COUNT = 21  # PALM1 + JOINT16 + TIP4


@dataclass(frozen=True)
class PregraspRuntimeIdentity:
    (
        'Static identity supplied independently by scene-asset lowering to '
        'cross-check cache keys. Search identity and object physics are not part of '
        'the hand asset. The four fields (source content, physical geometry, '
        'canonical schema, active routing) prevent a valid row-0 key from binding to '
        'a row-16 scene.'
    )

    source_content_hash: str  # asset source bundle SHA-256
    physical_geometry_hash: str  # SHA-256 of physical geometry, excluding ghost slots.
    canonical_schema_digest: str  # canonical ABI schema SHA-256
    routing_digest: str  # active joint routing SHA-256

    def __post_init__(self) -> None:
        'Strictly validate four lowercase SHA-256 values.'

        for field_name in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "routing_digest",
        ):
            digest = getattr(self, field_name)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    def validate_lookup_key(self, lookup_key: PregraspLookupKey) -> None:
        'Reject any disagreement between cache key and actual scene hand identity.'

        if lookup_key.source_content_hash != self.source_content_hash:
            raise ValueError("pregrasp lookup source content disagrees with runtime asset")
        if lookup_key.physical_geometry_hash != self.physical_geometry_hash:
            raise ValueError("pregrasp lookup physical geometry disagrees with runtime asset")
        if lookup_key.canonical_schema_digest != self.canonical_schema_digest:
            raise ValueError("pregrasp lookup canonical schema disagrees with runtime asset")
        if lookup_key.routing_digest != self.routing_digest:
            raise ValueError("pregrasp lookup routing disagrees with runtime asset")

    def validate_good_key(self, key: GoodPregraspKey) -> None:
        'Reject any disagreement between schema-3 catalog key and actual scene hand identity.'

        if key.source_content_hash != self.source_content_hash:
            raise ValueError("good-pregrasp source content disagrees with runtime asset")
        if key.physical_geometry_hash != self.physical_geometry_hash:
            raise ValueError("good-pregrasp physical geometry disagrees with runtime asset")
        if key.canonical_schema_digest != self.canonical_schema_digest:
            raise ValueError("good-pregrasp canonical schema disagrees with runtime asset")
        if key.routing_digest != self.routing_digest:
            raise ValueError("good-pregrasp routing disagrees with runtime asset")


def normalize_env_ids(
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    (
        'Normalize full/partial reset selection to one 1D torch.long index. None '
        'selects all envs [0,N). Preserve caller order; reject empty, duplicate, '
        'out-of-range, or non-1D selections.'
    )

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if env_ids is None:
        resolved = torch.arange(num_envs, dtype=torch.long, device=device)  # Full reset: K = N.
    else:
        resolved = torch.as_tensor(env_ids, dtype=torch.long, device=device)  # Normalize lists/tensors to the asset device.
    if resolved.ndim != 1 or resolved.numel() < 1:
        raise ValueError("env_ids must be a non-empty rank-1 selection")
    if bool(((resolved < 0) | (resolved >= num_envs)).any().item()):
        raise ValueError("env_ids contain an out-of-range environment")
    if torch.unique(resolved).numel() != resolved.numel():
        raise ValueError("env_ids must not contain duplicate environments")
    return resolved


def derive_tip_and_owner_masks(active_joint_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Derive TIP and PALM/JOINT/TIP owner masks from the depth-major joint mask '
        '[B,D=4,F=4]. Finger order is index/middle/ring/thumb. Each finger must be an '
        'active prefix starting at proximal depth 0; TIP is valid if that finger has '
        'at least one active revolute joint. Owner order is PALM, JOINT x16, TIP x4.'
    )

    if active_joint_mask.ndim != 2 or active_joint_mask.shape[1] != CANONICAL_JOINT_COUNT:
        raise ValueError("active_joint_mask must have shape [B,16]")
    if active_joint_mask.dtype != torch.bool:
        raise TypeError("active_joint_mask must be bool")
    by_depth_finger = active_joint_mask.reshape(active_joint_mask.shape[0], 4, 4)  # Shape [B,D,F].
    non_prefix = by_depth_finger[:, 1:] & ~by_depth_finger[:, :-1]
    if bool(non_prefix.any().item()):
        raise ValueError("each canonical finger joint mask must form a proximal compact prefix")
    active_tip_mask = by_depth_finger.any(dim=1)  # Shape [B,F], finger order index/middle/ring/thumb.
    palm_mask = torch.ones(active_joint_mask.shape[0], 1, dtype=torch.bool, device=active_joint_mask.device)
    active_owner_mask = torch.cat((palm_mask, active_joint_mask, active_tip_mask), dim=-1)  # Shape [B,21].
    return active_tip_mask, active_owner_mask


@dataclass(frozen=True)
class ResolvedPregraspBatch:
    (
        'Device batch for reset events after all fail-closed provider queries. The '
        'first axis follows env_ids exactly; canonical joint width is 16. q_state_rad '
        'writes actual PhysX state, while q_target_rad initializes the implicit PD '
        'target for later action accumulation. Object pose T_ho is in the semantic '
        'hand frame; quaternion order is (w,x,y,z).'
    )

    q_state_rad: torch.Tensor  # Actual reset state [K,16], rad.
    q_target_rad: torch.Tensor  # Controller preload target [K,16], rad.
    active_joint_mask: torch.Tensor  # Bool mask [K,16]; ghosts are false.
    object_position_h_m: torch.Tensor  # Shape [K,3], meters.
    object_quat_h_wxyz: torch.Tensor  # Shape [K,4], unit quaternion.
    record_digests: tuple[str, ...]  # Strict record-content digest for each row.
    lookup_digests: tuple[str, ...]  # Exact runtime lookup identity for each row.

    def __post_init__(self) -> None:
        'Validate shape, device, finite values, ghosts, and unit quaternion; never repair invalid provider output.'

        batch_size = self.q_state_rad.shape[0] if self.q_state_rad.ndim == 2 else -1  # Batch size K.
        expected_joint_shape = (batch_size, CANONICAL_JOINT_COUNT)  # Canonical transport [K,16].
        if batch_size < 1 or self.q_state_rad.shape != expected_joint_shape:
            raise ValueError("q_state_rad must have shape [K,16] with K>0")
        if self.q_target_rad.shape != expected_joint_shape or self.active_joint_mask.shape != expected_joint_shape:
            raise ValueError("q target and active mask must share [K,16] shape")
        if self.object_position_h_m.shape != (batch_size, 3) or self.object_quat_h_wxyz.shape != (batch_size, 4):
            raise ValueError("object hand-frame pose must have shapes [K,3] and [K,4]")
        tensors = (
            self.q_state_rad,
            self.q_target_rad,
            self.active_joint_mask,
            self.object_position_h_m,
            self.object_quat_h_wxyz,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("pregrasp batch tensors must share one device")
        if self.active_joint_mask.dtype != torch.bool:
            raise TypeError("active_joint_mask must be bool")
        numeric = (self.q_state_rad, self.q_target_rad, self.object_position_h_m, self.object_quat_h_wxyz)
        if any(not bool(torch.isfinite(tensor).all().item()) for tensor in numeric):
            raise ValueError("pregrasp batch tensors must be finite")
        ghost = ~self.active_joint_mask  # Storage-only canonical slots.
        if bool((self.q_state_rad[ghost] != 0.0).any().item()) or bool(
            (self.q_target_rad[ghost] != 0.0).any().item()
        ):
            raise ValueError("ghost joint state and target must be exactly zero")
        quaternion_norm = torch.linalg.vector_norm(self.object_quat_h_wxyz, dim=-1)  # Quaternion norm for q_ho.
        if not bool(torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-5, rtol=0.0)):
            raise ValueError("object quaternion must be unit length")
        if len(self.record_digests) != batch_size or len(self.lookup_digests) != batch_size:
            raise ValueError("record/lookup provenance must contain one digest per batch row")
        for digest in (*self.record_digests, *self.lookup_digests):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("pregrasp provenance must use lowercase SHA-256 digests")

    @property
    def batch_size(self) -> int:
        'Return batch environment count K.'

        return int(self.q_state_rad.shape[0])

    @classmethod
    def from_records(
        cls,
        records: Sequence[PregraspRecord],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ResolvedPregraspBatch:
        'Stack provider-validated records into a runtime tensor batch, preserving separate actual state and target.'

        if not records:
            raise ValueError("cannot build an empty pregrasp batch")
        candidates = [record.candidate for record in records]  # Schema validation already checked ghosts, units, and scale.
        return cls(
            q_state_rad=torch.tensor([candidate.q_state_rad for candidate in candidates], device=device, dtype=dtype),
            q_target_rad=torch.tensor(
                [candidate.q_target_rad for candidate in candidates], device=device, dtype=dtype
            ),
            active_joint_mask=torch.tensor(
                [candidate.active_joint_mask for candidate in candidates], device=device, dtype=torch.bool
            ),
            object_position_h_m=torch.tensor(
                [candidate.object_position_h_m for candidate in candidates], device=device, dtype=dtype
            ),
            object_quat_h_wxyz=torch.tensor(
                [candidate.object_orientation_wxyz for candidate in candidates], device=device, dtype=dtype
            ),
            record_digests=tuple(record.digest for record in records),
            lookup_digests=tuple(record.lookup_key.digest for record in records),
        )

    @classmethod
    def from_good_entries(
        cls,
        entries: Sequence[GoodPregraspEntry],
        *,
        rank: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ResolvedPregraspBatch:
        (
            'Stack the same-rank entries from a schema-3 Top-K catalog into a runtime '
            'reset batch. Rank is shared across assets and MVP fixes it at 0; the result '
            'preserves q0=u0, upright T_ho, and provenance in env order.'
        )

        if not entries:
            raise ValueError("cannot build an empty good-pregrasp batch")
        if rank < 0 or any(rank >= len(entry.members) for entry in entries):
            raise ValueError("good-pregrasp rank lies outside one or more Top-K entries")
        members = [entry.members[rank] for entry in entries]
        candidates = [member.candidate for member in members]
        return cls(
            q_state_rad=torch.tensor([candidate.q_state_rad for candidate in candidates], device=device, dtype=dtype),
            q_target_rad=torch.tensor(
                [candidate.q_target_rad for candidate in candidates], device=device, dtype=dtype
            ),
            active_joint_mask=torch.tensor(
                [candidate.active_joint_mask for candidate in candidates], device=device, dtype=torch.bool
            ),
            object_position_h_m=torch.tensor(
                [candidate.object_position_h_m for candidate in candidates], device=device, dtype=dtype
            ),
            object_quat_h_wxyz=torch.tensor(
                [candidate.object_orientation_h_wxyz for candidate in candidates], device=device, dtype=dtype
            ),
            record_digests=tuple(entry.digest for entry in entries),
            lookup_digests=tuple(entry.key.digest for entry in entries),
        )


class HeterogeneousPregraspState:
    (
        'Full-size per-env reset sidecar between reset events and the following '
        'action-term reset. Allocate all tensors for scene size N. install changes '
        'selected rows only and marks them valid; every other row keeps its target, '
        'mask, pose, and provenance unchanged across partial reset.'
    )

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        'Allocate controller state [N,16] and object-pose sidecar [N,7].'

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)  # Environment count N.
        self.device = torch.device(device)  # Must match the scene asset.
        self.q_state_rad = torch.zeros(num_envs, CANONICAL_JOINT_COUNT, device=device, dtype=dtype)
        self.q_target_rad = torch.zeros_like(self.q_state_rad)
        self.active_joint_mask = torch.zeros(num_envs, CANONICAL_JOINT_COUNT, device=device, dtype=torch.bool)
        self.active_tip_mask = torch.zeros(num_envs, CANONICAL_TIP_COUNT, device=device, dtype=torch.bool)
        self.active_owner_mask = torch.zeros(num_envs, CANONICAL_OWNER_COUNT, device=device, dtype=torch.bool)
        self.object_position_h_m = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self.object_quat_h_wxyz = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)  # An unresolved provider row cannot execute actions.
        self.record_digests: list[str | None] = [None] * num_envs  # Diagnostic provenance only; excluded from policy observations.
        self.lookup_digests: list[str | None] = [None] * num_envs

    def install(self, env_ids: torch.Tensor, batch: ResolvedPregraspBatch) -> None:
        'Atomically install selected rows; caller must resolve the batch before any PhysX write.'

        ids = normalize_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if batch.batch_size != ids.numel() or batch.q_state_rad.device != self.device:
            raise ValueError("pregrasp batch rows/device disagree with env_ids sidecar selection")
        self.q_state_rad[ids] = batch.q_state_rad
        self.q_target_rad[ids] = batch.q_target_rad
        self.active_joint_mask[ids] = batch.active_joint_mask
        tip_mask, owner_mask = derive_tip_and_owner_masks(batch.active_joint_mask)
        self.active_tip_mask[ids] = tip_mask
        self.active_owner_mask[ids] = owner_mask
        self.object_position_h_m[ids] = batch.object_position_h_m
        self.object_quat_h_wxyz[ids] = batch.object_quat_h_wxyz
        self.valid[ids] = True  # Publish valid only after every tensor copy succeeds.
        for local_index, env_id in enumerate(ids.detach().cpu().tolist()):
            self.record_digests[env_id] = batch.record_digests[local_index]
            self.lookup_digests[env_id] = batch.lookup_digests[local_index]

    def require(self, env_ids: torch.Tensor) -> torch.Tensor:
        'Return normalized ids and fail closed if any row lacks a valid provider result.'

        ids = normalize_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if not bool(self.valid[ids].all().item()):
            raise RuntimeError("pregrasp action/reset requested an unresolved environment")
        return ids


def compute_policy_step_masked_relative_target(
    previous_target: torch.Tensor,
    processed_delta: torch.Tensor,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    (
        'Apply one policy-step target transition; remain idempotent across physics '
        'decimation. Inputs share shape [B,16], and processed_delta is already '
        'scaled/clipped to at most 1/24 rad per step. Compute u_next = m * clip(u + '
        'processed_delta, q_min, q_max).'
    )

    expected = previous_target.shape  # Shape [B,16].
    tensors = (processed_delta, lower_limit, upper_limit, active_mask)
    if previous_target.ndim != 2 or any(tensor.shape != expected for tensor in tensors):
        raise ValueError("target, delta, limits and active mask must share rank-2 shape")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must be bool")
    active_delta = processed_delta * active_mask.to(dtype=processed_delta.dtype)  # Masked target delta m * delta_q_t.
    bounded = torch.clamp(previous_target + active_delta, min=lower_limit, max=upper_limit)
    return torch.where(active_mask, bounded, torch.zeros_like(bounded))  # Ghost target remains 0 rad.


def synchronize_action_reset(
    *,
    env_ids: torch.Tensor,
    sidecar: HeterogeneousPregraspState,
    joint_ids: Sequence[int] | torch.Tensor | slice,
    raw_actions: torch.Tensor,
    processed_actions: torch.Tensor,
    executed_actions: torch.Tensor,
    current_targets: torch.Tensor,
    previous_targets: torch.Tensor,
    pregrasp_targets: torch.Tensor,
) -> torch.Tensor:
    (
        'Sync action buffers for reset rows only and return their active mask. This '
        'is the event-to-ActionManager.reset boundary: non-reset actions, targets, '
        'and history must not change. Clear the three action snapshots for reset rows '
        'and initialize all three target buffers to the provider-authenticated q_t.'
    )

    ids = sidecar.require(env_ids)  # Fail closed on unresolved cache rows before writing action buffers.
    target_rows = sidecar.q_target_rad[ids][:, joint_ids]  # Two-stage indexing forms the outer product [K,J].
    mask_rows = sidecar.active_joint_mask[ids][:, joint_ids]  # Aligned with action joint order.
    buffers = (
        raw_actions,
        processed_actions,
        executed_actions,
        current_targets,
        previous_targets,
        pregrasp_targets,
    )
    if any(buffer.ndim != 2 or buffer.shape[0] != sidecar.num_envs for buffer in buffers):
        raise ValueError("action reset buffers must share full [num_envs,J] rows")
    if any(buffer.shape[1:] != target_rows.shape[1:] for buffer in buffers):
        raise ValueError("action reset buffers disagree with selected joint axis")
    raw_actions[ids] = 0.0
    processed_actions[ids] = 0.0
    executed_actions[ids] = 0.0
    current_targets[ids] = target_rows
    previous_targets[ids] = target_rows
    pregrasp_targets[ids] = target_rows
    return mask_rows


__all__ = [
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_OWNER_COUNT",
    "CANONICAL_TIP_COUNT",
    "HETERO_PREGRASP_STATE_ATTR",
    "HeterogeneousPregraspState",
    "PregraspRuntimeIdentity",
    "ResolvedPregraspBatch",
    "compute_policy_step_masked_relative_target",
    "derive_tip_and_owner_masks",
    "normalize_env_ids",
    "synchronize_action_reset",
]
