r"""Pregrasp partial-reset的纯Torch batch、sidecar与action同步合同。

ManagerBased reset先执行event、后执行``ActionManager.reset``。因此event写入的actual state
$\mathbf q_s$与PD preload target$\mathbf q_t$必须经full-size sidecar跨过这一生命周期边界；action term随后只
同步被reset rows。该模块不导入Isaac Lab，可用CPU tensor直接证伪ghost、stale row与partial indexing错误。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from anymani.pregrasp import GoodPregraspEntry, GoodPregraspKey, PregraspLookupKey, PregraspRecord

HETERO_PREGRASP_STATE_ATTR = "_anymani_hetero_pregrasp_reset_state"  # env上的唯一preload sidecar名称
CANONICAL_JOINT_COUNT = 16  # canonical-v1 fixed transport width；逻辑cardinality由active mask决定
CANONICAL_TIP_COUNT = 4  # index/middle/ring/thumb
CANONICAL_OWNER_COUNT = 21  # PALM1 + JOINT16 + TIP4


@dataclass(frozen=True)
class PregraspRuntimeIdentity:
    r"""由scene asset lowering独立提供、用于交叉核对cache key的静态身份。

    Search identity和object physics不属于hand asset本身；这里固定source content、physical geometry、canonical
    schema与active routing四个字段，防止“合法row0 key误绑row16 scene”仍被provider接受。
    """

    source_content_hash: str  # asset source bundle SHA-256
    physical_geometry_hash: str  # 排除ghost后的真实物理几何SHA-256
    canonical_schema_digest: str  # canonical ABI schema SHA-256
    routing_digest: str  # active joint routing SHA-256

    def __post_init__(self) -> None:
        r"""严格验证四个小写SHA-256。"""

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
        r"""拒绝cache key与实际scene hand identity的任何分歧。"""

        if lookup_key.source_content_hash != self.source_content_hash:
            raise ValueError("pregrasp lookup source content disagrees with runtime asset")
        if lookup_key.physical_geometry_hash != self.physical_geometry_hash:
            raise ValueError("pregrasp lookup physical geometry disagrees with runtime asset")
        if lookup_key.canonical_schema_digest != self.canonical_schema_digest:
            raise ValueError("pregrasp lookup canonical schema disagrees with runtime asset")
        if lookup_key.routing_digest != self.routing_digest:
            raise ValueError("pregrasp lookup routing disagrees with runtime asset")

    def validate_good_key(self, key: GoodPregraspKey) -> None:
        r"""拒绝schema-3 good-pregrasp key与实际scene hand identity的任何分歧。"""

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
    r"""把full/partial reset selection规约成唯一一维``torch.long``索引。

    Args:
        env_ids (Sequence[int] | torch.Tensor | None): reset环境；``None``表示$[0,N)$全部环境。
        num_envs (int): scene环境总数$N$。
        device (torch.device | str): sidecar与Isaac资产所在device。

    Returns:
        torch.Tensor: 形状$[K]$的device-local索引，保持caller顺序。

    Raises:
        ValueError: selection为空、重复、越界或不是一维。
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if env_ids is None:
        resolved = torch.arange(num_envs, dtype=torch.long, device=device)  # full reset $K=N$
    else:
        resolved = torch.as_tensor(env_ids, dtype=torch.long, device=device)  # list/tensor统一到asset device
    if resolved.ndim != 1 or resolved.numel() < 1:
        raise ValueError("env_ids must be a non-empty rank-1 selection")
    if bool(((resolved < 0) | (resolved >= num_envs)).any().item()):
        raise ValueError("env_ids contain an out-of-range environment")
    if torch.unique(resolved).numel() != resolved.numel():
        raise ValueError("env_ids must not contain duplicate environments")
    return resolved


def derive_tip_and_owner_masks(active_joint_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""由depth-major joint mask推导TIP与PALM/JOINT/TIP owner masks。

    Canonical joint轴reshape为$[B,D=4,F=4]$，finger顺序固定index/middle/ring/thumb。每指必须是从
    proximal depth 0开始的连续prefix；TIP在该指至少有一个active revolute时有效。Owner轴固定为
    ``[PALM, JOINT×16, TIP×4]``。
    """

    if active_joint_mask.ndim != 2 or active_joint_mask.shape[1] != CANONICAL_JOINT_COUNT:
        raise ValueError("active_joint_mask must have shape [B,16]")
    if active_joint_mask.dtype != torch.bool:
        raise TypeError("active_joint_mask must be bool")
    by_depth_finger = active_joint_mask.reshape(active_joint_mask.shape[0], 4, 4)  # $[B,D,F]$
    non_prefix = by_depth_finger[:, 1:] & ~by_depth_finger[:, :-1]
    if bool(non_prefix.any().item()):
        raise ValueError("each canonical finger joint mask must form a proximal compact prefix")
    active_tip_mask = by_depth_finger.any(dim=1)  # $[B,F]$，index/middle/ring/thumb
    palm_mask = torch.ones(active_joint_mask.shape[0], 1, dtype=torch.bool, device=active_joint_mask.device)
    active_owner_mask = torch.cat((palm_mask, active_joint_mask, active_tip_mask), dim=-1)  # $[B,21]$
    return active_tip_mask, active_owner_mask


@dataclass(frozen=True)
class ResolvedPregraspBatch:
    r"""Provider完成全部fail-closed查询后交付给reset event的device batch。

    张量第一轴严格对应``env_ids``顺序；canonical joint轴为16。``q_state_rad``写入实际PhysX state，
    ``q_target_rad``写入隐式PD target并由action term继续累积。Object pose是hand semantic frame中的
    $T_{ho}$，quaternion顺序固定$(w,x,y,z)$。
    """

    q_state_rad: torch.Tensor  # $[K,16]$ actual reset state，单位rad
    q_target_rad: torch.Tensor  # $[K,16]$ controller preload target，单位rad
    active_joint_mask: torch.Tensor  # bool $[K,16]$，ghost=False
    object_position_h_m: torch.Tensor  # $[K,3]$，单位m
    object_quat_h_wxyz: torch.Tensor  # $[K,4]$ unit quaternion
    record_digests: tuple[str, ...]  # 每行严格record content digest
    lookup_digests: tuple[str, ...]  # 每行exact runtime lookup identity

    def __post_init__(self) -> None:
        r"""验证shape、device、finite、ghost与unit quaternion，不修补坏provider输出。"""

        batch_size = self.q_state_rad.shape[0] if self.q_state_rad.ndim == 2 else -1  # $K$
        expected_joint_shape = (batch_size, CANONICAL_JOINT_COUNT)  # canonical transport$[K,16]$
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
        ghost = ~self.active_joint_mask  # storage-only canonical slots
        if bool((self.q_state_rad[ghost] != 0.0).any().item()) or bool(
            (self.q_target_rad[ghost] != 0.0).any().item()
        ):
            raise ValueError("ghost joint state and target must be exactly zero")
        quaternion_norm = torch.linalg.vector_norm(self.object_quat_h_wxyz, dim=-1)  # $\|q_{ho}\|_2$
        if not bool(torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-5, rtol=0.0)):
            raise ValueError("object quaternion must be unit length")
        if len(self.record_digests) != batch_size or len(self.lookup_digests) != batch_size:
            raise ValueError("record/lookup provenance must contain one digest per batch row")
        for digest in (*self.record_digests, *self.lookup_digests):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("pregrasp provenance must use lowercase SHA-256 digests")

    @property
    def batch_size(self) -> int:
        r"""返回batch环境数$K$。"""

        return int(self.q_state_rad.shape[0])

    @classmethod
    def from_records(
        cls,
        records: Sequence[PregraspRecord],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ResolvedPregraspBatch:
        r"""把已由provider严格验证的records堆叠成runtime tensor batch。

        Args:
            records (Sequence[PregraspRecord]): 与selected env rows同序的完整records。
            device (torch.device | str): Isaac scene device。
            dtype (torch.dtype): state tensor dtype，默认FP32。

        Returns:
            ResolvedPregraspBatch: 保留actual/target分离语义的device batch。
        """

        if not records:
            raise ValueError("cannot build an empty pregrasp batch")
        candidates = [record.candidate for record in records]  # schema已验证ghost、unit和scale
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
        r"""把schema-3 Top-K entries的同一rank堆叠为runtime reset batch。

        Args:
            entries (Sequence[GoodPregraspEntry]): 与selected env rows同序的exact catalog entries。
            rank (int): 所有资产共同消费的candidate rank；MVP固定0。
            device (torch.device | str): Isaac scene device。
            dtype (torch.dtype): state/pose tensor dtype，默认FP32。

        Returns:
            ResolvedPregraspBatch: $q_0=u_0$、upright $T_{ho,0}$与provenance同序batch。
        """

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
    r"""Full-size per-env reset sidecar，连接event与随后执行的action-term reset。

    所有tensor均按scene总环境数$N$分配。``install``只修改selected rows并把对应``valid``置真；未reset
    rows连同其target、mask、pose和provenance保持逐位不变，防止partial reset污染仍在运行的episodes。
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""分配$[N,16]$ controller state与$[N,7]$ object pose sidecar。"""

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)  # $N$
        self.device = torch.device(device)  # 与scene asset一致
        self.q_state_rad = torch.zeros(num_envs, CANONICAL_JOINT_COUNT, device=device, dtype=dtype)
        self.q_target_rad = torch.zeros_like(self.q_state_rad)
        self.active_joint_mask = torch.zeros(num_envs, CANONICAL_JOINT_COUNT, device=device, dtype=torch.bool)
        self.active_tip_mask = torch.zeros(num_envs, CANONICAL_TIP_COUNT, device=device, dtype=torch.bool)
        self.active_owner_mask = torch.zeros(num_envs, CANONICAL_OWNER_COUNT, device=device, dtype=torch.bool)
        self.object_position_h_m = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self.object_quat_h_wxyz = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)  # 尚未provider-resolve的row不能执行action
        self.record_digests: list[str | None] = [None] * num_envs  # diagnostics provenance，不进policy observation
        self.lookup_digests: list[str | None] = [None] * num_envs

    def install(self, env_ids: torch.Tensor, batch: ResolvedPregraspBatch) -> None:
        r"""原子语义地安装selected rows；caller必须在任何PhysX写入前完成batch解析。

        Args:
            env_ids (torch.Tensor): 唯一device-local索引$[K]$。
            batch (ResolvedPregraspBatch): 同序provider结果$[K,...]$。
        """

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
        self.valid[ids] = True  # 所有tensor复制完成后才发布valid commit marker
        for local_index, env_id in enumerate(ids.detach().cpu().tolist()):
            self.record_digests[env_id] = batch.record_digests[local_index]
            self.lookup_digests[env_id] = batch.lookup_digests[local_index]

    def require(self, env_ids: torch.Tensor) -> torch.Tensor:
        r"""返回规范化ids，并在任一row没有合法provider结果时fail closed。"""

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
    r"""计算一次policy-step target转移，physics decimation期间保持幂等。

    $$
    u_{t+1}=m\odot\operatorname{clip}(u_t+\Delta q_t,q_{\min},q_{\max}).
    $$

    五个输入共享$[B,16]$；``processed_delta``已经把raw action缩放/裁剪到每step最多$1/24$ rad。
    """

    expected = previous_target.shape  # $[B,16]$
    tensors = (processed_delta, lower_limit, upper_limit, active_mask)
    if previous_target.ndim != 2 or any(tensor.shape != expected for tensor in tensors):
        raise ValueError("target, delta, limits and active mask must share rank-2 shape")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must be bool")
    active_delta = processed_delta * active_mask.to(dtype=processed_delta.dtype)  # $m\odot\Delta q_t$
    bounded = torch.clamp(previous_target + active_delta, min=lower_limit, max=upper_limit)
    return torch.where(active_mask, bounded, torch.zeros_like(bounded))  # ghost target恒为0 rad


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
    r"""只同步reset rows的action buffers，并返回对应active mask。

    该纯函数实现``event -> ActionManager.reset``交界。非reset rows的动作、target与历史不得变化；reset rows
    的三个action snapshot清零，三个target buffers全部初始化为provider认证的$\mathbf q_t$。
    """

    ids = sidecar.require(env_ids)  # unresolved cache row在写action buffer前fail closed
    target_rows = sidecar.q_target_rad[ids][:, joint_ids]  # 两阶段索引得到outer-product$[K,J]$
    mask_rows = sidecar.active_joint_mask[ids][:, joint_ids]  # 与action joint order对齐
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
