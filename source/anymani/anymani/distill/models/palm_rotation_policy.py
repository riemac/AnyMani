"""Teacher actor-critic networks for hand rotation. Geometry tokens use [B,21,128], history uses 30 frames at 20 Hz, and policy actions use 16 masked joint slots."""


from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg
from .temporal_encoder import PerJointRawHistoryStack, PerJointTactileTemporalEncoder

JOINT_COUNT = 16
TIP_COUNT = 4
OWNER_COUNT = 21
HISTORY_LENGTH = 30
GEOMETRY_WIDTH = 128
JOINT_FRAME_WIDTH = 5
LOCAL_WIDTH = 64
HISTORY_WIDTH = 32
FINGER_WIDTH = 48
HAND_WIDTH = 64
BASE_ACTION_LIMIT = 0.8
RESIDUAL_LIMIT = 0.2
FILM_LIMIT = 0.25
RECOVERY_LIMIT_MARGIN_RAD = 0.02
RECOVERY_OUTWARD_ACTION_MIN = 0.05


def recovery_exploration_contract(
    sigma_floor: float | None, *, max_log_std: float, sigma_mode: str = "global"
) -> dict[str, object] | None:


    if sigma_floor is None:
        return None
    if (
        sigma_mode != "global"
        or not math.isfinite(sigma_floor)
        or sigma_floor <= 0.0
        or not math.isfinite(max_log_std)
        or math.log(sigma_floor) > max_log_std
    ):
        raise ValueError("recovery sigma floor requires global sigma and a finite positive value within its ceiling")
    return {
        "rule": "tip-silent-target-limit-outward-previous-action-v1",
        "sigma_floor": float(sigma_floor),
        "limit_margin_rad": RECOVERY_LIMIT_MARGIN_RAD,
        "previous_outward_action_min": RECOVERY_OUTWARD_ACTION_MIN,
        "contact_scope": "all-valid-tips-zero",
        "target_and_limits_units": "rad-divided-by-pi",
        "mean": "unchanged",
        "effective_log_sigma": "gate?max(global_log_std,log(sigma_floor)):global_log_std",
        "maximum_log_sigma": float(max_log_std),
    }


class _PalmRotationGraphBiasedTransformer(GraphBiasedTransformer):


    def _graph_bias(
        self,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
    ) -> torch.Tensor:


        matrices = (shortest_path, parent_direction, child_direction)
        if any(matrix.ndim not in {2, 3} or matrix.shape[-2] != matrix.shape[-1] for matrix in matrices):
            raise ValueError("graph relation matrices must have square shape [N_E,N_E] or [B,N_E,N_E]")
        if parent_direction.shape != shortest_path.shape or child_direction.shape != shortest_path.shape:
            raise ValueError("all graph relation matrices must have identical shape")

        shortest = shortest_path.clamp(min=0, max=self.max_graph_distance)
        parent = parent_direction.clamp(min=0, max=self.max_graph_distance)
        child = child_direction.clamp(min=0, max=self.max_graph_distance)
        bias = (
            self.shortest_path_bias(shortest) + self.parent_direction_bias(parent) + self.child_direction_bias(child)
        )
        if bias.ndim == 3:
            return bias.permute(2, 0, 1).contiguous()
        return bias.permute(0, 3, 1, 2).contiguous()


def _bool_mask(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:


    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype == torch.bool:
        return value
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(torch.isfinite(value) & ((value == 0) | (value == 1))),
        f"{name} numeric transport must contain finite 0/1 values",
    )
    return value.to(dtype=torch.bool)


def _phase_clock_input(
    phase_clock: torch.Tensor | None,
    *,
    enabled: bool,
    batch_size: int,
    reference: torch.Tensor,
    role: str,
    validated: bool,
) -> torch.Tensor | None:


    if not enabled:
        return None
    if phase_clock is None:
        raise ValueError(f"{role} phase_clock_enabled=True requires phase_clock with shape [{batch_size},2]")
    expected_shape = (batch_size, 2)
    if tuple(phase_clock.shape) != expected_shape:
        raise ValueError(
            f"{role} phase_clock must have shape {expected_shape}, got {tuple(phase_clock.shape)}"
        )
    if phase_clock.dtype != reference.dtype:
        raise ValueError(
            f"{role} phase_clock must have dtype {reference.dtype}, got {phase_clock.dtype}"
        )
    if phase_clock.device != reference.device:
        raise ValueError(
            f"{role} phase_clock must be on device {reference.device}, got {phase_clock.device}"
        )
    if not validated:
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.isfinite(phase_clock).all(),
            f"{role} phase_clock must contain finite sin/cos values",
        )
    return phase_clock


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:


    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_mean tokens/mask shapes disagree")
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_max(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:


    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_max tokens/mask shapes disagree")
    has_value = mask.any(dim=dim, keepdim=True)
    empty_slice = ~has_value
    safe_tokens = torch.where(empty_slice.unsqueeze(-1), torch.zeros_like(tokens), tokens)
    effective_mask = mask | empty_slice.expand_as(mask)
    minimum = torch.finfo(tokens.dtype).min
    result = safe_tokens.masked_fill(~effective_mask.unsqueeze(-1), minimum).amax(dim=dim)
    return result


@dataclass(frozen=True)
class PalmRotationGeometry:


    tokens: torch.Tensor
    owner_valid: torch.Tensor  # bool`[B,21]`
    shortest_path: torch.Tensor  # long`[B,21,21]`
    parent_direction: torch.Tensor  # long`[B,21,21]`
    child_direction: torch.Tensor  # long`[B,21,21]`

    def __post_init__(self) -> None:


        batch = self.tokens.shape[0]
        if self.tokens.shape != (batch, OWNER_COUNT, GEOMETRY_WIDTH):
            raise ValueError("palm-rotation geometry tokens must have shape [B,21,128]")
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="geometry owner_valid", shape=(batch, OWNER_COUNT)),
        )
        graph_shape = (batch, OWNER_COUNT, OWNER_COUNT)
        if any(
            matrix.shape != graph_shape for matrix in (self.shortest_path, self.parent_direction, self.child_direction)
        ):
            raise ValueError("palm-rotation graph matrices must have shape [B,21,21]")
        tensors = (self.tokens, self.owner_valid, self.shortest_path, self.parent_direction, self.child_direction)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("geometry tensors must share one device")


@dataclass(frozen=True)
class PalmRotationActorObservation:


    jnt_current: torch.Tensor  # `[B,16,5]` q/u/a/own-contact/TIP-contact
    jnt_history: torch.Tensor  # `[B,30,16,5]`
    jnt_limits: torch.Tensor  # `[B,16,2]` qmin/qmax divided by pi
    owner_contact: torch.Tensor  # `[B,21,1]` binary
    jnt_valid: torch.Tensor  # bool`[B,16]`
    tip_valid: torch.Tensor  # bool`[B,4]`
    owner_valid: torch.Tensor  # bool`[B,21]`

    def __post_init__(self) -> None:


        batch = self.jnt_current.shape[0]
        expected = {
            "jnt_current": (batch, JOINT_COUNT, JOINT_FRAME_WIDTH),
            "jnt_history": (batch, HISTORY_LENGTH, JOINT_COUNT, JOINT_FRAME_WIDTH),
            "jnt_limits": (batch, JOINT_COUNT, 2),
            "owner_contact": (batch, OWNER_COUNT, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"actor {name} must have shape {shape}")
        object.__setattr__(self, "jnt_valid", _bool_mask(self.jnt_valid, name="jnt_valid", shape=(batch, 16)))
        object.__setattr__(self, "tip_valid", _bool_mask(self.tip_valid, name="tip_valid", shape=(batch, 4)))
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="owner_valid", shape=(batch, 21)),
        )
        expected_owner = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=self.jnt_valid.device), self.jnt_valid, self.tip_valid),
            dim=-1,
        )
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(self.owner_valid == expected_owner), "actor owner masks disagree"
        )
        tensors = (
            self.jnt_current,
            self.jnt_history,
            self.jnt_limits,
            self.owner_contact,
            self.jnt_valid,
            self.tip_valid,
            self.owner_valid,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("actor observation tensors must share one device")

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> PalmRotationActorObservation:


        names = {"jnt_current", "jnt_history", "jnt_limits", "owner_contact", "jnt_valid", "tip_valid", "owner_valid"}
        missing = names - set(observation)
        if missing:
            raise KeyError(f"palm-rotation actor observation misses {sorted(missing)}")
        return cls(**{name: observation[name] for name in names})


@dataclass(frozen=True)
class PalmRotationCriticObservation:


    jnt_state: torch.Tensor  # `[B,16,4]` q/qd/u/a
    owner_contact: torch.Tensor  # `[B,21,2]` force N + bit
    obj: torch.Tensor  # `[B,1,15]`
    task: torch.Tensor  # `[B,1,8]`
    reward_release: torch.Tensor  # `[B,1]` cell-level lambda
    jnt_valid: torch.Tensor
    tip_valid: torch.Tensor
    owner_valid: torch.Tensor

    def __post_init__(self) -> None:


        batch = self.jnt_state.shape[0]
        expected = {
            "jnt_state": (batch, 16, 4),
            "owner_contact": (batch, 21, 2),
            "obj": (batch, 1, 15),
            "task": (batch, 1, 8),
            "reward_release": (batch, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"critic {name} must have shape {shape}")
        object.__setattr__(self, "jnt_valid", _bool_mask(self.jnt_valid, name="critic jnt_valid", shape=(batch, 16)))
        object.__setattr__(self, "tip_valid", _bool_mask(self.tip_valid, name="critic tip_valid", shape=(batch, 4)))
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="critic owner_valid", shape=(batch, 21)),
        )
        tensors = (
            self.jnt_state,
            self.owner_contact,
            self.obj,
            self.task,
            self.reward_release,
            self.jnt_valid,
            self.tip_valid,
            self.owner_valid,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("critic observation tensors must share one device")

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> PalmRotationCriticObservation:


        names = {
            "jnt_state",
            "owner_contact",
            "obj",
            "task",
            "reward_release",
            "jnt_valid",
            "tip_valid",
            "owner_valid",
        }
        missing = names - set(observation)
        if missing:
            raise KeyError(f"palm-rotation critic observation misses {sorted(missing)}")
        return cls(**{name: observation[name] for name in names})


@dataclass(frozen=True)
class PalmRotationActorOutput:


    mean: torch.Tensor
    film_modulation_rms: torch.Tensor
    log_std: torch.Tensor
    base_mean: torch.Tensor | None = None
    residual_mean: torch.Tensor | None = None
    direct_mean: torch.Tensor | None = None


class PalmRotationResidualActor(nn.Module):


    def __init__(
        self,
        *,
        residual_enabled: bool = True,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
    ) -> None:


        super().__init__()
        if initial_log_std > max_log_std:
            raise ValueError("initial_log_std must not exceed the exploration ceiling")
        if not 0.0 < base_action_limit <= 1.0 - RESIDUAL_LIMIT:
            raise ValueError("base_action_limit plus residual limit must remain within physical action bounds")
        self.residual_enabled = bool(residual_enabled)
        self.phase_clock_enabled = False
        if sigma_mode not in {"global", "conditional"} or (sigma_mode == "conditional" and not residual_enabled):
            raise ValueError("conditional sigma requires a contextual Actor")
        self.sigma_mode = sigma_mode
        self.max_log_std = float(max_log_std)
        recovery_exploration_contract(recovery_sigma_floor, max_log_std=max_log_std, sigma_mode=sigma_mode)
        self.recovery_sigma_floor = recovery_sigma_floor
        self.base_action_limit = float(base_action_limit)
        self.history_encoder_name = history_encoder
        if history_encoder == "tcn":
            self.history_encoder: nn.Module = PerJointTactileTemporalEncoder(
                joint_count=JOINT_COUNT,
                frame_dim=JOINT_FRAME_WIDTH,
                latent_dim=HISTORY_WIDTH,
                hidden_channels=(32, 32, 32),
            )  # `[B,30,16,5] -> [B,16,32]` learned temporal compression
            history_width = HISTORY_WIDTH
        elif history_encoder == "raw_stack":
            self.history_encoder = PerJointRawHistoryStack(
                joint_count=JOINT_COUNT,
                frame_dim=JOINT_FRAME_WIDTH,
            )
            history_width = HISTORY_LENGTH * JOINT_FRAME_WIDTH
        else:
            raise ValueError(f"unknown palm-rotation history encoder: {history_encoder!r}")
        local_input_width = JOINT_FRAME_WIDTH + 2 + 2 + history_width  # current/limits/lag/q/history
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_input_width),
            nn.Linear(local_input_width, LOCAL_WIDTH),
            nn.GELU(),
            nn.Linear(LOCAL_WIDTH, LOCAL_WIDTH),
        )
        self.joint_geometry_film = nn.Sequential(
            nn.LayerNorm(GEOMETRY_WIDTH),
            nn.Linear(GEOMETRY_WIDTH, 2 * LOCAL_WIDTH),
        )
        nn.init.zeros_(self.joint_geometry_film[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.joint_geometry_film[-1].bias)  # type: ignore[arg-type]
        self.tip_geometry_projection = nn.Linear(GEOMETRY_WIDTH, 32)
        self.finger_encoder = nn.Sequential(
            nn.LayerNorm(2 * LOCAL_WIDTH + 32 + 2),
            nn.Linear(2 * LOCAL_WIDTH + 32 + 2, FINGER_WIDTH),
            nn.GELU(),
            nn.Linear(FINGER_WIDTH, FINGER_WIDTH),
        )
        self.palm_geometry_projection = nn.Linear(GEOMETRY_WIDTH, 32)
        self.hand_encoder = nn.Sequential(
            nn.LayerNorm(2 * FINGER_WIDTH + 32 + 2),
            nn.Linear(2 * FINGER_WIDTH + 32 + 2, HAND_WIDTH),
            nn.GELU(),
            nn.Linear(HAND_WIDTH, HAND_WIDTH),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(LOCAL_WIDTH + HAND_WIDTH),
            nn.Linear(LOCAL_WIDTH + HAND_WIDTH, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.base_head[-1].weight, gain=0.01)  # type: ignore[arg-type]
        nn.init.zeros_(self.base_head[-1].bias)  # type: ignore[arg-type]


        self.geometry_adapter = nn.Linear(GEOMETRY_WIDTH, GEOMETRY_WIDTH)
        self.owner_contact_projection = nn.Linear(1, GEOMETRY_WIDTH, bias=False)
        self.palm_dynamic_projection = nn.Linear(HAND_WIDTH, GEOMETRY_WIDTH)
        self.joint_dynamic_projection = nn.Linear(LOCAL_WIDTH, GEOMETRY_WIDTH)
        self.tip_dynamic_projection = nn.Linear(FINGER_WIDTH, GEOMETRY_WIDTH)
        self.global_backbone = _PalmRotationGraphBiasedTransformer(
            GraphBiasedTransformerCfg(
                hidden_width=GEOMETRY_WIDTH,
                layers=1,
                attention_heads=4,
                feedforward_width=256,
                dropout=0.0,
            )
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(GEOMETRY_WIDTH + LOCAL_WIDTH),
            nn.Linear(GEOMETRY_WIDTH + LOCAL_WIDTH, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.residual_head[-1].bias)  # type: ignore[arg-type]
        self.global_log_std = nn.Parameter(torch.tensor(float(initial_log_std)))
        self.conditional_sigma_head = None
        if sigma_mode == "conditional":
            self.conditional_sigma_head = nn.Linear(GEOMETRY_WIDTH, 1)
            nn.init.zeros_(self.conditional_sigma_head.weight)
            nn.init.zeros_(self.conditional_sigma_head.bias)

    @torch.no_grad()
    def project_exploration_parameters(self) -> None:


        self.global_log_std.clamp_(max=self.max_log_std)
        if self.sigma_mode == "conditional":
            self.global_log_std.clamp_(min=math.log(0.05))

    def recovery_exploration_mask(self, observation: PalmRotationActorObservation) -> torch.Tensor:


        if self.recovery_sigma_floor is None:
            return torch.zeros_like(observation.jnt_valid)
        tip_contact = observation.owner_contact[:, 17:21, 0] > 0.5
        no_tip = observation.tip_valid.any(dim=-1) & ~(tip_contact & observation.tip_valid).any(dim=-1)
        target = observation.jnt_current[..., 1]
        previous_action = observation.jnt_current[..., 2]
        margin = RECOVERY_LIMIT_MARGIN_RAD / math.pi
        lower_outward = (target <= observation.jnt_limits[..., 0] + margin) & (
            previous_action < -RECOVERY_OUTWARD_ACTION_MIN
        )
        upper_outward = (target >= observation.jnt_limits[..., 1] - margin) & (
            previous_action > RECOVERY_OUTWARD_ACTION_MIN
        )
        return observation.jnt_valid & no_tip[:, None] & (lower_outward | upper_outward)

    def _policy_log_std(
        self, contextual: torch.Tensor | None, observation: PalmRotationActorObservation
    ) -> torch.Tensor:


        if self.sigma_mode == "global":
            if self.recovery_sigma_floor is None:
                return self.global_log_std
            floor = self.global_log_std.new_tensor(math.log(self.recovery_sigma_floor))
            log_std = torch.where(
                self.recovery_exploration_mask(observation),
                torch.maximum(self.global_log_std, floor),
                self.global_log_std,
            )
            return torch.where(observation.jnt_valid, log_std, torch.zeros_like(log_std))
        if contextual is None or self.conditional_sigma_head is None:
            raise RuntimeError("conditional sigma requires contextual joint features")
        offset = self.conditional_sigma_head(contextual[:, 1:17]).squeeze(-1)
        log_std = (self.global_log_std + math.log(2.0) * torch.tanh(offset)).clamp(math.log(0.05), self.max_log_std)
        return torch.where(observation.jnt_valid, log_std, torch.zeros_like(log_std))

    def _local_and_hand(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


        joint_weight = observation.jnt_valid.unsqueeze(-1).to(dtype=observation.jnt_current.dtype)
        history = self.history_encoder(observation.jnt_history, observation.jnt_valid)  # `[B,16,32]`
        current = observation.jnt_current * joint_weight  # `[B,16,5]`
        limits = observation.jnt_limits * joint_weight  # `[B,16,2]`
        tracking_lag = current[..., 1:2] - current[..., 0:1]
        span = (limits[..., 1:2] - limits[..., 0:1]).clamp_min(1.0e-6)
        normalized_q = 2.0 * (current[..., 0:1] - limits[..., 0:1]) / span - 1.0
        local_input = torch.cat((current, limits, tracking_lag, normalized_q, history), dim=-1)
        dynamic_local = self.local_encoder(local_input) * joint_weight
        gamma_raw, beta_raw = self.joint_geometry_film(geometry.tokens[:, 1:17]).chunk(2, dim=-1)
        gamma = FILM_LIMIT * torch.tanh(gamma_raw)
        beta = FILM_LIMIT * torch.tanh(beta_raw)
        local = ((1.0 + gamma) * dynamic_local + beta) * joint_weight
        film_modulation_rms = torch.linalg.vector_norm(local - dynamic_local, dim=-1) / math.sqrt(
            LOCAL_WIDTH
        )


        batch = local.shape[0]
        local_by_finger = local.reshape(batch, 4, 4, LOCAL_WIDTH).transpose(1, 2)
        mask_by_finger = observation.jnt_valid.reshape(batch, 4, 4).transpose(1, 2)
        finger_mean = masked_mean(local_by_finger, mask_by_finger, dim=2)
        finger_max = masked_max(local_by_finger, mask_by_finger, dim=2)
        joint_count = mask_by_finger.sum(dim=2, keepdim=True).to(dtype=local.dtype) / 4.0
        tip_geometry = self.tip_geometry_projection(geometry.tokens[:, 17:21])
        tip_contact = observation.owner_contact[:, 17:21]
        finger_input = torch.cat((finger_mean, finger_max, tip_geometry, tip_contact, joint_count), dim=-1)
        finger = self.finger_encoder(finger_input) * observation.tip_valid.unsqueeze(-1)  # `[B,4,48]`

        finger_mean_hand = masked_mean(finger, observation.tip_valid, dim=1)
        finger_max_hand = masked_max(finger, observation.tip_valid, dim=1)
        tip_count = observation.tip_valid.sum(dim=-1, keepdim=True).to(dtype=local.dtype) / 4.0
        palm_geometry = self.palm_geometry_projection(geometry.tokens[:, 0])
        palm_contact = observation.owner_contact[:, 0]
        hand_input = torch.cat((finger_mean_hand, finger_max_hand, palm_geometry, palm_contact, tip_count), dim=-1)
        hand = self.hand_encoder(hand_input)  # `[B,64]`
        return local, finger, hand, film_modulation_rms

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
    ) -> PalmRotationActorOutput:


        if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("actor observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid), "actor/geometry masks disagree"
        )
        local, finger, hand, film_modulation_rms = self._local_and_hand(observation, geometry)
        hand_per_joint = hand.unsqueeze(1).expand(-1, JOINT_COUNT, -1)
        raw_base = self.base_head(torch.cat((local, hand_per_joint), dim=-1)).squeeze(-1)
        base = self.base_action_limit * torch.tanh(raw_base)
        base = torch.where(observation.jnt_valid, base, torch.zeros_like(base))

        contextual = None
        if self.residual_enabled:
            contextual = self._contextual_tokens(observation, geometry, local, finger, hand)
            raw_residual = self.residual_head(torch.cat((contextual[:, 1:17], local), dim=-1)).squeeze(-1)
            residual = RESIDUAL_LIMIT * torch.tanh(raw_residual)
            residual = torch.where(observation.jnt_valid, residual, torch.zeros_like(residual))
        else:
            residual = torch.zeros_like(base)
        mean = base + residual
        return PalmRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, observation),
            base_mean=base,
            residual_mean=residual,
        )

    def _contextual_tokens(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        local: torch.Tensor,
        finger: torch.Tensor,
        hand: torch.Tensor,
    ) -> torch.Tensor:


        dynamic = torch.zeros_like(geometry.tokens)
        dynamic[:, 0] = self.palm_dynamic_projection(hand)
        dynamic[:, 1:17] = self.joint_dynamic_projection(local)
        dynamic[:, 17:21] = self.tip_dynamic_projection(finger)
        tokens = self.geometry_adapter(geometry.tokens) + dynamic
        tokens = tokens + self.owner_contact_projection(observation.owner_contact)
        return self.global_backbone(
            tokens,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )


class PalmRotationDirectActor(PalmRotationResidualActor):


    def __init__(
        self,
        *,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        local_skip: bool = True,
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
        phase_clock_enabled: bool = False,
    ) -> None:


        super().__init__(
            residual_enabled=True,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=BASE_ACTION_LIMIT,
            history_encoder=history_encoder,
            sigma_mode=sigma_mode,
            recovery_sigma_floor=recovery_sigma_floor,
        )
        del self.base_head
        del self.residual_head
        del self.residual_enabled
        self.local_skip = bool(local_skip)
        head_input_width = GEOMETRY_WIDTH + LOCAL_WIDTH if self.local_skip else GEOMETRY_WIDTH
        head_hidden_width = 64 if self.local_skip else 96
        self.direct_head = nn.Sequential(
            nn.LayerNorm(head_input_width),
            nn.Linear(head_input_width, head_hidden_width),
            nn.GELU(),
            nn.Linear(head_hidden_width, 1),
        )
        nn.init.orthogonal_(self.direct_head[-1].weight, gain=0.01)  # type: ignore[arg-type]
        nn.init.zeros_(self.direct_head[-1].bias)  # type: ignore[arg-type]
        self.phase_clock_enabled = bool(phase_clock_enabled)
        if self.phase_clock_enabled:

            self.phase_contextual_adapter = nn.Linear(2, GEOMETRY_WIDTH, bias=False)
            nn.init.zeros_(self.phase_contextual_adapter.weight)

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        *,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> PalmRotationActorOutput:


        if not _validated:
            if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
                raise ValueError("actor observation and geometry batch sizes disagree")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(observation.owner_valid == geometry.owner_valid), "actor/geometry masks disagree"
            )
        phase_clock = _phase_clock_input(
            phase_clock,
            enabled=self.phase_clock_enabled,
            batch_size=observation.jnt_current.shape[0],
            reference=observation.jnt_current,
            role="direct actor",
            validated=_validated,
        )
        local, finger, hand, film_modulation_rms = self._local_and_hand(observation, geometry)
        contextual = self._contextual_tokens(observation, geometry, local, finger, hand)
        contextual_joint = contextual[:, 1:17]
        if phase_clock is not None:

            phase_delta = self.phase_contextual_adapter(phase_clock).unsqueeze(1).expand(-1, JOINT_COUNT, -1)
            joint_mask = observation.jnt_valid.unsqueeze(-1).to(dtype=phase_delta.dtype)
            contextual_joint = contextual_joint + phase_delta * joint_mask
        direct_input = (
            torch.cat((contextual_joint, local), dim=-1) if self.local_skip else contextual_joint
        )
        raw_direct = self.direct_head(direct_input).squeeze(-1)
        mean = torch.tanh(raw_direct)
        mean = torch.where(observation.jnt_valid, mean, torch.zeros_like(mean))
        return PalmRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, observation),
            direct_mean=mean,
        )


class PalmRotationStructuredCritic(nn.Module):


    def __init__(self, *, phase_clock_enabled: bool = False) -> None:


        super().__init__()
        self.phase_clock_enabled = bool(phase_clock_enabled)
        owner_input_width = GEOMETRY_WIDTH + 2 + 4
        self.owner_adapter = nn.Sequential(
            nn.Linear(owner_input_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.backbone = _PalmRotationGraphBiasedTransformer(
            GraphBiasedTransformerCfg(
                hidden_width=128,
                layers=2,
                attention_heads=4,
                feedforward_width=256,
                dropout=0.0,
            )
        )
        self.object_adapter = nn.Sequential(nn.Linear(15, 128), nn.LayerNorm(128), nn.GELU())
        self.task_adapter = nn.Sequential(nn.Linear(9, 128), nn.LayerNorm(128), nn.GELU())
        self.value_head = nn.Sequential(
            nn.LayerNorm(7 * 128),
            nn.Linear(7 * 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        if self.phase_clock_enabled:

            self.phase_readout_adapter = nn.Linear(2, 7 * GEOMETRY_WIDTH, bias=False)
            nn.init.zeros_(self.phase_readout_adapter.weight)

    def forward(
        self,
        observation: PalmRotationCriticObservation,
        geometry: PalmRotationGeometry,
        *,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> torch.Tensor:


        if not _validated:
            if observation.jnt_state.shape[0] != geometry.tokens.shape[0]:
                raise ValueError("critic observation and geometry batch sizes disagree")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(observation.owner_valid == geometry.owner_valid), "critic/geometry masks disagree"
            )
        phase_clock = _phase_clock_input(
            phase_clock,
            enabled=self.phase_clock_enabled,
            batch_size=observation.jnt_state.shape[0],
            reference=observation.jnt_state,
            role="structured critic",
            validated=_validated,
        )

        owner_joint = torch.cat(
            (
                torch.zeros_like(observation.jnt_state[:, :1]),
                observation.jnt_state,
                torch.zeros_like(observation.jnt_state[:, :4]),
            ),
            dim=1,
        )
        owner_raw = torch.cat((geometry.tokens, observation.owner_contact, owner_joint), dim=-1)
        owner = self.owner_adapter(owner_raw) * observation.owner_valid.unsqueeze(-1)
        contextual = self.backbone(
            owner,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )
        palm = contextual[:, 0]
        joint = contextual[:, 1:17]
        tip = contextual[:, 17:21]
        joint_mean = masked_mean(joint, observation.jnt_valid, dim=1)
        joint_max = masked_max(joint, observation.jnt_valid, dim=1)
        tip_mean = masked_mean(tip, observation.tip_valid, dim=1)
        tip_max = masked_max(tip, observation.tip_valid, dim=1)
        object_hidden = self.object_adapter(observation.obj[:, 0])
        task_hidden = self.task_adapter(torch.cat((observation.task[:, 0], observation.reward_release), dim=-1))
        readout = torch.cat((palm, joint_mean, joint_max, tip_mean, tip_max, object_hidden, task_hidden), dim=-1)
        if phase_clock is not None:
            readout = readout + self.phase_readout_adapter(phase_clock)
        return self.value_head(readout).squeeze(-1)


class PalmRotationActorCritic(nn.Module):


    def __init__(
        self,
        *,
        arm: Literal["base", "residual", "direct", "direct_token"] = "residual",
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
        phase_clock_enabled: bool = False,
    ) -> None:


        super().__init__()
        self.phase_clock_enabled = bool(phase_clock_enabled)
        if self.phase_clock_enabled and arm not in {"direct", "direct_token"}:
            raise ValueError("phase_clock_enabled is supported only for direct or direct_token arm")
        if arm in {"direct", "direct_token"}:
            self.actor: PalmRotationResidualActor | PalmRotationDirectActor = PalmRotationDirectActor(
                initial_log_std=initial_log_std,
                max_log_std=max_log_std,
                history_encoder=history_encoder,
                local_skip=arm == "direct",
                sigma_mode=sigma_mode,
                recovery_sigma_floor=recovery_sigma_floor,
                phase_clock_enabled=self.phase_clock_enabled,
            )
        elif arm in {"base", "residual"}:
            self.actor = PalmRotationResidualActor(
                residual_enabled=arm == "residual",
                initial_log_std=initial_log_std,
                max_log_std=max_log_std,
                base_action_limit=base_action_limit,
                history_encoder=history_encoder,
                sigma_mode=sigma_mode,
                recovery_sigma_floor=recovery_sigma_floor,
            )
        else:
            raise ValueError(f"unsupported palm-rotation actor arm: {arm!r}")
        self.arm = arm
        self.critic = PalmRotationStructuredCritic(phase_clock_enabled=self.phase_clock_enabled)

    def trainable_parameter_sets(self) -> tuple[set[int], set[int]]:


        return {id(parameter) for parameter in self.actor.parameters()}, {
            id(parameter) for parameter in self.critic.parameters()
        }


def expanded_policy_log_std(log_std: torch.Tensor, mean: torch.Tensor, active: torch.Tensor) -> torch.Tensor:

    if log_std.numel() == 1:
        expanded = log_std.expand_as(mean)
    elif log_std.shape == mean.shape:
        expanded = log_std
    else:
        raise ValueError("policy log_std must be scalar or match the [batch,joint] mean")
    return torch.where(active, expanded, torch.zeros_like(expanded))


__all__ = [
    "expanded_policy_log_std",
    "BASE_ACTION_LIMIT",
    "GEOMETRY_WIDTH",
    "FILM_LIMIT",
    "HISTORY_LENGTH",
    "JOINT_COUNT",
    "OWNER_COUNT",
    "RESIDUAL_LIMIT",
    "TIP_COUNT",
    "PalmRotationActorCritic",
    "PalmRotationActorObservation",
    "PalmRotationActorOutput",
    "PalmRotationCriticObservation",
    "PalmRotationDirectActor",
    "PalmRotationGeometry",
    "PalmRotationResidualActor",
    "PalmRotationStructuredCritic",
    "masked_max",
    "masked_mean",
]
