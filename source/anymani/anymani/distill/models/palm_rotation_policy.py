r"""MVP80 small-base action policy、zero-init bounded residual与独立structured critic。

Actor raw input保留$O\to Z\to X\to H$阶段：task交付逐JOINT current/History30、limits、owner contact与masks；
冻结N040交付$q$-dependent owner geometry $Z^e\in\mathbb R^{B\times21\times128}$。Base路径不要求复杂
contextual token先学会动作，而由共享逐JOINT TCN、local MLP和finger-first set pooling直接输出：

$$
\mu^{base}_{j,t}=f_{base}(x_{j,t},s_t).
$$

一层graph-biased actor backbone只输出有界修正：

$$
\mu_{j,t}=\mu^{base}_{j,t}+0.2\tanh r_{j,t}.
$$

Residual最后一层权重与bias初始化为0，因此初始策略逐元素严格等于base；第一轮PPO先更新residual head，随后
梯度自然进入global trunk。Critic与actor完全分参，使用两层Pre-LN graph context、privileged object/contact
及critic-only LayerNorm-c，最终每environment输出一个hand-level scalar value。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg
from .temporal_encoder import PerJointTactileTemporalEncoder

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


def _bool_mask(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    r"""把task transport中的bool或0/1 mask规约为bool并验证shape。"""

    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype == torch.bool:
        return value
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # 避免GPU mask检查触发host sync
        torch.all(torch.isfinite(value) & ((value == 0) | (value == 1))),
        f"{name} numeric transport must contain finite 0/1 values",
    )
    return value.to(dtype=torch.bool)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    r"""沿entity轴计算有效项均值，ghost既不进分子也不进分母。"""

    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_mean tokens/mask shapes disagree")
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_max(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    r"""沿entity轴计算有效项max；全空slice返回0而不是dtype最小值。"""

    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_max tokens/mask shapes disagree")
    minimum = torch.finfo(tokens.dtype).min
    result = tokens.masked_fill(~mask.unsqueeze(-1), minimum).amax(dim=dim)
    has_value = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    return torch.where(has_value, result, torch.zeros_like(result))


@dataclass(frozen=True)
class PalmRotationGeometry:
    r"""冻结N040 owner tokens、valid mask与同序离散运动学图。"""

    tokens: torch.Tensor  # `[B,21,128]` FP32 policy边界
    owner_valid: torch.Tensor  # bool`[B,21]`
    shortest_path: torch.Tensor  # long`[B,21,21]`
    parent_direction: torch.Tensor  # long`[B,21,21]`
    child_direction: torch.Tensor  # long`[B,21,21]`

    def __post_init__(self) -> None:
        r"""验证geometry、mask、graph shape与共同device。"""

        batch = self.tokens.shape[0]
        if self.tokens.shape != (batch, OWNER_COUNT, GEOMETRY_WIDTH):
            raise ValueError("palm-rotation geometry tokens must have shape [B,21,128]")
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="geometry owner_valid", shape=(batch, OWNER_COUNT)),
        )
        graph_shape = (batch, OWNER_COUNT, OWNER_COUNT)
        if any(matrix.shape != graph_shape for matrix in (self.shortest_path, self.parent_direction, self.child_direction)):
            raise ValueError("palm-rotation graph matrices must have shape [B,21,21]")
        tensors = (self.tokens, self.owner_valid, self.shortest_path, self.parent_direction, self.child_direction)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("geometry tensors must share one device")


@dataclass(frozen=True)
class PalmRotationActorObservation:
    r"""Simulation-contact actor raw structured observation。"""

    jnt_current: torch.Tensor  # `[B,16,5]` q/u/a/own-contact/TIP-contact
    jnt_history: torch.Tensor  # `[B,30,16,5]`
    jnt_limits: torch.Tensor  # `[B,16,2]` qmin/qmax divided by pi
    owner_contact: torch.Tensor  # `[B,21,1]` binary
    jnt_valid: torch.Tensor  # bool`[B,16]`
    tip_valid: torch.Tensor  # bool`[B,4]`
    owner_valid: torch.Tensor  # bool`[B,21]`

    def __post_init__(self) -> None:
        r"""验证role/history axes、mask关系与共同device。"""

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
        r"""从ManagerBased named policy group恢复actor输入。"""

        names = {"jnt_current", "jnt_history", "jnt_limits", "owner_contact", "jnt_valid", "tip_valid", "owner_valid"}
        missing = names - set(observation)
        if missing:
            raise KeyError(f"palm-rotation actor observation misses {sorted(missing)}")
        return cls(**{name: observation[name] for name in names})


@dataclass(frozen=True)
class PalmRotationCriticObservation:
    r"""Privileged critic raw structured observation。"""

    jnt_state: torch.Tensor  # `[B,16,4]` q/qd/u/a
    owner_contact: torch.Tensor  # `[B,21,2]` force N + bit
    obj: torch.Tensor  # `[B,1,15]`
    task: torch.Tensor  # `[B,1,8]`
    reward_release: torch.Tensor  # `[B,1]` cell-level lambda
    jnt_valid: torch.Tensor
    tip_valid: torch.Tensor
    owner_valid: torch.Tensor

    def __post_init__(self) -> None:
        r"""验证privileged roles、masks与device。"""

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
        r"""从ManagerBased named critic group恢复privileged输入。"""

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
    r"""Factorized Gaussian mean分解与共享log-standard-deviation。"""

    mean: torch.Tensor  # `[B,16]`，base+residual
    base_mean: torch.Tensor  # `[B,16]`
    residual_mean: torch.Tensor  # `[B,16]`，绝对值不超过0.2
    film_modulation_rms: torch.Tensor  # `[B,16]`，geometry对dynamic local hidden的RMS改变量
    log_std: torch.Tensor  # scalar


class PalmRotationResidualActor(nn.Module):
    r"""Small local base为主、one-block full-hand action residual为辅的共享actor。"""

    def __init__(
        self,
        *,
        residual_enabled: bool = True,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
    ) -> None:
        r"""构造TCN、hierarchical pooling、base head与zero-init residual head。"""

        super().__init__()
        if initial_log_std > max_log_std:
            raise ValueError("initial_log_std must not exceed the exploration ceiling")
        if not 0.0 < base_action_limit <= 1.0 - RESIDUAL_LIMIT:
            raise ValueError("base_action_limit plus residual limit must remain within physical action bounds")
        self.residual_enabled = bool(residual_enabled)
        self.max_log_std = float(max_log_std)  # $\sigma_{max}=e^{-0.43}\approx0.65$，匹配N000 early budget
        self.base_action_limit = float(base_action_limit)  # base保留80% authority，residual保留20%
        self.history_encoder = PerJointTactileTemporalEncoder(
            joint_count=JOINT_COUNT,
            frame_dim=JOINT_FRAME_WIDTH,
            latent_dim=HISTORY_WIDTH,
            hidden_channels=(32, 32, 32),
        )
        local_input_width = JOINT_FRAME_WIDTH + 2 + 2 + HISTORY_WIDTH
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_input_width),
            nn.Linear(local_input_width, LOCAL_WIDTH),
            nn.GELU(),
            nn.Linear(LOCAL_WIDTH, LOCAL_WIDTH),
        )
        self.joint_geometry_film = nn.Sequential(
            nn.LayerNorm(GEOMETRY_WIDTH),
            nn.Linear(GEOMETRY_WIDTH, 2 * LOCAL_WIDTH),
        )  # $Z_j^e\mapsto(\gamma_j,\beta_j)$，不与低维dynamic vector直接拼接
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

        # Dynamic adapters只构造actor-specific$X^a$；冻结$Z^e$本身不被修改或纳入optimizer。
        self.geometry_adapter = nn.Linear(GEOMETRY_WIDTH, GEOMETRY_WIDTH)
        self.owner_contact_projection = nn.Linear(1, GEOMETRY_WIDTH, bias=False)
        self.palm_dynamic_projection = nn.Linear(HAND_WIDTH, GEOMETRY_WIDTH)
        self.joint_dynamic_projection = nn.Linear(LOCAL_WIDTH, GEOMETRY_WIDTH)
        self.tip_dynamic_projection = nn.Linear(FINGER_WIDTH, GEOMETRY_WIDTH)
        self.global_backbone = GraphBiasedTransformer(
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

    @torch.no_grad()
    def project_exploration_parameters(self) -> None:
        r"""把trainable global $\log\sigma$投影到N000 early-budget探索上界。

        PPO仍可根据数据降低标准差；entropy或policy gradient只能把它恢复到$-0.43$，不能再次进入物理动作
        30%以上被clamp、deterministic mean随噪声训练而抖动的区域。
        """

        self.global_log_std.clamp_(max=self.max_log_std)

    def _local_and_hand(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""形成dynamic-first FiLM local、per-finger与whole-hand summary。

        低维控制状态先形成$h_{t,j}^{dyn}$；冻结geometry只通过有界FiLM调制：

        $$
        h_{t,j}^{loc}
        =\left(1+0.25\tanh\gamma(Z_{t,j}^e)\right)\odot h_{t,j}^{dyn}
        +0.25\tanh\beta(Z_{t,j}^e).
        $$

        FiLM末层零初始化使初始$h^{loc}=h^{dyn}$，避免128维$Z^e$在训练起点覆盖$q,u,a$与History30。
        """

        joint_weight = observation.jnt_valid.unsqueeze(-1).to(dtype=observation.jnt_current.dtype)
        history = self.history_encoder(observation.jnt_history, observation.jnt_valid)  # `[B,16,32]`
        current = observation.jnt_current * joint_weight  # `[B,16,5]`
        limits = observation.jnt_limits * joint_weight  # `[B,16,2]`
        tracking_lag = current[..., 1:2] - current[..., 0:1]  # $(u-q)/\pi$
        span = (limits[..., 1:2] - limits[..., 0:1]).clamp_min(1.0e-6)
        normalized_q = 2.0 * (current[..., 0:1] - limits[..., 0:1]) / span - 1.0
        local_input = torch.cat((current, limits, tracking_lag, normalized_q, history), dim=-1)
        dynamic_local = self.local_encoder(local_input) * joint_weight  # $h^{dyn}$，`[B,16,64]`
        gamma_raw, beta_raw = self.joint_geometry_film(geometry.tokens[:, 1:17]).chunk(2, dim=-1)
        gamma = FILM_LIMIT * torch.tanh(gamma_raw)  # 有界multiplicative geometry modulation
        beta = FILM_LIMIT * torch.tanh(beta_raw)  # 有界additive geometry modulation
        local = ((1.0 + gamma) * dynamic_local + beta) * joint_weight  # $h^{loc}$，ghost严格为0
        film_modulation_rms = torch.sqrt((local - dynamic_local).square().mean(dim=-1))  # `[B,16]`

        # Canonical JOINT axis是depth-major；转成`[B,finger,depth,D]`后先在每根finger内pool。
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
        r"""输出base、bounded residual与最终masked Gaussian mean。"""

        if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("actor observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid), "actor/geometry masks disagree"
        )
        local, finger, hand, film_modulation_rms = self._local_and_hand(observation, geometry)
        hand_per_joint = hand.unsqueeze(1).expand(-1, JOINT_COUNT, -1)
        raw_base = self.base_head(torch.cat((local, hand_per_joint), dim=-1)).squeeze(-1)
        base = self.base_action_limit * torch.tanh(raw_base)  # $\mu^{base}\in[-0.8,0.8]$
        base = torch.where(observation.jnt_valid, base, torch.zeros_like(base))

        if self.residual_enabled:
            dynamic = torch.zeros_like(geometry.tokens)
            dynamic[:, 0] = self.palm_dynamic_projection(hand)
            dynamic[:, 1:17] = self.joint_dynamic_projection(local)
            dynamic[:, 17:21] = self.tip_dynamic_projection(finger)
            tokens = self.geometry_adapter(geometry.tokens) + dynamic
            tokens = tokens + self.owner_contact_projection(observation.owner_contact)
            contextual = self.global_backbone(
                tokens,
                geometry.shortest_path,
                geometry.parent_direction,
                geometry.child_direction,
                geometry.owner_valid,
            )
            raw_residual = self.residual_head(torch.cat((contextual[:, 1:17], local), dim=-1)).squeeze(-1)
            residual = RESIDUAL_LIMIT * torch.tanh(raw_residual)
            residual = torch.where(observation.jnt_valid, residual, torch.zeros_like(residual))
        else:
            residual = torch.zeros_like(base)
        mean = base + residual
        return PalmRotationActorOutput(
            mean=mean,
            base_mean=base,
            residual_mean=residual,
            film_modulation_rms=film_modulation_rms,
            log_std=self.global_log_std,
        )


class PalmRotationStructuredCritic(nn.Module):
    r"""两层graph context与critic-only LN-c的privileged hand-level value。"""

    def __init__(self) -> None:
        r"""构造owner/object/task adapters、两层Pre-LN backbone与scalar readout。"""

        super().__init__()
        owner_input_width = GEOMETRY_WIDTH + 2 + 4
        self.owner_adapter = nn.Sequential(
            nn.Linear(owner_input_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )  # Linear→LN→activation是TOPPO critic-only LN-c对应边界
        self.backbone = GraphBiasedTransformer(
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

    def forward(self, observation: PalmRotationCriticObservation, geometry: PalmRotationGeometry) -> torch.Tensor:
        r"""融合privileged owner/object/task state并输出`[B]` scalar value。"""

        if observation.jnt_state.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("critic observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid), "critic/geometry masks disagree"
        )
        owner_joint = torch.zeros(
            observation.jnt_state.shape[0],
            OWNER_COUNT,
            4,
            device=observation.jnt_state.device,
            dtype=observation.jnt_state.dtype,
        )
        owner_joint[:, 1:17] = observation.jnt_state
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
        return self.value_head(readout).squeeze(-1)


class PalmRotationActorCritic(nn.Module):
    r"""完全分参的actor/critic checkpoint namespace容器。"""

    def __init__(
        self,
        *,
        residual_enabled: bool = True,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
    ) -> None:
        r"""分别实例化actor与critic；冻结N040不属于本module。"""

        super().__init__()
        self.actor = PalmRotationResidualActor(
            residual_enabled=residual_enabled,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=base_action_limit,
        )
        self.critic = PalmRotationStructuredCritic()

    def trainable_parameter_sets(self) -> tuple[set[int], set[int]]:
        r"""返回actor/critic parameter object IDs供optimizer/checkpoint断言。"""

        return {id(parameter) for parameter in self.actor.parameters()}, {
            id(parameter) for parameter in self.critic.parameters()
        }


__all__ = [
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
    "PalmRotationGeometry",
    "PalmRotationResidualActor",
    "PalmRotationStructuredCritic",
    "masked_max",
    "masked_mean",
]
