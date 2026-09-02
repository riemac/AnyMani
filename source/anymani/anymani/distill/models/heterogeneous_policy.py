r"""轻量structured heterogeneous actor candidates与独立masked-pooling scalar critic。

Actor统一使用共享per-joint local encoder与scalar head。``gated_pool``以零初始化标量门加入全手masked-mean
coordination residual，初始函数严格等于local-only；``cross_attention``让JOINT queries读取21个geometry/TIP
memory tokens。Critic拥有完全独立参数，只共享同一份冻结geometry输入tensor，不共享任何learned module。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .structured_heterogeneous import (
    JOINT_COUNT,
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredActorOutput,
    StructuredCriticObservation,
    StructuredCriticOutput,
)
from .temporal_encoder import PerJointHistoryStackEncoder

CoordinationKind = Literal["local", "gated_pool", "cross_attention"]


def masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    r"""沿entity轴做mask-aware mean，分母至少1且ghost无gradient贡献。"""

    if tokens.ndim != valid_mask.ndim + 1 or tokens.shape[:-1] != valid_mask.shape:
        raise ValueError("tokens and valid_mask shapes must differ only by feature axis")
    weights = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    numerator = (tokens * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


@dataclass(frozen=True)
class StructuredActorCfg:
    r"""Actor capacity与coordination机制配置。"""

    hidden_width: int = 128
    temporal_width: int = 32
    coordination: CoordinationKind = "gated_pool"
    attention_heads: int = 4
    initial_log_std: float = -0.5

    def __post_init__(self) -> None:
        r"""验证width、heads与候选名称。"""

        if min(self.hidden_width, self.temporal_width, self.attention_heads) < 1:
            raise ValueError("actor widths/heads must be positive")
        if self.hidden_width % self.attention_heads != 0:
            raise ValueError("actor hidden width must be divisible by attention heads")
        if self.coordination not in {"local", "gated_pool", "cross_attention"}:
            raise ValueError(f"unsupported coordination candidate {self.coordination!r}")
        if not math.isfinite(self.initial_log_std):
            raise ValueError("actor initial_log_std must be finite")


@dataclass(frozen=True)
class StructuredCriticCfg:
    r"""Independent masked-pooling critic capacity。"""

    hidden_width: int = 128

    def __post_init__(self) -> None:
        r"""验证positive width。"""

        if self.hidden_width < 1:
            raise ValueError("critic hidden width must be positive")


class StructuredHeterogeneousActor(nn.Module):
    r"""共享per-joint Gaussian mean actor，输出ghost-zero mean与global scalar logstd。

    Joint local raw width为$3_{current}+2_{limits}+D_t$。Geometry JOINT view严格取统一owner tokens的
    ``1:17``，不构造第二latent。History encoder参数在16个joint之间共享。
    """

    def __init__(self, cfg: StructuredActorCfg = StructuredActorCfg()) -> None:
        r"""构造local trunk、可选coordination与共享head。"""

        super().__init__()
        self.cfg = cfg
        width = cfg.hidden_width
        self.history_encoder = PerJointHistoryStackEncoder(
            joint_count=JOINT_COUNT, frame_dim=4, latent_dim=cfg.temporal_width
        )
        local_input_width = 3 + 2 + cfg.temporal_width + 128
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_input_width),
            nn.Linear(local_input_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        if cfg.coordination == "gated_pool":
            self.coordination_projection = nn.Linear(width, width)
            self.coordination_gate = nn.Linear(width, width)
            self.coordination_scale = nn.Parameter(torch.zeros(()))  # $\alpha=0$精确退化local-only
        elif cfg.coordination == "cross_attention":
            self.owner_projection = nn.Linear(128, width)
            self.tip_contact_projection = nn.Linear(1, width, bias=False)
            self.cross_attention = nn.MultiheadAttention(
                width, cfg.attention_heads, dropout=0.0, batch_first=True
            )
            self.cross_attention_norm = nn.LayerNorm(width)
        output_layer = nn.Linear(width, 1)
        nn.init.orthogonal_(output_layer.weight, gain=0.01)  # type: ignore[arg-type]  # torch stub误标gain为int
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
        self.action_head = nn.Sequential(nn.LayerNorm(width), output_layer)
        self.global_log_std = nn.Parameter(torch.tensor(float(cfg.initial_log_std)))  # 唯一$\theta^{av}$

    def forward(
        self,
        observation: StructuredActorObservation,
        geometry: GeometryTokenBatch,
    ) -> StructuredActorOutput:
        r"""计算factorized Gaussian mean；所有invalid joints严格输出0。"""

        if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("actor observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # 避免GPU host sync的PyTorch runtime原语
            torch.all(observation.owner_valid == geometry.owner_valid),
            "actor and geometry owner masks disagree",
        )
        joint_mask = observation.jnt_valid
        joint_weight = joint_mask.unsqueeze(-1).to(dtype=observation.jnt_current.dtype)
        history = self.history_encoder(observation.jnt_history, joint_mask)
        current = observation.jnt_current * joint_weight
        limits = observation.jnt_limits * joint_weight
        geometry_tokens = geometry.tokens * geometry.owner_valid.unsqueeze(-1).to(dtype=geometry.tokens.dtype)
        joint_geometry = geometry_tokens[:, 1:17] * joint_weight
        local_input = torch.cat((current, limits, history, joint_geometry), dim=-1)
        local_hidden = self.local_encoder(local_input) * joint_weight

        if self.cfg.coordination == "local":
            contextual = local_hidden
        elif self.cfg.coordination == "gated_pool":
            pooled = masked_mean(local_hidden, joint_mask, dim=1)  # hand-level current/history summary$[B,D]$
            residual = self.coordination_projection(pooled).unsqueeze(1)
            gate = torch.sigmoid(self.coordination_gate(local_hidden))
            contextual = local_hidden + torch.tanh(self.coordination_scale) * gate * residual
        else:
            memory = self.owner_projection(geometry_tokens)
            tip_contact = observation.tip_contact * observation.tip_valid.unsqueeze(-1).to(
                dtype=observation.tip_contact.dtype
            )
            tip_delta = self.tip_contact_projection(tip_contact)
            memory = memory.clone()
            memory[:, 17:21] = memory[:, 17:21] + tip_delta
            attended, _ = self.cross_attention(
                local_hidden,
                memory,
                memory,
                key_padding_mask=~geometry.owner_valid,
                need_weights=False,
            )
            contextual = local_hidden + self.cross_attention_norm(attended)
        contextual = contextual * joint_weight
        mean = self.action_head(contextual).squeeze(-1)
        mean = torch.where(joint_mask, mean, torch.zeros_like(mean))
        return StructuredActorOutput(mean=mean, log_std=self.global_log_std)


class StructuredHeterogeneousCritic(nn.Module):
    r"""完全分参的privileged owner-token masked-pooling scalar critic。

    Owner input为$[Z^e,contact,scatter(joint\ state)]$；object/task各自编码后与owner masked mean融合。
    最终每environment只输出一个hand-level value。
    """

    def __init__(self, cfg: StructuredCriticCfg = StructuredCriticCfg()) -> None:
        r"""构造独立owner/object/task adapters与scalar head。"""

        super().__init__()
        self.cfg = cfg
        width = cfg.hidden_width
        self.owner_encoder = nn.Sequential(
            nn.LayerNorm(128 + 2 + 4),
            nn.Linear(128 + 2 + 4, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.object_encoder = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, width), nn.GELU())
        self.task_encoder = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, width), nn.GELU())
        self.value_head = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        observation: StructuredCriticObservation,
        geometry: GeometryTokenBatch,
    ) -> StructuredCriticOutput:
        r"""计算mask/permutation-invariant hand-level scalar value。"""

        if observation.jnt_state.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("critic observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # 避免GPU host sync的PyTorch runtime原语
            torch.all(observation.owner_valid == geometry.owner_valid),
            "critic and geometry owner masks disagree",
        )
        owner_mask = observation.owner_valid
        owner_weight = owner_mask.unsqueeze(-1).to(dtype=geometry.tokens.dtype)
        geometry_tokens = geometry.tokens * owner_weight
        owner_contact = observation.owner_contact * owner_weight
        joint_state = observation.jnt_state * observation.jnt_valid.unsqueeze(-1).to(
            dtype=observation.jnt_state.dtype
        )
        owner_joint_state = torch.zeros(
            observation.jnt_state.shape[0], 21, 4, dtype=joint_state.dtype, device=joint_state.device
        )
        owner_joint_state[:, 1:17] = joint_state
        owner_raw = torch.cat((geometry_tokens, owner_contact, owner_joint_state), dim=-1)
        owner_hidden = self.owner_encoder(owner_raw) * owner_weight
        pooled_owner = masked_mean(owner_hidden, owner_mask, dim=1)
        object_hidden = self.object_encoder(observation.obj[:, 0])
        task_hidden = self.task_encoder(observation.task[:, 0])
        fused = torch.cat((pooled_owner, object_hidden, task_hidden), dim=-1)
        value = self.value_head(fused).squeeze(-1)
        return StructuredCriticOutput(value=value)


class StructuredActorCriticPackage(nn.Module):
    r"""Actor/critic namespace容器；两者没有共享trainable parameters。"""

    def __init__(
        self,
        actor_cfg: StructuredActorCfg = StructuredActorCfg(),
        critic_cfg: StructuredCriticCfg = StructuredCriticCfg(),
    ) -> None:
        r"""分别实例化actor与critic modules。"""

        super().__init__()
        self.actor = StructuredHeterogeneousActor(actor_cfg)
        self.critic = StructuredHeterogeneousCritic(critic_cfg)

    def trainable_parameter_sets(self) -> tuple[set[int], set[int]]:
        r"""返回actor/critic parameter object IDs，供checkpoint/optimizer边界断言。"""

        return ({id(parameter) for parameter in self.actor.parameters()}, {id(parameter) for parameter in self.critic.parameters()})


__all__ = [
    "CoordinationKind",
    "StructuredActorCfg",
    "StructuredActorCriticPackage",
    "StructuredCriticCfg",
    "StructuredHeterogeneousActor",
    "StructuredHeterogeneousCritic",
    "masked_mean",
]
