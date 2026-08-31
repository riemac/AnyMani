r"""canonical masked embodiment policy：一次 owner graph forward 闭合 actor/critic。

输入 contract 是已经由 ``assets`` / ``robots`` / ``tasks`` lower 好的结构化 batch：

* ``owner_features`` 为 $[B,21,F_o]$，实体顺序为 PALM、16 个 JOINT、4 个 TIP；
* ``joint_features`` 为 $[B,16,F_j]$，只包含当前 q、q-dot、processed action 等动态状态；
* ``owner_valid_mask`` 与 ``joint_valid_mask`` 为 ``True=真实有效``；ghost joint/owner 恒为 False；
* graph relation 的三轴为 $[21,21]$ 或 $[B,21,21]$，并和 owner permutation 同步；
* ``asset_row`` 只用于 evidence-bank routing，不作为可学习 absolute slot embedding。

一次 forward 先把 JOINT state 注入同索引 JOINT owner，再运行共享的 graph-biased Transformer：

$$
H = \mathcal{T}_{\theta}(X_{owner}+\operatorname{scatter}(X_{joint}), G, M_{owner}).
$$

actor 只读取 JOINT owner：

$$
\mu_i=h_{\mu}(H_{owner(i)}),\qquad
\log\sigma_i=\log\sigma_{global},
$$

其中 $h_\mu$ 对全部 16 个 slots 共享，不持有 absolute joint embedding；critic 只从 PALM
context 读取 scalar value。于是同步置换 joint/owner features、graph 两轴和 masks 后，action
按同一置换等变，value 不变。inactive slot 在 head 输出、PPO probability 与环境 action 边界
继续由显式 mask 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg
from .input_adapters.geometry import StaticGeometryEvidence

CANONICAL_OWNER_COUNT = 21
"""PALM + 16 JOINT + 4 TIP 的固定 owner 数。"""

CANONICAL_JOINT_COUNT = 16
"""canonical v1 的固定 joint 数。"""


@dataclass(frozen=True)
class CanonicalPolicyCfg:
    r"""vanilla masked PPO 的结构容量合同。

    数值锚点：hidden $D=128$、2 个 Pre-LN blocks、4 heads、FFN width 256、dropout 0；
    它们与已通过的 heterogeneous forward/backward probe 对齐。第一版 exploration 只
    学一个 global scalar `log_std`，逐 token variance 留作后续消融。
    """

    owner_feature_dim: int = 32
    """PALM/JOINT/TIP owner 的 task/evidence 拼接特征维度 $F_o$。"""

    joint_feature_dim: int = 16
    """动态 JOINT state 特征维度 $F_j$，不含 owner graph relation。"""

    hidden_width: int = 128
    """共享 graph token 宽度 $D$。"""

    layers: int = 2
    """Pre-LN graph transformer block 数。"""

    attention_heads: int = 4
    """attention head 数 $H$，要求整除 hidden width。"""

    feedforward_width: int = 256
    """逐实体 FFN 宽度。"""

    dropout: float = 0.0
    """PPO baseline 默认确定性 retained trunk。"""

    initial_log_std: float = -0.5
    r"""全局 Gaussian exploration 的初始 $\log\sigma$，不是 per-joint 参数。"""

    geometry_entity_width: int = 128
    """retained geometry encoder 的统一 PALM/JOINT/TIP final-norm token 宽度。"""

    temporal_feature_dim: int = 0
    """可选逐JOINT History30摘要宽度；0表示current-frame/legacy route不注入时序latent。"""

    def __post_init__(self) -> None:
        r"""拒绝与 canonical entity/action schema 不兼容的容量。"""

        if (
            min(
                self.owner_feature_dim,
                self.joint_feature_dim,
                self.hidden_width,
                self.layers,
                self.attention_heads,
                self.feedforward_width,
                self.geometry_entity_width,
            )
            < 1
        ):
            raise ValueError("canonical policy dimensions must be positive")
        if self.hidden_width % self.attention_heads:
            raise ValueError("canonical policy hidden_width must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("canonical policy dropout must lie in [0,1)")
        if self.temporal_feature_dim < 0:
            raise ValueError("temporal_feature_dim must be non-negative")


@dataclass(frozen=True)
class EmbodimentPolicyInput:
    r"""一次 canonical policy forward 的结构化输入。"""

    owner_features: torch.Tensor
    """形状 `[B,21,F_o]`；PALM/JOINT/TIP 的 static evidence 与 task state。"""

    joint_features: torch.Tensor
    """形状 `[B,16,F_j]`；动态 q、q-dot、contact/action 等 JOINT state。"""

    owner_valid_mask: torch.Tensor
    """形状 `[B,21]`，``True`` 表示 PALM/JOINT/TIP owner 真实存在。"""

    joint_valid_mask: torch.Tensor
    """形状 `[B,16]`，``True`` 表示 active revolute joint。"""

    shortest_path: torch.Tensor
    """形状 `[21,21]` 或 `[B,21,21]` 的无向 graph distance bucket。"""

    parent_direction: torch.Tensor
    """形状 `[21,21]` 或 `[B,21,21]` 的 parent-direction bucket。"""

    child_direction: torch.Tensor
    """形状 `[21,21]` 或 `[B,21,21]` 的 child-direction bucket。"""

    asset_row: torch.Tensor
    """形状 `[B]` 的离散 evidence-bank row；不做 RMS/连续归一化。"""

    geometry_entities: torch.Tensor | None = None
    """可选 `[B,21,D]` retained unified geometry $Z$；JOINT view 不单独存储。"""

    temporal_features: torch.Tensor | None = None
    """可选 `[B,16,D_t]` 逐JOINT History30摘要；跨JOINT交互留给policy adapter。"""

    def __post_init__(self) -> None:
        r"""检查所有 batch/entity/action axes 的闭合。"""

        if self.owner_features.ndim != 3 or self.owner_features.shape[1] != CANONICAL_OWNER_COUNT:
            raise ValueError("owner_features must have shape [B,21,F_o]")
        if self.joint_features.ndim != 3 or self.joint_features.shape[1] != CANONICAL_JOINT_COUNT:
            raise ValueError("joint_features must have shape [B,16,F_j]")
        batch_size = self.owner_features.shape[0]
        if self.joint_features.shape[0] != batch_size:
            raise ValueError("owner_features and joint_features must share batch size")
        if (
            self.owner_valid_mask.shape != (batch_size, CANONICAL_OWNER_COUNT)
            or self.owner_valid_mask.dtype != torch.bool
        ):
            raise ValueError("owner_valid_mask must be bool [B,21]")
        if (
            self.joint_valid_mask.shape != (batch_size, CANONICAL_JOINT_COUNT)
            or self.joint_valid_mask.dtype != torch.bool
        ):
            raise ValueError("joint_valid_mask must be bool [B,16]")
        if self.asset_row.shape != (batch_size,):
            raise ValueError("asset_row must have shape [B]")
        if self.geometry_entities is not None:
            if self.geometry_entities.ndim != 3 or self.geometry_entities.shape[:2] != (
                batch_size,
                CANONICAL_OWNER_COUNT,
            ):
                raise ValueError("geometry_entities must have shape [B,21,D]")
        if self.temporal_features is not None:
            if self.temporal_features.ndim != 3 or self.temporal_features.shape[:2] != (
                batch_size,
                CANONICAL_JOINT_COUNT,
            ):
                raise ValueError("temporal_features must have shape [B,16,D_t]")
        for graph in (self.shortest_path, self.parent_direction, self.child_direction):
            if graph.shape not in {
                (CANONICAL_OWNER_COUNT, CANONICAL_OWNER_COUNT),
                (batch_size, CANONICAL_OWNER_COUNT, CANONICAL_OWNER_COUNT),
            }:
                raise ValueError("graph relations must have shape [21,21] or [B,21,21]")


@dataclass(frozen=True)
class EmbodimentPolicyOutput:
    r"""policy 的结构化输出；rl_games adapter 在边界处再 flatten。"""

    action_mean: torch.Tensor
    """形状 `[B,16]` 的共享 JOINT action mean；inactive slot 为 0。"""

    action_log_std: torch.Tensor
    """形状 `[B,16]`；所有位置 broadcast 同一个可学习 global scalar。"""

    value: torch.Tensor
    """形状 `[B,1]` 的 PALM-context symmetric critic value。"""

    joint_valid_mask: torch.Tensor
    """原样透传 `[B,16]` active mask，供 PPO probability contract 使用。"""

    owner_valid_mask: torch.Tensor
    """原样透传 `[B,21]` owner mask，供 diagnostics / auxiliary 使用。"""

    aux_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    """只含 diagnostics，不参与 rl_games tuple contract。"""


@dataclass(frozen=True)
class CanonicalEvidenceBank:
    r"""只读 canonical evidence bank 的 row-gather façade。

    bank tensors 的第一轴是 asset row；它们由 assets/distill representations 离线生成，模型
    只按 runtime `asset_row` gather，不从 URDF/link name 进行二次语义推断。
    """

    evidence: StaticGeometryEvidence
    """第一轴为 asset row 的 canonical raw geometry evidence。"""

    asset_ids: tuple[str, ...]
    """row 对应的稳定 source asset IDs。"""

    physical_geometry_hashes: tuple[str, ...]
    """row 对应的原始真实 geometry identity；不包含 ghost。"""

    def __post_init__(self) -> None:
        r"""验证 bank row、canonical owner/joint 轴和 provenance 对齐。"""

        if self.evidence.anchors.ndim != 3:
            raise ValueError("canonical evidence bank must be batched by asset row")
        row_count = self.evidence.anchors.shape[0]
        if len(self.asset_ids) != row_count or len(self.physical_geometry_hashes) != row_count:
            raise ValueError("canonical evidence provenance must align with asset rows")
        if self.evidence.home_surface_points.shape[1] != CANONICAL_OWNER_COUNT:
            raise ValueError("canonical evidence must contain 21 owner slots")
        if self.evidence.space_screws.shape[1] != CANONICAL_JOINT_COUNT:
            raise ValueError("canonical evidence must contain 16 joint slots")

    def gather(self, asset_row: torch.Tensor) -> StaticGeometryEvidence:
        r"""按 `[B]` asset row gather raw evidence、mask 与真实 graph。"""

        if asset_row.ndim != 1 or asset_row.dtype not in {torch.int32, torch.int64, torch.long}:
            raise ValueError("asset_row must be a rank-1 integer tensor")
        if torch.any(asset_row < 0) or torch.any(asset_row >= self.evidence.anchors.shape[0]):
            raise IndexError("asset_row contains an evidence-bank row outside the bank")
        evidence = self.evidence
        return StaticGeometryEvidence(
            anchors=evidence.anchors[asset_row],
            home_surface_points=evidence.home_surface_points[asset_row],
            home_surface_mask=evidence.home_surface_mask[asset_row],
            palm_normal=evidence.palm_normal[asset_row],
            space_screws=evidence.space_screws[asset_row],
            q_home=evidence.q_home[asset_row],
            entity_role=evidence.entity_role[asset_row],
            entity_joint_index=evidence.entity_joint_index[asset_row],
            joint_entity_index=evidence.joint_entity_index[asset_row],
            shortest_path=evidence.shortest_path[asset_row],
            parent_direction=evidence.parent_direction[asset_row],
            child_direction=evidence.child_direction[asset_row],
            entity_valid_mask=(
                evidence.entity_valid_mask[asset_row] if evidence.entity_valid_mask is not None else None
            ),
            joint_valid_mask=(evidence.joint_valid_mask[asset_row] if evidence.joint_valid_mask is not None else None),
            anchor_valid_mask=(
                evidence.anchor_valid_mask[asset_row] if evidence.anchor_valid_mask is not None else None
            ),
        )

    def to(self, device: torch.device | str) -> CanonicalEvidenceBank:
        r"""把静态 bank 一次搬到 policy device；provenance 保持 host tuple。"""

        target = torch.device(device)
        evidence = self.evidence

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return value.to(target) if value is not None else None

        return CanonicalEvidenceBank(
            evidence=StaticGeometryEvidence(
                anchors=evidence.anchors.to(target),
                home_surface_points=evidence.home_surface_points.to(target),
                home_surface_mask=evidence.home_surface_mask.to(target),
                palm_normal=evidence.palm_normal.to(target),
                space_screws=evidence.space_screws.to(target),
                q_home=evidence.q_home.to(target),
                entity_role=evidence.entity_role.to(target),
                entity_joint_index=evidence.entity_joint_index.to(target),
                joint_entity_index=evidence.joint_entity_index.to(target),
                shortest_path=evidence.shortest_path.to(target),
                parent_direction=evidence.parent_direction.to(target),
                child_direction=evidence.child_direction.to(target),
                entity_valid_mask=move(evidence.entity_valid_mask),
                joint_valid_mask=move(evidence.joint_valid_mask),
                anchor_valid_mask=move(evidence.anchor_valid_mask),
            ),
            asset_ids=self.asset_ids,
            physical_geometry_hashes=self.physical_geometry_hashes,
        )


class EmbodimentPolicy(nn.Module):
    r"""一次 canonical masked owner forward 的 actor/critic 装配。"""

    def __init__(self, config: CanonicalPolicyCfg = CanonicalPolicyCfg()) -> None:
        super().__init__()
        self.config = config
        self.owner_projection = nn.Sequential(
            nn.Linear(config.owner_feature_dim, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, config.hidden_width),
        )
        self.joint_projection = nn.Sequential(
            nn.Linear(config.joint_feature_dim, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, config.hidden_width),
        )
        self.geometry_projection = nn.Linear(config.geometry_entity_width, config.hidden_width)
        self.temporal_projection = (
            nn.Linear(config.temporal_feature_dim, config.hidden_width)
            if config.temporal_feature_dim > 0
            else None
        )  # History30 route才创建；legacy current-frame route不持有空时序参数
        self.backbone = GraphBiasedTransformer(
            GraphBiasedTransformerCfg(
                hidden_width=config.hidden_width,
                layers=config.layers,
                attention_heads=config.attention_heads,
                feedforward_width=config.feedforward_width,
                dropout=config.dropout,
            )
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.hidden_width),
            nn.Linear(config.hidden_width, 1),
        )  # contextual token→无量纲Gaussian mean；全有效JOINT共享最薄输出投影
        self.value_head = nn.Linear(config.hidden_width, 1)  # PALM context -> scalar critic
        self.global_log_std = nn.Parameter(torch.tensor(float(config.initial_log_std)))  # 唯一 exploration 参数

    def forward(self, inputs: EmbodimentPolicyInput) -> EmbodimentPolicyOutput:
        r"""执行一次 `[B,21]` owner graph forward 并返回结构化 actor/critic 输出。"""

        # owner feature 先投影到统一 D 维；不加入 slot index embedding，保证 token 置换等变。
        owner_tokens = self.owner_projection(inputs.owner_features)  # `[B,21,D]`
        joint_tokens = self.joint_projection(inputs.joint_features)  # `[B,16,D]`
        if inputs.temporal_features is not None:
            if self.temporal_projection is None:
                raise ValueError("temporal_features were provided but temporal_feature_dim is zero")
            if inputs.temporal_features.shape[-1] != self.config.temporal_feature_dim:
                raise ValueError(
                    "temporal feature width disagrees with policy config: "
                    f"actual={inputs.temporal_features.shape[-1]} expected={self.config.temporal_feature_dim}"
                )
            joint_tokens = joint_tokens + self.temporal_projection(inputs.temporal_features)
        if inputs.geometry_entities is not None:
            owner_tokens = owner_tokens + self.geometry_projection(inputs.geometry_entities)
        joint_mask_float = inputs.joint_valid_mask.unsqueeze(-1).to(dtype=joint_tokens.dtype)
        joint_tokens = joint_tokens * joint_mask_float  # ghost joint state 不能写入 owner latent
        owner_tokens = owner_tokens.clone()
        owner_tokens[:, 1 : 1 + CANONICAL_JOINT_COUNT] = (
            owner_tokens[:, 1 : 1 + CANONICAL_JOINT_COUNT] + joint_tokens
        )  # scatter JOINT state 到同索引 JOINT owner

        # joint mask 是 owner mask 的必要约束；ghost owner 永远不能作为 key/value 或 query。
        owner_mask = inputs.owner_valid_mask.clone()
        owner_mask[:, 1 : 1 + CANONICAL_JOINT_COUNT] &= inputs.joint_valid_mask
        tokens = self.backbone(
            owner_tokens,
            inputs.shortest_path,
            inputs.parent_direction,
            inputs.child_direction,
            entity_valid_mask=owner_mask,
        )  # `[B,21,D]`，一次 masked graph forward

        joint_context = tokens[:, 1 : 1 + CANONICAL_JOINT_COUNT]  # `[B,16,D]`
        action_mean = self.action_head(joint_context).squeeze(-1)  # 共享逐 JOINT mean head
        action_mean = action_mean * inputs.joint_valid_mask.to(dtype=action_mean.dtype)  # inactive mean 恒为 0
        action_log_std = self.global_log_std.expand_as(action_mean)  # 一个 scalar 广播为 `[B,16]`
        value = self.value_head(tokens[:, :1, :]).squeeze(-1)  # PALM context symmetric critic，`[B,1]`
        return EmbodimentPolicyOutput(
            action_mean=action_mean,
            action_log_std=action_log_std,
            value=value,
            joint_valid_mask=inputs.joint_valid_mask,
            owner_valid_mask=owner_mask,
            aux_outputs={"owner_latent": tokens, "asset_row": inputs.asset_row.to(dtype=action_mean.dtype)},
        )


__all__ = [
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_OWNER_COUNT",
    "CanonicalEvidenceBank",
    "CanonicalPolicyCfg",
    "EmbodimentPolicy",
    "EmbodimentPolicyInput",
    "EmbodimentPolicyOutput",
]
