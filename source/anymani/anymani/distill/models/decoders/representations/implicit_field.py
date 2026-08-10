r"""多带宽密度与 sampled-edge 距离灵敏度解码器。

解码器只在 SSL 期间存在。密度路径读取 joint-sign 偶的 owner 零阶表征与共享查询特征；
距离灵敏度路径额外读取对应 JOINT 的符号奇一阶表征。query stratum、最近点、距离标签、
Jacobian 与场标签都不进入 decoder 输入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImplicitFieldDecoderConfig:
    r"""训练期 field decoders 的类型宽度与容量。"""

    zero_order_width: int
    first_order_width: int
    query_width: int
    hidden_width: int = 128
    bandwidth_count: int = 4
    residual_blocks: int = 3

    def __post_init__(self) -> None:
        r"""验证 decoder 的所有宽度和带宽通道数严格为正。"""

        values = (
            self.zero_order_width,
            self.first_order_width,
            self.query_width,
            self.hidden_width,
            self.bandwidth_count,
            self.residual_blocks,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all implicit decoder widths/counts must be positive")


class _FiLMResidualBlock(nn.Module):
    r"""由 owner zero-order latent 调制的 query-wise 残差块。"""

    def __init__(self, hidden_width: int, condition_width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_width)  # 每个 query 独立规范化 hidden channels
        self.modulation = nn.Linear(condition_width, 2 * hidden_width)  # owner latent -> $(\gamma,\beta)$
        self.update = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        r"""对 `[B,G,N_Q,D]` query features 施加 owner-conditioned FiLM。"""

        gamma, beta = self.modulation(condition).chunk(2, dim=-1)  # 每项 `[B,G,N_Q,D]`
        modulated = self.normalization(hidden) * (1.0 + gamma) + beta  # FiLM：$(1+\gamma)h+\beta$
        return hidden + self.update(modulated)  # residual 保留 query 几何证据


class ConditionalDensityDecoder(nn.Module):
    r"""从 owner 零阶 latent 与共享 query feature 解码全部 Gaussian 带宽。"""

    def __init__(self, config: ImplicitFieldDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Linear(config.query_width, config.hidden_width)  # query relation -> decoder width
        self.blocks = nn.ModuleList(
            _FiLMResidualBlock(config.hidden_width, config.zero_order_width)
            for _ in range(config.residual_blocks)
        )
        self.output = nn.Linear(config.hidden_width, config.bandwidth_count)  # 每个 query 的 $L$ 个通道

    def forward(self, zero_order: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        r"""预测 `[B,G,N_Q,L]` 逐 owner 多带宽密度。

        `query_features` 不含 query stratum；相同物理 query 的预测函数不因采样来源标签改变。
        """

        if zero_order.ndim != 3 or query_features.ndim != 4:
            raise ValueError("zero_order/query_features must have shapes [B,G,D_0] and [B,G,N_Q,D_q]")
        if zero_order.shape[:2] != query_features.shape[:2]:
            raise ValueError("zero_order and query_features must share [B,G] axes")
        if zero_order.shape[-1] != self.config.zero_order_width:
            raise ValueError("zero_order width does not match decoder config")
        if query_features.shape[-1] != self.config.query_width:
            raise ValueError("query feature width does not match decoder config")

        hidden = self.query_projection(query_features)  # `[B,G,N_Q,D_h]`
        condition = zero_order.unsqueeze(2).expand(-1, -1, query_features.shape[2], -1)  # owner latent 广播到 query 轴
        for block in self.blocks:
            hidden = block(hidden, condition)  # 每层持续读取 owner latent，避免只在入口轻触几何记忆
        return torch.sigmoid(self.output(hidden))  # $\hat\rho\in(0,1)$；latent 本身不做 sigmoid


class DistanceSensitivityDecoder(nn.Module):
    r"""在 sampled owner–query–JOINT edges 上结构性读取符号奇 $\hat\kappa$。

    系数只读取 joint-sign 偶的 owner zero-order latent 与 query feature；最终与对应
    $z_i^{(1)}$ 做无偏置线性内积。因此 $z_i^{(1)}\mapsto-z_i^{(1)}$ 时输出严格翻号。
    """

    def __init__(self, config: ImplicitFieldDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.coefficient = nn.Sequential(
            nn.Linear(config.zero_order_width + config.query_width, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, config.first_order_width),
        )  # sign 偶上下文 -> 对一阶 carrier 的读取系数

    def forward(
        self,
        zero_order: torch.Tensor,
        first_order: torch.Tensor,
        query_features: torch.Tensor,
        owner_index: torch.Tensor,
        query_index: torch.Tensor,
        joint_index: torch.Tensor,
    ) -> torch.Tensor:
        r"""输出形状 `[B,E]`、单位由监督校准为 m/rad 的距离灵敏度。"""

        edge_count = owner_index.numel()  # sampled edge 数 $E$
        if query_index.shape != (edge_count,) or joint_index.shape != (edge_count,):
            raise ValueError("owner_index/query_index/joint_index must share shape [E]")
        owner_latent = zero_order.index_select(1, owner_index)  # `[B,E,D_0]`
        joint_latent = first_order.index_select(1, joint_index)  # `[B,E,D_1]`，joint-sign 奇
        selected_query = query_features[:, owner_index, query_index]  # `[B,E,D_q]`
        coefficient = self.coefficient(torch.cat((owner_latent, selected_query), dim=-1))  # `[B,E,D_1]` 偶
        return torch.sum(coefficient * joint_latent, dim=-1) / math.sqrt(self.config.first_order_width)  # 偶·奇=奇


__all__ = [
    "ConditionalDensityDecoder",
    "DistanceSensitivityDecoder",
    "ImplicitFieldDecoderConfig",
]
