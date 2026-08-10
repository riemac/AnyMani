r"""带运动学图软偏置的 Pre-LN 整手 Transformer。

同一次前向只处理一个结构模式，因此实体数 $N_E$ 固定且不使用 padding。注意力在
PALM/JOINT/TIP 实体轴保持全连接；最短路径、parent 方向和 child 方向只作为每层每头的
可学习加性偏置，不形成硬图 mask，从而保留跨手指通信。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _GraphTransformerLayer(nn.Module):
    r"""一个 Pre-LN 图偏置自注意力层。"""

    def __init__(
        self,
        hidden_width: int,
        attention_heads: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_width % attention_heads != 0:
            raise ValueError("hidden_width must be divisible by attention_heads")
        self.attention_heads = attention_heads  # 注意力头数 $H$
        self.head_width = hidden_width // attention_heads  # 每头宽度 $D_h=D/H$
        self.attention_norm = nn.LayerNorm(hidden_width)  # Pre-LN attention 输入规范化
        self.qkv = nn.Linear(hidden_width, 3 * hidden_width)  # 共享实体输入到 Q/K/V
        self.attention_output = nn.Linear(hidden_width, hidden_width)  # 多头拼接后的输出投影
        self.attention_dropout = nn.Dropout(dropout)  # dropout=0 时为 deterministic geometry contract
        self.feedforward_norm = nn.LayerNorm(hidden_width)  # Pre-LN FFN 输入规范化
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, hidden_width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, graph_bias: torch.Tensor) -> torch.Tensor:
        r"""执行全连接实体注意力。

        Args:
            tokens (torch.Tensor): 实体表征，形状 `[B,N_E,D]`。
            graph_bias (torch.Tensor): 每头结构偏置，形状 `[H,N_E,N_E]`。

        Returns:
            torch.Tensor: 更新后的实体表征，形状 `[B,N_E,D]`。
        """

        batch_size, entity_count, hidden_width = tokens.shape  # 当前同构 microbatch 的三条主轴
        normalized = self.attention_norm(tokens)  # Pre-LN，不改变 `[B,N_E,D]`
        qkv = self.qkv(normalized).reshape(
            batch_size, entity_count, 3, self.attention_heads, self.head_width
        )  # `[B,N_E,3,H,D_h]`
        query, key, value = qkv.unbind(dim=2)  # 每项 `[B,N_E,H,D_h]`
        query = query.transpose(1, 2)  # `[B,H,N_E,D_h]`
        key = key.transpose(1, 2)  # `[B,H,N_E,D_h]`
        value = value.transpose(1, 2)  # `[B,H,N_E,D_h]`

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)  # 缩放点积 `[B,H,N_E,N_E]`
        logits = logits + graph_bias.unsqueeze(0)  # soft graph bias，不屏蔽任何跨实体边
        weights = torch.softmax(logits, dim=-1)  # 每个 query entity 沿全部 key entities 归一化
        attended = torch.matmul(weights, value)  # `[B,H,N_E,D_h]` 全手上下文
        attended = attended.transpose(1, 2).reshape(batch_size, entity_count, hidden_width)  # 合并多头
        tokens = tokens + self.attention_dropout(self.attention_output(attended))  # attention residual

        normalized = self.feedforward_norm(tokens)  # 第二个 Pre-LN
        return tokens + self.feedforward(normalized)  # token-wise FFN residual


class GraphBiasedTransformer(nn.Module):
    r"""用离散运动学关系软调制的整手实体主干。"""

    def __init__(
        self,
        *,
        hidden_width: int,
        layers: int,
        attention_heads: int,
        feedforward_width: int,
        dropout: float,
        max_graph_distance: int,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("GraphBiasedTransformer requires at least one layer")
        self.max_graph_distance = max_graph_distance  # 超过该值的图距离统一落入末桶
        bucket_count = max_graph_distance + 1  # 包含 0/self 与截断后的最远桶
        self.shortest_path_bias = nn.Embedding(bucket_count, attention_heads)  # 无向最短路径偏置
        self.parent_direction_bias = nn.Embedding(bucket_count, attention_heads)  # parent 方向距离偏置
        self.child_direction_bias = nn.Embedding(bucket_count, attention_heads)  # child 方向距离偏置
        self.layers = nn.ModuleList(
            _GraphTransformerLayer(hidden_width, attention_heads, feedforward_width, dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_width)  # 所有 residual blocks 后的最终规范化

    def _graph_bias(
        self,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
    ) -> torch.Tensor:
        r"""把三种 `[N_E,N_E]` 图距离查表并相加成 `[H,N_E,N_E]`。"""

        matrices = (shortest_path, parent_direction, child_direction)
        if any(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] for matrix in matrices):
            raise ValueError("graph relation matrices must all have square shape [N_E,N_E]")
        if parent_direction.shape != shortest_path.shape or child_direction.shape != shortest_path.shape:
            raise ValueError("all graph relation matrices must have identical shape")

        shortest = shortest_path.clamp(min=0, max=self.max_graph_distance)  # 截断无向距离桶
        parent = parent_direction.clamp(min=0, max=self.max_graph_distance)  # 截断 parent 方向距离桶
        child = child_direction.clamp(min=0, max=self.max_graph_distance)  # 截断 child 方向距离桶
        bias = (
            self.shortest_path_bias(shortest)
            + self.parent_direction_bias(parent)
            + self.child_direction_bias(child)
        )  # `[N_E,N_E,H]`
        return bias.permute(2, 0, 1).contiguous()  # `[H,N_E,N_E]`

    def forward(
        self,
        tokens: torch.Tensor,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
    ) -> torch.Tensor:
        r"""上下文化同结构组的 PALM/JOINT/TIP 实体序列。"""

        graph_bias = self._graph_bias(shortest_path, parent_direction, child_direction)  # 结构模式共享静态偏置
        for layer in self.layers:
            tokens = layer(tokens, graph_bias)  # 全连接注意力；无 padding mask
        return self.final_norm(tokens)  # `[B,N_E,D]`


__all__ = ["GraphBiasedTransformer"]
