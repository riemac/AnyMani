"""Graph-biased Transformer over PALM, JOINT, and TIP tokens. Attention remains fully connected; graph distances add a per-head bias. Final tokens have shape [B,G,128]."""


from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBiasedTransformerCfg:


    hidden_width: int = 128
    layers: int = 4
    attention_heads: int = 4
    feedforward_width: int = 256
    dropout: float = 0.0
    max_graph_distance: int = 8

    def __post_init__(self) -> None:


        widths = (self.hidden_width, self.layers, self.attention_heads, self.feedforward_width)
        if any(value < 1 for value in widths) or self.max_graph_distance < 1:
            raise ValueError("graph-biased transformer widths/layers/distance must be positive")
        if self.hidden_width % self.attention_heads:
            raise ValueError("hidden_width must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


class _GraphTransformerLayer(nn.Module):


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
        self.attention_heads = attention_heads
        self.head_width = hidden_width // attention_heads
        self.attention_norm = nn.LayerNorm(hidden_width)
        self.qkv = nn.Linear(hidden_width, 3 * hidden_width)
        self.attention_output = nn.Linear(hidden_width, hidden_width)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_width)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, hidden_width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        graph_bias: torch.Tensor,
        entity_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        batch_size, entity_count, hidden_width = tokens.shape
        normalized = self.attention_norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch_size, entity_count, 3, self.attention_heads, self.head_width
        )  # `[B,N_E,3,H,D_h]`
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)  # `[B,H,N_E,D_h]`
        key = key.transpose(1, 2)  # `[B,H,N_E,D_h]`
        value = value.transpose(1, 2)  # `[B,H,N_E,D_h]`

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)  # `[B,H,N_E,N_E]`
        logits = logits + (graph_bias.unsqueeze(0) if graph_bias.ndim == 3 else graph_bias)
        if entity_valid_mask is not None:
            if entity_valid_mask.shape != (batch_size, entity_count) or entity_valid_mask.dtype != torch.bool:
                raise ValueError("entity_valid_mask must have bool shape [B,N_E]")
            logits = logits.masked_fill(
                ~entity_valid_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )
        weights = torch.softmax(logits, dim=-1)
        if entity_valid_mask is not None:
            weights = weights * entity_valid_mask[:, None, :, None]
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, entity_count, hidden_width)
        tokens = tokens + self.attention_dropout(self.attention_output(attended))
        if entity_valid_mask is not None:
            tokens = tokens * entity_valid_mask.unsqueeze(-1)

        normalized = self.feedforward_norm(tokens)
        tokens = tokens + self.feedforward(normalized)
        return tokens if entity_valid_mask is None else tokens * entity_valid_mask.unsqueeze(-1)


class GraphBiasedTransformer(nn.Module):


    def __init__(
        self,
        config: GraphBiasedTransformerCfg,
    ) -> None:


        super().__init__()
        self.config = config
        self.max_graph_distance = config.max_graph_distance
        bucket_count = config.max_graph_distance + 1
        self.shortest_path_bias = nn.Embedding(bucket_count, config.attention_heads)
        self.parent_direction_bias = nn.Embedding(bucket_count, config.attention_heads)
        self.child_direction_bias = nn.Embedding(bucket_count, config.attention_heads)
        self.layers = nn.ModuleList(
            _GraphTransformerLayer(
                config.hidden_width,
                config.attention_heads,
                config.feedforward_width,
                config.dropout,
            )
            for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_width)

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
        bucket_count = self.max_graph_distance + 1


        def lookup(index: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
            one_hot = torch.nn.functional.one_hot(index, num_classes=bucket_count).to(
                dtype=embedding.weight.dtype
            )
            return one_hot @ embedding.weight

        bias = (
            lookup(shortest, self.shortest_path_bias)
            + lookup(parent, self.parent_direction_bias)
            + lookup(child, self.child_direction_bias)
        )
        if bias.ndim == 3:
            return bias.permute(2, 0, 1).contiguous()  # `[H,N_E,N_E]`
        return bias.permute(0, 3, 1, 2).contiguous()  # `[B,H,N_E,N_E]`

    def forward(
        self,
        tokens: torch.Tensor,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
        entity_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        graph_bias = self._graph_bias(shortest_path, parent_direction, child_direction)
        if entity_valid_mask is not None:
            tokens = tokens * entity_valid_mask.unsqueeze(-1)
        for layer in self.layers:
            tokens = layer(tokens, graph_bias, entity_valid_mask)
        tokens = self.final_norm(tokens)  # `[B,N_E,D]`
        return tokens if entity_valid_mask is None else tokens * entity_valid_mask.unsqueeze(-1)


__all__ = ["GraphBiasedTransformer", "GraphBiasedTransformerCfg"]
