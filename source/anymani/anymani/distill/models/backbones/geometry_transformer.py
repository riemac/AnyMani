r"""带运动学图软偏置的 Pre-LN 整手 Transformer。

同一次前向只处理一个结构模式，因此实体数 $N_E$ 固定且不使用 padding。设第 $h$ 个
注意力头的实体输入为 $H\in\mathbb R^{B\times N_E\times D}$，图关系查表得到
$b_{ij}^{(h)}$，则注意力权重为：

$$
\alpha_{ij}^{(h)}
=
\operatorname{softmax}_j\left(
\frac{q_i^{(h)T}k_j^{(h)}}{\sqrt{D_h}}
+b_{ij}^{(h)}
\right).
$$

注意力在 PALM/JOINT/TIP 实体轴保持全连接；无向最短路径、parent 方向和 child 方向只
提供可学习加性偏置，不形成硬图掩码，从而允许不同手指根据当前几何表征直接通信。
前置 LayerNorm 和残差结构分别应用于注意力与前馈子层，最终再做一次 LayerNorm。

图关系矩阵的物理语义：

```text
shortest_path[i,j]     # 实体 i 与 j 的无向运动学最短距离
parent_direction[i,j]  # 从 i 沿父方向到 j 的有向距离；不可达时进入末桶
child_direction[i,j]   # 从 i 沿子方向到 j 的有向距离；不可达时进入末桶
```

图偏置只说明“哪些实体在运动学上相邻、方向如何”，不提供当前刚体位姿、当前表面点或动态
全对全变换。当前构型信息必须由 $q$ 与静态旋量/基准表面经过保留编码器形成，避免把监督答案
直接塞进主干。

同结构资产可以共享真实 $N_E$ 轴；跨结构批处理可显式填充到配置上限，并必须提供
``entity_valid_mask``。填充 token 没有独立 embedding，不能作为 key/value 被有效实体读取，且每层后
重新清零。无填充前向保留为 oracle，要求有效 token 输出与梯度逐元素等价。

PALM 与 TIP 虽不直接输出动作，仍保留独立实体表征并参加全连接注意力。这样 JOINT 可读取
手掌尺度、挂载布局、其他手指与末端形状；不能在进入主干前把整手无条件池化为单个向量，
否则逐归属体来源和一阶路由将不可恢复。

数值实现使用显式 Q/K/V 和加性偏置，而不是调用会隐藏掩码广播规则的高层封装。该选择便于
逐行核对 ``[B,H,N_E,N_E]`` 分数、图偏置和 softmax 轴，也允许性能测试准确测量实体主干成本。

NOTE: 图偏置是软先验而不是物理约束。非祖先灵敏度精确为零的合同由拓扑教师、解码器与损失
路径验证，不能仅因为图距离较远就假定注意力自动为零。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBiasedTransformerCfg:
    r"""全连接整手 Transformer 的容量与离散运动学图偏置配置。

    canonical 数值锚点为 $D=128$、4 个 Pre-LN blocks、4 heads、FFN width 256、dropout 0，
    图距离统一截断到 8。层数和宽度是待消融容量，不是物理常数；最短路径、parent 与 child
    三种 bias 则属于当前 backbone 的结构定义。
    """

    hidden_width: int = 128  # 实体 token 宽度 $D$
    layers: int = 4  # 四次关系加工；每层保持 Pre-LN attention/FFN residual 结构
    attention_heads: int = 4  # 多头数 $H$，每头宽度 $D_h=D/H$
    feedforward_width: int = 256  # 每层逐实体 FFN 中间宽度
    dropout: float = 0.0  # retained geometry contract 默认确定性前向
    max_graph_distance: int = 8  # 最短/父/子关系的末距离桶

    def __post_init__(self) -> None:
        r"""验证 attention 分头、残差深度与图桶均可形成合法网络。"""

        widths = (self.hidden_width, self.layers, self.attention_heads, self.feedforward_width)
        if any(value < 1 for value in widths) or self.max_graph_distance < 1:  # 空主干/空桶无物理意义
            raise ValueError("graph-biased transformer widths/layers/distance must be positive")
        if self.hidden_width % self.attention_heads:  # $D_h$ 必须为整数
            raise ValueError("hidden_width must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:  # PyTorch Dropout 的概率域
            raise ValueError("dropout must lie in [0,1)")


class _GraphTransformerLayer(nn.Module):
    r"""一个前置 LayerNorm 图偏置自注意力层。

    该内部层不拥有图关系查表，只消费已经形成的 ``[H,N_E,N_E]`` 加性偏置。这样同一结构模式
    的静态图偏置可以供整个批次复用，而不复制到 $B$ 轴。
    """

    def __init__(
        self,
        hidden_width: int,
        attention_heads: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        r"""构造单层多头注意力与逐实体前馈网络。

        Args:
            hidden_width (int): 实体隐藏宽度 $D$。
            attention_heads (int): 注意力头数 $H$。
            feedforward_width (int): 前馈中间宽度。
            dropout (float): 残差支路随机失活概率。

        Raises:
            ValueError: 当 $D$ 不能被 $H$ 整除时抛出。
        """

        super().__init__()
        if hidden_width % attention_heads != 0:
            raise ValueError("hidden_width must be divisible by attention_heads")
        self.attention_heads = attention_heads  # 注意力头数 $H$
        self.head_width = hidden_width // attention_heads  # 每头宽度 $D_h=D/H$
        self.attention_norm = nn.LayerNorm(hidden_width)  # 注意力子层的前置 LayerNorm
        self.qkv = nn.Linear(hidden_width, 3 * hidden_width)  # 共享实体输入到 Q/K/V
        self.attention_output = nn.Linear(hidden_width, hidden_width)  # 多头拼接后的输出投影
        self.attention_dropout = nn.Dropout(dropout)  # 随机失活为 0 时满足确定性几何合同
        self.feedforward_norm = nn.LayerNorm(hidden_width)  # 前馈子层的前置 LayerNorm
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
        r"""执行全连接实体注意力。

        Args:
            tokens (torch.Tensor): 实体表征，形状 `[B,N_E,D]`。
            graph_bias (torch.Tensor): 每头结构偏置，形状 `[H,N_E,N_E]` 或 `[B,H,N_E,N_E]`。
            entity_valid_mask (torch.Tensor | None): 可选 `[B,N_E]` 有效实体掩码。

        Returns:
            torch.Tensor: 更新后的实体表征，形状 `[B,N_E,D]`。
        """

        batch_size, entity_count, hidden_width = tokens.shape  # 当前同构微批次的三条主轴
        normalized = self.attention_norm(tokens)  # 前置 LayerNorm，不改变 `[B,N_E,D]`
        qkv = self.qkv(normalized).reshape(
            batch_size, entity_count, 3, self.attention_heads, self.head_width
        )  # `[B,N_E,3,H,D_h]`
        query, key, value = qkv.unbind(dim=2)  # 每项 `[B,N_E,H,D_h]`
        query = query.transpose(1, 2)  # `[B,H,N_E,D_h]`
        key = key.transpose(1, 2)  # `[B,H,N_E,D_h]`
        value = value.transpose(1, 2)  # `[B,H,N_E,D_h]`

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)  # `[B,H,N_E,N_E]`
        logits = logits + (graph_bias.unsqueeze(0) if graph_bias.ndim == 3 else graph_bias)  # 逐结构图偏置
        if entity_valid_mask is not None:
            if entity_valid_mask.shape != (batch_size, entity_count) or entity_valid_mask.dtype != torch.bool:
                raise ValueError("entity_valid_mask must have bool shape [B,N_E]")
            logits = logits.masked_fill(
                ~entity_valid_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )  # 有效 query 不得读取 padding key/value
        weights = torch.softmax(logits, dim=-1)  # 每个 query entity 沿全部 key entities 归一化
        if entity_valid_mask is not None:
            weights = weights * entity_valid_mask[:, None, :, None]  # padding query 行严格为零
        attended = torch.matmul(weights, value)  # `[B,H,N_E,D_h]` 全手上下文
        attended = attended.transpose(1, 2).reshape(batch_size, entity_count, hidden_width)  # 合并多头
        tokens = tokens + self.attention_dropout(self.attention_output(attended))  # 注意力残差
        if entity_valid_mask is not None:
            tokens = tokens * entity_valid_mask.unsqueeze(-1)  # 消除 projection bias 写入 padding token

        normalized = self.feedforward_norm(tokens)  # 第二个前置 LayerNorm
        tokens = tokens + self.feedforward(normalized)  # 逐实体前馈残差
        return tokens if entity_valid_mask is None else tokens * entity_valid_mask.unsqueeze(-1)


class GraphBiasedTransformer(nn.Module):
    r"""用离散运动学关系软调制的整手实体主干。

    `shortest_path` 表示无向运动学距离；`parent_direction` 与 `child_direction` 分别保留有向
    祖先/后代关系。三类关系各自查表后相加，使网络能区分“拓扑上同样相距一跳，但方向不同”
    的实体对。超过 ``max_graph_distance`` 的距离统一进入末桶，避免手指数或链长变化导致
    嵌入表无限增长。
    """

    def __init__(
        self,
        config: GraphBiasedTransformerCfg,
    ) -> None:
        r"""构造图关系嵌入与若干整手自注意力层。

        Args:
            config (GraphBiasedTransformerCfg): 实体宽度、层数、多头、FFN、dropout 与图桶。

        每种图关系为每个注意力头维护独立标量嵌入；参数量为
        $3(max\_distance+1)H$，不随实体数或手指数增长。
        """

        super().__init__()
        self.config = config  # checkpoint 外由 resolved experiment 保存完整 architecture contract
        self.max_graph_distance = config.max_graph_distance  # 超过该值的图距离统一落入末桶
        bucket_count = config.max_graph_distance + 1  # 包含 0/self 与截断后的最远桶
        self.shortest_path_bias = nn.Embedding(bucket_count, config.attention_heads)  # 无向最短路径偏置
        self.parent_direction_bias = nn.Embedding(bucket_count, config.attention_heads)  # parent 方向距离偏置
        self.child_direction_bias = nn.Embedding(bucket_count, config.attention_heads)  # child 方向距离偏置
        self.layers = nn.ModuleList(
            _GraphTransformerLayer(
                config.hidden_width,
                config.attention_heads,
                config.feedforward_width,
                config.dropout,
            )
            for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_width)  # 所有残差块后的最终规范化

    def _graph_bias(
        self,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
    ) -> torch.Tensor:
        r"""把三种图距离查表并相加成 `[H,N_E,N_E]` 或 `[B,H,N_E,N_E]`。

        查表结果只依赖静态结构模式，不依赖当前 $q$；当前几何变化由实体表征承担，避免把
        动态全对全 $SE(3)$ 答案直接泄漏进隐式路线。
        """

        matrices = (shortest_path, parent_direction, child_direction)
        if any(matrix.ndim not in {2, 3} or matrix.shape[-2] != matrix.shape[-1] for matrix in matrices):
            raise ValueError("graph relation matrices must have square shape [N_E,N_E] or [B,N_E,N_E]")
        if parent_direction.shape != shortest_path.shape or child_direction.shape != shortest_path.shape:
            raise ValueError("all graph relation matrices must have identical shape")

        shortest = shortest_path.clamp(min=0, max=self.max_graph_distance)  # 截断无向距离桶
        parent = parent_direction.clamp(min=0, max=self.max_graph_distance)  # 截断 parent 方向距离桶
        child = child_direction.clamp(min=0, max=self.max_graph_distance)  # 截断 child 方向距离桶
        bias = (
            self.shortest_path_bias(shortest)
            + self.parent_direction_bias(parent)
            + self.child_direction_bias(child)
        )  # `[N_E,N_E,H]` 或 `[B,N_E,N_E,H]`
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
        r"""上下文化同结构组的 PALM/JOINT/TIP 实体序列。

        Args:
            tokens (torch.Tensor): ``[B,N_E,D]`` 的实体输入。
            shortest_path (torch.Tensor): ``[N_E,N_E]`` 无向距离桶。
            parent_direction (torch.Tensor): ``[N_E,N_E]`` 父方向距离桶。
            child_direction (torch.Tensor): ``[N_E,N_E]`` 或 ``[B,N_E,N_E]`` 子方向距离桶。
            entity_valid_mask (torch.Tensor | None): 跨结构 padding 时的 `[B,N_E]` 掩码。

        Returns:
            torch.Tensor: ``[B,N_E,D]`` 的全手上下文化表征。
        """

        graph_bias = self._graph_bias(shortest_path, parent_direction, child_direction)  # 结构模式共享静态偏置
        if entity_valid_mask is not None:
            tokens = tokens * entity_valid_mask.unsqueeze(-1)  # 输入 padding token 不携带任意数值
        for layer in self.layers:
            tokens = layer(tokens, graph_bias, entity_valid_mask)  # 有效实体之间保持全连接
        tokens = self.final_norm(tokens)  # `[B,N_E,D]`
        return tokens if entity_valid_mask is None else tokens * entity_valid_mask.unsqueeze(-1)


__all__ = ["GraphBiasedTransformer", "GraphBiasedTransformerCfg"]
