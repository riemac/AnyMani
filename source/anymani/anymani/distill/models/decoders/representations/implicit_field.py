r"""多带宽密度与 sampled-edge 距离灵敏度解码器。

设 $z_g^{(0)}\in\mathbb R^{D_0}$ 是归属体 $g$ 的关节符号偶零阶表征，
$z_i^{(1)}\in\mathbb R^{D_1}$ 是 JOINT $i$ 的符号奇一阶表征，
$u(x)=\Psi_C(x)\in\mathbb R^{D_q}$ 是查询点相对完整 anchor 星座的共享特征。
密度解码器在每个 query 上同时输出全部 $L$ 个 Gaussian 带宽：

$$
D_\rho\left(z_g^{(0)},u(x)\right)
\longrightarrow
\hat{\boldsymbol\rho}_g(x;q)
\in(0,1)^L.
$$

距离灵敏度解码器只在 sampled owner--query--JOINT edges 上读取：

$$
D_\kappa\left(z_g^{(0)},z_i^{(1)},u(x),i\right)
\longrightarrow
\hat\kappa_{g,i}(x;q)
\quad[\mathrm{m/rad}].
$$

密度解码器使用 FiLM（特征线性调制）残差块，使归属体表征在每层持续调制查询特征，
而不是只在入口拼接一次。$D_\kappa$ 的最终读取对 $z_i^{(1)}$ 结构性线性且无偏置，
因此在成对关节符号重写下严格满足 $\hat\kappa'_{g,i}=s_i\hat\kappa_{g,i}$。

解码器只在 SSL 期间存在。查询点来源、最近点、距离标签、Jacobian 与场标签都不进入
解码器输入；SSL 完成后整个模块删除，不迁入 PPO。

张量轴约定：

```text
zero_order       : [B, G, D_0]
first_order      : [B, N_J, D_1]
query_features   : [B, G, N_Q, D_q]
density          : [B, G, N_Q, L]
sampled kappa    : [B, E]
```

$E$ 是抽样的归属体—查询点—JOINT 边数。密度路径保留完整归属体与查询点轴；灵敏度路径通过
`owner_index/query_index/joint_index` 只读取需要监督的边，避免实际生成
``[B,G,N_Q,N_J]`` 大张量。所有查询点始终同时输出全部 $L$ 个带宽，采样来源不分配独占带宽。

FiLM 条件来自同一归属体的 $z_g^{(0)}$，不会混入其他归属体的标签；跨归属体信息已经由
保留的整手 Transformer 写入 $z_g^{(0)}$。查询特征来自与基准表面共享参数的点—锚点前端，
因此模型不能通过另一套查询编码器形成训练期专用坐标捷径。

密度输出采用 sigmoid 是因为物理邻近场满足 $\rho\in(0,1]$。精确表面真值可等于 1，而有限
logit 只能逼近 1；因此精确 $d=0$ 点主要用于评估，训练查询点以近表面壳层和工作空间为主。
这一数值边界不意味着对潜变量使用 sigmoid；特别是一阶表征必须允许跨过零点。

距离灵敏度解码器不直接读取原始三维轴线。轴线符号已经在保留前端中压缩为
$z_i^{(1)}$，策略迁移时也只保留该类型化结果。解析最近点、表面点 Jacobian 与
$\kappa/g$ 教师只参与损失，不参与普通前向。

NOTE: 低重建误差不能单独证明零阶表征被使用。正式训练必须同时运行仅查询点基线与批内
表征打乱诊断，确认解码器没有绕过整手几何记忆。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImplicitFieldDecoderConfig:
    r"""训练期场解码器的类型宽度与容量。

    Attributes:
        zero_order_width (int): 归属体零阶表征宽度 $D_0$。
        first_order_width (int): JOINT 一阶表征宽度 $D_1$。
        query_width (int): 共享点—锚点查询特征宽度 $D_q$。
        hidden_width (int): FiLM/灵敏度解码器的中间宽度。
        bandwidth_count (int): 同时输出的 Gaussian 带宽数 $L$。
        residual_blocks (int): 密度解码器中连续 FiLM 残差块数量。

    数值默认 ``hidden_width=128``、``bandwidth_count=4``、``residual_blocks=3`` 是首个
    可运行容量锚点，不是已经由跨手型消融接受的最终容量。
    """

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
    r"""由归属体零阶表征调制的逐查询点残差块。

    对隐藏特征 $h$ 与归属体条件 $z_g^{(0)}$，调制形式为：

    $$
    \tilde h
    =
    \left(1+\gamma(z_g^{(0)})\right)\operatorname{LN}(h)
    +\beta(z_g^{(0)}).
    $$
    """

    def __init__(self, hidden_width: int, condition_width: int) -> None:
        r"""构造一次归一化、条件调制与残差更新。

        Args:
            hidden_width (int): 每个查询点的隐藏宽度。
            condition_width (int): 归属体零阶表征宽度 $D_0$。

        `modulation` 同时输出缩放与平移，故宽度为 ``2 * hidden_width``；缩放使用
        $1+\gamma$，使参数初始化附近保留接近恒等的调制路径。
        """

        super().__init__()
        self.normalization = nn.LayerNorm(hidden_width)  # 每个查询点独立规范化隐藏通道
        self.modulation = nn.Linear(condition_width, 2 * hidden_width)  # 归属体表征 -> $(\gamma,\beta)$
        self.update = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        r"""对 `[B,G,N_Q,D]` 查询特征施加归属体条件 FiLM。"""

        gamma, beta = self.modulation(condition).chunk(2, dim=-1)  # 每项 `[B,G,N_Q,D]`
        modulated = self.normalization(hidden) * (1.0 + gamma) + beta  # FiLM：$(1+\gamma)h+\beta$
        return hidden + self.update(modulated)  # 残差保留原查询几何证据


class ConditionalDensityDecoder(nn.Module):
    r"""从归属体零阶表征与共享查询特征解码全部 Gaussian 带宽。

    输入轴为 ``zero_order: [B,G,D_0]`` 和 ``query_features: [B,G,N_Q,D_q]``；输出为
    ``[B,G,N_Q,L]``。sigmoid 只约束邻近场范围 $(0,1)$，不约束 $Z^{(0)}$ 或
    $z_i^{(1)}$ 的符号与数值范围。
    """

    def __init__(self, config: ImplicitFieldDecoderConfig) -> None:
        r"""构造查询投影、连续 FiLM 残差块和多带宽输出层。

        Args:
            config (ImplicitFieldDecoderConfig): 输入类型宽度、隐藏宽度、带宽数与残差块数量。

        输出层宽度严格等于 $L$，与查询点数 $N_Q$ 无关；改变每批查询点数量不需要重建输出头。
        """

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

        `query_features` 不含查询点来源；相同物理查询点的预测函数不因采样来源标签改变。
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
        condition = zero_order.unsqueeze(2).expand(-1, -1, query_features.shape[2], -1)  # 归属体表征广播到查询轴
        for block in self.blocks:
            hidden = block(hidden, condition)  # 每层持续读取归属体表征，避免只在入口轻触几何记忆
        return torch.sigmoid(self.output(hidden))  # $\hat\rho\in(0,1)$；latent 本身不做 sigmoid


class DistanceSensitivityDecoder(nn.Module):
    r"""在抽样的归属体—查询点—JOINT 边上结构性读取符号奇 $\hat\kappa$。

    系数只读取关节符号偶的归属体零阶表征与查询特征；最终与对应
    $z_i^{(1)}$ 做无偏置线性内积。因此 $z_i^{(1)}\mapsto-z_i^{(1)}$ 时输出严格翻号：

    $$
    \hat\kappa_{g,i}
    =
    \frac{1}{\sqrt{D_1}}
    a\left(z_g^{(0)},u(x)\right)^Tz_i^{(1)}.
    $$

    $1/\sqrt{D_1}$ 只稳定不同一阶宽度下的初始点积尺度，不改变物理单位；m/rad 语义来自
    解析 $\kappa$ 教师监督。
    """

    def __init__(self, config: ImplicitFieldDecoderConfig) -> None:
        r"""构造只产生符号偶读取系数的灵敏度网络。

        Args:
            config (ImplicitFieldDecoderConfig): 零阶、一阶、查询与隐藏宽度。

        本构造函数不创建带偏置的一阶输出层。最终标量只能由偶系数与 $z_i^{(1)}$ 内积产生，
        从结构上排除“偏置项在符号翻转后保持不变”的奇偶违约。
        """

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

        if owner_index.ndim not in {1, 2} or query_index.shape != owner_index.shape or joint_index.shape != owner_index.shape:
            raise ValueError("owner/query/joint selectors must share [E] or [B,E] shape")
        if owner_index.ndim == 1:
            owner_latent = zero_order.index_select(1, owner_index)  # `[B,E,D_0]`
            joint_latent = first_order.index_select(1, joint_index)  # `[B,E,D_1]`，joint-sign 奇
            selected_query = query_features[:, owner_index, query_index]  # `[B,E,D_q]`
        else:
            if owner_index.shape[0] != zero_order.shape[0]:
                raise ValueError("batched selectors must share B with latent tensors")
            owner_gather = owner_index.unsqueeze(-1).expand(-1, -1, zero_order.shape[-1])
            joint_gather = joint_index.unsqueeze(-1).expand(-1, -1, first_order.shape[-1])
            owner_latent = torch.gather(zero_order, 1, owner_gather)  # 每个样本自己的 owner edge
            joint_latent = torch.gather(first_order, 1, joint_gather)  # 每个样本自己的 JOINT edge
            batch_index = torch.arange(zero_order.shape[0], device=zero_order.device).unsqueeze(1)
            selected_query = query_features[batch_index, owner_index, query_index]  # `[B,E,D_q]`
        coefficient = self.coefficient(torch.cat((owner_latent, selected_query), dim=-1))  # `[B,E,D_1]` 偶
        return torch.sum(coefficient * joint_latent, dim=-1) / math.sqrt(self.config.first_order_width)  # 偶·奇=奇


__all__ = [
    "ConditionalDensityDecoder",
    "DistanceSensitivityDecoder",
    "ImplicitFieldDecoderConfig",
]
