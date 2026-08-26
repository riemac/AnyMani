r"""统一 entity 表征上的显式 sigma 密度与 sampled-edge 距离灵敏度解码器。

设 $z_g,z_i\in\mathbb R^D$ 分别是统一 $Z$ 中的 owner token 与 JOINT token，
$u(x)=\Psi_C(x)\in\mathbb R^{D_q}$ 是查询点相对完整 anchor 星座的共享特征。
密度解码器对每个显式 `(query,sigma)` 条件输出一个 scalar：

$$
D_\rho\left(z_g,u(x),\log\frac{\sigma}{\sigma_{ref}}\right)
\longrightarrow
\hat\rho_{g,\sigma}(x;q)
\in(0,1).
$$

距离灵敏度解码器只在 sampled owner--query--JOINT edges 上读取：

$$
D_\kappa\left(z_g,z_i,u(x)\right)
\longrightarrow
\hat\kappa_{g,i}(x;q)
\quad[\mathrm{m/rad}].
$$

两个解码器都使用 FiLM（特征线性调制）残差块，使 retained token 在每层持续调制查询主路径。
$D_\kappa$ 每层由拼接条件 $[z_g\Vert z_i]$ 独立生成 $\gamma,\beta$，最后用带偏置线性层输出
无界 signed scalar。joint-sign 是完整物理坐标改写下的可观测测试合同，不由 latent parity 硬编码。

解码器只在 SSL 期间存在。查询点来源、最近点、距离标签、Jacobian 与场标签都不进入
解码器输入；SSL 完成后整个模块删除，不迁入 PPO。

张量轴约定：

```text
entities         : [B, G, D]
query_features   : [B, G, N_Q, D_q]
density          : [B, G, N_Q, N_sigma]
sampled kappa    : [B, E]
```

$E$ 是抽样的归属体—查询点—JOINT 边数。密度路径保留完整归属体与查询点轴；灵敏度路径通过
`owner_index/query_index/joint_index` 只读取需要监督的边，避免实际生成
``[B,G,N_Q,N_J]`` 大张量。$N_Q$ 与 $N_\sigma$ 都是数据轴，不进入网络固定宽度。

密度 FiLM 条件来自同一归属体的 $z_g$，不会混入其他归属体的标签；跨归属体信息已经由
保留的整手 Transformer 写入 $z_g$。查询特征来自与基准表面共享参数的点—锚点前端，
因此模型不能通过另一套查询编码器形成训练期专用坐标捷径。

密度输出采用 sigmoid 是因为物理邻近场满足 $\rho\in(0,1]$。精确表面真值可等于 1，而有限
logit 只能逼近 1；因此精确 $d=0$ 点主要用于评估，训练查询点以近表面壳层和工作空间为主。
这一数值边界不意味着对潜变量或 $\kappa$ 使用 sigmoid；二者都必须允许跨过零点。

距离灵敏度解码器不读取原始三维轴线、sigma 或 task one-hot。轴线与当前 $q$ 已在 Transformer
之前写入 JOINT token；解析最近点、表面点 Jacobian 与 $\kappa/g$ 教师只参与监督和诊断。

NOTE: 低重建误差不能单独证明零阶表征被使用。正式训练必须同时运行仅查询点基线与批内
表征打乱诊断，确认解码器没有绕过整手几何记忆。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScalarSigmaFiLMDensityDecoderCfg:
    r"""逐 `(owner,query,sigma)` 标量密度 reader 的 FiLM 容量与尺度条件。"""

    hidden_width: int = 128
    residual_blocks: int = 3
    sigma_reference_m: float = 0.016  # 16 mm；只定义无量纲 $\log(\sigma/\sigma_{ref})$

    def __post_init__(self) -> None:
        r"""验证 FiLM 主路径容量和参考带宽均严格为正。"""

        if self.hidden_width < 1 or self.residual_blocks < 1:
            raise ValueError("density decoder hidden width and residual blocks must be positive")
        if self.sigma_reference_m <= 0.0:
            raise ValueError("sigma_reference_m must be strictly positive")


@dataclass(frozen=True)
class DistanceSensitivityDecoderCfg:
    r"""query-main、owner/JOINT-conditioned 的距离灵敏度 FiLM reader 容量。"""

    hidden_width: int = 128  # 查询主路径与三个残差块的统一宽度
    residual_blocks: int = 3  # 每块独立从 $[z_o\Vert z_i]$ 生成 FiLM 参数

    def __post_init__(self) -> None:
        r"""拒绝空主路径或没有条件更新的退化 reader。"""

        if self.hidden_width < 1 or self.residual_blocks < 1:
            raise ValueError("sensitivity decoder hidden width and residual blocks must be positive")


@dataclass(frozen=True)
class GeometrySSLDecoderCfg:
    r"""聚合训练期密度与距离灵敏度 readers；整个配置不进入 retained checkpoint。"""

    density: ScalarSigmaFiLMDensityDecoderCfg = ScalarSigmaFiLMDensityDecoderCfg()
    sensitivity: DistanceSensitivityDecoderCfg = DistanceSensitivityDecoderCfg()


class _FiLMResidualBlock(nn.Module):
    r"""由归属体条件表征调制的逐查询点残差块。

    对隐藏特征 $h$ 与归属体条件 $z_g$，调制形式为：

    $$
    \tilde h
    =
    \left(1+\gamma(z_g)\right)\operatorname{LN}(h)
    +\beta(z_g).
    $$
    """

    def __init__(self, hidden_width: int, condition_width: int) -> None:
        r"""构造一次归一化、条件调制与残差更新。

        Args:
            hidden_width (int): 每个查询点的隐藏宽度。
            condition_width (int): owner 或 owner/JOINT 条件表征宽度。

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
    r"""从 query--sigma 主路径与归属体 FiLM 条件解码标量 Gaussian 密度。

    对每个 `(owner,query,sigma)`：

    $$
    D_\rho\left(z_g^{(0)}(q),u_{g,r}(x),\log\frac{\sigma}{\sigma_{ref}}\right)
    \longrightarrow \hat\rho_{g,r,\sigma}\in(0,1).
    $$

    ``query_features`` 与显式 sigma 进入主特征路径；``owner_latent`` 只作为每层 FiLM 条件。输出逻辑
    shape 仍为 ``[B,G,N_Q,N_sigma]``，但最后一轴是可变的采样轴，不是线性层固定输出宽度。
    """

    def __init__(
        self,
        config: ScalarSigmaFiLMDensityDecoderCfg,
        *,
        entity_width: int,
        query_width: int,
    ) -> None:
        r"""构造 query--sigma 投影、连续 owner FiLM 残差块和标量输出层。

        Args:
            config (ScalarSigmaFiLMDensityDecoderCfg): sigma reference、隐藏宽度与残差块数量。
            entity_width (int): encoder final-norm owner token 宽度 $D$。
            query_width (int): 共享点—anchor 前端派生的查询宽度 $D_q$。
        """

        super().__init__()
        self.config = config
        if entity_width < 1 or query_width < 1:
            raise ValueError("density decoder latent/query widths must be positive")
        self.entity_width = entity_width  # 派生宽度不在实验配置中重复声明
        self.query_width = query_width
        self.query_projection = nn.Linear(  # $[u(x)\Vert\log(\sigma/\sigma_{ref})]$ -> decoder width
            query_width + 1,
            config.hidden_width,
        )
        self.blocks = nn.ModuleList(
            _FiLMResidualBlock(config.hidden_width, entity_width)
            for _ in range(config.residual_blocks)
        )
        self.output = nn.Linear(config.hidden_width, 1)  # 每个 `(owner,query,sigma)` 只输出一个标量 $\rho$

    def forward(
        self,
        owner_latent: torch.Tensor,
        query_features: torch.Tensor,
        bandwidths: torch.Tensor,
    ) -> torch.Tensor:
        r"""向量化预测 ``[B,G,N_Q,N_sigma]`` 逐 owner 显式 sigma 密度。

        `query_features` 不含查询点来源；相同物理查询点的预测函数不因采样来源标签改变。
        ``bandwidths`` 接受跨 batch 共享的 ``[N_sigma]`` 或逐样本 ``[B,N_sigma]``，单位 m。
        """

        if owner_latent.ndim != 3 or query_features.ndim != 4:
            raise ValueError("owner_latent/query_features must have shapes [B,G,D] and [B,G,N_Q,D_q]")
        if owner_latent.shape[:2] != query_features.shape[:2]:
            raise ValueError("owner_latent and query_features must share [B,G] axes")
        if owner_latent.shape[-1] != self.entity_width:
            raise ValueError("owner latent width does not match decoder config")
        if query_features.shape[-1] != self.query_width:
            raise ValueError("query feature width does not match decoder config")
        bandwidths = bandwidths.detach()  # sigma 是外生物理条件，不建立优化或 q->sigma 梯度路径
        if bandwidths.ndim == 1:
            bandwidths = bandwidths.unsqueeze(0).expand(query_features.shape[0], -1)  # `[B,N_σ]`
        if bandwidths.ndim != 2 or bandwidths.shape[0] != query_features.shape[0] or torch.any(bandwidths <= 0.0):
            raise ValueError("bandwidths must have positive shape [N_sigma] or [B,N_sigma]")

        sigma_count = bandwidths.shape[1]  # $N_\sigma$ 是当前数据轴，可在不同调用中变化
        query = query_features.unsqueeze(3).expand(-1, -1, -1, sigma_count, -1)  # `[B,G,N_Q,N_σ,D_q]`
        log_sigma = torch.log(bandwidths / self.config.sigma_reference_m)  # `[B,N_σ]`，无量纲
        log_sigma = log_sigma[:, None, None, :, None].expand(  # 广播到每个 owner/query，不引入类别身份
            -1, query_features.shape[1], query_features.shape[2], -1, -1
        )
        hidden = self.query_projection(torch.cat((query, log_sigma), dim=-1))  # `[B,G,N_Q,N_σ,D_h]`
        condition = owner_latent[:, :, None, None, :].expand(  # 同一 owner latent 调制全部 query/sigma slots
            -1, -1, query_features.shape[2], sigma_count, -1
        )
        for block in self.blocks:
            hidden = block(hidden, condition)  # 每层持续读取归属体表征，避免只在入口轻触几何记忆
        return torch.sigmoid(self.output(hidden)).squeeze(-1)  # `[B,G,N_Q,N_σ]`，$\hat\rho\in(0,1)$


class DistanceSensitivityDecoder(nn.Module):
    r"""在 sampled owner—query—JOINT edges 上解码无界 signed $\hat\kappa$。

    model assembly 先从统一 $Z$ 路由 $z_o,z_i$ 与查询特征 $u(x)$。reader 以查询为主路径，
    每个残差块独立使用拼接条件 $c_{o,i}=[z_o\Vert z_i]$：

    $$
    h_0=W_qu(x),\qquad
    h_{l+1}=h_l+F_l\!\left((1+\gamma_l(c_{o,i}))\operatorname{LN}_l(h_l)+\beta_l(c_{o,i})\right),
    \qquad \hat\kappa=w^Th_3+b.
    $$

    canonical 数值锚点为 3 个 residual blocks、hidden width 128。输出层不使用 sigmoid/tanh，
    因为距离 Jacobian 元素 $\partial d_o(x;q)/\partial q_i$ 可正、可负且不具固定数值界。
    """

    def __init__(
        self,
        config: DistanceSensitivityDecoderCfg,
        *,
        entity_width: int,
        query_width: int,
    ) -> None:
        r"""构造 query projection、逐层双-token FiLM 与 signed scalar readout。

        Args:
            config (DistanceSensitivityDecoderCfg): 隐藏宽度与 FiLM residual block 数。
            entity_width (int): encoder final-norm entity token 宽度 $D$。
            query_width (int): 点—anchor 前端派生的 $D_q$。
        """

        super().__init__()
        self.config = config
        if min(entity_width, query_width) < 1:
            raise ValueError("sensitivity decoder entity/query widths must be positive")
        self.entity_width = entity_width  # owner 与 JOINT 来自同一 final-norm token 空间
        self.query_width = query_width  # $u(x)$ 不携带 sigma 或 query-stratum identity
        self.query_projection = nn.Linear(query_width, config.hidden_width)  # $u(x)\mapsto h_0\in\mathbb R^{128}$
        self.blocks = nn.ModuleList(
            _FiLMResidualBlock(config.hidden_width, 2 * entity_width)
            for _ in range(config.residual_blocks)
        )  # 每个 block 拥有独立 $\gamma_l,\beta_l,F_l$
        self.output = nn.Linear(config.hidden_width, 1)  # 无界 signed scalar，允许零点与正负响应

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        selected_query: torch.Tensor,
    ) -> torch.Tensor:
        r"""从已经路由的三类特征输出 `[B,E]` 距离灵敏度，监督单位为 m/rad。

        ``owner_latent`` 与 ``joint_latent`` 都是 `[B,E,D]` 的统一 entity tokens；
        ``selected_query`` 是 `[B,E,D_q]`。selector 解释属于 model assembly，本 reader 不持有
        owner/JOINT 轴规则，因而也不能绕过统一 $Z$ 读取 raw screw 或固定 slot identity。
        """

        if owner_latent.shape != joint_latent.shape or owner_latent.ndim != 3:
            raise ValueError("owner_latent and joint_latent must share [B,E,D] shape")
        if owner_latent.shape[-1] != self.entity_width:
            raise ValueError("owner/JOINT latent width does not match sensitivity decoder")
        if selected_query.shape != (*owner_latent.shape[:2], self.query_width):
            raise ValueError("selected_query must have shape [B,E,D_q]")
        hidden = self.query_projection(selected_query)  # `[B,E,128]` 的 query-main 初始状态
        condition = torch.cat((owner_latent, joint_latent), dim=-1)  # `[B,E,2D]` 的双-token 条件
        for block in self.blocks:
            hidden = block(hidden, condition)  # 独立 FiLM 后的两层 MLP residual update
        return self.output(hidden).squeeze(-1)  # `[B,E]`，不施加数值范围压缩


__all__ = [
    "ConditionalDensityDecoder",
    "DistanceSensitivityDecoderCfg",
    "DistanceSensitivityDecoder",
    "GeometrySSLDecoderCfg",
    "ScalarSigmaFiLMDensityDecoderCfg",
]
