r"""Anchor-relational Material-point Jacobian 的 disposable per-anchor reader。

Retained encoder 产生当前构型下的 owner token $Z_g(q)$ 与 selected JOINT token $Z_i(q)$；static query
frontend 为固定 material identity 与第 $k$ 个 PALM anchor 产生 $f_{gmk}^{static}$。Reader 预测：

$$
\widehat\Gamma_{gmki}
=
D_\theta\!\left(
f_{gmk}^{static}
+C_\theta([Z_g(q),Z_i(q)])
\right)
\in\mathbb R^4/\mathrm{rad}.
$$

最后四通道顺序固定为 `[height,radius,dot,chirality]`。同一个 reader 对全部 anchors、material points、
owners 与 JOINTs 共享参数；anchor axis 没有 index embedding，因此输入 $K$ 轴置换只同步置换输出。
该模块只拥有可学习 reader，不生成物理 target、不定义 loss，也不进入 retained-only checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AnchorRelationalJacobianDecoderCfg:
    r"""Owner/JOINT latent、static relation query 与 reader 隐藏容量。"""

    latent_width: int = 128  # retained owner/JOINT token 宽度 $D$
    relation_width: int = 64  # per material-point/anchor static relation feature 宽度 $D_r$
    hidden_width: int = 128  # 四通道输出前的共享 MLP 宽度

    def __post_init__(self) -> None:
        r"""拒绝空 latent、query 或 reader 容量。"""

        if min(self.latent_width, self.relation_width, self.hidden_width) < 1:
            raise ValueError("material-point Jacobian decoder widths must be positive")


class AnchorRelationalJacobianDecoder(nn.Module):
    r"""从 unified owner/JOINT Z 与 static material-anchor query 预测四通道 relation Jacobian。

    输入逻辑形状为：

    ```text
    owner_latent       [B,E,D]
    joint_latent       [B,E,D]
    static_pair_feature[B,E,K,D_r]
    output             [B,E,K,4]
    ```

    $E$ 是 sampled owner/material/JOINT edge，$K$ 是当前资产实际 anchor 数。跨结构 padding mask 由
    method/objective 持有；reader 不把 invalid anchor 或 edge 改写为可学习特殊 token。
    """

    def __init__(self, config: AnchorRelationalJacobianDecoderCfg = AnchorRelationalJacobianDecoderCfg()) -> None:
        r"""构造动态 edge context 投影与共享 per-anchor 四通道 MLP。"""

        super().__init__()
        self.config = config  # reader shape 与 artifact identity 的显式容量合同
        self.context_projection = nn.Sequential(
            nn.Linear(2 * config.latent_width, config.relation_width),
            nn.GELU(),
            nn.Linear(config.relation_width, config.relation_width),
        )  # $C_\theta([Z_g,Z_i])\in\mathbb R^{D_r}$
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.relation_width),
            nn.Linear(config.relation_width, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, 4),
        )  # 固定输出 `[height,radius,dot,chirality]`，单位由 objective 解释为 rad$^{-1}$

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        static_pair_feature: torch.Tensor,
    ) -> torch.Tensor:
        r"""对每个 sampled edge/anchor 预测四通道 relation sensitivity。

        Args:
            owner_latent (torch.Tensor): `[B,E,D]` 当前 owner representation。
            joint_latent (torch.Tensor): `[B,E,D]` selected JOINT representation。
            static_pair_feature (torch.Tensor): `[B,E,K,D_r]` 固定 material identity 相对 anchors 的关系特征。

        Returns:
            torch.Tensor: `[B,E,K,4]`，四通道顺序为 height/radius/dot/chirality。
        """

        if owner_latent.shape != joint_latent.shape or owner_latent.ndim != 3:
            raise ValueError("owner_latent and joint_latent must have identical [B,E,D] shape")
        if owner_latent.shape[-1] != self.config.latent_width:
            raise ValueError("owner/joint latent width does not match decoder config")
        if (
            static_pair_feature.ndim != 4
            or static_pair_feature.shape[:2] != owner_latent.shape[:2]
            or static_pair_feature.shape[-1] != self.config.relation_width
        ):
            raise ValueError("static_pair_feature must have aligned [B,E,K,D_r] shape")

        # 每条 owner/JOINT edge 形成一个动态 context，再沿无序 K 轴广播到每个实际 anchor。
        edge_context = self.context_projection(
            torch.cat((owner_latent, joint_latent), dim=-1)
        )  # `[B,E,D_r]`，包含当前 q 与整手图上下文
        fused = static_pair_feature + edge_context.unsqueeze(-2)  # `[B,E,K,D_r]`，不引入 anchor index
        return self.output_projection(fused)  # `[B,E,K,4]`


__all__ = ["AnchorRelationalJacobianDecoder", "AnchorRelationalJacobianDecoderCfg"]
