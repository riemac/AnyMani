r"""多锚点条件隐式场的零阶与 sampled-edge 一阶监督数据包。

对 batch size $B$、PALM/JOINT/TIP 归属体数 $G=N_E$、每个归属体 query 数 $N_Q$ 与
Gaussian 带宽数 $L$，零阶目标逻辑形状为：

```text
query_points : [B, G, N_Q, 3]  # `{h}`，m；fixed-query baseline 可由 [N_Q,3] 广播
distance     : [B, G, N_Q]     # $d_g$，m
density      : [B, G, N_Q, L]  # $\rho_{\sigma,g}$，无量纲
valid_mask   : [B, G, N_Q]     # True 表示该归属体/query target 有效
owner_role   : [G]             # PALM/JOINT/TIP；同结构 microbatch 共享并与实体轴同索引
```

field family、bandwidth、query layout、collision source、asset id、owner mapping 与 gauge-pair id
都必须进入 provenance。fixed-query vector decoder 与 conditional implicit decoder 消费同一物理
target，只在输出/readout contract 上不同。

multi-anchor route 还要求 target batch/provenance 能恢复 $K$ 个 mount-conditioned physical anchors
与每个 query 的 all-anchor relations。field scalar 仍由 physical query 和 posed surface 唯一决定；
anchors 改变的是 query 的 origin-independent 表示，不是另造一套物理 label。$K$ 始终表示锚点数，
不得再用于 query 轴。当前不训练独立 whole-hand union target。

若启用一阶监督，逻辑完整形状是 ``kappa: [B,G,N_Q,N_J]`` 与
``g: [B,G,N_Q,L,N_J]``，但默认只 materialize sampled owner--JOINT edges 或方向 JVP，
并保留最近点来源、祖先结构零与非光滑区域掩码。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import torch


class QueryStratum(IntEnum):
    r"""训练查询点的三种物理来源；枚举值只用于 provenance 与 loss weighting。"""

    WORKSPACE = 0  # 固定于 palm/anchor workspace，跨 q 复用
    OWNER_SHELL = 1  # 当前 target owner 的双侧近表面壳层
    ADJACENT = 2  # 同指邻接 owner surface 或刚体间隙


@dataclass(frozen=True)
class FieldTargetBatch:
    r"""逐归属体多带宽零阶监督。

    `query_stratum` 不属于 decoder 输入；它只允许 objective 按 50/25/25 来源分层统计或
    施加正权重。`owner_role[g]` 与实体轴、表面归属轴和 decoder owner 轴直接同索引。
    """

    query_points: torch.Tensor  # `[B,G,N_Q,3]`，`{h}`，m
    query_stratum: torch.Tensor  # `[B,G,N_Q]`，`QueryStratum`
    distance: torch.Tensor  # `[B,G,N_Q]`，m
    density: torch.Tensor  # `[B,G,N_Q,L]`，无量纲
    valid_mask: torch.Tensor  # `[B,G,N_Q]`，有效 target
    owner_role: torch.Tensor  # `[G]`，PALM/JOINT/TIP 整数角色
    bandwidths: torch.Tensor  # `[L]`，m
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        r"""验证 owner/query/bandwidth 轴和最小 frame/unit provenance。"""

        if self.query_points.ndim != 4 or self.query_points.shape[-1] != 3:
            raise ValueError(f"query_points must have shape [B,G,N_Q,3], got {tuple(self.query_points.shape)}")
        base_shape = self.query_points.shape[:-1]  # `[B,G,N_Q]`，所有零阶标量 target 的共同轴
        if self.query_stratum.shape != base_shape:
            raise ValueError("query_stratum must have shape [B,G,N_Q]")
        if self.distance.shape != base_shape or self.valid_mask.shape != base_shape:
            raise ValueError("distance and valid_mask must have shape [B,G,N_Q]")
        if self.bandwidths.ndim != 1 or self.density.shape != (*base_shape, self.bandwidths.numel()):
            raise ValueError("density must have shape [B,G,N_Q,L] and bandwidths must have shape [L]")
        if self.owner_role.shape != (base_shape[1],):
            raise ValueError(f"owner_role must have shape [G]={base_shape[1:2]}, got {tuple(self.owner_role.shape)}")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool")
        if torch.any(self.query_stratum < int(QueryStratum.WORKSPACE)) or torch.any(
            self.query_stratum > int(QueryStratum.ADJACENT)
        ):
            raise ValueError("query_stratum contains an unknown physical stratum")
        if self.provenance.get("frame") != "h" or self.provenance.get("length_unit") != "m":
            raise ValueError("FieldTargetBatch provenance must declare frame='h' and length_unit='m'")


@dataclass(frozen=True)
class SensitivityTargetBatch:
    r"""只在 sampled owner–query–JOINT edges 上 materialize 的一阶监督。

    设 sampled edge 数为 $E$。`owner_index/query_index/joint_index` 共同选择零阶 target
    中的一个 query 和一个 JOINT 坐标；每个 edge 对全部 $L$ 个带宽保存场灵敏度。非祖先
    edge 的 `ancestor_mask=False` 且 `kappa/field_sensitivity` 必须精确为零。
    """

    owner_index: torch.Tensor  # `[E]`
    query_index: torch.Tensor  # `[E]`
    joint_index: torch.Tensor  # `[E]`
    ancestor_mask: torch.Tensor  # `[E]`
    closest_point: torch.Tensor  # `[B,E,3]`，`{h}`，m
    closest_source: torch.Tensor  # `[B,E]`，稳定 component/face 来源 id
    uniqueness_margin: torch.Tensor  # `[B,E]`，m
    kappa: torch.Tensor  # `[B,E]`，m/rad
    field_sensitivity: torch.Tensor  # `[B,E,L]`，1/rad
    valid_mask: torch.Tensor  # `[B,E]`

    def __post_init__(self) -> None:
        r"""验证 sampled-edge 轴，并拒绝把非祖先目标写成非零。"""

        edge_count = self.owner_index.numel()  # sampled edge 数 $E$
        for name, selector in (
            ("owner_index", self.owner_index),
            ("query_index", self.query_index),
            ("joint_index", self.joint_index),
            ("ancestor_mask", self.ancestor_mask),
        ):
            if selector.shape != (edge_count,):
                raise ValueError(f"{name} must have shape [E], got {tuple(selector.shape)}")
        if self.closest_point.ndim != 3 or self.closest_point.shape[1:] != (edge_count, 3):
            raise ValueError("closest_point must have shape [B,E,3]")
        batch_size = self.closest_point.shape[0]  # microbatch 大小 $B$
        edge_shape = (batch_size, edge_count)  # 所有 sampled-edge 标量的共同轴
        for name, value in (
            ("closest_source", self.closest_source),
            ("uniqueness_margin", self.uniqueness_margin),
            ("kappa", self.kappa),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != edge_shape:
                raise ValueError(f"{name} must have shape [B,E]={edge_shape}, got {tuple(value.shape)}")
        if self.field_sensitivity.ndim != 3 or self.field_sensitivity.shape[:2] != edge_shape:
            raise ValueError("field_sensitivity must have shape [B,E,L]")
        if self.ancestor_mask.dtype != torch.bool or self.valid_mask.dtype != torch.bool:
            raise TypeError("ancestor_mask and valid_mask must use torch.bool")

        nonancestor = ~self.ancestor_mask  # `[E]`，跨指或 palm 等结构零 edges
        if torch.any(self.kappa[:, nonancestor] != 0) or torch.any(self.field_sensitivity[:, nonancestor] != 0):
            raise ValueError("non-ancestor sensitivity targets must be exactly zero")


__all__ = ["FieldTargetBatch", "QueryStratum", "SensitivityTargetBatch"]
