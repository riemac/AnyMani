r"""多锚点条件隐式场的零阶与抽样边一阶监督数据包。

对批大小 $B$、PALM/JOINT/TIP 归属体数 $G=N_E$、每个归属体查询点数 $N_Q$ 与
显式 sigma 样本数 $N_\sigma$，零阶目标逻辑形状为：

```text
query_points : [B, G, N_Q, 3]  # `{h}`，m；固定查询基线可由 [N_Q,3] 广播
distance     : [B, G, N_Q]     # $d_g$，m
density      : [B, G, N_Q, N_sigma]  # $\rho_{\sigma,g}$，无量纲
valid_mask   : [B, G, N_Q]     # True 表示该归属体/查询点监督有效
owner_role   : [G]             # PALM/JOINT/TIP；同结构微批次共享并与实体轴同索引
```

场定义、带宽、查询布局、碰撞来源、资产标识、归属映射与规范变换配对标识都必须进入来源信息。
固定查询向量解码器与条件隐式解码器消费同一物理监督，只在输出与读取方式上不同。

多锚点路线还要求监督批次和来源信息能恢复 $K$ 个挂载条件物理锚点，以及每个查询点相对全部
锚点的关系。场标量仍由物理查询点与当前表面唯一决定；锚点改变的是与原点无关的查询表示，
不是另造一套物理标签。$K$ 始终表示锚点数，不得再用于查询轴。当前不训练独立整手并集监督。

若启用一阶监督，逻辑完整形状是 ``kappa: [B,G,N_Q,N_J]`` 与
``g: [B,G,N_Q,N_sigma,N_J]``，但默认只实际生成抽样的归属体—JOINT 边或方向 JVP，
并保留最近点来源、祖先结构零与非光滑区域掩码。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import torch


class QueryStratum(IntEnum):
    r"""训练查询点的三种物理来源。

    枚举值只进入来源信息、分层指标和损失权重，不作为解码器输入。这样可以改变抽样测度，
    但不会让模型通过“查询来源标签”绕过真实几何表征。
    """

    WORKSPACE = 0  # 固定于手掌/锚点工作空间，跨 $q$ 复用
    OWNER_SHELL = 1  # 当前目标归属体的双侧近表面壳层
    ADJACENT = 2  # 同指邻接归属体表面或刚体间隙


@dataclass(frozen=True)
class FieldTargetBatch:
    r"""逐归属体多带宽零阶监督。

    `query_stratum` 不属于解码器输入；它只允许目标函数按 50/25/25 来源分层统计或施加正权重。
    `owner_role[g]` 与实体轴、表面归属轴和解码器归属体轴直接同索引。
    """

    query_points: torch.Tensor  # `[B,G,N_Q,3]`，`{h}`，m
    query_stratum: torch.Tensor  # `[B,G,N_Q]`，`QueryStratum`
    distance: torch.Tensor  # `[B,G,N_Q]`，m
    density: torch.Tensor  # `[B,G,N_Q,N_sigma]`，无量纲
    valid_mask: torch.Tensor  # `[B,G,N_Q]`，有效监督
    owner_role: torch.Tensor  # `[G]` 或跨结构 padding 后 `[B,G]`
    bandwidths: torch.Tensor  # `[N_sigma]` 或 `[B,N_sigma]`，m；后者保存实际 sigma realization
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        r"""验证归属体、查询点、带宽轴及最小坐标系/单位来源信息。

        这里不验证模型预测或采样比例，只保证一份数据包内部的物理轴闭合。来源信息必须显式声明
        ``frame='h'`` 与 ``length_unit='m'``，防止米、厘米或资产原始坐标被静默混合。
        """

        if self.query_points.ndim != 4 or self.query_points.shape[-1] != 3:
            raise ValueError(f"query_points must have shape [B,G,N_Q,3], got {tuple(self.query_points.shape)}")
        base_shape = self.query_points.shape[:-1]  # `[B,G,N_Q]`，所有零阶标量 target 的共同轴
        if self.query_stratum.shape != base_shape:
            raise ValueError("query_stratum must have shape [B,G,N_Q]")
        if self.distance.shape != base_shape or self.valid_mask.shape != base_shape:
            raise ValueError("distance and valid_mask must have shape [B,G,N_Q]")
        if self.bandwidths.ndim not in {1, 2}:
            raise ValueError("bandwidths must have shape [L] or [B,L]")
        if self.bandwidths.ndim == 2 and self.bandwidths.shape[0] != base_shape[0]:
            raise ValueError("sampled bandwidths [B,L] must share B with query targets")
        bandwidth_count = self.bandwidths.shape[-1]  # $N_\sigma$ 是数据采样轴，不是 decoder 固定宽度
        if self.density.shape != (*base_shape, bandwidth_count):
            raise ValueError("density must have shape [B,G,N_Q,N_sigma]")
        if torch.any(self.bandwidths <= 0.0):
            raise ValueError("bandwidths must be strictly positive")
        if self.owner_role.shape not in {(base_shape[1],), (base_shape[0], base_shape[1])}:
            raise ValueError("owner_role must have shape [G] or [B,G]")
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
    r"""只在抽样的归属体—查询点—JOINT 边上实际生成的一阶监督。

    设抽样边数为 $E$。`owner_index/query_index/joint_index` 共同选择零阶监督中的一个查询点
    和一个 JOINT 坐标；每条边对全部 $N_\sigma$ 个显式 sigma 保存场灵敏度。非祖先边的
    `ancestor_mask=False`，且 `kappa/field_sensitivity` 必须精确为零。
    """

    owner_index: torch.Tensor  # `[E]` 或跨结构 padding 后 `[B,E]`
    query_index: torch.Tensor  # 与 owner selector 同形状
    joint_index: torch.Tensor  # 与 owner selector 同形状
    ancestor_mask: torch.Tensor  # 与 owner selector 同形状；True 表示运动学祖先 / active edge
    active_mask: torch.Tensor  # 与 owner selector 同形状；True=active descendant，False=structure-zero
    closest_point: torch.Tensor  # `[B,E,3]`，`{h}`，m
    closest_source: torch.Tensor  # `[B,E]`，稳定碰撞部件/三角面来源标识
    uniqueness_margin: torch.Tensor  # `[B,E]`，m
    kappa: torch.Tensor  # `[B,E]`，m/rad
    field_sensitivity: torch.Tensor  # `[B,E,N_sigma]`，1/rad
    valid_mask: torch.Tensor  # `[B,E]`
    provenance: Mapping[str, str] = field(default_factory=dict)  # 最近点、mask、frame 与单位定义

    def __post_init__(self) -> None:
        r"""验证抽样边轴，并拒绝把非祖先目标写成非零。

        最近点唯一性间隔和有效掩码由目标后端给出；本数据类只验证形状与结构零，不在这里
        猜测何种间隔阈值代表可微区域。
        """

        if self.owner_index.ndim not in {1, 2}:
            raise ValueError("edge selectors must have shape [E] or [B,E]")
        edge_count = self.owner_index.shape[-1]  # 每个样本统一存储预算 $E$
        selector_shape = self.owner_index.shape
        for name, selector in (
            ("owner_index", self.owner_index),
            ("query_index", self.query_index),
            ("joint_index", self.joint_index),
            ("ancestor_mask", self.ancestor_mask),
            ("active_mask", self.active_mask),
        ):
            if selector.shape != selector_shape:
                raise ValueError(f"{name} must share selector shape {selector_shape}, got {tuple(selector.shape)}")
        if self.closest_point.ndim != 3 or self.closest_point.shape[1:] != (edge_count, 3):
            raise ValueError("closest_point must have shape [B,E,3]")
        batch_size = self.closest_point.shape[0]  # minibatch 大小 $B$
        if self.owner_index.ndim == 2 and self.owner_index.shape[0] != batch_size:
            raise ValueError("batched edge selectors must share B with closest_point")
        edge_shape = (batch_size, edge_count)  # 所有抽样边标量的共同轴
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
        if self.ancestor_mask.dtype != torch.bool or self.active_mask.dtype != torch.bool or self.valid_mask.dtype != torch.bool:
            raise TypeError("ancestor_mask, active_mask and valid_mask must use torch.bool")
        if torch.any(self.active_mask != self.ancestor_mask):
            raise ValueError("active_mask must match ancestor_mask: active edges are kinematic descendants")

        nonancestor = ~self.ancestor_mask  # 跨指或 palm 等结构零 edges
        if self.ancestor_mask.ndim == 1:
            invalid_kappa = self.kappa[:, nonancestor]
            invalid_field = self.field_sensitivity[:, nonancestor]
        else:
            invalid_kappa = self.kappa[nonancestor]
            invalid_field = self.field_sensitivity[nonancestor]
        if torch.any(invalid_kappa != 0) or torch.any(invalid_field != 0):
            raise ValueError("non-ancestor sensitivity targets must be exactly zero")
        if self.provenance and (
            self.provenance.get("frame") != "h"
            or self.provenance.get("distance_unit") != "m"
            or self.provenance.get("joint_unit") != "rad"
        ):
            raise ValueError("SensitivityTargetBatch provenance must declare frame='h', distance_unit='m', joint_unit='rad'")


__all__ = ["FieldTargetBatch", "QueryStratum", "SensitivityTargetBatch"]
