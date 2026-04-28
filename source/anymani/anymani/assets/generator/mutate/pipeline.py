# FIXME：post-mutate 是基于 Monte Carlo Sampling 的联合采样思想来构造的
r"""后序变异流水线：用开放 term container 编排多个局部连续参数工具。

本模块对应 `pre-made -> validator -> HandCfg -> post-mutate -> validator -> HandCfg` 里的 post-mutate 阶段

从工程优化的角度，目前我将后变异分为 3阶段:
1. Declare
2. Sample
3. Apply

# NOTE:
`joint_delete` 已明确回归 pre-made connectivity 主线，因此不再属于这里的 term；
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...validator import HandValidator, HandValidatorCfg
from ._base import MutatorBase
from ._distribution import ScalarDistributionCfg


# FIXME: 改成容器式
@dataclass
class HandMutatorCfg(AssetCfgBase):
    r"""post-mutate 开放式 term container。

    类似 IsaacLab 中 RewardsCfg 和 RewTerm 的关系。HandMutatorCfg 相当于 RewardsCfg，MutatorBaseCfg 相当于 RewTerm
    """

    class_type: type["HandMutator"] | None = None
    """关联的流水线运行时类。"""




class HandMutator:
    r"""post-mutate 流水线运行时壳。

    # FIXME: 需要重构。位于 Apply 层级，并行调用 MutatorBase 获取采样属性及参数，并行 apply 产生不同新变异 HandCfg
    """

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        self.cfg = cfg




__all__ = ["HandMutatorCfg", "HandMutator"]
