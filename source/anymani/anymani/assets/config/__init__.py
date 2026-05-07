"""资产生成声明式配置。

这里对齐 IsaacLab 的 `tasks/.../config/*.py` 风格：配置模块只声明可编辑的
`HandGeneratorCfg` 与少量 runner 级策略/路径常量，不承载执行逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..asset_base import AssetCfgBase


@dataclass
class AssetRunStrategyCfg(AssetCfgBase):
    r"""runner 级运行策略占位配置。

    这些字段目前只在统一 runner 层暴露和校验，还不进入 `HandGenerator`
    的正式 contract。这样后续可以先扩展运行编排，再决定是否下沉到生成器本体。
    """

    topology_selection_mode: Literal["all", "random_subset", "random_subset_with_full_hand"] = "all"
    """post-mutate 未来的拓扑挑选策略。当前仅 `all` 已实现。"""

    topology_selection_count: int | None = None
    """当策略不是 `all` 时，计划选择多少个 topology。当前仅作占位。"""


__all__ = ["AssetRunStrategyCfg"]
