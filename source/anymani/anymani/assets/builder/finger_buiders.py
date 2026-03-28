r"""TODO:自定义手指构建器配置类 `FingerBuilderCfg` 和运行时类 `FingerBuilderCfg`

计划拆分为两大主类，一个是 `Human-like hand`(类人手)，一个是 `Gripper-like hand`（夹爪手）
-  
- 
"""
from __future__ import annotations

from assets.asset_builders import FingerBuilderCfg, FingerBuilder
from assets.asset_base import FingerCfg
from assets.asset_schema_core import Vector6, Vector3

from dataclasses import dataclass, field
from typing import Any

@dataclass
class FulAcBuilderCfg(FingerBuilderCfg):
    r"""自定义手指构建器配置类。

    该声明式配置类包含的字段为构建类算法所需，而非单纯照搬 `FingerCfg` 的所有字段

    核心思想是 “算法里人易理解和显式控制的参数” 映射到 `FingerCfg` 的字段上
    """

    class_type: type["CustomFingerBuilder"] | None = None
    """关联的自定义手指构建器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = CustomFingerBuilder