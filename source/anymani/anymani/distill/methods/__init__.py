r"""Task-free embodiment methods：科学聚合根，不属于某一个训练阶段。

`ssl/` 只负责如何训练一个可训练 method；具体物理测度、模型与损失由本包的 concrete method 拥有。
"""

from .contracts import EmbodimentMethod, FeatureSpec, MethodStep, MethodUpdate
from .multi_anchor_gaussian_implicit_field import (
    MultiAnchorGaussianMethod,
    MultiAnchorGaussianMethodCfg,
)

__all__ = [
    "EmbodimentMethod",
    "FeatureSpec",
    "MethodStep",
    "MethodUpdate",
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianMethodCfg",
]
