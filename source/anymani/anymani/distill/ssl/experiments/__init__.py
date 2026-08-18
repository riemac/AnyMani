r"""Geometry SSL 的声明式实验组合。

实验模块拥有完整科学语义组合；``ssl.runtime`` 只执行已 resolve 的资产、q、window、optimizer
和 checkpoint 轴，不在运行时猜测模型或损失。当前正式候选通过 assets 层 dataset manifest
选择多个 mother lineages，不在 Python 实验模块中枚举 leaf asset IDs。
"""

from .canonical_residual_family import FORMAL_ASSET_DATASET_MANIFEST, CanonicalResidualFamilyCfg

__all__ = [
    "CanonicalResidualFamilyCfg",
    "FORMAL_ASSET_DATASET_MANIFEST",
]
