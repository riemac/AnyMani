r"""Geometry SSL 的声明式实验组合。

实验模块拥有完整科学语义组合；``ssl.runtime`` 只执行已 resolve 的资产、q、window、optimizer
和 checkpoint 轴，不在运行时猜测模型或损失。当前正式候选是 21-asset 同 topology canonical
residual family pilot。
"""

from .canonical_residual_family import (
    FORMAL_FAMILY,
    FORMAL_MOTHER,
    FORMAL_VARIANT_IDS,
    MOTHER_ASSET_ID,
    CanonicalResidualFamilyCfg,
)

__all__ = [
    "CanonicalResidualFamilyCfg",
    "FORMAL_FAMILY",
    "FORMAL_MOTHER",
    "FORMAL_VARIANT_IDS",
    "MOTHER_ASSET_ID",
]
