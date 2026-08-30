r"""Gaussian density + anchor-relational Material-point Jacobian concrete method。"""

from . import objectives as _objectives  # noqa: F401  # 绑定 density/Gamma objective callables
from .config import (
    DensityMaterialJacobianMethodCfg,
    DensityMaterialJacobianObjectivesCfg,
    DensityObjectiveCfg,
    GammaChannelScaleCfg,
    MaterialJacobianObjectiveCfg,
    MaterialPointSamplingCfg,
)
from .method import DensityMaterialJacobianMethod

__all__ = [
    "DensityMaterialJacobianMethod",
    "DensityMaterialJacobianMethodCfg",
    "DensityMaterialJacobianObjectivesCfg",
    "DensityObjectiveCfg",
    "GammaChannelScaleCfg",
    "MaterialJacobianObjectiveCfg",
    "MaterialPointSamplingCfg",
]
