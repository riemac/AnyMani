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
from .se3_config import SE3CoordinateRewriteCfg, SE3DensityMaterialJacobianMethodCfg
from .se3_method import SE3DensityMaterialJacobianMethod

__all__ = [
    "DensityMaterialJacobianMethod",
    "DensityMaterialJacobianMethodCfg",
    "DensityMaterialJacobianObjectivesCfg",
    "DensityObjectiveCfg",
    "GammaChannelScaleCfg",
    "MaterialJacobianObjectiveCfg",
    "MaterialPointSamplingCfg",
    "SE3CoordinateRewriteCfg",
    "SE3DensityMaterialJacobianMethod",
    "SE3DensityMaterialJacobianMethodCfg",
]
