r"""多锚点 Gaussian 隐式场：task-free 几何预训练的 concrete method。"""

from . import objectives as _objectives  # noqa: F401  # 绑定五项 ClassVar func
from .config import (
    ChainObjectiveCfg,
    DensityObjectiveCfg,
    DerivedFieldObjectiveCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianMethodCfg,
    MultiAnchorGaussianObjectivesCfg,
    SobolevObjectiveCfg,
)
from .method import MultiAnchorGaussianMethod

__all__ = [
    "ChainObjectiveCfg",
    "DensityObjectiveCfg",
    "DerivedFieldObjectiveCfg",
    "JointConfigurationMeasureCfg",
    "JointSignRewriteCfg",
    "KappaObjectiveCfg",
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorGaussianObjectivesCfg",
    "SobolevObjectiveCfg",
]
