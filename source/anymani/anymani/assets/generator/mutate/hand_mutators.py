r"""向后兼容的 re-export 桩（已拆分到各独立模块）。

本文件的原始内容已拆分到以下模块，请直接导入对应模块：

- ``.pipeline``       → HandMutatorCfg, HandMutator
- ``._distribution``  → ScalarDistributionCfg
- ``.joint_delete``   → JointDeleteCfg, JointDeleteMutator
- ``.mount_perturb``  → MountPerturbCfg, MountPerturbMutator
- ``.link_scale``     → LinkScaleCfg, LinkScaleMutator
- ``.tip_replace``    → TipReplaceCfg, TipReplaceMutator
- ``.limit_tweak``    → LimitTweakCfg, LimitTweakMutator

此文件保留仅为不破坏已有代码的 import 路径。
"""

from ._distribution import ScalarDistributionCfg
from .joint_delete import JointDeleteCfg, JointDeleteMutator
from .limit_tweak import LimitTweakCfg, LimitTweakMutator
from .link_scale import LinkScaleCfg, LinkScaleMutator
from .mount_perturb import MountPerturbCfg, MountPerturbMutator
from .pipeline import HandMutator, HandMutatorCfg, MutatorTerm
from .tip_replace import TipReplaceCfg, TipReplaceMutator

__all__ = [
    "ScalarDistributionCfg",
    "MutatorTerm",
    "HandMutatorCfg",
    "HandMutator",
    "JointDeleteCfg",
    "JointDeleteMutator",
    "MountPerturbCfg",
    "MountPerturbMutator",
    "LinkScaleCfg",
    "LinkScaleMutator",
    "TipReplaceCfg",
    "TipReplaceMutator",
    "LimitTweakCfg",
    "LimitTweakMutator",
]
