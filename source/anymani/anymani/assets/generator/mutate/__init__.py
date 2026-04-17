r"""手部资产生成器的后序变异工具包。

工具按职责分拆到独立模块，本包汇总所有公开符号：

- ``_base``         → MutatorBase（最小基类协议）
- ``_distribution`` → ScalarDistributionCfg（独立标量分布描述）
- ``joint_delete``  → JointDeleteCfg, JointDeleteMutator（关节删除 + 重连）
- ``link_scale``    → LinkScaleCfg, LinkScaleMutator（连杆长度缩放）
- ``tip_replace``   → TipReplaceCfg, TipReplaceMutator（指尖几何替换）
- ``limit_tweak``   → LimitTweakCfg, LimitTweakMutator（关节限位微调）
- ``mount_perturb`` → MountPerturbCfg, MountPerturbMutator（挂载点扰动）
- ``pipeline``      → MutatorTerm, HandMutatorCfg, HandMutator（流水线编排）
"""

from ._base import MutatorBase
from ._distribution import ScalarDistributionCfg, sample_scalar_distribution
from .joint_delete import JointDeleteCfg, JointDeleteMutator
from .limit_tweak import LimitTweakCfg, LimitTweakMutator
from .link_scale import LinkScaleCfg, LinkScaleMutator
from .mount_perturb import MountPerturbCfg, MountPerturbMutator
from .pipeline import HandMutator, HandMutatorCfg, MutatorTerm
from .tip_replace import TipReplaceCfg, TipReplaceMutator

__all__ = [
    # 基类
    "MutatorBase",
    "ScalarDistributionCfg",
    "sample_scalar_distribution",
    # 流水线
    "MutatorTerm",
    "HandMutatorCfg",
    "HandMutator",
    # 结构类
    "JointDeleteCfg",
    "JointDeleteMutator",
    # 几何类
    "TipReplaceCfg",
    "TipReplaceMutator",
    # 参数类
    "LinkScaleCfg",
    "LinkScaleMutator",
    "LimitTweakCfg",
    "LimitTweakMutator",
    "MountPerturbCfg",
    "MountPerturbMutator",
]
