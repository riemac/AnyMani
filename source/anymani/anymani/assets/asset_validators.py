r"""手部资产验证层的顶层抽象合同。

本文件与 `asset_generator.py` / `asset_builders.py` 类似，主要承担两类职责：

- 定义共享的验证报告对象与基类合同；
- 作为兼容入口 re-export joint / finger / palm / hand 四层验证器。

真正的层级正文下沉到 `validator/` 子目录，避免本文件再次膨胀。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .asset_base import AssetCfgBase


@dataclass
class ValidationIssue(AssetCfgBase):
    r"""单条验证问题。

    这里统一承载 fatal error / warning / info 三类问题，供 generator 后续做
    reject、统计或研究分析。
    """

    scope: str = "hand"
    """问题作用域，例如 `joint` / `finger` / `palm` / `hand`。"""

    severity: Literal["fatal", "warning", "info"] = "fatal"
    """问题严重级别。`fatal` 用于直接 reject，`warning/info` 用于软报告。"""

    code: str = "validation.issue"
    """问题编码。便于后续统计 reject reason 分布。"""

    message: str = ""
    """面向人阅读的诊断信息。"""

    details: dict[str, Any] = field(default_factory=dict)
    """可选的结构化补充信息，例如阈值、观测值与 lineage。"""


@dataclass
class ValidationReport(AssetCfgBase):
    r"""验证报告基类。"""

    target_name: str | None = None
    """被验证对象的逻辑名称。"""

    issues: list[ValidationIssue] = field(default_factory=list)
    """按生成顺序收集的验证问题列表。"""

    summary: dict[str, Any] = field(default_factory=dict)
    """可选的汇总信息，例如通过率、统计量或表示适配分数。"""


@dataclass
class ValidatorCfg(AssetCfgBase):
    r"""验证器配置基类。"""

    class_type: type["Validator"] | None = None
    """关联的验证器运行时类。"""

    strict: bool = True
    """是否把 fatal invariant 视为立即中断条件。"""


class Validator:
    r"""验证器基类。

    默认合同采用“硬错误 + 软报告”双轨：fatal invariant 负责保护结构边界，
    warning/info 则服务于研究分析与生成分布筛选。
    """

    cfg: ValidatorCfg

    def __init__(self, cfg: ValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> ValidationReport | None:
        r"""验证一个资产对象。

        Args:
            target (AssetCfgBase): 待验证资产。

        Returns:
            ValidationReport | None: 当前阶段只保留报告合同，不写正式规则实现。
        """
        pass

        # TODO:算法之一（通用验证合同）
        # ────────────────────────────────────────
        # 输入：任意实现了 `AssetCfgBase` 的资产对象。
        # 输出：`ValidationReport`，其中 `fatal` 用于 reject，`warning/info` 用于软诊断。
        #
        # ── 类型守卫 ──
        #   1. 检查 `target` 是否属于当前验证器所覆盖的层级。
        #   2. 若不匹配，则在正式实现阶段生成一条 `fatal` 级别的问题。
        #
        # ── 规则执行 ──
        #   1. 先跑 schema / topological invariant 这类硬边界。
        #   2. 再补充几何、物理、表征适配等软评估项。
        #
        # IDEA：顶层 `Validator` 只定义“问题对象长什么样”，不定义任何手型专属规则。


from .validator.finger_validators import FingerValidator, FingerValidatorCfg
from .validator.hand_validators import HandValidator, HandValidatorCfg
from .validator.joint_validators import JointValidator, JointValidatorCfg
from .validator.palm_validators import PalmValidator, PalmValidatorCfg


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "ValidatorCfg",
    "Validator",
    "JointValidatorCfg",
    "JointValidator",
    "FingerValidatorCfg",
    "FingerValidator",
    "PalmValidatorCfg",
    "PalmValidator",
    "HandValidatorCfg",
    "HandValidator",
]
