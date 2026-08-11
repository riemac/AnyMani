r"""验证器基础协议：ValidationResult + ValidatorBase。

这里定义的是整个验证层的最小公共语言：一个结果对象和一个接口约定。
所有层级的验证器（joint / finger / hand）都基于这两个类型工作。

设计说明
--------

### ValidationResult 的设计

验证结果分三类：

- ``errors``：严重违规，必须拒绝（如运动学链断裂、关节数为 0）
- ``warnings``：潜在问题，默认放行但可被 ``strict`` 模式升级为错误
  （如关节限位范围超出建议值、link 长度接近零）
- ``passed``：``True`` 当且仅当 ``errors`` 为空（warnings 不影响 passed）

在 ``strict=True`` 模式下，调用方可自行把 warnings 并入 errors 后重判 passed。

### 与 schema 层 __post_init__ 的分工

schema 层（`JointCfg.__post_init__` 等）已经捕获"明显非法"的输入（如
关节类型不合法、链断裂等）并立即抛出异常，属于**构造时硬校验**。

验证器层处理的是"物理/工程上的软约束"：

- 关节限位范围是否在建议区间内
- 整手 DOF 是否在合理范围
- link 长度是否在允许的最小/最大值之间

这类约束不会被嵌入 schema（避免限制太死），而是作为可配置规则放在验证器里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    r"""一次验证调用的结果包。

    ``passed`` 仅反映 ``errors``（严重违规）是否为空；
    ``warnings`` 的有无不影响 ``passed``。
    在 ``strict`` 调用模式下，调用方自行把 warnings 合并后重判 passed。
    """

    passed: bool = True
    """严重错误为空时为 ``True``。"""

    errors: list[str] = field(default_factory=list)
    """必须拒绝的严重违规消息列表。"""

    error_codes: list[str] = field(default_factory=list)
    """与 ``errors`` 对齐的稳定规则代码。

    人类可读消息可以携带 hand 名、阈值和几何测量值，因此不适合作为跨样本统计键；
    ``error_codes`` 只表达命中的规则，例如 ``hand.palm_thumb_family_mismatch``。
    """

    warnings: list[str] = field(default_factory=list)
    """潜在问题消息列表；默认放行，``strict`` 模式下升级为错误。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """验证器附加的结构化证据。

    # NOTE:
    原先 `ValidationResult` 只承载 human-readable errors / warnings。
    SDF clearance 这类几何规则需要把“证书边界”也传给 generator / sidecar：

    - 检测姿态是不是 post-mutate home pose；
    - 是否只用了 collision geometry；
    - 是否有 unsupported body 被跳过；
    - 这个结果不证明哪些更强 claim。

    因而这里加一个轻量 `metadata` 字段，而不把几何证书硬编码进基础协议。
    """

    def add_error(self, message: str, *, code: str) -> None:
        r"""追加一条人类错误消息及其稳定规则代码。

        Args:
            message (str): 携带样本名、阈值或测量证据的可读错误消息。
            code (str): 不随样本变化的规则标识，用于 run-level rejection 统计。
        """

        self.errors.append(str(message))  # errors 保持既有面向研究者的可读接口
        self.error_codes.append(str(code))  # code 与同位置的 error 一一对应，供 summary 聚合
        self.passed = False

    def merge(self, other: ValidationResult) -> ValidationResult:
        r"""把另一个验证结果并入自身（就地合并）。

        Args:
            other (ValidationResult): 待合并的验证结果。

        Returns:
            ValidationResult: 合并后的自身（支持链式调用）。
        """

        self.errors.extend(other.errors)
        self.error_codes.extend(other.error_codes)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        self.passed = len(self.errors) == 0
        return self

    def as_strict(self) -> ValidationResult:
        r"""返回一个把 warnings 升级为 errors 的新结果（不修改自身）。

        Returns:
            ValidationResult: 升级后的新 ValidationResult。
        """

        return ValidationResult(
            passed=len(self.errors) + len(self.warnings) == 0,
            errors=self.errors + self.warnings,
            error_codes=self.error_codes + ["strict.warning_promoted"] * len(self.warnings),
            warnings=[],
            metadata=dict(self.metadata),
        )

    def __bool__(self) -> bool:
        return self.passed


class ValidatorBase:
    r"""所有验证器的最小基类。

    子类只需实现对应层级的 ``validate()``，结果通过 ``ValidationResult``
    统一表达，而不是直接抛出异常（抛错由调用方根据 passed 决定）。
    """

    def validate(self, target: object) -> ValidationResult:
        r"""对目标对象执行验证。

        Args:
            target (object): 待验证的资产对象（JointCfg / FingerCfg / HandCfg 等）。

        Returns:
            ValidationResult: 验证结果包。
        """

        raise NotImplementedError


__all__ = ["ValidationResult", "ValidatorBase"]
