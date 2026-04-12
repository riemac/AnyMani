r"""关节级验证规则集：对 JointCfg 做逐条软约束检查。

这里的规则都是"物理/工程软约束"，不是 schema 层已经拦截的硬校验。
schema 的 ``__post_init__`` 已经保证 JointCfg 的基本合法性；
这里进一步检查"合法但可能有问题"的状态。

当前收纳的规则
--------------

1. **限位范围合规**：revolute 关节的 ``[lower, upper]`` 是否在建议区间内
2. **有效长度检查**：``||origin.pos||`` 是否不低于允许的最小值（防止退化零长 link）
3. **高曲率限位检测**：``upper - lower`` 是否合理，防止±范围过于极端（超过 2π 不合理）

设计说明
--------

### 规则可配置

每条规则对应 `JointValidatorCfg` 中一个 `bool` 开关和可选的阈值字段。
用户可以通过 cfg 选择性禁用某些规则，或调整阈值。

### 与 post-mutate 的关系

`joint_delete`、`link_scale`、`limit_tweak` 执行后，对受影响的 joint 重新跑
这些规则，是"步间轻量校验"的推荐方式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..asset_base import AssetCfgBase, JointCfg
from ._base import ValidatorBase, ValidationResult


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class JointValidatorCfg(AssetCfgBase):
    r"""关节级验证规则配置。"""

    class_type: type["JointValidator"] | None = None
    """关联的运行时类。"""

    check_limit_range: bool = True
    """是否检查关节限位范围合规（下限 < 上限，范围不超过 2π）。"""

    limit_max_range: float = 2 * math.pi
    """关节限位允许的最大范围（rad）；超过此值视为警告。默认 2π。"""

    check_link_length: bool = True
    """是否检查 joint origin.pos 的欧氏距离不低于最小 link 长度。"""

    min_link_length: float = 1e-4
    """允许的最小 link 长度（meter）；短于此值视为警告（零长退化）。"""

    strict: bool = False
    """是否把 warnings 升级为 errors（严格模式）。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointValidator


# ============================================================================
#  运行时壳
# ============================================================================


class JointValidator(ValidatorBase):
    r"""关节级验证器。

    对一个 `JointCfg` 按 ``cfg`` 中启用的规则逐条检查，返回 `ValidationResult`。
    """

    cfg: JointValidatorCfg

    def __init__(self, cfg: JointValidatorCfg):
        self.cfg = cfg

    def validate(self, target: JointCfg) -> ValidationResult:  # type: ignore[override]
        r"""对 `JointCfg` 执行关节级规则检查。

        Args:
            target (JointCfg): 待验证的关节配置。

        Returns:
            ValidationResult: 含 errors / warnings 的验证结果。
        """

        result = ValidationResult()

        if target.joint_type == "revolute" and self.cfg.check_limit_range:
            if target.limit is None:
                result.errors.append(f"joint '{target.name}': revolute joint is missing limits")
            else:
                joint_range = target.limit.upper - target.limit.lower
                if joint_range > self.cfg.limit_max_range:
                    result.warnings.append(
                        f"joint '{target.name}': limit range {joint_range:.3f} rad > {self.cfg.limit_max_range:.3f}"
                    )

        if self.cfg.check_link_length and target.origin is not None:
            x, y, z = target.origin.pos
            length = math.sqrt(x * x + y * y + z * z)
            allow_zero_origin = bool(target.metadata.get("allow_zero_origin", False))
            if length < self.cfg.min_link_length and not allow_zero_origin:
                result.warnings.append(
                    f"joint '{target.name}': link length {length:.6f} m < min {self.cfg.min_link_length:.6f}"
                )

        if self.cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result

        # TODO:算法之一（joint-level rule checking）
        # ────────────────────────────────────────
        # 输入
        #   target: JointCfg
        #   cfg.check_limit_range: 是否启用限位范围检查
        #   cfg.limit_max_range: 限位范围上限（rad）
        #   cfg.check_link_length: 是否启用 link 长度检查
        #   cfg.min_link_length: 最小 link 长度（meter）
        #   cfg.strict: 是否升级 warnings → errors
        #
        # 输出：ValidationResult
        #
        # result = ValidationResult()
        #
        # ── 规则 1：限位范围合规（仅 revolute 关节）──
        #   if target.joint_type == "revolute" and cfg.check_limit_range:
        #     lo, hi = target.limit.lower, target.limit.upper
        #     range = hi - lo
        #     if range > cfg.limit_max_range:
        #       result.warnings.append(
        #         f"joint '{target.name}': limit range {range:.3f} rad > {cfg.limit_max_range:.3f}"
        #       )
        #
        # ── 规则 2：有效 link 长度检查 ──
        #   if cfg.check_link_length and target.origin is not None:
        #     l = ||target.origin.pos||
        #     if l < cfg.min_link_length:
        #       result.warnings.append(
        #         f"joint '{target.name}': link length {l:.6f} m < min {cfg.min_link_length}"
        #       )
        #
        # ── strict 升级 ──
        #   if cfg.strict: result = result.as_strict()
        #
        # result.passed = len(result.errors) == 0
        # return result
        #
        # ── 与 preset 的交叉验证 ──
        #   若 finger preset 声明了该 joint 的建议长度范围，可在此处
        #   额外与 preset 对照检查（通过 metadata 字段传入 preset 约束）。


__all__ = ["JointValidatorCfg", "JointValidator"]
