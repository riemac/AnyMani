r"""手指级验证规则集：对 FingerCfg 做链式语义和 DOF 约束检查。

schema 层已经保证：
- 手指至少有 1 个 joint；
- 串联链的 parent/child 关系连续；
- finger 内部 joint 名唯一。

这里追加的是"工程合理性"软约束：

1. **最小 DOF 数**：革命关节数量是否不低于建议最小值
2. **Tip 唯一性**：是否恰好只有一个 ``is_tip=True`` 的 joint
3. **每根手指的深度上限**：joint 链长度是否不超过允许最大值
4. **逐关节合规**：对链内每个 joint 调用 JointValidator

设计说明
--------

### 分层调用

`FingerValidator` 内部会实例化 `JointValidator` 并对链内每个关节跑规则，
然后做 finger 级合并。这样 hand 级只需调用 `FingerValidator`，不需要
直接接触 joint 级规则。

### Tip 语义

当前约定：每根手指应有且仅有一个 ``is_tip=True`` 的 joint（末端关节）。
若发现零个或多于一个，记 warning（因为 schema 层没有硬性限制）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, FingerCfg
from ._base import ValidatorBase, ValidationResult
from .joint_rules import JointValidatorCfg, JointValidator


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class FingerValidatorCfg(AssetCfgBase):
    r"""手指级验证规则配置。"""

    class_type: type["FingerValidator"] | None = None
    """关联的运行时类。"""

    min_revolute_dof: int = 1
    """每根手指至少应有的 revolute 关节数；低于此值记 error。"""

    max_joint_depth: int | None = None
    """允许的最大链深度（joint 数量）；为 ``None`` 时不限制；超过记 warning。"""

    max_total_length: float | None = None
    """允许的手指最大总链长（meter）；为 ``None`` 时不检查。
    总链长定义为有效距离之和（从指根到指尖）：
    $L = \\sum_i \\|p_i\\|_2$。
    典型参考值：人手中指约 0.11 m，小型机器人手 0.05~0.09 m。"""

    check_tip_uniqueness: bool = True
    """是否检查每根手指恰好只有一个 ``is_tip=True`` 的末端关节。"""

    joint: JointValidatorCfg = field(default_factory=JointValidatorCfg)
    """关节级验证配置；finger 验证器内部会对每个 joint 跑此配置。"""

    strict: bool = False
    """是否把 warnings 升级为 errors。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerValidator


# ============================================================================
#  运行时壳
# ============================================================================


class FingerValidator(ValidatorBase):
    r"""手指级验证器。

    按 ``cfg`` 中启用的规则对 ``FingerCfg`` 做链式语义 + DOF 约束检查，
    并对链内每个关节调用 ``JointValidator``，合并返回。
    """

    cfg: FingerValidatorCfg

    def __init__(self, cfg: FingerValidatorCfg):
        self.cfg = cfg

    def validate(self, target: FingerCfg) -> ValidationResult:  # type: ignore[override]
        r"""对 `FingerCfg` 执行手指级规则检查。

        Args:
            target (FingerCfg): 待验证的手指配置。

        Returns:
            ValidationResult: 含所有关节级 + 手指级 errors / warnings 的合并结果。
        """

        result = ValidationResult()
        joint_validator = JointValidator(self.cfg.joint)

        for joint in target.joints:
            result.merge(joint_validator.validate(joint))

        revolute_count = sum(1 for joint in target.joints if joint.joint_type == "revolute")
        if revolute_count < self.cfg.min_revolute_dof:
            result.errors.append(
                f"finger '{target.name}': revolute dof {revolute_count} < min {self.cfg.min_revolute_dof}"
            )

        if self.cfg.max_joint_depth is not None and len(target.joints) > self.cfg.max_joint_depth:
            result.warnings.append(
                f"finger '{target.name}': depth {len(target.joints)} > max {self.cfg.max_joint_depth}"
            )

        if self.cfg.max_total_length is not None:
            total_length = sum(
                math.sqrt(joint.origin.pos[0] ** 2 + joint.origin.pos[1] ** 2 + joint.origin.pos[2] ** 2)
                for joint in target.joints
            )
            if total_length > self.cfg.max_total_length:
                result.warnings.append(
                    f"finger '{target.name}': total length {total_length * 100.0:.2f} cm > max "
                    f"{self.cfg.max_total_length * 100.0:.2f} cm"
                )

        if self.cfg.check_tip_uniqueness:
            tip_count = sum(1 for joint in target.joints if joint.is_tip)
            if tip_count != 1:
                result.warnings.append(
                    f"finger '{target.name}': expected exactly 1 tip joint, got {tip_count}"
                )

        if self.cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result

        # TODO:算法之一（finger-level rule checking）
        # ────────────────────────────────────────
        # 输入
        #   target: FingerCfg
        #   cfg: FingerValidatorCfg
        #
        # 输出：ValidationResult（包含 joint 级合并结果）
        #
        # result = ValidationResult()
        # joint_v = JointValidator(cfg.joint)
        #
        # ── 逐关节校验（下沉到 joint 级）──
        #   for joint in target.joints:
        #     jresult = joint_v.validate(joint)
        #     result.merge(jresult)   # 把 joint 错误/警告并入 finger 结果
        #
        # ── 规则 1：最小 revolute DOF ──
        #   revolute_count = sum(1 for j in target.joints if j.joint_type == "revolute")
        #   if revolute_count < cfg.min_revolute_dof:
        #     result.errors.append(
        #       f"finger '{target.name}': revolute dof {revolute_count} < min {cfg.min_revolute_dof}"
        #     )
        #
        # ── 规则 2：链深度上限 ──
        #   if cfg.max_joint_depth is not None and len(target.joints) > cfg.max_joint_depth:
        #     result.warnings.append(
        #       f"finger '{target.name}': depth {len(target.joints)} > max {cfg.max_joint_depth}"
        #     )
        #
        # ── 规则 3：手指总链长上限 ──
        #   if cfg.max_total_length is not None:
        #     total_len = sum(
        #       math.sqrt(sum(x**2 for x in j.origin.pos))
        #       for j in target.joints if j.origin is not None
        #     )  # 即 L = Σ ||p_i||_2
        #     if total_len > cfg.max_total_length:
        #       result.warnings.append(
        #         f"finger '{target.name}': total length {total_len*100:.1f} cm "
        #         f"> max {cfg.max_total_length*100:.1f} cm"
        #       )
        #
        # ── 规则 4：Tip 唯一性 ──
        #   if cfg.check_tip_uniqueness:
        #     tip_count = sum(1 for j in target.joints if j.is_tip)
        #     if tip_count != 1:
        #       result.warnings.append(
        #         f"finger '{target.name}': expected exactly 1 tip joint, got {tip_count}"
        #       )
        #
        # ── strict 升级 ──
        #   if cfg.strict: result = result.as_strict()
        #
        # result.passed = len(result.errors) == 0
        # return result


__all__ = ["FingerValidatorCfg", "FingerValidator"]
