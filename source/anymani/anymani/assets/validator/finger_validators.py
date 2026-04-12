r"""手指级验证器的声明式配置类和运行时类。

finger-level 验证主要围绕串联链的连续性、角色顺序和末端语义展开。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_validators import ValidationReport, Validator, ValidatorCfg


@dataclass
class FingerValidatorCfg(ValidatorCfg):
    r"""手指级验证器配置。"""

    class_type: type["FingerValidator"] | None = None
    """关联的手指级验证器类。"""

    min_joint_count: int = 1
    """手指最少应包含的 joint 数量。默认 1。"""

    check_chain_continuity: bool = True
    """是否检查 joint 串联链连续性。"""

    check_name_uniqueness: bool = True
    """是否检查 finger 内部 joint 名称唯一性。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerValidator


class FingerValidator(Validator):
    r"""手指级验证器。"""

    cfg: FingerValidatorCfg

    def __init__(self, cfg: FingerValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> ValidationReport | None:
        r"""验证一个手指级资产对象。"""
        pass

        # TODO:算法之一（finger chain 一致性检查）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `FingerCfg`。
        # 输出：`ValidationReport`。
        #
        # ── fatal invariant ──
        #   1. joint 数量是否至少满足 `min_joint_count`。
        #   2. joint.parent 与上一 joint.child 是否连续。
        #   3. finger 内部 joint 名称是否唯一。
        #
        # ── 软诊断 ──
        #   1. tip role 是否可被稳定识别。
        #   2. finger 内关节轴序列是否与 preset family 的建模直觉冲突。
        #
        # IDEA：finger-level validator 是 pre/post 的重要汇合点，因为 delete/regroup 后的第一层显式拓扑风险就在这里暴露。


__all__ = ["FingerValidatorCfg", "FingerValidator"]