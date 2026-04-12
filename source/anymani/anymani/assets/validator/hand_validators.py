r"""整手级验证器的声明式配置类和运行时类。

hand-level validator 负责汇总 joint/finger/palm 三层结果，并输出真正对 generator 有决策意义的
fatal reject 与 soft report。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_validators import ValidationReport, Validator, ValidatorCfg


@dataclass
class HandValidatorCfg(ValidatorCfg):
    r"""手级验证器配置。"""

    class_type: type["HandValidator"] | None = None
    """关联的手级验证器类。"""

    min_finger_count: int = 3
    """整手至少应保留的 finger 数量。默认 3。"""

    min_inter_finger_gap_cm: float = 1.0
    """相邻手指根部的最小推荐间距（厘米）。作为软阈值报告。"""

    emit_soft_report: bool = True
    """是否输出 warning/info 级别的软报告。默认开启。"""

    check_representation_readiness: bool = True
    """是否检查 hand 是否满足后续 joint-centric / SE(3) 表征前提。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandValidator


class HandValidator(Validator):
    r"""手级验证器。"""

    cfg: HandValidatorCfg

    def __init__(self, cfg: HandValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> ValidationReport | None:
        r"""验证一个整手级资产对象。"""
        pass

        # TODO:算法之一（hard error + soft report 汇总）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `HandCfg`。
        # 输出：`ValidationReport`。
        #
        # ── fatal invariant ──
        #   1. finger 数量是否至少满足 `min_finger_count`。
        #   2. 全局 joint/link/finger 名称是否唯一。
        #   3. 结构是否仍可被 exporter 稳定线性化。
        #
        # ── 软诊断 ──
        #   1. finger 根部间距是否小于推荐阈值。
        #   2. handedness / preset family / mount baseline 是否互相冲突。
        #   3. 是否满足后续 joint-centric Graph / relative SE(3) 表征前提。
        #
        # IDEA：hand-level validator 的报告要足够结构化，便于 generator 记录 reject reason 分布，而不是只返回 True/False。


__all__ = ["HandValidatorCfg", "HandValidator"]