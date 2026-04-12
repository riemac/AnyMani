r"""掌级验证器的声明式配置类和运行时类。

palm-level 验证主要围绕 design frame、collision-first 语义和 finger mount baseline。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_validators import ValidationReport, Validator, ValidatorCfg


@dataclass
class PalmValidatorCfg(ValidatorCfg):
    r"""掌级验证器配置。"""

    class_type: type["PalmValidator"] | None = None
    """关联的掌级验证器类。"""

    require_collision_first: bool = True
    """是否要求 palm 优先满足 collision-first 语义。"""

    check_mount_baseline: bool = True
    """是否检查 finger mount baseline 的存在性和基本自洽性。"""

    preserve_design_frame: bool = True
    """是否把 design frame 稳定性视为验证项。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmValidator


class PalmValidator(Validator):
    r"""掌级验证器。"""

    cfg: PalmValidatorCfg

    def __init__(self, cfg: PalmValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> ValidationReport | None:
        r"""验证一个掌级资产对象。"""
        pass

        # TODO:算法之一（palm-level 设计帧与挂载基准检查）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `PalmCfg`。
        # 输出：`ValidationReport`。
        #
        # ── fatal invariant ──
        #   1. palm 名称、origin、collision / visual 列表是否可被 exporter 稳定消费。
        #
        # ── 软诊断 ──
        #   1. 是否满足 collision-first。
        #   2. metadata 中是否有可供 hand-level regroup / export 使用的 mount baseline。
        #   3. design frame 是否与 preset 约定一致，避免 frame 语义漂移。
        #
        # IDEA：掌级验证器不直接计算 finger 间距；那是 hand-level 的组合问题。


__all__ = ["PalmValidatorCfg", "PalmValidator"]