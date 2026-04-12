r"""关节级验证器的声明式配置类和运行时类。

关节级验证器负责最底层的结构与几何一致性检查，是后续 finger / hand 级报告的
最小诊断单元。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_validators import ValidationReport, Validator, ValidatorCfg


@dataclass
class JointValidatorCfg(ValidatorCfg):
    r"""关节级验证器配置。"""

    class_type: type["JointValidator"] | None = None
    """关联的关节级验证器类。"""

    check_collision_frame_union: bool = True
    """是否检查 child link 的 collision 是否在同一局部系下可并集查询。"""

    check_axis_alignment: bool = True
    """是否检查 joint axis 与旋转中心/局部几何语义是否一致。"""

    reject_negative_geometry: bool = True
    """是否把负尺寸或非法 primitive 参数视为 fatal。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointValidator


class JointValidator(Validator):
    r"""关节级验证器。"""

    cfg: JointValidatorCfg

    def __init__(self, cfg: JointValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> ValidationReport | None:
        r"""验证一个关节级资产对象。"""
        pass

        # TODO:算法之一（joint-level fatal invariant + soft report）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `JointCfg`。
        # 输出：`ValidationReport`。
        #
        # ── fatal invariant ──
        #   1. joint_type、axis、limit、child link 名称是否自洽。
        #   2. primitive 参数是否为正，mesh 路径是否非空。
        #
        # ── 软诊断 ──
        #   1. child collision origins 是否在同一局部系下易于并集查询。
        #   2. axis / origin / child geometry 的相对关系是否符合 joint-centric 建模直觉。
        #
        # IDEA：joint-level validator 不负责整条 finger 的 chain 连续性，那是 finger-level 职责。


__all__ = ["JointValidatorCfg", "JointValidator"]