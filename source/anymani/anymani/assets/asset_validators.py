"""生成后手部资产的 validator 侧运行时对象。

validation 放在 pipeline 层，而不是塞进每个 schema 类内部，是因为并非
所有检查都同等“基础”。有些检查属于 schema 的内禀约束，有些则属于
当前科研阶段的策略性规则，比如“generator v1 暂时拒绝 mimic joint”。
把这两类都显式放在这里，后续才能按需放松或收紧规则。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg

HandRule = Callable[[HandCfg], None]


def reject_mimic_joints(hand: HandCfg) -> None:
    r"""在 generator v1 中拒绝 mimic joint。

    Args:
        hand (HandCfg): 待检查的手资产。

    Raises:
        NotImplementedError: 当任何 joint 使用 `mimic` 时抛出。
    """

    mimic_joints = [joint.name for joint in hand.iter_joints() if joint.mimic is not None]
    if mimic_joints:
        raise NotImplementedError(
            "Generator v1 does not support automatic mimic-hand generation; "
            f"found mimic joints: {mimic_joints}"
        )


@dataclass
class ValidatorCfg(AssetCfgBase):
    r"""验证器运行时对象的配置。

    validator 同时承担三类检查：

    - 交给 :class:`HandCfg` 的内禀完整性检查；
    - 当前 pipeline 阶段定义的策略性检查；
    - 用户额外注入的规则函数。
    """

    class_type: type["Validator"] | None = None
    """关联的 validator 运行时类。"""

    require_complete: bool = True
    """尚未解析完成的必填字段是否应当直接报错。"""

    reject_mimic: bool = True
    """第一版是否拒绝 mimic joint。"""

    rules: list[HandRule] = field(default_factory=list)
    """额外的自定义验证规则。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = Validator


class Validator:
    r"""用于验证生成结果的基础运行时对象。"""

    def __init__(self, cfg: ValidatorCfg):
        self.cfg = cfg

    def _run_builtin_checks(self, hand: HandCfg) -> None:
        r"""在自定义规则之前运行内置验证。

        Args:
            hand (HandCfg): 待验证的候选手资产。

        Raises:
            ValueError: 当必填字段仍未解析完成时抛出。
            NotImplementedError: 当当前 pipeline 配置拒绝某些特性时抛出，
                例如 v1 中的 mimic joint。
        """

        if self.cfg.require_complete:
            missing = hand.validate()
            if missing:
                raise ValueError(f"HandCfg contains unresolved required fields: {missing}")
        if self.cfg.reject_mimic:
            reject_mimic_joints(hand)

    def validate(self, hand: HandCfg) -> HandCfg:
        r"""验证一个已生成的 hand，并在成功时原样返回。

        Args:
            hand (HandCfg): 候选手资产。

        Returns:
            HandCfg: 原样返回的同一个 hand 对象，便于 pipeline 串联。
        """

        # 从调用者视角看，validation 维持函数式风格：
        # 返回同一个 hand 对象，这样 generator 代码就能保持
        # “构建 -> 验证 -> 导出”的清晰流水线读法。
        self._run_builtin_checks(hand)
        for rule in self.cfg.rules:
            rule(hand)
        return hand


__all__ = ["HandRule", "ValidatorCfg", "Validator", "reject_mimic_joints"]
