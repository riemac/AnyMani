"""手部资产生成的 Validator 层空骨架。

本文件只保留你主导的验证器框架，不再默认内置验证规则。尤其是：

- 不再默认拒绝 mimic joint
- 不再默认把 `HandCfg.validate()` 包装成运行时流程
- 不再替你预设“物理合理性”具体由哪些规则组成

这里现在只负责声明：未来验证算法将被分布在 joint / finger / palm / hand
四个层级里。
"""

from __future__ import annotations

from dataclasses import dataclass

from .asset_base import AssetCfgBase, FingerCfg, HandCfg, JointCfg, PalmCfg


@dataclass
class ValidatorCfg(AssetCfgBase):
    r"""验证器配置基类。"""

    class_type: type["Validator"] | None = None
    """关联的验证器运行时类。"""


@dataclass
class JointValidatorCfg(ValidatorCfg):
    r"""关节级验证器配置。"""

    class_type: type["Validator"] | None = None
    """关联的关节级验证器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointValidator


@dataclass
class FingerValidatorCfg(ValidatorCfg):
    r"""手指级验证器配置。"""

    class_type: type["Validator"] | None = None
    """关联的手指级验证器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerValidator


@dataclass
class PalmValidatorCfg(ValidatorCfg):
    r"""掌级验证器配置。"""

    class_type: type["Validator"] | None = None
    """关联的掌级验证器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmValidator


@dataclass
class HandValidatorCfg(ValidatorCfg):
    r"""手级验证器配置。"""

    class_type: type["Validator"] | None = None
    """关联的手级验证器类。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandValidator


class Validator:
    r"""验证器基类。

    这里的 `validate()` 只保留接口，表示“这里将来会放置物理合理性规则”，
    但当前不替你决定任何默认规则。
    """

    cfg: ValidatorCfg

    def __init__(self, cfg: ValidatorCfg):
        self.cfg = cfg

    def validate(self, target: AssetCfgBase) -> None:
        r"""验证一个资产对象。

        Args:
            target (AssetCfgBase): 待验证资产。

        Raises:
            NotImplementedError: 当前只是规则入口骨架，尚未填入真实实现。
        """

        raise NotImplementedError("Validator 骨架已保留，但具体验证规则需后续实现。")


class JointValidator(Validator):
    r"""关节级验证器。"""

    def __init__(self, cfg: JointValidatorCfg):
        super().__init__(cfg)

    def validate(self, target: AssetCfgBase) -> None:
        r"""验证一个 `JointCfg`。

        Args:
            target (AssetCfgBase): 待验证资产，预期应为 `JointCfg`。

        Raises:
            NotImplementedError: joint-level 验证规则尚未实现。
        """

        raise NotImplementedError("JointValidator 目前只保留骨架，等待 joint-level 规则实现。")


class FingerValidator(Validator):
    r"""手指级验证器。"""

    def __init__(self, cfg: FingerValidatorCfg):
        super().__init__(cfg)

    def validate(self, target: AssetCfgBase) -> None:
        r"""验证一个 `FingerCfg`。

        Args:
            target (AssetCfgBase): 待验证资产，预期应为 `FingerCfg`。

        Raises:
            NotImplementedError: finger-level 验证规则尚未实现。
        """

        raise NotImplementedError("FingerValidator 目前只保留骨架，等待 finger-level 规则实现。")


class PalmValidator(Validator):
    r"""掌级验证器。"""

    def __init__(self, cfg: PalmValidatorCfg):
        super().__init__(cfg)

    def validate(self, target: AssetCfgBase) -> None:
        r"""验证一个 `PalmCfg`。

        Args:
            target (AssetCfgBase): 待验证资产，预期应为 `PalmCfg`。

        Raises:
            NotImplementedError: palm-level 验证规则尚未实现。
        """

        raise NotImplementedError("PalmValidator 目前只保留骨架，等待 palm-level 规则实现。")


class HandValidator(Validator):
    r"""手级验证器。"""

    def __init__(self, cfg: HandValidatorCfg):
        super().__init__(cfg)

    def validate(self, target: AssetCfgBase) -> None:
        r"""验证一个 `HandCfg`。

        Args:
            target (AssetCfgBase): 待验证资产，预期应为 `HandCfg`。

        Raises:
            NotImplementedError: hand-level 验证规则尚未实现。
        """

        raise NotImplementedError("HandValidator 目前只保留骨架，等待 hand-level 规则实现。")


__all__ = [
    "ValidatorCfg",
    "JointValidatorCfg",
    "FingerValidatorCfg",
    "PalmValidatorCfg",
    "HandValidatorCfg",
    "Validator",
    "JointValidator",
    "FingerValidator",
    "PalmValidator",
    "HandValidator",
]
