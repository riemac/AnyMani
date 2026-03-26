"""手部资产生成的 Exporter 层空骨架。

这一层只保留“导出职责”本身，不再内置任何默认导出行为。当前不再默认：

- 把 `HandCfg` 序列化成 JSON
- 自动落盘调试文件
- 替你决定正式导出的 URDF / yaml / 附带资产格式细节

这里保留四个导出层级，服务你后续自己定义的导出策略：

- `JointExporter`
- `FingerExporter`
- `PalmExporter`
- `HandExporter`
"""

from __future__ import annotations

from .asset_base import AssetCfgBase
from .asset_base import FingerCfg, HandCfg, JointCfg, PalmCfg


class Exporter:
    r"""导出器基类。

    这里的 `export()` 只说明“从资产配置导出外部文件或附带资产”这一职责，
    但不再提供默认实现。
    """

    def __init__(self):
        pass

    def export(self, target: AssetCfgBase) -> None:
        r"""导出资产对象。

        Args:
            target (AssetCfgBase): 待导出的资产对象。

        Raises:
            NotImplementedError: 当前只是导出入口骨架，尚未填入真实实现。
        """

        raise NotImplementedError("Exporter 骨架已保留，但具体导出逻辑需后续实现。")


class JointExporter(Exporter):
    r"""关节级导出器。

    预期职责：

    - 从 `JointCfg` 导出自包含的最小调试资产
    - 在需要时生成快速检验用的局部 URDF / yaml / 附带资源
    """

    def __init__(self):
        super().__init__()

    def export(self, target: AssetCfgBase) -> None:
        r"""导出一个 `JointCfg`。

        Args:
            target (AssetCfgBase): 待导出的资产对象，预期应为 `JointCfg`。

        Raises:
            NotImplementedError: 关节级导出逻辑尚未实现。
        """

        raise NotImplementedError("JointExporter 目前只保留骨架，等待 joint-level 导出实现。")


class FingerExporter(Exporter):
    r"""手指级导出器。

    预期职责：

    - 从 `FingerCfg` 导出 finger 级自包含资产
    - 在需要时生成快速检验用 finger-level URDF
    """

    def __init__(self):
        super().__init__()

    def export(self, target: AssetCfgBase) -> None:
        r"""导出一个 `FingerCfg`。

        Args:
            target (AssetCfgBase): 待导出的资产对象，预期应为 `FingerCfg`。

        Raises:
            NotImplementedError: 手指级导出逻辑尚未实现。
        """

        raise NotImplementedError("FingerExporter 目前只保留骨架，等待 finger-level 导出实现。")


class PalmExporter(Exporter):
    r"""掌级导出器。

    预期职责：

    - 从 `PalmCfg` 导出掌级自包含资产
    - 在需要时生成快速检验用 palm-level URDF
    """

    def __init__(self):
        super().__init__()

    def export(self, target: AssetCfgBase) -> None:
        r"""导出一个 `PalmCfg`。

        Args:
            target (AssetCfgBase): 待导出的资产对象，预期应为 `PalmCfg`。

        Raises:
            NotImplementedError: 掌级导出逻辑尚未实现。
        """

        raise NotImplementedError("PalmExporter 目前只保留骨架，等待 palm-level 导出实现。")


class HandExporter(Exporter):
    r"""手级导出器。

    预期职责：

    - 从 `HandCfg` 导出正式 URDF / yaml / 附带资产
    - 在需要时导出整手级自包含 URDF 用于快速检验
    """

    def __init__(self):
        super().__init__()

    def export(self, target: AssetCfgBase) -> None:
        r"""导出一个 `HandCfg`。

        Args:
            target (AssetCfgBase): 待导出的资产对象，预期应为 `HandCfg`。

        Raises:
            NotImplementedError: 手级导出逻辑尚未实现。
        """

        raise NotImplementedError("HandExporter 目前只保留骨架，等待 hand-level 导出实现。")


__all__ = [
    "Exporter",
    "JointExporter",
    "FingerExporter",
    "PalmExporter",
    "HandExporter",
]
