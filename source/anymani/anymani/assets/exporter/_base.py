r"""导出器基础协议：ExportResult + ExporterBase。

每个导出器的职责是把内存中的 `HandCfg`（或子结构）序列化成落盘产物。
这里不规定产物格式，只规定接口约定和结果容器。

设计说明
--------

### ExportResult 的设计

导出结果记录：

- ``written``：已成功写入的文件路径列表
- ``skipped``：因已存在且 ``overwrite=False`` 而跳过的路径列表
- ``errors``：写入失败的 (路径, 异常消息) 列表

这三个字段允许调用方做"不打断流程的失败记录"——某张 sidecar 写失败时，
URDF 已经写好的那部分不应丢失。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportResult:
    r"""一次导出调用的结果包。

    支持 ``merge()`` 串联多个子导出器的结果。
    """

    written: list[Path] = field(default_factory=list)
    """成功写入的文件路径列表。"""

    skipped: list[Path] = field(default_factory=list)
    """跳过的文件路径列表（已存在且不允许覆盖）。"""

    errors: list[tuple[Path, str]] = field(default_factory=list)
    """写入失败的 (路径, 错误消息) 列表。"""

    def merge(self, other: "ExportResult") -> "ExportResult":
        r"""就地合并另一个 ExportResult，返回自身。

        Args:
            other (ExportResult): 待合并结果。

        Returns:
            ExportResult: 合并后的自身（支持链式调用）。
        """

        self.written.extend(other.written)
        self.skipped.extend(other.skipped)
        self.errors.extend(other.errors)
        return self

    @property
    def ok(self) -> bool:
        """若 ``errors`` 为空则为 ``True``。"""

        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.ok


class ExporterBase:
    r"""所有导出器的最小基类。

    子类实现 ``export()``，返回 ``ExportResult`` 而不是直接抛出异常，
    允许调用方选择如何处理失败。
    """

    def export(self, target: object, output_dir: Path) -> ExportResult:
        r"""把 ``target`` 序列化到 ``output_dir`` 下的文件。

        Args:
            target (object): 待导出的资产（通常为 HandCfg 或其子结构）。
            output_dir (Path): 产物落盘目录。

        Returns:
            ExportResult: 含写入/跳过/错误路径的结果包。
        """

        raise NotImplementedError


__all__ = ["ExportResult", "ExporterBase"]
