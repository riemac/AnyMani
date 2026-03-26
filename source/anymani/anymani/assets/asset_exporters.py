"""生成后手部资产的 exporter 侧运行时对象。

exporter v1 刻意保持保守。当前阶段项目还在收敛“生成语义”，所以
exporter 代码主要服务于检查、调试和轻量序列化，而不是过早把项目锁进
一套很重的 URDF 写出抽象栈里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg


@dataclass
class ExporterCfg(AssetCfgBase):
    r"""导出器运行时对象的配置。

    当前 exporter 的目标很简单：把 canonical :class:`HandCfg`
    转成普通 Python 字典，并在需要时额外落一个 JSON 方便调试或记录实验。
    """

    class_type: type["Exporter"] | None = None
    """关联的 exporter 运行时类。"""

    output_dir: str | None = None
    """可选输出目录。"""

    dump_json: bool = False
    """是否额外导出一份 `HandCfg` 的 JSON 快照。"""

    json_file_name: str = "hand.json"
    """在 `dump_json=True` 时使用的输出文件名。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = Exporter


class Exporter:
    r"""用于序列化生成结果的基础运行时对象。"""

    def __init__(self, cfg: ExporterCfg):
        self.cfg = cfg

    def serialize(self, hand: HandCfg) -> dict[str, Any]:
        r"""把 hand 序列化成普通字典。

        Args:
            hand (HandCfg): canonical hand schema 对象。

        Returns:
            dict[str, Any]: 转换后的嵌套映射。
        """

        return hand.to_dict()

    def export(self, hand: HandCfg) -> dict[str, Any]:
        r"""根据 exporter 配置导出 hand。

        v1 的 exporter 行为刻意保持克制：
        其主要工作是序列化 `HandCfg`，并可选地写出一份 JSON 调试产物。
        """

        payload = self.serialize(hand)
        if self.cfg.dump_json and self.cfg.output_dir is not None:
            output_dir = Path(self.cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.cfg.json_file_name
            # 落盘的 JSON 主要用于人工检查和实验留痕。
            # 它还不是最终的 URDF / SDF 权威导出路径。
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload


__all__ = ["ExporterCfg", "Exporter"]
