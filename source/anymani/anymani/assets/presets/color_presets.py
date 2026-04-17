r"""recolored 调色盘数据表。

本文件只保存**稳定的颜色数据**，不承担运行时“怎么把颜色分配到哪条 link”
这类逻辑。真正的 lowering / 推断 / 覆盖规则统一放在 `generator/_recolor.py`，
这样可以把：

- `presets/`：稳定数据
- `generator/`：运行时决策

这两层职责明确拆开，避免又把科研语义和流程逻辑揉在一起。
"""

from __future__ import annotations


DEFAULT_COLOR_PRESET_NAME = "anatomy_v1"
"""默认的 anatomy 语义调色盘名。

当前对应你已经确认的首轮规则：

- palm / LEAP fixed root：红
- CMC1 / MCP1：黄
- CMC2 / MCP2：青
- PIP：绿
- DIP：蓝
- TIP：紫
"""


COLOR_PRESETS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "anatomy_v1": {
        "palm": (1.0, 0.0, 0.0, 1.0),
        "root_fixed": (1.0, 0.0, 0.0, 1.0),
        "cmc1": (1.0, 1.0, 0.0, 1.0),
        "mcp1": (1.0, 1.0, 0.0, 1.0),
        "cmc2": (0.0, 1.0, 1.0, 1.0),
        "mcp2": (0.0, 1.0, 1.0, 1.0),
        "pip": (0.0, 1.0, 0.0, 1.0),
        "dip": (0.0, 0.0, 1.0, 1.0),
        "tip": (1.0, 0.0, 1.0, 1.0),
    },
}
"""palette_name -> semantic part -> RGBA 的稳定数据表。"""


__all__ = ["COLOR_PRESETS", "DEFAULT_COLOR_PRESET_NAME"]
