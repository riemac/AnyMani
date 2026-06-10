r"""recolored 调色盘数据表。

本文件只保存**稳定的颜色数据**，不承担运行时“怎么把颜色分配到哪条 link”
这类逻辑。真正的 lowering / 推断 / 覆盖规则统一放在 `generator/presentation/recolor.py`，
这样可以把：

- `presets/`：稳定数据
- `generator/`：运行时决策

这两层职责明确拆开，避免又把科研语义和流程逻辑揉在一起。
"""

from __future__ import annotations

DEFAULT_COLOR_PRESET_NAME = "anatomy_soft_v1"
"""默认的 anatomy 语义调色盘名。

当前对应你已经确认的柔和 anatomy 调色规则：

- palm / LEAP fixed root：红
- CMC1 / MCP1：黄
- CMC2 / MCP2：青
- thumb MCP：绿（与 non-thumb PIP 同属第三活动段的 visual cue）
- PIP：绿
- DIP：紫，用来区分第二段靛蓝和真正 distal link
- TIP：乳白，统一 custom tip 与 procedural `cs` 的接触皮肤视觉语义

# NOTE:
旧的高饱和 `anatomy_v1` 已出清。当前实验入口和 `recolored=True` 都收敛到
`anatomy_soft_v1`，避免维护两套实际不再使用的视觉 contract。
"""


COLOR_PRESETS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "anatomy_soft_v1": {
        # 这组颜色保留 anatomy 分段语义，但降低饱和度与明度。
        # 直观效果是“红仍然是 palm，黄/青/绿仍标示近端关节段”，但不再像显示器 RGB 原色那样刺眼。
        "palm": (0.6039215686274509, 0.14901960784313725, 0.14901960784313725, 1.0),
        "root_fixed": (0.6039215686274509, 0.14901960784313725, 0.14901960784313725, 1.0),
        "cmc1": (0.8666666666666667, 0.8666666666666667, 0.050980392156862744, 1.0),
        "mcp1": (0.8666666666666667, 0.8666666666666667, 0.050980392156862744, 1.0),
        # cmc2 / mcp2 使用深青色，保留“第二活动段”的青色 cue，同时避免纯青色在 viewer 里过亮。
        "cmc2": (0.047058823529411764, 0.4392156862745098, 0.48627450980392156, 1.0),
        "mcp2": (0.047058823529411764, 0.4392156862745098, 0.48627450980392156, 1.0),
        "mcp": (0.043137254901960784, 0.3215686274509804, 0.2235294117647059, 1.0),
        "pip": (0.043137254901960784, 0.3215686274509804, 0.2235294117647059, 1.0),
        "dip": (0.35294117647058826, 0.23137254901960785, 0.4470588235294118, 1.0),
        "tip": (0.92, 0.88, 0.78, 1.0),
    },
}
"""palette_name -> semantic part -> RGBA 的稳定数据表。"""


__all__ = ["COLOR_PRESETS", "DEFAULT_COLOR_PRESET_NAME"]
