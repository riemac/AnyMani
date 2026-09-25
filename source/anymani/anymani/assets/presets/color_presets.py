"Defines visual-only palettes. The active anatomy palette is anatomy_soft_v1; colors do not change collision geometry or physical identity."

from __future__ import annotations

DEFAULT_COLOR_PRESET_NAME = "anatomy_soft_v1"
"Default visualization palette name."


COLOR_PRESETS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "anatomy_soft_v1": {


        "palm": (0.6039215686274509, 0.14901960784313725, 0.14901960784313725, 1.0),
        "root_fixed": (0.6039215686274509, 0.14901960784313725, 0.14901960784313725, 1.0),
        "cmc1": (0.8666666666666667, 0.8666666666666667, 0.050980392156862744, 1.0),
        "mcp1": (0.8666666666666667, 0.8666666666666667, 0.050980392156862744, 1.0),

        "cmc2": (0.047058823529411764, 0.4392156862745098, 0.48627450980392156, 1.0),
        "mcp2": (0.047058823529411764, 0.4392156862745098, 0.48627450980392156, 1.0),
        "mcp": (0.043137254901960784, 0.3215686274509804, 0.2235294117647059, 1.0),
        "pip": (0.043137254901960784, 0.3215686274509804, 0.2235294117647059, 1.0),
        "dip": (0.35294117647058826, 0.23137254901960785, 0.4470588235294118, 1.0),
        "tip": (0.92, 0.88, 0.78, 1.0),
    },
}
"Named visual palettes; these values do not change collision geometry."


__all__ = ["COLOR_PRESETS", "DEFAULT_COLOR_PRESET_NAME"]
