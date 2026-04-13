"""preset quick-check 预览脚本子包。

这里收纳的是“为了调 preset 而快速生成可巡检 URDF”的脚本入口，而不是正式生成链。
把它们放在 `assets/presets/preview/`，是为了让工具和它服务的数据层尽量邻近：

- finger / palm / mount preset 在 `assets/presets/`
- quick-check 入口也放在 `assets/presets/preview/`

这样未来即使把 `assets` 单独抽成一个 Python 包，这部分依然可以整体带走。
"""

__all__: list[str] = []
