r"""手部资产生成器子包。

本子包的定位是用户直接面向的生成入口层：

- `hand_generator` 负责整手级生成调度
- `mutate` 负责后序变异工具壳

这里不承载具体实现，只作为稳定的包级入口与导出点。
"""

from .hand_generator import (
    HandGenerationResult,
    HandGenerator,
    HandGeneratorCfg,
    render_hand_tree_mermaid,
    render_hand_tree_txt,
)

__all__ = [
    "HandGenerationResult",
    "HandGenerator",
    "HandGeneratorCfg",
    "render_hand_tree_txt",
    "render_hand_tree_mermaid",
]