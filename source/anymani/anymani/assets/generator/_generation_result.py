"""生成结果包定义。

这个文件只负责承载 `HandGenerationResult`，不再混入生成主流程本身。

之所以单独拆文件，是因为结果包本质上是一个“跨生成 / 导出 / 展示层共享的数据壳”，
它既不属于：

- pre-made legality
- connectivity lower
- batch orchestration

也不应该继续占据 `hand_generator.py` 的阅读注意力。

# NOTE:
`render_trees()` 仍保留在结果包上，是因为从使用体验上看：

```python
result.render_trees()
```

这种写法比“再去找另一个工具模块手动调用渲染函数”更顺手。
但它内部会延迟导入树渲染工具，避免把主 façade 和展示层再度缠回一起。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..asset_base import HandCfg


@dataclass
class HandGenerationResult:
    r"""一次生成调用的轻量结果包。

    这个结果包的设计目标是“按需承载产物”，而不是强迫每次都生成完整产物链。
    若用户只想看结构，可以只填 `hand_cfg`；若用户想落盘，则可以同时填：

    - `urdf_path`
    - `sidecar_path`

    此外，为了科研排障方便，结果包也允许按需缓存：

    - ASCII 树
    """

    hand_cfg: HandCfg | None = None
    """内存中的手部配置；轻量模式下可直接返回。"""

    urdf_path: Path | None = None
    """导出的 URDF 路径；若未请求导出则为 `None`。"""

    sidecar_path: Path | None = None
    """附带元数据文件路径；例如 yaml / json sidecar。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """生成过程的辅助信息，例如 preset 名、随机种子、拒绝原因统计等。"""

    tree_txt: str | None = None
    """ASCII 树状可视化；通过 `render_trees()` 填充，也可落盘为 `.txt` 文件。"""

    def render_trees(self) -> "HandGenerationResult":
        r"""从 `self.hand_cfg` 就地生成 ASCII 树状可视化，并返回自身。

        Returns:
            HandGenerationResult: 返回自身，便于链式写法：
                `result.render_trees()`
        """

        if self.hand_cfg is not None:
            # 延迟导入展示层工具，避免把结果包定义再次和 `hand_generator.py`
            # 的主调度逻辑缠回一个文件。
            from ._tree_render import render_hand_tree_txt

            self.tree_txt = render_hand_tree_txt(self.hand_cfg)  # 终端友好的 ASCII 树
        return self


__all__ = [
    "HandGenerationResult",
]
