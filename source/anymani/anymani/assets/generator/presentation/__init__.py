r"""generator 展示层入口。

本子包只负责“怎样把已经生成的资产解释给人看”，例如 recolor lowering 和
ASCII tree 渲染，不介入生成主流程和物理变异。
"""

from .recolor import RecolorSpec, describe_recolor_spec, normalize_recolor_spec, resolve_visual_recolor_materials
from .tree_render import render_hand_tree_txt

__all__ = [
    "RecolorSpec",
    "describe_recolor_spec",
    "normalize_recolor_spec",
    "resolve_visual_recolor_materials",
    "render_hand_tree_txt",
]
