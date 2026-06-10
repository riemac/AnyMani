r"""URDF visual recolor helper。

这层 helper 的职责，不是“真正写出 XML”，而是把用户在 `HandGeneratorCfg.recolored`
里写下的高层科研意图，lower 成 URDF writer 能直接消费的
`link_name -> MaterialCfg` 映射。

之所以把它放在 `generator/` 而不是 `exporter/`，是因为它本质上属于
“生成阶段的附加语义决策”：

- exporter 只负责把已经确定好的 material 写成 `<visual><material><color .../>`
- 哪个 link 应该染成什么颜色，则由 generator 结合 hand topology 决定
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from ...asset_base import HandCfg
from ...asset_schema_core import MaterialCfg, _ensure_tuple
from ...presets.color_presets import COLOR_PRESETS, DEFAULT_COLOR_PRESET_NAME

RgbaTuple: TypeAlias = tuple[float, float, float, float]
RecolorSpec: TypeAlias = str | dict[str, RgbaTuple] | bool | None


def normalize_recolor_spec(recolored: Any) -> RecolorSpec:
    r"""规范化 `HandGeneratorCfg.recolored`。

    当前对外 contract 按已确认的三档收敛：

    - `None` / `False`：关闭
    - `str`：命名 palette
    - `dict[child_link_name, rgba]`：按 link 名做局部覆盖

    此外，为了兼容这个字段早期作为“布尔开关草稿”存在的历史，`True` 在这里被
    收敛为默认 palette `anatomy_soft_v1`。这样老脚本若先写了 `recolored=True`，
    会直接进入当前正式使用的柔和 anatomy 调色 contract。
    """

    if recolored is None or recolored is False:
        return None
    if recolored is True:
        return DEFAULT_COLOR_PRESET_NAME
    if isinstance(recolored, str):
        normalized_name = recolored.strip()
        if not normalized_name:
            raise ValueError("recolored preset name cannot be empty")
        return normalized_name
    if isinstance(recolored, Mapping):
        normalized: dict[str, RgbaTuple] = {}
        for link_name, rgba in recolored.items():
            normalized_name = str(link_name).strip()
            if not normalized_name:
                raise ValueError("recolored override contains an empty child-link name")
            packed = _ensure_tuple(rgba, length=4, field_name=f"recolored[{normalized_name!r}]")
            normalized[normalized_name] = cast(RgbaTuple, tuple(float(value) for value in packed))
        return normalized
    raise TypeError(
        "recolored must be None/False, a palette name string, or a dict[child_link_name, rgba]; "
        f"got {type(recolored).__name__}"
    )


def describe_recolor_spec(recolored: RecolorSpec) -> dict[str, Any] | None:
    r"""给 sidecar / debug 元数据生成一份紧凑摘要。"""

    normalized = normalize_recolor_spec(recolored)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return {"mode": "preset", "preset": normalized}
    if isinstance(normalized, dict):
        return {"mode": "overrides", "links": sorted(normalized)}
    raise TypeError(f"Unexpected normalized recolor payload: {normalized!r}")


def resolve_visual_recolor_materials(hand_cfg: HandCfg, recolored: RecolorSpec) -> dict[str, MaterialCfg]:
    r"""把高层 recolor 语义解析成 `link_name -> MaterialCfg` 映射。

    返回的键是**最终要写进 URDF `<link name="...">` 的 link 名**，这样 exporter
    只需要在写 `<visual>` 时查一次当前 link 即可，不必再理解 anatomy 规则。
    """

    normalized = normalize_recolor_spec(recolored)
    if normalized is None:
        return {}
    if isinstance(normalized, str):
        palette = _resolve_named_palette(normalized)
        return {
            link_name: _make_material(link_name=link_name, rgba=palette_rgba)
            for link_name, palette_rgba in _resolve_named_palette_targets(hand_cfg, palette).items()
        }
    if isinstance(normalized, dict):
        return {
            link_name: _make_material(link_name=link_name, rgba=rgba)
            for link_name, rgba in normalized.items()
        }
    raise TypeError(f"Unexpected normalized recolor payload: {normalized!r}")


def _resolve_named_palette(preset_name: str) -> dict[str, RgbaTuple]:
    r"""按名字取稳定 palette 数据。"""

    if preset_name not in COLOR_PRESETS:
        raise ValueError(
            f"Unknown recolored palette {preset_name!r}; available presets are {sorted(COLOR_PRESETS)!r}"
        )
    return dict(COLOR_PRESETS[preset_name])


def _resolve_named_palette_targets(hand_cfg: HandCfg, palette: Mapping[str, RgbaTuple]) -> dict[str, RgbaTuple]:
    r"""把语义 palette 展开成当前 hand 的具体 link 名。

    这里最重要的分工是：

    - palette 只知道 `palm / mcp1 / cmc2 / tip` 这种**语义类别**
    - 当前 helper 负责把这些语义类别映射到具体 hand 的真实 link 名
    """

    resolved: dict[str, RgbaTuple] = {}
    if "palm" in palette:
        resolved[hand_cfg.palm.name] = palette["palm"]  # palm 名不一定永远字面等于 `"palm"`，因此要从 hand cfg 读真值

    for finger in hand_cfg.fingers:
        for joint in finger.joints:
            semantic_key = _infer_semantic_color_key(str(joint.child))
            if semantic_key is None or semantic_key not in palette:
                continue
            resolved[str(joint.child)] = palette[semantic_key]
    return resolved


def _infer_semantic_color_key(link_name: str) -> str | None:
    r"""从 link 名里反推出 anatomy 语义类别。"""

    if link_name.endswith("_root_fixed_link"):
        return "root_fixed"

    for semantic_key in ("cmc1", "cmc2", "mcp1", "mcp2", "mcp", "pip", "dip", "tip"):
        if link_name.endswith(f"_{semantic_key}"):
            return semantic_key
    return None


def _make_material(*, link_name: str, rgba: RgbaTuple) -> MaterialCfg:
    r"""把一个 RGBA 元组包装成稳定的 `MaterialCfg`。"""

    return MaterialCfg(
        name=f"{link_name}_recolor",
        rgba=rgba,
    )


__all__ = [
    "RgbaTuple",
    "RecolorSpec",
    "normalize_recolor_spec",
    "describe_recolor_spec",
    "resolve_visual_recolor_materials",
]
