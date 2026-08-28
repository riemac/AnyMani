r"""Generated-hand URDF debug color 的 USD material restore。

该路径只服务 GUI/render，不改变 collision、mass、joint、drive 或 root pose。训练默认不启用；当
``restore_visual_materials=True`` 时，adapter 从 reference URDF 预计算 topology-level visual plan，
spawn wrapper 把颜色绑定到可编辑 ``/<link>/visuals`` ancestor，避免遍历 instance proxies。
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils

from anymani.assets.bank import UrdfRgba
from anymani.assets.bank.urdf_utils import parse_urdf_visual_rgba_by_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualMaterialRestorePlan:
    r"""同拓扑 hand selection 共享的 URDF visual/link/color 恢复计划。"""

    source_urdf_path: Path
    """提供 visual/link/color contract 的 reference ``hand.urdf``。"""

    visual_rgba_by_name: dict[str, UrdfRgba]
    """URDF visual name→RGBA；仅服务 GUI/debug。"""

    visual_link_by_name: dict[str, str]
    """URDF visual name→parent link；定位 spawned USD ``/<link>/visuals``。"""


def serialize_visual_material_restore_plan(plan: VisualMaterialRestorePlan) -> dict[str, object]:
    r"""把计划降为 ``UrdfFileCfg.to_dict()`` 可 JSON-hash 的容器。"""

    return {
        "source_urdf_path": str(plan.source_urdf_path),
        "visual_rgba_by_name": {visual_name: list(rgba) for visual_name, rgba in plan.visual_rgba_by_name.items()},
        "visual_link_by_name": dict(plan.visual_link_by_name),
    }


def deserialize_visual_material_restore_plan(payload: object) -> VisualMaterialRestorePlan | None:
    r"""从 cfg JSON-safe payload 恢复内部 visual plan；非法 payload 返回 ``None``。"""

    if not isinstance(payload, dict):
        return None
    source_urdf_path = payload.get("source_urdf_path")
    visual_rgba_by_name = payload.get("visual_rgba_by_name")
    visual_link_by_name = payload.get("visual_link_by_name")
    if not isinstance(source_urdf_path, str) or not isinstance(visual_rgba_by_name, dict):
        return None
    if not isinstance(visual_link_by_name, dict):
        return None
    return VisualMaterialRestorePlan(
        source_urdf_path=Path(source_urdf_path),
        visual_rgba_by_name={
            str(visual_name): tuple(float(value) for value in rgba)  # type: ignore[misc]
            for visual_name, rgba in visual_rgba_by_name.items()
        },
        visual_link_by_name={str(visual_name): str(link_name) for visual_name, link_name in visual_link_by_name.items()},
    )


def spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    r"""调用官方 URDF spawn，再恢复 generated hand 的 per-visual debug color。"""

    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    spawned_prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    visual_material_plan = deserialize_visual_material_restore_plan(
        getattr(cfg, "_anymani_visual_material_plan", None)
    )  # adapter 预计算的 same-topology plan
    if visual_material_plan is None:
        visual_material_plan = build_visual_material_restore_plan(Path(cfg.asset_path))  # direct-wrapper fallback
    restore_visual_materials_on_spawned_prim(
        spawned_prim,
        visual_material_plan.visual_rgba_by_name,
        visual_material_plan.visual_link_by_name,
    )
    return spawned_prim


def build_visual_material_restore_plan(urdf_path: Path) -> VisualMaterialRestorePlan:
    r"""从 reference URDF 构造 same-topology variants 共享的 debug color plan。"""

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    return VisualMaterialRestorePlan(
        source_urdf_path=resolved_urdf_path,
        visual_rgba_by_name=parse_urdf_visual_rgba_by_name(resolved_urdf_path),
        visual_link_by_name=parse_urdf_visual_link_by_name(resolved_urdf_path),
    )


def parse_urdf_visual_link_by_name(urdf_path: Path) -> dict[str, str]:
    r"""解析 URDF visual name 对应的 parent link name。"""

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")
    root = ET.parse(resolved_urdf_path).getroot()  # 纯 XML parse，不触碰 USD/Isaac state
    link_by_visual_name: dict[str, str] = {}
    for link_elem in root.findall("./link"):
        link_name = link_elem.attrib.get("name")  # 对应 spawned USD 一级 body prim
        if not link_name:
            continue
        for visual_elem in link_elem.findall("./visual"):
            visual_name = visual_elem.attrib.get("name")  # debug color key
            if visual_name:
                link_by_visual_name[visual_name] = link_name
    return link_by_visual_name


def restore_visual_materials_on_spawned_prim(
    spawned_prim,
    visual_rgba_by_name: dict[str, UrdfRgba],
    visual_link_by_name: dict[str, str],
) -> None:
    r"""在 spawned hand 的可编辑 visual ancestors 上绑定 URDF colors。"""

    if len(visual_rgba_by_name) == 0:
        return
    visual_prims = find_spawned_visual_prims_by_name(spawned_prim, visual_link_by_name)
    bound_target_by_path: dict[str, str] = {}
    missing_visual_names: list[str] = []
    for visual_name, rgba in visual_rgba_by_name.items():
        visual_prim = visual_prims.get(visual_name)
        if visual_prim is None:
            missing_visual_names.append(visual_name)
            continue
        target_prim = nearest_editable_material_binding_prim(visual_prim)
        target_path = str(target_prim.GetPath())
        previous_visual_name = bound_target_by_path.get(target_path)
        if previous_visual_name is not None and visual_rgba_by_name[previous_visual_name][:3] != rgba[:3]:
            logger.warning(
                "Skip URDF visual color for %s because editable USD target %s was already bound for %s.",
                visual_name,
                target_path,
                previous_visual_name,
            )
            continue
        try:
            bind_urdf_preview_surface(spawned_prim, target_prim, visual_name, rgba)
        except Exception as exc:
            logger.warning("Failed to restore URDF visual color for %s on %s: %s", visual_name, target_path, exc)
            continue
        bound_target_by_path[target_path] = visual_name
    if missing_visual_names:
        logger.warning(
            "Could not find %d URDF visual prims under spawned hand %s; examples: %s",
            len(missing_visual_names),
            spawned_prim.GetPath(),
            missing_visual_names[:5],
        )


def find_spawned_visual_prims_by_name(spawned_prim, visual_link_by_name: dict[str, str]) -> dict[str, Any]:
    r"""按 ``/<link>/visuals`` 查找 editable targets，不遍历 instance proxies。"""

    visual_prims: dict[str, Any] = {}
    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    for visual_name, link_name in visual_link_by_name.items():
        target_path = f"{root_path}/{link_name}/visuals"  # mesh 子树可为 proxy，ancestor 可编辑
        target_prim = stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            visual_prims[visual_name] = target_prim
    return visual_prims


def nearest_editable_material_binding_prim(visual_prim):
    r"""选择最近的非 instance-proxy material-binding ancestor。"""

    target_prim = visual_prim
    while target_prim.IsInstanceProxy():
        parent_prim = target_prim.GetParent()
        if not parent_prim.IsValid():
            break
        target_prim = parent_prim
    return target_prim


def bind_urdf_preview_surface(spawned_prim, target_prim, visual_name: str, rgba: UrdfRgba) -> None:
    r"""创建并绑定表示 URDF RGB 的 USD PreviewSurface material。"""

    from pxr import UsdShade

    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    looks_path = f"{root_path}/Looks"
    material_path = f"{looks_path}/{sanitize_usd_prim_name('urdf_' + visual_name)}"
    if not stage.GetPrimAtPath(looks_path).IsValid():
        stage.DefinePrim(looks_path, "Scope")
    if not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(rgba[0], rgba[1], rgba[2]),
            roughness=0.5,
            metallic=0.0,
        )
        material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    material_binding_api = (
        UsdShade.MaterialBindingAPI(target_prim)
        if target_prim.HasAPI(UsdShade.MaterialBindingAPI)
        else UsdShade.MaterialBindingAPI.Apply(target_prim)
    )
    material_binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def sanitize_usd_prim_name(raw_name: str) -> str:
    r"""把 URDF visual name 转成保守合法的 USD prim-name 片段。"""

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    return f"_{sanitized}" if sanitized == "" or sanitized[0].isdigit() else sanitized


__all__ = [
    "VisualMaterialRestorePlan",
    "build_visual_material_restore_plan",
    "serialize_visual_material_restore_plan",
    "spawn_urdf_with_restored_visual_materials",
]
