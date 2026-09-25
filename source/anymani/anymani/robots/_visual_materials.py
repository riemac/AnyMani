(
    'Restore generated-hand URDF colors as render-only USD materials for '
    'GUI/recording. This path never changes collision, mass, joints, drives, root '
    'pose, canonical URDF bytes, or physical/cache identity. Resolve source '
    'colors from assets.bank.visual_material_source, then build a per-child '
    'JSON-safe plan linking source visual/link names to canonical visual/link '
    'names and RGBA. Bind only at editable /<canonical-link>/visuals ancestors; '
    'never traverse instance proxies. If provenance or a unique geometry-backed '
    'mapping is missing, report the visual and keep the importer appearance.'
)

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils

from anymani.assets.bank import UrdfRgba
from anymani.assets.bank.visual_material_source import (
    UrdfVisualRecord,
    VisualMaterialResolution,
    parse_urdf_visual_source,
    resolve_visual_material_source,
)

logger = logging.getLogger(__name__)

VisualMaterialPlanProvenance = Literal["source", "parent", "none"]
'Identity of the source that supplied render-plan color evidence.'


@dataclass(frozen=True)
class VisualMaterialRestorePlan:
    (
        'Per-visual render contract for one source/canonical child. Keep source RGBA '
        'and source link maps keyed by source visual name; record canonical visual '
        'renames explicitly so wrappers never guess.'
    )

    source_urdf_path: Path
    'Current source-child URDF; canonical child retains this provenance.'

    visual_rgba_by_name: dict[str, UrdfRgba]
    'Source visual name to RGBA; read-only, never written back to URDF.'

    visual_link_by_name: dict[str, str]
    'Source visual name to canonical link; only uniquely mapped visuals.'

    canonical_visual_name_by_source_name: dict[str, str] = field(default_factory=dict)
    'Source visual name to canonical visual name; explicit rename evidence.'

    source_visual_link_by_name: dict[str, str] = field(default_factory=dict)
    'Source visual name to source link for per-visual audit.'

    source_urdf_sha256: str = ""
    'Current source URDF byte hash; for render audit, not canonical cache key.'

    color_source_urdf_path: Path | None = None
    'URDF that supplied colors; a validated parent may serve older revisions.'

    color_source_urdf_sha256: str | None = None
    'Byte hash of the color source, or None when evidence is unavailable.'

    canonical_urdf_path: Path | None = None
    'Canonical/native URDF actually spawned for this child.'

    canonical_urdf_sha256: str = ""
    'Current child URDF byte hash; audit evidence that the plan did not rewrite the artifact.'

    provenance: VisualMaterialPlanProvenance = "none"
    'Color source: source, verified parent, or none.'

    unresolved_visual_names: tuple[str, ...] = ()
    'Names with source RGBA but no unique canonical target.'

    geometry_signature_by_source_name: dict[str, tuple[str, ...]] = field(default_factory=dict)
    'Source visual name to geometry evidence for runtime comparison.'

    resolution_report: tuple[dict[str, object], ...] = ()
    'Source/parent parsing results and rejection reasons; JSON-safe.'


def serialize_visual_material_restore_plan(plan: VisualMaterialRestorePlan) -> dict[str, object]:
    (
        'Convert the plan to native JSON-hashable containers for '
        'UrdfFileCfg.to_dict(). This carries render evidence only; it does not modify '
        'the canonical artifact or stable USD cache physical identity.'
    )

    return {
        "source_urdf_path": str(plan.source_urdf_path),
        "source_urdf_sha256": plan.source_urdf_sha256,
        "color_source_urdf_path": str(plan.color_source_urdf_path) if plan.color_source_urdf_path else None,
        "color_source_urdf_sha256": plan.color_source_urdf_sha256,
        "canonical_urdf_path": str(plan.canonical_urdf_path) if plan.canonical_urdf_path else None,
        "canonical_urdf_sha256": plan.canonical_urdf_sha256,
        "visual_rgba_by_name": {
            visual_name: [float(value) for value in rgba]
            for visual_name, rgba in sorted(plan.visual_rgba_by_name.items())
        },
        "visual_link_by_name": dict(sorted(plan.visual_link_by_name.items())),
        "canonical_visual_name_by_source_name": dict(sorted(plan.canonical_visual_name_by_source_name.items())),
        "source_visual_link_by_name": dict(sorted(plan.source_visual_link_by_name.items())),
        "provenance": plan.provenance,
        "unresolved_visual_names": list(plan.unresolved_visual_names),
        "geometry_signature_by_source_name": {
            visual_name: list(signature)
            for visual_name, signature in sorted(plan.geometry_signature_by_source_name.items())
        },
        "resolution_report": [dict(item) for item in plan.resolution_report],
    }


def deserialize_visual_material_restore_plan(payload: object) -> VisualMaterialRestorePlan | None:
    'Restore an internal plan from JSON-safe payload; fail closed to None on invalid data.'

    if not isinstance(payload, Mapping):
        return None
    source_urdf_raw = payload.get("source_urdf_path")
    raw_rgba = payload.get("visual_rgba_by_name")
    raw_links = payload.get("visual_link_by_name")
    if not isinstance(source_urdf_raw, str) or not isinstance(raw_rgba, Mapping) or not isinstance(raw_links, Mapping):
        return None
    try:
        rgba_by_name = {str(name): _coerce_rgba(values) for name, values in raw_rgba.items()}
        links = {str(name): str(value) for name, value in raw_links.items()}
        canonical_names = _coerce_string_mapping(payload.get("canonical_visual_name_by_source_name", {}))
        source_links = _coerce_string_mapping(payload.get("source_visual_link_by_name", {}))
        geometry = _coerce_geometry_mapping(payload.get("geometry_signature_by_source_name", {}))
        provenance = payload.get("provenance", "source" if rgba_by_name else "none")
        if provenance not in {"source", "parent", "none"}:
            return None
        unresolved_raw = payload.get("unresolved_visual_names", [])
        if isinstance(unresolved_raw, (str, bytes)) or not isinstance(unresolved_raw, (list, tuple)):
            return None
        unresolved = tuple(str(value) for value in unresolved_raw)
        color_source_raw = payload.get("color_source_urdf_path")
        canonical_raw = payload.get("canonical_urdf_path")
        if color_source_raw is not None and not isinstance(color_source_raw, str):
            return None
        if canonical_raw is not None and not isinstance(canonical_raw, str):
            return None
    except (TypeError, ValueError):
        return None
    return VisualMaterialRestorePlan(
        source_urdf_path=Path(source_urdf_raw),
        visual_rgba_by_name=rgba_by_name,
        visual_link_by_name=links,
        canonical_visual_name_by_source_name=canonical_names,
        source_visual_link_by_name=source_links or dict(links),
        source_urdf_sha256=str(payload.get("source_urdf_sha256", "")),
        color_source_urdf_path=Path(color_source_raw) if color_source_raw else None,
        color_source_urdf_sha256=(
            str(payload["color_source_urdf_sha256"]) if payload.get("color_source_urdf_sha256") else None
        ),
        canonical_urdf_path=Path(canonical_raw) if canonical_raw else None,
        canonical_urdf_sha256=str(payload.get("canonical_urdf_sha256", "")),
        provenance=provenance,
        unresolved_visual_names=unresolved,
        geometry_signature_by_source_name=geometry,
        resolution_report=tuple(
            {str(key): value for key, value in item.items()}
            for item in payload.get("resolution_report", [])
            if isinstance(item, Mapping)
        ),
    )


def spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    "Call the official URDF spawn, then restore source debug colors from this child's plan."

    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    spawned_prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    visual_material_plan = deserialize_visual_material_restore_plan(getattr(cfg, "_anymani_visual_material_plan", None))
    if visual_material_plan is None:
        # Only a direct wrapper call without precomputed cfg payload reads the current asset;
        # HandSpawnAdapter always plans source/canonical materials before creating each child cfg.
        visual_material_plan = build_visual_material_restore_plan(Path(cfg.asset_path))
    restore_visual_materials_on_spawned_prim(
        spawned_prim,
        visual_material_plan.visual_rgba_by_name,
        visual_material_plan.visual_link_by_name,
        canonical_visual_name_by_source_name=visual_material_plan.canonical_visual_name_by_source_name,
        unresolved_visual_names=visual_material_plan.unresolved_visual_names,
    )
    return spawned_prim


def build_visual_material_restore_plan(
    urdf_path: Path,
    *,
    canonical_urdf_path: Path | None = None,
    source_sidecar: Mapping[str, Any] | None = None,
    canonical_joint_name_by_source_name: Mapping[str, str] | None = None,
) -> VisualMaterialRestorePlan:
    (
        "Build one child's render plan from the current source URDF and optional "
        'canonical target. Keep source palette/provenance keyed by source visual '
        'names; use explicit source-to-canonical joint mappings only for target '
        'links. For a native source target, visual/link identity is unchanged. Return '
        'a per-child JSON-safe color/mapping plan.'
    )

    resolution = resolve_visual_material_source(urdf_path, sidecar=source_sidecar)
    if resolution.provenance == "none" and resolution.report:
        logger.warning(
            "No reliable URDF visual color source for %s; importer appearance is retained. evidence=%s",
            resolution.source.urdf_path,
            resolution.report,
        )
    source = resolution.source
    canonical_path = (
        Path(canonical_urdf_path).expanduser().resolve(strict=False) if canonical_urdf_path else source.urdf_path
    )
    canonical = source if canonical_path == source.urdf_path else parse_urdf_visual_source(canonical_path)
    source_to_canonical_link = _derive_canonical_link_mapping(
        source.urdf_path,
        canonical.urdf_path,
        source_records=source.visuals,
        canonical_records=canonical.visuals,
        canonical_joint_name_by_source_name=canonical_joint_name_by_source_name,
    )
    (
        canonical_visual_names,
        canonical_links,
        unresolved_names,
        geometry_signatures,
        visual_report,
    ) = _map_source_visuals_to_canonical(
        resolution,
        canonical,
        source_to_canonical_link=source_to_canonical_link,
    )
    for source_record in source.visuals:
        geometry_signatures.setdefault(source_record.name, source_record.geometry_signature)
    return VisualMaterialRestorePlan(
        source_urdf_path=source.urdf_path,
        visual_rgba_by_name=dict(resolution.visual_rgba_by_name),
        visual_link_by_name=canonical_links,
        canonical_visual_name_by_source_name=canonical_visual_names,
        source_visual_link_by_name=dict(resolution.visual_link_by_name),
        source_urdf_sha256=source.urdf_sha256,
        color_source_urdf_path=resolution.color_source.urdf_path if resolution.color_source else None,
        color_source_urdf_sha256=resolution.color_source.urdf_sha256 if resolution.color_source else None,
        canonical_urdf_path=canonical.urdf_path,
        canonical_urdf_sha256=canonical.urdf_sha256,
        provenance=resolution.provenance,
        unresolved_visual_names=unresolved_names,
        geometry_signature_by_source_name=geometry_signatures,
        resolution_report=tuple(resolution.report) + tuple(visual_report),
    )


def parse_urdf_visual_link_by_name(urdf_path: Path) -> dict[str, str]:
    (
        'Parse the source parent link for a URDF visual name. This compatibility '
        'helper reads XML only; full color provenance uses parse_urdf_visual_source '
        'and geometry evidence.'
    )

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")
    root = ET.parse(resolved_urdf_path).getroot()  # Parse XML only; do not touch USD/Isaac state.
    link_by_visual_name: dict[str, str] = {}
    for link_elem in root.findall("./link"):
        link_name = link_elem.attrib.get("name")
        if not link_name:
            continue
        for visual_elem in link_elem.findall("./visual"):
            visual_name = visual_elem.attrib.get("name")
            if visual_name:
                link_by_visual_name[visual_name] = link_name
    return link_by_visual_name


def restore_visual_materials_on_spawned_prim(
    spawned_prim: Any,
    visual_rgba_by_name: Mapping[str, UrdfRgba],
    visual_link_by_name: Mapping[str, str],
    *,
    canonical_visual_name_by_source_name: Mapping[str, str] | None = None,
    unresolved_visual_names: tuple[str, ...] = (),
) -> None:
    (
        'Bind source RGBA to editable visual ancestors on one spawned hand. Keep each '
        "heterogeneous child's palette and explicit canonical rename mapping separate "
        'to avoid coloring the wrong link.'
    )

    if len(visual_rgba_by_name) == 0:
        return
    canonical_visual_name_by_source_name = canonical_visual_name_by_source_name or {}
    blocked = set(unresolved_visual_names)
    visual_prims = find_spawned_visual_prims_by_name(
        spawned_prim,
        visual_link_by_name,
        canonical_visual_name_by_source_name=canonical_visual_name_by_source_name,
    )
    bound_target_by_path: dict[str, str] = {}
    missing_visual_names: list[str] = []
    for source_visual_name, rgba in visual_rgba_by_name.items():
        if source_visual_name in blocked:
            missing_visual_names.append(source_visual_name)
            continue
        visual_prim = visual_prims.get(source_visual_name)
        if visual_prim is None:
            missing_visual_names.append(source_visual_name)
            continue
        target_prim = nearest_editable_material_binding_prim(visual_prim)
        target_path = str(target_prim.GetPath())
        previous_visual_name = bound_target_by_path.get(target_path)
        if previous_visual_name is not None and visual_rgba_by_name[previous_visual_name][:3] != rgba[:3]:
            logger.warning(
                "Skip URDF visual color for %s because editable USD target %s was already bound for %s.",
                source_visual_name,
                target_path,
                previous_visual_name,
            )
            continue
        try:
            canonical_visual_name = canonical_visual_name_by_source_name.get(
                source_visual_name, source_visual_name
            )  # material path follows spawned canonical visual name
            bind_urdf_preview_surface(spawned_prim, target_prim, canonical_visual_name, rgba)
        except Exception as exc:
            logger.warning("Failed to restore URDF visual color for %s on %s: %s", source_visual_name, target_path, exc)
            continue
        bound_target_by_path[target_path] = source_visual_name
    if missing_visual_names:
        logger.warning(
            "Could not restore %d URDF visual colors under spawned hand %s; examples: %s",
            len(missing_visual_names),
            spawned_prim.GetPath(),
            missing_visual_names[:5],
        )


def find_spawned_visual_prims_by_name(
    spawned_prim: Any,
    visual_link_by_name: Mapping[str, str],
    *,
    canonical_visual_name_by_source_name: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    'Find editable targets under /<canonical-link>/visuals; do not traverse instance proxies.'

    # Canonical visual name is mainly for audit/explicit mapping; locate the editable ancestor
    # by canonical link so visual prim renames do not rely on stale source paths.
    _ = canonical_visual_name_by_source_name
    visual_prims: dict[str, Any] = {}
    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    for source_visual_name, canonical_link_name in visual_link_by_name.items():
        target_path = f"{root_path}/{canonical_link_name}/visuals"
        target_prim = stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            visual_prims[source_visual_name] = target_prim
    return visual_prims


def nearest_editable_material_binding_prim(visual_prim: Any) -> Any:
    'Select the nearest material-binding ancestor that is not an instance proxy.'

    target_prim = visual_prim
    while target_prim.IsInstanceProxy():
        parent_prim = target_prim.GetParent()
        if not parent_prim.IsValid():
            break
        target_prim = parent_prim
    return target_prim


def bind_urdf_preview_surface(spawned_prim: Any, target_prim: Any, visual_name: str, rgba: UrdfRgba) -> None:
    'Create and bind a USD PreviewSurface material for source URDF RGB.'

    import isaaclab.sim as sim_utils
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


def audit_visual_material_restore_plan(plan: VisualMaterialRestorePlan) -> dict[str, object]:
    (
        'Return read-only per-visual render evidence for main-thread audit. Values '
        'are JSON-safe; this function does not access USD stage, spawn assets, or '
        'change physics/renderer state.'
    )

    records: list[dict[str, object]] = []
    source_names = tuple(dict.fromkeys((*plan.source_visual_link_by_name, *plan.visual_rgba_by_name)))
    for source_name in source_names:
        rgba = plan.visual_rgba_by_name.get(source_name)
        canonical_name = plan.canonical_visual_name_by_source_name.get(source_name)
        canonical_link = plan.visual_link_by_name.get(source_name)
        source_link = plan.source_visual_link_by_name.get(source_name)
        has_color = rgba is not None
        status = (
            "unresolved"
            if has_color and source_name in plan.unresolved_visual_names
            else "ready" if has_color else "unrestored"
        )
        reason = None if has_color else "no_reliable_source_color"
        records.append({
            "status": status,
            "source_visual_name": source_name,
            "canonical_visual_name": canonical_name,
            "source_link_name": source_link,
            "canonical_link_name": canonical_link,
            "rgba": [float(value) for value in rgba] if rgba is not None else None,
            "geometry_signature": list(plan.geometry_signature_by_source_name.get(source_name, ())),
            "color_provenance": plan.provenance,
            "source_urdf_sha256": plan.source_urdf_sha256,
            "color_source_urdf_sha256": plan.color_source_urdf_sha256,
            "reason": reason,
        })
    return {
        "source_urdf_path": str(plan.source_urdf_path),
        "source_urdf_sha256": plan.source_urdf_sha256,
        "canonical_urdf_path": str(plan.canonical_urdf_path) if plan.canonical_urdf_path else None,
        "canonical_urdf_sha256": plan.canonical_urdf_sha256,
        "color_source_urdf_path": str(plan.color_source_urdf_path) if plan.color_source_urdf_path else None,
        "color_source_urdf_sha256": plan.color_source_urdf_sha256,
        "provenance": plan.provenance,
        "visuals": records,
        "resolution_report": [dict(item) for item in plan.resolution_report],
    }


def _map_source_visuals_to_canonical(
    resolution: VisualMaterialResolution,
    canonical: Any,
    *,
    source_to_canonical_link: Mapping[str, str],
) -> tuple[
    dict[str, str],
    dict[str, str],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    tuple[dict[str, object], ...],
]:
    'Build a unique source-to-canonical mapping from link, visual name, and geometry evidence.'

    canonical_visual_names: dict[str, str] = {}
    canonical_links: dict[str, str] = {}
    unresolved: list[str] = []
    geometry_by_name: dict[str, tuple[str, ...]] = {}
    report: list[dict[str, object]] = []
    source_records = {record.name: record for record in resolution.source.visuals}
    used_targets: set[str] = set()
    for source_name, rgba in resolution.visual_rgba_by_name.items():
        source_record = source_records.get(source_name)
        if source_record is None:
            unresolved.append(source_name)
            report.append(
                {"status": "unresolved", "source_visual_name": source_name, "reason": "source_record_missing"}
            )
            continue
        geometry_by_name[source_name] = source_record.geometry_signature
        expected_link = source_to_canonical_link.get(source_record.link_name)
        candidates = _canonical_visual_candidates(
            source_record,
            canonical.visuals,
            expected_link=expected_link,
            used_targets=used_targets,
        )
        if len(candidates) != 1:
            unresolved.append(source_name)
            report.append({
                "status": "unresolved",
                "source_visual_name": source_name,
                "source_link_name": source_record.link_name,
                "expected_canonical_link_name": expected_link,
                "reason": "canonical_visual_geometry_ambiguous_or_mismatched",
                "candidate_count": len(candidates),
                "rgba": [float(value) for value in rgba],
            })
            continue
        target = candidates[0]
        canonical_visual_names[source_name] = target.name
        canonical_links[source_name] = target.link_name
        used_targets.add(_visual_target_key(target))
        report.append({
            "status": "mapping_ready",
            "source_visual_name": source_name,
            "canonical_visual_name": target.name,
            "source_link_name": source_record.link_name,
            "canonical_link_name": target.link_name,
            "geometry_signature": list(source_record.geometry_signature),
            "geometry_match": source_record.geometry_signature == target.geometry_signature,
            "rgba": [float(value) for value in rgba],
        })
    return canonical_visual_names, canonical_links, tuple(unresolved), geometry_by_name, tuple(report)


def _canonical_visual_candidates(
    source_record: UrdfVisualRecord,
    canonical_records: tuple[UrdfVisualRecord, ...],
    *,
    expected_link: str | None,
    used_targets: set[str],
) -> list[UrdfVisualRecord]:
    'Return only a uniquely supported canonical visual candidate; never guess from color similarity.'

    available = [record for record in canonical_records if _visual_target_key(record) not in used_targets]
    if expected_link is not None:
        on_link = [record for record in available if record.link_name == expected_link]
        geometric_on_link = [
            record for record in on_link if record.geometry_signature == source_record.geometry_signature
        ]
        if len(geometric_on_link) == 1:
            return geometric_on_link
        if geometric_on_link:
            return geometric_on_link
    same_name = [
        record
        for record in available
        if record.name == source_record.name and record.geometry_signature == source_record.geometry_signature
    ]
    if len(same_name) == 1:
        return same_name
    geometric = [record for record in available if record.geometry_signature == source_record.geometry_signature]
    return geometric if len(geometric) == 1 else []


def _derive_canonical_link_mapping(
    source_urdf_path: Path,
    canonical_urdf_path: Path,
    *,
    source_records: tuple[UrdfVisualRecord, ...],
    canonical_records: tuple[UrdfVisualRecord, ...],
    canonical_joint_name_by_source_name: Mapping[str, str] | None,
) -> dict[str, str]:
    'Infer source-to-canonical link mapping from verified joint renames, the joint graph, and same-name visuals.'

    source_root = ET.parse(source_urdf_path).getroot()
    canonical_root = ET.parse(canonical_urdf_path).getroot()
    source_joints = _joint_elements(source_root)
    canonical_joints = _joint_elements(canonical_root)
    canonical_by_name = {joint.attrib.get("name"): joint for joint in canonical_joints if joint.attrib.get("name")}
    link_mapping: dict[str, str] = {}
    source_link_names = {record.link_name for record in source_records}
    canonical_link_names = {record.link_name for record in canonical_records}
    for link_name in source_link_names:
        if link_name in canonical_link_names:
            link_mapping[link_name] = link_name
    if "palm" in source_link_names and "palm" in canonical_link_names:
        link_mapping["palm"] = "palm"

    # Use the validated source-to-canonical joint contract; do not infer depth/order from names.
    verified_joint_map = dict(canonical_joint_name_by_source_name or {})
    for source_joint in source_joints:
        source_name = source_joint.attrib.get("name")
        if not source_name:
            continue
        canonical_name = verified_joint_map.get(source_name)
        if canonical_name is None and source_name in canonical_by_name:
            canonical_name = source_name
        if canonical_name is None:
            inferred = _infer_canonical_joint_name(source_name, canonical_by_name)
            canonical_name = inferred
        canonical_joint = canonical_by_name.get(canonical_name) if canonical_name else None
        if canonical_joint is None:
            continue
        source_parent = source_joint.find("./parent")
        source_child = source_joint.find("./child")
        canonical_parent = canonical_joint.find("./parent")
        canonical_child = canonical_joint.find("./child")
        if (
            source_parent is not None
            and source_child is not None
            and canonical_parent is not None
            and canonical_child is not None
        ):
            source_parent_name = source_parent.attrib.get("link")
            source_child_name = source_child.attrib.get("link")
            canonical_parent_name = canonical_parent.attrib.get("link")
            canonical_child_name = canonical_child.attrib.get("link")
            if source_parent_name and canonical_parent_name:
                link_mapping.setdefault(source_parent_name, canonical_parent_name)
            if source_child_name and canonical_child_name:
                link_mapping.setdefault(source_child_name, canonical_child_name)

    # Root/TIP fixed joints may be renamed during canonical lowering; finger slot and endpoints provide independent evidence.
    for source_joint in source_joints:
        source_name = source_joint.attrib.get("name", "")
        source_parent = source_joint.find("./parent")
        source_child = source_joint.find("./child")
        if source_parent is None or source_child is None:
            continue
        source_parent_name = source_parent.attrib.get("link")
        source_child_name = source_child.attrib.get("link")
        if not source_parent_name or not source_child_name:
            continue
        slot = _infer_slot(source_name, source_parent_name, source_child_name)
        if slot is None:
            continue
        if source_joint.attrib.get("type") == "fixed":
            candidates = [
                f"{slot}_root_fixed",
                f"{slot}_tip_fixed",
            ]
            for candidate_name in candidates:
                canonical_joint = canonical_by_name.get(candidate_name)
                if canonical_joint is None:
                    continue
                canonical_parent = canonical_joint.find("./parent")
                canonical_child = canonical_joint.find("./child")
                if canonical_parent is None or canonical_child is None:
                    continue
                canonical_parent_name = canonical_parent.attrib.get("link")
                canonical_child_name = canonical_child.attrib.get("link")
                if candidate_name.endswith("_root_fixed") and source_parent_name == "palm":
                    if canonical_parent_name and canonical_child_name:
                        link_mapping.setdefault(source_parent_name, canonical_parent_name)
                        link_mapping.setdefault(source_child_name, canonical_child_name)
                elif candidate_name.endswith("_tip_fixed") and canonical_child_name:
                    link_mapping.setdefault(source_child_name, canonical_child_name)

    # Use a same-name visual as final source-link evidence only when geometry matches; otherwise leave it unmapped.
    canonical_by_visual_name = {record.name: record for record in canonical_records}
    for source_record in source_records:
        target = canonical_by_visual_name.get(source_record.name)
        if target is not None and target.geometry_signature == source_record.geometry_signature:
            link_mapping.setdefault(source_record.link_name, target.link_name)
    return link_mapping


def _joint_elements(root: ET.Element) -> tuple[ET.Element, ...]:
    'Return top-level URDF joints only; do not treat nested metadata as kinematic edges.'

    return tuple(root.findall("./joint"))


def _infer_canonical_joint_name(source_name: str, canonical_by_name: Mapping[str | None, ET.Element]) -> str | None:
    'Infer joint renames from canonical v1 <slot>_j<depth> names.'

    match = re.search(r"(?P<slot>thumb|index|middle|ring)_j(?P<depth>[0-9]+)$", source_name)
    if match is None:
        return None
    candidate = f"{match.group('slot')}_j{match.group('depth')}"
    return candidate if candidate in canonical_by_name else None


def _infer_slot(*names: str) -> str | None:
    'Extract a finger slot supported by the canonical schema from joint/link names.'

    for slot in ("thumb", "index", "middle", "ring"):
        if any(re.search(rf"(?:^|[_-]){slot}(?:[_-]|$)", name) for name in names):
            return slot
    return None


def _visual_target_key(record: UrdfVisualRecord) -> str:
    'Build a stable canonical visual target key; duplicate visual names on different links are allowed.'

    return f"{record.link_name}\x00{record.name}"


def _coerce_rgba(values: object) -> UrdfRgba:
    'Validate four-channel JSON-safe colors; reject NaN/Inf and wrong shapes.'

    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("RGBA payload must be a four-value list")
    numbers = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("RGBA payload must contain finite values")
    return numbers  # type: ignore[return-value]


def _coerce_string_mapping(value: object) -> dict[str, str]:
    'Validate a JSON string-to-string mapping.'

    if not isinstance(value, Mapping):
        raise ValueError("mapping payload must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _coerce_geometry_mapping(value: object) -> dict[str, tuple[str, ...]]:
    'Validate a visual-name-to-geometry-signature mapping.'

    if not isinstance(value, Mapping):
        raise ValueError("geometry mapping payload must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for key, signature in value.items():
        if not isinstance(signature, (list, tuple)):
            raise ValueError("geometry signature must be a list")
        result[str(key)] = tuple(str(item) for item in signature)
    return result


def sanitize_usd_prim_name(raw_name: str) -> str:
    'Convert a URDF visual name to a conservative USD prim-name fragment.'

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    return f"_{sanitized}" if sanitized == "" or sanitized[0].isdigit() else sanitized


__all__ = [
    "VisualMaterialRestorePlan",
    "audit_visual_material_restore_plan",
    "bind_urdf_preview_surface",
    "build_visual_material_restore_plan",
    "deserialize_visual_material_restore_plan",
    "find_spawned_visual_prims_by_name",
    "nearest_editable_material_binding_prim",
    "parse_urdf_visual_link_by_name",
    "restore_visual_materials_on_spawned_prim",
    "sanitize_usd_prim_name",
    "serialize_visual_material_restore_plan",
    "spawn_urdf_with_restored_visual_materials",
]
