"Resolves visual material data from the source URDF and records the exact files used."

from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .hand_container import UrdfRgba

VisualMaterialProvenance = Literal["source", "parent", "none"]
"Source of the visual color selected for an exported mesh."


@dataclass(frozen=True)
class UrdfVisualRecord:
    "Visual element and material values read from one source URDF link."

    name: str
    "Stable semantic identifier preserved in the output metadata."

    link_name: str
    "Stable semantic identifier preserved in the output metadata."

    rgba: UrdfRgba | None
    "Visual red, green, blue, and alpha values in the normalized interval [0, 1]."

    geometry_signature: tuple[str, ...]
    "Stable digest of the physical geometry fields used for uniqueness checks."


@dataclass(frozen=True)
class VisualMaterialSource:
    "Raw visual materials, colors, and mesh references from a hand bundle."

    urdf_path: Path
    "Filesystem location resolved relative to the declared asset root when not absolute."

    urdf_sha256: str
    "SHA-256 of the exact source URDF bytes."

    visuals: tuple[UrdfVisualRecord, ...]
    "Named visual records with source mesh and material references."

    @property
    def visual_rgba_by_name(self) -> dict[str, UrdfRgba]:
        "Returns normalized RGBA values for each named visual element."

        result: dict[str, UrdfRgba] = {}
        for visual in self.visuals:
            if visual.rgba is not None and visual.name not in result:
                result[visual.name] = visual.rgba
        return result

    @property
    def visual_link_by_name(self) -> dict[str, str]:
        "Returns the source link for one unique named visual."

        result: dict[str, str] = {}
        for visual in self.visuals:
            result.setdefault(visual.name, visual.link_name)
        return result

    @property
    def duplicate_visual_names(self) -> tuple[str, ...]:
        "Finds visual links whose source names collide before material routing."

        counts: dict[str, int] = {}
        for visual in self.visuals:
            counts[visual.name] = counts.get(visual.name, 0) + 1
        return tuple(sorted(name for name, count in counts.items() if count > 1))


@dataclass(frozen=True)
class VisualMaterialResolution:
    "Resolved display colors and source evidence for each named visual."

    source: VisualMaterialSource
    "Source record or parent object for the current lowering operation."

    color_source: VisualMaterialSource | None
    "Chosen source for the exported visual material color."

    visual_rgba_by_name: dict[str, UrdfRgba]
    "Stable semantic identifier preserved in the output metadata."

    visual_link_by_name: dict[str, str]
    "Stable semantic identifier preserved in the output metadata."

    provenance: VisualMaterialProvenance
    "Source and generation history preserved for identity and audit."

    report: tuple[dict[str, object], ...]
    "Structured evidence produced by the current validation or build stage."

    def to_dict(self) -> dict[str, object]:
        'Serializes the typed object as a dictionary.'

        return {
            "source_urdf_path": str(self.source.urdf_path),
            "source_urdf_sha256": self.source.urdf_sha256,
            "color_source_urdf_path": str(self.color_source.urdf_path) if self.color_source else None,
            "color_source_urdf_sha256": self.color_source.urdf_sha256 if self.color_source else None,
            "visual_rgba_by_name": {
                name: [float(value) for value in rgba] for name, rgba in sorted(self.visual_rgba_by_name.items())
            },
            "visual_link_by_name": dict(sorted(self.visual_link_by_name.items())),
            "provenance": self.provenance,
            "report": [dict(item) for item in self.report],
        }


def parse_urdf_visual_source(urdf_path: Path) -> VisualMaterialSource:
    'Loads and validates urdf visual source.'

    resolved_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_path}")

    root = ET.parse(resolved_path).getroot()
    visuals: list[UrdfVisualRecord] = []
    for link_elem in root.findall("./link"):
        link_name = link_elem.attrib.get("name")
        if not link_name:
            continue
        for visual_elem in link_elem.findall("./visual"):
            visual_name = visual_elem.attrib.get("name")
            if not visual_name:
                continue
            visuals.append(
                UrdfVisualRecord(
                    name=visual_name,
                    link_name=link_name,
                    rgba=_parse_visual_rgba(visual_elem, visual_name=visual_name, urdf_path=resolved_path),
                    geometry_signature=_geometry_signature(visual_elem, urdf_path=resolved_path),
                )
            )
    return VisualMaterialSource(
        urdf_path=resolved_path,
        urdf_sha256=_sha256_file(resolved_path),
        visuals=tuple(visuals),
    )


def resolve_visual_material_source(
    urdf_path: Path,
    *,
    sidecar: Mapping[str, Any] | None = None,
) -> VisualMaterialResolution:
    'Resolves visual material source.'

    source = parse_urdf_visual_source(urdf_path)
    source_colors = source.visual_rgba_by_name
    source_links = source.visual_link_by_name
    if source_colors:
        report: tuple[dict[str, object], ...]
        if source.duplicate_visual_names:
            report = (
                {
                    "status": "rejected",
                    "reason": "duplicate_source_visual_name",
                    "visual_names": list(source.duplicate_visual_names),
                },
            )
            return VisualMaterialResolution(source, None, {}, source_links, "none", report)
        report = (
            {
                "status": "accepted",
                "provenance": "source",
                "visual_count": len(source_colors),
            },
        )
        return VisualMaterialResolution(source, source, source_colors, source_links, "source", report)

    parent_resolution, parent_report = _resolve_revision_parent(source, sidecar=sidecar)
    if parent_resolution is None:
        return VisualMaterialResolution(
            source,
            None,
            {},
            source_links,
            "none",
            parent_report
            or (
                {
                    "status": "unrestored",
                    "reason": "no_reliable_source_color",
                },
            ),
        )
    color_source, colors, report = parent_resolution
    return VisualMaterialResolution(source, color_source, colors, source_links, "parent", report)


def _resolve_revision_parent(
    source: VisualMaterialSource,
    *,
    sidecar: Mapping[str, Any] | None,
) -> tuple[
    tuple[VisualMaterialSource, dict[str, UrdfRgba], tuple[dict[str, object], ...]] | None,
    tuple[dict[str, object], ...],
]:

    revision = sidecar.get("asset_revision") if isinstance(sidecar, Mapping) else None
    if not isinstance(revision, Mapping):
        return None, ()
    raw_bundle = revision.get("parent_bundle")
    expected_hash = revision.get("parent_urdf_sha256")
    if (
        not isinstance(raw_bundle, str)
        or not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None
    ):
        return None, (
            {
                "status": "rejected",
                "reason": "invalid_parent_provenance_fields",
            },
        )

    parent_bundle_path = Path(raw_bundle).expanduser()
    if not parent_bundle_path.is_absolute():
        parent_bundle_path = source.urdf_path.parent / parent_bundle_path
    parent_bundle = parent_bundle_path.resolve(strict=False)
    parent_urdf = parent_bundle if parent_bundle.name == "hand.urdf" else parent_bundle / "hand.urdf"
    if not parent_urdf.is_file():
        return None, (
            {
                "status": "rejected",
                "reason": "parent_urdf_missing",
                "parent_bundle": str(parent_bundle),
            },
        )
    actual_hash = _sha256_file(parent_urdf)
    if actual_hash.lower() != expected_hash.lower():
        return None, (
            {
                "status": "rejected",
                "reason": "parent_urdf_sha256_mismatch",
                "parent_bundle": str(parent_bundle),
                "parent_urdf_sha256_expected": expected_hash,
                "parent_urdf_sha256_actual": actual_hash,
            },
        )

    try:
        parent = parse_urdf_visual_source(parent_urdf)
    except (ET.ParseError, OSError, ValueError):

        return None, (
            {
                "status": "rejected",
                "reason": "parent_urdf_parse_failed",
                "parent_bundle": str(parent_bundle),
            },
        )
    if parent.duplicate_visual_names or source.duplicate_visual_names:
        return None, (
            {
                "status": "rejected",
                "reason": "duplicate_parent_or_source_visual_name",
            },
        )
    if _contains_weak_mesh_evidence(source.visuals) or _contains_weak_mesh_evidence(parent.visuals):
        return None, (
            {
                "status": "rejected",
                "reason": "parent_mesh_evidence_missing",
            },
        )
    matches, matching_report = _match_visual_geometry(source.visuals, parent.visuals)
    if matches is None:
        return None, matching_report
    parent_colors = parent.visual_rgba_by_name
    if not parent_colors:
        return None, (
            {
                "status": "rejected",
                "reason": "parent_has_no_visual_color",
            },
        )
    colors = {
        current_name: parent_colors[parent_name]
        for current_name, parent_name in matches.items()
        if parent_name in parent_colors
    }
    if not colors:
        return None, (
            {
                "status": "rejected",
                "reason": "parent_color_names_do_not_match_source_geometry",
            },
        )
    accepted_report: list[dict[str, object]] = [{
        "status": "accepted",
        "provenance": "parent",
        "parent_bundle": str(parent_bundle),
        "parent_urdf_sha256_expected": expected_hash,
        "parent_urdf_sha256_actual": actual_hash,
    }]
    accepted_report.extend(matching_report)
    return (parent, colors, tuple(accepted_report)), ()


def _match_visual_geometry(
    current: tuple[UrdfVisualRecord, ...],
    parent: tuple[UrdfVisualRecord, ...],
) -> tuple[dict[str, str], tuple[dict[str, object], ...]] | tuple[None, tuple[dict[str, object], ...]]:

    if len(current) != len(parent):
        return None, (
            {
                "status": "rejected",
                "reason": "parent_visual_count_mismatch",
                "current_visual_count": len(current),
                "parent_visual_count": len(parent),
            },
        )
    unused_current = list(current)
    mapping: dict[str, str] = {}
    report: list[dict[str, object]] = []
    for parent_visual in parent:
        exact = [
            item
            for item in unused_current
            if item.name == parent_visual.name
            and item.link_name == parent_visual.link_name
            and item.geometry_signature == parent_visual.geometry_signature
        ]
        candidate_pool = exact
        candidates = candidate_pool if len(candidate_pool) == 1 else []
        if len(candidates) != 1:
            return None, (
                {
                    "status": "rejected",
                    "reason": "parent_visual_geometry_ambiguous_or_mismatched",
                    "parent_visual_name": parent_visual.name,
                    "parent_link_name": parent_visual.link_name,
                    "candidate_count": len(candidate_pool),
                },
            )
        current_visual = candidates[0]
        unused_current.remove(current_visual)
        mapping[current_visual.name] = parent_visual.name
        report.append({
            "status": "geometry_match",
            "current_visual_name": current_visual.name,
            "parent_visual_name": parent_visual.name,
            "current_link_name": current_visual.link_name,
            "parent_link_name": parent_visual.link_name,
            "geometry_signature": list(current_visual.geometry_signature),
            "parent_has_color": parent_visual.rgba is not None,
        })
    return mapping, tuple(report)


def _parse_visual_rgba(visual_elem: ET.Element, *, visual_name: str, urdf_path: Path) -> UrdfRgba | None:

    color_elem = visual_elem.find("./material/color")
    if color_elem is None or color_elem.attrib.get("rgba") is None:
        return None
    raw_values = color_elem.attrib["rgba"].split()
    if len(raw_values) != 4:
        raise ValueError(f"visual {visual_name!r} in {urdf_path} has invalid rgba field")
    try:
        values = tuple(float(value) for value in raw_values)
    except ValueError as exc:
        raise ValueError(f"visual {visual_name!r} in {urdf_path} has non-float rgba field") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"visual {visual_name!r} in {urdf_path} has non-finite rgba field")
    return values  # type: ignore[return-value]


def _geometry_signature(visual_elem: ET.Element, *, urdf_path: Path) -> tuple[str, ...]:

    geometry_elem = visual_elem.find("./geometry")
    if geometry_elem is None:
        return ("missing_geometry",)
    children = list(geometry_elem)
    if len(children) != 1:
        return ("geometry_children", str(len(children)))
    primitive = children[0]
    tag = primitive.tag.rsplit("}", 1)[-1]
    origin_elem = visual_elem.find("./origin")
    origin_signature = (
        "origin=" + ",".join(f"{key}:{_normalize_scalar(value)}" for key, value in sorted(origin_elem.attrib.items()))
        if origin_elem is not None
        else "origin=default"
    )
    attributes = tuple(
        f"{key}={_normalize_scalar(value)}" for key, value in sorted(primitive.attrib.items()) if key != "filename"
    )
    if tag != "mesh":
        return (tag, origin_signature, *attributes)
    raw_uri = primitive.attrib.get("filename", "").strip()
    mesh_path = _resolve_mesh_evidence_path(raw_uri, urdf_path=urdf_path)
    mesh_digest = _sha256_file(mesh_path) if mesh_path is not None and mesh_path.is_file() else ""
    if mesh_digest:

        return ("mesh", origin_signature, f"sha256={mesh_digest}", *attributes)

    return ("mesh", origin_signature, "sha256=missing", f"basename={Path(raw_uri).name}", *attributes)


def _resolve_mesh_evidence_path(raw_uri: str, *, urdf_path: Path) -> Path | None:

    if not raw_uri or raw_uri.startswith("package://"):
        return None
    mesh_path = Path(raw_uri).expanduser()
    return (
        mesh_path.resolve(strict=False)
        if mesh_path.is_absolute()
        else (urdf_path.parent / mesh_path).resolve(strict=False)
    )


def _normalize_scalar(raw_value: str) -> str:

    value = raw_value.strip()
    try:
        parsed = float(value)
    except ValueError:
        return value
    return f"{parsed:.12g}" if math.isfinite(parsed) else value


def _contains_weak_mesh_evidence(visuals: tuple[UrdfVisualRecord, ...]) -> bool:

    return any(record.geometry_signature[:2] == ("mesh", "sha256=missing") for record in visuals)


def _sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "UrdfVisualRecord",
    "VisualMaterialResolution",
    "VisualMaterialSource",
    "parse_urdf_visual_source",
    "resolve_visual_material_source",
]
