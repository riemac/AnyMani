"Applies the declared axial geometry proposal while preserving joint-frame semantics."

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ...asset_base import HandCfg
from ...asset_schema_core import EllipticCylinderGeometryCfg, PoseCfg
from .base import PatchOp

_AXIAL_KINDS = {"box", "cylinder", "elliptic_cylinder"}
_AXIAL_COMPOSER = "axial_geometry_v1"
_FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ProximalOverlapContribution:
    "Resolved change to proximal overlap while preserving the distal boundary."

    joint_name: str
    child_link: str
    delta_ratio: float
    base_overhang: float
    source_child_span: float
    parent_span_before: float
    max_parent_overlap_ratio: float


@dataclass(frozen=True)
class ResolvedAxialGeometry:
    "Updated link lengths, cross-sections, and downstream origins from one proposal."

    scaled_length: float
    final_length: float
    proximal_delta: float
    scaled_child_span: float | None
    base_overhang: float | None
    final_overhang: float | None
    max_overhang: float | None

    def overlap_provenance(self, *, joint_name: str, child_link: str) -> dict[str, Any]:
        "Records the source and resolved proximal-overlap values for one link."

        assert self.scaled_child_span is not None
        assert self.base_overhang is not None
        assert self.final_overhang is not None
        assert self.max_overhang is not None
        return {
            "joint_name": joint_name,
            "child_link": child_link,
            "scaled_child_span_m": self.scaled_child_span,
            "base_overhang_m": self.base_overhang,
            "final_overhang_m": self.final_overhang,
            "max_overhang_m": self.max_overhang,
            "geometry_delta_m": self.proximal_delta,
        }


@dataclass(frozen=True)
class AxialGeometryEdit:
    "Deferred geometry patch describing one link-scale or overlap change."

    finger_index: int
    joint_index: int
    joint_name: str
    child_link: str
    source_length: float
    source_cross_section: tuple[float, float] | None
    keep_center: bool
    scaled_length: float | None = None
    scaled_cross_section: tuple[float, float] | None = None
    overlap: ProximalOverlapContribution | None = None

    @property
    def path(self) -> tuple[Any, ...]:
        "Returns the ordered downstream path affected by an axial geometry edit."

        return ("finger", self.finger_index, "joint", self.joint_index, "link_geometry")

    def merge(self, other: AxialGeometryEdit) -> AxialGeometryEdit:
        "Combines geometry edits from the same source configuration without applying them early."

        if (self.finger_index, self.joint_index, self.joint_name, self.child_link) != (
            other.finger_index,
            other.joint_index,
            other.joint_name,
            other.child_link,
        ):
            raise ValueError("axial geometry edits must target the same joint identity")
        if not math.isclose(self.source_length, other.source_length, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
            raise ValueError(f"axial geometry source length mismatch for {self.joint_name!r}")
        if self.source_cross_section != other.source_cross_section or self.keep_center != other.keep_center:
            raise ValueError(f"axial geometry source frame mismatch for {self.joint_name!r}")
        if self.scaled_length is not None and other.scaled_length is not None:
            raise ValueError(f"duplicate link-scale edit for {self.joint_name!r}")
        if self.overlap is not None and other.overlap is not None:
            raise ValueError(f"duplicate proximal-overlap edit for {self.joint_name!r}")

        return replace(
            self,
            scaled_length=self.scaled_length if self.scaled_length is not None else other.scaled_length,
            scaled_cross_section=(
                self.scaled_cross_section
                if self.scaled_cross_section is not None
                else other.scaled_cross_section
            ),
            overlap=self.overlap if self.overlap is not None else other.overlap,
        )

    def resolve(self) -> ResolvedAxialGeometry:
        "Resolves declared source paths, typed sidecars, and the required mesh closure."

        scaled_length = float(self.scaled_length if self.scaled_length is not None else self.source_length)
        if scaled_length <= _FLOAT_TOLERANCE:
            raise ValueError(f"scaled axial length must be positive for {self.joint_name!r}")

        if self.overlap is None:
            return ResolvedAxialGeometry(
                scaled_length=scaled_length,
                final_length=scaled_length,
                proximal_delta=0.0,
                scaled_child_span=None,
                base_overhang=None,
                final_overhang=None,
                max_overhang=None,
            )

        contribution = self.overlap
        scaled_child_span = contribution.source_child_span + (scaled_length - self.source_length)
        if scaled_child_span <= _FLOAT_TOLERANCE:
            raise ValueError(f"child effective span must be positive for {self.joint_name!r}")
        if contribution.parent_span_before <= _FLOAT_TOLERANCE:
            raise ValueError(f"parent effective span must be positive for {self.joint_name!r}")

        raw_overhang = contribution.base_overhang + contribution.delta_ratio * scaled_child_span
        max_overhang = contribution.max_parent_overlap_ratio * contribution.parent_span_before
        final_overhang = max(0.0, min(raw_overhang, max_overhang))
        proximal_delta = final_overhang - contribution.base_overhang
        final_length = scaled_length + proximal_delta
        if final_length <= _FLOAT_TOLERANCE:
            raise ValueError(f"proximal edit collapses axial geometry for {self.joint_name!r}")

        return ResolvedAxialGeometry(
            scaled_length=scaled_length,
            final_length=final_length,
            proximal_delta=proximal_delta,
            scaled_child_span=scaled_child_span,
            base_overhang=contribution.base_overhang,
            final_overhang=final_overhang,
            max_overhang=max_overhang,
        )

    def apply(self, target: HandCfg) -> None:
        "Applies resolved axial dimensions and downstream origin updates to a hand copy."

        resolved = self.resolve()
        joint = target.fingers[self.finger_index].joints[self.joint_index]
        _set_joint_primary_geometry(
            joint,
            old_length=self.source_length,
            new_length=resolved.final_length,
            old_cross_section=self.source_cross_section,
            new_cross_section=self.scaled_cross_section,
            keep_center=self.keep_center,
            proximal_delta=resolved.proximal_delta,
        )


def make_axial_geometry_patch_op(edit: AxialGeometryEdit) -> PatchOp:
    'Builds axial geometry patch op.'

    def apply(target: HandCfg, *, payload=edit) -> None:
        payload.apply(target)

    def compose(other: PatchOp, *, payload=edit) -> PatchOp:
        if not isinstance(other.payload, AxialGeometryEdit):
            raise ValueError(f"invalid axial geometry payload for {payload.joint_name!r}")
        return make_axial_geometry_patch_op(payload.merge(other.payload))

    def finalize_metadata(metadata: dict[str, Any], *, payload=edit) -> None:
        if payload.overlap is None:
            return
        resolved = payload.resolve()
        result = resolved.overlap_provenance(
            joint_name=payload.joint_name,
            child_link=payload.child_link,
        )
        contribution = payload.overlap
        result.update(
            {
                "delta_ratio": contribution.delta_ratio,
                "source_child_span_m": contribution.source_child_span,
                "parent_span_before_m": contribution.parent_span_before,
                "max_parent_overlap_ratio": contribution.max_parent_overlap_ratio,
            }
        )
        post_samples = metadata.setdefault("post_mutate_samples", {})
        overlap_sample = post_samples.setdefault("link_proximal_overlap", {})
        overlap_sample.setdefault("owner_results", {})[payload.joint_name] = result
        top_level = metadata.setdefault("post_mutate_link_proximal_overlap", overlap_sample)
        if isinstance(top_level, dict):
            top_level.setdefault("owner_results", {})[payload.joint_name] = result

    return PatchOp(
        path=edit.path,
        apply=apply,
        composer=_AXIAL_COMPOSER,
        payload=edit,
        compose=compose,
        finalize_metadata=finalize_metadata,
    )


def joint_primary_length(joint: Any) -> float | None:
    "Returns the principal link length along the declared anatomical axis."

    geometry = joint.collisions[0].geometry if joint.collisions else None
    if geometry is None and joint.visuals:
        geometry = joint.visuals[0].geometry
    if geometry is None:
        return None
    if geometry.kind == "box":
        return float(geometry.size[1])
    if geometry.kind in {"cylinder", "elliptic_cylinder"}:
        return float(geometry.length)
    return None


def joint_cross_section(joint: Any) -> tuple[float, float] | None:
    "Returns the width and height used by the selected joint-child geometry."

    geometry = joint.collisions[0].geometry if joint.collisions else None
    if geometry is None and joint.visuals:
        geometry = joint.visuals[0].geometry
    if geometry is None:
        return None
    if geometry.kind == "box":
        return float(geometry.size[0]), float(geometry.size[2])
    if geometry.kind == "cylinder":
        diameter = 2.0 * float(geometry.radius)
        return diameter, diameter
    if geometry.kind == "elliptic_cylinder":
        return 2.0 * float(geometry.radius_x), 2.0 * float(geometry.radius_z)
    return None


def validate_axial_geometry(joint: Any) -> None:
    'Validates axial geometry.'

    elements = [*joint.collisions, *joint.visuals]
    if not elements:
        raise ValueError(f"proximal overlap requires collision/visual geometry for {joint.name!r}")
    unsupported = sorted({element.geometry.kind for element in elements} - _AXIAL_KINDS)
    if unsupported:
        raise ValueError(
            f"proximal overlap supports only {_AXIAL_KINDS}, got {unsupported} for {joint.name!r}"
        )


def _set_joint_primary_geometry(
    joint: Any,
    *,
    old_length: float,
    new_length: float,
    old_cross_section: tuple[float, float] | None,
    new_cross_section: tuple[float, float] | None,
    keep_center: bool,
    proximal_delta: float,
) -> None:

    for collection_name in ("collisions", "visuals"):
        collection = getattr(joint, collection_name)
        for index, element in enumerate(collection):
            geometry = element.geometry
            if geometry.kind == "box":
                size = geometry.size
                width = float(size[0]) if new_cross_section is None else float(new_cross_section[0])
                height = float(size[2]) if new_cross_section is None else float(new_cross_section[1])
                geometry = geometry.replace(size=(width, new_length, height))
            elif geometry.kind == "cylinder":
                if new_cross_section is None:
                    geometry = geometry.replace(length=new_length)
                else:
                    radius_x = 0.5 * float(new_cross_section[0])
                    radius_z = 0.5 * float(new_cross_section[1])
                    if math.isclose(radius_x, radius_z, rel_tol=0.0, abs_tol=1e-12):
                        geometry = geometry.replace(radius=radius_x, length=new_length)
                    else:
                        geometry = EllipticCylinderGeometryCfg(
                            radius_x=radius_x,
                            radius_z=radius_z,
                            length=new_length,
                        )
            elif geometry.kind == "elliptic_cylinder":
                if new_cross_section is None:
                    geometry = geometry.replace(length=new_length)
                else:
                    radius_x = 0.5 * float(new_cross_section[0])
                    radius_z = 0.5 * float(new_cross_section[1])
                    if math.isclose(radius_x, radius_z, rel_tol=0.0, abs_tol=1e-12):
                        from ...asset_schema_core import CylinderGeometryCfg

                        geometry = CylinderGeometryCfg(radius=radius_x, length=new_length)
                    else:
                        geometry = geometry.replace(
                            radius_x=radius_x,
                            radius_z=radius_z,
                            length=new_length,
                        )
            else:
                raise ValueError(f"unsupported axial geometry kind={geometry.kind!r} for {joint.name!r}")

            origin = element.origin
            if keep_center:
                if abs(proximal_delta) > _FLOAT_TOLERANCE:
                    raise ValueError(f"centered joint cannot receive proximal overlap: {joint.name!r}")
                new_origin = origin.copy()
            else:
                offset_y = origin.pos[1] - old_length / 2.0

                scaled_center_y = new_length / 2.0 + offset_y - proximal_delta
                new_origin = PoseCfg(
                    pos=(origin.pos[0], scaled_center_y, origin.pos[2]),
                    rpy=origin.rpy,
                )
            collection[index] = element.replace(geometry=geometry, origin=new_origin)
            if joint.inertial is not None and index == 0:
                joint.inertial = joint.inertial.replace(origin=new_origin)


__all__ = [
    "AxialGeometryEdit",
    "ProximalOverlapContribution",
    "joint_cross_section",
    "joint_primary_length",
    "make_axial_geometry_patch_op",
    "validate_axial_geometry",
]
