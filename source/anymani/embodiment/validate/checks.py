from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from embodiment.schema.hir_v01 import HandHIR, is_movable_joint_type


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def validate_hir(hir: HandHIR) -> ValidationReport:
    report = ValidationReport()

    if hir.root_link is None:
        report.errors.append("root_link is None")

    # root uniqueness from graph map
    if hir.graph.parent_map:
        roots = [k for k, p in hir.graph.parent_map.items() if p is None]
        if len(roots) == 0:
            report.errors.append("graph has no root")
        elif len(roots) > 1:
            report.warnings.append(f"graph has multiple roots: {roots}")

    # parent/child link existence
    for j in hir.joints.values():
        if j.parent_link not in hir.links:
            report.errors.append(f"joint {j.joint_id}: missing parent link {j.parent_link}")
        if j.child_link not in hir.links:
            report.errors.append(f"joint {j.joint_id}: missing child link {j.child_link}")

    # axis validity for movable joints
    for j in hir.joints.values():
        if not is_movable_joint_type(j.joint_type):
            continue
        ax = j.axis_local
        norm = (ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) ** 0.5
        if norm < 1e-9:
            report.errors.append(f"joint {j.joint_id}: zero axis for movable joint")

    # limit sanity
    for j in hir.joints.values():
        if j.limit_lower is not None and j.limit_upper is not None and j.limit_lower > j.limit_upper:
            report.errors.append(f"joint {j.joint_id}: lower > upper")

    # collision sanity
    supported = {"box", "sphere", "capsule", "cylinder", "mesh"}
    for link_name, link in hir.links.items():
        if not link.collisions:
            report.warnings.append(f"link {link_name}: no collision items")
        for i, c in enumerate(link.collisions):
            if c.geom_type not in supported:
                report.errors.append(f"link {link_name} collision[{i}]: unsupported geom_type {c.geom_type}")
                continue
            if c.geom_type == "box":
                if c.size is None or any(v <= 0 for v in c.size):
                    report.errors.append(f"link {link_name} collision[{i}]: invalid box size")
            elif c.geom_type == "sphere":
                if c.radius is None or c.radius <= 0:
                    report.errors.append(f"link {link_name} collision[{i}]: invalid sphere radius")
            elif c.geom_type in {"capsule", "cylinder"}:
                if c.radius is None or c.length is None or c.radius <= 0 or c.length <= 0:
                    report.errors.append(f"link {link_name} collision[{i}]: invalid {c.geom_type} params")
            elif c.geom_type == "mesh":
                if not c.mesh_file:
                    report.errors.append(f"link {link_name} collision[{i}]: mesh filename missing")

    # graph cycle check from root
    if hir.root_link is not None and hir.graph.child_map:
        visited: Set[str] = set()
        stack: Set[str] = set()

        def dfs(u: str):
            if u in stack:
                report.errors.append(f"cycle detected at {u}")
                return
            if u in visited:
                return
            visited.add(u)
            stack.add(u)
            for v in hir.graph.child_map.get(u, []):
                dfs(v)
            stack.remove(u)

        dfs(hir.root_link)

    # dof mapping consistency
    movable_set = {j.joint_id for j in hir.joints.values() if is_movable_joint_type(j.joint_type)}
    mapped_set = set(hir.graph.joint_name_to_joint_i.keys())
    if movable_set != mapped_set:
        report.errors.append("joint_name_to_joint_i mismatch with movable joints")
    if hir.graph.dof_count != len(mapped_set):
        report.errors.append("dof_count mismatch with joint_name_to_joint_i")

    # tips should be leaf links
    leaf_links = {lk for lk, children in hir.graph.child_map.items() if len(children) == 0}
    for tip in hir.tips:
        if tip.tip_link not in leaf_links:
            report.warnings.append(f"tip {tip.tip_link} is not a leaf link")

    if len(hir.tips) == 0:
        report.errors.append("no tips detected")

    return report


def validate_hir_basic(hir: HandHIR) -> List[str]:
    """Backward-compatible helper used by early drafts."""
    return validate_hir(hir).errors
