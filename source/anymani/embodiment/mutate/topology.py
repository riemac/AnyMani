from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Set

from embodiment.schema.hir_v01 import HandHIR, TipSpec, is_movable_joint_type


def _collect_descendants(start_link: str, child_map: Dict[str, List[str]]) -> Set[str]:
    out: Set[str] = set()
    stack = [start_link]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(child_map.get(cur, []))
    return out


def _rebuild_graph(hir: HandHIR):
    parent_map: Dict[str, Optional[str]] = {k: None for k in hir.links.keys()}
    child_map: Dict[str, List[str]] = {k: [] for k in hir.links.keys()}
    for j in hir.joints.values():
        if j.parent_link not in hir.links or j.child_link not in hir.links:
            continue
        parent_map[j.child_link] = j.parent_link
        child_map[j.parent_link].append(j.child_link)

    roots = [k for k, v in parent_map.items() if v is None]
    hir.root_link = sorted(roots)[0] if roots else None

    movable = sorted([j.joint_id for j in hir.joints.values() if is_movable_joint_type(j.joint_type)])
    hir.graph.root_link = hir.root_link
    hir.graph.parent_map = parent_map
    hir.graph.child_map = child_map
    hir.graph.joint_name_to_joint_i = {name: i for i, name in enumerate(movable)}
    hir.graph.dof_count = len(movable)


def _refresh_tips(hir: HandHIR):
    child_to_joint = {j.child_link: j for j in hir.joints.values()}
    tips: List[TipSpec] = []
    for link_name, children in hir.graph.child_map.items():
        if children:
            continue
        j = child_to_joint.get(link_name)
        role = "unknown_tip"
        if j is not None:
            role = "fixed_tip" if j.joint_type == "fixed" else "kinematic_tip"
        tips.append(TipSpec(tip_link=link_name, parent_link=hir.graph.parent_map.get(link_name), tip_role=role))
    hir.tips = tips


def drop_last_joint_of_finger(hir: HandHIR, finger_id: str) -> HandHIR:
    """Drop the last joint in one finger and prune its descendant subtree."""
    out = deepcopy(hir)
    finger = next((f for f in out.fingers if f.finger_id == finger_id), None)
    if finger is None or not finger.chain_joint_ids:
        return out

    last_joint_name = finger.chain_joint_ids[-1]
    last_joint = out.joints.get(last_joint_name)
    if last_joint is None:
        return out

    prune_root = last_joint.child_link
    descendants = _collect_descendants(prune_root, out.graph.child_map)

    # Remove joints touching pruned links.
    joints_to_remove = []
    for jn, js in out.joints.items():
        if js.child_link in descendants or js.parent_link in descendants:
            joints_to_remove.append(jn)
    for jn in joints_to_remove:
        out.joints.pop(jn, None)

    # Remove pruned links.
    for lk in descendants:
        out.links.pop(lk, None)

    # Rebuild finger chains.
    for f in out.fingers:
        f.chain_joint_ids = [jn for jn in f.chain_joint_ids if jn in out.joints]
        f.chain_link_ids = [lk for lk in f.chain_link_ids if lk in out.links]

    _rebuild_graph(out)
    _refresh_tips(out)
    return out
