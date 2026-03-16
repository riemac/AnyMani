from __future__ import annotations

from copy import deepcopy

from embodiment.schema.hir_v01 import HandHIR


def scale_finger_origins(hir: HandHIR, finger_id: str, z_scale: float = 1.05) -> HandHIR:
    """Scale z-offset of one finger chain's joint origins."""
    out = deepcopy(hir)
    finger = next((f for f in out.fingers if f.finger_id == finger_id), None)
    if finger is None:
        return out

    for jn in finger.chain_joint_ids:
        js = out.joints.get(jn)
        if js is None:
            continue
        x, y, z = js.pose_parent_to_joint.xyz
        js.pose_parent_to_joint.xyz = (x, y, z * z_scale)
    return out


def widen_joint_limits(hir: HandHIR, ratio: float = 1.02, max_abs: float = 3.14) -> HandHIR:
    """Widen movable-joint limits around their centers."""
    out = deepcopy(hir)
    for js in out.joints.values():
        if js.limit_lower is None or js.limit_upper is None:
            continue
        center = 0.5 * (js.limit_lower + js.limit_upper)
        half = 0.5 * (js.limit_upper - js.limit_lower) * ratio
        lower = max(-max_abs, center - half)
        upper = min(max_abs, center + half)
        if lower < upper:
            js.limit_lower = lower
            js.limit_upper = upper
    return out
