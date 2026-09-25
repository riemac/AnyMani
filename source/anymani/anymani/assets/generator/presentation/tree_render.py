"Renders the hand tree for inspection without affecting geometry identity."

from __future__ import annotations

import math
from typing import Any

from ...asset_base import HandCfg


def _axis_label(axis: tuple[float, float, float]) -> str:

    labels = ("X", "Y", "Z")
    idx = max(range(3), key=lambda i: abs(axis[i]))
    sign = "-" if axis[idx] < 0 else "+"
    return f"{sign}{labels[idx]}"


def _link_length(origin: Any) -> float:

    if origin is None:
        return 0.0
    x, y, z = origin.pos
    return math.sqrt(x * x + y * y + z * z)


def _fmt_vec(v: tuple[float, float, float]) -> str:

    x, y, z = v
    return f"({x:+.3f}, {y:+.3f}, {z:+.3f})"


def render_hand_tree_txt(hand_cfg: HandCfg) -> str:
    "Renders a deterministic text tree from the ordered hand joints and links."

    lines: list[str] = []


    dof = hand_cfg.dof_count
    lines.append(
        f"{hand_cfg.palm.name}"
        f"  [family={hand_cfg.family} · {hand_cfg.handedness} · dof={dof}]"
    )

    n_fingers = len(hand_cfg.fingers)
    for f_idx, finger in enumerate(hand_cfg.fingers):
        is_last_finger = f_idx == n_fingers - 1
        f_branch = "└── " if is_last_finger else "├── "
        f_cont = "    " if is_last_finger else "│   "


        mount_pos = _fmt_vec(finger.mount.pos) if finger.mount else "(+0.000, +0.000, +0.000)"
        mount_rpy = _fmt_vec(finger.mount.rpy) if finger.mount else "(+0.000, +0.000, +0.000)"
        lines.append(f"{f_branch}[{finger.name}]  mount={mount_pos} m  rpy={mount_rpy} rad")

        n_joints = len(finger.joints)
        for j_idx, joint in enumerate(finger.joints):
            is_last = j_idx == n_joints - 1
            j_prefix = f"{f_cont}{'└── ' if is_last else '├── '}"


            axis_str = _axis_label(joint.axis) if joint.joint_type != "fixed" else "fixed"
            length = _link_length(joint.origin)


            limit_str = ""
            if joint.limit is not None and joint.joint_type == "revolute":
                lo = joint.limit.lower
                hi = joint.limit.upper
                limit_str = f"  [{lo:+.2f}, {hi:+.2f}] rad"

            tip_str = "  ★ TIP" if joint.is_tip else ""

            lines.append(
                f"{j_prefix}{joint.name}  →  {joint.child}"
                f"  {joint.joint_type}  axis={axis_str}  len={length:.4f} m"
                f"{limit_str}{tip_str}"
            )

    return "\n".join(lines)
__all__ = [
    "render_hand_tree_txt",
]
