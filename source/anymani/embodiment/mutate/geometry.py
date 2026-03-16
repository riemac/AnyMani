from __future__ import annotations

from copy import deepcopy
from typing import Tuple

from embodiment.schema.hir_v01 import HandHIR


def scale_box_collisions_of_link(hir: HandHIR, link_id: str, scale: Tuple[float, float, float]) -> HandHIR:
    """Scale all box collision sizes under a target link."""
    out = deepcopy(hir)
    link = out.links.get(link_id)
    if link is None:
        return out
    sx, sy, sz = scale
    for c in link.collisions:
        if c.geom_type != "box" or c.size is None:
            continue
        x, y, z = c.size
        nx, ny, nz = x * sx, y * sy, z * sz
        if nx > 0 and ny > 0 and nz > 0:
            c.size = (nx, ny, nz)
    return out


def scale_sphere_collisions_of_link(hir: HandHIR, link_id: str, scale: float) -> HandHIR:
    """Scale all sphere collision radii under a target link."""
    out = deepcopy(hir)
    link = out.links.get(link_id)
    if link is None:
        return out
    for c in link.collisions:
        if c.geom_type == "sphere" and c.radius is not None:
            nr = c.radius * scale
            if nr > 0:
                c.radius = nr
    return out


def mesh_to_box_proxy(hir: HandHIR, link_id: str, proxy_size=(0.01, 0.01, 0.01)) -> HandHIR:
    """Convert mesh collisions to conservative box proxies for robustness tests."""
    out = deepcopy(hir)
    link = out.links.get(link_id)
    if link is None:
        return out
    for c in link.collisions:
        if c.geom_type != "mesh":
            continue
        c.geom_type = "box"
        c.mesh_file = None
        c.mesh_scale = None
        c.radius = None
        c.length = None
        c.size = proxy_size
    return out
