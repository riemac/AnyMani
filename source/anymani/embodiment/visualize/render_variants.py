from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_xyz(text: str | None) -> Tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    vals = text.split()
    if len(vals) != 3:
        return 0.0, 0.0, 0.0
    return float(vals[0]), float(vals[1]), float(vals[2])


def _parse_kinematic_positions(urdf_path: Path) -> Tuple[Dict[str, Tuple[float, float, float]], List[Tuple[str, str]]]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    links = []
    parent_map: Dict[str, str | None] = {}
    child_map: Dict[str, List[str]] = {}
    joint_xyz: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    for elem in list(root):
        if elem.tag.endswith("link"):
            name = elem.attrib.get("name")
            if name:
                links.append(name)
                parent_map.setdefault(name, None)
                child_map.setdefault(name, [])

    for elem in list(root):
        if not elem.tag.endswith("joint"):
            continue
        parent = None
        child = None
        xyz = (0.0, 0.0, 0.0)
        for c in list(elem):
            tag = c.tag.split("}", 1)[-1]
            if tag == "parent":
                parent = c.attrib.get("link")
            elif tag == "child":
                child = c.attrib.get("link")
            elif tag == "origin":
                xyz = _parse_xyz(c.attrib.get("xyz"))
        if not parent or not child:
            continue
        if parent not in parent_map or child not in parent_map:
            continue
        parent_map[child] = parent
        child_map[parent].append(child)
        joint_xyz[(parent, child)] = xyz

    roots = [lk for lk, p in parent_map.items() if p is None]
    root_link = sorted(roots)[0] if roots else (links[0] if links else "")

    pos: Dict[str, Tuple[float, float, float]] = {root_link: (0.0, 0.0, 0.0)}
    edges: List[Tuple[str, str]] = []
    stack = [root_link]
    while stack:
        cur = stack.pop()
        cx, cy, cz = pos[cur]
        for ch in child_map.get(cur, []):
            ox, oy, oz = joint_xyz.get((cur, ch), (0.0, 0.0, 0.0))
            pos[ch] = (cx + ox, cy + oy, cz + oz)
            edges.append((cur, ch))
            stack.append(ch)

    # isolated links fallback
    for lk in links:
        pos.setdefault(lk, (0.0, 0.0, 0.0))
    return pos, edges


def _render_one(urdf_path: Path, png_path: Path, elev: float, azim: float, dpi: int):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PNG rendering") from exc

    pos, edges = _parse_kinematic_positions(urdf_path)
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    zs = [v[2] for v in pos.values()]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=35)

    for p, c in edges:
        x1, y1, z1 = pos[p]
        x2, y2, z2 = pos[c]
        ax.plot([x1, x2], [y1, y2], [z1, z2], linewidth=1.4)

    ax.set_title(urdf_path.stem)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def render_urdf_directory_to_png(
    urdf_dir: str,
    output_dir: str,
    limit: int | None = None,
    elev: float = 24.0,
    azim: float = 38.0,
    dpi: int = 180,
) -> dict:
    """Render URDF variants in one directory into PNG snapshots.

    The current renderer is kinematic-structure based (joint tree), so it does not
    require heavy 3D dependencies and works well for quick visual QA.
    """
    src = Path(urdf_dir)
    out = Path(output_dir)
    files = sorted(src.glob("*.urdf"))
    if limit is not None:
        files = files[:limit]

    rendered = []
    failed = []
    for urdf_file in files:
        png_file = out / f"{urdf_file.stem}.png"
        try:
            _render_one(urdf_file, png_file, elev=elev, azim=azim, dpi=dpi)
            rendered.append({"urdf": str(urdf_file), "png": str(png_file)})
        except Exception as exc:  # pragma: no cover
            failed.append({"urdf": str(urdf_file), "error": str(exc)})

    index_md = out / "index.md"
    out.mkdir(parents=True, exist_ok=True)
    with index_md.open("w", encoding="utf-8") as f:
        f.write("# URDF Variant Preview\n\n")
        for item in rendered:
            urdf_name = Path(item["urdf"]).name
            png_name = Path(item["png"]).name
            f.write(f"- {urdf_name} -> {png_name}\n")

    return {
        "rendered_count": len(rendered),
        "failed_count": len(failed),
        "rendered": rendered,
        "failed": failed,
        "index": str(index_md),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    result = render_urdf_directory_to_png(
        urdf_dir=args.urdf_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )
    print(result)


if __name__ == "__main__":
    main()
