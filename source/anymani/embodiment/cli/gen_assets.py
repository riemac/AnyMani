from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from embodiment.io.hir_to_urdf import emit_hir_to_urdf
from embodiment.io.urdf_to_hir import parse_urdf_to_hir
from embodiment.mutate.geometry import scale_box_collisions_of_link, scale_sphere_collisions_of_link
from embodiment.mutate.kinematics import scale_finger_origins, widen_joint_limits
from embodiment.mutate.topology import drop_last_joint_of_finger
from embodiment.validate.checks import validate_hir
from embodiment.visualize.render_variants import render_urdf_directory_to_png


def _apply_recipe(base_hir, sample_i: int):
    cur = base_hir
    if cur.fingers:
        finger = cur.fingers[sample_i % len(cur.fingers)]
        if sample_i % 4 == 0:
            cur = drop_last_joint_of_finger(cur, finger.finger_id)
            # refresh finger reference after topology change
            if cur.fingers:
                finger = cur.fingers[min(sample_i % len(cur.fingers), len(cur.fingers) - 1)]
        scale = 0.96 + 0.02 * (sample_i % 5)
        cur = scale_finger_origins(cur, finger.finger_id, z_scale=scale)

    cur = widen_joint_limits(cur, ratio=1.0 + 0.01 * (sample_i % 3))

    if cur.links:
        first_link = sorted(cur.links.keys())[0]
        bx = 0.98 + 0.02 * ((sample_i % 4) / 3.0)
        by = 0.98 + 0.02 * (((sample_i + 1) % 4) / 3.0)
        bz = 0.98 + 0.02 * (((sample_i + 2) % 4) / 3.0)
        cur = scale_box_collisions_of_link(cur, first_link, (bx, by, bz))
        cur = scale_sphere_collisions_of_link(cur, first_link, 0.98 + 0.03 * ((sample_i % 5) / 4.0))

    return cur


def _meta_dict(hir, variant_id: str, source_urdf: str, errors: list[str], warnings: list[str]):
    return {
        "schema_version": hir.hir_version,
        "variant_id": variant_id,
        "embodiment_name": f"{hir.family}_{hir.hand_id}_{variant_id}",
        "hand_id": hir.hand_id,
        "family": hir.family,
        "handedness": hir.handedness,
        "source_urdf": source_urdf,
        "dof_count": hir.graph.dof_count,
        "joint_name_to_joint_i": hir.graph.joint_name_to_joint_i,
        "tips": [t.tip_link for t in hir.tips],
        "validation": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        },
    }


def generate_assets(
    input_urdf: str,
    out_dir: str,
    family: str,
    count: int = 20,
    seed: int = 42,
    handedness: str = "unknown",
    render_png: bool = True,
) -> dict:
    random.seed(seed)

    out_root = Path(out_dir)
    urdf_dir = out_root / "urdf"
    meta_dir = out_root / "meta"
    png_dir = out_root / "png"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    base = parse_urdf_to_hir(
        urdf_path=input_urdf,
        family=family,
        handedness=handedness,
        source_urdf=input_urdf,
    )

    manifest = []
    errors = []
    generated = 0
    attempts = 0
    max_attempts = max(count * 8, 40)

    while generated < count and attempts < max_attempts:
        attempts += 1
        variant_id = f"{generated:04d}"
        candidate = _apply_recipe(base, sample_i=attempts)
        report = validate_hir(candidate)
        if not report.passed:
            errors.append({"attempt": attempts, "errors": report.errors})
            continue

        urdf_path = urdf_dir / f"{variant_id}.urdf"
        meta_path = meta_dir / f"{variant_id}.json"
        emit_hir_to_urdf(candidate, str(urdf_path))

        meta = _meta_dict(
            hir=candidate,
            variant_id=variant_id,
            source_urdf=input_urdf,
            errors=report.errors,
            warnings=report.warnings,
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        manifest.append(
            {
                "variant_id": variant_id,
                "urdf": str(urdf_path),
                "meta": str(meta_path),
                "dof_count": candidate.graph.dof_count,
                "tip_count": len(candidate.tips),
                "warning_count": len(report.warnings),
            }
        )
        generated += 1

    pass_rate = 0.0 if attempts == 0 else generated / attempts

    manifest_json = out_root / "manifest.json"
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "family": family,
                "input_urdf": input_urdf,
                "requested_count": count,
                "generated_count": generated,
                "attempts": attempts,
                "pass_rate": pass_rate,
                "items": manifest,
                "failed_attempts": errors,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    manifest_csv = out_root / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant_id", "urdf", "meta", "dof_count", "tip_count", "warning_count"])
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)

    # Index for URDF Visualizer extension usage.
    viewer_index = out_root / "urdf_visualizer_index.md"
    with viewer_index.open("w", encoding="utf-8") as f:
        f.write("# URDF Visualizer Index\n\n")
        f.write("Open these URDF files with URDF Visualizer extension.\n\n")
        for row in manifest:
            f.write(f"- {Path(row['urdf']).name}\n")

    render_result = {"rendered_count": 0, "failed_count": 0, "error": None}
    if render_png:
        try:
            render_result = render_urdf_directory_to_png(
                urdf_dir=str(urdf_dir),
                output_dir=str(png_dir),
            )
        except Exception as exc:  # pragma: no cover
            render_result = {"rendered_count": 0, "failed_count": generated, "error": str(exc)}

    summary = {
        "family": family,
        "generated_count": generated,
        "requested_count": count,
        "attempts": attempts,
        "pass_rate": pass_rate,
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "viewer_index": str(viewer_index),
        "render": render_result,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_urdf", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--handedness", default="unknown", choices=["left", "right", "unknown"])
    parser.add_argument("--no_render_png", action="store_true")
    args = parser.parse_args()

    summary = generate_assets(
        input_urdf=args.input_urdf,
        out_dir=args.out_dir,
        family=args.family,
        count=args.count,
        seed=args.seed,
        handedness=args.handedness,
        render_png=not args.no_render_png,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
