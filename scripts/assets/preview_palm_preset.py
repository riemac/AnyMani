"""Quick-check a palm preset by exporting a standalone palm URDF."""

from __future__ import annotations

import argparse
import sys

from _common import bootstrap_python_path, infer_family_from_palm_preset, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.exporter import PalmExporter, PalmExporterCfg
from anymani.assets.presets import resolve_human_like_mounts, resolve_palm_builder_cfg


parser = argparse.ArgumentParser(description="Quick-check a standalone palm URDF from a registered palm preset.")
parser.add_argument("--preset", type=str, required=True, help="已注册的 palm preset 名，例如 `single_box_allegro`。")
parser.add_argument("--handedness", type=str, default="right", choices=("left", "right"), help="用于解析 mount preview 的左右手。")
parser.add_argument("--output-dir", type=str, default=None, help="导出目录；不写则自动创建临时目录。")


def main() -> int:
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir, prefix="anymani_palm_preview")

    palm_cfg = resolve_palm_builder_cfg(args.preset)
    palm = palm_cfg.class_type(palm_cfg).build()

    family = infer_family_from_palm_preset(args.preset)
    mounts = resolve_human_like_mounts(
        family=family,
        handedness=args.handedness,
        palm_cfg=palm_cfg,
    )
    if mounts:
        palm = palm.replace(
            metadata={
                **palm.metadata,
                "preview_preset": args.preset,
                "finger_mounts": mounts,
            }
        )

    export_result = PalmExporter(PalmExporterCfg()).export(palm, output_dir)
    if not export_result.ok:
        raise RuntimeError(f"PalmExporter failed: {export_result.errors}")

    print_export_result(label=f"palm preview [{args.preset}]", output_dir=output_dir, written=export_result.written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
