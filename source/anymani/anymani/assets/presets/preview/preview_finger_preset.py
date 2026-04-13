"""Quick-check a finger preset by exporting a standalone finger URDF."""

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.exporter import FingerExporter, FingerExporterCfg
from anymani.assets.presets import get_finger_builder_preset


parser = argparse.ArgumentParser(description="Quick-check a standalone finger URDF from a registered finger preset.")
parser.add_argument("--preset", type=str, required=True, help="已注册的 finger preset 名。")
parser.add_argument("--name", type=str, default="preview_finger", help="导出时使用的 finger 逻辑名。")
parser.add_argument("--parent-link", type=str, default="palm", help="finger 根部 parent link 名。")
parser.add_argument("--output-dir", type=str, default=None, help="导出目录；不写则自动创建临时目录。")


def main() -> int:
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir, prefix="anymani_finger_preview")

    builder_cfg = get_finger_builder_preset(args.preset).replace(name=args.name, parent_link=args.parent_link)
    finger = builder_cfg.class_type(builder_cfg).build()

    export_result = FingerExporter(FingerExporterCfg()).export(finger, output_dir)
    if not export_result.ok:
        raise RuntimeError(f"FingerExporter failed: {export_result.errors}")

    print_export_result(label=f"finger preview [{args.preset}]", output_dir=output_dir, written=export_result.written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
