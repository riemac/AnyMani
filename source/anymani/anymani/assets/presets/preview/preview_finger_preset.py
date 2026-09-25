"Declares preview finger preset values used by reproducible hand construction."

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.exporter import FingerExporter, FingerExporterCfg  # noqa: E402
from anymani.assets.presets import get_finger_builder_preset  # noqa: E402


parser = argparse.ArgumentParser(description="Quick-check a standalone finger URDF from a registered finger preset.")
parser.add_argument("--preset", type=str, required=True, help='Name of a registered finger preset.')
parser.add_argument("--name", type=str, default="preview_finger", help='Logical finger name used for export.')
parser.add_argument("--parent-link", type=str, default="palm", help='Parent link name for the finger root.')
parser.add_argument("--output-dir", type=str, default=None, help='Export directory; create a temporary directory if omitted.')


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
