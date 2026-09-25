"Declares preview joint preset values used by reproducible hand construction."

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.exporter import JointExporter, JointExporterCfg  # noqa: E402
from anymani.assets.presets import get_finger_builder_preset  # noqa: E402


parser = argparse.ArgumentParser(description="Quick-check a standalone joint URDF sliced from a finger preset.")
parser.add_argument("--finger-preset", type=str, required=True, help='Name of a registered finger preset.')
parser.add_argument("--joint-index", type=int, default=0, help='Index of the joint to extract.')
parser.add_argument("--finger-name", type=str, default="preview_finger", help='Logical name for the preview finger.')
parser.add_argument("--parent-link", type=str, default="palm", help='Parent link name for the preview finger.')
parser.add_argument("--output-dir", type=str, default=None, help='Export directory; create a temporary directory if omitted.')


def main() -> int:
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir, prefix="anymani_joint_preview")

    builder_cfg = get_finger_builder_preset(args.finger_preset).replace(
        name=args.finger_name,
        parent_link=args.parent_link,
    )
    finger = builder_cfg.class_type(builder_cfg).build()

    if not 0 <= args.joint_index < len(finger.joints):
        raise IndexError(f"joint-index {args.joint_index} out of range for preset {args.finger_preset!r}")

    joint = finger.joints[args.joint_index]
    export_result = JointExporter(JointExporterCfg()).export(joint, output_dir)
    if not export_result.ok:
        raise RuntimeError(f"JointExporter failed: {export_result.errors}")

    print_export_result(
        label=f"joint preview [{args.finger_preset}#{args.joint_index}]",
        output_dir=output_dir,
        written=export_result.written,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
