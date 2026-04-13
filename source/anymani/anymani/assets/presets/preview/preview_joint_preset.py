"""Quick-check a single joint sliced from a finger preset.

# Question:
当前项目还没有独立的 joint preset 注册表；
因此 joint 级 quick-check 先采用“finger preset + joint index 裁剪”的工作流。
"""

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.exporter import JointExporter, JointExporterCfg
from anymani.assets.presets import get_finger_builder_preset


parser = argparse.ArgumentParser(description="Quick-check a standalone joint URDF sliced from a finger preset.")
parser.add_argument("--finger-preset", type=str, required=True, help="已注册的 finger preset 名。")
parser.add_argument("--joint-index", type=int, default=0, help="要裁出的 joint 序号。")
parser.add_argument("--finger-name", type=str, default="preview_finger", help="用于构建局部 finger 的逻辑名。")
parser.add_argument("--parent-link", type=str, default="palm", help="局部 finger 的 parent link 名。")
parser.add_argument("--output-dir", type=str, default=None, help="导出目录；不写则自动创建临时目录。")


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
