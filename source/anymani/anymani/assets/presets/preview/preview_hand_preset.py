"""Quick-check a hand preset combination via the formal `HandGenerator` path."""

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.assets.presets import make_human_like_builder_cfg


parser = argparse.ArgumentParser(description="Quick-check a full hand URDF/bundle from preset combinations.")
parser.add_argument("--family", type=str, required=True, help="hand family 名，例如 `allegro` / `leap`。")
parser.add_argument("--handedness", type=str, default="right", choices=("left", "right"), help="左右手。")
parser.add_argument("--name", type=str, default="hand_preview", help="导出时使用的 hand 名。")
parser.add_argument("--palm-preset", type=str, default=None, help="palm preset 名；缺省时按 family 推成 `com_{family}`。")
parser.add_argument("--finger-preset", type=str, default=None, help="非拇指 finger preset 名；缺省时按 family 推成 `{family}_non_thumb_v1`。")
parser.add_argument("--thumb-preset", type=str, default=None, help="thumb preset 名；缺省时按 family 推成 `{family}_thumb_v1`。")
parser.add_argument(
    "--artifact-level",
    type=str,
    default="urdf",
    choices=("hand_cfg", "urdf", "bundle"),
    help="正式 HandGenerator 导出粒度；quick-check 默认只写 URDF。",
)
parser.add_argument("--output-dir", type=str, default=None, help="导出目录；不写则自动创建临时目录。")


def main() -> int:
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir, prefix="anymani_hand_preview")

    palm_preset = args.palm_preset or f"com_{args.family}"
    finger_preset = args.finger_preset or f"{args.family}_non_thumb_v1"
    thumb_preset = args.thumb_preset or f"{args.family}_thumb_v1"

    made_cfg = make_human_like_builder_cfg(
        name=args.name,
        family=args.family,
        handedness=args.handedness,
        palm_cfg=palm_preset,
        finger_cfg=finger_preset,
        thumb_cfg=thumb_preset,
    )
    generator_cfg = HandGeneratorCfg(
        mode="full",
        artifact_level=args.artifact_level,
        output_dir=output_dir,
        Made=made_cfg,
    )
    result = HandGenerator(generator_cfg).generate()
    if result is None:
        raise RuntimeError("HandGenerator returned None for preview request.")

    written = []
    if result.urdf_path is not None:
        written.append(result.urdf_path)
    if result.sidecar_path is not None:
        written.append(result.sidecar_path)
    print_export_result(
        label=f"hand preview [{args.family}:{palm_preset} + {finger_preset} + {thumb_preset}]",
        output_dir=output_dir,
        written=written,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
