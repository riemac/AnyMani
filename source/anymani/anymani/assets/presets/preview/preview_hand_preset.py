"Declares preview hand preset values used by reproducible hand construction."

from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from _common import bootstrap_python_path, print_export_result, resolve_output_dir
else:
    from ._common import bootstrap_python_path, print_export_result, resolve_output_dir

bootstrap_python_path()

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg  # noqa: E402
from anymani.assets.presets import (  # noqa: E402
    get_hand_connectivity_preset_data,
    get_hand_builder_preset_data,
    make_human_like_builder_cfg,
    make_human_like_builder_cfg_from_preset,
)


parser = argparse.ArgumentParser(description="Quick-check a full hand URDF/bundle from preset combinations.")
parser.add_argument("--hand-preset", type=str, default=None, help='Registered hand preset; set it to preview a complete hand.')
parser.add_argument("--family", type=str, default=None, help='Hand family, for example `allegro` or `leap`.')
parser.add_argument("--handedness", type=str, default=None, choices=("left", "right"), help='Hand side; may override the hand preset.')
parser.add_argument("--name", type=str, default=None, help='Logical hand name for export; defaults to the preset name.')
parser.add_argument("--palm-preset", type=str, default=None, help='Palm preset; defaults to `com_{family}` for the selected family.')
parser.add_argument("--finger-preset", type=str, default=None, help='Non-thumb finger preset; defaults to `{family}_non_thumb_v1` for the selected family.')
parser.add_argument("--thumb-preset", type=str, default=None, help='Thumb preset; defaults to `{family}_thumb_v1` for the selected family.')
parser.add_argument(
    "--connectivity-preset",
    type=str,
    default=None,
    help='Optional hand-level connectivity preset for legal joint/child-link combinations; it does not select a fingertip. Requires `--hand-preset`.',
)
parser.add_argument(
    "--artifact-level",
    type=str,
    default="urdf",
    choices=("hand_cfg", "urdf", "bundle"),
    help='HandGenerator artifact level; quick checks write only the URDF by default.',
)
parser.add_argument(
    "--output-dir", type=str, default=None, help='Export directory; create a temporary directory if omitted.')


def _build_made_cfg_and_label(args) -> tuple[object, str]:

    if args.hand_preset is not None:
        preset_data = get_hand_builder_preset_data(args.hand_preset)
        effective_family = args.family or preset_data["family"]
        effective_handedness = args.handedness or preset_data["handedness"]
        effective_name = args.name or preset_data["name"]
        effective_palm = args.palm_preset or preset_data["palm_cfg"]
        effective_finger = args.finger_preset or preset_data["finger_cfg"]
        effective_thumb = args.thumb_preset or preset_data["thumb_cfg"]
        made_cfg = make_human_like_builder_cfg_from_preset(
            args.hand_preset,
            name=effective_name,
            family=effective_family,
            handedness=effective_handedness,
            palm_cfg=effective_palm,
            finger_cfg=effective_finger,
            thumb_cfg=effective_thumb,
        )
        label = (
            f"hand preview [{args.hand_preset}:{effective_handedness}"
            f" -> {effective_family}:{effective_palm} + {effective_finger} + {effective_thumb}]"
        )
        return made_cfg, label

    if args.family is None:
        parser.error("Either provide --hand-preset, or provide --family for the split preset path.")

    effective_handedness = args.handedness or "right"
    effective_name = args.name or "hand_preview"
    palm_preset = args.palm_preset or f"com_{args.family}"
    finger_preset = args.finger_preset or f"{args.family}_non_thumb_v1"
    thumb_preset = args.thumb_preset or f"{args.family}_thumb_v1"
    made_cfg = make_human_like_builder_cfg(
        name=effective_name,
        family=args.family,
        handedness=effective_handedness,
        palm_cfg=palm_preset,
        finger_cfg=finger_preset,
        thumb_cfg=thumb_preset,
    )
    label = f"hand preview [{args.family}:{palm_preset} + {finger_preset} + {thumb_preset}]"
    return made_cfg, label


def main() -> int:
    args = parser.parse_args()
    if args.connectivity_preset is not None and args.hand_preset is None:
        parser.error("`--connectivity-preset` currently requires `--hand-preset`, because pre-made connectivity is keyed by base hand preset.")

    output_dir = resolve_output_dir(args.output_dir, prefix="anymani_hand_preview")
    made_cfg, label = _build_made_cfg_and_label(args)
    if args.connectivity_preset is not None:
        label = f"{label} + connectivity={args.connectivity_preset}"
    slot_level_connectivity = None
    if args.connectivity_preset is not None and args.hand_preset is not None:
        hand_connectivity = get_hand_connectivity_preset_data(args.connectivity_preset)
        slot_level_connectivity = {
            args.hand_preset: {
                slot_name: [recipe_name]
                for slot_name, recipe_name in hand_connectivity.finger_slots.items()
            }
        }
    generator_cfg = HandGeneratorCfg(
        mode="made",
        artifact_level=args.artifact_level,
        output_dir=output_dir,
        handedness=getattr(made_cfg, "handedness", "all"),
        Made=made_cfg,
        hand_presets=[args.hand_preset] if args.connectivity_preset is not None and args.hand_preset is not None else [],
        connectivity_presets=slot_level_connectivity,
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
        label=label,
        output_dir=output_dir,
        written=written,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
