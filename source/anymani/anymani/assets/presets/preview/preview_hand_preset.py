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
from anymani.assets.presets import (
    get_hand_builder_preset_data,
    make_human_like_builder_cfg,
    make_human_like_builder_cfg_from_preset,
)


parser = argparse.ArgumentParser(description="Quick-check a full hand URDF/bundle from preset combinations.")
parser.add_argument("--hand-preset", type=str, default=None, help="已注册的 hand preset 名；提供后可直接整手预览。")
parser.add_argument("--family", type=str, default=None, help="hand family 名，例如 `allegro` / `leap`。")
parser.add_argument("--handedness", type=str, default=None, choices=("left", "right"), help="左右手；hand preset 路径下可作为覆盖项。")
parser.add_argument("--name", type=str, default=None, help="导出时使用的 hand 名；缺省时沿用 hand preset 内建名字。")
parser.add_argument("--palm-preset", type=str, default=None, help="palm preset 名；缺省时按 family 推成 `com_{family}`。")
parser.add_argument("--finger-preset", type=str, default=None, help="非拇指 finger preset 名；缺省时按 family 推成 `{family}_non_thumb_v1`。")
parser.add_argument("--thumb-preset", type=str, default=None, help="thumb preset 名；缺省时按 family 推成 `{family}_thumb_v1`。")
parser.add_argument(
    "--connectivity-preset",
    type=str,
    default=None,
    help="可选的 hand-level connectivity preset 名；只描述合法 joint / child-link 组合，不绑定 fingertip。当前要求同时提供 `--hand-preset`。",
)
parser.add_argument(
    "--artifact-level",
    type=str,
    default="urdf",
    choices=("hand_cfg", "urdf", "bundle"),
    help="正式 HandGenerator 导出粒度；quick-check 默认只写 URDF。",
)
parser.add_argument(
    "--output-layout",
    type=str,
    default="recursive",
    choices=("flat", "recursive"),
    help="当启用 connectivity preset 时，控制 pre-made 产物采用递归式还是扁平式目录。",
)
parser.add_argument("--output-dir", type=str, default=None, help="导出目录；不写则自动创建临时目录。")


def _build_made_cfg_and_label(args) -> tuple[object, str]:
    r"""根据 CLI 输入构造 `Made` cfg，并返回用于打印的预览标签。

    当前 hand preview 支持两条入口：

    1. `--hand-preset xxx`：直接使用 hand preset 作为整手组合锚点；
    2. `--family ... --palm-preset ... --finger-preset ... --thumb-preset ...`：
       沿用原先的拆分组合入口。

    之所以保留两条路，而不是强迫只剩 hand preset，是因为科研调参有两类场景：

    - 快速复现某个稳定 hand 组合时，更适合直接喊 hand preset 名；
    - 正在试探 palm/finger/thumb 组合边界时，仍需要拆分入口做局部替换。

    Args:
        args: `argparse` 解析后的命令行参数。

    Returns:
        tuple[object, str]: `(made_cfg, label)`，其中 `made_cfg` 会直接喂给
        `HandGeneratorCfg.Made`，`label` 用于统一打印导出来源。
    """

    if args.hand_preset is not None:
        preset_data = get_hand_builder_preset_data(args.hand_preset)  # 先读出 hand preset 原始组合，便于显示与覆盖
        effective_family = args.family or preset_data["family"]  # family 默认沿用 hand preset
        effective_handedness = args.handedness or preset_data["handedness"]  # handedness 缺省沿用 hand preset
        effective_name = args.name or preset_data["name"]  # hand 名优先显式覆盖，否则沿用 preset 内建名
        effective_palm = args.palm_preset or preset_data["palm_cfg"]  # palm/finger/thumb 都允许把 hand preset 当模板覆盖
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

    effective_handedness = args.handedness or "right"  # 历史 split 路径默认仍保持右手
    effective_name = args.name or "hand_preview"  # 历史 split 路径默认 hand 名
    palm_preset = args.palm_preset or f"com_{args.family}"  # 沿用旧逻辑：palm 默认推 `com_{family}`
    finger_preset = args.finger_preset or f"{args.family}_non_thumb_v1"  # 非拇指默认推 family 对应 preset
    thumb_preset = args.thumb_preset or f"{args.family}_thumb_v1"  # 拇指默认推 family 对应 thumb preset
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
    generator_cfg = HandGeneratorCfg(
        mode="full",
        artifact_level=args.artifact_level,
        output_dir=output_dir,
        handedness=getattr(made_cfg, "handedness", "all"),
        Made=made_cfg,
        hand_presets=[args.hand_preset] if args.connectivity_preset is not None and args.hand_preset is not None else [],
        connectivity_presets=(
            {args.hand_preset: [args.connectivity_preset]}
            if args.connectivity_preset is not None and args.hand_preset is not None
            else None
        ),
        output_layout=args.output_layout,
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
