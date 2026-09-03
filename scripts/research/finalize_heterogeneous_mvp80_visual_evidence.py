r"""封闭strict-v5五页全景/近景截图的identity与审查清单。

本工具不自行推断视觉质量；它验证80个close-ups、5×3个时间帧及mapping与最终manifest一一对应，
并把调用方已完成的多模态审查结论和数值cold-reset证据共同写入可审计JSON。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL = ROOT / "outputs/pregrasp/visual/heterogeneous_rotation_mvp80_dexcube_s1p1_v5"
DEFAULT_MANIFEST = ROOT / (
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml"
)
DEFAULT_OUTPUT = DEFAULT_VISUAL / "visual-evidence.json"


def _sha256(path: Path) -> str:
    r"""计算截图、mapping和数值artifact的SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_record(path: Path) -> dict[str, object]:
    r"""验证PNG可解码并返回path、像素shape与digest。"""

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width < 640 or height < 360:
        raise RuntimeError(f"visual evidence image is too small for morphology review: {path}")
    return {"path": str(path), "sha256": _sha256(path), "width": width, "height": height}


def main() -> int:
    r"""验证截图coverage并发布用户授权的primary multimodal审查结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verdict", choices=("passed", "rejected"), required=True)
    args = parser.parse_args()
    visual_root = args.visual_dir if args.visual_dir.is_absolute() else ROOT / args.visual_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = tuple(int(row) for row in manifest["selected_rows"])
    if len(rows) != 80 or len(set(rows)) != 80:
        raise RuntimeError("visual evidence requires the final unique 80-row manifest")

    pages = []
    observed_offsets: set[int] = set()
    for offset in range(0, 80, 16):
        stop = offset + 15
        mapping_path = visual_root / f"page-{offset:02d}-{stop:02d}-mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapped_rows = tuple(int(item["dataset_row"]) for item in mapping)
        mapped_offsets = tuple(int(item["manifest_offset"]) for item in mapping)
        if mapped_rows != rows[offset : stop + 1] or mapped_offsets != tuple(range(offset, stop + 1)):
            raise RuntimeError(f"visual page {offset}-{stop} mapping disagrees with final manifest")
        overview = [
            _image_record(visual_root / f"page-{offset:02d}-{stop:02d}-rank0-hold-step{step:04d}.png")
            for step in (1, 24, 120)
        ]
        closeups = []
        for item in mapping:
            manifest_offset = int(item["manifest_offset"])
            matches = tuple(
                visual_root.glob(
                    f"offset-{manifest_offset:02d}-row-{int(item['dataset_row']):04d}-"
                    f"asset-{item['asset_id']}-rank0-hold-closeup.png"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(f"visual offset {manifest_offset} requires exactly one close-up")
            closeups.append(_image_record(matches[0]))
            observed_offsets.add(manifest_offset)
        sheet = visual_root / f"contact-sheet-offset-{offset:02d}-{stop:02d}.png"
        pages.append(
            {
                "offset_range": [offset, stop],
                "mapping_path": str(mapping_path),
                "mapping_sha256": _sha256(mapping_path),
                "overview_frames": overview,
                "closeups": closeups,
                "contact_sheet": _image_record(sheet),
                "review": {
                    "verdict": args.verdict,
                    "criteria": [
                        "upright object",
                        "object over palm/support region",
                        "available fingers form a non-degenerate envelope",
                        "no visible gross interpenetration or ejection",
                    ],
                },
            }
        )
    if observed_offsets != set(range(80)):
        raise RuntimeError("visual evidence does not cover every final manifest offset exactly once")

    strict_summary = ROOT / "outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/final-summary.json"
    runtime_smoke = ROOT / "outputs/hetero/runtime-smokes/palm-rotation-mvp-mvp80.json"
    result = {
        "artifact_type": "anymani.good_pregrasp.visual_evidence",
        "schema_version": "1.0.0",
        "reviewer": "primary-agent-multimodal-review-delegated-by-user",
        "verdict": args.verdict,
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "strict_summary": {"path": str(strict_summary), "sha256": _sha256(strict_summary)},
        "runtime_hold": {"path": str(runtime_smoke), "sha256": _sha256(runtime_smoke)},
        "pages": pages,
        "interpretation": (
            "Visual evidence checks reset pose plausibility; strict physical metrics and the 80/80 one-second task hold "
            "remain the decisive dynamic safety evidence. This verdict is not a rotation-learning claim."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print({"output": str(output_path), "pages": len(pages), "closeups": len(observed_offsets), "verdict": args.verdict})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
