from __future__ import annotations

import json
from pathlib import Path

from embodiment.cli.gen_assets import generate_assets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_pipeline_generate_small(tmp_path: Path):
    root = _repo_root()
    leap_urdf = root / "source" / "anymani" / "assets" / "leap_hand_sim_urdf" / "leap_hand" / "robot.urdf"
    out_dir = tmp_path / "leap_small"
    summary = generate_assets(
        input_urdf=str(leap_urdf),
        out_dir=str(out_dir),
        family="leap",
        count=3,
        render_png=False,
    )

    assert summary["generated_count"] == 3
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "manifest.csv").exists()
    assert (out_dir / "urdf_visualizer_index.md").exists()

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["generated_count"] == 3
    assert len(data["items"]) == 3


def test_pipeline_render_png(tmp_path: Path):
    root = _repo_root()
    leap_urdf = root / "source" / "anymani" / "assets" / "leap_hand_sim_urdf" / "leap_hand" / "robot.urdf"
    out_dir = tmp_path / "leap_png"
    summary = generate_assets(
        input_urdf=str(leap_urdf),
        out_dir=str(out_dir),
        family="leap",
        count=2,
        render_png=True,
    )

    if summary["render"].get("error") is None:
        assert summary["render"]["rendered_count"] >= 1
        assert (out_dir / "png" / "index.md").exists()
