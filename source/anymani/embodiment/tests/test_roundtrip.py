from __future__ import annotations

from pathlib import Path

from embodiment.io.hir_to_urdf import emit_hir_to_urdf
from embodiment.io.urdf_to_hir import parse_urdf_to_hir
from embodiment.validate.checks import validate_hir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_roundtrip_leap(tmp_path: Path):
    root = _repo_root()
    src = root / "source" / "anymani" / "assets" / "leap_hand_sim_urdf" / "leap_hand" / "robot.urdf"
    assert src.exists(), f"missing test asset: {src}"

    hir = parse_urdf_to_hir(str(src), family="leap")
    report = validate_hir(hir)
    assert report.passed, f"validation errors: {report.errors[:3]}"

    out = tmp_path / "leap_roundtrip.urdf"
    emit_hir_to_urdf(hir, str(out))
    assert out.exists()

    hir2 = parse_urdf_to_hir(str(out), family="leap")
    report2 = validate_hir(hir2)
    assert report2.passed, f"roundtrip errors: {report2.errors[:3]}"


def test_roundtrip_allegro(tmp_path: Path):
    root = _repo_root()
    src = root.parent / "hora" / "assets" / "allegro" / "allegro.urdf"
    assert src.exists(), f"missing test asset: {src}"

    hir = parse_urdf_to_hir(str(src), family="allegro")
    report = validate_hir(hir)
    assert report.passed, f"validation errors: {report.errors[:3]}"

    out = tmp_path / "allegro_roundtrip.urdf"
    emit_hir_to_urdf(hir, str(out))
    assert out.exists()
