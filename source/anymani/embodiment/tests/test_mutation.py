from __future__ import annotations

from pathlib import Path

from embodiment.io.urdf_to_hir import parse_urdf_to_hir
from embodiment.mutate.geometry import scale_box_collisions_of_link
from embodiment.mutate.kinematics import scale_finger_origins, widen_joint_limits
from embodiment.mutate.topology import drop_last_joint_of_finger
from embodiment.validate.checks import validate_hir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_topology_mutation_keeps_validity():
    src = _repo_root() / "source" / "anymani" / "assets" / "leap_hand_sim_urdf" / "leap_hand" / "robot.urdf"
    hir = parse_urdf_to_hir(str(src), family="leap")
    if not hir.fingers:
        return
    out = drop_last_joint_of_finger(hir, hir.fingers[0].finger_id)
    report = validate_hir(out)
    assert report.passed, f"topology mutation errors: {report.errors[:3]}"


def test_kinematics_mutation_keeps_validity():
    src = _repo_root() / "source" / "anymani" / "assets" / "leap_hand_sim_urdf" / "leap_hand" / "robot.urdf"
    hir = parse_urdf_to_hir(str(src), family="leap")
    if not hir.fingers:
        return
    out = scale_finger_origins(hir, hir.fingers[0].finger_id, z_scale=1.03)
    out = widen_joint_limits(out, ratio=1.01)
    report = validate_hir(out)
    assert report.passed, f"kinematics mutation errors: {report.errors[:3]}"


def test_geometry_mutation_keeps_validity():
    src = _repo_root().parent / "hora" / "assets" / "allegro" / "allegro.urdf"
    hir = parse_urdf_to_hir(str(src), family="allegro")
    if not hir.links:
        return
    first_link = sorted(hir.links.keys())[0]
    out = scale_box_collisions_of_link(hir, first_link, (1.01, 1.0, 0.99))
    report = validate_hir(out)
    assert report.passed, f"geometry mutation errors: {report.errors[:3]}"
