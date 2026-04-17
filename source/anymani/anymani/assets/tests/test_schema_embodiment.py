import pytest

from assets.asset_schema_core import InertialCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg


def _make_joint(name: str, parent: str, child: str | None = None) -> JointCfg:
    return JointCfg(
        name=name,
        parent=parent,
        child=child,
        collisions=[{"type": "box", "size": (0.01, 0.01, 0.02)}],
        inertial=InertialCfg.from_box((0.01, 0.01, 0.02), density=600.0),
    )


def test_finger_cfg_rejects_broken_chain():
    first = _make_joint("0", "palm", "link_a")
    second = _make_joint("1", "wrong_parent", "link_b")
    with pytest.raises(ValueError):
        FingerCfg(name="index", joints=[first, second])


def test_hand_cfg_accepts_minimal_valid_structure():
    first = _make_joint("0", "palm", "link_a")
    second = _make_joint("1", "link_a", "link_b")
    finger = FingerCfg(name="index", joints=[first, second])
    hand = HandCfg(name="demo_hand", palm=PalmCfg(), fingers=[finger])
    assert hand.dof_count == 2
