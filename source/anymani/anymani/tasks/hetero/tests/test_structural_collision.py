r"""Generated canonical structural collision pair语义合同。"""

from __future__ import annotations

from anymani.tasks.hetero.contact_layout import build_canonical_contact_layout, structural_collision_filter_pairs


def test_structural_filter_removes_palm_and_same_finger_but_keeps_cross_finger() -> None:
    r"""Palm–finger与same-finger进入filter，index–thumb cross-finger不进入。"""

    layout = build_canonical_contact_layout()
    pairs = set(structural_collision_filter_pairs(layout.palm_link, layout.finger_link_chains))
    assert tuple(sorted(("palm", "index_tip"))) in pairs
    assert tuple(sorted(("index_link_j0", "index_tip"))) in pairs
    assert tuple(sorted(("index_tip", "thumb_tip"))) not in pairs
    assert all(left != right for left, right in pairs)
