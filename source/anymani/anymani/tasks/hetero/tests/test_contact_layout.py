r"""Canonical contact sensor role/order/owner mapping合同。"""

from __future__ import annotations

import torch

from anymani.tasks.hetero.contact_layout import active_contact_sensor_mask, build_canonical_contact_layout


def test_layout_has_fixed_24_sensor_abi_and_distinct_scene_state_orders() -> None:
    r"""State为TIP+non-tip+PALM，scene为TIP+PALM+non-tip，不能按位置混用。"""

    layout = build_canonical_contact_layout()
    assert len(layout.fingertip_sensor_names) == 4
    assert len(layout.finger_non_tip_sensor_names) == 19
    assert len(layout.state_sensor_names) == 24
    assert len(layout.scene_sensor_names) == 24
    assert layout.state_sensor_names[0:4] == (
        "contact_index_tip",
        "contact_middle_tip",
        "contact_ring_tip",
        "contact_thumb_tip",
    )
    assert layout.state_sensor_names[-1] == "contact_palm"
    assert layout.scene_sensor_names[4] == "contact_palm"


def test_sensor_owner_mapping_assigns_roots_and_palm_to_palm_owner() -> None:
    r"""Roots在bad-contact role中保留，但critic owner reduction归PALM owner 0。"""

    layout = build_canonical_contact_layout()
    assert layout.sensor_owner_indices[:4] == (17, 18, 19, 20)
    assert layout.sensor_owner_indices[4:7] == (0, 0, 0)
    assert layout.sensor_owner_indices[-1] == 0
    # index_link_j0..j3的owner依次为depth-major 1,5,9,13。
    assert layout.sensor_owner_indices[7:11] == (1, 5, 9, 13)


def test_active_sensor_mask_zeroes_ghost_link_slots_without_changing_abi() -> None:
    r"""10-DoF mask仍返回24 slots；ghost joint sensors为False，PALM恒True。"""

    layout = build_canonical_contact_layout()
    joint_mask = torch.tensor(
        ((True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False),)
    )
    sensor_mask = active_contact_sensor_mask(joint_mask, layout)
    assert sensor_mask.shape == (1, 24)
    assert bool(sensor_mask[0, -1])
    assert int(sensor_mask.sum().item()) == 4 + 3 + 10 + 1
