r"""N000 native actor packet到当前canonical student packet的纯数据合同。"""

from __future__ import annotations

import math

import torch
from anymani.distill.diagnostics.recording.rl.n000_teacher import (
    build_n000_student_actor_packet,
    canonical_from_native_indices,
    contact_bits_to_owner,
    native_from_canonical_indices,
    sensor_owner_indices_from_sidecar,
)

FINGERS = ("index", "middle", "ring", "thumb")
CANONICAL_JOINTS = tuple(f"{finger}_j{depth}" for depth in range(4) for finger in FINGERS)
NATIVE_JOINTS = tuple(f"{finger}_j{depth}" for finger in FINGERS for depth in range(4))


def _sidecar() -> dict[str, object]:
    r"""构造包含root、4个revolute与TIP child语义的最小旧资产sidecar。"""

    fingers = []
    for finger in FINGERS:
        joints = []
        if finger != "thumb":
            joints.append(
                {
                    "name": f"{finger}_root_fixed",
                    "joint_type": "fixed",
                    "child": f"{finger}_root",
                    "is_tip": False,
                }
            )
        joints.extend(
            {
                "name": f"{finger}_j{depth}",
                "joint_type": "revolute",
                "child": f"{finger}_link_{depth}",
                "is_tip": False,
            }
            for depth in range(4)
        )
        joints.append(
            {
                "name": f"{finger}_tip_fixed",
                "joint_type": "fixed",
                "child": f"{finger}_tip",
                "is_tip": True,
            }
        )
        fingers.append({"name": finger, "joints": joints})
    return {"hand_cfg": {"palm": {"name": "palm"}, "fingers": fingers}}


def test_native_joint_axis_is_reordered_by_name_not_position() -> None:
    r"""旧finger-major轴必须显式变成当前depth-major轴。"""

    assert canonical_from_native_indices(NATIVE_JOINTS, CANONICAL_JOINTS) == (
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    )
    assert native_from_canonical_indices((1, 2, 3, 0)) == (3, 0, 1, 2)


def test_sidecar_contact_links_map_to_canonical_owners() -> None:
    r"""Root归PALM、revolute child归对应JOINT、TIP归对应TIP owner。"""

    tip_links = tuple(f"{finger}_tip" for finger in FINGERS)
    non_tip_links = tuple(
        link
        for finger in FINGERS
        for link in (
            *((f"{finger}_root",) if finger != "thumb" else ()),
            *(f"{finger}_link_{depth}" for depth in range(4)),
        )
    )
    owner_indices = sensor_owner_indices_from_sidecar(
        _sidecar(),
        state_sensor_links=(*tip_links, *non_tip_links, "palm"),
        canonical_joint_names=CANONICAL_JOINTS,
        canonical_finger_names=FINGERS,
    )

    assert owner_indices[:4] == (17, 18, 19, 20)
    assert owner_indices[4:9] == (0, 1, 5, 9, 13)
    assert owner_indices[-5:] == (4, 8, 12, 16, 0)

    contacts = torch.zeros(2, len(owner_indices), dtype=torch.bool)
    contacts[0, 4] = True  # index root -> PALM owner
    contacts[0, 5] = True  # index_j0 -> owner 1
    contacts[1, 3] = True  # thumb TIP -> owner 20
    owner = contact_bits_to_owner(contacts, owner_indices, owner_count=21)
    assert owner[0, 0] and owner[0, 1] and int(owner[0].sum()) == 2
    assert owner[1, 20] and int(owner[1].sum()) == 1


def test_teacher_history_forms_exact_current_student_packet() -> None:
    r"""Q/U/A、own-contact、TIP广播与limits必须共同遵守canonical joint轴。"""

    batch, history = 2, 30
    permutation = canonical_from_native_indices(NATIVE_JOINTS, CANONICAL_JOINTS)
    q_native = torch.arange(16, dtype=torch.float32).expand(batch, history, -1)
    u_native = q_native + 100.0
    action_native = q_native + 200.0
    tip = torch.tensor([1.0, 0.0, 1.0, 0.0]).expand(batch, history, -1)
    teacher_history = torch.cat((q_native, u_native, action_native, tip), dim=-1)
    own_history = torch.zeros(batch, history, 16, dtype=torch.bool)
    own_history[:, :, 5] = True
    owner_contact = torch.zeros(batch, 21, dtype=torch.bool)
    owner_contact[:, 6] = True
    limits_native = torch.stack((q_native[:, 0] - 1.0, q_native[:, 0] + 1.0), dim=-1)

    packet = build_n000_student_actor_packet(
        teacher_history_native=teacher_history,
        own_joint_contact_history=own_history,
        owner_contact=owner_contact,
        soft_joint_limits_native_rad=limits_native,
        canonical_from_native=permutation,
    )

    expected_q = torch.tensor(permutation, dtype=torch.float32)
    assert packet.jnt_history.shape == (batch, 30, 16, 5)
    assert torch.equal(packet.jnt_history[0, -1, :, 0], expected_q)
    assert torch.equal(packet.jnt_history[0, -1, :, 1], expected_q + 100.0)
    assert torch.equal(packet.jnt_history[0, -1, :, 2], expected_q + 200.0)
    assert packet.jnt_history[0, -1, 5, 3] == 1.0
    assert torch.equal(packet.jnt_history[0, -1, :, 4], tip[0, -1].repeat(4))
    assert torch.equal(packet.jnt_current, packet.jnt_history[:, -1])
    assert torch.allclose(packet.jnt_limits[0, :, 0], (expected_q - 1.0) / math.pi)
    assert packet.owner_contact.shape == (batch, 21, 1) and packet.owner_contact[:, 6].all()
    assert packet.jnt_valid.all() and packet.tip_valid.all() and packet.owner_valid.all()
