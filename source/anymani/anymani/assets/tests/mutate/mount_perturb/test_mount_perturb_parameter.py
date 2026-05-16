"""`mount_perturb` 参数与 mode 合同回归测试。"""

from __future__ import annotations

import math

import pytest

from assets.asset_schema_core import PoseCfg
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import MountPerturbCfg, MountPerturbMutator
from assets.presets import make_human_like_builder_cfg
from assets.units import deg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_mount_perturb_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供 mutate 测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """构造和 URDF 一致的固定轴 `rpy` 旋转矩阵。"""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_multiply(lhs, rhs):
    """计算 $R = R_1 R_2$，供局部 frame 右乘测试复用。"""

    rhs_cols = (
        (rhs[0][0], rhs[1][0], rhs[2][0]),
        (rhs[0][1], rhs[1][1], rhs[2][1]),
        (rhs[0][2], rhs[1][2], rhs[2][2]),
    )
    return tuple(
        (
            row[0] * rhs_cols[0][0] + row[1] * rhs_cols[0][1] + row[2] * rhs_cols[0][2],
            row[0] * rhs_cols[1][0] + row[1] * rhs_cols[1][1] + row[2] * rhs_cols[1][2],
            row[0] * rhs_cols[2][0] + row[1] * rhs_cols[2][1] + row[2] * rhs_cols[2][2],
        )
        for row in lhs
    )


def _rotz(theta: float):
    """构造局部 $z$ 轴小旋转矩阵。"""

    c, s = math.cos(theta), math.sin(theta)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_mount_perturb_mutator_changes_only_target_finger_mount():
    """`mount_perturb` 应消费外部采样值；未给参数的 finger 保持不变。"""

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index").mount
    before_middle = _finger_by_name(hand, "middle").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            self_mode="general",
            pos_radius=0.001,
            rot_radius=0.02,
        )
    ).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "general",
                "finger_deltas": {
                    "index": {
                        "delta_pos_local": (0.001, 0.001, 0.001),
                        "delta_rotvec_local": (0.02, 0.02, 0.02),
                    }
                },
            }
        },
    )

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index").mount
    after_middle = _finger_by_name(mutated, "middle").mount
    assert after_index.pos != before_index.pos
    assert after_index.rpy != before_index.rpy
    assert after_middle.pos == before_middle.pos
    assert after_middle.rpy == before_middle.rpy


def test_mount_perturb_cfg_accepts_explicit_degree_helpers_while_runtime_consumes_radians():
    r"""角度显式 helper 应在 authoring 侧完成换算，runtime 不再持有 `disturb_unit`。"""

    cfg = MountPerturbCfg(
        self_mode="general",
        rot_radius=deg(5.0),
    )

    assert math.isclose(cfg.rot_radius, math.radians(5.0), rel_tol=0.0, abs_tol=1e-12)

    specs = MountPerturbMutator(cfg).describe_sampling(_build_allegro_hand())
    assert callable(specs["sample"])


def test_removed_disturb_unit_field_is_rejected_eagerly():
    r"""旧 `disturb_unit` 已被一次性删除，不再保留双轨配置入口。"""

    with pytest.raises(TypeError, match="disturb_unit"):
        MountPerturbCfg(disturb_unit="rad")


def test_mount_perturb_cfg_rejects_probabilities_not_summing_to_one():
    r"""dict `self_mode` 的概率和必须严格为 1。"""

    with pytest.raises(ValueError, match="sum to 1"):
        MountPerturbCfg(
            self_mode={"identity": 0.4, "index_ring_yaw_rot": 0.5},
            mirror_yaw_range=(0.0, 0.1),
            thumb_rot_radius=0.05,
        )


def test_mount_perturb_cfg_rejects_removed_sample_space_and_legacy_general_ranges():
    r"""旧 `sample_space/pos_range/rot_range` 接口应被一次性出清。"""

    with pytest.raises(TypeError, match="sample_space"):
        MountPerturbCfg(sample_space={"pos": "ellipsoid", "rot": "ellipsoid"})

    with pytest.raises(TypeError, match="pos_range"):
        MountPerturbCfg(self_mode="general", pos_range=(0.0, 0.1))

    with pytest.raises(TypeError, match="rot_range"):
        MountPerturbCfg(self_mode="general", rot_range=(0.0, 0.1))


def test_identity_mode_preserves_mount_and_records_mode_provenance():
    r"""`identity` mode 应显式保留 pre-made mount，同时记录 provenance。"""

    hand = _build_allegro_hand()
    before_mounts = {finger.name: finger.mount for finger in hand.fingers}

    mutated = MountPerturbMutator(MountPerturbCfg(self_mode="identity")).mutate(
        hand,
        sampled_params={"sample": {"resolved_self_mode": "identity"}},
    )

    assert mutated is not None
    assert {finger.name: finger.mount for finger in mutated.fingers} == before_mounts
    assert mutated.metadata["post_mutate_samples"]["mount_perturb"]["resolved_self_mode"] == "identity"


def test_index_ring_yaw_mode_mirrors_non_thumb_boundaries_and_keeps_middle_fixed():
    r"""`index_ring_yaw_rot` 应对 index/ring 写入反号 yaw，middle 保持不动。"""

    hand = _build_allegro_hand()
    for finger in hand.fingers:
        finger.mount = PoseCfg(pos=finger.mount.pos, rpy=(0.0, 0.0, 0.0))
    before_middle = _finger_by_name(hand, "middle").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            self_mode="index_ring_yaw_rot",
            mirror_yaw_range=(0.0, 0.1),
            thumb_rot_radius=0.05,
        )
    ).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "index_ring_yaw_rot",
                "mirror_yaw": 0.1,
                "thumb_delta_pos_local": (0.0, 0.0, 0.0),
                "thumb_delta_rotvec_local": (0.0, 0.0, 0.05),
            }
        },
    )

    assert mutated is not None
    assert math.isclose(_finger_by_name(mutated, "index").mount.rpy[2], -0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_finger_by_name(mutated, "ring").mount.rpy[2], 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert _finger_by_name(mutated, "middle").mount == before_middle
    assert math.isclose(_finger_by_name(mutated, "thumb").mount.rpy[2], 0.05, rel_tol=0.0, abs_tol=1e-12)


def test_index_ring_x_mode_mirrors_palm_frame_spacing_and_keeps_middle_fixed():
    r"""`index_ring_x_pos` 应围绕 middle 在 palm-frame $x$ 上镜像平移。"""

    hand = _build_allegro_hand()
    for finger in hand.fingers:
        finger.mount = PoseCfg(pos=finger.mount.pos, rpy=(0.0, 0.0, 0.0))
    before_index = _finger_by_name(hand, "index").mount
    before_ring = _finger_by_name(hand, "ring").mount
    before_middle = _finger_by_name(hand, "middle").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            self_mode="index_ring_x_pos",
            mirror_x_range=(-0.01, 0.01),
            thumb_pos_radius=0.01,
        )
    ).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "index_ring_x_pos",
                "mirror_x": 0.01,
                "thumb_delta_pos_local": (0.0, 0.0, 0.0),
                "thumb_delta_rotvec_local": (0.0, 0.0, 0.0),
            }
        },
    )

    assert mutated is not None
    assert math.isclose(_finger_by_name(mutated, "index").mount.pos[0], before_index.pos[0] + 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_finger_by_name(mutated, "ring").mount.pos[0], before_ring.pos[0] - 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert _finger_by_name(mutated, "middle").mount == before_middle


def test_index_ring_mode_combines_mirror_x_and_mirror_yaw():
    r"""`index_ring` 应同时具备镜像横向位移与镜像 yaw。"""

    hand = _build_allegro_hand()
    for finger in hand.fingers:
        finger.mount = PoseCfg(pos=finger.mount.pos, rpy=(0.0, 0.0, 0.0))
    before_index = _finger_by_name(hand, "index").mount
    before_ring = _finger_by_name(hand, "ring").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            self_mode="index_ring",
            mirror_x_range=(-0.01, 0.01),
            mirror_yaw_range=(-0.1, 0.1),
        )
    ).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "index_ring",
                "mirror_x": 0.01,
                "mirror_yaw": 0.1,
            }
        },
    )

    assert mutated is not None
    assert math.isclose(_finger_by_name(mutated, "index").mount.pos[0], before_index.pos[0] + 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_finger_by_name(mutated, "ring").mount.pos[0], before_ring.pos[0] - 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_finger_by_name(mutated, "index").mount.rpy[2], -0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_finger_by_name(mutated, "ring").mount.rpy[2], 0.1, rel_tol=0.0, abs_tol=1e-12)


def test_index_ring_mode_degrades_gracefully_when_mirror_pair_is_incomplete():
    r"""缺失 ring 时不应报错，而应只保留 thumb 独立扰动。"""

    hand = _build_allegro_hand()
    hand.fingers = [finger for finger in hand.fingers if finger.name != "ring"]
    before_index = _finger_by_name(hand, "index").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            self_mode="index_ring",
            mirror_x_range=(-0.01, 0.01),
            mirror_yaw_range=(-0.1, 0.1),
            thumb_pos_radius=0.01,
        )
    ).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "index_ring",
                "mirror_x": 0.01,
                "mirror_yaw": 0.1,
                "thumb_delta_pos_local": (0.0, 0.0, 0.01),
                "thumb_delta_rotvec_local": (0.0, 0.0, 0.0),
            }
        },
    )

    assert mutated is not None
    assert _finger_by_name(mutated, "index").mount == before_index
    assert mutated.metadata["post_mutate_samples"]["mount_perturb"]["mirror_pair_applied"] is False


def test_general_mode_position_delta_is_interpreted_in_local_mount_frame():
    r"""局部位置增量应先经当前 mount 姿态旋转，再写回 palm frame。"""

    hand = _build_allegro_hand()
    index = _finger_by_name(hand, "index")
    index.mount = PoseCfg(pos=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, math.pi / 2.0))

    mutated = MountPerturbMutator(MountPerturbCfg(self_mode="general", pos_radius=0.01)).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "general",
                "finger_deltas": {
                    "index": {
                        "delta_pos_local": (0.01, 0.0, 0.0),
                        "delta_rotvec_local": (0.0, 0.0, 0.0),
                    }
                },
            }
        },
    )

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index").mount
    assert math.isclose(after_index.pos[0], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index.pos[1], 2.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index.pos[2], 3.0, rel_tol=0.0, abs_tol=1e-12)


def test_general_mode_rotation_delta_uses_local_frame_right_multiplication():
    r"""局部 yaw 增量应满足 $R' = R \, R_z(\delta)$，而不是固定 frame 左乘。"""

    hand = _build_allegro_hand()
    index = _finger_by_name(hand, "index")
    index.mount = PoseCfg(pos=(0.0, 0.0, 0.0), rpy=(0.0, math.pi / 2.0, 0.0))

    mutated = MountPerturbMutator(MountPerturbCfg(self_mode="general", rot_radius=0.1)).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "general",
                "finger_deltas": {
                    "index": {
                        "delta_pos_local": (0.0, 0.0, 0.0),
                        "delta_rotvec_local": (0.0, 0.0, 0.1),
                    }
                },
            }
        },
    )

    assert mutated is not None
    after_matrix = _rotation_matrix_from_rpy(_finger_by_name(mutated, "index").mount.rpy)
    expected_matrix = _matrix_multiply(_rotation_matrix_from_rpy((0.0, math.pi / 2.0, 0.0)), _rotz(0.1))
    naive_left_matrix = _rotation_matrix_from_rpy((0.0, math.pi / 2.0, 0.1))

    max_naive_gap = 0.0
    for row_index in range(3):
        for col_index in range(3):
            assert math.isclose(after_matrix[row_index][col_index], expected_matrix[row_index][col_index], rel_tol=0.0, abs_tol=1e-9)
            max_naive_gap = max(max_naive_gap, abs(after_matrix[row_index][col_index] - naive_left_matrix[row_index][col_index]))
    assert max_naive_gap > 1e-3
