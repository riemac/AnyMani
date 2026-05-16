r"""post-mutate SDF clearance validator 回归测试。

这组测试锁住本轮最核心的科研语义：

1. SDF sign convention 是 outside positive / surface zero / inside negative；
2. finger-finger clearance 使用 symmetric sampled surface SDF；
3. certificate 必须诚实写出验证域与 non-certified claim；
4. unsupported geometry 默认 fail-hard，warn_skip 只能得到 incomplete certificate。
"""

from __future__ import annotations

import math

from assets.asset_schema_core import CollisionGeometryCfg, PoseCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.validator._collision_geometry import CollisionBodyRecord, extract_finger_collision_bodies
from assets.validator._sdf_clearance import SdfClearanceConfig, evaluate_finger_sdf_clearance, signed_distance_to_body
from assets.validator.hand_rules import HandValidator, HandValidatorCfg


def _body(kind: str = "box") -> CollisionBodyRecord:
    r"""构造一个位于原点的测试 body record。"""

    collision = CollisionGeometryCfg(name="box_col", geometry={"type": kind, "size": (1.0, 1.0, 1.0)})
    return CollisionBodyRecord(
        finger_name="index",
        joint_name="index_j0",
        link_name="index_link",
        body_name="box_col",
        body_path="index/index_j0/index_link/box_col",
        geometry_kind=collision.geometry.kind,
        geometry=collision.geometry,
        world_pose=PoseCfg(),
    )


def _two_box_hand(*, separation: float, second_kind: str = "box") -> HandCfg:
    r"""构造两根单 link finger，collision box 中心间距由 separation 控制。"""

    first = FingerCfg(
        name="index",
        parent_link="palm",
        mount=PoseCfg(),
        joints=[
            JointCfg(
                name="index_j0",
                parent="palm",
                child="index_link",
                origin=PoseCfg(),
                collisions=[
                    CollisionGeometryCfg(name="index_col", geometry={"type": "box", "size": (0.02, 0.02, 0.02)})
                ],
            )
        ],
    )
    if second_kind == "mesh":
        second_collision = CollisionGeometryCfg(name="middle_col", geometry={"type": "mesh", "file_path": "dummy.obj"})
    else:
        second_collision = CollisionGeometryCfg(name="middle_col", geometry={"type": second_kind, "size": (0.02, 0.02, 0.02)})
    second = FingerCfg(
        name="middle",
        parent_link="palm",
        mount=PoseCfg(pos=(separation, 0.0, 0.0)),
        joints=[
            JointCfg(
                name="middle_j0",
                parent="palm",
                child="middle_link",
                origin=PoseCfg(),
                collisions=[second_collision],
            )
        ],
    )
    return HandCfg(
        name="two_box_hand",
        palm=PalmCfg(name="palm"),
        fingers=[first, second],
        family="unit_test",
        handedness="right",
    )


def test_box_sdf_sign_convention_outside_surface_inside():
    r"""box SDF 的符号约定必须与 certificate 文档一致。"""

    body = _body()

    assert signed_distance_to_body((0.75, 0.0, 0.0), body) > 0.0
    assert math.isclose(signed_distance_to_body((0.5, 0.0, 0.0), body), 0.0, abs_tol=1e-9)
    assert signed_distance_to_body((0.0, 0.0, 0.0), body) < 0.0


def test_sdf_clearance_rejects_overlap_and_accepts_separated_boxes():
    r"""两个 finger box 穿膜时拒绝，间隔超过 margin 时通过。"""

    overlap = _two_box_hand(separation=0.015)
    separated = _two_box_hand(separation=0.04)

    overlap_result = evaluate_finger_sdf_clearance(overlap, SdfClearanceConfig(min_clearance=0.005))
    separated_result = evaluate_finger_sdf_clearance(separated, SdfClearanceConfig(min_clearance=0.005))

    assert overlap_result.passed is False
    assert overlap_result.violations[0].clearance < 0.005
    assert separated_result.passed is True


def test_hand_validator_attaches_complete_sdf_certificate_for_post_mutate():
    r"""post-mutate validator 应把 SDF certificate 放进 ValidationResult.metadata。"""

    hand = _two_box_hand(separation=0.04)
    result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                min_finger_spacing=0.005,
            )
        )
    ).validate_post_mutate(hand)

    certificate = result.metadata["finger_spacing_certificate"]

    assert result.passed is True
    assert certificate["pose_scope"] == "post_mutate_home_pose"
    assert certificate["geometry_scope"] == "collision_geometry_only"
    assert certificate["sdf_kind"] == "sampled_surface_sdf_approx"
    assert certificate["complete"] is True
    assert certificate["skipped_bodies"] == []
    assert certificate["device"] in {"cpu", "cuda"}
    assert "mesh_exact_clearance" in certificate["not_certified"]


def test_unsupported_mesh_default_fail_hard_and_warn_skip_is_incomplete():
    r"""mesh collision 默认硬失败；warn_skip 只允许产生 incomplete certificate。"""

    hand = _two_box_hand(separation=0.04, second_kind="mesh")

    default_result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                min_finger_spacing=0.005,
            )
        )
    ).validate_post_mutate(hand)

    warn_result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                min_finger_spacing=0.005,
                sdf_unsupported_geometry_policy="warn_skip",
            )
        )
    ).validate_post_mutate(hand)

    assert default_result.passed is False
    assert any("unsupported collision geometry" in error for error in default_result.errors)
    assert warn_result.passed is False
    assert warn_result.metadata["finger_spacing_certificate"]["complete"] is False
    assert warn_result.metadata["finger_spacing_certificate"]["skipped_bodies"]


def test_custom_tip_mesh_with_explicit_sdf_proxy_is_complete():
    r"""带显式 `sdf_proxy` 的 custom tip mesh 可用外包盒进入完整 SDF certificate。"""

    hand = _two_box_hand(separation=0.04, second_kind="mesh")
    middle_tip = hand.fingers[1].joints[0]
    middle_tip.metadata["sdf_proxy"] = {
        "type": "box",
        "size": (0.02, 0.02, 0.02),
        "origin": {"pos": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)},
        "source": "unit_test_box_proxy",
    }

    result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                min_finger_spacing=0.005,
            )
        )
    ).validate_post_mutate(hand)

    certificate = result.metadata["finger_spacing_certificate"]

    assert result.passed is True
    assert certificate["complete"] is True
    assert certificate["skipped_bodies"] == []
    extraction = extract_finger_collision_bodies(hand)
    assert extraction.bodies_by_finger["middle"][0].body_path.endswith("#sdf_proxy")


def test_extract_collision_observes_mutated_mount_transform():
    r"""collision 抽取必须使用 hand 当前 mount，而不是 pre-mutate 旧状态。"""

    hand = _two_box_hand(separation=0.123)
    extraction = extract_finger_collision_bodies(hand)
    middle_body = extraction.bodies_by_finger["middle"][0]

    assert math.isclose(middle_body.world_pose.pos[0], 0.123, abs_tol=1e-9)


def test_extract_collision_composes_rotated_collision_origin_in_world_frame():
    r"""collision.origin 不能再用分量相加；必须受 link 姿态旋转影响。

    这个测试专门锁住本轮真实 bug：

    - 若 link 自身绕 $z$ 旋转 $90^\circ$；
    - collision 在 link 局部 $+x$ 上偏移 $1$cm；
    - 那么 world 下它应落到 $+y$，而不是继续留在 $+x$。
    """

    hand = HandCfg(
        name="rotated_collision_hand",
        palm=PalmCfg(name="palm"),
        fingers=[
            FingerCfg(
                name="index",
                parent_link="palm",
                mount=PoseCfg(),
                joints=[
                    JointCfg(
                        name="index_j0",
                        parent="palm",
                        child="index_link",
                        origin=PoseCfg(rpy=(0.0, 0.0, math.pi / 2.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_col",
                                geometry={"type": "box", "size": (0.01, 0.01, 0.01)},
                                origin=PoseCfg(pos=(0.01, 0.0, 0.0)),
                            )
                        ],
                    )
                ],
            )
        ],
        family="unit_test",
        handedness="right",
    )

    extraction = extract_finger_collision_bodies(hand)
    body = extraction.bodies_by_finger["index"][0]

    assert math.isclose(body.world_pose.pos[0], 0.0, abs_tol=1e-6)
    assert math.isclose(body.world_pose.pos[1], 0.01, abs_tol=1e-6)
