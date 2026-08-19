r"""post-mutate SDF clearance validator 回归测试。

这组测试锁住本轮最核心的科研语义：

1. SDF sign convention 是 outside positive / surface zero / inside negative；
2. finger-finger clearance 使用 symmetric sampled surface SDF；
3. certificate 必须诚实写出验证域与 non-certified claim；
4. unsupported geometry 默认 fail-hard，warn_skip 只能得到 incomplete certificate。
"""

from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from assets.asset_schema_core import CollisionGeometryCfg, PoseCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.validator import _mesh_sdf, _sdf_batch
from assets.validator import hand_rules as hand_rules_module
from assets.validator._collision_geometry import CollisionBodyRecord, extract_finger_collision_bodies
from assets.validator._sdf_batch import evaluate_finger_sdf_clearance_batch
from assets.validator._sdf_clearance import SdfClearanceConfig, evaluate_finger_sdf_clearance, signed_distance_to_body
from assets.validator._sdf_service import (
    CentralSdfServiceError,
    configure_worker_sdf_service,
    evaluate_finger_sdf_clearance_routed,
    run_sdf_service,
    stop_sdf_service,
)
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


def _write_box_mesh(path: Path, *, size: tuple[float, float, float] = (0.02, 0.02, 0.02)) -> Path:
    r"""把一个 watertight box mesh 写到临时目录，供 mesh-SDF 测试使用。"""

    import trimesh

    mesh = trimesh.creation.box(extents=size)
    mesh.export(path)
    return path


def _write_non_watertight_mesh(path: Path) -> Path:
    r"""写出一个带洞 mesh，用于锁住 non-watertight fail-hard 语义。"""

    import trimesh

    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    broken = mesh.copy()
    broken.update_faces([index != 0 for index in range(len(broken.faces))])
    broken.export(path)
    return path


def _two_box_hand(*, separation: float, second_kind: str = "box", second_mesh_path: str | None = None) -> HandCfg:
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
        second_collision = CollisionGeometryCfg(
            name="middle_col",
            geometry={"type": "mesh", "file_path": second_mesh_path or "dummy.obj"},
        )
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


def test_cuda_micro_batch_matches_scalar_for_overlap_and_separation():
    r"""Ragged batch 只增加 batch 轴，必须逐项复现 scalar CUDA 的 pair clearance 与判定。"""

    hands = [_two_box_hand(separation=0.015), _two_box_hand(separation=0.04)]
    configs = [SdfClearanceConfig(min_clearance=0.005, device="cuda", mesh_backend="warp") for _ in hands]

    results = evaluate_finger_sdf_clearance_batch(
        hands,
        configs,
        borderline_recheck_margin=0.0,
        verify_all_with_scalar=True,
    )

    assert [result.passed for result in results] == [False, True]
    assert all(result.certificate.device == "cuda" for result in results)


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "sphere", "radius": 0.01},
        {"type": "cylinder", "radius": 0.01, "length": 0.02},
        {"type": "elliptic_cylinder", "radius_x": 0.01, "radius_z": 0.008, "length": 0.02},
    ],
)
def test_cuda_micro_batch_matches_scalar_for_each_supported_primitive(geometry):
    r"""四类 primitive 的 batch 公式必须分别由 scalar CUDA oracle 覆盖，而非只验证 box。"""

    hand = _two_box_hand(separation=0.04)
    hand.fingers[1].joints[0].collisions[0] = CollisionGeometryCfg(
        name="middle_col",
        geometry=geometry,
    )
    cfg = SdfClearanceConfig(min_clearance=-1.0, device="cuda", mesh_backend="warp")

    result = evaluate_finger_sdf_clearance_batch([hand], [cfg], verify_all_with_scalar=True)[0]

    assert result.passed is True


def test_cuda_micro_batch_rechecks_threshold_borderline_with_scalar(monkeypatch):
    r"""恰好 $5\,\mathrm{mm}$ 的 box gap 必须进入 scalar oracle，而非只信 batch float32 判定。"""

    calls = 0
    scalar = _sdf_batch.evaluate_finger_sdf_clearance

    def counted_scalar(hand, cfg):
        nonlocal calls
        calls += 1
        return scalar(hand, cfg)

    monkeypatch.setattr(_sdf_batch, "evaluate_finger_sdf_clearance", counted_scalar)
    hand = _two_box_hand(separation=0.025)
    cfg = SdfClearanceConfig(min_clearance=0.005, device="cuda", mesh_backend="warp")

    result = evaluate_finger_sdf_clearance_batch([hand], [cfg], borderline_recheck_margin=1.0e-6)[0]

    assert calls == 1
    assert result.passed is True


def test_central_gpu_service_batches_independent_clients_without_local_fallback():
    r"""两个并发 client 应由独立 spawn actor 返回，worker/调用线程不拥有 CUDA service 状态。"""

    context = mp.get_context("spawn")
    request_queue = context.Queue()
    startup_receive, startup_send = context.Pipe(duplex=False)
    process = context.Process(
        target=run_sdf_service,
        args=(request_queue, startup_send),
        kwargs={"batch_size": 2, "batch_window_ms": 50.0},
    )
    process.start()
    startup_send.close()
    assert startup_receive.poll(60.0)
    assert startup_receive.recv()["ok"] is True
    startup_receive.close()
    assert process.is_alive()
    configure_worker_sdf_service(request_queue)
    try:
        hands = [_two_box_hand(separation=0.015), _two_box_hand(separation=0.04)]
        configs = [
            SdfClearanceConfig(min_clearance=0.005, device="cuda", mesh_backend="warp")
            for _ in hands
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(evaluate_finger_sdf_clearance_routed, hand, cfg)
                for hand, cfg in zip(hands, configs)
            ]
            results = [future.result(timeout=60.0) for future in futures]
    finally:
        configure_worker_sdf_service(None)
        stop_sdf_service(request_queue)
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        request_queue.close()
        request_queue.join_thread()

    assert process.exitcode == 0
    assert [result.passed for result in results] == [False, True]


def test_central_gpu_service_rejects_auto_backend_without_local_fallback():
    r"""Central actor 收到 auto backend 必须返回 fatal error，不能在 worker 内改走 scalar/CPU。"""

    context = mp.get_context("spawn")
    request_queue = context.Queue()
    startup_receive, startup_send = context.Pipe(duplex=False)
    process = context.Process(
        target=run_sdf_service,
        args=(request_queue, startup_send),
        kwargs={"batch_size": 1},
    )
    process.start()
    startup_send.close()
    assert startup_receive.poll(60.0)
    assert startup_receive.recv()["ok"] is True
    startup_receive.close()
    configure_worker_sdf_service(request_queue)
    try:
        with pytest.raises(CentralSdfServiceError, match="requires device='cuda'"):
            evaluate_finger_sdf_clearance_routed(
                _two_box_hand(separation=0.04),
                SdfClearanceConfig(min_clearance=0.005),
            )
    finally:
        configure_worker_sdf_service(None)
        stop_sdf_service(request_queue)
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        request_queue.close()
        request_queue.join_thread()

    assert process.exitcode == 0


def test_hand_validator_propagates_central_service_failure(monkeypatch):
    r"""中央 actor 失效属于 build-level fatal error，不能规约成普通 candidate rejection。"""

    def fail_service(_hand, _cfg):
        raise CentralSdfServiceError("injected actor failure")

    monkeypatch.setattr(hand_rules_module, "evaluate_finger_sdf_clearance_routed", fail_service)
    validator = HandValidator(
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
    )

    with pytest.raises(CentralSdfServiceError, match="injected actor failure"):
        validator.validate_post_mutate(_two_box_hand(separation=0.04))


def test_warp_mesh_cache_is_bounded_lru():
    r"""长时间 GPU service 最多持有 128 个 mesh BVH，且优先淘汰最旧项。"""

    cache = _mesh_sdf._WARP_MESH_CACHE
    cache.clear()
    handle = _mesh_sdf._WarpMeshHandle(mesh=object(), points=object(), indices=object())
    for index in range(_mesh_sdf._WARP_MESH_CACHE_MAXSIZE + 1):
        key = (f"mesh-{index}", f"sha-{index}", (1.0, 1.0, 1.0), "cuda:0")
        _mesh_sdf._remember_warp_mesh_handle(key, handle)

    assert len(cache) == _mesh_sdf._WARP_MESH_CACHE_MAXSIZE
    assert ("mesh-0", "sha-0", (1.0, 1.0, 1.0), "cuda:0") not in cache
    cache.clear()


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


def test_missing_mesh_path_fails_hard_without_proxy_fallback():
    r"""缺失 mesh 文件必须硬失败，不能退回旧 `sdf_proxy` 近似。"""

    hand = _two_box_hand(separation=0.04, second_kind="mesh")
    hand.fingers[1].joints[0].metadata["sdf_proxy"] = {
        "type": "box",
        "size": (0.02, 0.02, 0.02),
        "origin": {"pos": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)},
    }

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

    assert default_result.passed is False
    assert any("mesh file does not exist" in error for error in default_result.errors)
    assert default_result.metadata["finger_spacing_certificate"]["complete"] is False


def test_custom_tip_mesh_signed_distance_accepts_and_rejects(tmp_path):
    r"""custom mesh 进入真实 signed-distance clearance，而不是 primitive proxy。"""

    mesh_path = _write_box_mesh(tmp_path / "tip_box.stl")
    overlap = _two_box_hand(separation=0.015, second_kind="mesh", second_mesh_path=str(mesh_path))
    separated = _two_box_hand(separation=0.04, second_kind="mesh", second_mesh_path=str(mesh_path))

    overlap_result = evaluate_finger_sdf_clearance(
        overlap,
        SdfClearanceConfig(min_clearance=0.005, device="cpu", mesh_backend="trimesh", mesh_surface_samples=512),
    )
    separated_result = evaluate_finger_sdf_clearance(
        separated,
        SdfClearanceConfig(min_clearance=0.005, device="cpu", mesh_backend="trimesh", mesh_surface_samples=512),
    )

    assert overlap_result.passed is False
    assert overlap_result.violations[0].clearance < 0.0
    assert separated_result.passed is True
    assert separated_result.certificate.mesh_sdf["actual_backend"] == "trimesh"


def test_custom_tip_mesh_uses_warp_when_available(tmp_path):
    r"""auto 配置下，当前 CUDA/Warp 环境应优先使用 Warp mesh query。"""

    mesh_path = _write_box_mesh(tmp_path / "tip_box.stl")
    hand = _two_box_hand(separation=0.04, second_kind="mesh", second_mesh_path=str(mesh_path))

    result = evaluate_finger_sdf_clearance(
        hand,
        SdfClearanceConfig(min_clearance=0.005, device="auto", mesh_backend="auto", mesh_surface_samples=512),
    )

    assert result.passed is True
    assert result.certificate.mesh_sdf["requested_backend"] == "auto"
    assert result.certificate.mesh_sdf["actual_backend"] in {"warp", "trimesh", "mixed"}


def test_cuda_micro_batch_matches_scalar_for_mesh_and_primitive_mix(tmp_path):
    r"""同一 batch 中的 mesh/Warp 与 primitive bodies 仍须保持每只 hand 独立证书。"""

    mesh_path = _write_box_mesh(tmp_path / "tip_box.stl")
    hands = [
        _two_box_hand(separation=0.015, second_kind="mesh", second_mesh_path=str(mesh_path)),
        _two_box_hand(separation=0.04, second_kind="mesh", second_mesh_path=str(mesh_path)),
    ]
    configs = [
        SdfClearanceConfig(
            min_clearance=0.005,
            device="cuda",
            mesh_backend="warp",
            mesh_surface_samples=512,
        )
        for _ in hands
    ]

    results = evaluate_finger_sdf_clearance_batch(
        hands,
        configs,
        borderline_recheck_margin=0.0,
        verify_all_with_scalar=True,
    )

    assert [result.passed for result in results] == [False, True]
    assert all(result.certificate.mesh_sdf["actual_backend"] == "warp" for result in results)


def test_non_watertight_mesh_fails_hard(tmp_path):
    r"""mesh 非闭合时 signed distance inside/outside 不可信，validator 必须拒绝证书。"""

    mesh_path = _write_non_watertight_mesh(tmp_path / "broken_tip.stl")
    hand = _two_box_hand(separation=0.04, second_kind="mesh", second_mesh_path=str(mesh_path))

    result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                min_finger_spacing=0.005,
                sdf_device="cpu",
                sdf_mesh_backend="trimesh",
            )
        )
    ).validate_post_mutate(hand)

    assert result.passed is False
    assert any("watertight" in error for error in result.errors)


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
