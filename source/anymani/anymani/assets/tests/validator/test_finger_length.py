r"""post-mutate finger axial length validator 回归测试。

这组测试锁住本轮已经明确对齐的科研语义：

1. 长度不是 `joint.origin` 范数求和，而是 collision geometry union
   沿 nominal distal axis 的投影宽度；
2. non-thumb 轴采用规范 distal 方向；
3. thumb 的 axis 忽略 CMC1 定轴，但 CMC1 geometry 仍参与长度包络；
4. custom mesh tip 直接用 mesh vertices 投影，不走 sampled surface 或 SDF；
5. HandValidator 的 post-mutate 路径应把超阈值 finger 作为 hard reject。
"""

from __future__ import annotations

import math
from pathlib import Path

from assets.asset_schema_core import CollisionGeometryCfg, PoseCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.validator._finger_length import FingerLengthConfig, evaluate_finger_axial_length
from assets.validator.hand_rules import HandValidator, HandValidatorCfg


def _write_box_mesh(path: Path, *, size: tuple[float, float, float]) -> Path:
    r"""写出一个 watertight box mesh，供 custom tip length 测试使用。"""

    import trimesh

    mesh = trimesh.creation.box(extents=size)  # `trimesh.creation.box` 默认给出中心在原点的闭合三角网格
    mesh.export(path)
    return path


def _single_non_thumb_hand() -> HandCfg:
    r"""构造一根沿 local/world $+y$ 伸展的简化 non-thumb finger。

    长度构造刻意保持解析可算：

    - 第一段 box 长 $4$cm，中心在 $y=0$，投影区间 $[-2, 2]$cm；
    - 第二段 box 长 $4$cm，joint origin 上移 $4$cm，投影区间 $[2, 6]$cm；
    - 指尖 sphere 半径 $1$cm，joint origin 再上移 $4$cm，区间 $[7, 9]$cm。

    因此全 finger 的轴向真实长度应为：
    $$
    L = 9 - (-2) = 11 \text{ cm}.
    $$
    """

    return HandCfg(
        name="single_non_thumb_hand",
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
                        child="index_link_0",
                        origin=PoseCfg(),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_link_0_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="index_j1",
                        parent="index_link_0",
                        child="index_link_1",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_link_1_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="index_tip_fixed",
                        parent="index_link_1",
                        child="index_tip",
                        joint_type="fixed",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_tip_col",
                                geometry={"type": "sphere", "radius": 0.01},
                            )
                        ],
                        is_tip=True,
                    ),
                ],
            )
        ],
        family="unit_test",
        handedness="right",
    )


def _single_thumb_hand() -> HandCfg:
    r"""构造一根简化 thumb，用于锁住“忽略 CMC1 定轴，但保留 CMC1 包络”的语义。

    这里故意把 CMC1 的 collision center 放到 $y=-3$cm：

    - 若 axis 错误地由 CMC1 主导，或直接把 CMC1 排除出点集，长度都会变短；
    - 正确语义应当是：axis 由 `CMC2 -> DIP` 定义，但投影区间仍包含 CMC1。

    解析长度：

    - CMC1 小 box：区间 $[-4, -2]$cm；
    - CMC2 box：区间 $[-2, 2]$cm；
    - MCP  box：区间 $[2, 6]$cm；
    - DIP  box：区间 $[6, 10]$cm；
    - TIP sphere：区间 $[11, 13]$cm。

    因而：
    $$
    L = 13 - (-4) = 17 \text{ cm}.
    $$
    """

    return HandCfg(
        name="single_thumb_hand",
        palm=PalmCfg(name="palm"),
        fingers=[
            FingerCfg(
                name="thumb",
                parent_link="palm",
                mount=PoseCfg(),
                joints=[
                    JointCfg(
                        name="thumb_j0",
                        parent="palm",
                        child="thumb_cmc1",
                        origin=PoseCfg(),
                        collisions=[
                            CollisionGeometryCfg(
                                name="thumb_cmc1_col",
                                geometry={"type": "box", "size": (0.02, 0.02, 0.02)},
                                origin=PoseCfg(pos=(0.0, -0.03, 0.0)),
                            )
                        ],
                    ),
                    JointCfg(
                        name="thumb_j1",
                        parent="thumb_cmc1",
                        child="thumb_cmc2",
                        origin=PoseCfg(pos=(0.04, 0.0, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="thumb_cmc2_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="thumb_j2",
                        parent="thumb_cmc2",
                        child="thumb_mcp",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="thumb_mcp_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="thumb_j3",
                        parent="thumb_mcp",
                        child="thumb_dip",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="thumb_dip_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="thumb_tip_fixed",
                        parent="thumb_dip",
                        child="thumb_tip",
                        joint_type="fixed",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="thumb_tip_col",
                                geometry={"type": "sphere", "radius": 0.01},
                            )
                        ],
                        is_tip=True,
                    ),
                ],
            )
        ],
        family="unit_test",
        handedness="right",
    )


def test_non_thumb_axial_length_uses_geometry_projection_extent():
    r"""non-thumb 长度应等于 collision geometry union 在 distal axis 上的投影宽度。"""

    result = evaluate_finger_axial_length(_single_non_thumb_hand(), FingerLengthConfig())
    measurement = result.measurements[0]

    assert result.passed is True
    assert measurement.finger_name == "index"
    assert measurement.axis_source == "root_link_local_+y"
    assert math.isclose(measurement.axial_length, 0.11, abs_tol=1e-9)


def test_thumb_axis_ignores_cmc1_but_length_envelope_still_includes_cmc1_geometry():
    r"""thumb 的 axis 定义应跳过 CMC1，但长度包络仍要把 CMC1 collision 计入。"""

    result = evaluate_finger_axial_length(_single_thumb_hand(), FingerLengthConfig())
    measurement = result.measurements[0]

    assert result.passed is True
    assert measurement.finger_name == "thumb"
    assert measurement.axis_source == "thumb_cmc2_to_thumb_dip"
    assert math.isclose(measurement.axial_length, 0.17, abs_tol=1e-9)


def test_custom_tip_mesh_uses_vertices_projection_instead_of_joint_origin_proxy(tmp_path):
    r"""custom mesh tip 的长度应直接来自 mesh vertices 投影，而不是旧式 origin 近似。"""

    mesh_path = _write_box_mesh(tmp_path / "tip_box.stl", size=(0.02, 0.06, 0.02))
    hand = HandCfg(
        name="mesh_tip_hand",
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
                        child="index_link_0",
                        origin=PoseCfg(),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_link_0_col",
                                geometry={"type": "box", "size": (0.02, 0.04, 0.02)},
                            )
                        ],
                    ),
                    JointCfg(
                        name="index_tip_fixed",
                        parent="index_link_0",
                        child="index_tip",
                        joint_type="fixed",
                        origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
                        collisions=[
                            CollisionGeometryCfg(
                                name="index_tip_mesh_col",
                                geometry={"type": "mesh", "file_path": str(mesh_path)},
                                origin=PoseCfg(pos=(0.0, 0.03, 0.0)),
                            )
                        ],
                        is_tip=True,
                    ),
                ],
            )
        ],
        family="unit_test",
        handedness="right",
    )

    result = evaluate_finger_axial_length(hand, FingerLengthConfig())
    measurement = result.measurements[0]

    # 第一段 box: [-2, 2]cm；mesh tip box 高 6cm、中心在 7cm，因此 tip 区间 [4, 10]cm，总长 12cm。
    assert math.isclose(measurement.axial_length, 0.12, abs_tol=1e-6)


def test_hand_validator_rejects_post_mutate_finger_that_exceeds_role_threshold():
    r"""post-mutate hand validator 应把超阈值 finger axial length 当作 hard reject。"""

    hand = _single_non_thumb_hand()
    result = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_min=None,
                finger_count_min=None,
                finger_count_max=None,
                require_thumb=False,
                require_non_thumb_with_min_revolute_dof=None,
                check_finger_spacing=False,
                check_finger_length=True,
                max_non_thumb_length=0.10,
            )
        )
    ).validate_post_mutate(hand)

    certificate = result.metadata["finger_length_certificate"]

    assert result.passed is False
    assert certificate["length_kind"] == "axial_projection_extent"
    assert any("axial_length" in error for error in result.errors)
