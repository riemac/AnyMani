r"""pre-made connectivity lowering 回归测试。

这里锁住的是当前 `JointDeleteMutator` 作为 connectivity lower 执行器的两类核心语义：

1. 显式删除指定 joint；
2. 删除后 finger 链重新接通；
3. `merge` 会把被删 joint 的几何并入上一保留容器；
4. `drop` 会按“配置项消失后的物理缩短”语义重连 surviving 链；
5. 这条工具服务 pre-made connectivity lower，而不是 post-mutate 容器。

# NOTE:
第 4 条正是这轮新增的关键保护：
当删除中间 joint / child-link 时，剩余的 distal joint / tip 不应继续保留
被删段累计长度，而应接回“第一段被删 joint 的挂接位姿”。
"""

from __future__ import annotations

import math

from assets.asset_schema_core import PoseCfg
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.hand_generator import HandGeneratorCfg
from assets.generator.premade.connectivity import apply_connectivity_preset
from assets.generator.premade.connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from assets.generator.premade.topology import build_base_hand
from assets.handedness import mirror_pose_about_yz, mirror_revolute_axis_about_yz
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_joint_delete_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的 Allegro `HandCfg`。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def _assert_pose_is_yz_mirror(left: PoseCfg, right: PoseCfg, *, tol: float = 1.0e-9) -> None:
    r"""验证 connectivity lowering 后的局部位姿仍满足严格 YZ 反射。"""

    expected = mirror_pose_about_yz(right)  # 真值为 $\mathbf p_L=S\mathbf p_R, R_L=SR_RS$
    for actual_value, expected_value in zip(left.pos, expected.pos, strict=True):
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tol)  # 平移单位 m
    for actual_value, expected_value in zip(left.rpy, expected.rpy, strict=True):
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tol)  # RPY 单位 rad


def test_joint_delete_mutator_relinks_chain_and_merges_deleted_geometry():
    """`joint_delete` 应删除目标 joint、重接链、压紧 joint 名，并把几何并到上一保留容器。"""

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    before_collision_count = len(before_index.joints[1].collisions)

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j2",),
            regroup_strategy="merge",
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_j2", "index_tip"]
    assert [joint.child for joint in after_index.joints] == ["index_mcp1", "index_mcp2", "index_dip", "index_tip"]
    assert all(current.parent == previous.child for previous, current in zip(after_index.joints[:-1], after_index.joints[1:]))
    assert len(after_index.joints[1].collisions) > before_collision_count
    assert mutated.dof_count == hand.dof_count - 1


def test_joint_delete_drop_relinks_tip_to_first_deleted_joint_origin():
    r"""`drop` 应按物理缩短语义把 tip 接回第一段被删 joint 的挂接位姿。

    当前用 Allegro index 测这个最清楚：

    - 原链：`j0 -> j1 -> j2 -> j3 -> tip`
    - 删除：`j2, j3`
    - 目标：`tip` 直接接到 `j1` 后面

    这里最关键的不是“链还连着”这么弱的条件，而是：
    `tip.origin` 必须等于原始 `j2.origin`，而不是 `j2 + j3 + tip` 的累计位姿。
    """

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    original_first_deleted_origin = before_index.joints[2].origin.copy()  # `j2.origin`：删除序列中的第一段挂接位姿
    original_tip_origin = before_index.joints[-1].origin.copy()  # 原 tip 仍挂在 `j3` 之后，只用于防止误回退

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j2", "index_j3"),
            regroup_strategy="drop",
            respect_preset=False,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_tip"]
    assert all(current.parent == previous.child for previous, current in zip(after_index.joints[:-1], after_index.joints[1:]))
    assert after_index.joints[-1].parent == after_index.joints[1].child
    assert after_index.joints[-1].origin.pos == original_first_deleted_origin.pos
    assert after_index.joints[-1].origin.rpy == original_first_deleted_origin.rpy
    assert after_index.joints[-1].origin.pos != original_tip_origin.pos


def test_joint_delete_drop_reanchors_remaining_chain_to_mount_when_root_joint_is_deleted():
    r"""删除指根 joint 时，剩余链的第一个 surviving joint 必须贴回 finger 挂载点。

    虽然正式合法 pre-made 变体一般不会删指根 joint，
    但 `JointDeleteMutator` 作为通用工具，仍必须保证：

    - 若删除第一段运动关节；
    - 下一段 surviving joint 会重新挂到 `finger.parent_link`；
    - 且其 origin 应等于“第一段被删 joint 的挂接位姿”。

    对 Allegro 而言，`j0.origin = 0`，因此删除 `j0` 后，
    新的首 joint `j1` 应直接贴回 `palm` 挂载点。
    """

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    original_root_origin = before_index.joints[0].origin.copy()  # Allegro `j0` 默认就是 finger 挂载点自身

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j0",),
            regroup_strategy="drop",
            respect_preset=False,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_j2", "index_tip"]
    assert [joint.child for joint in after_index.joints] == ["index_mcp2", "index_pip", "index_dip", "index_tip"]
    assert after_index.joints[0].parent == after_index.parent_link
    assert after_index.joints[0].origin.pos == original_root_origin.pos
    assert after_index.joints[0].origin.rpy == original_root_origin.rpy


def test_mixed_connectivity_lowering_preserves_strict_paired_handedness_contract() -> None:
    r"""跨 family 装配与 joint drop 后，左右输出仍须保持 same-$q$ 严格镜像。

    测试 topology 使用 LEAP palm/thumb，并把 index、ring 换成 Allegro，middle
    保持 LEAP；随后各 finger 使用不同的 drop recipe。该组合同时覆盖：

    - mixed non-thumb family 的不同 mount/joint-axis 锚点；
    - 删除活动 joint 后的重命名与 parent-child 重接；
    - fixed tip 回接到第一段被删 joint 的局部 origin；
    - 左右手共享同名活动链、同 limits 与同一个广义坐标 $q$。

    对所有 surviving joint，目标合同为：

    $$
    T_{L,j}=S T_{R,j} S,\qquad
    \mathbf a_{L,j}=\det(S)S\mathbf a_{R,j},\qquad
    \mathcal Q_{L,j}=\mathcal Q_{R,j}.
    $$
    """

    cfg = HandGeneratorCfg(
        mode="made",  # 只需 pre-made topology/connectivity lowering，不执行 post-mutate
        artifact_level="hand_cfg",  # 测试命题是内存运动链合同，不引入文件 I/O
        handedness="all",  # 同一 cfg 同时提供物理 left/right topology registry
        hand_presets=["single_palm_leap"],  # LEAP palm 保证 thumb family 与 palm 严格绑定
        mixed=True,  # non-thumb slot 允许跨 LEAP/Allegro family
        missing=False,  # 保留四指，避免缺指变量混入 connectivity 命题
        Validate=None,  # 合法性 validator 与 handedness 数学合同正交
    )
    slot_families = "thumb_leap__index_allegro__middle_leap__ring_allegro"  # 目标 mixed morphology
    connectivity_name = "thumb-drop_j3__index-drop_j2_j3__middle-drop_j3__ring-drop_j3"  # 四指不同降低路径
    lowered_by_side = {}

    # 左右各自从同一离散 morphology/recipe 构建；唯一差异只允许是物理 handedness lowering。
    for handedness in ("left", "right"):
        topology_name = f"single_palm_leap__{handedness}__mixed__{slot_families}"  # registry 内部稳定 key
        base_hand, _builder_name = build_base_hand(cfg, hand_preset_name=topology_name)  # 含 mixed provenance 的整手
        lowered, metadata = apply_connectivity_preset(
            cfg,
            base_hand,
            connectivity_preset_name=connectivity_name,
            hand_preset_name=topology_name,
        )  # 显式执行 joint delete、drop regroup 与稳定重命名
        assert metadata["slot_family_map"] == {
            "thumb": "leap",
            "index": "allegro",
            "middle": "leap",
            "ring": "allegro",
        }  # thumb 始终服从 palm family；只有 non-thumb 发生 family 混合
        lowered_by_side[handedness] = lowered

    left = lowered_by_side["left"]  # 物理左手，已经完成整手 YZ lowering 与 connectivity lowering
    right = lowered_by_side["right"]  # canonical 右手对应的同 topology/connectivity 输出
    assert left.dof_count == right.dof_count == 11  # $3+2+3+3=11$ 个 surviving revolute DOF
    assert [finger.name for finger in left.fingers] == [finger.name for finger in right.fingers]
    assert [joint.name for joint in left.iter_joints()] == [joint.name for joint in right.iter_joints()]

    # 逐 finger/joint 核对局部 SE(3)、伪向量 axis 与合法 $q$ 域；fixed tip 也包含在 origin 检查中。
    for left_finger, right_finger in zip(left.fingers, right.fingers, strict=True):
        _assert_pose_is_yz_mirror(left_finger.mount, right_finger.mount)  # mixed mount 仍满足 $T_L=ST_RS$
        assert len(left_finger.joints) == len(right_finger.joints)  # connectivity 不得产生侧别相关链深
        for left_joint, right_joint in zip(left_finger.joints, right_finger.joints, strict=True):
            assert left_joint.name == right_joint.name  # policy/URDF joint identity 不随 handedness 改写
            assert left_joint.parent == right_joint.parent
            assert left_joint.child == right_joint.child
            assert left_joint.joint_type == right_joint.joint_type
            _assert_pose_is_yz_mirror(left_joint.origin, right_joint.origin)  # drop/reanchor 后局部位姿仍镜像
            if right_joint.joint_type == "revolute":
                expected_axis = mirror_revolute_axis_about_yz(right_joint.axis)  # $\det(S)S\mathbf a_R$
                for actual_value, expected_value in zip(left_joint.axis, expected_axis, strict=True):
                    assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-9)
                assert left_joint.limit == right_joint.limit  # same-$q$ 不允许 limits 反号、换序或侧别漂移
