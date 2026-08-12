"""`HandMutator` 容器级回归测试。"""

from __future__ import annotations

import math

from assets.asset_schema_core import PoseCfg
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import (
    HandMutator,
    HandMutatorCfg,
    LimitTweakCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from assets.handedness import mirror_pose_about_yz, mirror_revolute_axis_about_yz
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_mutate_pipeline_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供 mutate 测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _build_allegro_hand_with_handedness(handedness: str):
    r"""从同一 canonical recipe 构建目标 handedness，供 paired mutate contract 使用。"""

    cfg = _make_allegro_builder_cfg().replace(
        name=f"allegro_mutate_pipeline_{handedness}",
        handedness=handedness,
    )  # 除 handedness 外不改变任何 morphology 锚点
    return HumanLikeHandBuilder(cfg).build()


def _joint_by_name(hand, joint_name: str):
    """按名字取 joint。"""

    for joint in hand.iter_joints():
        if joint.name == joint_name:
            return joint
    raise KeyError(joint_name)


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def _assert_pose_is_mirror(left: PoseCfg, right: PoseCfg, *, tol: float = 1e-9) -> None:
    r"""比较 paired mutate 输出是否仍满足 YZ 平面严格镜像。"""

    expected = mirror_pose_about_yz(right)  # $\mathbf p_L=S\mathbf p_R, R_L=SR_RS$
    for actual_value, expected_value in zip(left.pos, expected.pos, strict=True):
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tol)
    for actual_value, expected_value in zip(left.rpy, expected.rpy, strict=True):
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tol)


class DemoParameterMutatorCfg(HandMutatorCfg):
    """用类属性声明 term，锁住新的 IsaacLab 风格 container 用法。"""

    limit = LimitTweakCfg(
        disturb_object="shared",
        disturb_type="add",
        joint_range=(0.05, 0.05),
        clip={"abs": 0.1},
    )
    mount = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
        rot_radius=0.02,
    )


class DemoLinkScaleThenTipReplaceMutatorCfg(HandMutatorCfg):
    r"""锁住 `link_scale -> tip_replace` 的组合 patch 语义。

    这份测试 cfg 故意保持和默认 post-mutate 配置相同的声明顺序：

    1. `link_scale` 先把末节 link 的有效长度 $L_i$ 改为 $L_i'$；
    2. `link_scale` 同时把下游 fixed `tip_joint.origin` 推进到新的远端边界；
    3. `tip_replace` 再替换末端 child link 的接触皮肤 / mesh embodiment。

    科研语义上，步骤 2 是运动链边界条件，步骤 3 是接触几何假设；两者不能
    相互覆盖。若 `tip_replace` 整体替换 `JointCfg`，就会把步骤 2 已经求出的
    $y_{tip}' = y_{tip} + (L_i' - L_i)$ 回滚到原始 $y_{tip}$。
    """

    link_scale = LinkScaleCfg(
        self_mode="only_length",  # 只扰动主长度方向，避免 shared width/height 干扰本测试的 tip 边界断言
        scale_type="rel",  # `1.2` 解释为 $L_i'=1.2L_i$，而不是绝对增量
        link_scale=(1.2, 1.2),  # 固定采样锚点，保证回归测试确定
        distrib="uniform",  # 分布字段仍显式给出，保持和真实 mutator cfg 形状一致
    )
    tip_replace = TipReplaceCfg(
        target_fingers=("index",),  # 只替换 index tip，便于把断言集中到一条运动链
        self_mode="same",  # 单指目标下 `same` 与 `general` 等价，但更贴近默认配置
        tip_range=["round"],  # custom mesh tip 会暴露 origin 覆盖 bug，不再是 primitive 两段几何
        scale=(1.0, 1.0),  # 不缩放 tip mesh，避免把 anchor/scale 误差混进本测试
    )


class DemoMountAndLimitMutatorCfg(HandMutatorCfg):
    r"""锁住 mount 局部增量与 joint-limit 再标定的 paired-handedness 语义。

    `mount_perturb` 的局部增量在 canonical right mount frame 中解释：

    $$
    T'_{PM}=T_{PM}\Delta_M.
    $$

    `limit_tweak` 只改变同名关节的合法广义坐标区间
    $[q_{\min},q_{\max}]$。它不改变轴、位姿或左右手的 $q$ 符号，因此同一
    sample 作用于一对物理手后，mount 仍严格镜像，limits 则逐值相同。
    """

    mount_perturb = MountPerturbCfg(
        self_mode="general",  # 强制局部椭球增量模式，测试不依赖随机 mode 路由
        pos_radius=(0.003, 0.002, 0.001),  # mount-frame 平移半轴，单位 m
        rot_radius=(0.04, 0.03, 0.02),  # mount-frame 旋转向量半轴，单位 rad
    )
    limit_tweak = LimitTweakCfg(
        self_mode="disturb",  # 每个活动 joint 消费显式给定的独立上下界增量
        disturb_object="independent",  # $q_{\min}$ 与 $q_{\max}$ 可取不同增量
        disturb_type="add",  # $q'=q+\delta q$，不引入与原 limit 符号相关的比例语义
        joint_range=(-0.1, 0.1),  # 手工 sample 的合法声明域，单位 rad
    )


def test_hand_mutator_pipeline_accepts_declared_terms_and_step_validation():
    """`HandMutatorCfg` 应按声明顺序解析 term，并接受上游采样值。"""

    hand = _build_allegro_hand()
    cfg = DemoParameterMutatorCfg()

    mutated = HandMutator(cfg).mutate(
        hand,
        sampled_params={
            "limit": {"index_j0": 0.05},
            "mount": {
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
        },
    )

    assert mutated is not None
    assert [name for name, _ in cfg.ordered_terms()] == ["limit", "mount"]
    assert _joint_by_name(mutated, "index_j0").limit.lower != _joint_by_name(hand, "index_j0").limit.lower
    assert _finger_by_name(mutated, "index").mount.pos != _finger_by_name(hand, "index").mount.pos


def test_hand_mutator_pipeline_preserves_link_scaled_tip_origin_when_replacing_tip():
    r"""`tip_replace` 不应覆盖 `link_scale` 已更新的末端 joint origin。

    回归目标来自真实 post-mutate bug：所有 term 都基于同一个原始 `HandCfg`
    规划 patch，然后在 `HandPatch.apply()` 里顺序写入。`link_scale` 已经把
    index 末节后的 fixed tip joint 推到新的远端边界：

    $$
    y_{tip}' = y_{tip} + (L_i' - L_i)
             = y_{tip} + (1.2L_i - L_i).
    $$

    `tip_replace` 的物理职责是替换末端接触几何，不是重设运动链边界；因此
    最终 tip joint 的 origin 必须保留上式求出的 apply-time 结果。
    """

    hand = _build_allegro_hand()
    before_tip = _finger_by_name(hand, "index").tip_joint
    before_distal = _joint_by_name(hand, "index_j3")
    before_tip_pos = before_tip.origin.pos  # 原始 fixed tip joint frame，相对 index_dip 的局部位姿
    before_distal_length = before_distal.collisions[0].geometry.size[1]  # index 末节主体长度 $L_i$，单位 m

    mutated = HandMutator(DemoLinkScaleThenTipReplaceMutatorCfg()).mutate(
        hand,
        sampled_params={
            "link_scale": {
                "sample": {
                    "resolved_self_mode": "only_length",  # 强制走只改长度的 mode，排除横截面重解算
                    "joint_length_scale": {"index_j3": 1.2},  # $L_i'=1.2L_i$，只作用于 index 末节
                    "length_scale": {},  # 由 mutator 写回的 provenance 字段，输入侧保持空
                    "width_scale": None,  # only_length mode 不应消费 shared width
                    "height_scale": None,  # only_length mode 不应消费 shared height
                }
            },
            "tip_replace": {
                "sample": {
                    "resolved_self_mode": "same",  # 单指目标下共享一份 tip spec
                    "finger_specs": {"index": {"tip_type": "round", "scale": 1.0}},  # 确定性 custom mesh tip
                }
            },
        },
    )

    assert mutated is not None
    after_tip = _finger_by_name(mutated, "index").tip_joint
    expected_tip_y = before_tip_pos[1] + before_distal_length * (1.2 - 1.0)  # $y_{tip}'=y_{tip}+\Delta L_i$
    assert after_tip.collisions[0].geometry.kind == "mesh"
    assert after_tip.metadata["post_mutate_tip_type"] == "round"
    assert not math.isclose(after_tip.origin.pos[1], before_tip_pos[1], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.origin.pos[0], before_tip_pos[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.origin.pos[1], expected_tip_y, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.origin.pos[2], before_tip_pos[2], rel_tol=0.0, abs_tol=1e-12)


def test_same_post_mutate_sample_preserves_strict_left_right_mirror_contract():
    r"""相同随机样本作用于左右手后，输出仍应是一对同 $q$ 的严格镜像。

    该命题要求 post-mutate 的公式在 canonical right 空间解释。尤其 Allegro thumb
    的 CMC1 宽高缩放会重解算 CMC2 的 local $x/y/z$；若直接对物理 left 使用
    right-hand 侧边界公式，CMC2 的 $x$ 会被重新写回右手符号，重现历史偏移错误。
    """

    right = _build_allegro_hand_with_handedness("right")  # canonical morphology 真源
    left = _build_allegro_hand_with_handedness("left")  # 同一真源的物理镜像
    mutator = HandMutator(DemoLinkScaleThenTipReplaceMutatorCfg())
    sampled_params = {
        "link_scale": {
            "sample": {
                "resolved_self_mode": "only_length",
                "joint_length_scale": {
                    "thumb_j0": 1.2,
                    "thumb_j1": 1.1,
                    "index_j3": 1.2,
                },  # CMC1 与后续串联段同时变化，覆盖 thumb 专用/普通推进两条公式
                "length_scale": {},
                "width_scale": None,
                "height_scale": None,
            }
        },
        "tip_replace": {
            "sample": {
                "resolved_self_mode": "same",
                "finger_specs": {"index": {"tip_type": "round", "scale": 1.0}},
            }
        },
    }

    mutated_right = mutator.mutate(right, sampled_params=sampled_params)  # 在 canonical right 中解释样本
    mutated_left = mutator.mutate(left, sampled_params=sampled_params)  # 应内部 canonicalize 后再恢复 left

    assert mutated_right is not None
    assert mutated_left is not None
    assert [joint.name for joint in mutated_left.iter_joints()] == [joint.name for joint in mutated_right.iter_joints()]
    for left_finger, right_finger in zip(mutated_left.fingers, mutated_right.fingers, strict=True):
        _assert_pose_is_mirror(left_finger.mount, right_finger.mount)  # mount 保持整手反射关系
        for left_joint, right_joint in zip(left_finger.joints, right_finger.joints, strict=True):
            _assert_pose_is_mirror(left_joint.origin, right_joint.origin)  # CMC2 与 tip 下游位姿都不能漂移
            if right_joint.joint_type == "revolute":
                expected_axis = mirror_revolute_axis_about_yz(right_joint.axis)  # same-$q$ 伪向量合同
                for actual_value, expected_value in zip(left_joint.axis, expected_axis, strict=True):
                    assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-9)
                assert left_joint.limit == right_joint.limit  # mutate 不得为 handedness 改写合法 $q$ 域


def test_same_mount_and_limit_sample_preserves_paired_handedness_contract() -> None:
    r"""局部 mount 扰动与 limit 微调不得破坏 same-$q$ 左右手合同。

    相同结构化 sample 先在 canonical right 空间解释，再各自 lower 到目标物理侧。
    对任意被扰动的 finger mount，应继续满足：

    $$
    \mathbf p_L'=S\mathbf p_R',\qquad R_L'=SR_R'S.
    $$

    对任意活动关节，左右手共享同一广义坐标，因此必须满足：

    $$
    [q_{\min,L}',q_{\max,L}']=[q_{\min,R}',q_{\max,R}'].
    $$
    """

    right = _build_allegro_hand_with_handedness("right")  # canonical morphology 真源
    left = _build_allegro_hand_with_handedness("left")  # 同一 morphology 的物理 YZ 镜像
    mutator = HandMutator(DemoMountAndLimitMutatorCfg())  # 两侧复用同一个确定性 term 配置
    sampled_params = {
        "mount_perturb": {
            "sample": {
                "resolved_self_mode": "general",  # 局部右乘语义 $T'_{PM}=T_{PM}\Delta_M$
                "finger_deltas": {
                    "index": {
                        "delta_pos_local": (0.0020, -0.0010, 0.0005),  # 局部平移增量，单位 m
                        "delta_rotvec_local": (0.030, -0.020, 0.010),  # 局部旋转向量增量，单位 rad
                    },
                    "thumb": {
                        "delta_pos_local": (-0.0015, 0.0008, -0.0004),  # 非零三轴 thumb 局部平移，单位 m
                        "delta_rotvec_local": (-0.025, 0.015, 0.005),  # 非零三轴 thumb 局部转动，单位 rad
                    },
                },
            }
        },
        "limit_tweak": {
            "sample": {
                "resolved_self_mode": "disturb",  # 每个同名关节直接消费同一组边界增量
                "joint_deltas": {
                    "index_j0": {"lower": 0.04, "upper": -0.03},  # 收窄 index 根关节角域，单位 rad
                    "thumb_j1": {"lower": -0.02, "upper": 0.05},  # 放宽 thumb CMC2 角域，单位 rad
                },
            }
        },
    }

    mutated_right = mutator.mutate(right, sampled_params=sampled_params)  # canonical right 直接应用 sample
    mutated_left = mutator.mutate(left, sampled_params=sampled_params)  # left 先 canonicalize，再恢复物理镜像

    assert mutated_right is not None
    assert mutated_left is not None
    assert [joint.name for joint in mutated_left.iter_joints()] == [joint.name for joint in mutated_right.iter_joints()]
    for left_finger, right_finger in zip(mutated_left.fingers, mutated_right.fingers, strict=True):
        _assert_pose_is_mirror(left_finger.mount, right_finger.mount)  # 局部增量后仍满足 $T_L'=ST_R'S$
        for left_joint, right_joint in zip(left_finger.joints, right_finger.joints, strict=True):
            _assert_pose_is_mirror(left_joint.origin, right_joint.origin)  # limit term 不得污染运动链局部位姿
            if right_joint.joint_type == "revolute":
                assert left_joint.limit == right_joint.limit  # same-$q$ 两侧必须共享完全相同的合法角域
