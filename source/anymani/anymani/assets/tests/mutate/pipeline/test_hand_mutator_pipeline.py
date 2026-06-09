"""`HandMutator` 容器级回归测试。"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import (
    HandMutator,
    HandMutatorCfg,
    LimitTweakCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
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
