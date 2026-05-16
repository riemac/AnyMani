"""`HandMutator` 容器级回归测试。"""

from __future__ import annotations

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import HandMutator, HandMutatorCfg, LimitTweakCfg, MountPerturbCfg
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
