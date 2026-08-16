from __future__ import annotations

import math

import pytest

from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.generator.mutate import (
    HandMutator,
    HandMutatorCfg,
    LinkProximalOverlapCfg,
    LinkProximalOverlapMutator,
    LinkScaleCfg,
)
from assets.presets import make_human_like_builder_cfg


def _build_hand(*, family: str, handedness: str = "right"):
    r"""构造一只保持官方 family 几何锚点的 deterministic generated hand。

    LEAP 用于检验从零 overhang 开始的正向延伸；Allegro 用于检验已有负
    mesh offset 的缩减。两者都直接走 builder 真源，不依赖 generated 目录。
    """

    return HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name=f"{family}_proximal_overlap_{handedness}",
            family=family,
            handedness=handedness,
            palm_cfg=f"com_{family}",
            finger_cfg=f"{family}_non_thumb_v1",
            thumb_cfg=f"{family}_thumb_v1",
        )
    ).build()


def _joint_by_name(hand, joint_name: str):
    r"""按稳定 joint identity 返回 joint-centric child-link 描述。"""

    for joint in hand.iter_joints():
        if joint.name == joint_name:
            return joint
    raise KeyError(joint_name)


def _axial_support(joint) -> tuple[float, float]:
    r"""返回主体 primitive 在 child-link 局部 $y$ 轴上的闭区间 $[a,b]$。

    当前 regular generated link 的 box/cylinder/elliptic-cylinder 都以局部
    $y$ 为生长轴。测试只读取 box mother，以最小公式直接锁住 proximal 与
    distal boundary，而不是只比较长度这种不足以区分双端伸缩的量。
    """

    geometry = joint.collisions[0].geometry  # collision 是 SSL 与物理几何的权威真源
    assert geometry.kind == "box"
    length = float(geometry.size[1])  # $L$，局部 $y$ 轴全长，单位 m
    center_y = float(joint.collisions[0].origin.pos[1])  # $c_y$，collision frame 中心
    return center_y - length / 2.0, center_y + length / 2.0  # $[c_y-L/2,c_y+L/2]$


def _overlap_cfg(*, self_mode: str = "disturb", ratio: tuple[float, float] = (-0.1, 0.2)):
    r"""构造固定科研语义的近端重叠配置。"""

    return LinkProximalOverlapCfg(
        self_mode=self_mode,
        overhang_delta_ratio=ratio,
        max_parent_overlap_ratio=0.5,
        distrib="uniform",
        boundary_policy="clip",
    )


def test_positive_ratio_extends_only_the_child_proximal_boundary() -> None:
    r"""LEAP 零 overhang link 应只向 parent 内部延伸，distal boundary 保持不动。

    对 `index_j1`，基础区间为 $[0,s_i]$。显式给定
    $\eta_i=0.2$ 后应得到 $[-0.2s_i,s_i]$；本 joint 与下一 joint 的 frame
    都不得因近端几何编辑发生位移。
    """

    hand = _build_hand(family="leap")
    before_target = _joint_by_name(hand, "index_j1")
    before_first_active = _joint_by_name(hand, "index_j0")
    before_next = _joint_by_name(hand, "index_j2")
    before_root_fixed = _joint_by_name(hand, "index_root_fixed")
    before_tip = _joint_by_name(hand, "index_tip")
    before_support = _axial_support(before_target)

    mutated = LinkProximalOverlapMutator(_overlap_cfg()).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "disturb",
                "joint_delta_ratio": {"index_j1": 0.2},
            }
        },
    )

    assert mutated is not None
    after_target = _joint_by_name(mutated, "index_j1")
    after_support = _axial_support(after_target)
    child_span = before_support[1]  # LEAP 当前 $d_i=0$，因此 distal boundary 就是有效 span
    expected_overhang = 0.2 * child_span  # $o_i'=\eta_i s_i$，尚未触发 parent-relative cap
    assert math.isclose(after_support[0], -expected_overhang, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_support[1], before_support[1], rel_tol=0.0, abs_tol=1e-12)
    assert after_target.origin == before_target.origin  # joint $i$ frame 不动
    assert _joint_by_name(mutated, "index_j2").origin == before_next.origin  # downstream frame 不动
    assert _joint_by_name(mutated, "index_j0") == before_first_active  # 首 active child 是硬排除项
    assert _joint_by_name(mutated, "index_root_fixed") == before_root_fixed  # palm-side fixed root 不参与
    assert _joint_by_name(mutated, "index_tip") == before_tip  # fixed tip 不参与
    assert after_target.visuals[0].geometry == after_target.collisions[0].geometry
    assert after_target.visuals[0].origin == after_target.collisions[0].origin


def test_negative_ratio_reduces_existing_allegro_overhang_without_making_gap() -> None:
    r"""负 ratio 应从 proximal side 缩减 Allegro 既有 overhang，并以零为下界。

    当前 official-aligned Allegro preset 把 `index_j3` 的 proximal boundary 放在
    $d_i=-6$ mm。负扰动先按 child effective span 产生 signed delta，再将最终
    overhang 裁到 $[0,\kappa E_{i-1}^0]$；distal boundary 始终保持原值。
    """

    hand = _build_hand(family="allegro")
    before = _joint_by_name(hand, "index_j3")
    before_support = _axial_support(before)
    assert math.isclose(before_support[0], -0.006, rel_tol=0.0, abs_tol=1e-12)

    mutated = LinkProximalOverlapMutator(_overlap_cfg()).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "disturb",
                "joint_delta_ratio": {"index_j3": -0.1},
            }
        },
    )

    assert mutated is not None
    after = _joint_by_name(mutated, "index_j3")
    after_support = _axial_support(after)
    child_span = before_support[1]  # $s_i=L_i+d_i=16$ mm
    expected_overhang = 0.006 - 0.1 * child_span  # $o_i'=o_i^0+\eta_i s_i$
    assert math.isclose(after_support[0], -expected_overhang, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_support[1], before_support[1], rel_tol=0.0, abs_tol=1e-12)

    fully_retracted = LinkProximalOverlapMutator(_overlap_cfg(ratio=(-1.0, -1.0))).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "disturb",
                "joint_delta_ratio": {"index_j3": -1.0},
            }
        },
    )
    assert fully_retracted is not None
    retracted_support = _axial_support(_joint_by_name(fully_retracted, "index_j3"))
    assert math.isclose(retracted_support[0], 0.0, rel_tol=0.0, abs_tol=1e-12)  # 不允许产生正 gap
    assert math.isclose(retracted_support[1], before_support[1], rel_tol=0.0, abs_tol=1e-12)


def test_parent_relative_cap_limits_final_total_overhang() -> None:
    r"""最终总 overhang 不得超过变异前 parent 净 span 的可配置比例。"""

    hand = _build_hand(family="leap")
    target = _joint_by_name(hand, "index_j1")
    parent_span = float(target.origin.pos[1])  # actual surviving chain 中 parent 到当前 joint 的净 span

    mutated = LinkProximalOverlapMutator(_overlap_cfg(ratio=(10.0, 10.0))).mutate(
        hand,
        sampled_params={
            "sample": {
                "resolved_self_mode": "disturb",
                "joint_delta_ratio": {"index_j1": 10.0},
            }
        },
    )

    assert mutated is not None
    support = _axial_support(_joint_by_name(mutated, "index_j1"))
    assert math.isclose(-support[0], 0.5 * parent_span, rel_tol=0.0, abs_tol=1e-12)


class _ScaleThenOverlapCfg(HandMutatorCfg):
    r"""先声明 link scale、后声明 proximal overlap 的组合测试配置。"""

    link_scale = LinkScaleCfg(
        self_mode="only_length",
        scale_type="rel",
        link_scale=(1.2, 1.2),
        distrib="uniform",
    )
    link_proximal_overlap = _overlap_cfg(ratio=(0.2, 0.2))


class _OverlapThenScaleCfg(HandMutatorCfg):
    r"""反向声明同一组 term，用于证伪依赖 class attribute 顺序的实现。"""

    link_proximal_overlap = _overlap_cfg(ratio=(0.2, 0.2))
    link_scale = LinkScaleCfg(
        self_mode="only_length",
        scale_type="rel",
        link_scale=(1.2, 1.2),
        distrib="uniform",
    )


@pytest.mark.parametrize("cfg_type", [_ScaleThenOverlapCfg, _OverlapThenScaleCfg])
def test_link_scale_and_overlap_compose_once_independent_of_term_order(cfg_type) -> None:
    r"""两个 term 应合成为一次轴向几何写入，并保持 link-scale distal boundary。

    `index_j1` 先由 $L_i$ 变成 $L_i^s=1.2L_i$，随后以缩放后的
    $s_i^\star=L_i^s$ 计算 $o_i'=0.2s_i^\star$。overlap 不得再次推进
    `index_j2.origin`，最终几何远端必须与 link-scale 后的 joint span 重合。
    """

    hand = _build_hand(family="leap")
    before_target = _joint_by_name(hand, "index_j1")
    before_next = _joint_by_name(hand, "index_j2")
    before_length = float(before_target.collisions[0].geometry.size[1])
    scaled_length = 1.2 * before_length
    sampled_params = {
        "link_scale": {
            "sample": {
                "resolved_self_mode": "only_length",
                "joint_length_scale": {"index_j1": 1.2},
                "length_scale": {},
                "width_scale": None,
                "height_scale": None,
            }
        },
        "link_proximal_overlap": {
            "sample": {
                "resolved_self_mode": "disturb",
                "joint_delta_ratio": {"index_j1": 0.2},
            }
        },
    }

    mutated = HandMutator(cfg_type()).mutate(hand, sampled_params=sampled_params)

    assert mutated is not None
    after_target = _joint_by_name(mutated, "index_j1")
    after_next = _joint_by_name(mutated, "index_j2")
    support = _axial_support(after_target)
    assert math.isclose(support[0], -0.2 * scaled_length, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(support[1], scaled_length, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_next.origin.pos[1], scaled_length, rel_tol=0.0, abs_tol=1e-12)
    assert after_target.origin == before_target.origin
    assert after_next.origin.pos[0] == before_next.origin.pos[0]
    assert after_next.origin.pos[2] == before_next.origin.pos[2]


def test_overlap_cfg_rejects_invalid_parent_ratio_and_mode_probabilities() -> None:
    r"""科研配置应在运行前拒绝突破半 parent span 的 cap 与错误 mode 概率。"""

    with pytest.raises(ValueError, match="max_parent_overlap_ratio"):
        LinkProximalOverlapCfg(
            overhang_delta_ratio=(-0.1, 0.2),
            max_parent_overlap_ratio=0.6,
        )
    with pytest.raises(ValueError, match="probabilities must sum"):
        LinkProximalOverlapCfg(
            self_mode={"identity": 0.2, "disturb": 0.2},
            overhang_delta_ratio=(-0.1, 0.2),
        )
