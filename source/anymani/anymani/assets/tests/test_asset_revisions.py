"""局部限位修订的科研合同：只扩大指定短指的屈曲范围。"""

from copy import deepcopy

import pytest
from anymani.assets.asset_revisions import widen_two_joint_allegro_flexion
from anymani.assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg


def _hand() -> HandCfg:
    """构造短指、完整指和短拇指共存的最小串联手。"""

    fingers = []  # 分别覆盖 two-joint、three-joint、four-joint 与排除 thumb
    for name, dof in (("index", 2), ("middle", 3), ("ring", 4), ("thumb", 2)):
        joints = []  # 末端 fixed TIP 不得被误计入活动关节数
        parent = "palm"  # 每条 chain 独立从掌部开始
        for depth in range(dof):
            suffix = ("mcp1", "mcp2", "pip", "dip")[depth]  # 显式 anatomy slot
            child = f"{name}_{suffix}"  # j1 的 child 负责标识 MCP2
            joints.append(
                JointCfg(
                    name=f"{name}_j{depth}",
                    parent=parent,
                    child=child,
                    limit={"lower": -0.196, "upper": 1.61, "effort": 10.0, "velocity": 3.14},
                )
            )
            parent = child  # 保持 schema 要求的串联链闭合
        joints.append(JointCfg(name=f"{name}_tip", parent=parent, child=f"{name}_tip", joint_type="fixed", limit=None))
        fingers.append(FingerCfg(name=name, joints=joints))
    return HandCfg(
        name="revision_contract",
        family="allegro",
        handedness="right",
        fingers=fingers,
        metadata={
            "premade_connectivity": {
                "slot_family_map": {name: "allegro" for name in ("index", "middle", "ring", "thumb")}
            }
        },
    )


def test_only_second_flexion_of_exactly_two_joint_non_thumb_changes():
    """原 HandCfg 不变；副本只允许 index_j1.upper 一项发生变化。"""

    original = _hand()  # 2/3/4 DOF non-thumb 加 2 DOF thumb 的边界组合
    before = deepcopy(original.to_dict())  # 完整物理/schema 快照，不只比关节数量
    revised, edits = widen_two_joint_allegro_flexion(original, upper_rad=2.23)
    assert original.to_dict() == before  # 不原地覆写旧资产
    assert len(edits) == 1 and edits[0].joint_name == "index_j1"
    assert edits[0].old_upper_rad == 1.61 and edits[0].new_upper_rad == 2.23
    changed = revised.to_dict()  # 逐字段证明 geometry/effort/velocity/thumb/healthy fingers 未动
    assert changed["fingers"][0]["joints"][1]["limit"]["upper"] == 2.23
    changed["fingers"][0]["joints"][1]["limit"]["upper"] = 1.61
    assert changed == before


def test_slot_family_excludes_leap_and_expansion_never_narrows():
    """base palm 不代替 finger family；重复修订不能把更大上限压低。"""

    original = _hand()  # 即使 palm 来自 Allegro，LEAP slot 仍必须跳过
    original.metadata["premade_connectivity"]["slot_family_map"]["index"] = "leap"
    revised, edits = widen_two_joint_allegro_flexion(original)
    assert not edits and revised.to_dict() == original.to_dict()
    original = _hand()  # 已扩大到 2.4 rad 的资产不能被默认 2.23 rad 缩窄
    original.fingers[0].joints[1].limit.upper = 2.4
    revised, edits = widen_two_joint_allegro_flexion(original)
    assert not edits and revised.to_dict() == original.to_dict()


def test_rejects_ambiguous_two_joint_anatomy_before_editing():
    """两活动关节未必就是 MCP1+MCP2，错误 anatomy 必须拒绝。"""

    hand = _hand()  # 修改 child label 模拟保留 j0+j2 后重编号的另一种拓扑
    hand.fingers[0].joints[1].child = "index_pip"
    with pytest.raises(ValueError, match="MCP1.*MCP2"):
        widen_two_joint_allegro_flexion(hand)


@pytest.mark.parametrize("upper", [float("nan"), float("inf"), 0.0, -1.0])
def test_rejects_invalid_upper(upper):
    """输入必须是有限、正值的弧度上限。"""

    with pytest.raises(ValueError, match="finite.*positive"):
        widen_two_joint_allegro_flexion(_hand(), upper_rad=upper)
