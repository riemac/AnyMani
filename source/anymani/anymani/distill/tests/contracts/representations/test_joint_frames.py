r"""共享学生静态运动学及单点FK真值的纯张量合同。

三个命题分别证伪：不同手指被错误串联、关节自身旋转错误地移动轴上原点、
以及ghost/错误父节点被当作物理关节。这里使用解析平面机构，不启动仿真。
"""

import pytest
import torch
from anymani.distill.representations.sources.joint_frames import JointKinematicsBank


def _bank() -> JointKinematicsBank:
    """一根两关节链加独立手指，按canonical交错编号0、4与1放置。"""
    feature = torch.zeros(1, 16, 15, dtype=torch.float64)  # 每关节15维静态运动学。
    valid = torch.zeros(1, 16, dtype=torch.bool)  # 只有三个真实关节。
    parent = torch.full((1, 16), -1, dtype=torch.long)  # -1表示统一手部参考系。
    depth = torch.full((1, 16), -1, dtype=torch.long)  # ghost不参与任意深度推进。
    for slot, position in ((0, (0.1, 0.0, 0.0)), (1, (-0.1, 0.0, 0.0)), (4, (0.05, 0.0, 0.0))):
        feature[0, slot, :3] = torch.tensor(position) / 0.1  # 位置输入归一至L=.1m。
        feature[0, slot, 3:12] = torch.eye(3).reshape(-1)  # 零角父子方向一致。
        feature[0, slot, 14] = 1.0  # 三个关节均绕本地z轴旋转。
        valid[0, slot] = True  # 精确物理mask。
        depth[0, slot] = 0  # 先设成独立根。
    parent[0, 4], depth[0, 4] = 0, 1  # 第二深度关节4只受关节0影响。
    return JointKinematicsBank(feature, parent, depth, valid)


def test_fk_origin_and_other_finger_independence() -> None:
    """轴上原点不受自身q影响，另一根手指的角度不能进入本链。"""
    bank = _bank()  # 解析静态真值。
    q = torch.zeros(2, 16, dtype=torch.float64)  # 两个状态仅改变独立手指与末端自身角。
    q[:, 0] = torch.pi / 2  # 根旋转90度把子关节的+x位移转为+y。
    q[1, 1], q[1, 4] = 0.71, -1.2  # 不应该改变关节4的原点。
    out = bank.joint_origins(q, torch.zeros(2, dtype=torch.long))  # 输出米制手坐标。
    expected = torch.tensor([[0.1, 0.05, 0.0]] * 2, dtype=torch.float64)
    torch.testing.assert_close(out[:, 4], expected, atol=1e-9, rtol=1e-7)
    assert torch.count_nonzero(out[:, ~bank.valid[0]]) == 0  # ghost始终精确零。


def test_fk_gradient_of_terminal_origin_is_zero_for_own_joint() -> None:
    """FK辅助目标所监督的是关节原点，不暗中加入离轴指尖点。"""
    bank = _bank()  # 目标与动作的语义区别必须可检验。
    q = torch.zeros(1, 16, dtype=torch.float64, requires_grad=True)
    out = bank.joint_origins(q, torch.zeros(1, dtype=torch.long))
    gradient = torch.autograd.grad(out[0, 4, 1], q)[0]  # 子关节原点y对各q的导数。
    torch.testing.assert_close(gradient[0, 0], torch.tensor(0.05, dtype=torch.float64))
    assert gradient[0, 4] == 0  # 自身轴上点不因自身转动发生平移。
    assert gradient[0, 1] == 0  # 不同手指严格独立。


def test_invalid_parent_and_shape_rejected() -> None:
    """父节点必须有效且位于前一深度，避免错误图产生貌似合理的FK。"""
    bank = _bank()  # 测试故意把真实父指向ghost。
    parent = bank.parent_slot.clone()
    parent[0, 4] = 2  # slot2为ghost，禁止参与FK。
    with pytest.raises(ValueError, match="parent"):
        JointKinematicsBank(bank.features, parent, bank.depth, bank.valid)
    with pytest.raises(ValueError, match="shape"):
        bank.joint_origins(torch.zeros(1, 15), torch.zeros(1, dtype=torch.long))
