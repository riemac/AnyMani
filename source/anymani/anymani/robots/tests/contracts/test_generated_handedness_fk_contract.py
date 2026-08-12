r"""Generated 左右手在非零广义坐标下的严格镜像 FK 合同。

静态 builder 字段满足镜像并不自动保证导出后的完整运动链仍正确。首关节 mount
需要在 exporter 中折叠，sidecar 需要保存同一 joint/owner 顺序，robots lowering
还需要把 fixed descendants、空间旋量和祖先掩码组合成动态位姿。因此本文件验证
完整的跨层路径：

$$
\text{HandGenerator}
\to \{\text{URDF},\text{sidecar},\text{meshes}\}
\to \text{HandBank}
\to \text{POE/FK}.
$$

右手是 canonical 真源，左手关于 palm 的 YZ 平面反射。令

$$
S=\operatorname{diag}(-1,1,1),\qquad
H=\operatorname{diag}(-1,1,1,1).
$$

左右手共享 joint identity、顺序、limits 和同一个广义坐标 $q$；对每个
PALM/JOINT/TIP owner，在任意合法构型上必须满足：

$$
T_{L,g}(q)=H\,T_{R,g}(q)\,H.
$$

该合同同时参数化 Allegro 与 LEAP，因为两者的 thumb axes、fixed-root 结构和
fingertip embodiment 不同；只验证一个 family 无法覆盖原始 handedness 失效面。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandBank, HandBankCfg
from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.robots.geometry_kinematics import forward_owner_transforms, lower_hand_geometry_semantics

pytestmark = pytest.mark.contract

_FAMILY_CASES = (
    ("single_palm_allegro", "allegro"),
    ("single_palm_leap", "leap"),
)
"""两类 canonical single-palm morphology；均为 4 fingers / 16 revolute DOF。"""


def _full_connectivity_pool(hand_preset: str, family: str) -> dict[str, dict[str, list[str]]]:
    r"""构造唯一 full-chain connectivity 组合，使一次 run 只生成一对左右手。

    Args:
        hand_preset (str): canonical hand preset 名称。
        family (str): finger recipe family，当前为 ``allegro`` 或 ``leap``。

    Returns:
        dict[str, dict[str, list[str]]]: 每个 slot 仅含一个 full-chain recipe 的
        connectivity façade；左右 topology 因而共享完全相同的离散选择。
    """

    thumb_recipe = f"{family}_thumb_full"  # thumb 与 palm family 绑定，保留完整 4-DOF 链
    non_thumb_recipe = f"{family}_non_thumb_full"  # index/middle/ring 各保留完整 4-DOF 链
    return {
        hand_preset: {
            "thumb": [thumb_recipe],
            "index": [non_thumb_recipe],
            "middle": [non_thumb_recipe],
            "ring": [non_thumb_recipe],
        }
    }


def _generate_and_load_pair(tmp_path: Path, *, hand_preset: str, family: str):
    r"""生成左右完整 bundle，并通过默认严格 Bank 安全门读取几何语义。

    Args:
        tmp_path (Path): pytest 隔离目录；所有 URDF、sidecar 与 meshes 均写入此处。
        hand_preset (str): 本次测试的 canonical hand preset。
        family (str): 对应 finger/connectivity family。

    Returns:
        dict[str, HandContainer]: 以 ``left`` / ``right`` 为键的已解析 bundle。
    """

    output_root = tmp_path / family  # family 独立 run 根，避免参数化用例互相发现资产
    cfg = HandGeneratorCfg(
        mode="made",  # 只验证 canonical pre-made lowering，不引入随机 post-mutate
        artifact_level="bundle",  # 必须真实写出 URDF、sidecar、tree 与 materialized meshes
        output_dir=output_root,
        handedness="all",  # 同一 run 生成物理 left/right，防止配置漂移
        hand_presets=[hand_preset],
        connectivity_presets=_full_connectivity_pool(hand_preset, family),
        mixed=False,  # family 内基准 pair；mixed/drop 由 assets connectivity 合同独立覆盖
        missing=False,  # 四指同构保证 owner identity 可逐项配对
        Validate=None,  # 测试命题是 handedness/FK；机械可用性 validator 与其正交
        premade_parallel=False,  # 两个样本顺序执行，使失败栈保持确定且易于核对
    )
    results = list(HandGenerator(cfg).generate_batch())  # 物化一对可由真实 importer 消费的 bundle
    assert len(results) == 2
    assert all(result.urdf_path is not None for result in results)
    run_roots = {result.urdf_path.parents[2] for result in results if result.urdf_path is not None}
    assert len(run_roots) == 1  # 左右 bundle 必须属于同一个带 summary 的生成 run
    run_root = run_roots.pop()

    # 默认 `allow_legacy_left_handedness=False`：新 left 必须凭严格证书直接通过，不能靠审计 override。
    selection = HandBank(
        HandBankCfg(
            source_mode="pre_made",
            selection_mode="all",
            pre_made_path=run_root,
            require_geometry_semantics=True,
        )
    ).resolve()
    assert len(selection.assets) == 2
    return {str(asset.sidecar["handedness"]): asset for asset in selection.assets}


def _nonzero_legal_configurations(joint_limits: torch.Tensor) -> torch.Tensor:
    r"""在每个 joint 的公共 limits 内构造两个确定性非零构型。

    对第 $i$ 个 joint 采用随索引变化的内插系数，避免所有关节处于相同归一化相位：

    $$
    q_i=q_{\min,i}+\alpha_i(q_{\max,i}-q_{\min,i}),
    \qquad \alpha_i\in[0.23,0.77].
    $$

    第二个样本使用 $1-\alpha_i$，从而同时激励每条 finger chain 的正负功能相位，
    又与 limit 边界保持至少 $23\%$ 的区间余量。

    Args:
        joint_limits (torch.Tensor): 形状 ``[N_J,2]`` 的 $(q_{min},q_{max})$，单位 rad。

    Returns:
        torch.Tensor: 形状 ``[2,N_J]`` 的合法非零构型，单位 rad。
    """

    joint_count = joint_limits.shape[0]  # 当前 generated full hand 为 $N_J=16$
    alpha = torch.linspace(0.23, 0.77, joint_count, dtype=joint_limits.dtype)  # 每关节不同相位
    lower = joint_limits[:, 0]  # $q_{\min}$，形状 `[N_J]`，单位 rad
    width = joint_limits[:, 1] - lower  # 合法角域宽度 $q_{\max}-q_{\min}$，单位 rad
    q_first = lower + alpha * width  # 第一组内部构型，不触碰 limit clipping
    q_second = lower + (1.0 - alpha) * width  # 第二组镜像采样相位，仍使用相同物理 $q$ 符号
    q = torch.stack((q_first, q_second), dim=0)  # `[B=2,N_J]`
    assert torch.count_nonzero(q).item() == q.numel()  # 本合同不得退化成只验证 home/zero pose
    return q


@pytest.mark.parametrize(("hand_preset", "family"), _FAMILY_CASES)
def test_exported_pair_has_strict_mirrored_owner_fk_at_same_nonzero_q(
    tmp_path: Path,
    hand_preset: str,
    family: str,
) -> None:
    r"""导出左右手在同一非零 $q$ 上的全部 owner 位姿必须严格镜像。

    Args:
        tmp_path (Path): 当前参数化用例的隔离生成目录。
        hand_preset (str): canonical hand preset 名称。
        family (str): 当前 hand/finger family。
    """

    pair = _generate_and_load_pair(tmp_path, hand_preset=hand_preset, family=family)  # 真实落盘/Bank 路径
    left_semantics = pair["left"].geometry_semantics  # 新证书 left 的完整 fixed/revolute 运动链
    right_semantics = pair["right"].geometry_semantics  # canonical right 的同构运动链
    assert left_semantics is not None
    assert right_semantics is not None

    left_spec = lower_hand_geometry_semantics(left_semantics, dtype=torch.float64)  # 双精度 FK 数值 oracle
    right_spec = lower_hand_geometry_semantics(right_semantics, dtype=torch.float64)
    assert left_spec.joint_names == right_spec.joint_names  # names 同时承担 same-$q$ policy identity
    assert left_spec.owner_ids == right_spec.owner_ids  # PALM/JOINT/TIP owner 必须可逐项配对
    assert left_spec.joint_limits is not None
    assert right_spec.joint_limits is not None
    torch.testing.assert_close(left_spec.joint_limits, right_spec.joint_limits, atol=0.0, rtol=0.0)

    q = _nonzero_legal_configurations(right_spec.joint_limits)  # 左右两侧共享同一 `[2,16]` 物理角
    left_fk = forward_owner_transforms(left_spec, q)  # `[2,G,4,4]`，owner local -> palm/hand frame
    right_fk = forward_owner_transforms(right_spec, q)
    reflection = torch.diag(torch.tensor((-1.0, 1.0, 1.0, 1.0), dtype=torch.float64))  # 齐次 $H$
    expected_left_fk = reflection @ right_fk @ reflection  # $T_L(q)=H T_R(q) H$

    # 双精度 POE 的余量只吸收 RPY/三角函数舍入；厘米级或轴符号错误会远超该阈值。
    torch.testing.assert_close(left_fk, expected_left_fk, atol=2.0e-9, rtol=2.0e-9)
