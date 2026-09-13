r"""新旧手族梯度审计的纯张量合同，不启动Isaac或修改训练参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from anymani.distill.rl.scripts.audit_palm_rotation_checkpoint_gradients import (
    _family_gradient_evidence,
    _family_half_gradient_sums,
    _training_argv,
)


def test_family_gradient_sums_match_full_batch_with_unequal_slice_counts() -> None:
    r"""先累计样本梯度和再除完整分母；微批中各half数量不同也不能改变估计量。"""

    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))  # 未使用参数仍占据固定坐标。
    features = torch.arange(32, dtype=torch.float64).reshape(16, 2) / 10.0
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    halves = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    parameters = (parameter, unused)
    full_sums, full_counts = _family_half_gradient_sums(features @ parameter, groups, halves, parameters)
    pieces = [
        _family_half_gradient_sums(features[s] @ parameter, groups[s], halves[s], parameters)
        for s in (slice(0, 8), slice(8, 16))
    ]
    torch.testing.assert_close(pieces[0][0] + pieces[1][0], full_sums)
    torch.testing.assert_close(pieces[0][1] + pieces[1][1], full_counts)
    assert full_sums.shape == (2, 2, 3)  # 两族、两半、完整参数坐标，含unused维度。
    assert torch.count_nonzero(full_sums[..., -1]) == 0
    assert parameter.grad is None and unused.grad is None  # autograd.grad不写主梯度。


def test_family_fairgrad_reuses_two_task_scale_invariant_direction() -> None:
    r"""正交但范数相差100倍时，解析FairGrad方向为(1,1)，而非普通均值。"""

    gradients = torch.tensor([[[100.0, 0.0], [100.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    counts = torch.full((2, 2), 3, dtype=torch.long)
    dense, evidence = _family_gradient_evidence(gradients * counts[..., None], counts)
    torch.testing.assert_close(dense["fairgrad_direction"], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(dense["mean_direction"], torch.tensor([50.0, 0.5], dtype=torch.float64))
    assert evidence["fairgrad"]["active_tasks"] == 2
    assert evidence["fairgrad"]["shared_conflict_blocked"] is False
    assert evidence["candidates"]["fairgrad"]["norm"] == pytest.approx(2.0**0.5)
    assert evidence["candidates"]["fairgrad_mean_norm"]["norm"] == pytest.approx(evidence["candidates"]["mean"]["norm"])
    scaled = gradients * torch.tensor([7.0, 0.2])[:, None, None]
    scaled_dense, _ = _family_gradient_evidence(scaled * counts[..., None], counts)
    torch.testing.assert_close(scaled_dense["fairgrad_direction"], dense["fairgrad_direction"])


def test_family_opposition_is_recorded_as_blocked_not_common_progress() -> None:
    r"""严格反向的两族梯度没有共同严格下降方向，保留SSL的阻塞政策。"""

    sums = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [-1.0, 0.0]]])
    dense, evidence = _family_gradient_evidence(sums, torch.ones((2, 2), dtype=torch.long))
    assert evidence["fairgrad"]["shared_conflict_blocked"] is True
    assert torch.count_nonzero(dense["fairgrad_direction"]) == 0
    assert evidence["candidates"]["fairgrad"]["group_loss_decrease_rates"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="balanced"):
        _family_gradient_evidence(sums, torch.tensor([[1, 1], [2, 2]]))


def test_checkpoint_audit_restores_explicit_joint_pose_anchor(tmp_path: Path) -> None:
    r"""新warm分支取消初姿锚，审计不能回落到训练入口默认的−0.5。"""

    args = argparse.Namespace(
        cohort_lock=tmp_path / "cohort.json", checkpoint=tmp_path / "model.pth", output=tmp_path / "audit"
    )
    checkpoint = {
        "epoch": 160,
        "anymani_identity": {
            "identity_schema_version": "4.0.0",
            "policy": {"arm": "direct_token"},
            "training": {
                "minibatch_count": 20,
                "num_envs": 1024,
                "seed": 42,
                "gradient_accumulation_steps": 5,
                "gradient_probe_frequency": 0,
                "full_gradient_shadow_frequency": 0,
                "advantage_normalization_scope": "per_asset_rollout",
                "history_encoder": "tcn",
                "joint_pose_anchor_weight": 0.0,
            },
        },
    }
    argv = _training_argv(args, checkpoint)
    assert argv[argv.index("--joint_pose_anchor_weight") + 1] == "0.0"
