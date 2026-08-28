r"""Two-asset heterogeneous N000 tactile-rotation runtime smoke。

该 smoke 显式启动 Isaac Sim，验证 2048 formal route 的最小前缀：两个 unique canonical prototype、
两个独立 env、69D actor、103D critic、16D masked action、round-robin asset rows、ghost 物理零与
frozen-$Z$ manifest identity。它不声明 2048/4096 容量，只证伪 schema/runtime wiring 错误。

运行：

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=30s 600s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_heterogeneous_tactile_rotation_runtime.py -q -s
```
"""

from __future__ import annotations

# ruff: noqa: E402,I001
# AppLauncher 必须早于 pxr/omni/task runtime imports；smoke prefix env var 也必须早于 task cfg import。
import os

os.environ["ANYMANI_HETEROGENEOUS_ASSET_LIMIT"] = "2"

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch

import anymani.tasks.gm  # noqa: F401  # 注册 gm-owned task alias
from anymani.distill.rl.frozen_z import build_frozen_z_provider_from_canonical_artifacts
from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
    HETEROGENEOUS_CANONICAL_ARTIFACTS,
    HETEROGENEOUS_CONTACT_LAYOUT,
    HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
    PPO_DATASET,
)
from anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg import (
    HeterogeneousTactileRotationEnvCfg,
)

TASK_ID = "AnyMani-GM-HeterogeneousAsset-TactileRotation-v0"
"""tasks-owned environment ID；PPO alias 复用完全相同的 env cfg。"""


@pytest.mark.isaacsim
def test_two_asset_reset_step_observation_routing_and_frozen_z() -> None:
    r"""两个 canonical prototypes 在单一 PhysX articulation 中闭合 reset/step 与学习接口。"""

    cfg = HeterogeneousTactileRotationEnvCfg()
    assert cfg.scene.num_envs == len(HETEROGENEOUS_CANONICAL_ARTIFACTS) == 2
    assert len(HETEROGENEOUS_CONTACT_LAYOUT.fingertip_sensor_names) == 4
    assert len(HETEROGENEOUS_CONTACT_LAYOUT.finger_non_tip_sensor_names) == 19
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        runtime_env = env.unwrapped
        runtime_env.sim._app_control_on_stop_handle = None  # smoke 自己关闭 Kit lifecycle
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 69)
        assert obs["critic"].shape == (2, 103)
        assert env.action_space.shape == (2, 16)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(obs["critic"]).all()

        asset_rows = runtime_env._anymani_canonical_asset_row
        active_mask = runtime_env._anymani_canonical_active_joint_mask
        torch.testing.assert_close(asset_rows, torch.tensor([0, 1], device=asset_rows.device))
        expected_masks = torch.tensor(
            [artifact.routing.active_joint_mask for artifact in HETEROGENEOUS_CANONICAL_ARTIFACTS],
            dtype=torch.bool,
            device=active_mask.device,
        )
        torch.testing.assert_close(active_mask, expected_masks)
        torch.testing.assert_close(obs["policy"][:, -17], asset_rows.to(dtype=obs["policy"].dtype))
        torch.testing.assert_close(obs["policy"][:, -16:] > 0.5, active_mask)

        for step_id in range(4):
            zero_action = torch.zeros(2, 16, device=runtime_env.device)
            obs, reward, terminated, truncated, _ = env.step(zero_action)
            assert obs["policy"].shape == (2, 69), step_id
            assert obs["critic"].shape == (2, 103), step_id
            assert torch.isfinite(reward).all(), step_id
            assert terminated.shape == truncated.shape == reward.shape

        robot = runtime_env.scene["robot"]
        assert robot.num_joints == 16
        assert robot.num_bodies == 25
        if (~active_mask).any():
            assert float(robot.data.joint_pos[~active_mask].abs().max()) < 1.0e-5
            assert float(robot.data.joint_vel[~active_mask].abs().max()) < 1.0e-3

        provider = build_frozen_z_provider_from_canonical_artifacts(
            HETEROGENEOUS_CANONICAL_ARTIFACTS,
            dataset_digest=PPO_DATASET.source_sha256,
            manifest_digest=HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
        )
        assert provider.z_table.shape == (2, 21, 128)
        assert provider.identity["asset_ids"] == [artifact.asset_id for artifact in HETEROGENEOUS_CANONICAL_ARTIFACTS]
        assert provider.identity["identity_digest"]
    finally:
        env.close()


# 不在 pytest teardown 调用 ``simulation_app.close()``：Isaac Sim 5.1 会在 pytest 输出失败详情和
# 最终退出码前终止解释器，把真实失败伪装成 shell code 0。该 smoke 总在独立 benchmark process group
# 中运行；测试结束后 Python 自然退出，父 recorder 在 timeout 时也会回收整组进程。
