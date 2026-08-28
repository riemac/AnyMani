r"""Five-mother canonical single-PhysX-batch runtime smoke。

该 smoke 依赖 Isaac Sim / PhysX / ContactSensor，不能由默认 contract pytest 收集；显式
运行时验证五个 source DOF strata 在同一 articulation 中 reset/step，116D actor obs、
contact routing、per-env active mask 与 ghost position/velocity tolerance 闭合。
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import anymani.distill.rl  # noqa: E402,F401
import anymani.tasks.gm  # noqa: E402,F401
from anymani.tasks.gm.canonical_unified_env_cfg import (  # noqa: E402
    CANONICAL_ARTIFACTS,
    CANONICAL_CONTACT_LAYOUT,
    CanonicalUnifiedInHandEnvCfg,
)


@pytest.mark.isaacsim
def test_canonical_five_mother_single_articulation_reset_step() -> None:
    r"""五个 source DOF strata 同时 reset/step，ghost 状态留在物理容差内。"""

    cfg = CanonicalUnifiedInHandEnvCfg()
    env = gym.make("AnyMani-GM-Canonical-InHand-v0", cfg=cfg)
    try:
        assert [artifact.routing.source_dof_count for artifact in CANONICAL_ARTIFACTS] == [7, 9, 12, 14, 16]
        assert CANONICAL_CONTACT_LAYOUT.fingertip_link_names == (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
        )
        obs, _ = env.reset()
        assert obs["policy"].shape == (cfg.scene.num_envs, 116)

        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 16
        assert robot.num_bodies == 25
        active_mask = env.unwrapped._anymani_canonical_active_joint_mask
        assert active_mask.shape == (cfg.scene.num_envs, 16)
        assert int((~active_mask).sum()) == 704  # 32 * ((9+7+4+2+0) ghost DOFs)
        asset_rows = env.unwrapped._anymani_canonical_asset_row
        expected_rows = torch.arange(cfg.scene.num_envs, device=asset_rows.device) % len(CANONICAL_ARTIFACTS)
        torch.testing.assert_close(asset_rows, expected_rows)
        expected_masks = torch.tensor(
            [artifact.routing.active_joint_mask for artifact in CANONICAL_ARTIFACTS],
            dtype=torch.bool,
            device=active_mask.device,
        )
        torch.testing.assert_close(active_mask, expected_masks[asset_rows])

        for _ in range(4):
            zero_action = torch.zeros(cfg.scene.num_envs, 16, device=env.unwrapped.device)
            obs, reward, terminated, truncated, _ = env.step(zero_action)
            assert obs["policy"].shape == (cfg.scene.num_envs, 116)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(robot.data.joint_pos).all()
            assert torch.isfinite(robot.data.joint_vel).all()

        ghost_positions = robot.data.joint_pos[~active_mask].abs()
        ghost_velocities = robot.data.joint_vel[~active_mask].abs()
        print(
            "[canonical-smoke] "
            f"ghost_max_position={float(ghost_positions.max()):.9e} "
            f"ghost_max_velocity={float(ghost_velocities.max()):.9e}"
        )
        ghost_indices = torch.nonzero(~active_mask, as_tuple=False)
        worst_position = int(torch.argmax(ghost_positions))
        worst_env, worst_joint = (int(value) for value in ghost_indices[worst_position])
        assert float(ghost_positions.max()) < 1.0e-5, {
            "value": float(ghost_positions.max()),
            "env": worst_env,
            "joint": robot.joint_names[worst_joint],
        }
        assert float(ghost_velocities.max()) < 1.0e-3, float(ghost_velocities.max())
    finally:
        env.close()


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免显式 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()
