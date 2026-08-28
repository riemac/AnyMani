#!/usr/bin/env python3
r"""参数化验证 heterogeneous Stage/PhysX/reset/step 容量与 routing。

``asset_count`` 选择 ``ppo.yaml.train`` 的有序前缀；``replicas`` 决定独立环境数
``num_envs = asset_count * replicas``。MultiAssetSpawner 和 canonical reset event 都使用 round-robin，故 replica
``r`` 的 row 轴严格重复 ``0..asset_count-1``，共享 mask/geometry identity，但物体与 simulator state 独立。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import cast


def _parse_args() -> argparse.Namespace:
    r"""解析 prototype、replica 与连续 policy-step 数。"""

    parser = argparse.ArgumentParser(description="Smoke AnyMani heterogeneous runtime capacity.")
    parser.add_argument("--asset_count", type=int, required=True)
    parser.add_argument("--replicas", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.asset_count <= 2048:
        parser.error("--asset_count must be within [1,2048]")
    if args.steps < 1:
        parser.error("--steps must be positive")
    return args


def main() -> int:
    r"""创建真实 Stage/PhysX scene，reset 并持续执行 masked zero actions。"""

    args = _parse_args()
    if args.asset_count == 2048:
        os.environ.pop("ANYMANI_HETEROGENEOUS_ASSET_LIMIT", None)  # formal route 命中完整 prepared cache
    else:
        os.environ["ANYMANI_HETEROGENEOUS_ASSET_LIMIT"] = str(args.asset_count)

    from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase

    record_optional_rl_phase("app_launcher", "start", headless=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _simulation_app = app_launcher.app
    record_optional_rl_phase("app_launcher", "complete")

    import anymani.tasks.gm  # noqa: F401  # 注册 tasks-owned environment ID
    import gymnasium as gym
    import torch
    from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
        HETEROGENEOUS_ACTIVE_MASK_ROWS,
        HETEROGENEOUS_CANONICAL_ARTIFACTS,
    )
    from anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg import (
        HeterogeneousTactileRotationEnvCfg,
    )
    from isaaclab.envs import ManagerBasedRLEnv

    num_envs = args.asset_count * args.replicas
    cfg = HeterogeneousTactileRotationEnvCfg()
    cfg.scene.num_envs = num_envs
    record_optional_rl_phase(
        "environment_construct",
        "start",
        asset_count=args.asset_count,
        replicas=args.replicas,
        num_envs=num_envs,
    )
    env = gym.make("AnyMani-GM-HeterogeneousAsset-TactileRotation-v0", cfg=cfg)
    record_optional_rl_phase("environment_construct", "complete", num_envs=num_envs)
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        record_optional_rl_phase("runtime_reset", "start", num_envs=num_envs)
        reset_output = env.reset()
        obs = cast(dict[str, torch.Tensor], reset_output[0])
        record_optional_rl_phase("runtime_reset", "complete", num_envs=num_envs)

        if obs["policy"].shape != (num_envs, 69) or obs["critic"].shape != (num_envs, 103):
            raise RuntimeError(
                f"unexpected observation shapes policy={tuple(obs['policy'].shape)} critic={tuple(obs['critic'].shape)}"
            )
        if env.action_space.shape != (num_envs, 16):
            raise RuntimeError(f"unexpected action shape {env.action_space.shape}")
        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(obs["critic"]).all():
            raise RuntimeError("non-finite observation after reset")

        expected_rows = torch.arange(num_envs, device=runtime_env.device, dtype=torch.long) % args.asset_count
        asset_rows = cast(torch.Tensor, getattr(runtime_env, "_anymani_canonical_asset_row"))
        torch.testing.assert_close(asset_rows, expected_rows)
        source_masks = torch.tensor(HETEROGENEOUS_ACTIVE_MASK_ROWS, dtype=torch.bool, device=runtime_env.device)
        expected_masks = source_masks[expected_rows]
        active_mask = cast(torch.Tensor, getattr(runtime_env, "_anymani_canonical_active_joint_mask"))
        torch.testing.assert_close(active_mask, expected_masks)

        action = torch.zeros(num_envs, 16, device=runtime_env.device)
        record_optional_rl_phase("runtime_step", "start", num_envs=num_envs, policy_steps=args.steps)
        step_started = time.perf_counter()
        for step_id in range(args.steps):
            step_output = env.step(action)
            obs = cast(dict[str, torch.Tensor], step_output[0])
            reward = cast(torch.Tensor, step_output[1])
            terminated = cast(torch.Tensor, step_output[2])
            truncated = cast(torch.Tensor, step_output[3])
            if not torch.isfinite(reward).all():
                raise RuntimeError(f"non-finite reward at policy step {step_id}")
            if terminated.shape != truncated.shape or terminated.shape != reward.shape:
                raise RuntimeError(f"done/reward shape mismatch at policy step {step_id}")
        step_seconds = time.perf_counter() - step_started
        record_optional_rl_phase(
            "runtime_step",
            "complete",
            num_envs=num_envs,
            policy_steps=args.steps,
            environment_steps=num_envs * args.steps,
            step_seconds=step_seconds,
            environment_steps_per_second=num_envs * args.steps / step_seconds,
        )

        record_optional_rl_phase("runtime_contract", "start", num_envs=num_envs)
        robot = runtime_env.scene["robot"]
        if robot.num_joints != 16 or robot.num_bodies != 25:
            raise RuntimeError(f"canonical articulation shape is {robot.num_joints} joints/{robot.num_bodies} bodies")
        ghost_limit_max = 0.0
        ghost_q_max = 0.0
        ghost_qd_max = 0.0
        if (~active_mask).any():
            ghost_limit_max = float(robot.data.joint_pos_limits[~active_mask].abs().max())
            ghost_q_max = float(robot.data.joint_pos[~active_mask].abs().max())
            ghost_qd_max = float(robot.data.joint_vel[~active_mask].abs().max())
            actor_ghost = torch.cat(
                (obs["policy"][:, :16], obs["policy"][:, 16:32], obs["policy"][:, 32:48]), dim=1
            )  # q、target、last action
            actor_ghost_mask = torch.cat((~active_mask, ~active_mask, ~active_mask), dim=1)
            critic_ghost = torch.cat(
                (
                    obs["critic"][:, :16],
                    obs["critic"][:, 16:32],
                    obs["critic"][:, 32:48],
                    obs["critic"][:, 48:64],
                ),
                dim=1,
            )  # q、qd、target、last action
            critic_ghost_mask = torch.cat((~active_mask, ~active_mask, ~active_mask, ~active_mask), dim=1)
            if ghost_limit_max != 0.0:
                raise RuntimeError(f"ghost hard limits are not exact zero: {ghost_limit_max:.3e}")
            if torch.count_nonzero(actor_ghost[actor_ghost_mask]).item() != 0:
                raise RuntimeError("actor ghost q/target/last-action observation is not exact zero")
            if torch.count_nonzero(critic_ghost[critic_ghost_mask]).item() != 0:
                raise RuntimeError("critic ghost q/qd/target/last-action observation is not exact zero")
            # PhysX 的零宽 revolute limit 仍允许亚毫弧度级 solver penetration；该 state 已从 obs 精确屏蔽。
            if ghost_q_max >= 1.0e-3 or ghost_qd_max >= 2.0e-3:
                raise RuntimeError(f"ghost state drift q={ghost_q_max:.3e}, qd={ghost_qd_max:.3e}")
        record_optional_rl_phase(
            "runtime_contract",
            "complete",
            num_envs=num_envs,
            ghost_limit_abs_max=ghost_limit_max,
            ghost_joint_pos_abs_max=ghost_q_max,
            ghost_joint_vel_abs_max=ghost_qd_max,
        )
        print(
            {
                "asset_count": len(HETEROGENEOUS_CANONICAL_ARTIFACTS),
                "replicas": args.replicas,
                "num_envs": num_envs,
                "policy_steps": args.steps,
                "environment_steps_per_second": num_envs * args.steps / step_seconds,
                "first_asset_id": HETEROGENEOUS_CANONICAL_ARTIFACTS[0].asset_id,
                "last_asset_id": HETEROGENEOUS_CANONICAL_ARTIFACTS[-1].asset_id,
            },
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    # Isaac/Kit 的退出钩子可能在未处理异常后以 code 0 终止解释器；父 recorder 必须收到真实失败码。
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
