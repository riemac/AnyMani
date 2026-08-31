r"""Standalone验证two-asset N040 History30 environment ABI。

该入口采用项目benchmark相同的AppLauncher→task import→gym.make生命周期，并以最终JSON作为唯一通过信号。
它不依赖pytest对Kit进程退出码的解释。
"""

from __future__ import annotations

import json
import os
import traceback
from typing import cast


def main() -> int:
    r"""构造2-asset scene，验证reset History30与一步buffer推进。"""

    os.environ["ANYMANI_HETEROGENEOUS_ASSET_LIMIT"] = "2"
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import anymani.tasks.gm  # noqa: F401  # 注册tasks-ownedHistory30 variant
        import gymnasium as gym
        import torch
        from anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg import (
            HeterogeneousN040HistoryTactileRotationEnvCfg,
        )
        from isaaclab.envs import ManagerBasedRLEnv

        cfg = HeterogeneousN040HistoryTactileRotationEnvCfg()
        env = gym.make("AnyMani-GM-HeterogeneousAsset-TactileRotation-History30-v0", cfg=cfg)
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        obs, _ = env.reset()
        history = obs["policy"][:, : 30 * 16 * 4].reshape(2, 30, 16, 4)
        if obs["policy"].shape != (2, 1969) or obs["critic"].shape != (2, 127):
            raise RuntimeError(f"unexpected observation shapes policy={obs['policy'].shape} critic={obs['critic'].shape}")
        torch.testing.assert_close(history[:, 0], history[:, -1])
        action_term = runtime_env.action_manager.get_term("hand_joint_pos")
        next_obs, _, _, _, _ = env.step(torch.zeros(2, 16, device=runtime_env.device))
        next_history = next_obs["policy"][:, : 30 * 16 * 4].reshape(2, 30, 16, 4)
        torch.testing.assert_close(next_history[:, :-1], history[:, 1:])
        print(
            json.dumps(
                {
                    "policy_shape": list(obs["policy"].shape),
                    "critic_shape": list(obs["critic"].shape),
                    "action_shape": list(env.action_space.shape),
                    "action_scale_rad": float(action_term.cfg.scale),
                    "history_prefix_repeated": True,
                    "history_step_shift": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 2
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
