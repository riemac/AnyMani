"""Play (visualize) a student checkpoint trained with online distillation.

This is a project-side utility script under AnyRotate to replay *student* policies
trained via the teacher-student distillation pipeline.

Key features:
- Uses IsaacLab's AppLauncher (same as training scripts).
- Registers custom rl_games networks (e.g. se3_tcn).
- Loads the student task config from the Gym registry and replays a given checkpoint.
- Optionally records a short video.

Typical usage:
  isaaclab.sh -p AnyRotate/source/leaphand/leaphand/distillation/play_distill.py \
    --task Template-Leaphand-se3-Tactile-Manager-v0 \
    --checkpoint <path/to/nn/last_*.pth> \
    --num-envs 1

Notes:
- If the student network consumes observation history (e.g. 50-step history), the
  first ~N steps after reset can look odd because the history buffer is being
  filled.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _default_video_dir(checkpoint_path: str) -> str:
    ckpt = Path(checkpoint_path)
    # expected layout: <experiment_dir>/nn/<checkpoint>.pth
    if ckpt.parent.name == "nn" and ckpt.parent.parent.exists():
        return str(ckpt.parent.parent / "videos" / "play")

    # fallback to a deterministic project folder
    project_root = Path(__file__).resolve().parents[4]
    return str(project_root / "logs" / "rl_games" / "videos" / "play")


parser = argparse.ArgumentParser(description="Play a student checkpoint trained with distillation (rl_games).")
parser.add_argument("--task", type=str, required=True, help="Gym task id for the student env.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student rl_games .pth checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num-envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record a short video.")
parser.add_argument("--video-length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# Append AppLauncher cli args (includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _run(env_cfg, agent_cfg):
    import gymnasium as gym
    import torch

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    import leaphand.tasks.manager_based.leaphand  # noqa: F401

    # Register custom rl_games networks used by this project (e.g. se3_tcn).
    import leaphand.rl_games.se3_tcn_network  # noqa: F401

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the log directory for the environment (best-effort)
    try:
        ckpt_path = retrieve_file_path(args_cli.checkpoint)
        experiment_dir = str(Path(ckpt_path).resolve().parent.parent)
        env_cfg.log_dir = experiment_dir
    except Exception:  # noqa: BLE001
        pass

    # rl-games env params
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"].get("env", {}).get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"].get("env", {}).get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_dir = _default_video_dir(args_cli.checkpoint)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    resume_path = retrieve_file_path(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    import isaaclab_tasks  # noqa: F401
    import leaphand.tasks.manager_based.leaphand  # noqa: F401

    from isaaclab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        _run(env_cfg, agent_cfg)

    main()
    simulation_app.close()
