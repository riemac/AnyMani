r"""AnyMani distill 的统一 RL checkpoint 回放入口。

本入口和 `anymani.distill.train` 成对存在：`train.py` 负责 rl_games PPO 训练，
`play.py` 负责加载训练 checkpoint，在同一个 tasks-owned env contract 中做可视化回放。
它不新增 play-only Gym alias，也不复制 env cfg；回放仍使用
`AnyMani-GM-SingleAsset-MLP-v0` 这一训练 task，确保 obs/action schema 与 checkpoint
完全一致。

推荐 GUI 回放命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.play \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 1 \
  --checkpoint /path/to/checkpoint.pth \
  --real-time
```
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

from anymani.assets.bank.path_utils import resolve_anymani_root
from isaaclab.app import AppLauncher

ANYMANI_ROOT = resolve_anymani_root()
"""AnyMani 仓库根目录；回放 checkpoint 搜索从这里的 `logs/distill/rl_games` 开始。"""

DEFAULT_TASK = "AnyMani-GM-SingleAsset-MLP-v0"
"""当前 single-asset MLP policy 的 distill Gym task id。"""

DEFAULT_NUM_ENVS = 1
"""默认只开 1 个 env 做人工 GUI 观察，避免多 env 视觉重叠干扰行为判断。"""


parser = argparse.ArgumentParser(description="Play AnyMani distill rl_games checkpoints.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task id registered by anymani.distill.rl.")
parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of envs for playback.")
parser.add_argument("--seed", type=int, default=None, help="Playback seed; -1 samples a random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path. This is the safest mode.")
parser.add_argument("--run_name", type=str, default=None, help="Run directory under AnyMani logs when checkpoint is omitted.")
parser.add_argument("--use_last_checkpoint", action="store_true", default=True, help="Use the latest checkpoint in a run.")
parser.add_argument("--use_best_checkpoint", action="store_true", default=False, help="Use `<agent_name>.pth` instead of latest.")
parser.add_argument("--rl_games_strict", action="store_true", default=False, help="Require local rl_games v1.6.5 commit.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=1000, help="Recorded video length in policy steps.")
parser.add_argument("--steps", type=int, default=None, help="Optional max playback steps before auto-exit.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run at env step real time if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录像需要 camera pipeline；GUI 人工回放不强制打开 video recorder。
if args_cli.video:
    args_cli.enable_cameras = True  # AppLauncher 参数必须在 app 启动前设置

# `hydra_task_config` 会读取 sys.argv 中剩余的 Hydra override；这里剥离本入口自己的 CLI。
sys.argv = [sys.argv[0]] + hydra_args

# 所有 Isaac Sim / rl_games runtime import 都应在 AppLauncher 之后发生。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import anymani.distill.rl  # noqa: F401, E402  # 注册 distill-owned Gym aliases
import anymani.tasks.gm  # noqa: F401, E402  # 显式注册 tasks-owned GM env aliases
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

# 必须在 import `rl_games.*` 前固定本地源码路径，确保 train/play 使用同一 backend。
backend_info = prefer_local_rl_games(strict=args_cli.rl_games_strict)

from anymani.distill.rl.rl_games_networks import register_anymani_rl_games_networks  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

register_anymani_rl_games_networks()
"""注册未来 Transformer teacher adapter；当前 single-asset MLP 回放不会使用该网络。"""


def _resolve_seed(agent_cfg: dict) -> int:
    r"""解析回放 seed，并保持 env / agent seed 一致。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。

    Returns:
        int: 最终用于 env reset 的 seed。
    """

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)  # 显式 `-1` 表示本次回放随机种子
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = int(args_cli.seed)  # CLI seed 覆盖 YAML seed
    return int(agent_cfg["params"]["seed"])


def _distill_log_root(agent_cfg: dict) -> Path:
    r"""返回当前 agent 的 AnyMani distill rl_games 日志根目录。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。

    Returns:
        Path: `AnyMani/logs/distill/rl_games/<config_name>`。
    """

    config_name = agent_cfg["params"]["config"]["name"]  # 例如 `gm_single_asset_mlp`
    return ANYMANI_ROOT / "logs" / "distill" / "rl_games" / config_name


def _resolve_checkpoint(agent_cfg: dict) -> str:
    r"""解析回放 checkpoint 路径。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。

    Returns:
        str: 可传给 `agent.restore(...)` 的 checkpoint 路径。
    """

    if args_cli.checkpoint is not None:
        return retrieve_file_path(args_cli.checkpoint)  # 显式 checkpoint 最可复现，优先级最高

    log_root_path = _distill_log_root(agent_cfg)
    run_dir = args_cli.run_name or agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth" if args_cli.use_best_checkpoint else ".*"
    print(f"[INFO] Searching checkpoint under: {log_root_path / str(run_dir)}")
    return get_checkpoint_path(str(log_root_path), run_dir, checkpoint_file, other_dirs=["nn"])


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    r"""加载 rl_games checkpoint 并执行可视化回放。

    Args:
        env_cfg: Isaac Lab env cfg，由 distill Gym registry 加载。
        agent_cfg (dict): rl_games YAML，由 distill Gym registry 加载。
    """

    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = _resolve_seed(agent_cfg)  # env reset 随机性必须在构造前固定

    resume_path = _resolve_checkpoint(agent_cfg)
    log_dir = os.path.dirname(os.path.dirname(resume_path))  # `<run>/nn/<ckpt>.pth -> <run>`
    env_cfg.log_dir = log_dir
    print(f"[INFO] Using rl_games from: {backend_info.package_file}")
    print(f"[INFO] Loading checkpoint: {resume_path}")

    rl_device = agent_cfg["params"]["config"].get("device", args_cli.device)  # policy inference device
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # rl_games player 当前只消费 single-agent env

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]  # RlGamesVecEnvWrapper 的 grouped obs contract
    _ = agent.get_batch_size(obs, 1)  # 告诉 rl_games 当前 obs 是 batched rollout
    if agent.is_rnn:
        agent.init_rnn()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        timestep += 1
        if args_cli.video and timestep >= int(args_cli.video_length):
            break
        if args_cli.steps is not None and timestep >= int(args_cli.steps):
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
