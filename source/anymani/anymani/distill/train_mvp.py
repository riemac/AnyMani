r"""Self-contained MVP training entry for AnyMani distill.

本入口是层次通才策略训练的 MVP 入口，不再把 GM teacher 训练逻辑散落到项目根部
`scripts/rl_games/`。职责边界：

- `tasks/gm` 定义环境 MDP；
- `distill/rl` 注册 GM teacher task、固定 rl_games backend、注册 Transformer 网络；
- 本文件负责启动 Isaac Sim、创建 env、包装 rl_games、运行 PPO。

运行示例：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.train_mvp --task AnyMani-GM-Teacher-Debug-v0 --num_envs 16 --headless

# 异构手 MLP 可行性 smoke：3 个 hand variants × 每种 100 envs，只验证训练闭环可运行。
python -m anymani.distill.train_mvp \
  --task AnyMani-GM-Heterogeneous-MLP-Smoke-v0 \
  --num_envs 300 \
  --max_iterations 1 \
  --headless
```
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train AnyMani distill policies with rl_games.")
parser.add_argument("--task", type=str, default="AnyMani-GM-Teacher-Debug-v0", help="Gym task id registered by distill.")
parser.add_argument("--num_envs", type=int, default=None, help="Override env parallel count.")
parser.add_argument("--seed", type=int, default=None, help="Training seed; -1 samples a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override rl_games max_epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for resume.")
parser.add_argument("--sigma", type=float, default=None, help="Optional fixed policy sigma override.")
parser.add_argument("--experiment_name", type=str, default=None, help="Optional log subdirectory name.")
parser.add_argument("--rl_games_strict", action="store_true", default=False, help="Require local rl_games commit to match v1.6.5.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# IsaacLab 的 `hydra_task_config` 会在 `main()` 阶段再次解析 `sys.argv`。
# 这里必须只把真正的 Hydra overrides 留给它；否则 `--task`、`--num_envs`、`--headless`
# 这类已经由 argparse 消费的训练入口参数会被 Hydra 当作未知参数并中断训练。
sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import anymani.distill.rl  # noqa: F401  # 注册 distill 自包含 GM teacher tasks
import gymnasium as gym
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

backend_info = prefer_local_rl_games(strict=args_cli.rl_games_strict)

import isaaclab_tasks  # noqa: F401
from anymani.distill.rl.rl_games_networks import register_anymani_rl_games_networks
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

register_anymani_rl_games_networks()


def _fix_minibatch_size(agent_cfg: dict, num_envs: int) -> None:
    r"""保证小规模 smoke 时 rl_games batch/minibatch 可整除。

    Args:
        agent_cfg (dict): rl_games agent 配置。
        num_envs (int): 实际并行 env 数。
    """

    cfg = agent_cfg["params"]["config"]
    horizon_length = int(cfg.get("horizon_length", 1))
    batch_size = max(1, int(num_envs) * horizon_length)
    minibatch_size = int(cfg.get("minibatch_size", batch_size))
    if minibatch_size > batch_size:
        print(f"[WARN] minibatch_size={minibatch_size} > batch_size={batch_size}; using {batch_size}.")
        cfg["minibatch_size"] = batch_size
        return
    if batch_size % minibatch_size != 0:
        fixed = math.gcd(batch_size, minibatch_size) or batch_size
        print(f"[WARN] batch_size={batch_size} is not divisible by minibatch_size={minibatch_size}; using {fixed}.")
        cfg["minibatch_size"] = fixed


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    r"""Train GM teacher with distill-owned rl_games pipeline.

    Args:
        env_cfg: Isaac Lab env cfg loaded from distill gym registry。
        agent_cfg (dict): rl_games YAML loaded from `distill/rl/agents`。
    """

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading checkpoint: {resume_path}")
    else:
        resume_path = None

    rl_device = agent_cfg["params"]["config"].get("device", args_cli.device)
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["config"]["num_actors"] = env_cfg.scene.num_envs

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.abspath(os.path.join("logs", "distill", "rl_games", config_name))
    log_dir = args_cli.experiment_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    agent_cfg["params"]["config"]["rl_games_backend_file"] = str(backend_info.package_file)
    agent_cfg["params"]["config"]["rl_games_backend_commit"] = backend_info.git_commit

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    _fix_minibatch_size(agent_cfg, env.unwrapped.num_envs)

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner_args = {"train": True, "play": False, "sigma": args_cli.sigma}
    if resume_path is not None:
        runner_args["checkpoint"] = resume_path
    runner.run(runner_args)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
