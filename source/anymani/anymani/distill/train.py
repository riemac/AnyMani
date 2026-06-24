r"""AnyMani distill 的统一 RL 训练入口。

本入口当前服务“单资产 MLP MDP probe”这一条正式训练主线。它不定义环境语义：
`tasks/gm/single_asset_env_cfg.py` 仍然拥有 scene、obs、action、reward、reset、
termination；这里仅负责启动 Isaac Sim、加载 distill registry 中的 rl_games YAML、
固定本地 rl_games backend、写出训练复现实验目录并运行 PPO。

当前科研问题是：

$$
\text{fixed generated hand asset} + \text{current GM MDP}
\Longrightarrow \text{can learn in-hand reorientation?}
$$

如果该问题尚未回答，`distill` 不应再扩散出 teacher debug、heterogeneous MVP、
play alias 或临时 smoke 训练入口。旧路线已经出清，避免读者误以为它们仍是当前
可选建模方案。

运行示例：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.train \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 4096 \
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

from anymani.assets.bank.path_utils import resolve_anymani_root
from isaaclab.app import AppLauncher

ANYMANI_ROOT = resolve_anymani_root()
"""AnyMani 仓库根目录；训练日志锚定到这里，避免受 shell 当前工作目录影响。"""

DEFAULT_TASK = "AnyMani-GM-SingleAsset-MLP-v0"
"""当前正式 RL 训练路线：single-asset GM MDP probe 的 distill Gym task id。"""

DEFAULT_NUM_ENVS = 4096
"""默认并行环境数；5070Ti 16GB 上优先使用 4096 env，可通过 CLI 降到 2048/1024 做排查。"""


parser = argparse.ArgumentParser(description="Train AnyMani distill RL policies with rl_games.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task id registered by anymani.distill.rl.")
parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Override env parallel count.")
parser.add_argument("--seed", type=int, default=None, help="Training seed; -1 samples a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override rl_games max_epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for resume.")
parser.add_argument("--sigma", type=float, default=None, help="Optional fixed policy sigma override.")
parser.add_argument("--experiment_name", type=str, default=None, help="Optional run directory name under train_dir.")
parser.add_argument("--rl_games_strict", action="store_true", default=False, help="Require local rl_games v1.6.5 commit.")
parser.add_argument("--video", action="store_true", default=False, help="Record RGB rollout snippets during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in policy steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Step interval between recorded videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录像需要 Isaac Sim camera pipeline；不录像时保持 headless 训练路径最轻。
if args_cli.video:
    args_cli.enable_cameras = True  # IsaacLab AppLauncher 边界参数，必须在启动 app 前设置

# IsaacLab 的 `hydra_task_config` 会在 `main()` 阶段再次解析 `sys.argv`。
# 这里仅把 Hydra override 留给它，避免 `--task`、`--num_envs` 等 argparse 参数被重复解析。
sys.argv = [sys.argv[0]] + hydra_args

# 所有 Isaac Sim / USD / PhysX 相关 import 都应发生在 AppLauncher 之后。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import anymani.distill.rl  # noqa: F401, E402  # 注册当前 distill-owned training aliases
import anymani.tasks.gm  # noqa: F401, E402  # 显式注册 tasks-owned env aliases，便于 Hydra 解析 env cfg
import gymnasium as gym  # noqa: E402
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

# 固定本地 rl_games 源码优先级；后续 `rl_games.*` import 均应来自该路径。
backend_info = prefer_local_rl_games(strict=args_cli.rl_games_strict)

import isaaclab_tasks  # noqa: F401, E402
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
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

register_anymani_rl_games_networks()
"""注册未来 Transformer teacher adapter；当前 single-asset MLP YAML 不会使用该网络。"""


def _fix_minibatch_size(agent_cfg: dict, num_envs: int) -> None:
    r"""保证小规模 smoke / debug 训练时 PPO batch 与 minibatch 可整除。

    rl_games 内部 batch size 为：

    $$
    B = N_{\text{env}} \times H,
    $$

    其中 $N_{\text{env}}$ 是实际并行环境数，$H$ 是 `horizon_length`。正式训练用
    4096 env 时 YAML 已经整除；但我们经常用 2/8/64 env 做 smoke，此时必须在
    `runner.load()` 前把 `minibatch_size` 修正到合法值，否则训练入口会在第一轮
    rollout 后因 shape contract 失败。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。
        num_envs (int): 实际 env 数，即 wrapper 后的 `env.unwrapped.num_envs`。
    """

    cfg = agent_cfg["params"]["config"]  # rl_games PPO 训练超参字典
    horizon_length = int(cfg.get("horizon_length", 1))  # 每个 env 一轮 rollout 的步数 $H$
    batch_size = max(1, int(num_envs) * horizon_length)  # PPO batch size $B=N_{\text{env}}H$
    minibatch_size = int(cfg.get("minibatch_size", batch_size))  # YAML 目标 minibatch size
    if minibatch_size > batch_size:
        print(f"[WARN] minibatch_size={minibatch_size} > batch_size={batch_size}; using {batch_size}.")
        cfg["minibatch_size"] = batch_size  # 最小 smoke 时一个 batch 只做一个 minibatch
        return
    if batch_size % minibatch_size != 0:
        fixed = math.gcd(batch_size, minibatch_size) or batch_size  # gcd 保证整除且尽量接近原配置
        print(f"[WARN] batch_size={batch_size} is not divisible by minibatch_size={minibatch_size}; using {fixed}.")
        cfg["minibatch_size"] = fixed


def _resolve_seed(agent_cfg: dict) -> int:
    r"""解析训练 seed，并把 CLI 覆盖写回 rl_games agent cfg。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。

    Returns:
        int: 最终用于 env 构造和 agent 初始化的 seed。
    """

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)  # 显式 `-1` 表示研究阶段要求随机 seed
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = int(args_cli.seed)  # CLI seed 优先于 YAML seed
    return int(agent_cfg["params"]["seed"])  # env_cfg.seed 与 agent seed 必须一致


def _configure_log_dir(agent_cfg: dict) -> tuple[str, str]:
    r"""配置 AnyMani-root anchored rl_games 日志目录。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。

    Returns:
        tuple[str, str]: `(log_root_path, log_dir)`，分别对应 train_dir 与 run name。
    """

    config_name = agent_cfg["params"]["config"]["name"]  # YAML 中的实验族名，例如 `gm_single_asset_mlp`
    log_root_path = str(ANYMANI_ROOT / "logs" / "distill" / "rl_games" / config_name)
    log_dir = args_cli.experiment_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root_path  # rl_games TensorBoard / checkpoint 根目录
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir  # 单次 run 目录名
    agent_cfg["params"]["config"]["rl_games_backend_file"] = str(backend_info.package_file)  # backend 复现信息
    agent_cfg["params"]["config"]["rl_games_backend_commit"] = backend_info.git_commit  # backend 复现信息
    print(f"[INFO] Logging experiment in directory: {os.path.join(log_root_path, log_dir)}")
    return log_root_path, log_dir


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    r"""启动 AnyMani distill RL 训练。

    Args:
        env_cfg: Isaac Lab env cfg，由 Gym registry 的 `env_cfg_entry_point` 加载。
        agent_cfg (dict): rl_games YAML，由 Gym registry 的 `rl_games_cfg_entry_point` 加载。
    """

    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = _resolve_seed(agent_cfg)  # 某些 reset 随机化发生在 env 构造期，必须提前写入

    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = int(args_cli.max_iterations)  # 短训 / smoke 覆盖 epoch 数
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)  # 支持 omniverse / local 路径解析
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading checkpoint: {resume_path}")
    else:
        resume_path = None

    rl_device = agent_cfg["params"]["config"].get("device", args_cli.device)  # rl_games 训练 device
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["config"]["num_actors"] = env_cfg.scene.num_envs

    log_root_path, log_dir = _configure_log_dir(agent_cfg)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)  # IsaacLab env extras / IO descriptor 的日志锚点
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # rl_games PPO 当前只消费 single-agent env

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % int(args_cli.video_interval) == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)  # obs clamp，上游 YAML 控制
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)  # action clamp，上游 YAML 控制
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
