r"""TODO: MLP PPO 训练单资产母体的入口。

本文件服务于 `tasks/gm/single_asset_env_cfg.py` 定义的 single-asset MDP probe。
它不是最终层次通才 teacher 训练入口，也不承担 transformer / token geometry /
teacher-student distillation。当前阶段只用 MLP PPO 回答一个问题：

$$
\text{fixed pre-made hand asset} + \text{current GM MDP}
\Longrightarrow \text{can learn in-hand reorientation?}
$$

若答案是否定的，优先排查 MDP / reset / reward / action / asset physics；若答案是肯定的，
再把 same-topology post-mutate variants 与 transformer teacher 加回来。

== 当前已确认的训练边界 ==

- 训练对象：`right_t4_i4_m4_r4` pre-made topology mother asset；不是 post-mutate bank。
- 网络：rl_games 内置 `actor_critic` MLP；不使用 `anymani_gm_transformer`。
- 环境：新 single-asset env cfg，但复用 `tasks/gm/mdp` 现有 term。
- reset：第一轮沿用随机 no-cache scaffold，不先做 fixed grasp / grasp cache。
- command：第一轮收窄为 fixed `{h}` z 轴 + episode 目标，贴近 LEAP 官方 z-axis 成功基线；
  random axis + subgoal resample 留给单资产闭环跑通后的下一轮难度恢复。
- 观测：按当前 MDP scaffold，不引入 `distill/models` 的 PALM / JOINT / TIP geometry tokenizer。

== 推荐执行顺序 ==

1. random smoke：验证 env 构造、reset、step、reward/done shape；
2. one-epoch smoke：验证 rl_games rollout / backward / optimizer / checkpoint 闭环；
3. short train：几十到几百 epoch 看 reward、success、fall rate 是否出现非随机趋势；
4. failure triage：若无趋势，再按 reset → reward scale → action scale → contact sensor → command frame 排查。

== 第一轮不要急着调的内容 ==

- 不加 transformer；
- 不加 dynamic SE(3) attention bias；
- 不加 SSL / aux head；
- 不加 tip BPS / mesh geometry；
- 不做 multi-asset routing；
- 不做 sim2sim / real URDF alignment。

这些都属于后续阶段。若在单资产 MLP 尚未收敛时加入，会让训练失败重新变得不可归因。

== 建议记录的关键日志 ==

除 rl_games 默认 reward / loss 外，单资产 MDP probe 至少应能从 env extras 或 TensorBoard 中看到：

```text
orientation_error       # SO(3) geodesic error，单位 rad
keypoint_error          # AnyRotate 风格六点姿态误差，单位 m
goal_success_count      # 每个 episode 完成的 subgoal 数
axis_progress           # 沿 command axis 累计旋转进度，单位 rad
object_fall_rate        # object_out_of_hand 触发比例
fingertip_contact_count # 有效指尖接触数量
non_tip_contact_rate    # 非指尖接触比例
action_l2 / action_rate # 动作幅值与抖动，用于判断 scale=0.1 是否过激
```

TOAGENT:
    实现阶段可将本文件改为真正的 `argparse + AppLauncher + rl_games Runner` 入口，
    或者改成薄 wrapper 调用现有 `anymani.distill.train_mvp`。无论选择哪种方式，都要保留
    上述训练边界与日志验收语义；它们是本阶段曲线解释的实验 contract。
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
"""AnyMani 仓库根目录；日志路径锚定到这里，避免受 shell 当前工作目录影响。"""

DEFAULT_SINGLE_ASSET_TASK = "AnyMani-GM-SingleAsset-MLP-v0"
"""单资产 MLP PPO 默认 Gym task id；由 `anymani.distill.rl` 注册。"""

DEFAULT_SINGLE_ASSET_NUM_ENVS = 2048
"""默认并行环境数；4070Ti 16GB 上可通过 `--num_envs 4096` 放大。"""


parser = argparse.ArgumentParser(description="Train the AnyMani single-asset GM MDP probe with rl_games MLP PPO.")
parser.add_argument("--task", type=str, default=DEFAULT_SINGLE_ASSET_TASK, help="Single-asset Gym task id registered by distill.")
parser.add_argument("--num_envs", type=int, default=DEFAULT_SINGLE_ASSET_NUM_ENVS, help="Override env parallel count.")
parser.add_argument("--seed", type=int, default=None, help="Training seed; -1 samples a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override rl_games max_epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for resume.")
parser.add_argument("--sigma", type=float, default=None, help="Optional fixed policy sigma override.")
parser.add_argument("--experiment_name", type=str, default=None, help="Optional log subdirectory name.")
parser.add_argument("--rl_games_strict", action="store_true", default=False, help="Require local rl_games commit to match v1.6.5.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# IsaacLab 的 `hydra_task_config` 会二次解析 `sys.argv`。这里保留 Hydra override，移除本入口已消费的 CLI 参数。
sys.argv = [sys.argv[0]] + hydra_args

# 单资产训练仍需要先启动 Isaac Sim，再 import gym/env/rl_games 运行时模块。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import anymani.distill.rl  # noqa: F401, E402  # 注册 `AnyMani-GM-SingleAsset-MLP-v0`
import anymani.tasks.gm  # noqa: F401, E402  # 注册 tasks-owned single-asset smoke aliases
import gymnasium as gym  # noqa: E402
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

backend_info = prefer_local_rl_games(strict=args_cli.rl_games_strict)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


def _fix_minibatch_size(agent_cfg: dict, num_envs: int) -> None:
    r"""保证单资产 smoke / debug 规模下 rl_games batch 与 minibatch 可整除。

    Args:
        agent_cfg (dict): rl_games agent 配置字典。
        num_envs (int): 实际并行环境数。
    """

    cfg = agent_cfg["params"]["config"]  # rl_games PPO 训练超参字典
    horizon_length = int(cfg.get("horizon_length", 1))  # 每个 env 一轮 rollout 的步数
    batch_size = max(1, int(num_envs) * horizon_length)  # PPO batch size = num_envs × horizon_length
    minibatch_size = int(cfg.get("minibatch_size", batch_size))  # YAML 中的目标 minibatch size
    if minibatch_size > batch_size:
        print(f"[WARN] minibatch_size={minibatch_size} > batch_size={batch_size}; using {batch_size}.")
        cfg["minibatch_size"] = batch_size
        return
    if batch_size % minibatch_size != 0:
        fixed = math.gcd(batch_size, minibatch_size) or batch_size  # 最大公约数保证整除且尽量接近原配置
        print(f"[WARN] batch_size={batch_size} is not divisible by minibatch_size={minibatch_size}; using {fixed}.")
        cfg["minibatch_size"] = fixed


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    r"""启动单资产 MLP PPO 训练。

    Args:
        env_cfg: Isaac Lab env cfg，由 `AnyMani-GM-SingleAsset-MLP-v0` 的 registry 加载。
        agent_cfg (dict): rl_games YAML，由 `gm_single_asset_mlp_ppo.yaml` 加载。
    """

    # 单资产 probe 默认高并行；CLI 仍可覆盖为 64/128 做 smoke 或 4096 做吞吐训练。
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # seed=-1 表示显式要求随机种子；否则沿用 YAML 或用户传入的固定 seed。
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    # `max_iterations` 只覆盖 rl_games `max_epochs`，用于 1-5 epoch smoke 或短训。
    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading checkpoint: {resume_path}")
    else:
        resume_path = None

    # rl_games 的 device 字段以 agent cfg 为准；num_actors 则与实际 env wrapper 对齐。
    rl_device = agent_cfg["params"]["config"].get("device", args_cli.device)
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["config"]["num_actors"] = env_cfg.scene.num_envs

    # 单资产训练日志与 teacher debug 分开，避免 MDP probe 曲线混入 transformer 试验目录。
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = str(ANYMANI_ROOT / "logs" / "distill" / "rl_games" / config_name)
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
