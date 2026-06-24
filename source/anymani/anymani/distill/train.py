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
import time
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
import torch  # noqa: E402
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
from rl_games.algos_torch import torch_ext  # noqa: E402
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


class RslStyleIsaacAlgoObserver(IsaacAlgoObserver):
    r"""把 rl_games 训练统计追加打印成 RSL-RL 风格 summary block。

    rl_games 默认 console 主要输出：

    ```text
    fps step: ... fps total: ... epoch: k/K frames: ...
    ```

    这对吞吐很有用，但对科研排查不够直观；RSL-RL 风格 block 会把 value loss、
    surrogate loss、entropy、mean reward、episode length、action std 和 IsaacLab
    episode extras 放在同一个可扫读面板里。这里不改变 TensorBoard scalar，也不改变
    PPO 更新，只在 `after_print_stats()` 阶段补充人类可读 console。
    """

    def after_init(self, algo) -> None:
        r"""在 rl_games algorithm 初始化后 hook `write_stats()` 缓存训练标量。

        Args:
            algo: rl_games algorithm 实例；接口由 rl_games runtime 提供。
        """

        super().after_init(algo)
        self._last_train_stats: dict[str, float | int | None] | None = None  # 最近一次 PPO epoch 的训练标量
        self._last_print_key: tuple[int, int] | None = None  # 防止同一 epoch/frame 重复打印 RSL block
        original_write_stats = algo.write_stats  # rl_games 原始 TensorBoard/stat writer

        def write_stats_with_cache(
            total_time,
            epoch_num,
            step_time,
            play_time,
            update_time,
            a_losses,
            c_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            frame,
            scaled_time,
            scaled_play_time,
            curr_frames,
        ):
            r"""缓存 rl_games 传给 writer 的训练标量，再调用原始 writer。"""

            self._last_train_stats = {
                "total_time": total_time,
                "epoch_num": epoch_num,
                "step_time": step_time,
                "play_time": play_time,
                "update_time": update_time,
                "a_loss": self._mean_list(a_losses),
                "c_loss": self._mean_list(c_losses),
                "entropy": self._mean_list(entropies),
                "kl": self._mean_list(kls),
                "last_lr": last_lr,
                "lr_mul": lr_mul,
                "frame": frame,
                "scaled_time": scaled_time,
                "scaled_play_time": scaled_play_time,
                "curr_frames": curr_frames,
            }
            return original_write_stats(
                total_time,
                epoch_num,
                step_time,
                play_time,
                update_time,
                a_losses,
                c_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
                frame,
                scaled_time,
                scaled_play_time,
                curr_frames,
            )

        algo.write_stats = write_stats_with_cache  # 只包一层统计缓存，不改变 rl_games 算法状态

    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        r"""在 rl_games 默认 stats 后追加 RSL-RL 风格 block。"""

        print_key = (int(epoch_num), int(frame))  # 一个 epoch/frame 只打印一次，避免 observer 重入重复输出
        if self._last_print_key == print_key:
            return
        self._last_print_key = print_key

        episode_summary = self._episode_summary()  # IsaacLab extras，如 reward term / success / fall rate 等
        super().after_print_stats(frame, epoch_num, total_time)

        if self._last_train_stats is None:
            return
        self._print_rsl_style_summary(self._last_train_stats, episode_summary)

    def _episode_summary(self) -> dict[str, float]:
        r"""聚合 IsaacLab episode extras，得到每个 key 的均值。

        Returns:
            dict[str, float]: episode extras 的均值；若尚未有 episode 结束则为空字典。
        """

        summary: dict[str, float] = {}
        if not self.ep_infos:
            return summary
        for key in self.ep_infos[0]:
            values = []
            for ep_info in self.ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor([value], device=self.algo.device, dtype=torch.float32)
                if value.ndim == 0:
                    value = value.unsqueeze(0)  # scalar -> `[1]`，便于和 batch extras 拼接
                values.append(value.to(self.algo.device, dtype=torch.float32).flatten())
            if values:
                summary[key] = torch.cat(values).mean().item()  # extras 均值，单位由 env term 自身定义
        return summary

    @staticmethod
    def _mean_list(values) -> float | None:
        r"""把 rl_games loss list 聚合成 Python float。"""

        if not values:
            return None
        return torch_ext.mean_list(values).item()

    @staticmethod
    def _scalar(value) -> float | None:
        r"""把 rl_games / torch scalar 统一转成 Python float。"""

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean().item()
        return float(value)

    def _mean_reward(self) -> float | None:
        r"""读取 rl_games 维护的 episode reward 均值。"""

        if self.algo.game_rewards.current_size == 0:
            return None
        return self._scalar(self.algo.game_rewards.get_mean()[0])

    def _mean_episode_length(self) -> float | None:
        r"""读取 rl_games 维护的 episode length 均值。"""

        if self.algo.game_lengths.current_size == 0:
            return None
        return self._scalar(self.algo.game_lengths.get_mean())

    def _mean_action_std(self) -> float | None:
        r"""估计当前策略 action std，帮助判断探索是否过强或过弱。"""

        model = getattr(self.algo, "model", None)
        network = getattr(model, "a2c_network", None)
        sigma = getattr(network, "sigma", None)
        if sigma is None:
            return None
        with torch.no_grad():
            sigma_value = sigma.detach().float()
            if "LogStd" in type(model).__qualname__:
                sigma_value = torch.exp(sigma_value)  # logstd parameter -> std
            return sigma_value.mean().item()

    def _line(self, label: str, value: str, pad: int = 40) -> str:
        r"""生成 RSL-RL block 中右对齐的一行文本。"""

        return f"{label + ':':>{pad}} {value}\n"

    def _print_rsl_style_summary(self, stats: dict[str, float | int | None], episode_summary: dict[str, float], width: int = 80) -> None:
        r"""打印单个 PPO epoch 的 RSL-RL 风格摘要。"""

        epoch_num = int(stats["epoch_num"])
        max_epochs = int(getattr(self.algo, "max_epochs", -1))
        done_epochs = max(epoch_num, 1)
        remaining_epochs = max(max_epochs - epoch_num, 0) if max_epochs > 0 else 0
        total_time = float(stats["total_time"] or 0.0)
        eta = total_time / done_epochs * remaining_epochs if remaining_epochs > 0 else 0.0
        total_fps = float(stats["curr_frames"] or 0.0) / max(float(stats["scaled_time"] or 0.0), 1.0e-9)

        log_string = f"{'#' * width}\n"
        log_string += f"{f' Learning iteration {epoch_num}/{max_epochs} '.center(width)}\n\n"
        log_string += self._line("Total steps", f"{int(stats['frame'] or 0)}")
        log_string += self._line("Steps per second", f"{total_fps:.0f}")
        log_string += self._line("Collection time", f"{float(stats['scaled_play_time'] or 0.0):.3f}s")
        log_string += self._line("Learning time", f"{float(stats['update_time'] or 0.0):.3f}s")
        if stats["c_loss"] is not None:
            log_string += self._line("Mean value loss", f"{float(stats['c_loss']):.4f}")
        if stats["a_loss"] is not None:
            log_string += self._line("Mean surrogate loss", f"{float(stats['a_loss']):.4f}")
        if stats["entropy"] is not None:
            log_string += self._line("Mean entropy loss", f"{float(stats['entropy']):.4f}")
        mean_reward = self._mean_reward()
        if mean_reward is not None:
            log_string += self._line("Mean reward", f"{mean_reward:.2f}")
        mean_episode_length = self._mean_episode_length()
        if mean_episode_length is not None:
            log_string += self._line("Mean episode length", f"{mean_episode_length:.2f}")
        mean_action_std = self._mean_action_std()
        if mean_action_std is not None:
            log_string += self._line("Mean action std", f"{mean_action_std:.2f}")
        for key, value in episode_summary.items():
            log_string += self._line(key, f"{value:.4f}")
        log_string += f"{'-' * width}\n"
        log_string += self._line("Iteration time", f"{float(stats['scaled_time'] or 0.0):.2f}s")
        log_string += self._line("Time elapsed", time.strftime("%H:%M:%S", time.gmtime(total_time)))
        log_string += self._line("ETA", time.strftime("%H:%M:%S", time.gmtime(eta)))
        print(log_string)


def _make_isaac_algo_observer(agent_cfg: dict) -> IsaacAlgoObserver:
    r"""按 YAML 配置创建 rl_games observer。"""

    if agent_cfg["params"]["config"].get("rsl_style_console", False):
        return RslStyleIsaacAlgoObserver()
    return IsaacAlgoObserver()


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

    runner = Runner(_make_isaac_algo_observer(agent_cfg))
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
