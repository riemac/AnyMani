r"""Self-contained MVP smoke test entry for AnyMani distill RL tasks.

本入口用于验证 `distill` 注册的 GM teacher 环境能完成 reset / step。它替代外层
`scripts/random_agent.py` 在新训练管线中的角色，避免训练、测试脚本继续散落在项目根部。

运行示例：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.smoke_mvp --task AnyMani-GM-Teacher-Debug-v0 --num_envs 4 --steps 100 --headless
```
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random-action smoke test for AnyMani distill tasks.")
parser.add_argument("--task", type=str, default="AnyMani-GM-Teacher-Debug-v0", help="Distill-registered Gym task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Override env parallel count.")
parser.add_argument("--steps", type=int, default=100, help="Number of random-action steps to execute.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric for debugging USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import anymani.distill.rl  # noqa: F401  # 注册 distill 自包含 task
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    r"""执行有限步随机动作 smoke。

    该测试只声明“环境能构造、reset、step，且动作/观测空间维度闭合”。它不声明
    训练已经收敛，也不替代 viewer 中对手-物初始姿态的人工检查。
    """

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] observation_space={env.observation_space}")
    print(f"[INFO] action_space={env.action_space}")

    obs, _ = env.reset()
    print(f"[INFO] reset obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    for step in range(int(args_cli.steps)):
        with torch.inference_mode():
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)
            if step == 0 or (step + 1) == int(args_cli.steps):
                print(
                    f"[INFO] step={step + 1} reward_mean={reward.mean().item():.6f} "
                    f"terminated={terminated.float().mean().item():.3f} truncated={truncated.float().mean().item():.3f}"
                )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
