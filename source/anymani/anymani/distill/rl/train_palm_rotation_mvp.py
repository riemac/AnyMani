r"""80手掌托旋转MVP的rl_games正式训练入口。

本入口在Isaac config import之前冻结``ppo_mvp80.yaml``与环境数，随后构造：

``ManagerBasedRLEnv -> BF16-N040 structured vec-env -> custom dual-optimizer PPO``。

默认执行residual arm的30M matched pulse：2560 env、horizon 30、391 updates。1280-env fallback
自动把默认updates翻倍以保持相同transition预算。``--smoke``使用80 env、horizon 4、1 update，
只验证完整rollout/buffer/minibatch/backward/checkpoint数据流，不构成学习证据。

运行示例：

```bash
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.rl.train_palm_rotation_mvp \
  --headless --arm residual --num_envs 2560
```
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

from anymani.assets.bank.path_utils import resolve_anymani_root

ANYMANI_ROOT = resolve_anymani_root()
TASK_ID = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"
DEFAULT_MANIFEST = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml")


def _load_mvp80_rows(path: Path) -> tuple[int, ...]:
    r"""读取版本化manifest并返回80个唯一formal rows。

    Args:
        path (Path): 仓库相对或绝对``ppo_mvp80.yaml``路径。

    Returns:
        tuple[int, ...]: 有序80-row axis。
    """

    resolved = path if path.is_absolute() else ANYMANI_ROOT / path  # 不依赖shell cwd
    payload = resolved.read_bytes()  # exact checkpoint provenance bytes
    document = yaml.safe_load(payload)
    if not isinstance(document, dict):
        raise ValueError("MVP80 manifest must contain a YAML mapping")
    rows = tuple(int(row) for row in document.get("selected_rows", ()))  # formal ppo.yaml rows
    if len(rows) != 80 or len(set(rows)) != 80:
        raise ValueError(f"MVP80 training requires exactly 80 unique selected_rows, got {len(rows)}")
    return rows


parser = argparse.ArgumentParser(description="Train the 80-hand palm-rotation MVP with structured rl_games PPO.")
parser.add_argument("--asset_manifest", type=Path, default=DEFAULT_MANIFEST, help="Versioned 80-row selection manifest.")
parser.add_argument("--num_envs", type=int, default=None, help="2560 formal, 1280 fallback, or 80 with --smoke.")
parser.add_argument("--arm", choices=("base", "residual"), default="residual", help="Matched actor arm.")
parser.add_argument("--seed", type=int, default=42, help="Formal protocol uses 42, then 43/44 after seed42 passes.")
parser.add_argument("--max_updates", type=int, default=None, help="Override PPO updates; default preserves 30M pulse budget.")
parser.add_argument("--minibatches", type=int, default=None, help="Formal default 16; every minibatch remains asset-balanced.")
parser.add_argument("--checkpoint", type=str, default=None, help="Full actor/critic/optimizers/curriculum checkpoint.")
parser.add_argument("--experiment_name", type=str, default=None, help="Run name under logs/distill/rl_games.")
parser.add_argument("--sigma", type=float, default=None, help="Optional rl_games play-time sigma override.")
parser.add_argument("--smoke", action="store_true", help="Use 80 env, horizon 4, one mini-epoch/update integration mode.")
parser.add_argument("--rl_games_strict", action="store_true", help="Require pinned local rl_games v1.6.5 commit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, launcher_unknown_args = parser.parse_known_args()

# Static generated scene、mask、pregrasp routing在config import时构造，故必须先设置完整80-row/env axis。
selected_rows = _load_mvp80_rows(args_cli.asset_manifest)
num_envs = int(args_cli.num_envs if args_cli.num_envs is not None else (80 if args_cli.smoke else 2560))
if args_cli.smoke:
    if num_envs < 80 or num_envs % 80 != 0:
        raise ValueError("smoke num_envs must be a positive multiple of all 80 assets")
elif num_envs not in (1280, 2560):
    raise ValueError("formal MVP permits only 2560 envs or the 1280-env memory fallback")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)  # ordered support axis
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(num_envs)  # static scene/mask/reset/command routing axis

# 专用入口不执行Hydra config round-trip：contact/pregrasp typed contracts含frozen dataclasses，不应降成dict后原位修改。
sys.argv = [sys.argv[0], *launcher_unknown_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

import anymani.distill.rl  # noqa: F401, E402  # 注册MVP rl_games alias
import anymani.tasks.hetero  # noqa: F401, E402  # 注册tasks-owned raw environment alias
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

# 所有rl_games.*模块必须在本地backend pin之后导入，避免site-packages 1.6.1静默替换v1.6.5源码。
backend_info = prefer_local_rl_games(strict=bool(args_cli.rl_games_strict))
torch.backends.cuda.matmul.allow_tf32 = False  # actor/critic FP32不允许静默降为TF32 matmul
torch.backends.cudnn.allow_tf32 = False  # temporal convolution同样维持FP32；N040另有局部BF16 autocast

from rl_games.common import env_configurations, vecenv  # noqa: E402

from anymani.distill.rl.observers import OneShotIsaacAlgoObserver  # noqa: E402
from anymani.distill.rl.palm_rotation_ppo import (  # noqa: E402
    PalmRotationPpoRunner,
    register_palm_rotation_ppo,
)
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
)
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_vecenv import (  # noqa: E402
    PalmRotationRlGamesGpuEnv,
    PalmRotationRlGamesVecEnv,
)
from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (  # noqa: E402
    GOOD_PREGRASP_RESET_CFG,
    GeneratedPalmRotationMvpEnvCfg,
)
from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING  # noqa: E402


def _resolve_seed(agent_cfg: dict[str, Any]) -> int:
    r"""将CLI seed写入agent/env；``-1``仅用于显式非正式随机probe。"""

    seed = random.randint(0, 10000) if int(args_cli.seed) == -1 else int(args_cli.seed)
    agent_cfg["params"]["seed"] = seed  # rl_games model/rollout RNG
    return seed


def _configure_budget(agent_cfg: dict[str, Any]) -> tuple[int, int, int]:
    r"""设置horizon/minibatch/updates并保持每份逐资产等量采样。

    正式actor的History30会把一个environment sample展开为16条joint temporal sequences。四份切分在1280/2560
    env下产生过大的反向activation；正式默认改为16份，使每份仍含全部80 assets但只改变优化microbatch，
    不改变rollout batch、每样本复用次数或global advantage denominator。

    Returns:
        tuple[int, int, int]: ``(horizon, minibatch_size, max_updates)``。
    """

    config = agent_cfg["params"]["config"]  # rl_games PPO config
    horizon = 4 if args_cli.smoke else 30  # smoke覆盖完整buffer但不等待30个physics steps
    batch_size = num_envs * horizon  # $B=N_{env}H$
    minibatch_count = int(args_cli.minibatches if args_cli.minibatches is not None else (4 if args_cli.smoke else 16))
    if minibatch_count < 1 or (batch_size // 80) % minibatch_count != 0:
        raise ValueError("per-asset rollout samples must be divisible into all stratified minibatches")
    minibatch_size = batch_size // minibatch_count  # 每个minibatch始终含全部80 assets
    default_updates = 1 if args_cli.smoke else 391 * (2560 // num_envs)  # matched约30.03M transitions
    max_updates = int(args_cli.max_updates if args_cli.max_updates is not None else default_updates)
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    config["horizon_length"] = horizon
    config["minibatch_size"] = minibatch_size
    config["mini_epochs"] = 1 if args_cli.smoke else 5
    config["max_epochs"] = max_updates
    config["num_actors"] = num_envs
    config["asset_count"] = 80
    config["name"] = f"heterogeneous_palm_rotation_mvp_{args_cli.arm}"
    agent_cfg["params"]["network"]["palm_rotation"]["residual_enabled"] = args_cli.arm == "residual"
    return horizon, minibatch_size, max_updates


def _configure_log_dir(agent_cfg: dict[str, Any], *, checkpoint: str | None) -> tuple[Path, Path]:
    r"""建立新run目录，或从checkpoint严格恢复原run目录。

    Resume不能生成新的timestamp目录，因为Parquet shard inventory与checkpoint共同定义同一条训练轨迹。
    ``<run>/nn/<checkpoint>.pth``是唯一接受的恢复布局；显式experiment name若存在必须与路径一致。
    """

    config = agent_cfg["params"]["config"]
    root = ANYMANI_ROOT / "logs" / "distill" / "rl_games" / str(config["name"])
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()  # exact checkpoint artifact
        run_dir = checkpoint_path.parent.parent  # `<run>/nn/file.pth` -> `<run>`
        if checkpoint_path.parent.name != "nn" or run_dir.parent.resolve() != root.resolve():
            raise ValueError(f"resume checkpoint must belong to the expected arm run root: {root}")
        if args_cli.experiment_name is not None and args_cli.experiment_name != run_dir.name:
            raise ValueError("--experiment_name must match the checkpoint-owned run directory on resume")
        run_name = run_dir.name
    else:
        run_name = args_cli.experiment_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = root / run_name
    config["train_dir"] = str(root)  # rl_games checkpoint/TensorBoard parent
    config["full_experiment_name"] = run_name
    config["rl_games_backend_file"] = str(backend_info.package_file)
    config["rl_games_backend_commit"] = backend_info.git_commit
    run_dir.joinpath("params").mkdir(parents=True, exist_ok=True)
    return root, run_dir


def main() -> None:
    r"""构造exact 80-hand environment、cached-N040 transport和custom PPO Runner。"""

    env_cfg: ManagerBasedRLEnvCfg = GeneratedPalmRotationMvpEnvCfg()  # typed task cfg，不经过dict round-trip
    agent_path = (
        ANYMANI_ROOT
        / "source/anymani/anymani/distill/rl/agents/heterogeneous_palm_rotation_mvp_ppo.yaml"
    )
    agent_cfg = yaml.safe_load(agent_path.read_text(encoding="utf-8"))  # versioned rl_games config
    if not isinstance(agent_cfg, dict):
        raise TypeError("palm-rotation rl_games YAML must contain a mapping")
    seed = _resolve_seed(agent_cfg)  # task随机状态与PPO RNG统一
    horizon, minibatch_size, max_updates = _configure_budget(agent_cfg)
    env_cfg.scene.num_envs = num_envs  # 与pre-import static routing严格相等
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = seed
    rl_device = str(agent_cfg["params"]["config"].get("device", env_cfg.sim.device))
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    checkpoint = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None
    _, run_dir = _configure_log_dir(agent_cfg, checkpoint=checkpoint)
    if checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
    env_cfg.log_dir = str(run_dir)  # task diagnostics与PPO artifact共用run root

    # Environment必须先实例化，N040 canonical evidence使用同一ASSET_BINDING与device。
    env = gym.make(TASK_ID, cfg=env_cfg)
    provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=rl_device)
    prototype_index = torch.tensor(
        ASSET_BINDING.asset_index_by_env(num_envs),
        dtype=torch.long,
        device=rl_device,
    )  # exact round-robin$e\bmod80$
    transport = PalmRotationRlGamesVecEnv(
        env,
        geometry_provider=provider,
        prototype_index=prototype_index,
        rl_device=rl_device,
        clip_observations=float(agent_cfg["params"]["env"]["clip_observations"]),
        clip_actions=float(agent_cfg["params"]["env"]["clip_actions"]),
    )

    # Runtime identity必须在Runner build前注入network，checkpoint restore才能先验证再加载model tensors。
    ppo_cfg = agent_cfg["params"]["config"]  # resolved optimizer/sampling contract
    identity = build_palm_rotation_method_identity(
        provider_identity=provider.identity,
        manifest_path=args_cli.asset_manifest,
        selected_rows=selected_rows,
        pregrasp=GOOD_PREGRASP_RESET_CFG,
        arm=str(args_cli.arm),
        run_contract={
            "seed": seed,
            "num_envs": num_envs,
            "horizon_length": horizon,
            "minibatch_size": minibatch_size,
            "minibatch_count": (num_envs * horizon) // minibatch_size,
            "gradient_accumulation_steps": int(ppo_cfg["gradient_accumulation_steps"]),
            "mini_epochs": int(ppo_cfg["mini_epochs"]),
            "gamma": float(ppo_cfg["gamma"]),
            "gae_lambda": float(ppo_cfg["tau"]),
            "ppo_clip": float(ppo_cfg["e_clip"]),
            "entropy_coef": float(ppo_cfg["entropy_coef"]),
            "grad_norm": float(ppo_cfg["grad_norm"]),
            "actor_base_lr": float(ppo_cfg["learning_rate"]),
            "adaptive_lr_max": float(ppo_cfg["adaptive_lr_max"]),
            "actor_residual_lr": float(ppo_cfg["residual_learning_rate"]),
            "critic_lr": float(ppo_cfg["critic_learning_rate"]),
            "lr_schedule": str(ppo_cfg["lr_schedule"]),
            "normalize_advantage": bool(ppo_cfg["normalize_advantage"]),
            "normalize_value": bool(ppo_cfg["normalize_value"]),
            "initial_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["initial_log_std"]),
            "max_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["max_log_std"]),
            "base_action_limit": float(agent_cfg["params"]["network"]["palm_rotation"]["base_action_limit"]),
            "rl_games_backend_commit": backend_info.git_commit,
            "device": rl_device,
        },
    )
    agent_cfg["params"]["network"]["anymani_identity"] = identity
    agent_cfg["params"]["config"]["num_actors"] = transport.num_envs
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    (run_dir / "params" / "runtime_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 进程内注册custom vec-env、network/model/agent；不修改外部rl_games源码或全局安装。
    vecenv.register(
        "AnyManiPalmRotationWrapper",
        lambda config_name, num_actors, **kwargs: PalmRotationRlGamesGpuEnv(
            config_name,
            num_actors,
            env=transport,
        ),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "AnyManiPalmRotationWrapper", "env_creator": lambda **kwargs: transport},
    )
    register_palm_rotation_ppo()
    runner = PalmRotationPpoRunner(OneShotIsaacAlgoObserver())
    precision_flags = enforce_palm_rotation_precision()  # Runner.__init__会开启TF32，必须在model build前覆盖
    runner.load(agent_cfg)
    runner.reset()
    if enforce_palm_rotation_precision() != precision_flags:
        raise RuntimeError("palm-rotation precision flags changed during Runner model construction")

    # Resume解析后仍由custom agent先核对identity，再恢复actor/critic/optimizers/curriculum。
    runner_args: dict[str, Any] = {"train": True, "play": False, "sigma": args_cli.sigma}
    if checkpoint is not None:
        runner_args["checkpoint"] = checkpoint
    print(
        json.dumps(
            {
                "task": TASK_ID,
                "arm": args_cli.arm,
                "seed": seed,
                "num_envs": num_envs,
                "horizon": horizon,
                "batch_size": num_envs * horizon,
                "minibatch_size": minibatch_size,
                "mini_epochs": agent_cfg["params"]["config"]["mini_epochs"],
                "max_updates": max_updates,
                "identity_digest": identity["identity_digest"],
                "run_dir": str(run_dir),
            },
            sort_keys=True,
        )
    )
    try:
        runner.run(runner_args)  # PPO rollout/update/checkpoint lifecycle
    finally:
        transport.close()  # failure也释放PhysX/CUDA scene resources


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
