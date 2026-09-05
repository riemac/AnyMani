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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, cast

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


def _select_support_rows(mvp80_rows: tuple[int, ...], raw_support_rows: str | None) -> tuple[int, ...]:
    r"""返回默认完整MVP80轴或其显式有序子集。

    Single-embodiment closure与8/20/40 progressive probes只允许消费已冻结MVP80 manifest中的rows，使
    strict-v5 catalog、物理身份与最终80手目标保持同源。空值恢复完整80轴；显式子集不按manifest重新排序，
    调用方顺序就是selection-local prototype axis。

    Args:
        mvp80_rows (tuple[int, ...]): 版本化80-row manifest顺序。
        raw_support_rows (str | None): 逗号分隔formal row列表。

    Returns:
        tuple[int, ...]: 当前run实际消费的有序支持集。
    """

    if raw_support_rows is None:
        return mvp80_rows  # 默认行为逐值保持既有正式80手
    selected = tuple(int(item.strip()) for item in raw_support_rows.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--support_rows must contain unique formal rows")
    outside = tuple(row for row in selected if row not in set(mvp80_rows))
    if outside:
        raise ValueError(f"--support_rows must be a subset of the frozen MVP80 manifest, got outside rows={outside}")
    return selected


parser = argparse.ArgumentParser(description="Train the 80-hand palm-rotation MVP with structured rl_games PPO.")
parser.add_argument(
    "--asset_manifest", type=Path, default=DEFAULT_MANIFEST, help="Versioned 80-row selection manifest."
)
parser.add_argument(
    "--support_rows",
    type=str,
    default=None,
    help="Explicit ordered subset of frozen MVP80 rows for single/few-embodiment closure.",
)
parser.add_argument(
    "--cohort_lock",
    type=Path,
    default=None,
    help="Member-level cohort lock; mutually exclusive with --support_rows and the legacy MVP80 axis.",
)
parser.add_argument("--num_envs", type=int, default=None, help="2560 formal, 1280 fallback, or 80 with --smoke.")
parser.add_argument(
    "--arm",
    choices=("base", "residual", "direct", "direct_token"),
    default="residual",
    help="Actor arm; direct preserves the historical local skip, while direct_token reads only contextual tokens.",
)
parser.add_argument(
    "--history_encoder",
    choices=("tcn", "raw_stack"),
    default="tcn",
    help="Per-joint History30 route; raw_stack feeds all 150 lagged scalars to the local FiLM MLP.",
)
parser.add_argument("--seed", type=int, default=42, help="Formal protocol uses 42, then 43/44 after seed42 passes.")
parser.add_argument(
    "--max_updates", type=int, default=None, help="Override PPO updates; default preserves 30M pulse budget."
)
parser.add_argument(
    "--reward_release_start_turns",
    type=float,
    default=1.0,
    help="Cell-median positive-turn EMA where stability/contact shaping begins; baseline is 1.0 turn.",
)
parser.add_argument(
    "--reward_release_end_turns",
    type=float,
    default=2.0,
    help="Cell-median positive-turn EMA where stability/contact shaping reaches full weight.",
)
parser.add_argument(
    "--minibatches", type=int, default=None, help="Formal default 16; every minibatch remains asset-balanced."
)
parser.add_argument(
    "--gradient_probe_frequency",
    type=int,
    default=None,
    help="Head-level per-asset Gram cadence；0关闭，正式缺省500。",
)
parser.add_argument(
    "--full_gradient_shadow_frequency",
    type=int,
    default=None,
    help="完整Actor global/per-asset replica-half梯度shadow cadence；0关闭且不改变主optimizer。",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=None,
    help="Activation slices per logical optimizer step; use 8 slices/2 accumulation to preserve 4 logical steps.",
)
parser.add_argument(
    "--advantage_normalization_scope",
    choices=("global", "per_asset_rollout"),
    default=None,
    help="Actor GAE normalization population；per_asset_rollout使用每资产完整rollout moments。",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Full actor/critic/optimizers/curriculum checkpoint.")
parser.add_argument(
    "--actor_init_checkpoint",
    type=str,
    default=None,
    help="只加载Actor参数的新run初始化；Critic/optimizers/normalizer/curricula全部重置。",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Run name under logs/distill/rl_games.")
parser.add_argument("--sigma", type=float, default=None, help="Optional rl_games play-time sigma override.")
parser.add_argument(
    "--tf32", action="store_true", help="Explicit TF32 numeric-speed candidate; tensors and Adam remain FP32."
)
parser.add_argument(
    "--torch_compile",
    choices=("default", "reduce-overhead"),
    default=None,
    help="Compile the PPO model after checkpoint restore; default YAML remains eager.",
)
parser.add_argument(
    "--smoke", action="store_true", help="Use 80 env, horizon 4, one mini-epoch/update integration mode."
)
parser.add_argument("--rl_games_strict", action="store_true", help="Require pinned local rl_games v1.6.5 commit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, launcher_unknown_args = parser.parse_known_args()
if args_cli.checkpoint is not None and args_cli.actor_init_checkpoint is not None:
    raise ValueError("--checkpoint and --actor_init_checkpoint are mutually exclusive")
if args_cli.reward_release_start_turns < 0.0:
    raise ValueError("--reward_release_start_turns must be non-negative")
if args_cli.reward_release_end_turns <= args_cli.reward_release_start_turns:
    raise ValueError("--reward_release_end_turns must exceed --reward_release_start_turns")

# Static scene在Isaac config import时构造，因此先解析轻量lock root并冻结process环境变量。
if args_cli.cohort_lock is not None:
    if args_cli.support_rows is not None:
        raise ValueError("--cohort_lock and --support_rows are mutually exclusive")
    cohort_lock_path = (
        args_cli.cohort_lock
        if args_cli.cohort_lock.is_absolute()
        else (ANYMANI_ROOT / args_cli.cohort_lock).resolve(strict=True)
    )
    cohort_document = yaml.safe_load(cohort_lock_path.read_text(encoding="utf-8"))
    if not isinstance(cohort_document, dict) or cohort_document.get("schema_version") not in {
        "1.0.0",
        "1.1.0",
        "1.2.0",
    }:
        raise ValueError("--cohort_lock must contain a supported schema-1.x member-level cohort")
    cohort_members = cohort_document.get("members")
    if not isinstance(cohort_members, list) or not cohort_members:
        raise ValueError("--cohort_lock must contain a non-empty members list")
    if not args_cli.smoke and cohort_document.get("schema_version") != "1.2.0":
        raise ValueError("formal cohort PPO requires a schema-1.2 canonical-final lock")
    selected_rows = tuple(range(len(cohort_members)))  # selection-local diagnostic/prototype axis
    asset_count = len(selected_rows)
    support_manifest_path = cohort_lock_path  # exact membership/order bytes enter method identity
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(cohort_lock_path)
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
else:
    mvp80_rows = _load_mvp80_rows(args_cli.asset_manifest)  # 版本化最终支持集，不因closure改写
    selected_rows = _select_support_rows(mvp80_rows, args_cli.support_rows)  # 当前run prototype axis
    asset_count = len(selected_rows)  # $A\in\{1,\ldots,80\}$；默认$A=80$
    support_manifest_path = args_cli.asset_manifest
    os.environ.pop("ANYMANI_HETERO_COHORT_LOCK", None)
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)
num_envs = int(args_cli.num_envs if args_cli.num_envs is not None else (80 if args_cli.smoke else 2560))
if args_cli.smoke:
    if num_envs < asset_count or num_envs % asset_count != 0:
        raise ValueError("smoke num_envs must be a positive multiple of the selected support assets")
elif args_cli.cohort_lock is not None and (args_cli.num_envs is None or num_envs % asset_count != 0):
    raise ValueError("cohort training requires explicit --num_envs divisible by cohort asset count")
elif asset_count == 80 and num_envs not in (1280, 2560):
    raise ValueError("full MVP80 permits only 2560 envs or the 1280-env memory fallback")
elif asset_count < 80 and (args_cli.num_envs is None or num_envs % asset_count != 0):
    raise ValueError("subset closure requires explicit --num_envs divisible by its selected asset count")
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
torch.backends.cuda.matmul.allow_tf32 = bool(args_cli.tf32)  # Runner之前先声明；构造后还会重新强制
torch.backends.cudnn.allow_tf32 = bool(args_cli.tf32)  # temporal convolution与matmul共享显式候选身份

from rl_games.common import env_configurations, vecenv  # noqa: E402

from anymani.distill.rl.observers import OneShotIsaacAlgoObserver  # noqa: E402
from anymani.distill.rl.palm_rotation_ppo import (  # noqa: E402
    PalmRotationPpoRunner,
    register_palm_rotation_ppo,
    validate_gradient_probe_compile_compatibility,
)
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
    palm_rotation_code_provenance,
)
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_vecenv import (  # noqa: E402
    PalmRotationRlGamesGpuEnv,
    PalmRotationRlGamesVecEnv,
)
from anymani.distill.rl.runtime.palm_rotation_warm_start import (  # noqa: E402
    inspect_actor_init_checkpoint,
    inspect_resumed_actor_warm_start,
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
    if minibatch_count < 1 or (batch_size // asset_count) % minibatch_count != 0:
        raise ValueError("per-asset rollout samples must be divisible into all stratified minibatches")
    minibatch_size = batch_size // minibatch_count  # 每个minibatch始终含全部80 assets
    accumulation_steps = int(
        args_cli.gradient_accumulation_steps
        if args_cli.gradient_accumulation_steps is not None
        else agent_cfg["params"]["config"]["gradient_accumulation_steps"]
    )
    if accumulation_steps < 1 or minibatch_count % accumulation_steps != 0:
        raise ValueError("gradient accumulation steps must be positive and divide activation minibatches")
    if not args_cli.smoke and asset_count < 80 and args_cli.max_updates is None:
        raise ValueError("subset closure requires an explicit --max_updates scientific budget")
    default_updates = 1 if args_cli.smoke else 391 * (2560 // num_envs)  # 完整MVP matched约30.03M transitions
    max_updates = int(args_cli.max_updates if args_cli.max_updates is not None else default_updates)
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    config["horizon_length"] = horizon
    config["minibatch_size"] = minibatch_size
    config["gradient_accumulation_steps"] = accumulation_steps
    config["mini_epochs"] = 1 if args_cli.smoke else 5
    config["max_epochs"] = max_updates
    config["num_actors"] = num_envs
    config["asset_count"] = asset_count
    if args_cli.gradient_probe_frequency is not None:
        if args_cli.gradient_probe_frequency < 0:
            raise ValueError("--gradient_probe_frequency must be non-negative")
        config["gradient_probe_frequency"] = int(args_cli.gradient_probe_frequency)
    if args_cli.full_gradient_shadow_frequency is not None:
        if args_cli.full_gradient_shadow_frequency < 0:
            raise ValueError("--full_gradient_shadow_frequency must be non-negative")
        config["full_gradient_shadow_frequency"] = int(args_cli.full_gradient_shadow_frequency)
    if args_cli.advantage_normalization_scope is not None:
        config["advantage_normalization_scope"] = str(args_cli.advantage_normalization_scope)
    config["name"] = f"heterogeneous_palm_rotation_mvp_{args_cli.arm}"
    config["torch_compile"] = False  # 外部Runner wrapper保持eager，避免Isaac进程静默zero-update退出
    agent_cfg["params"]["network"]["palm_rotation"]["arm"] = args_cli.arm
    agent_cfg["params"]["network"]["palm_rotation"]["history_encoder"] = args_cli.history_encoder
    agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"] = args_cli.torch_compile
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
    reward_release_params = env_cfg.curriculum.reward_release.params  # type: ignore[union-attr]  # 训练MDP课程配置
    reward_release_params["release_start_turns"] = float(args_cli.reward_release_start_turns)
    reward_release_params["release_end_turns"] = float(args_cli.reward_release_end_turns)
    reward_release_ema_alpha = float(
        cast(Any, reward_release_params["ema_alpha"])
    )  # episode-cohort EMA更新率，baseline 0.05
    agent_path = ANYMANI_ROOT / "source/anymani/anymani/distill/rl/agents/heterogeneous_palm_rotation_mvp_ppo.yaml"
    agent_cfg = yaml.safe_load(agent_path.read_text(encoding="utf-8"))  # versioned rl_games config
    if not isinstance(agent_cfg, dict):
        raise TypeError("palm-rotation rl_games YAML must contain a mapping")
    seed = _resolve_seed(agent_cfg)  # task随机状态与PPO RNG统一
    horizon, minibatch_size, max_updates = _configure_budget(agent_cfg)
    validate_gradient_probe_compile_compatibility(
        args_cli.torch_compile,
        int(agent_cfg["params"]["config"]["gradient_probe_frequency"]),
        int(agent_cfg["params"]["config"]["full_gradient_shadow_frequency"]),
    )  # fail-fast，禁止在u500才由AOTAutograd donated buffers终止长run
    env_cfg.scene.num_envs = num_envs  # 与pre-import static routing严格相等
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = seed
    rl_device = str(agent_cfg["params"]["config"].get("device", env_cfg.sim.device))
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    checkpoint = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None
    actor_init_checkpoint = (
        retrieve_file_path(args_cli.actor_init_checkpoint) if args_cli.actor_init_checkpoint is not None else None
    )
    _, run_dir = _configure_log_dir(agent_cfg, checkpoint=checkpoint)
    if checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["full_checkpoint_resume"] = checkpoint is not None
    env_cfg.log_dir = str(run_dir)  # task diagnostics与PPO artifact共用run root

    # Environment必须先实例化，N040 canonical evidence使用同一ASSET_BINDING与device。
    env = gym.make(TASK_ID, cfg=env_cfg)
    provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=rl_device)
    actor_warm_start = (
        inspect_actor_init_checkpoint(
            actor_init_checkpoint,
            target_arm=str(args_cli.arm),
            target_history_encoder=str(args_cli.history_encoder),
            target_provider_identity=provider.identity,
        )
        if actor_init_checkpoint is not None
        else (inspect_resumed_actor_warm_start(checkpoint) if checkpoint is not None else None)
    )
    if actor_init_checkpoint is not None:
        agent_cfg["params"]["config"]["actor_init_checkpoint"] = actor_init_checkpoint
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
        manifest_path=support_manifest_path,
        selected_rows=selected_rows,
        pregrasp=GOOD_PREGRASP_RESET_CFG,
        arm=str(args_cli.arm),
        run_contract={
            "seed": seed,
            "num_envs": num_envs,
            "asset_count": asset_count,
            "cohort_id": ASSET_BINDING.cohort_id,
            "cohort_lock_sha256": ASSET_BINDING.cohort_lock_sha256,
            "source_member_keys": list(ASSET_BINDING.source_member_keys),
            "actor_warm_start": actor_warm_start,
            "horizon_length": horizon,
            "minibatch_size": minibatch_size,
            "minibatch_count": (num_envs * horizon) // minibatch_size,
            "gradient_accumulation_steps": int(ppo_cfg["gradient_accumulation_steps"]),
            "mini_epochs": int(ppo_cfg["mini_epochs"]),
            "gradient_probe_frequency": int(ppo_cfg["gradient_probe_frequency"]),
            "full_gradient_shadow_frequency": int(ppo_cfg["full_gradient_shadow_frequency"]),
            "gamma": float(ppo_cfg["gamma"]),
            "gae_lambda": float(ppo_cfg["tau"]),
            "ppo_clip": float(ppo_cfg["e_clip"]),
            "entropy_coef": float(ppo_cfg["entropy_coef"]),
            "grad_norm": float(ppo_cfg["grad_norm"]),
            "actor_base_lr": float(ppo_cfg["learning_rate"]),
            "adaptive_lr_max": float(ppo_cfg["adaptive_lr_max"]),
            "actor_residual_lr": float(ppo_cfg["residual_learning_rate"]),
            "actor_contextual_lr": float(ppo_cfg["contextual_learning_rate"]),
            "critic_lr": float(ppo_cfg["critic_learning_rate"]),
            "lr_schedule": str(ppo_cfg["lr_schedule"]),
            "normalize_advantage": bool(ppo_cfg["normalize_advantage"]),
            "advantage_normalization_scope": str(ppo_cfg["advantage_normalization_scope"]),
            "reward_release_start_turns": float(args_cli.reward_release_start_turns),
            "reward_release_end_turns": float(args_cli.reward_release_end_turns),
            "reward_release_ema_alpha": reward_release_ema_alpha,
            "normalize_value": bool(ppo_cfg["normalize_value"]),
            "initial_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["initial_log_std"]),
            "max_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["max_log_std"]),
            "base_action_limit": float(agent_cfg["params"]["network"]["palm_rotation"]["base_action_limit"]),
            "history_encoder": str(agent_cfg["params"]["network"]["palm_rotation"]["history_encoder"]),
            "allow_tf32": bool(args_cli.tf32),
            "torch_compile": agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"],
            "rl_games_backend_commit": backend_info.git_commit,
            "device": rl_device,
        },
    )
    agent_cfg["params"]["network"]["anymani_identity"] = identity
    agent_cfg["params"]["config"]["num_actors"] = transport.num_envs
    agent_cfg["params"]["config"]["code_provenance"] = palm_rotation_code_provenance()
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
    precision_flags = enforce_palm_rotation_precision(
        allow_tf32=bool(args_cli.tf32)
    )  # Runner.__init__会改global flags，必须在model build前恢复声明模式
    runner.load(agent_cfg)
    runner.reset()
    if enforce_palm_rotation_precision(allow_tf32=bool(args_cli.tf32)) != precision_flags:
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
                "history_encoder": args_cli.history_encoder,
                "tf32": bool(args_cli.tf32),
                "torch_compile": agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"],
                "seed": seed,
                "num_envs": num_envs,
                "asset_count": asset_count,
                "cohort_id": ASSET_BINDING.cohort_id,
                "horizon": horizon,
                "batch_size": num_envs * horizon,
                "minibatch_size": minibatch_size,
                "gradient_accumulation_steps": agent_cfg["params"]["config"]["gradient_accumulation_steps"],
                "advantage_normalization_scope": agent_cfg["params"]["config"]["advantage_normalization_scope"],
                "reward_release_start_turns": float(args_cli.reward_release_start_turns),
                "reward_release_end_turns": float(args_cli.reward_release_end_turns),
                "reward_release_ema_alpha": reward_release_ema_alpha,
                "full_gradient_shadow_frequency": agent_cfg["params"]["config"]["full_gradient_shadow_frequency"],
                "mini_epochs": agent_cfg["params"]["config"]["mini_epochs"],
                "max_updates": max_updates,
                "identity_digest": identity["identity_digest"],
                "run_dir": str(run_dir),
                "actor_init_checkpoint_sha256": (
                    actor_warm_start["checkpoint_sha256"] if actor_warm_start is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    try:
        runner.run(runner_args)  # PPO rollout/update/checkpoint lifecycle
        metrics_path = run_dir / "metrics.parquet"  # 正常预算结束必须由custom agent finalize
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            raise RuntimeError("palm-rotation Runner returned without any finalized PPO update metrics")
    finally:
        transport.close()  # failure也释放PhysX/CUDA scene resources


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)  # Kit shutdown不得把训练异常覆盖成exit 0
