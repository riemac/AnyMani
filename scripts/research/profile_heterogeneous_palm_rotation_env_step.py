r"""对掌旋ManagerBasedRLEnv policy step做matched串行化组件计时。

该profile在相同128 env、6个120 Hz substeps、20 Hz policy下比较单资产row与member-level cohort。每个被测方法
前后调用``torch.cuda.synchronize``，因此结果是用于归因的serialized wall time，不是生产吞吐；正式吞吐仍以
PPO Parquet的异步``environment_step_seconds``为准。脚本不修改IsaacLab，只在当前env实例上临时包装方法。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from isaaclab.app import AppLauncher


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    r"""在task config import前解析互斥asset routing与计时步数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if (args.cohort_lock is None) == (args.rows is None):
        parser.error("exactly one of --cohort-lock or --rows is required")
    if args.num_envs < 1 or args.warmup_steps < 1 or args.profile_steps < 1:
        parser.error("environment and step counts must be positive")
    return args, unknown


ARGS, UNKNOWN_ARGS = _parse_args()
if ARGS.cohort_lock is not None:
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(ARGS.cohort_lock.expanduser().resolve(strict=True))
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
else:
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = str(ARGS.rows)
    os.environ.pop("ANYMANI_HETERO_COHORT_LOCK", None)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(ARGS.num_envs)
sys.argv = [sys.argv[0], *UNKNOWN_ARGS]
app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app


def main() -> None:
    r"""构造真实task、包装关键方法并原子发布组件wall-time JSON。"""

    import anymani.distill.rl  # noqa: F401  # 注册distill-owned RLGames Gym alias
    import anymani.tasks.hetero  # noqa: F401  # 注册底层heterogeneous task
    import gymnasium as gym
    import torch
    from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import GeneratedPalmRotationMvpEnvCfg
    from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING
    from isaaclab.envs import ManagerBasedRLEnv

    cfg = GeneratedPalmRotationMvpEnvCfg()
    cfg.scene.num_envs = int(ARGS.num_envs)
    env = gym.make("AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0", cfg=cfg)
    unwrapped = cast(ManagerBasedRLEnv, env.unwrapped)
    actions = torch.zeros(ARGS.num_envs, 16, device=unwrapped.device)  # 固定零动作隔离policy变化
    try:
        env.reset()
        for _ in range(int(ARGS.warmup_steps)):
            env.step(actions)  # 初始化lazy contact/observation/PhysX buffers，不进入profile

        totals: dict[str, float] = defaultdict(float)
        calls: dict[str, int] = defaultdict(int)

        def wrap(owner: Any, method_name: str, label: str) -> None:
            r"""在一个bound method前后同步CUDA并累计serialized wall time。"""

            original = getattr(owner, method_name)

            def measured(*args: Any, **kwargs: Any) -> Any:
                r"""执行一次真实方法并把其同步wall time归入固定label。"""

                torch.cuda.synchronize()
                start = time.perf_counter()
                result = original(*args, **kwargs)
                torch.cuda.synchronize()
                totals[label] += time.perf_counter() - start
                calls[label] += 1
                return result

            setattr(owner, method_name, measured)

        # 顺序对应ManagerBasedRLEnv.step：action→6×physics→termination/reward→command→observation。
        wrap(unwrapped.action_manager, "process_action", "action_process")
        wrap(unwrapped.action_manager, "apply_action", "action_apply")
        wrap(unwrapped.scene, "write_data_to_sim", "scene_write")
        wrap(unwrapped.sim, "step", "simulation_step")
        wrap(unwrapped.scene, "update", "scene_update")
        wrap(unwrapped.termination_manager, "compute", "termination")
        wrap(unwrapped.reward_manager, "compute", "reward")
        wrap(unwrapped.command_manager, "compute", "command")
        wrap(unwrapped.observation_manager, "compute", "observation")

        torch.cuda.synchronize()
        overall_start = time.perf_counter()
        for _ in range(int(ARGS.profile_steps)):
            env.step(actions)
        torch.cuda.synchronize()
        overall_seconds = time.perf_counter() - overall_start

        components = {
            label: {
                "calls": calls[label],
                "total_seconds": totals[label],
                "mean_call_seconds": totals[label] / calls[label],
                "seconds_per_policy_step": totals[label] / int(ARGS.profile_steps),
            }
            for label in sorted(totals)
        }
        document = {
            "artifact_type": "anymani.heterogeneous_palm_rotation_env_step_profile",
            "schema_version": "1.0.0",
            "timing_semantics": "cuda-synchronized-serialized-component-wall-time",
            "asset_count": ASSET_BINDING.asset_count,
            "cohort_id": ASSET_BINDING.cohort_id,
            "cohort_lock_sha256": ASSET_BINDING.cohort_lock_sha256 or None,
            "source_member_keys": list(ASSET_BINDING.source_member_keys),
            "num_envs": int(ARGS.num_envs),
            "warmup_steps": int(ARGS.warmup_steps),
            "profile_steps": int(ARGS.profile_steps),
            "decimation": int(cfg.decimation),
            "overall_seconds": overall_seconds,
            "overall_seconds_per_policy_step": overall_seconds / int(ARGS.profile_steps),
            "component_seconds_sum": sum(totals.values()),
            "components": components,
        }
        output = ARGS.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(document, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
