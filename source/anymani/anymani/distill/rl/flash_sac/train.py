r"""AnyMani共享掌旋的FlashSAC训练入口，预算以真实新transition计数。

示例：python -m anymani.distill.rl.flash_sac.train --headless --cohort_lock <canonical.lock.yaml>
主环境即唯一采样环境；首30秒与完整回合表现从训练产物提取，不自动启动评价回放。
总预算默认8192000，预热100352也包含其中；stop_after_transitions可以分段验证并完整续接。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import FlashSACConfig
from .runtime import prepare_scene_route

ANYMANI_ROOT = Path(__file__).resolve().parents[6]  # <repo>/source/anymani/anymani/distill/rl/flash_sac/train.py。


def _record_failure(error: BaseException, run_dir: Path, training: Any = None) -> None:
    r"""在Kit关闭之前持久化首个Python失败，供父进程核对真实退出原因。"""
    failure: dict[str, Any] = {"exception_type": type(error).__name__, "message": str(error)}  # wrapper原始错误字段。
    if training is not None:
        failure.update(collected_transitions=training.learner.collected_transitions,
                       attempted_transitions=training.attempted_transitions)  # 失败尝试仍计入交互预算。
    payload = json.dumps(failure, ensure_ascii=False) + "\n"  # 仅保存小型事实，不从错误状态发布模型。
    destinations = [run_dir / "python_failure.json"]  # 训练run自身保存同一失败事实。
    evidence_dir = os.environ.get("ANYMANI_RL_EVIDENCE_DIR")  # wrapper分配的case证据目录。
    if evidence_dir:
        destinations.append(Path(evidence_dir) / "python_failure.json")  # 防止应用退出码掩盖原异常。
    for destination in destinations:
        if not destination.exists():
            destination.write_text(payload)  # 后续清理错误不覆盖首个因果入口。


def main() -> None:
    r"""先冻结静态场景，再启动Isaac并组装SAC；异常先记账，再清理仿真资源。"""
    from isaaclab.app import AppLauncher  # 只在真实CLI运行时导入应用启动器。

    defaults = FlashSACConfig()  # 不启动模型/GPU，仅提供带物理语义的候选默认值。
    parser = argparse.ArgumentParser(description=__doc__)  # 参数与输出配置明确对应。
    parser.add_argument("--run_dir", type=Path, default=None)  # 明确覆盖默认输出位置。
    parser.add_argument("--resume", type=Path, default=None)  # 仅完整SAC checkpoint，不接受PPO权重。
    parser.add_argument("--seed", type=int, default=defaults.seed)  # 网络/物理随机性来源。
    parser.add_argument("--num_envs", type=int, default=defaults.num_envs)  # 每资产等量副本。
    parser.add_argument("--total_transitions", type=int, default=defaults.total_transitions)  # 包含预热。
    parser.add_argument("--learning_starts", type=int, default=defaults.learning_starts)  # 只控制学习启动量。
    parser.add_argument("--replay_capacity", type=int, default=defaults.replay_capacity)  # 紧凑动态transition数。
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)  # 严格资产均衡batch。
    parser.add_argument("--actor_variant", choices=("structured", "flash_mlp"), default=defaults.actor_variant)
    parser.add_argument("--stop_after_transitions", type=int, default=None)  # 总量坐标上的完整进程停止边界。
    parser.add_argument("--checkpoint_interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--console_interval", type=int, default=defaults.console_interval)
    parser.add_argument("--gpu_headroom_mib", type=int, default=500)  # driver剩余显存，覆盖PhysX之外的分配。
    parser.add_argument("--no_tf32", action="store_true")  # 默认显式允许FP32网络的TF32内部乘法。
    AppLauncher.add_app_launcher_args(parser)  # headless、device等Isaac选项。
    parser.add_argument("--cohort_lock", type=Path, required=True)  # AppLauncher预解析完成后声明必需的成员清单。
    args, unknown = parser.parse_known_args()  # Kit剩余参数由AppLauncher消费。
    cohort = prepare_scene_route(args.cohort_lock, args.num_envs)  # 必须在任务配置import之前。
    config = replace(
        defaults, seed=args.seed, num_envs=args.num_envs, asset_count=len(cohort["members"]),
        total_transitions=args.total_transitions, learning_starts=args.learning_starts,
        replay_capacity=args.replay_capacity, batch_size=args.batch_size, actor_variant=args.actor_variant,
        checkpoint_interval=args.checkpoint_interval, console_interval=args.console_interval,
        gpu_headroom_bytes=args.gpu_headroom_mib * 1024**2,  # MiB转换为字节，方法身份保留实际值。
    )
    if config.asset_count != 256:
        raise ValueError("the first FlashSAC comparison requires the unchanged 256-asset cohort")
    if config.total_transitions > 8_192_000:
        raise ValueError("this comparison entry is capped at the authorized 8192000 new transitions")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")  # 自动路径只标识运行，不参与科学统计。
    run_dir = (args.run_dir or ANYMANI_ROOT / "logs/distill/rl_games/heterogeneous_palm_rotation_flash_sac" /
               f"orientation-exponential-{config.actor_variant}-s{config.seed}-{stamp}").resolve()
    if (run_dir / "metrics.jsonl").exists():
        raise FileExistsError("existing run metrics cannot be overwritten; resume into a new run directory")
    (run_dir / "params").mkdir(parents=True, exist_ok=True)  # 保存实际CLI与配置，尚未启动GPU。
    (run_dir / "params/flash_sac.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (run_dir / "params/command.json").write_text(json.dumps({"argv": sys.argv, "run_dir": str(run_dir)}, indent=2) + "\n")
    provenance = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ANYMANI_ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ANYMANI_ROOT, text=True)),
        "python_version": sys.version,  # Git来源独立于实际源码内容身份。
    }
    (run_dir / "params/code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sys.argv = [sys.argv[0], *unknown]  # 不将学习器选项传给Kit解释。
    app_launcher = AppLauncher(args)  # 创建本次训练唯一的Isaac应用。
    app = app_launcher.app  # 后续任何异常也必须关闭应用。
    environment = None  # 用于区分场景尚未创建与已有可关闭环境。
    training = None  # 初始化失败时尚无采样计数。
    try:
        import numpy as np

        from .runtime import create_environment
        from .training import FlashSACTrainingRun

        random.seed(config.seed)  # Python任务随机性可复现，并在checkpoint中保存。
        np.random.seed(config.seed)  # NumPy任务随机性独立于Torch策略采样。
        environment, provider, _, identity = create_environment(
            args.cohort_lock.resolve(), run_dir, config=config, device=args.device, allow_tf32=not args.no_tf32,
        )  # 原物理/奖励、冻结N040和原生SAC身份。
        (run_dir / "params/runtime_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
        training = FlashSACTrainingRun(config, environment, provider, identity, run_dir, resume=args.resume)
        print(f"[FlashSAC START] run={run_dir} assets={config.asset_count} envs={config.num_envs} "
              f"budget={config.total_transitions} warmup={config.learning_starts}", flush=True)
        report = training.run(stop_after_transitions=args.stop_after_transitions)  # 预热与验证采样均计预算。
        print(f"[FlashSAC FINISHED] status={report['status']} transitions={report['transitions']}", flush=True)
    except BaseException as error:
        _record_failure(error, run_dir, training)  # 原始Python错误和交互计数先持久化，再关闭Kit。
        raise  # 不能让Kit关闭时的退出码覆盖真正的训练错误。
    finally:
        try:
            if environment is not None:
                environment.close()  # 写出自然结束/右删失回合，恢复reset回调并释放场景。
        except BaseException as error:
            _record_failure(error, run_dir, training)  # 右删失记录或环境清理失败也需要被wrapper识别。
            raise
        finally:
            app.close()  # 始终只关闭本次拥有的应用。


if __name__ == "__main__":
    main()  # 由显式训练命令启动，不在包import时创建应用。
