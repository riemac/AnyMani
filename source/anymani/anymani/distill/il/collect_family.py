r"""先初始化选定的冻结runtime，再通过明确callback采集族教师。

入口只在AppLauncher之前使用标准库。通过PYTHONPATH选择经过恢复核验的LEAP
旧runtime或当前Allegro runtime，两者都调用同一记录合同，物理推进由原evaluator负责。
原evaluator的main显式接收collector；不通过替换权重或绕过identity检查加载旧教师。
Python错误在Kit shutdown之前写入独立status，避免旧入口以exit0掩盖未生成数据。
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path


def main() -> int:
    """解析采集参数，剩余CLI逐项交给原有固定物理评价入口。"""
    parser = argparse.ArgumentParser(description="Collect a frozen family teacher for offline shared students.")
    parser.add_argument("--dataset_output", type=Path, required=True)  # 新HDF5路径，不能覆盖。
    parser.add_argument("--collection_family", choices=("leap_right", "allegro_right"), required=True)
    parser.add_argument("--collection_action_mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--collection_action_seed", type=int, default=None)  # sample独立随机流。
    parser.add_argument("--dataset_stride", type=int, default=4)  # 原始20Hz帧仍完整保存。
    parser.add_argument("--collector_status", type=Path, required=True)  # 必须在Kit退出前封存。
    args, evaluator_args = parser.parse_known_args()
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]  # 仅去掉CLI分隔符，不重解释任何物理参数。
    if args.dataset_output.exists() or args.collector_status.exists():
        raise FileExistsError("collection data and status paths must be new")
    args.collector_status.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *evaluator_args]  # checkpoint/cohort/R16/H600仍由evaluator解析。
    backend = None  # 初始化失败也必须能写清失败阶段。
    collector = None  # 所有句柄属于本主进程，异常时负责关闭。
    code = 1
    try:
        backend = runpy.run_module(
            "anymani.distill.rl.evaluate_palm_rotation_mvp", run_name="family_evaluator_backend"
        )  # 非__main__导入只完成launcher与模块装配，尚未执行rollout。
        from anymani.distill.il.family_collection import FrozenFamilyCollection
        from anymani.distill.il.family_dataset import read_family_metadata

        collector = FrozenFamilyCollection(
            args.dataset_output,
            family=args.collection_family,
            mode=args.collection_action_mode,
            seed=args.collection_action_seed,
            sample_stride=args.dataset_stride,
        )  # 新callback无环境控制权；行为模式在数据来源中显式标识。
        backend["main"](collector=collector)
        metadata = read_family_metadata(args.dataset_output)  # 完成标志/数组轴仍需独立检查。
        args.collector_status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "python_exit_code": 0,
                    "dataset": str(args.dataset_output.resolve()),
                    "frames": metadata["recorded_steps"],
                    "sample_steps": metadata["sample_count"],
                    "env_count": metadata["env_count"],
                    "qualified_episodes": metadata["qualified_episode_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        code = 0  # 成功必须同时有collector完整数据，而不是仅子进程exit0。
    except BaseException as error:
        args.collector_status.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "python_exit_code": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_policy_steps": collector.executed_steps if collector is not None else 0,
                    "attempted_policy_steps": collector._step if collector is not None else 0,
                    "env_count": int(collector.active.numel()) if collector is not None and collector.active is not None else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        sys.stderr.flush()  # Kit关闭可能直接终止解释器；错误先刷出。
    finally:
        if collector is not None:
            collector.close()  # 未finalize的数据保持incomplete，不升级成完整数据。
        if backend is not None:
            backend["simulation_app"].close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
