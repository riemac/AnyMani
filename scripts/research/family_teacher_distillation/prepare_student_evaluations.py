"""为已定义的共享学生和冻结评价成员生成可恢复的显式命令。

这里只发布命令，不读取最终未见手成绩或选择模型。LEAP 使用已核验的历史
运行版本，Allegro 使用当前版本，两个运行版本均执行同一个学生 TorchScript。
每份命令保留完整 nominal cohort，供后续统计初始化失败成员。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(case: Path, model_run: Path, root: Path) -> list[Path]:
    """按冻结 registry 生成八个评价任务，并核对已有命令的真实目标。"""
    registry = json.loads((case / "evaluation-populations/registry.json").read_text())
    paths = []
    for population in registry["populations"]:
        family, role = population["family"], population["role"]
        suffix = f"{family}-train" if population["split"] == "training" else role
        output = case / f"{model_run.name}-{suffix}-r16"
        launch_path = output / "launch.json"
        if launch_path.exists():
            existing = json.loads(launch_path.read_text())
            command = existing["command"]
            if (
                Path(command[command.index("--student_checkpoint") + 1]).resolve() != model_run / "model/best.pt"
                or command[command.index("--cohort_lock") + 1] != population["runtime_cohort"]
            ):
                raise ValueError(f"existing evaluation targets another model/population: {output}")
            paths.append(launch_path)
            continue
        template = json.loads((case / population["template"] / "launch.json").read_text())["command"]
        teacher = template[template.index("--checkpoint") + 1]
        runtime = case / "leap-runtime" if family == "leap" else root
        command = [
            "/home/hac/isaac/env_isaaclab/bin/python", str(root / "scripts/benchmarks/benchmark_heterogeneous_rl.py"),
            "--output_dir", str(output / "runtime"), "--live_output", "--timeout_s", "1800", "--",
            "/usr/bin/env", "TERM=xterm-256color", "VIRTUAL_ENV=/home/hac/isaac/env_isaaclab",
            f"PYTHONPATH={runtime / 'source/anymani'}", "OMP_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=0",
            "PYTHONDONTWRITEBYTECODE=1", "ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS=0",
            f"ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT={population['catalog']}",
            "/home/hac/isaac/IsaacLab/isaaclab.sh", "-p", "-m", "anymani.distill.il.evaluate_family",
            "--student_checkpoint", str(model_run / "model/best.pt"),
            "--student_torchscript", str(model_run / "deployment/actor.ts"),
            "--reference_teacher_checkpoint", teacher,
            "--student_status", str(output / "student-status.json"),
            "--headless", "--rl_games_strict", "--cohort_lock", population["runtime_cohort"],
            "--num_replicas", "16", "--steps", "600", "--trace_stride", "1", "--trace_rewards",
            "--reference", str(root / "logs/benchmarks/n000-fixed30-r16-20260905.json"),
            "--output", str(output / "evaluation.json"),
        ]
        if family == "leap" or population["split"] != "training":
            command.append("--cohort_transfer")  # 原语义检查保留，明确声明新的评价成员轴。
        launch = {
            "name": output.name, "cwd": str(runtime), "command": command,
            "expected_artifacts": [str(output / name) for name in ("student-status.json", "evaluation.json", "evaluation.h5")],
            "status_artifact": str(output / "student-status.json"), "status": "prepared",
            "population": population, "model_run": str(model_run),
            "maximum_policy_interactions": population["ready_assets"] * 16 * 600,
            "nominal_evaluation_interactions": population["nominal_assets"] * 16 * 600,
            "purpose": "Frozen shared-student evaluation with unchanged full nominal denominator.",
            "selection_boundary": "Final strict-unseen scores never select or tune checkpoints.",
            "registry_sha256": hashlib.sha256((case / "evaluation-populations/registry.json").read_bytes()).hexdigest(),
        }
        output.mkdir(parents=True, exist_ok=False)
        launch_path.write_text(json.dumps(launch, indent=2) + "\n")
        paths.append(launch_path)
    return paths


def main() -> None:
    """显式指定当前 case、模型运行与原仓库，不从目录名字猜测实验语义。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    paths = prepare(args.case.resolve(), args.model_run.resolve(), args.root.resolve())
    print(json.dumps([str(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()
