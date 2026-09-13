"""封存一次已完成的离线训练，导出同一学生供两族运行版本使用。"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha256(path: Path) -> str:
    """读取真实文件字节用于配置、模型和部署来源绑定。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """在成功训练后导出并验证；已有部署文件只能按原 SHA 复用。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root, case, model_run = args.root.resolve(), args.case.resolve(), args.model_run.resolve()
    report_path = model_run / "model/training-report.json"
    report = json.loads(report_path.read_text())
    if report["status"] not in {"completed", "time_limit"} or not report["epochs"]:
        raise ValueError("model is not a completed budgeted training run")
    checkpoint, deployment = model_run / "model/best.pt", model_run / "deployment/actor.ts"
    sidecar_path = Path(str(deployment) + ".json")
    if deployment.exists() or sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar["checkpoint_sha256"] != sha256(checkpoint) or sidecar["torchscript_sha256"] != sha256(deployment):
            raise ValueError("existing deployment does not match the completed model")
    else:
        import torch
        from anymani.distill.il.family_student import export_family_student_torchscript

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sidecar = export_family_student_torchscript(checkpoint, deployment, device="cuda:0")
        (deployment.parent / "export-report.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    origin = {
        "status": "bound-to-completed-training-run", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sidecar["checkpoint_sha256"], "torchscript_sha256": sidecar["torchscript_sha256"],
        "seed": report["run_identity"]["seed"], "variant": report["run_identity"]["representation"],
        "run_identity": report["run_identity"], "training_report": str(report_path),
        "training_report_sha256": sha256(report_path), "launch_sha256": sha256(model_run / "launch.json"),
        "best_epoch": report["best_epoch"],
    }
    (model_run / "model-origin.json").write_text(json.dumps(origin, indent=2) + "\n")
    module = runpy.run_path(str(root / "scripts/research/family_teacher_distillation/prepare_student_evaluations.py"))
    paths = module["prepare"](case, model_run, root)
    (model_run / "evaluation-launches.json").write_text(json.dumps([str(path) for path in paths], indent=2) + "\n")
    print(json.dumps({"model": model_run.name, "checkpoint_sha256": origin["checkpoint_sha256"], "evaluations": len(paths)}))


if __name__ == "__main__":
    main()
