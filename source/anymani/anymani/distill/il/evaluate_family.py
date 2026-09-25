"""Evaluate a frozen shared student on fixed hand-specific pregrasps."""

from __future__ import annotations
import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one frozen offline family student.", allow_abbrev=False)
    parser.add_argument("--student_checkpoint", type=Path, required=True)
    parser.add_argument("--student_torchscript", type=Path, required=True)
    parser.add_argument("--student_sidecar", type=Path, default=None)
    parser.add_argument("--reference_teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--student_status", type=Path, required=True)
    args, evaluator_args = parser.parse_known_args()
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]
    if any((value == "--checkpoint" or value.startswith("--checkpoint=") for value in evaluator_args)):
        raise ValueError("use --reference_teacher_checkpoint for the environment reference")
    if args.student_status.exists():
        raise FileExistsError("student status must use a new path")
    args.student_status.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--checkpoint", str(args.reference_teacher_checkpoint), *evaluator_args]
    backend = None
    student = None
    code = 1
    try:
        backend = runpy.run_module(
            "anymani.distill.rl.evaluate_palm_rotation_mvp", run_name="family_student_evaluator_backend"
        )
        from anymani.distill.il.family_evaluation import FrozenFamilyStudent

        student = FrozenFamilyStudent(
            args.student_checkpoint, args.student_torchscript, sidecar_path=args.student_sidecar
        )
        backend["main"](actor_override=student)
        output = backend["args_cli"].output.resolve()
        document = json.loads(output.read_text())
        if document.get("artifact_type") != "anymani.family_student_fixed_evaluation":
            raise RuntimeError("evaluation artifact does not identify the offline student")
        args.student_status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "python_exit_code": 0,
                    "evaluation": str(output),
                    "student_checkpoint": str(args.student_checkpoint.resolve()),
                    "student_torchscript": str(args.student_torchscript.resolve()),
                    "student": student.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        code = 0
    except BaseException as error:
        args.student_status.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "python_exit_code": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_policy_steps": getattr(student, "executed_steps", 0),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        if backend is not None:
            backend["simulation_app"].close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
