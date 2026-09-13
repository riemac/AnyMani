r"""在经过复现核验的族环境中评价真正的离线共享学生。

参考教师 checkpoint 只提供冻结的环境、N040 与物理控制合同；被执行的动作
始终来自显式传入的 student TorchScript。产物独立记录两种身份，不能把
离线 Actor 包装成 PPO checkpoint 或继承教师的训练 frame 数。

入口在 AppLauncher 前只使用标准库，允许通过 PYTHONPATH 选择 LEAP 的
历史 runtime。主 evaluator 拥有 reset/step/终止统计，student callback
只消费合法 Actor 观察及共同静态运动学信息。
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path


def main() -> int:
    """解析模型来源，执行固定首轨迹评价，并在 Kit 关闭前封存成败状态。"""
    # --reference 属于下游 evaluator 的统计参考，不能被当作本层
    # --reference_teacher_checkpoint 的缩写而覆盖教师权重路径。
    parser = argparse.ArgumentParser(description="Evaluate one frozen offline family student.", allow_abbrev=False)
    parser.add_argument("--student_checkpoint", type=Path, required=True)
    parser.add_argument("--student_torchscript", type=Path, required=True)
    parser.add_argument("--student_sidecar", type=Path, default=None)
    parser.add_argument("--reference_teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--student_status", type=Path, required=True)
    args, evaluator_args = parser.parse_known_args()
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]  # 保留原 evaluator 的物理评价参数语义。
    if any(value == "--checkpoint" or value.startswith("--checkpoint=") for value in evaluator_args):
        raise ValueError("use --reference_teacher_checkpoint for the environment reference")
    if args.student_status.exists():
        raise FileExistsError("student status must use a new path")
    args.student_status.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--checkpoint", str(args.reference_teacher_checkpoint), *evaluator_args]
    backend = None  # Launcher 失败时也能形成 Python 错误记录。
    student = None
    code = 1
    try:
        backend = runpy.run_module(
            "anymani.distill.rl.evaluate_palm_rotation_mvp", run_name="family_student_evaluator_backend"
        )  # 非 __main__ 导入只完成装配，随后显式调用受控入口。
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
                }, ensure_ascii=False, indent=2,
            ) + "\n"
        )
        code = 0  # exit0 必须与真实学生产物同时成立。
    except BaseException as error:
        args.student_status.write_text(
            json.dumps(
                {
                    "status": "failed", "python_exit_code": 1,
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_policy_steps": getattr(student, "executed_steps", 0),
                }, ensure_ascii=False, indent=2,
            ) + "\n"
        )
        traceback.print_exc()
        sys.stderr.flush()  # 某些 Kit 关闭路径会吞掉尚未传播的异常，先写证据。
    finally:
        if backend is not None:
            backend["simulation_app"].close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
