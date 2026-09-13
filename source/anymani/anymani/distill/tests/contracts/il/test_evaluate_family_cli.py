"""学生 CLI 必须区分参考教师权重与原 evaluator 的 N000 统计参考文件。"""

from __future__ import annotations

import sys

from anymani.distill.il import evaluate_family


def test_reference_statistics_argument_cannot_replace_teacher_checkpoint(monkeypatch, tmp_path):
    """在 Launcher 前截获真实转发 argv，防止 argparse 前缀缩写吞掉 --reference。"""
    teacher = tmp_path / "teacher.pth"
    reference = tmp_path / "n000.json"
    captured = []

    def capture_before_launcher(*_args, **_kwargs):
        captured.extend(sys.argv)
        raise RuntimeError("test interception before AppLauncher")

    monkeypatch.setattr(evaluate_family.runpy, "run_module", capture_before_launcher)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_family", "--student_checkpoint", str(tmp_path / "student.pt"),
        "--student_torchscript", str(tmp_path / "actor.ts"),
        "--reference_teacher_checkpoint", str(teacher),
        "--student_status", str(tmp_path / "status.json"),
        "--reference", str(reference), "--headless",
    ])
    assert evaluate_family.main() == 1  # 故意在 Launcher 前终止；不启动 Isaac。
    assert captured[captured.index("--checkpoint") + 1] == str(teacher)
    assert captured[captured.index("--reference") + 1] == str(reference)
    assert "--headless" in captured
