r"""LEAP non-thumb 根部 fixed 段回归测试。

这组测试专门保护 `Leap-Non-Thumb.png` 里的一个关键科研语义：

- palm 与 `{0}` 之间存在一段长度为 $l_f$ 的真实 fixed 根部段；
- 这段 fixed 根部段不能再被“偷折算”为第一个 revolute joint 的前置位移；
- 因此 preview 产物里必须真的出现一段独立的 fixed root mesh。
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

from assets.presets import get_finger_builder_preset


REPO_ROOT = Path(__file__).resolve().parents[5]
PREVIEW_FINGER_SCRIPT = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "presets" / "preview" / "preview_finger_preset.py"


def test_leap_non_thumb_builds_explicit_fixed_root_segment():
    r"""LEAP non-thumb 应显式生成 `l_f` 根部 fixed joint/link。"""

    cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(name="index", parent_link="palm")
    finger = cfg.class_type(cfg).build()

    root_fixed, j0, j1, j2, j3, tip = finger.joints

    assert len(finger.joints) == 6  # root fixed + 4 revolute + tip fixed
    assert finger.dof_count == 4  # 加入 fixed 根部段不应改变 revolute DOF 总数

    assert root_fixed.joint_type == "fixed"
    assert root_fixed.parent == "palm"
    assert root_fixed.child == "index_root_fixed_link"
    assert root_fixed.metadata["fixed_root_segment"] is True
    assert root_fixed.visuals[0].geometry.kind == "box"
    assert root_fixed.collisions[0].geometry.kind == "box"
    assert math.isclose(root_fixed.visuals[0].geometry.size[1], 0.013, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(root_fixed.visuals[0].origin.pos[1], 0.0065, rel_tol=0.0, abs_tol=1e-9)

    # `{0}` 应从 fixed 根部段顶端长出，而不是直接从 palm 根长出。
    assert j0.joint_type == "revolute"
    assert j0.parent == root_fixed.child
    assert j0.child == "index_mcp1"
    assert math.isclose(j0.origin.pos[1], 0.013, rel_tol=0.0, abs_tol=1e-9)
    assert all(current.parent == previous.child for previous, current in zip(finger.joints[:-1], finger.joints[1:]))
    assert tip.joint_type == "fixed"
    assert tip.child == "index_tip"


def test_preview_finger_script_exports_leap_non_thumb_with_fixed_root(tmp_path):
    r"""LEAP non-thumb preview URDF 中应能直接看到根部 fixed 段。"""

    output_dir = tmp_path / "leap_non_thumb_preview"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREVIEW_FINGER_SCRIPT),
            "--preset",
            "leap_non_thumb_v1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    urdf_path = output_dir / "finger.urdf"
    text = urdf_path.read_text(encoding="utf-8")

    assert urdf_path.is_file()
    assert "finger preview [leap_non_thumb_v1]" in completed.stdout
    assert 'joint name="preview_finger_root_fixed" type="fixed"' in text
    assert 'link name="preview_finger_root_fixed_link"' in text
