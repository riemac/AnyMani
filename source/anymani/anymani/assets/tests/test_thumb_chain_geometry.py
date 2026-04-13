r"""thumb 几何链路回归测试。

这组测试不再满足于“thumb 能 build 出来、第二个关节不是零偏移”，而是直接把
`Thumb.png` 里的科研公式锁成代码契约。这样做的核心目的，是让后续你微调测量值时，
测试能区分：

1. 只是数值小幅改动；
2. 还是 `CMC1 -> CMC2` 这条真正关键的逻辑链路又被改坏了。
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

from assets.presets import get_finger_builder_preset


REPO_ROOT = Path(__file__).resolve().parents[5]
PREVIEW_FINGER_SCRIPT = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "presets" / "preview" / "preview_finger_preset.py"


def _assert_vec_close(actual, expected, *, tol: float = 1e-9) -> None:
    """逐分量比较向量，避免测试里到处手写 `isclose`。"""

    assert len(actual) == len(expected)
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=tol) for a, b in zip(actual, expected, strict=True))


def test_allegro_thumb_chain_matches_thumb_png_contract():
    r"""`allegro_thumb_v1` 应精确符合 `Thumb.png` 的链式几何关系。

    我们在这里直接把用户给出的 Allegro thumb 公式写成断言：

    - `CMC1` 轴为 $x$
    - `CMC2` 轴为 $y$
    - 其余 MCP / IP 为 $z$
    - `CMC2` origin:
      $$
      (x_1,y_1,z_1)=((w_{cmc1}-w)/2,\ d_{0y}+l_0/2,\ d_{0z}-(h_{cmc1}-h)/2)
      $$
    - 后续 joint:
      $$
      y_i=l_{i-1}+d_{i-1,y}
      $$
    - tip:
      $$
      y_{tip}=l_{N-1}+d_{N-1,y}
      $$
    """

    cfg = get_finger_builder_preset("allegro_thumb_v1").replace(name="thumb", parent_link="palm")
    finger = cfg.class_type(cfg).build()

    j0, j1, j2, j3, tip = finger.joints

    _assert_vec_close(j0.axis, (1.0, 0.0, 0.0))
    _assert_vec_close(j1.axis, (0.0, 1.0, 0.0))
    _assert_vec_close(j2.axis, (0.0, 0.0, 1.0))
    _assert_vec_close(j3.axis, (0.0, 0.0, 1.0))

    _assert_vec_close(j0.origin.pos, (0.0, 0.0, 0.0))
    _assert_vec_close(j1.origin.pos, (0.008, 0.0315, 0.011))
    _assert_vec_close(j2.origin.pos, (0.0, 0.015, 0.0))
    _assert_vec_close(j3.origin.pos, (0.0, 0.043, 0.0))
    _assert_vec_close(tip.origin.pos, (0.0, 0.031, 0.0))

    # `CMC1` 的 mesh frame 在零偏移时应与 joint frame 重合，因此这里直接锁住
    # 它采用显式 $(d_{0y}, d_{0z})$，而不是 regular link 的 “length/2 + d_y” 旧约。
    _assert_vec_close(j0.visuals[0].origin.pos, (0.0, 0.009, 0.0145))
    _assert_vec_close(j0.collisions[0].origin.pos, (0.0, 0.009, 0.0145))


def test_leap_thumb_preset_keeps_shared_thumb_axis_contract():
    r"""LEAP 虽暂不恢复 custom tip，但共享 thumb 轴语义仍必须一致。"""

    cfg = get_finger_builder_preset("leap_thumb_v1")
    assert cfg.axes == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]


def test_preview_finger_script_exports_allegro_thumb_urdf(tmp_path):
    r"""`preview_finger_preset.py --preset allegro_thumb_v1` 应稳定产出 thumb URDF。"""

    output_dir = tmp_path / "thumb_preview"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREVIEW_FINGER_SCRIPT),
            "--preset",
            "allegro_thumb_v1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    urdf_path = output_dir / "finger.urdf"
    assert urdf_path.is_file()
    assert "finger preview [allegro_thumb_v1]" in completed.stdout
