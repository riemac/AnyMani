r"""hand preset 与 hand preview 入口回归测试。

这组测试锁住三件直接面向科研调参的契约：

1. `hand_presets.py` 中的 hand preset 必须保持“整手组合锚点”可读；
2. 同一份 canonical hand preset 在 handedness 改成 left 时，应由
   `HumanLikeHandBuilder` 自动完成 thumb 的左右手唯一映射；
3. `preview_hand_preset.py` 必须支持只输入 hand preset 名，就直接产出
   可巡检的整手 URDF。

# NOTE:
这里故意同时测“纯 Python build 路径”和“python 脚本直跑路径”，因为用户的目标
并不只是内部 API 可用，更是要有一条最短反馈回路，能直接让 VS Code URDF viewer
看到结果。
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.presets import get_hand_builder_preset_data, make_human_like_builder_cfg_from_preset


REPO_ROOT = Path(__file__).resolve().parents[5]  # 仓库根目录 `/home/hac/isaac/AnyMani`
PREVIEW_HAND_SCRIPT = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "presets" / "preview" / "preview_hand_preset.py"


def test_hand_preset_registry_keeps_single_palm_allegro_combination_readable():
    r"""`single_palm_allegro` 应显式记录整手组合锚点，而不是隐式猜测。

    这里锁住的是“组合可读性”本身：

    - family 应是 `allegro`
    - palm 应是 `single_box_allegro`
    - 非拇指/拇指应分别指向 Allegro 对应 preset
    - handedness 默认应保留为 registry 内建值
    """

    preset = get_hand_builder_preset_data("single_palm_allegro")

    assert preset["name"] == "single_palm_allegro"
    assert preset["family"] == "allegro"
    assert preset["handedness"] == "right"
    assert preset["palm_cfg"] == "single_box_allegro"
    assert preset["finger_cfg"] == "allegro_non_thumb_v1"
    assert preset["thumb_cfg"] == "allegro_thumb_v1"
    assert preset["mirror_thumb_mount_for_left"] is True


def test_same_hand_preset_can_switch_left_and_right_via_builder_thumb_mapping():
    r"""同一份 hand preset 应允许通过 handedness 覆盖切到 left-hand。

    这里验证的不是 mount preset 层的字符串分支，而是更高层的科研语义：

    - hand preset 只保存一套 canonical thumb 锚点；
    - handedness 改成 `left` 后，由 hand builder 执行唯一映射；
    - non-thumb 在 palm frame 下保持不变；
    - thumb 的 $x$ 与 yaw 满足镜像关系。
    """

    right = HumanLikeHandBuilder(
        make_human_like_builder_cfg_from_preset(
            "single_palm_allegro",
            name="single_palm_allegro_right",
            handedness="right",
        )
    ).build()
    left = HumanLikeHandBuilder(
        make_human_like_builder_cfg_from_preset(
            "single_palm_allegro",
            name="single_palm_allegro_left",
            handedness="left",
        )
    ).build()

    right_index = next(finger for finger in right.fingers if finger.name == "index")
    left_index = next(finger for finger in left.fingers if finger.name == "index")
    right_thumb = next(finger for finger in right.fingers if finger.name == "thumb")
    left_thumb = next(finger for finger in left.fingers if finger.name == "thumb")

    assert left_index.mount.pos == right_index.mount.pos
    assert left_index.mount.rpy == right_index.mount.rpy
    assert math.isclose(left_thumb.mount.pos[0], -right_thumb.mount.pos[0], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(left_thumb.mount.pos[1], right_thumb.mount.pos[1], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(left_thumb.mount.pos[2], right_thumb.mount.pos[2], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(left_thumb.mount.rpy[2], -right_thumb.mount.rpy[2], rel_tol=0.0, abs_tol=1e-6)


def test_preview_hand_script_accepts_hand_preset_name_and_writes_urdf(tmp_path):
    r"""`preview_hand_preset.py` 应支持 `--hand-preset` 直达整手 quick-check。

    这个测试直接走用户最终会使用的命令行路径：

    $$
    \text{python preview\_hand\_preset.py --hand-preset ...}
    \xrightarrow{\text{HandGenerator}} \text{hand.urdf}
    $$

    如果这条链断掉，科研侧的“改 preset -> 立刻看 URDF”反馈回路就失效了。
    """

    output_dir = tmp_path / "preview_outputs"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREVIEW_HAND_SCRIPT),
            "--hand-preset",
            "single_palm_allegro",
            "--handedness",
            "left",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    written_urdfs = list(output_dir.rglob("hand.urdf"))  # `HandGenerator` 会在 hash 子目录下落 `hand.urdf`
    assert len(written_urdfs) == 1
    assert written_urdfs[0].is_file()
    assert "single_palm_allegro:left" in completed.stdout
