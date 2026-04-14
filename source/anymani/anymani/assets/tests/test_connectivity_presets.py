"""pre-made connectivity preset 与枚举主链回归测试。

这组测试锁住的是本轮新增的 pre-made 关键契约：

1. 合法注册的主体只覆盖 joint / child-link connectivity，不绑定 fingertip；
2. `HandGeneratorCfg` 继续作为唯一 façade，能够直接消费
   `hand_preset` / `connectivity_preset` / `hand_preset_names` / `connectivity_preset_names`；
3. `generate_batch(sampling_strategy="enumerate")` 已能枚举
   `base hand preset × connectivity preset` 的离散空间；
4. `preview_hand_preset.py` 现在可以带上 `--connectivity-preset` 直接导出
   pre-made 产物，形成最短人工巡检回路。
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import (
    get_hand_connectivity_preset_data,
    list_hand_connectivity_preset_names,
)


REPO_ROOT = Path(__file__).resolve().parents[5]  # 仓库根目录 `/home/hac/isaac/AnyMani`
PREVIEW_HAND_SCRIPT = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "presets" / "preview" / "preview_hand_preset.py"


def test_connectivity_registry_exposes_family_specific_full_alias():
    r"""`family_full` 应作为 hand-level full-chain alias 稳定存在。

    这里锁住的不是“名字好不好看”这种表层细节，而是更重要的 provenance 语义：

    - `allegro_full` / `leap_full` 要稳定可喊；
    - 它们内部仍应明确指向 thumb / non-thumb 的 full chain recipe；
    - 合法注册的数据主体不应夹带 tip 几何字段。
    """

    allegro_names = list_hand_connectivity_preset_names("allegro")
    preset = get_hand_connectivity_preset_data("allegro_full")

    assert "allegro_full" in allegro_names
    assert preset.family == "allegro"
    assert preset.finger_slots["thumb"] == "allegro_thumb_r4"
    assert preset.finger_slots["index"] == "allegro_non_thumb_r4"
    assert "tip" not in preset.metadata


def test_hand_generator_applies_connectivity_preset_and_exports_recursive_bundle(tmp_path):
    r"""带 connectivity preset 的单样本生成应走 recursive pre-made 输出。

    这里验证的核心是：

    $$
    \text{single\_palm\_allegro}
    \xrightarrow{\text{connectivity }\,t3/i2/m2/r2}
    \text{dof}=3+2+2+2=9
    $$

    同时，sidecar 里应显式保留 `base_hand_preset` 与 `connectivity_preset`。
    """

    result = HandGenerator(
        HandGeneratorCfg(
            mode="full",
            artifact_level="bundle",
            output_dir=tmp_path,
            hand_preset="single_palm_allegro",
            connectivity_preset="allegro_t3_i2_m2_r2",
            output_layout="recursive",
        )
    ).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.hand_cfg.dof_count == 9
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    assert result.urdf_path.parent.parent.name == "allegro_t3_i2_m2_r2"
    assert result.urdf_path.parent.parent.parent.name == "single_palm_allegro"
    assert result.urdf_path.parent.parent.parent.parent.name == "pre_made"

    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["base_hand_preset"] == "single_palm_allegro"
    assert sidecar["connectivity_preset"] == "allegro_t3_i2_m2_r2"
    assert sidecar["per_finger_connectivity"]["thumb"]["retained_revolute"] == 3
    assert sidecar["per_finger_connectivity"]["index"]["retained_revolute"] == 2


def test_hand_generator_enumerate_walks_registered_connectivity_space(tmp_path):
    r"""`generate_batch()` 的 enumerate 路线应显式遍历注册过的 connectivity preset。

    本测试不再接受旧的 “enumerate 尚未实现” 状态，而是锁住：

    - `hand_preset_names`
    - `connectivity_preset_names`
    - `max_enumerate`

    这三个 façade 字段已经能稳定驱动 pre-made 枚举。
    """

    generator = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="hand_cfg",
            output_dir=tmp_path,
            sampling_strategy="enumerate",
            hand_preset_names=("single_palm_allegro",),
            connectivity_preset_names=("allegro_full", "allegro_t3_i2_m2_r2"),
            max_enumerate=2,
        )
    )

    results = list(generator.generate_batch())
    connectivity_to_dof = {
        result.metadata["connectivity_preset"]: result.hand_cfg.dof_count
        for result in results
        if result.hand_cfg is not None
    }

    assert len(results) == 2
    assert connectivity_to_dof["allegro_full"] == 16
    assert connectivity_to_dof["allegro_t3_i2_m2_r2"] == 9


def test_preview_hand_script_accepts_connectivity_preset_and_writes_recursive_output(tmp_path):
    r"""`preview_hand_preset.py` 现在应支持 `--connectivity-preset` 直达 pre-made 预览。

    这里直接走用户最终会复制粘贴的命令行路径，而不是只测内部 API，
    因为科研侧真正关心的是“改 recipe 后立刻能看到 URDF”。
    """

    output_dir = tmp_path / "preview_outputs"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREVIEW_HAND_SCRIPT),
            "--hand-preset",
            "single_palm_allegro",
            "--connectivity-preset",
            "allegro_t3_i2_m2_r2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    written_urdfs = list(output_dir.rglob("hand.urdf"))
    assert len(written_urdfs) == 1
    assert "connectivity=allegro_t3_i2_m2_r2" in completed.stdout
    assert "single_palm_allegro" in written_urdfs[0].as_posix()
    assert "allegro_t3_i2_m2_r2" in written_urdfs[0].as_posix()
