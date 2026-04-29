"""pre-made connectivity preset 与枚举主链回归测试。

这组测试锁住的是这轮重写后的 pre-made 关键契约：

1. connectivity registry 的主体是**显式 joint / child-link delete recipe**，
   不再把科研语义压扁成 `retained_revolute=k`；
2. `HandGeneratorCfg` 的 pre-made façade 直接暴露
   `hand_presets` 与 slot-level `connectivity_presets`；
3. pre-made 主线在 joint-centric 语义下应使用 `drop`：
   删除 joint 时同步删除其 child-link 几何，而不是 merge 回父段；
4. `generate_batch()` 已按固定 pre-made 语义显式遍历
   `base hand preset × slot-level connectivity pool` 的离散空间；
5. `preview_hand_preset.py` 现在仍可通过 `--connectivity-preset` 形成最短人工巡检回路。
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import (
    get_finger_connectivity_preset_data,
    get_hand_connectivity_preset_data,
    list_hand_connectivity_preset_names,
)


REPO_ROOT = Path(__file__).resolve().parents[6]  # 仓库根目录 `/home/hac/isaac/AnyMani`
PREVIEW_HAND_SCRIPT = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "presets" / "preview" / "preview_hand_preset.py"


def _slot_level_selection(hand_preset: str, connectivity_preset: str) -> dict[str, dict[str, list[str]]]:
    r"""把 hand-level alias 展开成当前生成器真正接受的 slot-level candidate pool。

    这个 helper 只服务测试迁移：科研侧平时仍然可以在 registry 层用
    `allegro_full` / `leap_t3_i3_m2_r2` 这类 alias 做人工查询，但 generator façade
    现在只接受 `hand_preset -> {slot -> [finger_connectivity_name]}`。
    """

    hand_connectivity = get_hand_connectivity_preset_data(connectivity_preset)
    return {
        hand_preset: {
            slot_name: [recipe_name]
            for slot_name, recipe_name in hand_connectivity.finger_slots.items()
        }
    }


def test_connectivity_registry_exposes_family_specific_full_alias():
    r"""`family_full` 应作为 hand-level full-chain alias 稳定存在。

    这里锁住的不是“名字好不好看”这种表层细节，而是更重要的 provenance 语义：

    - `allegro_full` / `leap_full` 要稳定可喊；
    - 它们内部仍应明确指向 thumb / non-thumb 的 full chain delete recipe；
    - finger-level registry 应显式写出 deleted joint 后缀，而不是只剩计数。
    """

    allegro_names = list_hand_connectivity_preset_names("allegro")
    preset = get_hand_connectivity_preset_data("allegro_full")
    drop_recipe = get_finger_connectivity_preset_data("allegro_non_thumb_drop_j2_j3")

    assert "allegro_full" in allegro_names
    assert preset.family == "allegro"
    assert preset.finger_slots["thumb"] == "allegro_thumb_full"
    assert preset.finger_slots["index"] == "allegro_non_thumb_full"
    assert preset.metadata["index_deleted_joint_suffixes"] == []
    assert drop_recipe.deleted_joint_suffixes == ("j2", "j3")
    assert drop_recipe.regroup_strategy == "drop"


def test_hand_generator_applies_connectivity_preset_and_drops_deleted_child_link_geometry(tmp_path):
    r"""带 connectivity preset 的单样本生成应删除被裁剪段的 child-link 几何。

    这里验证的核心是：

    $$
    \text{single\_palm\_leap}
    \xrightarrow{\text{connectivity }\,t3/i3/m2/r2,\ \text{drop geometry}}
    \text{dof}=3+3+2+2=10
    $$

    用户这轮指出的问题正是：旧实现把被删 joint 的 mesh merge 回父段，
    导致 URDF 里还能看到残留几何。这里直接锁住 joint-centric 语义：

    - `index_j2` / `index_j3` 必须从链里消失；
    - `index_j2_col` / `index_j3_col` 这类 child-link 几何也必须消失；
    - sidecar 必须明确记录这是一次 `drop` 语义的 connectivity 裁剪。
    """

    result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="bundle",
            output_dir=tmp_path,
            handedness="right",
            hand_presets=["single_palm_leap"],
            connectivity_presets=_slot_level_selection("single_palm_leap", "leap_t3_i3_m2_r2"),
            mixed=False,
            missing=False,
            output_layout="recursive",
        )
    ).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.hand_cfg.dof_count == 10
    assert result.urdf_path is not None and result.urdf_path.is_file()
    assert result.sidecar_path is not None and result.sidecar_path.is_file()
    assert result.urdf_path.parent.parent.name == "right_t3_i3_m2_r2"
    assert result.urdf_path.parent.parent.parent.name == "single_palm_leap"
    summary_path = result.urdf_path.parent.parent.parent.parent / "summary.yaml"
    assert summary_path.is_file()

    index = next(finger for finger in result.hand_cfg.fingers if finger.name == "index")
    surviving_joint_names = [joint.name for joint in index.joints]
    surviving_collision_names = [collision.name for joint in index.joints for collision in joint.collisions]

    assert surviving_joint_names == ["index_root_fixed", "index_j0", "index_j1", "index_j2", "index_tip"]
    assert "index_j3_col" not in surviving_collision_names

    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    assert sidecar["base_hand_preset"] == "single_palm_leap"
    assert sidecar["handedness"] == "right"
    assert sidecar["topology_group_name"] == "single_palm_leap"
    assert sidecar["topology_name"] == "right_t3_i3_m2_r2"
    assert sidecar["connectivity_preset"] == "thumb-drop_j3__index-drop_j3__middle-drop_j2_j3__ring-drop_j2_j3"
    assert sidecar["per_finger_connectivity"]["thumb"]["deleted_joints"] == ["thumb_j3"]
    assert sidecar["per_finger_connectivity"]["index"]["deleted_joint_suffixes"] == ["j3"]
    assert sidecar["per_finger_connectivity"]["index"]["deleted_joints"] == ["index_j3"]
    assert sidecar["per_finger_connectivity"]["index"]["regroup_strategy"] == "drop"
    assert sidecar["per_finger_connectivity"]["index"]["remaining_revolute"] == 3
    assert sidecar["hand_cfg"]["family"] == "leap"
    assert sidecar["hand_cfg"]["handedness"] == "right"
    assert sidecar["hand_cfg"]["metadata"]["premade_connectivity"]["base_hand_preset"] == "single_palm_leap"
    assert summary["run"]["mode"] == "made"
    assert summary["config"]["handedness"] == "right"
    assert summary["stats"]["succeeded"] == 1
    assert summary["stats"]["topology_count"] == 1


def test_hand_generator_enumerate_walks_registered_connectivity_space(tmp_path):
    r"""`generate_batch()` 的 enumerate 路线应显式遍历注册过的 connectivity preset。

    本测试不再接受旧的 “enumerate 尚未实现” 状态，而是锁住：

    - `hand_presets`
    - `connectivity_presets`
    - `max_enumerate`

    这三个 façade 字段已经能稳定驱动 pre-made 枚举。
    """

    generator = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="hand_cfg",
            output_dir=tmp_path,
            handedness="right",
            hand_presets=["single_palm_allegro"],
            mixed=False,
            missing=False,
            connectivity_presets={
                "single_palm_allegro": {
                    "thumb": ["allegro_thumb_full"],
                    "index": ["allegro_non_thumb_full"],
                    "middle": ["allegro_non_thumb_full"],
                    "ring": ["allegro_non_thumb_full", "allegro_non_thumb_drop_j2_j3"],
                }
            },
            max_enumerate=2,
        )
    )

    results = list(generator.generate_batch())
    topology_to_dof = {
        result.metadata["topology_name"]: result.hand_cfg.dof_count
        for result in results
        if result.hand_cfg is not None
    }

    assert len(results) == 2
    assert topology_to_dof["right_t4_i4_m4_r4"] == 16
    assert topology_to_dof["right_t4_i4_m4_r2"] == 14


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
            "--handedness",
            "right",
            "--connectivity-preset",
            "allegro_t3_i3_m2_r2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    written_urdfs = list(output_dir.rglob("hand.urdf"))
    assert len(written_urdfs) == 1
    assert "connectivity=allegro_t3_i3_m2_r2" in completed.stdout
    assert "single_palm_allegro" in written_urdfs[0].as_posix()
    assert "right_t3_i3_m2_r2" in written_urdfs[0].as_posix()
    assert len(list(output_dir.glob("*/summary.yaml"))) == 1
