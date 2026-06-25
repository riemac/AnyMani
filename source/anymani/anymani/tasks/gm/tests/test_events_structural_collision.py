r"""Pure contract tests for generated structural collision filtering.

这些测试不启动 Isaac Sim，也不 author USD stage。它们只锁住 AnyMani generated hand
结构碰撞过滤的 pair-level 科研语义：finger-palm 与 same-finger 内部碰撞被过滤，
不同 fingers 之间的碰撞保留。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

EVENTS_PATH = Path(__file__).resolve().parents[1] / "mdp" / "events.py"
r"""被测试的 events 源文件路径；用 path import 避免触发 `anymani.tasks.gm.mdp` 全量导入。"""


def _load_events_module() -> types.ModuleType:
    r"""加载 `events.py`，并 stub 掉本测试不需要的 IsaacLab runtime 类型。"""

    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    envs = types.ModuleType("isaaclab.envs")
    managers = types.ModuleType("isaaclab.managers")
    assets.RigidObject = object
    envs.ManagerBasedRLEnv = object
    managers.SceneEntityCfg = lambda name: types.SimpleNamespace(name=name)
    previous = {name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.assets", "isaaclab.envs", "isaaclab.managers")}
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.assets": assets,
            "isaaclab.envs": envs,
            "isaaclab.managers": managers,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("anymani.tasks.gm.mdp.events_for_test", EVENTS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclass string annotation resolution 需要模块已注册
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("anymani.tasks.gm.mdp.events_for_test", None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_generated_structural_collision_filter_keeps_cross_finger_collisions() -> None:
    r"""结构过滤应只移除 palm-finger 与 same-finger pair，不移除 cross-finger pair。"""

    module = _load_events_module()
    pairs = set(
        module.generated_structural_collision_filter_pairs(
            palm_link_name="palm",
            finger_link_chains=(
                ("index_root", "index_tip"),
                ("thumb_root", "thumb_tip"),
            ),
        )
    )

    assert tuple(sorted(("palm", "index_root"))) in pairs
    assert tuple(sorted(("index_root", "index_tip"))) in pairs
    assert tuple(sorted(("thumb_root", "thumb_tip"))) in pairs
    assert tuple(sorted(("index_tip", "thumb_tip"))) not in pairs
    assert tuple(sorted(("index_root", "thumb_root"))) not in pairs


def test_generated_structural_collision_filter_uses_filtered_pairs_api() -> None:
    r"""runtime authoring 应使用 pairwise `FilteredPairsAPI`，不要回退到 collision group。"""

    source = EVENTS_PATH.read_text(encoding="utf-8")

    assert "FilteredPairsAPI.Apply" in source
    assert '"PhysicsCollisionGroup"' not in source
    assert "CollectionAPI:colliders" not in source
    assert "CreateFilteredPairsRel" in source
