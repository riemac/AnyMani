"""pre-made 子系统的兼容 re-export 壳。

当前真正的实现已经按职责拆成：

- `_premade_normalize.py`
- `_premade_topology.py`
- `_premade_connectivity.py`
- `_premade_identity.py`

这个文件只保留为内部兼容薄壳，避免一次重构同时改太多 import。
后续若确认没有剩余调用方，可以继续删掉本文件并让调用方直连子模块。
"""

from ._premade_connectivity import (
    apply_connectivity_preset,
    connectivity_names_for_hand_preset,
    resolve_deleted_joint_names,
    resolve_single_premade_selection,
)
from ._premade_identity import resolve_export_root, stable_premade_id
from ._premade_normalize import normalize_connectivity_mapping, normalize_name_list
from ._premade_topology import (
    PremadeTopologySpec,
    build_base_hand,
    build_premade_topology_registry,
    candidate_hand_preset_names,
    resolve_premade_topology_spec,
)

__all__ = [
    "PremadeTopologySpec",
    "apply_connectivity_preset",
    "build_base_hand",
    "build_premade_topology_registry",
    "candidate_hand_preset_names",
    "connectivity_names_for_hand_preset",
    "normalize_connectivity_mapping",
    "normalize_name_list",
    "resolve_deleted_joint_names",
    "resolve_export_root",
    "resolve_premade_topology_spec",
    "resolve_single_premade_selection",
    "stable_premade_id",
]
