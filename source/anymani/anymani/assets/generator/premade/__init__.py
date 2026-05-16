r"""pre-made 子系统入口。

本子包汇总 pre-made 阶段的拓扑展开、connectivity 选择、identity 命名和
样本级并行调度工具。这里不放 `HandGeneratorCfg`，只放 façade 会复用的
阶段性 helper。
"""

from .batch import PremadeTask, PremadeWorkerResult, build_premade_tasks, run_premade_parallel, run_premade_serial
from .connectivity import (
    apply_connectivity_preset,
    connectivity_names_for_hand_preset,
    resolve_deleted_joint_names,
    resolve_single_premade_selection,
)
from .connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from .identity import resolve_export_root, stable_premade_id
from .normalize import normalize_connectivity_mapping, normalize_name_list
from .topology import (
    PremadeTopologySpec,
    build_base_hand,
    build_premade_topology_registry,
    candidate_hand_preset_names,
    extract_premade_topology_metadata,
    resolve_premade_topology_spec,
    slot_finger_kind,
)

__all__ = [
    "PremadeTask",
    "PremadeWorkerResult",
    "build_premade_tasks",
    "run_premade_parallel",
    "run_premade_serial",
    "apply_connectivity_preset",
    "connectivity_names_for_hand_preset",
    "resolve_deleted_joint_names",
    "resolve_single_premade_selection",
    "JointDeleteCfg",
    "JointDeleteMutator",
    "resolve_export_root",
    "stable_premade_id",
    "normalize_connectivity_mapping",
    "normalize_name_list",
    "PremadeTopologySpec",
    "build_base_hand",
    "build_premade_topology_registry",
    "candidate_hand_preset_names",
    "extract_premade_topology_metadata",
    "resolve_premade_topology_spec",
    "slot_finger_kind",
]
