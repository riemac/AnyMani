r"""RL 运行阶段、资源曲线与失败包络的结构化记录入口。

本包只记录 RL runtime 已经产生的事实，不启动 Isaac Sim、不创建策略，也不解释 reward。
``RlRunRecorder`` 由被测进程写阶段事件；父进程通过 ``read_linux_process_resources``
采样目标 PID，并把 NVML 口径的显存作为显式字段传入同一 recorder。
"""

from .cells import MorphologyCell, balanced_morphology_rows, morphology_cell_from_routing
from .n000_teacher import (
    N000StudentActorPacket,
    build_n000_student_actor_packet,
    canonical_from_native_indices,
    contact_bits_to_owner,
    native_from_canonical_indices,
    sensor_owner_indices_from_sidecar,
)
from .runtime import (
    FATAL_RUNTIME_PATTERNS,
    RlRunRecorder,
    read_linux_process_resources,
    record_optional_rl_phase,
    scan_appended_fatal_log,
)

__all__ = [
    "FATAL_RUNTIME_PATTERNS",
    "MorphologyCell",
    "N000StudentActorPacket",
    "RlRunRecorder",
    "balanced_morphology_rows",
    "build_n000_student_actor_packet",
    "canonical_from_native_indices",
    "contact_bits_to_owner",
    "morphology_cell_from_routing",
    "read_linux_process_resources",
    "record_optional_rl_phase",
    "scan_appended_fatal_log",
    "native_from_canonical_indices",
    "sensor_owner_indices_from_sidecar",
]
