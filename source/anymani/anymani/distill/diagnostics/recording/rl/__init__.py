r"""RL 运行阶段、资源曲线与失败包络的结构化记录入口。

本包只记录 RL runtime 已经产生的事实，不启动 Isaac Sim、不创建策略，也不解释 reward。
``RlRunRecorder`` 由被测进程写阶段事件；父进程通过 ``read_linux_process_resources``
采样目标 PID，并把 NVML 口径的显存作为显式字段传入同一 recorder。
"""

from .cells import MorphologyCell, balanced_morphology_rows, morphology_cell_from_routing
from .runtime import RlRunRecorder, read_linux_process_resources, record_optional_rl_phase

__all__ = [
    "MorphologyCell",
    "RlRunRecorder",
    "balanced_morphology_rows",
    "morphology_cell_from_routing",
    "read_linux_process_resources",
    "record_optional_rl_phase",
]
