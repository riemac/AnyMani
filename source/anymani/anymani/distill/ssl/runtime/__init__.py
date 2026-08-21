r"""Schema 4 embodiment pretraining runtime 的稳定公开接口。

resident window 与旧 geometry utilities 继续提供底层 tensor/lease 能力；最高训练入口由
``ssl.experiment.EmbodimentPretrain`` 组合，不在 runtime 包中恢复集中式 experiment config。
"""

from .evaluation import MultiAnchorEvaluation, MultiAnchorEvaluationCfg
from .pretrainer import EmbodimentPretrainTrainer, EmbodimentPretrainTrainerCfg
from .run import PretrainRun, PretrainRunCfg
from .sampling import (
    OnlineMinibatchSchedule,
    OnlineSamplingCfg,
    OnlineSamplingState,
    ScheduledMinibatch,
)
from .scheduler import GeometrySSLRuntimeCfg, GeometrySSLRuntimeState, ResidentGeometryAssetWindow

__all__ = [
    "EmbodimentPretrainTrainer",
    "EmbodimentPretrainTrainerCfg",
    "GeometrySSLRuntimeCfg",
    "GeometrySSLRuntimeState",
    "ResidentGeometryAssetWindow",
    "MultiAnchorEvaluation",
    "MultiAnchorEvaluationCfg",
    "OnlineMinibatchSchedule",
    "OnlineSamplingCfg",
    "OnlineSamplingState",
    "PretrainRun",
    "PretrainRunCfg",
    "ScheduledMinibatch",
]
