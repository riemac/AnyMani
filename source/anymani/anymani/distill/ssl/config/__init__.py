r"""SSL stage 级组合配置公开接口。

物理对象、learnable component 与 objective 的局部配置仍由各自 owner 包定义；本包只组合
data split、target/query、训练预算、optimizer、precision、checkpoint 和 evaluation protocol，
并验证 Sobolev、derived-gradient 与带宽轴等跨组件科学约束。当前公开 geometry SSL 的冻结 dataclass、Hydra mapping bridge 与 resolved artifact writer。
"""

from .geometry_ssl import (
    GeometrySSLAssetCfg,
    GeometrySSLAssetManifest,
    GeometrySSLExperimentCfg,
    GeometrySSLOptimizerCfg,
    GeometrySSLTrainLoopCfg,
    experiment_config_from_dict,
    resolved_config_dict,
    write_resolved_experiment_files,
)

__all__ = [
    "GeometrySSLAssetCfg",
    "GeometrySSLAssetManifest",
    "GeometrySSLExperimentCfg",
    "GeometrySSLOptimizerCfg",
    "GeometrySSLTrainLoopCfg",
    "experiment_config_from_dict",
    "resolved_config_dict",
    "write_resolved_experiment_files",
]
