r"""Geometry SSL 的 Hydra CLI façade。

该入口只负责注册声明式 experiment、把 Hydra mapping 重建为冻结 dataclass，并调用
``ssl.runtime.trainer.run_geometry_ssl_pretraining``。资产解析、resident window、目标、validation、
checkpoint 与训练循环都属于 runtime 子包，不能重新堆回 CLI 文件。

运行入口：

``python -m anymani.distill.ssl.pretrain --config-name geometry_ssl_canonical_residual_family``
"""

from __future__ import annotations

import hydra  # Geometry SSL 唯一命令行入口
from hydra.core.config_store import ConfigStore  # 注册默认配置与完整 experiment mapping
from omegaconf import DictConfig, OmegaConf  # CLI override resolve 与基础容器转换

from anymani.distill.ssl.config import (
    GeometrySSLExperimentCfg,
    experiment_config_from_dict,
    resolved_config_dict,
)
from anymani.distill.ssl.experiments import experiment_configs
from anymani.distill.ssl.runtime.trainer import run_geometry_ssl_pretraining

ConfigStore.instance().store(
    name="geometry_ssl",
    node=resolved_config_dict(GeometrySSLExperimentCfg()),
)  # 底层默认配置只提供显式 path override 入口
for experiment_name, experiment_config in experiment_configs().items():
    ConfigStore.instance().store(
        name=experiment_name,
        node=resolved_config_dict(experiment_config),
    )  # 每个 experiment 注册完整、声明式 resolved mapping


@hydra.main(version_base="1.3", config_name="geometry_ssl")
def main(config: DictConfig) -> None:
    r"""解析全部 Hydra overrides，重建冻结配置后启动训练。

    Args:
        config (DictConfig): Hydra 合成的可变 mapping；进入 runtime 前必须重新执行全部
            dataclass 数值、路径与轴合同。
    """

    payload = OmegaConf.to_container(config, resolve=True)  # interpolation 全部求值
    if not isinstance(payload, dict):  # 根配置必须是 mapping
        raise TypeError("resolved Hydra geometry SSL config must be a mapping")
    normalized_payload = {str(key): value for key, value in payload.items()}  # 收窄 DictKeyType
    resolved = experiment_config_from_dict(normalized_payload)  # 逐层 dataclass 验证
    output_dir = run_geometry_ssl_pretraining(resolved)  # runtime 执行完整科研生命周期
    print(output_dir)  # shell/调度器唯一 stdout 结果：artifact root


if __name__ == "__main__":  # ``python -m anymani.distill.ssl.pretrain``
    main()


__all__ = ["main", "run_geometry_ssl_pretraining"]
