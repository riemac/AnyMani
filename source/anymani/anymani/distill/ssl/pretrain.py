r"""Schema 4 embodiment pretraining 的 Hydra CLI façade。

入口只登记 concrete ConfigStore schemas、组合 packaged YAML 并恢复 concrete dataclasses；资产、物理
teacher、method objective、optimizer 与 checkpoint 全部由对应 runtime role 拥有。

运行入口：``python -m anymani.distill.ssl.pretrain``。
"""

from __future__ import annotations

import hydra  # Geometry SSL 唯一命令行入口
from omegaconf import DictConfig, OmegaConf  # CLI composition 与 concrete object 恢复

from anymani.distill.ssl.config_store import register_pretraining_configs
from anymani.distill.ssl.experiment import EmbodimentPretrain, EmbodimentPretrainCfg

register_pretraining_configs()


@hydra.main(
    version_base="1.3",
    config_path="pkg://anymani.distill.presets.ssl",
    config_name="canonical_multi_anchor_gaussian",
)
def main(config: DictConfig) -> None:
    r"""解析全部 Hydra overrides，重建冻结配置后启动训练。

    Args:
        config (DictConfig): Hydra 合成的可变 mapping；进入 runtime 前必须重新执行全部
            dataclass 数值、路径与轴合同。
    """

    resolved = OmegaConf.to_object(config)
    if not isinstance(resolved, EmbodimentPretrainCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentPretrainCfg: {type(resolved)!r}")
    resolved.validate_composed()
    output_dir = EmbodimentPretrain(resolved).run()  # 唯一有副作用的 runtime 入口
    print(output_dir)  # shell/调度器唯一 stdout 结果：artifact root


if __name__ == "__main__":  # ``python -m anymani.distill.ssl.pretrain``
    main()


__all__ = ["main"]
