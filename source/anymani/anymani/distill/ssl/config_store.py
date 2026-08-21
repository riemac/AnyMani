r"""Embodiment pretraining 的 Hydra 根 schema 注册。

完整实验由 Python façade 装配。Hydra 只登记根配置并接受命令行覆盖，不再用分片 YAML
拼装 method、model 或 objective。
"""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from .experiment import EmbodimentPretrainCfg
from .experiments.multi_anchor_gaussion_implicit_field import EXPERIMENT


def _mutable_schema(config: Any) -> DictConfig:
    r"""保留 dataclass object type，但允许 Hydra 在 compose 期间写入覆盖字段。"""

    node = OmegaConf.structured(config)

    def thaw(value: Any) -> None:
        r"""递归解除 nested frozen dataclass nodes 的 compose-time readonly 标志。"""

        if isinstance(value, (DictConfig, ListConfig)):
            OmegaConf.set_readonly(value, False)
            if isinstance(value, DictConfig):
                children = (value._get_node(key) for key in value.keys())
            else:
                children = (value._get_node(index) for index in range(len(value)))
            for child in children:
                thaw(child)

    thaw(node)
    return node


def register_pretraining_configs() -> None:
    r"""注册 schema 根与完整 Python 实验。"""

    store = ConfigStore.instance()
    store.store(name="embodiment_pretrain_schema", node=_mutable_schema(EmbodimentPretrainCfg))
    store.store(name="multi_anchor_gaussian_implicit_field", node=_mutable_schema(EXPERIMENT))
    store.store(name="canonical_multi_anchor_gaussian", node=_mutable_schema(EXPERIMENT))


__all__ = ["register_pretraining_configs"]
