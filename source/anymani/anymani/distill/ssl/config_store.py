r"""Embodiment pretraining 的 Hydra ConfigStore 注册。

完整实验由 Python façade 装配。Hydra 只登记该根配置并接受命令行覆盖；资产 YAML 只存在于
`assets/datasets/`，本模块不再检索任何 distill YAML。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from .experiment import EmbodimentPretrainCfg
from .experiments.multi_anchor_gaussion_implicit_field import EXPERIMENT

CANONICAL_EXPERIMENT_NAME = "multi_anchor_gaussian_implicit_field"
"""Hydra 根配置名；与 Python 实验模块的科学名称一致，不是日志目录名。"""


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
    r"""登记唯一的完整 Python 实验，供 CLI 与测试 compose。"""

    store = ConfigStore.instance()
    store.store(name=CANONICAL_EXPERIMENT_NAME, node=_mutable_schema(EXPERIMENT))


def compose_pretrain_cfg(overrides: Sequence[str] = ()) -> EmbodimentPretrainCfg:
    r"""从 ConfigStore 恢复 schema 5 根配置，不读取 distill YAML。

    Args:
        overrides (Sequence[str]): Hydra 点路径覆盖，例如 ``trainer.optimizer.learning_rate=1e-4``。

    Returns:
        EmbodimentPretrainCfg: 仍带 concrete role dataclass 的冻结根配置。
    """

    register_pretraining_configs()
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=CANONICAL_EXPERIMENT_NAME, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentPretrainCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentPretrainCfg: {type(resolved)!r}")
    return resolved


__all__ = [
    "CANONICAL_EXPERIMENT_NAME",
    "compose_pretrain_cfg",
    "register_pretraining_configs",
]
