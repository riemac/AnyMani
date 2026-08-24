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
from .experiments.multi_anchor_gaussion_implicit_field import (
    EVALUATION_EXPERIMENT,
    EXPERIMENT,
    VALIDATION_EXPERIMENT,
)
from .post_training import EmbodimentEvaluationCfg, EmbodimentValidationCfg

CANONICAL_EXPERIMENT_NAME = "multi_anchor_gaussian_implicit_field"
"""Hydra 根配置名；与 Python 实验模块的科学名称一致，不是日志目录名。"""

CANONICAL_VALIDATION_NAME = "multi_anchor_gaussian_implicit_field_validation"
"""独立 validation 的完整 Python 根配置名。"""

CANONICAL_EVALUATION_NAME = "multi_anchor_gaussian_implicit_field_evaluation"
"""独立 evaluation 的完整 Python 根配置名。"""


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
    r"""登记 pretrain、validation 与 evaluation 三份完整 Python 实验。"""

    store = ConfigStore.instance()
    store.store(name=CANONICAL_EXPERIMENT_NAME, node=_mutable_schema(EXPERIMENT))
    store.store(name=CANONICAL_VALIDATION_NAME, node=_mutable_schema(VALIDATION_EXPERIMENT))
    store.store(name=CANONICAL_EVALUATION_NAME, node=_mutable_schema(EVALUATION_EXPERIMENT))


def compose_pretrain_cfg(overrides: Sequence[str] = ()) -> EmbodimentPretrainCfg:
    r"""从 ConfigStore 恢复 schema 7 pure-pretrain 根配置，不读取 distill YAML。

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


def compose_validation_cfg(overrides: Sequence[str] = ()) -> EmbodimentValidationCfg:
    r"""从 ConfigStore 恢复独立 validation 根配置。"""

    register_pretraining_configs()
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=CANONICAL_VALIDATION_NAME, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentValidationCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentValidationCfg: {type(resolved)!r}")
    return resolved


def compose_evaluation_cfg(overrides: Sequence[str] = ()) -> EmbodimentEvaluationCfg:
    r"""从 ConfigStore 恢复独立 evaluation 根配置。"""

    register_pretraining_configs()
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=CANONICAL_EVALUATION_NAME, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentEvaluationCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentEvaluationCfg: {type(resolved)!r}")
    return resolved


__all__ = [
    "CANONICAL_EXPERIMENT_NAME",
    "CANONICAL_EVALUATION_NAME",
    "CANONICAL_VALIDATION_NAME",
    "compose_evaluation_cfg",
    "compose_pretrain_cfg",
    "compose_validation_cfg",
    "register_pretraining_configs",
]
