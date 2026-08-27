r"""Embodiment pretraining 的 Hydra ConfigStore 注册。

完整实验由 Python façade 装配。Hydra 只登记该根配置并接受命令行覆盖；资产 YAML 只存在于
`assets/datasets/`，本模块不再检索任何 distill YAML。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from .experiment import EmbodimentPretrainCfg
from .experiments import DEFAULT_EXPERIMENT_NAME, ExperimentPreset, available_experiments, load_experiment
from .post_training import EmbodimentEvaluationCfg, EmbodimentValidationCfg

CANONICAL_EXPERIMENT_NAME = DEFAULT_EXPERIMENT_NAME
"""D019--D035 schema-8 正式实验快照的 Hydra 根配置名。"""

CANONICAL_VALIDATION_NAME = "geometry_ssl_multitask_representation_v0_7_3_validation"
"""独立 validation 的完整 Python 根配置名。"""

CANONICAL_EVALUATION_NAME = "geometry_ssl_multitask_representation_v0_7_3_evaluation"
"""独立 evaluation 的完整 Python 根配置名。"""

LEGACY_EXPERIMENT_NAME = "multi_anchor_gaussian_implicit_field"
LEGACY_VALIDATION_NAME = "multi_anchor_gaussian_implicit_field_validation"
LEGACY_EVALUATION_NAME = "multi_anchor_gaussian_implicit_field_evaluation"
"""历史通用 preset 的注册名；版本化训练 CLI 不再以它作为默认实验。"""


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
    r"""登记 registry 中所有完整 Python 实验的 pretrain 与可选事后配置。"""

    store = ConfigStore.instance()
    for name in available_experiments():
        _register_preset(store, load_experiment(name))


def _register_preset(store: ConfigStore, preset: ExperimentPreset) -> tuple[str, str, str | None]:
    r"""把一个 registry entry 映射为 Hydra 的 pretrain/validation/evaluation 名称。"""

    store.store(name=preset.name, node=_mutable_schema(preset.pretrain))
    validation_name = f"{preset.name}_validation"
    evaluation_name = f"{preset.name}_evaluation"
    if preset.validation is not None:
        store.store(name=validation_name, node=_mutable_schema(preset.validation))
    if preset.evaluation is not None:
        store.store(name=evaluation_name, node=_mutable_schema(preset.evaluation))
    return preset.name, validation_name, evaluation_name if preset.evaluation is not None else None


def _register_selected_preset(preset: ExperimentPreset) -> tuple[str, str, str | None]:
    r"""为路径加载的快照建立不与 registry 冲突的临时 Hydra 名称。"""

    store = ConfigStore.instance()
    return _register_preset(store, preset)


def compose_pretrain_cfg(
    overrides: Sequence[str] = (), *, config_ref: str | Path = CANONICAL_EXPERIMENT_NAME
) -> EmbodimentPretrainCfg:
    r"""从 ConfigStore 恢复 schema 8 pure-pretrain 根配置，不读取 distill YAML。

    Args:
        overrides (Sequence[str]): Hydra 点路径覆盖，例如 ``trainer.optimizer.learning_rate=1e-4``。

    Returns:
        EmbodimentPretrainCfg: 仍带 concrete role dataclass 的冻结根配置。
    """

    preset = load_experiment(config_ref)
    config_name, _, _ = _register_selected_preset(preset)
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=config_name, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentPretrainCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentPretrainCfg: {type(resolved)!r}")
    return resolved


def compose_validation_cfg(
    overrides: Sequence[str] = (), *, config_ref: str | Path = CANONICAL_EXPERIMENT_NAME
) -> EmbodimentValidationCfg:
    r"""从 ConfigStore 恢复独立 validation 根配置。"""

    preset = load_experiment(config_ref)
    _, config_name, _ = _register_selected_preset(preset)
    if preset.validation is None:
        raise ValueError(f"experiment {preset.name!r} does not define a validation configuration")
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=config_name, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentValidationCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentValidationCfg: {type(resolved)!r}")
    return resolved


def compose_evaluation_cfg(
    overrides: Sequence[str] = (), *, config_ref: str | Path = CANONICAL_EXPERIMENT_NAME
) -> EmbodimentEvaluationCfg:
    r"""从 ConfigStore 恢复独立 evaluation 根配置。"""

    preset = load_experiment(config_ref)
    _, _, config_name = _register_selected_preset(preset)
    if preset.evaluation is None or config_name is None:
        raise ValueError(f"experiment {preset.name!r} does not define an evaluation configuration")
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=config_name, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentEvaluationCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentEvaluationCfg: {type(resolved)!r}")
    return resolved


__all__ = [
    "CANONICAL_EXPERIMENT_NAME",
    "CANONICAL_EVALUATION_NAME",
    "CANONICAL_VALIDATION_NAME",
    "LEGACY_EVALUATION_NAME",
    "LEGACY_EXPERIMENT_NAME",
    "LEGACY_VALIDATION_NAME",
    "compose_evaluation_cfg",
    "compose_pretrain_cfg",
    "compose_validation_cfg",
    "register_pretraining_configs",
]
