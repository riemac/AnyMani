"""Register and compose the selected versioned SSL configuration."""


from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from .experiment import EmbodimentPretrainCfg
from .experiments import DEFAULT_EXPERIMENT_NAME, ExperimentPreset, available_experiments, load_experiment
from .post_training import EmbodimentEvaluationCfg

CANONICAL_EXPERIMENT_NAME = DEFAULT_EXPERIMENT_NAME
"""Hydra root for the schema-9 geometry pretraining snapshot."""

CANONICAL_EVALUATION_NAME = "geometry_ssl_multitask_representation_v0_7_5_evaluation"
"""Python root configuration for independent evaluation."""

def _mutable_schema(config: Any) -> DictConfig:


    node = OmegaConf.structured(config)

    def thaw(value: Any) -> None:


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


    store = ConfigStore.instance()
    for name in available_experiments():
        _register_preset(store, load_experiment(name))


def _register_preset(store: ConfigStore, preset: ExperimentPreset) -> tuple[str, str | None]:


    store.store(name=preset.name, node=_mutable_schema(preset.pretrain))
    evaluation_name = f"{preset.name}_evaluation"
    if preset.evaluation is not None:
        store.store(name=evaluation_name, node=_mutable_schema(preset.evaluation))
    return preset.name, evaluation_name if preset.evaluation is not None else None


def _register_selected_preset(preset: ExperimentPreset) -> tuple[str, str | None]:


    store = ConfigStore.instance()
    return _register_preset(store, preset)


def compose_pretrain_cfg(
    overrides: Sequence[str] = (), *, config_ref: str | Path = CANONICAL_EXPERIMENT_NAME
) -> EmbodimentPretrainCfg:


    preset = load_experiment(config_ref)
    config_name, _ = _register_selected_preset(preset)
    with initialize(version_base="1.3", config_path=None):
        composed = compose(config_name=config_name, overrides=list(overrides))
    resolved = OmegaConf.to_object(composed)
    if not isinstance(resolved, EmbodimentPretrainCfg):
        raise TypeError(f"Hydra root did not restore EmbodimentPretrainCfg: {type(resolved)!r}")
    return resolved


def compose_evaluation_cfg(
    overrides: Sequence[str] = (), *, config_ref: str | Path = CANONICAL_EXPERIMENT_NAME
) -> EmbodimentEvaluationCfg:


    preset = load_experiment(config_ref)
    _, config_name = _register_selected_preset(preset)
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
    "compose_evaluation_cfg",
    "compose_pretrain_cfg",
    "register_pretraining_configs",
]
