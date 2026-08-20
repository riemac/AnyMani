r"""Embodiment pretraining concrete Hydra groups 的类型注册。

该模块只把稳定配置名称关联到 dataclass schema。runtime 构造仍由每个 cfg 的 ``runtime_type``
完成；新增组件不需要在这里编写字段解析、构造参数或跨组件连接逻辑。
"""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.objectives.representations.geometry_terms import (
    ChainObjectiveTermCfg,
    DensityObjectiveTermCfg,
    DerivedFieldObjectiveTermCfg,
    KappaObjectiveTermCfg,
    PairedParityObjectiveTermCfg,
    SobolevObjectiveTermCfg,
)
from anymani.distill.representations.geometry import GeometryRepresentationCfg

from .experiment import EmbodimentPretrainCfg
from .experiments.multi_anchor_gaussion_implicit_field import EXPERIMENT
from .methods import MultiAnchorGaussianMethodCfg
from .runtime.evaluation import MultiAnchorEvaluationCfg
from .runtime.pretrainer import EmbodimentPretrainTrainerCfg
from .runtime.run import PretrainRunCfg


def _mutable_schema(config: Any) -> DictConfig:
    r"""保留 dataclass object type，但允许 Hydra defaults 在 compose 期间写入字段。

    ``frozen=True`` 是 runtime 科研合同；Hydra compose 则需要一个短暂可合并的 DictConfig view。
    最终 ``OmegaConf.to_object`` 仍按原 dataclass 类型构造冻结对象。
    """

    node = OmegaConf.structured(config)

    def thaw(value: Any) -> None:
        r"""递归解除 nested frozen dataclass nodes 的 compose-time readonly 标志。"""

        if isinstance(value, (DictConfig, ListConfig)):
            OmegaConf.set_readonly(value, False)
            if isinstance(value, DictConfig):
                children = (value._get_node(key) for key in value.keys())  # 不求值 MISSING role slots
            else:
                children = (value._get_node(index) for index in range(len(value)))
            for child in children:
                thaw(child)

    thaw(node)
    return node


def register_pretraining_configs() -> None:
    r"""注册 schema 3 根与当前 concrete component groups。"""

    store = ConfigStore.instance()
    store.store(name="embodiment_pretrain_schema", node=_mutable_schema(EmbodimentPretrainCfg))
    store.store(name="multi_anchor_gaussian_implicit_field", node=_mutable_schema(EXPERIMENT))
    store.store(group="method", name="schema_multi_anchor_gaussian", node=_mutable_schema(MultiAnchorGaussianMethodCfg))
    store.store(
        group="representation",
        name="schema_multi_anchor_gaussian",
        node=_mutable_schema(GeometryRepresentationCfg),
    )
    store.store(group="model", name="schema_multi_anchor_gaussian", node=_mutable_schema(GeometrySSLModelCfg))
    store.store(group="objective_term", name="schema_density", node=_mutable_schema(DensityObjectiveTermCfg))
    store.store(group="objective_term", name="schema_kappa", node=_mutable_schema(KappaObjectiveTermCfg))
    store.store(
        group="objective_term",
        name="schema_derived_field",
        node=_mutable_schema(DerivedFieldObjectiveTermCfg),
    )
    store.store(group="objective_term", name="schema_sobolev", node=_mutable_schema(SobolevObjectiveTermCfg))
    store.store(group="objective_term", name="schema_chain", node=_mutable_schema(ChainObjectiveTermCfg))
    store.store(group="objective_term", name="schema_paired", node=_mutable_schema(PairedParityObjectiveTermCfg))
    store.store(group="trainer", name="schema_single_gpu_16gb", node=_mutable_schema(EmbodimentPretrainTrainerCfg))
    store.store(
        group="evaluation",
        name="schema_multi_anchor_gaussian",
        node=_mutable_schema(MultiAnchorEvaluationCfg),
    )
    store.store(
        group="run",
        name="schema_canonical_multi_anchor_gaussian",
        node=_mutable_schema(PretrainRunCfg),
    )


__all__ = ["register_pretraining_configs"]
