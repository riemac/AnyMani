"""Schema-9 training configuration and lifecycle entry point."""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import MISSING, OmegaConf

from .contracts import build_runtime

EMBODIMENT_PRETRAIN_SCHEMA_VERSION = "9.0.0"
"""Independent training and evaluation lifecycles with epoch recovery."""


class EmbodimentPretrain:


    def __init__(
        self,
        config: EmbodimentPretrainCfg,
        *,
        output_dir: Path | None = None,
        config_identity: Mapping[str, str] | None = None,
    ) -> None:


        self.config = config
        self.output_dir = output_dir
        self.config_identity = dict(config_identity or {})

    def run(self) -> Path:


        self.config.validate_composed()
        print("[SSL] Building data runtime...")
        data = build_runtime(self.config.data)
        print("[SSL] Building method runtime...")
        method = build_runtime(self.config.method)
        print("[SSL] Building trainer runtime...")
        trainer = build_runtime(self.config.trainer)
        print("[SSL] Building run configuration...")
        run = build_runtime(self.config.run)
        print("[SSL] Starting training lifecycle...")
        resolved_config = resolved_config_dict(self.config)
        if self.config_identity:
            resolved_config["experiment_identity"] = dict(self.config_identity)
        return trainer.fit(
            data=data,
            method=method,
            run=run,
            output_dir_override=self.output_dir,
            resolved_config=resolved_config,
        )


@dataclass(frozen=True)
class EmbodimentPretrainCfg:


    runtime_type: ClassVar[type[EmbodimentPretrain]] = EmbodimentPretrain
    schema_version: str = EMBODIMENT_PRETRAIN_SCHEMA_VERSION
    data: Any = MISSING
    method: Any = MISSING
    trainer: Any = MISSING
    run: Any = MISSING

    def validate_composed(self) -> None:


        if self.schema_version != EMBODIMENT_PRETRAIN_SCHEMA_VERSION:
            raise ValueError(f"embodiment pretraining schema must be exactly {EMBODIMENT_PRETRAIN_SCHEMA_VERSION}")
        missing = tuple(
            role
            for role in ("data", "method", "trainer", "run")
            if getattr(self, role) == MISSING or getattr(self, role) == "???"
        )
        if missing:
            raise ValueError(f"embodiment pretraining config is missing component roles: {missing}")
        invalid = tuple(
            role
            for role in ("data", "method", "trainer", "run")
            if not callable(getattr(type(getattr(self, role)), "runtime_type", None))
        )
        if invalid:
            raise TypeError(f"embodiment pretraining roles lack runtime_type bindings: {invalid}")


def resolved_config_dict(
    config: EmbodimentPretrainCfg, *, config_identity: Mapping[str, str] | None = None
) -> dict[str, Any]:


    container = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    if not isinstance(container, dict):
        raise TypeError("resolved embodiment pretraining config must be a mapping")
    resolved = {str(key): value for key, value in container.items()}
    if config_identity:
        resolved["experiment_identity"] = dict(config_identity)
    return resolved


__all__ = [
    "EMBODIMENT_PRETRAIN_SCHEMA_VERSION",
    "EmbodimentPretrain",
    "EmbodimentPretrainCfg",
    "resolved_config_dict",
]
