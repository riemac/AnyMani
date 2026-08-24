r"""Schema 6 embodiment pretraining 的最高声明配置与唯一副作用入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import MISSING, OmegaConf

from .contracts import build_runtime

EMBODIMENT_PRETRAIN_SCHEMA_VERSION = "6.0.0"
"""显式 epoch/minibatch/microbatch 边界与 epoch-boundary resume。"""


class EmbodimentPretrain:
    r"""装配 data/method/trainer/run runtimes，并把完整实验生命周期交给 Trainer。"""

    def __init__(self, config: EmbodimentPretrainCfg, *, output_dir: Path | None = None) -> None:
        r"""只保存完整配置和可选测试输出目录；不解析资产、不初始化 CUDA。"""

        self.config = config
        self.output_dir = output_dir

    def run(self) -> Path:
        r"""构造四个 role runtime，并执行一次完整 pretraining 生命周期。"""

        self.config.validate_composed()  # 在任何 IO/CUDA 前拒绝缺失或错误的 component group
        print("[SSL] Building data runtime...")
        data = build_runtime(self.config.data)
        print("[SSL] Building method runtime...")
        method = build_runtime(self.config.method)
        print("[SSL] Building trainer runtime...")
        trainer = build_runtime(self.config.trainer)
        print("[SSL] Building run configuration...")
        run = build_runtime(self.config.run)
        print("[SSL] Starting training lifecycle...")
        return trainer.fit(
            data=data,
            method=method,
            run=run,
            output_dir_override=self.output_dir,
            resolved_config=resolved_config_dict(self.config),
        )


@dataclass(frozen=True)
class EmbodimentPretrainCfg:
    r"""Hydra 只组合 data、method、trainer 与 run 四个 concrete roles。

    四个槽位故意使用 ``Any``：OmegaConf 若把它们标为抽象基类，会在 structured compose 后丢失
    concrete dataclass 类型及 ``runtime_type``。科学兼容性由 method/trainer compile gate 检查，根配置
    不知道 Gaussian、hand、decoder 或 objective 字段。
    """

    runtime_type: ClassVar[type[EmbodimentPretrain]] = EmbodimentPretrain
    schema_version: str = EMBODIMENT_PRETRAIN_SCHEMA_VERSION
    data: Any = MISSING
    method: Any = MISSING
    trainer: Any = MISSING
    run: Any = MISSING

    def validate_composed(self) -> None:
        r"""验证 schema 与四个 role 都已由 concrete Python experiment 填充。"""

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


def resolved_config_dict(config: EmbodimentPretrainCfg) -> dict[str, Any]:
    r"""把 concrete structured config 解析为 checkpoint/YAML 使用的基础 mapping。"""

    container = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    if not isinstance(container, dict):
        raise TypeError("resolved embodiment pretraining config must be a mapping")
    return {str(key): value for key, value in container.items()}


__all__ = [
    "EMBODIMENT_PRETRAIN_SCHEMA_VERSION",
    "EmbodimentPretrain",
    "EmbodimentPretrainCfg",
    "resolved_config_dict",
]
