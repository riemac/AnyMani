r"""Geometry SSL pure-pretrain runtime 的 resume 科学合同。

底层 tensor payload 的原子读写由 ``ssl.checkpoint`` 拥有；本模块只定义 runtime 必须恢复的
minibatch/Sobol/RNG 状态，并拒绝当前 CLI 与 checkpoint 之间的科学配置或数据身份漂移。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path  # immutable epoch checkpoint 与 mutable alias 发布路径

from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict


def resume_scientific_config(config: EmbodimentPretrainCfg | dict[str, object]) -> dict[str, object]:
    r"""返回 resume 必须一致的科学配置，只排除 output/resume 定位。"""

    payload = resolved_config_dict(config) if isinstance(config, EmbodimentPretrainCfg) else dict(config)
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("resolved geometry SSL config lacks run mapping")
    payload["run"] = {
        key: value for key, value in run.items() if key not in {"output_dir", "experiment_name", "resume_checkpoint"}
    }  # seed/deterministic_algorithms 属于科学轨迹，只排除 artifact 定位字段
    return payload


def require_resume_scientific_config(
    current: EmbodimentPretrainCfg | dict[str, object],
    checkpoint_resolved: dict[str, object],
) -> None:
    r"""拒绝当前 CLI 与 checkpoint 的任一 scientific config 漂移。"""

    schema = checkpoint_resolved.get("schema_version")
    if schema != "8.0.0":
        raise ValueError("resume checkpoint must contain schema 8 resolved configuration")
    expected = resume_scientific_config(checkpoint_resolved)
    actual = resume_scientific_config(current)
    if actual != expected:
        changed_sections = tuple(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        raise ValueError(f"resume scientific config mismatch in sections={changed_sections}")


def require_resume_metadata_identity(
    current: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> None:
    r"""拒绝配置无法表达的代码、公式、参数分区、source producer 或 worktree 漂移。"""

    fields = (
        "code_revision",
        "package_version",
        "geometry_semantics_schema",
        "declared_objective",
        "objective_formula",
        "fairgrad_formula",
        "parameter_partition",
        "source_artifact",
        "worktree_dirty",
        "worktree_fingerprint",
    )
    changed = tuple(name for name in fields if current.get(name) != checkpoint.get(name))
    if changed:
        raise ValueError(f"resume checkpoint metadata identity mismatch in fields={changed}")


def publish_checkpoint_alias(alias_path: Path, immutable_path: Path) -> None:
    r"""把 immutable checkpoint 以同文件系统原子 hard-link alias 发布。"""

    if not immutable_path.is_file():
        raise FileNotFoundError(f"immutable checkpoint does not exist: {immutable_path}")
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias_path.with_suffix(alias_path.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    temporary.hardlink_to(immutable_path)  # 同目录同文件系统，共享 checkpoint inode
    temporary.replace(alias_path)


__all__ = [
    "publish_checkpoint_alias",
    "require_resume_metadata_identity",
    "require_resume_scientific_config",
]
