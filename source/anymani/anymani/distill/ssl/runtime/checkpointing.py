r"""Geometry SSL pure-pretrain runtime 的 resume 科学合同。

底层 tensor payload 的原子读写由 ``ssl.checkpoint`` 拥有；本模块只定义 runtime 必须恢复的
minibatch/Sobol/RNG 状态，并拒绝当前 CLI 与 checkpoint 之间的科学配置或数据身份漂移。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path  # immutable epoch checkpoint 与 mutable alias 发布路径

from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict


def resume_scientific_config(config: EmbodimentPretrainCfg | dict[str, object]) -> dict[str, object]:
    r"""返回 resume 必须一致的科学配置，只排除 output/resume 定位与源码变更授权开关。"""

    payload = deepcopy(resolved_config_dict(config) if isinstance(config, EmbodimentPretrainCfg) else dict(config))
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("resolved geometry SSL config lacks run mapping")
    payload["run"] = {
        key: value
        for key, value in run.items()
        if key
        not in {
            "output_dir",
            "experiment_name",
            "resume_checkpoint",
            "new_run",
            "allow_worktree_change",
            "extend_completed_run",
            "extension_source_package_version",
        }
    }  # seed/deterministic_algorithms 属于科学轨迹，只排除 artifact 定位字段
    return payload


def require_resume_scientific_config(
    current: EmbodimentPretrainCfg | dict[str, object],
    checkpoint_resolved: dict[str, object],
    *,
    allow_completed_budget_extension: bool = False,
    allow_experiment_identity_change: bool = False,
) -> None:
    r"""拒绝 scientific drift；显式 completed extension 只允许总 epoch 上界严格增加。"""

    schema = checkpoint_resolved.get("schema_version")
    if schema != "9.0.0":
        raise ValueError("resume checkpoint must contain schema 9 resolved configuration")
    expected = resume_scientific_config(checkpoint_resolved)
    actual = resume_scientific_config(current)
    if allow_experiment_identity_change:
        expected_identity = expected.pop("experiment_identity", None)
        actual_identity = actual.pop("experiment_identity", None)
        if not isinstance(expected_identity, dict) or not isinstance(actual_identity, dict):
            raise ValueError("experiment identity migration requires two identity mappings")
        if any(
            expected_identity.get(name) != actual_identity.get(name)
            for name in ("name", "module", "path")
        ):
            raise ValueError("experiment identity migration may change only snapshot sha256")
    if allow_completed_budget_extension:
        expected_trainer = expected.get("trainer")
        actual_trainer = actual.get("trainer")
        if not isinstance(expected_trainer, dict) or not isinstance(actual_trainer, dict):
            raise ValueError("completed budget extension requires trainer mappings")
        old_max_epochs = expected_trainer.get("max_epochs")
        new_max_epochs = actual_trainer.get("max_epochs")
        if not isinstance(old_max_epochs, int) or not isinstance(new_max_epochs, int):
            raise ValueError("completed budget extension requires integer max_epochs")
        if new_max_epochs <= old_max_epochs:
            raise ValueError("completed budget extension must strictly increase max_epochs")
        expected_trainer["max_epochs"] = new_max_epochs
    if actual != expected:
        changed_sections = tuple(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        raise ValueError(f"resume scientific config mismatch in sections={changed_sections}")


def require_resume_metadata_identity(
    current: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    allow_worktree_change: bool = False,
    extension_source_package_version: str = "",
) -> None:
    r"""拒绝代码/公式/source 漂移，只在显式授权时放行已验证源码修复的 code/worktree 变化。

    ``allow_worktree_change`` 是受控 recovery migration 开关，而不是科学配置覆盖。它允许研究者在
    已完成针对性验证的源码修复后继续同一 incomplete run，并放行该修复产生的 code revision 与
    worktree fingerprint 变化；仍要求 checkpoint 与当前进程都处于相同 dirty 状态，并继续严格比较
    package、geometry、objective、FairGrad、参数分区和 source artifact identity。这样不会把源码变化
    静默当成同一训练轨迹。
    """

    fields = (
        "geometry_semantics_schema",
        "declared_objective",
        "objective_formula",
        "fairgrad_formula",
        "parameter_partition",
        "source_artifact",
        "worktree_dirty",
    )
    if extension_source_package_version:
        if checkpoint.get("package_version") != extension_source_package_version:
            raise ValueError("resume checkpoint package_version does not match explicit extension source package")
    else:
        fields = ("package_version", *fields)
    changed = tuple(name for name in fields if current.get(name) != checkpoint.get(name))
    if not allow_worktree_change and current.get("code_revision") != checkpoint.get("code_revision"):
        changed += ("code_revision",)
    if current.get("worktree_dirty") != checkpoint.get("worktree_dirty"):
        changed += ("worktree_dirty",)
    if not allow_worktree_change and current.get("worktree_fingerprint") != checkpoint.get("worktree_fingerprint"):
        changed += ("worktree_fingerprint",)
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
