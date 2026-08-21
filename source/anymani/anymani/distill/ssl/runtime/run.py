r"""Embodiment pretraining 的 run identity、复现设置与 artifact 根目录。"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar


class PretrainRun:
    r"""只管理一次运行的定位与代码/package lineage，不拥有训练算法。"""

    def __init__(self, config: PretrainRunCfg) -> None:
        r"""保存 output/resume/reproducibility 声明；构造阶段不创建目录。"""

        self.config = config

    def prepare_output_dir(self, override: Path | None = None) -> Path:
        r"""创建本次唯一 artifact 根目录；测试可显式覆盖 timestamp 路径。"""

        output_dir = override
        if output_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path(self.config.output_dir) / self.config.experiment_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def code_revision() -> str:
        r"""尽力记录 Git HEAD；非 Git 安装显式返回 ``unknown``。"""

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return result.stdout.strip() or "unknown"

    @staticmethod
    def package_version() -> str:
        r"""读取 editable/installed AnyMani distribution version。"""

        try:
            return version("anymani")
        except PackageNotFoundError:
            return "editable-unknown"

    def checkpoint_metadata(
        self,
        *,
        geometry_semantics_schema: str,
        asset_manifest: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        declared_objective: Mapping[str, float],
        calibration_artifact_hash: str = "",
        worktree_dirty: bool = False,
        worktree_fingerprint: str = "",
    ) -> Any:
        r"""构造一次 run 共用的 schema 4 checkpoint lineage。"""

        from anymani.distill.ssl.checkpoint import GeometrySSLCheckpointMetadata

        return GeometrySSLCheckpointMetadata(
            code_revision=self.code_revision(),
            package_version=self.package_version(),
            geometry_semantics_schema=geometry_semantics_schema,
            asset_manifest=asset_manifest,
            resolved_config=resolved_config,
            declared_objective=declared_objective,
            calibration_artifact_hash=calibration_artifact_hash,
            worktree_dirty=worktree_dirty,
            worktree_fingerprint=worktree_fingerprint,
        )

    @staticmethod
    def save_full_checkpoint(path: Path, **payload: Any) -> None:
        r"""把 full resume payload 写出职责路由到 checkpoint owner。"""

        from anymani.distill.ssl.checkpoint import save_geometry_ssl_checkpoint

        save_geometry_ssl_checkpoint(path, **payload)

    @staticmethod
    def save_retained_artifact(path: Path, **payload: Any) -> None:
        r"""把 validation-best standalone export 路由到 checkpoint owner。"""

        from anymani.distill.ssl.checkpoint import save_retained_geometry_artifact

        save_retained_geometry_artifact(path, **payload)


@dataclass(frozen=True)
class PretrainRunCfg:
    r"""输出、resume 与全局确定性配置，不包含 optimizer 或科学采样预算。"""

    runtime_type: ClassVar[type[PretrainRun]] = PretrainRun
    output_dir: str = "logs/ssl"
    experiment_name: str = "multi_anchor_gaussian"
    resume_checkpoint: str = ""
    seed: int = 0  # model 初始化及各 role 派生 seed 的唯一根
    deterministic_algorithms: bool = True
    phase: str = "pretrain"  # `calibrate_objectives` 或 `pretrain`
    calibration_artifact: str = ""  # pretrain 可显式加载的 calibration YAML

    def __post_init__(self) -> None:
        r"""拒绝空 experiment identity、负随机种子和未知 phase。"""

        if not self.output_dir or not self.experiment_name:
            raise ValueError("pretraining run requires output_dir and experiment_name")
        if self.seed < 0:
            raise ValueError("pretraining run seed must be non-negative")
        if self.phase not in {"calibrate_objectives", "pretrain"}:
            raise ValueError("run.phase must be 'calibrate_objectives' or 'pretrain'")


__all__ = ["PretrainRun", "PretrainRunCfg"]
