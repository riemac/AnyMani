r"""Geometry SSL 事后 validation/evaluation 的独立声明配置与 façade。

训练配置不包含本模块的任何字段。两个阶段只消费已经完成的 schema-7 full checkpoint，
不会创建 optimizer、改变参数或回写源训练目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import MISSING, OmegaConf

from .contracts import build_runtime

VALIDATION_SCHEMA_VERSION = "1.0.0"
"""独立 validation 配置与结果合同的首个稳定版本。"""

EVALUATION_SCHEMA_VERSION = "1.0.0"
"""独立 evaluation 配置与结果合同的首个稳定版本。"""


@dataclass(frozen=True)
class ValidationCfg:
    r"""固定 validation bank、teacher-baseline selection 和 GPU 资源配置。"""

    q_per_asset: int = 64
    assets_per_minibatch: int = 2
    q_per_asset_per_minibatch: int = 2
    selection_metrics: tuple[str, ...] = ("density", "kappa")
    seed_offset: int = 1_000_003
    max_resident_assets: int = 64
    device: str = "cuda:0"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        r"""验证固定 q-bank、selection 指标和资源上限。"""

        object.__setattr__(self, "selection_metrics", tuple(self.selection_metrics))
        counts = (
            self.q_per_asset,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
            self.seed_offset,
            self.max_resident_assets,
        )
        if min(counts) < 1 or not self.selection_metrics:
            raise ValueError("validation q/batch/seed/resource values and selection metrics must be non-empty")
        if self.max_resident_assets < self.assets_per_minibatch:
            raise ValueError("validation max_resident_assets must cover one asset minibatch")
        _validate_cuda_float32(self.device, self.dtype, role="validation")


@dataclass(frozen=True)
class EvaluationCfg:
    r"""冻结 checkpoint 后的 unseen suites、可选训练 q-bank 与消融配置。"""

    q_per_asset: int = 64
    assets_per_minibatch: int = 2
    q_per_asset_per_minibatch: int = 2
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "joint_token_shuffle",
    )
    selection_metrics: tuple[str, ...] = ("density", "kappa")
    bootstrap_replicates: int = 2_000
    evaluation_seed_offset: int = 2_000_003
    training_q_bank_seed_offset: int = 3_000_003
    bootstrap_seed_offset: int = 4_000_003
    max_resident_assets: int = 64
    device: str = "cuda:0"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        r"""验证固定测度预算、消融集合与互不重叠的随机域。"""

        object.__setattr__(self, "final_ablations", tuple(self.final_ablations))
        object.__setattr__(self, "selection_metrics", tuple(self.selection_metrics))
        counts = (
            self.q_per_asset,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
            self.bootstrap_replicates,
            self.max_resident_assets,
        )
        offsets = (
            self.evaluation_seed_offset,
            self.training_q_bank_seed_offset,
            self.bootstrap_seed_offset,
        )
        if min(counts) < 1 or not self.final_ablations or not self.selection_metrics:
            raise ValueError("evaluation q/batch/bootstrap/resource budgets and metric sets must be non-empty")
        if min(offsets) < 1 or len(set(offsets)) != len(offsets):
            raise ValueError("evaluation seed offsets must be positive and distinct")
        if self.max_resident_assets < self.assets_per_minibatch:
            raise ValueError("evaluation max_resident_assets must cover one asset minibatch")
        _validate_cuda_float32(self.device, self.dtype, role="evaluation")


def _validate_cuda_float32(device: str, dtype: str, *, role: str) -> None:
    r"""保持 Warp fixed-bank 路径的 CUDA float32 资源合同。"""

    if not (device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError(f"{role} device must be 'cuda' or 'cuda:<index>'")
    if dtype != "float32":
        raise ValueError(f"current Warp {role} path requires dtype='float32'")


class ValidationRun:
    r"""管理一次显式 validation 的输入 checkpoint 与独立 artifact 目录。"""

    def __init__(self, config: ValidationRunCfg) -> None:
        r"""保存运行声明；构造阶段不访问 checkpoint 或文件系统。"""

        self.config = config

    def resolve_output_dir(self, override: Path | None = None) -> Path:
        r"""只解析 validation 输出路径；安全 gate 通过前不创建目录。"""

        output_dir = override
        if output_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path(self.config.output_dir) / self.config.experiment_name / timestamp
        return output_dir.expanduser().resolve(strict=False)

    def prepare_output_dir(self, override: Path | None = None) -> Path:
        r"""创建 validation 独占目录，不复用或回写训练 run。"""

        output_dir = self.resolve_output_dir(override)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


@dataclass(frozen=True)
class ValidationRunCfg:
    r"""显式 baseline/candidate 列表和 validation 复现随机种子。"""

    runtime_type: ClassVar[type[ValidationRun]] = ValidationRun
    output_dir: str = "logs/ssl"
    experiment_name: str = "canonical_multi_anchor_gaussian_validation"
    baseline_checkpoint: str = ""
    checkpoints: tuple[str, ...] = ()
    seed: int = 20260813
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        r"""规范候选 tuple，并验证不依赖 checkpoint IO 的运行字段。"""

        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if not self.output_dir or not self.experiment_name or self.seed < 0:
            raise ValueError("validation run requires output identity and non-negative seed")

    def validate_inputs(self) -> None:
        r"""在执行边界要求一个 baseline 和至少一个显式候选 checkpoint。"""

        if not self.baseline_checkpoint or not self.checkpoints:
            raise ValueError("validation requires --baseline_checkpoint and at least one --checkpoint")
        normalized = tuple(str(Path(path).expanduser().resolve()) for path in self.checkpoints)
        if len(set(normalized)) != len(normalized):
            raise ValueError("validation checkpoint list must not contain duplicate paths")


class EvaluationRun:
    r"""管理一次显式 evaluation 的目标 checkpoint、可选 baseline 与输出目录。"""

    def __init__(self, config: EvaluationRunCfg) -> None:
        r"""保存运行声明；构造阶段不访问 checkpoint 或 CUDA。"""

        self.config = config

    def resolve_output_dir(self, override: Path | None = None) -> Path:
        r"""只解析 evaluation 输出路径；安全 gate 通过前不创建目录。"""

        output_dir = override
        if output_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path(self.config.output_dir) / self.config.experiment_name / timestamp
        return output_dir.expanduser().resolve(strict=False)

    def prepare_output_dir(self, override: Path | None = None) -> Path:
        r"""创建 evaluation 独占目录。"""

        output_dir = self.resolve_output_dir(override)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


@dataclass(frozen=True)
class EvaluationRunCfg:
    r"""目标 checkpoint、可选 epoch-0 baseline 与 evaluation 随机种子。"""

    runtime_type: ClassVar[type[EvaluationRun]] = EvaluationRun
    output_dir: str = "logs/ssl"
    experiment_name: str = "canonical_multi_anchor_gaussian_evaluation"
    checkpoint: str = ""
    baseline_checkpoint: str = ""
    seed: int = 20260813
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        r"""验证输出身份和随机种子；checkpoint 必填性延迟到 CLI 执行边界。"""

        if not self.output_dir or not self.experiment_name or self.seed < 0:
            raise ValueError("evaluation run requires output identity and non-negative seed")

    def validate_inputs(self) -> None:
        r"""要求一个显式目标 checkpoint；baseline 始终可选。"""

        if not self.checkpoint:
            raise ValueError("evaluation requires --checkpoint")


class EmbodimentValidation:
    r"""装配 data/method/validation/run，并执行一次独立 checkpoint selection。"""

    def __init__(self, config: EmbodimentValidationCfg, *, output_dir: Path | None = None) -> None:
        r"""保存完整配置和可选测试输出目录，不初始化 CUDA。"""

        self.config = config
        self.output_dir = output_dir

    def run(self) -> Path:
        r"""构造四个 role runtime，并交给独立 validation 内核。"""

        self.config.validate_composed()
        data = build_runtime(self.config.data)
        method = build_runtime(self.config.method)
        run = build_runtime(self.config.run)
        from .runtime.post_training import validate_checkpoints

        return validate_checkpoints(
            data=data,
            method=method,
            config=self.config.validation,
            run=run,
            output_dir_override=self.output_dir,
            resolved_config=resolved_post_training_config_dict(self.config),
        )


@dataclass(frozen=True)
class EmbodimentValidationCfg:
    r"""独立 validation 的 data/method/validation/run 四角色根配置。"""

    schema_version: str = VALIDATION_SCHEMA_VERSION
    data: Any = MISSING
    method: Any = MISSING
    validation: ValidationCfg = MISSING
    run: Any = MISSING

    def validate_composed(self) -> None:
        r"""验证 schema、运行输入与所有 concrete roles。"""

        _validate_root(self, schema=VALIDATION_SCHEMA_VERSION, roles=("data", "method", "run"))
        if not isinstance(self.validation, ValidationCfg):
            raise TypeError("validation root requires a concrete ValidationCfg")
        self.run.validate_inputs()


class EmbodimentEvaluation:
    r"""装配 data/method/evaluation/run，并执行一次独立 held-out evaluation。"""

    def __init__(self, config: EmbodimentEvaluationCfg, *, output_dir: Path | None = None) -> None:
        r"""保存完整配置和可选测试输出目录，不读取 checkpoint。"""

        self.config = config
        self.output_dir = output_dir

    def run(self) -> Path:
        r"""构造四个 role runtime，并交给独立 evaluation 内核。"""

        self.config.validate_composed()
        data = build_runtime(self.config.data)
        method = build_runtime(self.config.method)
        run = build_runtime(self.config.run)
        from .runtime.post_training import evaluate_checkpoint

        return evaluate_checkpoint(
            data=data,
            method=method,
            config=self.config.evaluation,
            run=run,
            output_dir_override=self.output_dir,
            resolved_config=resolved_post_training_config_dict(self.config),
        )


@dataclass(frozen=True)
class EmbodimentEvaluationCfg:
    r"""独立 evaluation 的 data/method/evaluation/run 四角色根配置。"""

    schema_version: str = EVALUATION_SCHEMA_VERSION
    data: Any = MISSING
    method: Any = MISSING
    evaluation: EvaluationCfg = MISSING
    run: Any = MISSING

    def validate_composed(self) -> None:
        r"""验证 schema、运行输入与所有 concrete roles。"""

        _validate_root(self, schema=EVALUATION_SCHEMA_VERSION, roles=("data", "method", "run"))
        if not isinstance(self.evaluation, EvaluationCfg):
            raise TypeError("evaluation root requires a concrete EvaluationCfg")
        self.run.validate_inputs()


def _validate_root(config: Any, *, schema: str, roles: tuple[str, ...]) -> None:
    r"""共享两个事后阶段的 schema 与 concrete runtime 绑定检查。"""

    if config.schema_version != schema:
        raise ValueError(f"post-training schema must be exactly {schema}")
    missing = tuple(
        role for role in roles if getattr(config, role) == MISSING or getattr(config, role) == "???"
    )
    if missing:
        raise ValueError(f"post-training config is missing component roles: {missing}")
    invalid = tuple(
        role for role in roles if not callable(getattr(type(getattr(config, role)), "runtime_type", None))
    )
    if invalid:
        raise TypeError(f"post-training roles lack runtime_type bindings: {invalid}")


def resolved_post_training_config_dict(config: Any) -> dict[str, Any]:
    r"""把 concrete post-training config 解析为可审计基础 mapping。"""

    container = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    if not isinstance(container, dict):
        raise TypeError("resolved post-training config must be a mapping")
    return {str(key): value for key, value in container.items()}


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "EmbodimentEvaluation",
    "EmbodimentEvaluationCfg",
    "EmbodimentValidation",
    "EmbodimentValidationCfg",
    "EvaluationCfg",
    "EvaluationRun",
    "EvaluationRunCfg",
    "ValidationCfg",
    "ValidationRun",
    "ValidationRunCfg",
    "resolved_post_training_config_dict",
]
