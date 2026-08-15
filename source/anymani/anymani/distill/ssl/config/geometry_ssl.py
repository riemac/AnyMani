r"""几何 SSL 的结构化 Hydra/OmegaConf 实验合同。

配置只声明资产路径、静态物化、online sampling、模型、目标、优化与记录策略。资产里的 owner、
运动学、q_home、limits 和 frame 由 ``assets`` 的 typed semantics 交付，动态 lower 由
``representations.sources`` 完成，不能被训练 YAML 覆盖。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field  # 冻结配置与 manifest 基础类型导出
from pathlib import Path  # resolved run artifacts 路径
from typing import Any  # OmegaConf 基础容器的嵌套 value 类型

import yaml  # resolved config/manifest 使用人类可读 YAML
from omegaconf import OmegaConf  # Hydra interpolation 与结构化 dataclass 桥接

from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg  # retained/disposable 容量
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.objectives.representations.field_reconstruction import GeometryFieldObjectiveCfg  # 六项权重
from anymani.distill.representations.geometry import GeometryRepresentationCfg  # source->query->target->layout
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg  # 50/25/25 query
from anymani.distill.representations.sources.geometry_source import GeometrySourceCfg  # home/anchor physical source
from anymani.distill.representations.targets.geometry_field import (
    GaussianProximityFieldCfg,  # sigma measure
    GeometryFieldTargetCfg,  # differential target
)


@dataclass(frozen=True)
class GeometrySSLAssetCfg:
    r"""generated train/validation 与隔离 official evaluation 资产入口。

    train/validation 只允许 generated 资产；official 路径在 pretrain 中按 ``source_kind='official'``
    fail-closed 解析，但不物化 teacher、不构建 optimizer batch。路径级检查先挡住明显复用，bank resolve
    后 manifest 再以 SHA-256 ``content_hash`` 检查重命名/复制导致的隐蔽泄漏。
    """

    family_paths: tuple[str, ...] = ()  # mother+variants；非空时自动按 physical hash 分组
    mother_asset_id: str = ""  # family 模式中固定进入训练的 mother ID
    validation_asset_count: int = 4  # 期望 held-out asset 数；完整 group 可使实际数调整
    split_seed: int = 20260813  # physical group 确定性划分 seed
    train_paths: tuple[str, ...] = ()  # 显式 generated optimizer 数据来源
    validation_paths: tuple[str, ...] = ()  # 显式 generated fixed held-out bank
    official_evaluation_paths: tuple[str, ...] = ()  # 冻结后 zero-shot/adaptation 身份

    def __post_init__(self) -> None:
        r"""拒绝路径级 split 泄漏；内容哈希级检查在 bank resolve 后执行。"""

        train = set(self.train_paths)  # 路径字符串去重后的训练集合
        validation = set(self.validation_paths)  # 路径字符串 validation 集合
        official = set(self.official_evaluation_paths)  # official 隔离集合
        if train & validation or train & official or validation & official:  # 任意 pair 交集
            raise ValueError("train/validation/official asset paths must be disjoint")  # 配置构造即拒绝
        if self.family_paths and (self.train_paths or self.validation_paths):
            raise ValueError("family_paths cannot be combined with explicit train/validation paths")
        if self.family_paths and not self.mother_asset_id:
            raise ValueError("family_paths require mother_asset_id for the fixed train group")
        if self.validation_asset_count < 0:
            raise ValueError("validation_asset_count must be non-negative")


@dataclass(frozen=True)
class GeometryCoverageCfg:
    r"""每资产 q coverage 与同一 sigma realization 内的 q 相关结构。"""

    epochs: int = 20
    q_per_asset_per_epoch: int = 256
    q_per_asset_per_realization: int = 2  # 同一 q 子批次共享 sigma，不能只当显存旋钮

    def __post_init__(self) -> None:
        if min(self.epochs, self.q_per_asset_per_epoch, self.q_per_asset_per_realization) < 1:
            raise ValueError("coverage epochs and q budgets must be positive")
        if self.q_per_asset_per_epoch % self.q_per_asset_per_realization:
            raise ValueError("q_per_asset_per_epoch must divide into q realization blocks")


@dataclass(frozen=True)
class GeometryCalibrationCfg:
    r"""train-only encoder-gradient calibration 的固定预算与裁剪域。"""

    batches: int = 8
    min_weight: float = 1.0e-2
    max_weight: float = 1.0e3

    def __post_init__(self) -> None:
        if self.batches < 1 or self.min_weight <= 0.0 or self.max_weight < self.min_weight:
            raise ValueError("calibration batches or weight bounds are invalid")


@dataclass(frozen=True)
class GeometryValidationCfg:
    r"""固定 held-out morphology bank、checkpoint cadence 与选择指标。"""

    q_per_asset: int = 64
    every_optimizer_updates: int = 250
    selection_metrics: tuple[str, ...] = ("density", "kappa", "derived_field")
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    )
    bootstrap_replicates: int = 2_000

    def __post_init__(self) -> None:
        if self.q_per_asset < 1 or self.every_optimizer_updates < 1 or self.bootstrap_replicates < 1:
            raise ValueError("validation q/cadence/bootstrap budget must be positive")
        if self.selection_metrics != ("density", "kappa", "derived_field"):
            raise ValueError("runtime currently requires density/kappa/derived_field selection metrics")
        expected_ablations = (
            "query_only",
            "same_asset_q_shuffle",
            "cross_asset_shuffle",
            "first_order_zero",
            "first_order_joint_shuffle",
            "first_order_sign_flip",
        )
        if self.final_ablations != expected_ablations:
            raise ValueError("runtime currently requires the six declared canonical final ablations")


@dataclass(frozen=True)
class GeometryReproducibilityCfg:
    r"""seed domain 与 deterministic backend 合同。"""

    seed: int = 0
    deterministic_algorithms: bool = True
    seed_domains: tuple[str, ...] = ("model", "sobol_q", "query", "sigma", "edge", "validation", "bootstrap")

    def __post_init__(self) -> None:
        expected = ("model", "sobol_q", "query", "sigma", "edge", "validation", "bootstrap")
        if self.seed < 0 or self.seed_domains != expected:
            raise ValueError("reproducibility seed and all runtime seed domains must be explicit")


@dataclass(frozen=True)
class GeometrySSLProtocolCfg:
    r"""coverage、calibration、validation、reproducibility 与 safety limit 的协议组合。"""

    coverage: GeometryCoverageCfg = field(default_factory=GeometryCoverageCfg)
    calibration: GeometryCalibrationCfg = field(default_factory=GeometryCalibrationCfg)
    validation: GeometryValidationCfg = field(default_factory=GeometryValidationCfg)
    reproducibility: GeometryReproducibilityCfg = field(default_factory=GeometryReproducibilityCfg)
    run_safety_step_limit: int = 30_000  # safety limit，不是 canonical 正式 optimizer budget

    def __post_init__(self) -> None:
        if self.run_safety_step_limit < 1:
            raise ValueError("run_safety_step_limit must be positive")


@dataclass(frozen=True)
class GeometrySSLTrainerCfg:
    r"""Hydra trainer preset 注入的优化器、microbatch、resident window 与记录 cadence。"""

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_gradient_norm: float = 10.0
    assets_per_microbatch: int = 2  # $A_{mb}$
    gradient_accumulation_steps: int = 4  # $N_{acc}$
    max_resident_assets: int = 20
    device: str = "cuda:0"
    dtype: str = "float32"
    log_every_updates: int = 10
    checkpoint_every_updates: int = 1_000

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.max_gradient_norm <= 0.0:
            raise ValueError("trainer optimizer values are invalid")
        if self.assets_per_microbatch < 1 or self.gradient_accumulation_steps < 1 or self.max_resident_assets < 1:
            raise ValueError("trainer batch/accumulation/resident counts must be positive")
        if self.log_every_updates < 1 or self.checkpoint_every_updates < 1:
            raise ValueError("trainer logging/checkpoint intervals must be positive")
        if not (self.device == "cuda" or (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("geometry SSL trainer device must be 'cuda' or 'cuda:<index>'")
        if self.dtype != "float32":
            raise ValueError("geometry SSL trainer dtype must be 'float32'")


@dataclass(frozen=True)
class GeometrySSLRunCfg:
    r"""只负责 output identity 与完整 checkpoint resume，不拥有训练超参。"""

    output_dir: str = "logs/geometry_ssl"
    experiment_name: str = "multi_anchor_geometry_ssl"
    resume_checkpoint: str = ""


@dataclass(frozen=True)
class GeometrySSLTrainingBudget:
    r"""由资产数、coverage 与 trainer 派生的可审计离散训练预算。"""

    train_asset_count: int
    microbatches_per_epoch: int
    optimizer_updates_per_epoch: int
    total_optimizer_updates: int
    total_q_samples: int
    nominal_microbatch_q: int
    nominal_effective_q: int
    mean_effective_q: float


def derive_geometry_ssl_training_budget(
    config: GeometrySSLExperimentCfg,
    *,
    train_asset_count: int,
) -> GeometrySSLTrainingBudget:
    r"""按 resident-window 分组规则解析实际 microbatch/update/q 预算。"""

    if train_asset_count < 1:
        raise ValueError("train_asset_count must be positive")
    assets_per_microbatch = config.trainer.assets_per_microbatch
    resident = config.trainer.max_resident_assets
    full_windows, remainder = divmod(train_asset_count, resident)
    groups_per_epoch_pass = full_windows * ((resident + assets_per_microbatch - 1) // assets_per_microbatch)
    if remainder:
        groups_per_epoch_pass += (remainder + assets_per_microbatch - 1) // assets_per_microbatch
    realization_q = config.protocol.coverage.q_per_asset_per_realization
    repeats = config.protocol.coverage.q_per_asset_per_epoch // realization_q
    microbatches = groups_per_epoch_pass * repeats
    accumulation = config.trainer.gradient_accumulation_steps
    if microbatches % accumulation:
        raise ValueError("microbatches_per_epoch must be divisible by gradient_accumulation_steps")
    updates = microbatches // accumulation
    epochs = config.protocol.coverage.epochs
    total_q = train_asset_count * config.protocol.coverage.q_per_asset_per_epoch * epochs
    nominal_microbatch_q = assets_per_microbatch * realization_q
    nominal_effective_q = nominal_microbatch_q * accumulation
    return GeometrySSLTrainingBudget(
        train_asset_count=train_asset_count,
        microbatches_per_epoch=microbatches,
        optimizer_updates_per_epoch=updates,
        total_optimizer_updates=updates * epochs,
        total_q_samples=total_q,
        nominal_microbatch_q=nominal_microbatch_q,
        nominal_effective_q=nominal_effective_q,
        mean_effective_q=(train_asset_count * config.protocol.coverage.q_per_asset_per_epoch) / updates,
    )


@dataclass(frozen=True)
class GeometrySSLExperimentCfg:
    r"""schema 2.0.0 的完整声明式 geometry SSL resolved config。

    该对象是 checkpoint、``resolved_config.yaml`` 与 CLI 的共同事实源。它不允许配置 owner、link、
    q_home 或 limits；这些静态物理事实只来自 ``HandContainer.geometry_semantics``。
    """

    schema_version: str = "2.0.0"  # 实验配置 schema
    assets: GeometrySSLAssetCfg = field(default_factory=GeometrySSLAssetCfg)  # split paths
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)  # source->field->query->target
    model: GeometrySSLModelCfg = field(default_factory=GeometrySSLModelCfg)  # retained + SSL-only readers
    objective: GeometryFieldObjectiveCfg = field(default_factory=GeometryFieldObjectiveCfg)  # 六项损失权重
    protocol: GeometrySSLProtocolCfg = field(default_factory=GeometrySSLProtocolCfg)  # scientific sampling/evidence
    trainer: GeometrySSLTrainerCfg = field(default_factory=GeometrySSLTrainerCfg)  # Hydra runtime preset
    run: GeometrySSLRunCfg = field(default_factory=GeometrySSLRunCfg)  # output/resume only

    def __post_init__(self) -> None:
        r"""验证 model sigma reference 位于 target 正带宽域内。

        sigma 数量是 target 的动态数据轴，不再与 decoder 输出宽度闭合；这里只要求 reference 为合法
        物理尺度，避免 ``log(sigma/sigma_reference)`` 接收退化配置。
        """

        if self.schema_version != "2.0.0":
            raise ValueError("geometry SSL experiment schema must be exactly 2.0.0")
        if self.trainer.max_resident_assets < self.trainer.assets_per_microbatch:
            raise ValueError("resident asset cap must fit one asset microbatch")


@dataclass(frozen=True)
class GeometrySSLAssetManifest:
    r"""resolve 后按内容哈希冻结的资产 split 证据。

    每条记录包含 asset ID、content/physical/configuration hashes、source kind、topology、family、
    handedness、JOINT/owner 数。physical hash 是学习映射 leakage 判据；content hash 继续防止完全相同
    sidecar 复制或重命名后跨 split。
    """

    schema_version: str  # manifest schema
    train: tuple[dict[str, str], ...]  # generated optimizer split
    validation: tuple[dict[str, str], ...]  # generated fixed held-out split
    official_evaluation: tuple[dict[str, str], ...]  # 冻结后隔离 split
    split_strategy: str = "explicit"  # explicit 或 physical_group
    split_seed: int = 0  # physical_group 模式的 deterministic seed
    requested_validation_asset_count: int = 0  # group split 目标数量
    actual_validation_asset_count: int = 0  # 完整 group 约束后的实际数量

    def __post_init__(self) -> None:
        r"""拒绝任何 content 或 physical identity 跨 split 重用。"""

        groups = tuple(  # 每个 split 的唯一内容集合
            {record["content_hash"] for record in split}  # 路径/ID 不参与判据
            for split in (self.train, self.validation, self.official_evaluation)  # 固定三组顺序
        )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:  # pairwise disjoint
            raise ValueError("asset content hashes leak across train/validation/official splits")  # 硬停止
        physical_groups = tuple(
            {record["physical_geometry_hash"] for record in split if record.get("physical_geometry_hash")}
            for split in (self.train, self.validation, self.official_evaluation)
        )
        if (
            physical_groups[0] & physical_groups[1]
            or physical_groups[0] & physical_groups[2]
            or physical_groups[1] & physical_groups[2]
        ):
            raise ValueError("physical geometry hashes leak across train/validation/official splits")


def resolved_config_dict(config: GeometrySSLExperimentCfg) -> dict[str, Any]:
    r"""经 OmegaConf interpolation resolution 后转为无 Python 对象的普通 mapping。

    Returns:
        dict[str, Any]: 可写 YAML/checkpoint 的完整 resolved mapping。
    """

    container = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)  # 解析 interpolation
    if not isinstance(container, dict):  # 实验根必须为 mapping
        raise TypeError("resolved geometry SSL config must be a mapping")  # 拒绝 list/scalar 根
    return {str(key): value for key, value in container.items()}  # 收窄键类型供 YAML/checkpoint


def experiment_config_from_dict(payload: dict[str, Any]) -> GeometrySSLExperimentCfg:
    r"""把 Hydra 的可变 DictConfig payload 重建为冻结且逐层验证的实验 dataclasses。

    Hydra 注册普通 mapping 以允许命令行 override；正式运行前必须回到冻结 dataclass，使每层
    ``__post_init__`` 数值/轴合同重新执行。
    """

    if str(payload.get("schema_version", "")) != "2.0.0":
        raise ValueError(
            "geometry SSL resolved config schema must be exactly 2.0.0; "
            "schema 1.x/checkpoints are intentionally fail-closed"
        )
    required_contract_fields = {
        "representation": {"source", "field", "query", "target", "layout"},
        "model": {"encoder", "ssl_decoders"},
        "protocol": {"coverage", "calibration", "validation", "reproducibility", "run_safety_step_limit"},
        "trainer": {"assets_per_microbatch", "gradient_accumulation_steps", "max_resident_assets"},
        "run": {"output_dir", "experiment_name", "resume_checkpoint"},
    }
    missing = {
        section: tuple(sorted(names - set(dict(payload.get(section, {})))))
        for section, names in required_contract_fields.items()
        if names - set(dict(payload.get(section, {})))
    }
    if missing:
        raise ValueError(
            "resolved config predates the online-query/explicit-sigma contract; "
            f"missing_fields={missing}"
        )

    assets_payload = dict(payload["assets"])  # Hydra ListConfig -> 基础容器
    assets = GeometrySSLAssetCfg(  # 路径轴冻结为 tuple
        family_paths=tuple(assets_payload.get("family_paths", ())),  # automatic grouped family
        mother_asset_id=str(assets_payload.get("mother_asset_id", "")),  # fixed train group
        validation_asset_count=int(assets_payload.get("validation_asset_count", 4)),  # held-out target
        split_seed=int(assets_payload.get("split_seed", 20260813)),  # group split seed
        train_paths=tuple(assets_payload["train_paths"]),  # generated train
        validation_paths=tuple(assets_payload["validation_paths"]),  # generated held-out
        official_evaluation_paths=tuple(assets_payload["official_evaluation_paths"]),  # official only
    )
    representation_payload = dict(payload["representation"])
    field_payload = dict(representation_payload["field"])
    field_payload["bandwidth_centers_m"] = tuple(field_payload["bandwidth_centers_m"])
    field_payload["validation_bandwidths_m"] = tuple(field_payload["validation_bandwidths_m"])
    representation = GeometryRepresentationCfg(
        source=GeometrySourceCfg(**dict(representation_payload["source"])),
        field=GaussianProximityFieldCfg(**field_payload),
        query=SpatialQuerySamplerCfg(**dict(representation_payload["query"])),
        target=GeometryFieldTargetCfg(**dict(representation_payload["target"])),
        layout=GeometryPaddingCfg(**dict(representation_payload["layout"])),
    )
    model_payload = dict(payload["model"])
    encoder_payload = dict(model_payload["encoder"])
    encoder = GeometryEncoderCfg(
        frontend=SO2AnchorFrontendCfg(**dict(encoder_payload["frontend"])),
        backbone=GraphBiasedTransformerCfg(**dict(encoder_payload["backbone"])),
        heads=GeometryLatentHeadsCfg(**dict(encoder_payload["heads"])),
    )
    decoder_payload = dict(model_payload["ssl_decoders"])
    model = GeometrySSLModelCfg(
        encoder=encoder,
        ssl_decoders=GeometrySSLDecoderCfg(
            density=ScalarSigmaFiLMDensityDecoderCfg(**dict(decoder_payload["density"])),
            sensitivity=DistanceSensitivityDecoderCfg(**dict(decoder_payload["sensitivity"])),
        ),
    )
    protocol_payload = dict(payload["protocol"])
    validation_payload = dict(protocol_payload["validation"])
    reproducibility_payload = dict(protocol_payload["reproducibility"])
    protocol = GeometrySSLProtocolCfg(
        coverage=GeometryCoverageCfg(**dict(protocol_payload["coverage"])),
        calibration=GeometryCalibrationCfg(**dict(protocol_payload["calibration"])),
        validation=GeometryValidationCfg(
            **{
                **validation_payload,
                "selection_metrics": tuple(validation_payload["selection_metrics"]),
                "final_ablations": tuple(validation_payload["final_ablations"]),
            }
        ),
        reproducibility=GeometryReproducibilityCfg(
            **{
                **reproducibility_payload,
                "seed_domains": tuple(reproducibility_payload["seed_domains"]),
            }
        ),
        run_safety_step_limit=int(protocol_payload["run_safety_step_limit"]),
    )
    return GeometrySSLExperimentCfg(  # 构造顺序触发全部子配置验证
        schema_version=str(payload["schema_version"]),  # 实验 schema
        assets=assets,  # split 路径
        representation=representation,  # source/field/query/target/layout
        model=model,  # encoder+SSL-only readers
        objective=GeometryFieldObjectiveCfg(**dict(payload["objective"])),  # 六项权重
        protocol=protocol,  # coverage/calibration/validation/reproducibility
        trainer=GeometrySSLTrainerCfg(**dict(payload["trainer"])),  # Hydra preset
        run=GeometrySSLRunCfg(**dict(payload["run"])),  # output/resume
    )


def write_resolved_experiment_files(
    output_dir: Path,  # 当前 run 唯一目录
    *,
    config: GeometrySSLExperimentCfg,  # 完整 resolved 实验配置
    manifest: GeometrySSLAssetManifest,  # 内容哈希 split
) -> None:
    r"""在训练开始前写入 resolved config 与资产 manifest YAML。

    两份文件均先于 GPU cache/optimizer 建立，因此后续物化失败仍保留“本次尝试使用了什么”的证据。
    """

    output_dir.mkdir(parents=True, exist_ok=True)  # 只创建 resolved run 目录
    (output_dir / "resolved_config.yaml").write_text(  # 完整配置事实源
        yaml.safe_dump(resolved_config_dict(config), sort_keys=False, allow_unicode=True),  # 保留字段顺序/中文
        encoding="utf-8",  # 跨平台确定编码
    )
    (output_dir / "asset_manifest.yaml").write_text(  # split 与 morphology 身份
        yaml.safe_dump(asdict(manifest), sort_keys=False, allow_unicode=True),  # tuple 安全序列化为 YAML sequence
        encoding="utf-8",  # 明确编码
    )


__all__ = [  # SSL 结构化配置公开面
    "GeometrySSLAssetCfg",  # split paths
    "GeometrySSLAssetManifest",  # resolved split evidence
    "GeometryCalibrationCfg",
    "GeometryCoverageCfg",
    "GeometrySSLExperimentCfg",  # 根配置
    "GeometrySSLProtocolCfg",
    "GeometrySSLRunCfg",
    "GeometrySSLTrainerCfg",
    "GeometrySSLTrainingBudget",
    "GeometryReproducibilityCfg",
    "GeometryValidationCfg",
    "experiment_config_from_dict",  # Hydra bridge
    "derive_geometry_ssl_training_budget",
    "resolved_config_dict",  # checkpoint/YAML bridge
    "write_resolved_experiment_files",  # run artifacts
]
