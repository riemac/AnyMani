r"""几何 SSL 的结构化 Hydra/OmegaConf 实验合同。

配置只声明资产路径、静态物化、online sampling、模型、目标、优化与记录策略。资产里的 owner、
运动学、q_home、limits 和 frame 仍由 sidecar/robots 决定，不能被训练 YAML 覆盖。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field  # 冻结配置与 manifest 基础类型导出
from pathlib import Path  # resolved run artifacts 路径
from typing import Any  # OmegaConf 基础容器的嵌套 value 类型

import yaml  # resolved config/manifest 使用人类可读 YAML
from omegaconf import OmegaConf  # Hydra interpolation 与结构化 dataclass 桥接

from anymani.distill.models.geometry_ssl import GeometrySSLModelConfig  # retained/disposable 容量
from anymani.distill.models.input_adapters.geometry import (  # encoder 与跨结构上限
    GeometryEncoderConfig,
    GeometryPaddingCfg,
)
from anymani.distill.objectives.representations.field_reconstruction import GeometrySSLWeights  # 五项权重
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg  # 50/25/25 query
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg  # 带宽/edge/mask
from anymani.distill.ssl.dataset import GeometryAssetMaterializationCfg  # home/anchor/workspace cache


@dataclass(frozen=True)
class GeometrySSLAssetCfg:
    r"""generated train/validation 与隔离 official evaluation 资产入口。

    train/validation 只允许 generated 资产；official 路径在 pretrain 中按 ``source_kind='official'``
    fail-closed 解析，但不物化 teacher、不构建 optimizer batch。路径级检查先挡住明显复用，bank resolve
    后 manifest 再以 SHA-256 ``content_hash`` 检查重命名/复制导致的隐蔽泄漏。
    """

    train_paths: tuple[str, ...] = ()  # generated optimizer 数据来源
    validation_paths: tuple[str, ...] = ()  # generated 固定 held-out bank
    official_evaluation_paths: tuple[str, ...] = ()  # 冻结后 zero-shot/adaptation 身份

    def __post_init__(self) -> None:
        r"""拒绝路径级 split 泄漏；内容哈希级检查在 bank resolve 后执行。"""

        train = set(self.train_paths)  # 路径字符串去重后的训练集合
        validation = set(self.validation_paths)  # 路径字符串 validation 集合
        official = set(self.official_evaluation_paths)  # official 隔离集合
        if train & validation or train & official or validation & official:  # 任意 pair 交集
            raise ValueError("train/validation/official asset paths must be disjoint")  # 配置构造即拒绝


@dataclass(frozen=True)
class GeometrySSLOptimizerCfg:
    r"""AdamW 与梯度裁剪配置。

    默认 $\eta=3\times10^{-4}$、解耦权重衰减 $10^{-4}$；梯度总范数上限 10 只防止非有限爆炸，
    不替代各物理损失项的 generated-only 梯度范数校准。
    """

    learning_rate: float = 3.0e-4  # AdamW 学习率 $\eta$
    weight_decay: float = 1.0e-4  # 解耦参数衰减系数
    max_gradient_norm: float = 10.0  # clip 前全参数 L2 范数上限

    def __post_init__(self) -> None:
        r"""验证学习率严格为正，weight decay 非负，gradient norm 上限严格为正。"""

        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.max_gradient_norm <= 0.0:  # 数值域
            raise ValueError("optimizer learning rate/weight decay/gradient norm are invalid")  # fail before run


@dataclass(frozen=True)
class GeometrySSLTrainLoopCfg:
    r"""训练步数、微批次、复现实验与输出频率。

    ``batch_size`` 是每次 forward 的资产/q 样本数；有效 batch 为
    $B_{eff}=B\times N_{acc}$。gradient accumulation 按 microbatch loss 除以 $N_{acc}$，因此不改变
    总梯度均值。validation 使用启动时固定的一份 Sobol q/query/teacher bank。
    """

    steps: int = 10_000  # optimizer 更新次数
    batch_size: int = 16  # 单次 GPU forward 的 $B$
    gradient_accumulation_steps: int = 1  # 每次 optimizer step 的 microbatches
    seed: int = 0  # model、Sobol、query/edge 路由总种子
    device: str = "cuda:0"  # Warp 主路径要求 CUDA
    dtype: str = "float32"  # OmegaConf 1.3 structured config 不支持 Literal
    log_every_steps: int = 10  # TensorBoard/JSONL 标量周期
    checkpoint_every_steps: int = 1_000  # 完整+retained checkpoint 周期
    validation_every_steps: int = 250  # 固定 generated held-out bank 周期
    output_dir: str = "logs/geometry_ssl"  # 项目运行证据根
    experiment_name: str = "multi_anchor_geometry_ssl"  # 稳定实验目录名

    def __post_init__(self) -> None:
        r"""拒绝空训练、空 batch、空累积或不可触发的记录周期。"""

        counts = (  # 全部为离散正整数语义
            self.steps,  # optimizer steps
            self.batch_size,  # microbatch B
            self.gradient_accumulation_steps,  # accumulation count
            self.log_every_steps,  # logging interval
            self.checkpoint_every_steps,  # checkpoint interval
            self.validation_every_steps,  # validation interval
        )
        if any(value < 1 for value in counts):  # 0 会让生命周期分支不可达或除零
            raise ValueError("all training counts and intervals must be positive")  # 配置闸门
        cuda_device = self.device == "cuda" or (
            self.device.startswith("cuda:") and self.device.removeprefix("cuda:").isdigit()
        )
        if not cuda_device:  # online nearest-surface teacher 不提供 CPU fallback
            raise ValueError("geometry SSL training device must be 'cuda' or 'cuda:<index>'")
        if self.dtype != "float32":  # Warp PyTorch bridge 的主路径只接受 CUDA float32
            raise ValueError("geometry SSL training dtype must be 'float32'")  # 禁止声明无法执行的配置


@dataclass(frozen=True)
class GeometrySSLExperimentCfg:
    r"""首版完整可运行几何 SSL resolved config。

    该对象是 checkpoint、``resolved_config.yaml`` 与 CLI 的共同事实源。它不允许配置 owner、link、
    q_home 或 limits；这些静态物理事实只来自 ``HandContainer.geometry_semantics``。
    """

    schema_version: str = "1.0.0"  # 实验配置 schema
    assets: GeometrySSLAssetCfg = field(default_factory=GeometrySSLAssetCfg)  # split paths
    materialization: GeometryAssetMaterializationCfg = field(default_factory=GeometryAssetMaterializationCfg)  # cache
    query: SpatialQuerySamplerCfg = field(default_factory=SpatialQuerySamplerCfg)  # query 测度
    target: GeometryFieldTargetCfg = field(default_factory=GeometryFieldTargetCfg)  # teacher 物理超参
    padding: GeometryPaddingCfg = field(default_factory=GeometryPaddingCfg)  # 20 JOINT/26 owner 上限
    model: GeometrySSLModelConfig = field(default_factory=GeometrySSLModelConfig)  # 网络容量
    objective: GeometrySSLWeights = field(default_factory=GeometrySSLWeights)  # 五项损失权重
    optimizer: GeometrySSLOptimizerCfg = field(default_factory=GeometrySSLOptimizerCfg)  # AdamW
    train: GeometrySSLTrainLoopCfg = field(default_factory=GeometrySSLTrainLoopCfg)  # 生命周期

    def __post_init__(self) -> None:
        r"""闭合 target bandwidth $L$ 与 density decoder 输出轴 $L$。"""

        if self.model.bandwidth_count != len(self.target.bandwidths_m):  # 逐带宽监督必须同轴
            raise ValueError("model bandwidth_count must equal target bandwidth count")  # 不广播/截断


@dataclass(frozen=True)
class GeometrySSLAssetManifest:
    r"""resolve 后按内容哈希冻结的资产 split 证据。

    每条记录包含 asset ID、SHA-256、source kind、topology、family、handedness、JOINT/owner 数。
    内容哈希而非路径是 leakage 判据，因此复制或重命名资产也不能跨 split。
    """

    schema_version: str  # manifest schema
    train: tuple[dict[str, str], ...]  # generated optimizer split
    validation: tuple[dict[str, str], ...]  # generated fixed held-out split
    official_evaluation: tuple[dict[str, str], ...]  # 冻结后隔离 split

    def __post_init__(self) -> None:
        r"""拒绝任何 ``content_hash`` 跨 train/validation/official 重用。"""

        groups = tuple(  # 每个 split 的唯一内容集合
            {record["content_hash"] for record in split}  # 路径/ID 不参与判据
            for split in (self.train, self.validation, self.official_evaluation)  # 固定三组顺序
        )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:  # pairwise disjoint
            raise ValueError("asset content hashes leak across train/validation/official splits")  # 硬停止


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

    assets_payload = dict(payload["assets"])  # Hydra ListConfig -> 基础容器
    assets = GeometrySSLAssetCfg(  # 路径轴冻结为 tuple
        train_paths=tuple(assets_payload["train_paths"]),  # generated train
        validation_paths=tuple(assets_payload["validation_paths"]),  # generated held-out
        official_evaluation_paths=tuple(assets_payload["official_evaluation_paths"]),  # official only
    )
    target_payload = dict(payload["target"])  # teacher mapping
    target_payload["bandwidths_m"] = tuple(target_payload["bandwidths_m"])  # 固定物理 $\\sigma_\\ell$ 轴
    model_payload = dict(payload["model"])  # retained/disposable model mapping
    model_payload["encoder"] = GeometryEncoderConfig(**dict(model_payload["encoder"]))  # nested encoder
    return GeometrySSLExperimentCfg(  # 构造顺序触发全部子配置验证
        schema_version=str(payload["schema_version"]),  # 实验 schema
        assets=assets,  # split 路径
        materialization=GeometryAssetMaterializationCfg(**dict(payload["materialization"])),  # CPU cache
        query=SpatialQuerySamplerCfg(**dict(payload["query"])),  # 50/25/25
        target=GeometryFieldTargetCfg(**target_payload),  # $d/\\rho/\\kappa/g$
        padding=GeometryPaddingCfg(**dict(payload["padding"])),  # 20/5/26
        model=GeometrySSLModelConfig(**model_payload),  # encoder+decoder
        objective=GeometrySSLWeights(**dict(payload["objective"])),  # 五项权重
        optimizer=GeometrySSLOptimizerCfg(**dict(payload["optimizer"])),  # AdamW
        train=GeometrySSLTrainLoopCfg(**dict(payload["train"])),  # loop/logging
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
    "GeometrySSLExperimentCfg",  # 根配置
    "GeometrySSLOptimizerCfg",  # optimizer
    "GeometrySSLTrainLoopCfg",  # loop
    "experiment_config_from_dict",  # Hydra bridge
    "resolved_config_dict",  # checkpoint/YAML bridge
    "write_resolved_experiment_files",  # run artifacts
]
