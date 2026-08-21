r"""Schema 4 full checkpoint 与 standalone retained artifact 合同。

完整 checkpoint 只服务预训练 resume，保存 encoder、两个 SSL-only readers、optimizer、采样状态与审计
metadata。IL/PPO 只能读取独立 retained artifact；其 loader 严格报告 missing/unexpected keys，且 artifact
不携带 optimizer、teacher、reader 或 objective 配置。
"""

from __future__ import annotations

from collections.abc import Mapping  # metadata/retained state 只要求只读 mapping 合同
from dataclasses import asdict, dataclass  # metadata 冻结后序列化为基础类型
from pathlib import Path  # checkpoint 路径由 resolved run directory 管理
from typing import Any  # optimizer/config metadata 含嵌套基础类型

import torch  # tensor/optimizer state 使用官方 state_dict 序列化

from anymani.distill.methods.contracts import FeatureSpec
from anymani.distill.models.geometry_ssl import GeometrySSLModel  # 完整 retained+disposable 组装
from anymani.distill.models.input_adapters.geometry import ImplicitGeometryEncoder  # PPO/IL transfer 目标

CHECKPOINT_SCHEMA_VERSION = "4.0.0"  # 五项 objective、声明权重与 calibration artifact hash
RETAINED_ARTIFACT_SCHEMA_VERSION = "4.0.0"  # standalone transfer artifact 与 full checkpoint 同代但不同类型


@dataclass(frozen=True)
class GeometrySSLCheckpointMetadata:
    r"""随 tensor state 一起保存、可独立审计的科研合同。

    ``asset_manifest`` 证明 train、validation 与具名 evaluation suites 的 dataset provenance、内容哈希和
    physical identity；``resolved_config`` 固定 target 带宽、query 混合、模型容量、loss 权重与
    optimizer。frame/unit 字符串让脱离源码读取 checkpoint 的分析程序仍能拒绝米/厘米、
    rad/normalized-q 混淆。
    """

    code_revision: str  # Git commit；无法解析时显式 `unknown`
    package_version: str  # installed/editable AnyMani version
    geometry_semantics_schema: str  # assets 静态语义 schema 版本
    asset_manifest: Mapping[str, Any]  # train/validation/evaluation 的展开 provenance 与物理身份
    resolved_config: Mapping[str, Any]  # Hydra/OmegaConf interpolation 后完整配置
    declared_objective: Mapping[str, float]  # OBJECTIVES_CFG 中显式写出的五项权重
    calibration_artifact_hash: str = ""  # 前向预实验 artifact 的 SHA-256；空表示未加载
    worktree_dirty: bool = False  # True 表示运行代码不是干净 HEAD
    worktree_fingerprint: str = ""  # dirty/untracked manifest 指纹，不把大 diff 写入每个 checkpoint
    frame_contract: str = "query/closest/surface in hand frame {h}"  # 全部几何点 frame
    unit_contract: str = "length=m,joint=rad,density=dimensionless,kappa=m/rad,g=1/rad"  # SI 量纲
    retained_namespaces: tuple[str, ...] = ("encoder.",)  # 迁入 PPO/IL
    disposable_namespaces: tuple[str, ...] = (  # SSL 结束后整体删除
        "density_decoder.",
        "sensitivity_decoder.",
    )


@dataclass(frozen=True)
class RetainedLoadReport:
    r"""PPO/IL 初始化时必须由调用者记录的严格 key 报告。

    即使调用者选择 ``strict=False`` 做受控诊断，也必须得到具体 missing/unexpected keys；不能只返回
    一个布尔值掩盖部分加载。正式 transfer 默认 strict，任一非空字段都终止加载。
    """

    missing_keys: tuple[str, ...]  # 当前 encoder 期望但 checkpoint 缺失
    unexpected_keys: tuple[str, ...]  # checkpoint 含有但当前 encoder 不认识


def save_geometry_ssl_checkpoint(
    path: Path,  # 正式 `.pt` 输出路径
    *,
    model: GeometrySSLModel,  # 完整 SSL 模型
    optimizer: torch.optim.Optimizer,  # resume 所需动量/二阶矩
    step: int,  # 已完成 optimizer steps
    metadata: GeometrySSLCheckpointMetadata,  # 科研复现实验合同
    runtime_state: Mapping[str, Any] | None = None,  # epoch/window/Sobol cursor resume state
) -> None:
    r"""保存只服务预训练 resume/审计的完整 state。

    ``model_state`` 包含 retained encoder 与 disposable readers；PPO/IL 不读取该 payload，只消费独立
    retained artifact。先写同目录临时文件再 rename，避免异常终止后正式路径指向半写 payload。

    Raises:
        ValueError: step 为负时抛出。
    """

    if step < 0:  # step 是训练生命周期坐标，不能出现未开始前负索引
        raise ValueError("checkpoint step must be non-negative")  # fail before filesystem mutation
    path.parent.mkdir(parents=True, exist_ok=True)  # 只创建明确 checkpoint parent
    temporary = path.with_suffix(path.suffix + ".tmp")  # 同文件系统原子 rename 前置文件
    torch.save(  # PyTorch tensors/optimizer state 保持 dtype/device metadata
        {  # 顶层字段由 `_load_payload` 严格验证
            "schema_version": CHECKPOINT_SCHEMA_VERSION,  # payload reader 路由键
            "step": int(step),  # Python int，resume 从下一 step 继续
            "model_state": model.state_dict(),  # encoder + 两个 disposable decoders
            "optimizer_state": optimizer.state_dict(),  # AdamW moments/param groups
            "metadata": asdict(metadata),  # 只含 weights-only loader 支持的基础类型
            "runtime_state": dict(runtime_state or {}),  # 资产窗口与每资产 q 游标
        },
        temporary,  # 正式路径在完整写入前保持不存在/旧版本
    )
    temporary.replace(path)  # 同一文件系统 rename，避免中断留下半个正式 checkpoint


def load_geometry_ssl_checkpoint(
    path: Path,  # 完整 SSL checkpoint
    *,
    model: GeometrySSLModel,  # 配置必须与 checkpoint 对齐
    optimizer: torch.optim.Optimizer | None = None,  # 可选 resume optimizer
    map_location: str | torch.device = "cpu",  # 默认先安全映射 CPU
) -> tuple[int, dict[str, Any]]:
    r"""严格恢复完整 SSL 模型，若提供 optimizer 则同时恢复训练状态。

    Returns:
        tuple[int, dict[str, Any]]: 已保存 step 与 metadata；模型/optimizer 原位更新。
    """

    payload = _load_payload(path, map_location=map_location)  # 先验证 schema/字段再修改模型
    model.load_state_dict(payload["model_state"], strict=True)  # 完整 namespace 必须逐 key 对齐
    if optimizer is not None:  # inference/审计可只恢复模型
        optimizer.load_state_dict(payload["optimizer_state"])  # 恢复 AdamW moments 与 param groups
    return int(payload["step"]), dict(payload["metadata"])  # 调用者决定下一 step 与记录方式


def load_geometry_ssl_runtime_state(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    r"""读取 checkpoint 中的 runtime cursor，不装配模型或 optimizer。"""

    payload = _load_payload(path, map_location=map_location)
    runtime_state = payload.get("runtime_state", {})
    if not isinstance(runtime_state, Mapping):
        raise ValueError("geometry SSL checkpoint runtime_state must be a mapping")
    return dict(runtime_state)


def save_retained_geometry_artifact(
    path: Path,
    *,
    model: GeometrySSLModel,
    feature_spec: FeatureSpec,
    metadata: GeometrySSLCheckpointMetadata,
    source_checkpoint: Path,
) -> None:
    r"""原子写出只含 retained encoder 的 schema 4 standalone artifact。

    该 payload 不保存 optimizer、training q sampler、query/target backend、decoder 或 objective；下游
    只能从 ``retained_state`` 和输入/特征合同构造部署期消费路径。
    """

    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"retained artifact source checkpoint does not exist: {source_checkpoint}")
    retained = model.retained_state_dict()
    if not retained or any(not key.startswith("encoder.") for key in retained):
        raise ValueError("retained artifact requires a non-empty encoder-only state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": RETAINED_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "retained_geometry_encoder",
            "retained_state": retained,
            "retained_model_config": {"encoder": asdict(model.config.encoder)},
            "feature_spec": asdict(feature_spec),
            "input_contract": {
                "frame": metadata.frame_contract,
                "units": metadata.unit_contract,
                "retained_inputs": "physical q + static geometry evidence",
            },
            "lineage": {
                "source_checkpoint": str(source_checkpoint),
                "code_revision": metadata.code_revision,
                "package_version": metadata.package_version,
                "geometry_semantics_schema": metadata.geometry_semantics_schema,
                "asset_manifest": dict(metadata.asset_manifest),
            },
        },
        temporary,
    )
    temporary.replace(path)


def load_retained_geometry_artifact(
    path: Path,
    *,
    encoder: ImplicitGeometryEncoder,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> RetainedLoadReport:
    r"""严格加载 standalone retained artifact，拒绝把 full checkpoint 当作 transfer artifact。"""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != RETAINED_ARTIFACT_SCHEMA_VERSION:
        actual = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported retained artifact schema={actual!r}; expected {RETAINED_ARTIFACT_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_type") != "retained_geometry_encoder":
        raise ValueError("retained artifact type is not retained_geometry_encoder")
    required = {"retained_state", "retained_model_config", "feature_spec", "input_contract", "lineage"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"retained artifact is missing fields: {sorted(missing)}")
    forbidden = ("optimizer_state", "runtime_state", "model_state", "query_backend", "target_backend", "objective")
    leaked = tuple(name for name in forbidden if name in payload)
    if leaked:
        raise ValueError(f"retained artifact contains disposable namespaces: {leaked}")
    retained = payload.get("retained_state")
    if not isinstance(retained, Mapping):
        raise ValueError("retained artifact retained_state must be a mapping")
    unexpected_namespace = tuple(str(key) for key in retained if not str(key).startswith("encoder."))
    if unexpected_namespace:
        raise ValueError(f"retained artifact contains non-encoder namespaces: {unexpected_namespace}")
    encoder_state = {str(key)[len("encoder.") :]: value for key, value in retained.items()}
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    report = RetainedLoadReport(tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys))
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError(
            f"retained encoder key mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )
    return report


def _load_payload(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    r"""读取 checkpoint 并首先验证 schema version 与顶层字段。

    ``weights_only=True`` 禁止任意 pickle object construction；payload 仅允许 tensor、容器和基础类型。
    """

    payload = torch.load(path, map_location=map_location, weights_only=True)  # 不执行任意 Python 对象
    if not isinstance(payload, dict):  # 顶层 schema 固定 mapping
        raise TypeError("geometry SSL checkpoint payload must be a mapping")  # 拒绝裸 state_dict 猜测
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:  # 无隐式兼容/迁移
        raise ValueError(  # 报告实际/期望版本
            f"unsupported geometry SSL checkpoint schema={payload.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    required = {  # 完整 resume 所需字段；transfer 使用独立 retained artifact reader
        "step",
        "model_state",
        "optimizer_state",
        "metadata",
    }
    missing = required - payload.keys()  # 顶层缺失字段集合
    if missing:  # 先完整报告，不在后续 KeyError 逐个失败
        raise ValueError(f"geometry SSL checkpoint is missing fields: {sorted(missing)}")
    return payload  # schema 已闭合的基础 mapping


__all__ = [  # checkpoint 模块稳定公开面
    "CHECKPOINT_SCHEMA_VERSION",  # payload schema 常量
    "RETAINED_ARTIFACT_SCHEMA_VERSION",
    "GeometrySSLCheckpointMetadata",  # metadata schema
    "RetainedLoadReport",  # transfer 审计报告
    "load_geometry_ssl_checkpoint",  # SSL resume
    "load_geometry_ssl_runtime_state",  # window/Sobol cursor resume
    "load_retained_geometry_artifact",
    "save_retained_geometry_artifact",
    "save_geometry_ssl_checkpoint",  # 原子完整保存
]
