r"""几何 SSL checkpoint 的完整 resume 与 retained-only transfer 合同。

完整 checkpoint 保存 encoder、两个 SSL-only decoders、optimizer、step 与可审计 metadata；迁入 IL/PPO 时
只允许读取 ``encoder.`` namespace。加载函数严格报告 missing/unexpected keys，不以兼容包装静默吞掉
结构变化。
"""

from __future__ import annotations

from collections.abc import Mapping  # metadata/retained state 只要求只读 mapping 合同
from dataclasses import asdict, dataclass  # metadata 冻结后序列化为基础类型
from pathlib import Path  # checkpoint 路径由 resolved run directory 管理
from typing import Any  # optimizer/config metadata 含嵌套基础类型

import torch  # tensor/optimizer state 使用官方 state_dict 序列化

from anymani.distill.models.geometry_ssl import GeometrySSLModel  # 完整 retained+disposable 组装
from anymani.distill.models.input_adapters.geometry import ImplicitGeometryEncoder  # PPO/IL transfer 目标

CHECKPOINT_SCHEMA_VERSION = "1.0.0"  # 顶层 payload schema；与资产语义 schema 独立


@dataclass(frozen=True)
class GeometrySSLCheckpointMetadata:
    r"""随 tensor state 一起保存、可独立审计的科研合同。

    ``asset_manifest`` 证明训练/validation/official split 内容哈希；``resolved_config`` 固定 target
    带宽、query 混合、模型容量、loss 权重与 optimizer。frame/unit 字符串让脱离源码读取 checkpoint
    的分析程序仍能拒绝米/厘米、rad/normalized-q 混淆。
    """

    code_revision: str  # Git commit；无法解析时显式 `unknown`
    package_version: str  # installed/editable AnyMani version
    geometry_semantics_schema: str  # assets 静态语义 schema 版本
    asset_manifest: Mapping[str, Any]  # train/validation/official 内容哈希 split
    resolved_config: Mapping[str, Any]  # Hydra/OmegaConf interpolation 后完整配置
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
) -> None:
    r"""保存可 resume 的完整 state 和无需 decoder 装配即可读取的 retained state。

    完整 ``model_state`` 服务 SSL resume；冗余的 ``retained_state`` 以 ``encoder.`` 前缀冻结 transfer
    边界，使 PPO 代码无需实例化任何 decoder。先写同目录临时文件再 rename，避免异常终止后正式路径
    指向半写 payload。

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
            "retained_state": model.retained_state_dict(),  # 仅 `encoder.*`
            "optimizer_state": optimizer.state_dict(),  # AdamW moments/param groups
            "metadata": asdict(metadata),  # 只含 weights-only loader 支持的基础类型
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


def load_retained_geometry_encoder(
    path: Path,  # 几何 SSL 完整 checkpoint
    *,
    encoder: ImplicitGeometryEncoder,  # PPO/IL 已装配的 retained encoder
    strict: bool = True,  # 正式 transfer 必须 True
    map_location: str | torch.device = "cpu",  # 默认 CPU 预检 key
) -> RetainedLoadReport:
    r"""只加载 ``encoder.`` state，明确拒绝把 SSL decoder 带进 PPO/IL。

    ``retained_state`` 中每个 key 必须以 ``encoder.`` 开头；去掉前缀后用 ``strict=False`` 获取完整
    incompatibility report，再由本函数的 ``strict`` 语义决定是否抛错。该两阶段流程既提供可审计信息，
    又避免 PyTorch strict 异常只展示截断上下文。

    Returns:
        RetainedLoadReport: missing/unexpected key 元组。
    """

    payload = _load_payload(path, map_location=map_location)  # weights-only 安全加载 + schema 验证
    retained = payload.get("retained_state")  # 读取冗余、明确的 transfer state
    if not isinstance(retained, Mapping) or not retained:  # 空/错误类型均不可 transfer
        raise ValueError("checkpoint does not contain a non-empty retained_state")  # 不退回 model_state 猜 namespace
    prefix = "encoder."  # 完整与 retained checkpoint 共享的稳定前缀
    unexpected_namespace = tuple(  # 审计任何 decoder/未知状态泄漏
        str(key) for key in retained if not str(key).startswith(prefix)
    )
    if unexpected_namespace:  # 生命周期边界错误优先于普通 key mismatch
        raise ValueError(f"retained_state contains non-encoder namespaces: {unexpected_namespace}")
    encoder_state = {  # 目标对象本身已是 encoder，故只在此边界去前缀
        str(key)[len(prefix) :]: value for key, value in retained.items()
    }
    incompatible = encoder.load_state_dict(encoder_state, strict=False)  # 收集而非吞掉 mismatch
    report = RetainedLoadReport(  # 转成不可变、可序列化报告
        tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys)
    )
    if strict and (report.missing_keys or report.unexpected_keys):  # 正式 transfer 的硬闸门
        raise RuntimeError(  # 同时展示两类 key，便于定位模型配置漂移
            f"retained encoder key mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )
    return report  # strict=False 诊断仍必须由调用者记录该报告


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
    required = {  # 完整 resume 与 retained transfer 共同所需字段
        "step",
        "model_state",
        "retained_state",
        "optimizer_state",
        "metadata",
    }
    missing = required - payload.keys()  # 顶层缺失字段集合
    if missing:  # 先完整报告，不在后续 KeyError 逐个失败
        raise ValueError(f"geometry SSL checkpoint is missing fields: {sorted(missing)}")
    return payload  # schema 已闭合的基础 mapping


__all__ = [  # checkpoint 模块稳定公开面
    "CHECKPOINT_SCHEMA_VERSION",  # payload schema 常量
    "GeometrySSLCheckpointMetadata",  # metadata schema
    "RetainedLoadReport",  # transfer 审计报告
    "load_geometry_ssl_checkpoint",  # SSL resume
    "load_retained_geometry_encoder",  # PPO/IL transfer
    "save_geometry_ssl_checkpoint",  # 原子完整保存
]
