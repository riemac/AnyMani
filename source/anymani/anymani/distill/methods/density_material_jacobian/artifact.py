r"""Density + Gamma method 的 schema-5 encoder-only retained artifact。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.artifact import (
    RETAINED_ARTIFACT_SCHEMA_VERSION,
    RetainedLoadReport,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)


@dataclass(frozen=True)
class SE3RetainedEncoderArtifact:
    r"""严格恢复后的 N040 encoder 与不可变来源信息。

    Schema-5 artifact 只保留 $E_\phi(\mathcal E_{static},q)\mapsto Z$ 所需的 encoder master
    parameters。PPO 的冻结/微调日程属于下游策略，本类型只证明文件身份、数学 frontend identity、
    architecture config 与 state keys 完整一致。

    Attributes:
        encoder (SE3InvariantGeometryEncoder): 从 artifact 自描述 config 重建并严格加载的 N040 encoder。
        load_report (RetainedLoadReport): PyTorch missing/unexpected state keys；严格成功时二者均为空。
        artifact_sha256 (str): 完整 artifact bytes 的 SHA-256 十六进制摘要。
        path (Path): 已解析的 artifact 绝对路径。
        feature_spec (Mapping[str, Any]): unified PALM/JOINT/TIP entity 与 JOINT view 合同。
        input_contract (Mapping[str, Any]): frame、单位与 retained input 边界。
        lineage (Mapping[str, Any]): SSL checkpoint、dataset、code 与 worktree 来源信息。
    """

    encoder: SE3InvariantGeometryEncoder
    load_report: RetainedLoadReport
    artifact_sha256: str
    path: Path
    feature_spec: Mapping[str, Any]
    input_contract: Mapping[str, Any]
    lineage: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    r"""流式计算 artifact SHA-256，避免把 checkpoint bytes 额外复制到内存。

    Args:
        path (Path): 已存在的本地 artifact 文件。

    Returns:
        str: 64 字符小写十六进制 SHA-256。
    """

    digest = hashlib.sha256()  # 完整文件内容身份；不使用路径或 mtime 作为科研 provenance
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):  # 1 MiB block，峰值内存与文件大小解耦
            digest.update(block)  # SHA-256 顺序吸收原始 checkpoint bytes
    return digest.hexdigest()  # 固定 256-bit artifact identity


def load_se3_retained_encoder_artifact(
    path: Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
) -> SE3RetainedEncoderArtifact:
    r"""严格校验并恢复 N040 proper-$SE(3)$ encoder-only artifact。

    加载顺序先核对完整文件 SHA，再解析 schema/type/config/state。这样错误路径、截断文件或同名旧
    encoder 都会在构造 PPO model 前 fail closed。函数直接实例化
    :class:`SE3InvariantGeometryEncoder`，不会创建 density/Gamma readers、teacher、optimizer 或 trainer。

    Args:
        path (Path): schema-5 `retained_encoder.pt` 文件路径。
        expected_sha256 (str): 预注册的 64 字符 artifact SHA-256。
        map_location (str | torch.device): `torch.load` 的目标设备；默认先在 CPU 完成身份预检。

    Returns:
        SE3RetainedEncoderArtifact: 严格恢复的 encoder、load report 与来源信息。

    Raises:
        FileNotFoundError: artifact 路径不存在或不是普通文件。
        ValueError: SHA、schema、artifact type、encoder type、config 或 namespace 不符合 N040 合同。
        RuntimeError: encoder state 存在 missing/unexpected keys。
    """

    resolved_path = path.expanduser().resolve()  # provenance 使用无歧义绝对路径，不依赖 shell cwd
    if not resolved_path.is_file():
        raise FileNotFoundError(f"retained artifact does not exist: {resolved_path}")
    if len(expected_sha256) != 64:
        raise ValueError("expected retained artifact SHA-256 must contain 64 hexadecimal characters")
    actual_sha256 = _file_sha256(resolved_path)  # 在 torch.load 前拒绝错误/损坏 artifact
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"retained artifact SHA-256 mismatch: expected={expected_sha256.lower()}, actual={actual_sha256}"
        )

    # Schema-5 使用 `weights_only=True`；payload 中只允许基础 Python 容器与 tensors。
    payload = torch.load(resolved_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != RETAINED_ARTIFACT_SCHEMA_VERSION:
        actual_schema = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported retained artifact schema={actual_schema!r}; expected {RETAINED_ARTIFACT_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_type") != "retained_geometry_encoder":
        raise ValueError("retained artifact type is not retained_geometry_encoder")
    required = {"retained_state", "retained_model_config", "feature_spec", "input_contract", "lineage"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"retained artifact is missing fields: {sorted(missing)}")
    forbidden = ("optimizer_state", "trainer_state", "method_state", "query_backend", "target_backend", "objective")
    leaked = tuple(name for name in forbidden if name in payload)
    if leaked:
        raise ValueError(f"retained artifact contains disposable fields: {leaked}")

    # N031/N040 参数 shape 同形；必须读取显式 encoder identity，禁止从 keys 猜测 frontend 数学。
    model_config = payload.get("retained_model_config")
    if not isinstance(model_config, Mapping) or set(model_config) != {"encoder_type", "encoder"}:
        raise ValueError("retained_model_config must contain exactly encoder_type and encoder")
    if model_config.get("encoder_type") != "se3_invariant":
        raise ValueError("retained artifact encoder_type is not se3_invariant")
    encoder_payload = model_config.get("encoder")
    if not isinstance(encoder_payload, Mapping) or set(encoder_payload) != {"frontend", "backbone"}:
        raise ValueError("N040 encoder config must contain exactly frontend and backbone")
    frontend_payload = encoder_payload.get("frontend")
    backbone_payload = encoder_payload.get("backbone")
    if not isinstance(frontend_payload, Mapping) or not isinstance(backbone_payload, Mapping):
        raise ValueError("N040 frontend/backbone configs must be mappings")
    encoder_config = SE3InvariantGeometryEncoderCfg(
        frontend=SE3InvariantAnchorFrontendCfg(**dict(frontend_payload)),  # line-anchor widths 与米制尺度
        backbone=GraphBiasedTransformerCfg(**dict(backbone_payload)),  # 4-layer graph-biased retained trunk
    )
    if asdict(encoder_config) != dict(encoder_payload):
        raise ValueError("N040 encoder config cannot be reconstructed without drift")
    encoder = SE3InvariantGeometryEncoder(encoder_config)  # 只创建 retained encoder，不创建 disposable readers

    # State namespace 必须全部属于 `encoder.`；去前缀后逐 key 严格恢复 FP32 master parameters。
    retained_state = payload.get("retained_state")
    if not isinstance(retained_state, Mapping) or not retained_state:
        raise ValueError("retained artifact retained_state must be a non-empty mapping")
    unexpected_namespace = tuple(str(key) for key in retained_state if not str(key).startswith("encoder."))
    if unexpected_namespace:
        raise ValueError(f"retained artifact contains non-encoder namespaces: {unexpected_namespace}")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in retained_state.values()):
        raise ValueError("retained artifact requires FP32 tensor encoder state")
    encoder_state = {str(key)[len("encoder.") :]: value for key, value in retained_state.items()}
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    report = RetainedLoadReport(tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys))
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"retained N040 encoder key mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )

    # Metadata 保持 artifact 原值；下游 runtime identity 会另行加入 PPO dataset 与 evidence digest。
    feature_spec = payload.get("feature_spec")
    input_contract = payload.get("input_contract")
    lineage = payload.get("lineage")
    if not isinstance(feature_spec, Mapping):
        raise ValueError("retained artifact feature_spec must be a mapping")
    if not isinstance(input_contract, Mapping):
        raise ValueError("retained artifact input_contract must be a mapping")
    if not isinstance(lineage, Mapping):
        raise ValueError("retained artifact lineage must be a mapping")
    return SE3RetainedEncoderArtifact(
        encoder=encoder,
        load_report=report,
        artifact_sha256=actual_sha256,
        path=resolved_path,
        feature_spec={str(key): value for key, value in feature_spec.items()},
        input_contract={str(key): value for key, value in input_contract.items()},
        lineage={str(key): value for key, value in lineage.items()},
    )


def build_retained_artifact(
    method: Any,
    *,
    metadata: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:
    r"""发布只含 unified encoder 的 schema-5 artifact，不泄漏 density/Gamma readers。"""

    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"retained artifact source checkpoint does not exist: {source_checkpoint}")
    raw = method.retained_state_dict()
    if not raw or any(not name.startswith("encoder.") for name in raw):
        raise ValueError("retained artifact requires non-empty encoder-only state")
    if any(value.dtype != torch.float32 for value in raw.values()):
        raise ValueError("retained artifact requires FP32 encoder master parameters")
    retained = {
        name: value.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, value in raw.items()
    }
    resolved = metadata.get("resolved_config", {})
    trainer = resolved.get("trainer", {}) if isinstance(resolved, Mapping) else {}
    precision = trainer.get("execution", {}) if isinstance(trainer, Mapping) else {}
    source_artifact = metadata.get("source_artifact", {})
    if not isinstance(precision, Mapping) or not isinstance(source_artifact, Mapping):
        raise ValueError("retained artifact lineage lacks precision or source identity")
    encoder_config = method.config.model.encoder  # retained architecture 的强类型配置
    encoder_type = (
        "se3_invariant" if isinstance(encoder_config, SE3InvariantGeometryEncoderCfg) else "legacy_so2"
    )  # schema-5 consumer 不能仅凭同形 state keys 推断 frontend 数学
    feature_spec = method.feature_spec()  # frame contract 是 artifact 自描述语义的唯一来源
    return {
        "schema_version": "5.0.0",
        "artifact_type": "retained_geometry_encoder",
        "retained_state": retained,
        "retained_model_config": {
            "encoder_type": encoder_type,
            "encoder": asdict(encoder_config),
        },
        "feature_spec": asdict(feature_spec),
        "input_contract": {
            "frame": feature_spec.frame_contract,
            "units": "length=m,joint=rad,density=dimensionless,Gamma=rad^-1",
            "retained_inputs": "physical q + static geometry evidence",
            "discarded_ssl_readers": "density,material_jacobian",
        },
        "lineage": {
            "source_checkpoint": str(source_checkpoint),
            "checkpoint_schema_version": "9.0.0",
            "code_revision": metadata.get("code_revision", "unknown"),
            "package_version": metadata.get("package_version", "unknown"),
            "asset_manifest": dict(metadata.get("asset_manifest", {})),
            "dataset_identity": dict(metadata.get("dataset_identity", {})),
            "execution_precision": dict(precision),
            "source_artifact": dict(source_artifact),
            "parameter_partition": dict(metadata.get("parameter_partition", {})),
            "worktree_dirty": bool(metadata.get("worktree_dirty", False)),
            "worktree_fingerprint": str(metadata.get("worktree_fingerprint", "")),
        },
    }


__all__ = [
    "SE3RetainedEncoderArtifact",
    "build_retained_artifact",
    "load_se3_retained_encoder_artifact",
]
