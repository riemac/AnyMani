r"""掌托旋转PPO的显式Actor-only checkpoint迁移。

``--checkpoint``仍由rl_games执行完整同identity续训；本模块为首次``--actor_init_checkpoint``加载Actor，并在随后
full resume时从target checkpoint恢复同一份只读迁移血缘。首次迁移只读取``a2c_network.package.actor.*``，不接触
Critic、两套optimizer、value normalizer、课程、ADR、诊断cursor或随机状态。Source与target可使用不同cohort/task
reward，但必须共享actor arm、History30 encoder与retained N040 artifact，否则张量即使shape碰巧相同也拒绝迁移。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anymani.assets.bank.path_utils import resolve_anymani_root

ACTOR_CHECKPOINT_PREFIX = "a2c_network.package.actor."
"""rl_games完整model state中唯一允许warm-start加载的namespace。"""

ACTOR_WARM_START_SCHEMA_VERSION = "1.0.0"
"""写入target method identity的Actor-only迁移证据schema。"""


def _sha256(path: Path) -> str:
    r"""流式计算checkpoint byte identity。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)  # 1 MiB有界读取，不把checkpoint复制到第二份bytes
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    r"""仓库内checkpoint记录相对路径，外部artifact保留absolute路径。"""

    root = resolve_anymani_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _checkpoint_document(path: Path) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    r"""加载完整checkpoint并返回根、source identity与model state三个validated mappings。"""

    document = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(document, dict):
        raise TypeError("actor-init checkpoint root must be a mapping")
    identity = document.get("anymani_identity")
    model = document.get("model")
    if not isinstance(identity, Mapping) or identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
        raise ValueError("actor-init checkpoint requires a schema-3/4 AnyMani identity")
    if not isinstance(model, Mapping):
        raise ValueError("actor-init checkpoint is missing model state")
    return document, identity, model


def inspect_actor_init_checkpoint(
    path: str | Path,
    *,
    target_arm: str,
    target_history_encoder: str,
    target_provider_identity: Mapping[str, Any],
) -> dict[str, Any]:
    r"""验证迁移兼容性并返回写入target identity的JSON-safe证据。

    Args:
        path (str | Path): 完整source PPO checkpoint。
        target_arm (str): 目标``base/residual/direct/direct_token``动作生成结构。
        target_history_encoder (str): 目标``tcn/raw_stack``时间编码器。
        target_provider_identity (Mapping[str, Any]): 目标N040 provider identity，必须含retained artifact SHA。

    Returns:
        dict[str, Any]: Parent SHA、source arm/cohort/task、加载namespace与显式重置组件。
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    _document, identity, model = _checkpoint_document(resolved)
    policy = identity.get("policy")
    training = identity.get("training")
    source_provider = identity.get("geometry_provider")
    if not isinstance(policy, Mapping) or not isinstance(training, Mapping) or not isinstance(source_provider, Mapping):
        raise ValueError("actor-init checkpoint identity lacks policy/training/geometry provider")
    source_arm = str(policy.get("arm", ""))
    source_history = str(training.get("history_encoder", ""))
    if source_arm != target_arm:
        raise ValueError(f"actor-init arm mismatch: source={source_arm!r}, target={target_arm!r}")
    if source_history != target_history_encoder:
        raise ValueError(
            f"actor-init History30 encoder mismatch: source={source_history!r}, target={target_history_encoder!r}"
        )

    source_retained = source_provider.get("retained_artifact")
    target_retained = target_provider_identity.get("retained_artifact")
    if not isinstance(source_retained, Mapping) or not isinstance(target_retained, Mapping):
        raise ValueError("actor-init requires source and target retained-artifact identities")
    source_retained_sha = str(source_retained.get("sha256", ""))
    target_retained_sha = str(target_retained.get("sha256", ""))
    if not source_retained_sha or source_retained_sha != target_retained_sha:
        raise ValueError("actor-init source and target use different retained N040 artifacts")

    actor_keys = sorted(str(key) for key in model if str(key).startswith(ACTOR_CHECKPOINT_PREFIX))
    if not actor_keys:
        raise ValueError("actor-init checkpoint contains no actor namespace tensors")
    task_contract = identity.get("task_contract")
    task_contract = dict(task_contract) if isinstance(task_contract, Mapping) else {}
    return {
        "schema_version": ACTOR_WARM_START_SCHEMA_VERSION,
        "checkpoint_path": _relative_or_absolute(resolved),
        "checkpoint_sha256": _sha256(resolved),
        "source_identity_digest": str(identity.get("identity_digest", "")),
        "source_arm": source_arm,
        "source_history_encoder": source_history,
        "source_cohort_id": str(training.get("cohort_id", "")),
        "source_cohort_lock_sha256": str(training.get("cohort_lock_sha256", "")),
        "source_task_id": str(identity.get("task_id", "")),
        "source_task_contract": task_contract,
        "retained_artifact_sha256": source_retained_sha,
        "source_namespace": ACTOR_CHECKPOINT_PREFIX,
        "target_namespace": "package.actor.",
        "loaded_tensor_count": len(actor_keys),
        "reset_components": [
            "critic",
            "actor_optimizer",
            "critic_optimizer",
            "value_normalizer",
            "reward_curriculum",
            "adr_state",
            "diagnostics_recorder",
            "training_counters",
            "random_states",
        ],
    }


def inspect_resumed_actor_warm_start(path: str | Path) -> dict[str, Any] | None:
    r"""从target full checkpoint恢复首次Actor-only初始化血缘，不再次加载parent Actor。

    Full resume必须重建与checkpoint完全相同的method identity。若首次run由warm-start初始化，该证据已经写入
    ``anymani_identity.training.actor_warm_start``；续训应原样保留它，而不是要求已经包含全部训练状态的u20/u500
    checkpoint再次与parent checkpoint执行Actor-only加载。
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    _document, identity, _model = _checkpoint_document(resolved)
    training = identity.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("resume checkpoint identity lacks training contract")
    evidence = training.get("actor_warm_start")
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != ACTOR_WARM_START_SCHEMA_VERSION:
        raise ValueError("resume checkpoint has malformed actor-warm-start provenance")
    checkpoint_sha = evidence.get("checkpoint_sha256")
    reset_components = evidence.get("reset_components")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("resume actor-warm-start provenance lacks a SHA-256 parent identity")
    if not isinstance(reset_components, list) or "critic" not in reset_components:
        raise ValueError("resume actor-warm-start provenance lacks the reset boundary")
    return dict(evidence)


def should_load_actor_init_checkpoint(
    *, actor_init_path: str, warm_start: object, full_checkpoint_resume: bool
) -> bool:
    r"""区分首次Actor-only迁移与随后由target checkpoint执行的完整续训。"""

    has_actor_init_path = bool(actor_init_path.strip())
    has_warm_start_identity = isinstance(warm_start, Mapping)
    if full_checkpoint_resume:
        if has_actor_init_path:
            raise ValueError("full checkpoint resume cannot also load an actor-init checkpoint")
        return False  # target checkpoint稍后统一恢复Actor、Critic、optimizers与所有continuation state
    if has_actor_init_path != has_warm_start_identity:
        raise ValueError("fresh actor-init path and runtime warm-start identity must be jointly present or absent")
    return has_actor_init_path


def load_actor_init_checkpoint(
    actor: nn.Module,
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
) -> tuple[str, ...]:
    r"""只把source Actor namespace严格加载到目标Actor，并返回有序tensor keys。

    ``strict=True``同时拒绝跨arm结构、缺参数和多参数；调用方应在fresh optimizer构造前执行本函数。
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    actual_sha = _sha256(resolved)
    if actual_sha != expected_checkpoint_sha256:
        raise ValueError(
            f"actor-init checkpoint bytes drifted after identity inspection: expected={expected_checkpoint_sha256}, "
            f"actual={actual_sha}"
        )
    _document, _identity, model = _checkpoint_document(resolved)
    actor_state = {
        str(key)[len(ACTOR_CHECKPOINT_PREFIX) :]: value
        for key, value in model.items()
        if str(key).startswith(ACTOR_CHECKPOINT_PREFIX)
    }
    if not actor_state or any(not isinstance(value, torch.Tensor) for value in actor_state.values()):
        raise ValueError("actor-init namespace must contain only non-empty tensor state")
    actor.load_state_dict(actor_state, strict=True)  # 唯一参数写入点；不访问checkpoint其它keys
    return tuple(sorted(actor_state))


__all__ = [
    "ACTOR_CHECKPOINT_PREFIX",
    "ACTOR_WARM_START_SCHEMA_VERSION",
    "inspect_actor_init_checkpoint",
    "inspect_resumed_actor_warm_start",
    "load_actor_init_checkpoint",
    "should_load_actor_init_checkpoint",
]
