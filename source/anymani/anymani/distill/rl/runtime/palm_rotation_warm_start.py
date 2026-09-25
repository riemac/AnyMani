'Validate Actor and Critic warm starts before loading checkpoint tensors.'

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anymani.assets.bank.path_utils import resolve_anymani_root

ACTOR_CHECKPOINT_PREFIX = "a2c_network.package.actor."
'Definition for Actor checkpoint prefix (a2c_network.package.actor.).'

ACTOR_WARM_START_SCHEMA_VERSION = "1.0.0"
'Definition for Actor warm start schema version (1.0.0).'

PHASE_CONTEXTUAL_ADAPTER_KEY = "phase_contextual_adapter.weight"
'Definition for phase contextual adapter key (phase_contextual_adapter.weight).'

PHASE_READOUT_ADAPTER_KEY = "phase_readout_adapter.weight"
'Definition for phase readout adapter key (phase_readout_adapter.weight).'

PHASE_PERIOD_MIN_STEPS = 2
'Definition for phase period min steps (2).'


def _sha256(path: Path) -> str:
    'Handle sha256.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    'Handle relative or absolute.'

    root = resolve_anymani_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _checkpoint_document(path: Path) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    'Handle checkpoint document.'

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


def _validate_phase_period(value: object, *, field_name: str) -> int | None:
    'Validate phase period.'

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < PHASE_PERIOD_MIN_STEPS:
        raise ValueError(f"{field_name} must be None or an integer >= {PHASE_PERIOD_MIN_STEPS}")
    return value


def _namespace_state(
    model: Mapping[str, Any],
    *,
    prefix: str,
    namespace_name: str,
) -> dict[str, torch.Tensor]:
    'Handle namespace state.'

    selected = {str(key)[len(prefix) :]: value for key, value in model.items() if str(key).startswith(prefix)}
    if not selected:
        raise ValueError(f"actor-init checkpoint contains no {namespace_name} namespace tensors")
    for key, value in selected.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{namespace_name} namespace value {key!r} is not a tensor")
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{namespace_name} namespace value {key!r} is non-finite")
    return selected


def _load_module_state_with_optional_phase(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    namespace_name: str,
    phase_key: str,
    allow_phase_clock_adaptation: bool,
) -> tuple[str, ...]:
    'Load module state with optional phase.'

    target_state = module.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    missing = target_keys - source_keys
    unexpected = source_keys - target_keys
    allowed_missing = {phase_key} if allow_phase_clock_adaptation else set()
    if unexpected or missing - allowed_missing or len(missing & allowed_missing) > 1:
        raise ValueError(
            f"{namespace_name} state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}; "
            "only one declared phase adapter weight may be missing"
        )
    if allow_phase_clock_adaptation and missing and missing != {phase_key}:
        raise ValueError(
            f"{namespace_name} phase adaptation requires exactly one missing weight {phase_key!r}; "
            f"missing={sorted(missing)}"
        )
    if not allow_phase_clock_adaptation and missing:
        raise ValueError(f"{namespace_name} state is missing weights: {sorted(missing)}")

    validated: dict[str, torch.Tensor] = {}
    for key in sorted(target_keys & source_keys):
        source_value = source_state[key]
        target_value = target_state[key]
        if tuple(source_value.shape) != tuple(target_value.shape) or source_value.dtype != target_value.dtype:
            raise ValueError(
                f"{namespace_name} weight {key!r} shape/dtype mismatch: source={tuple(source_value.shape)}/"
                f"{source_value.dtype}, target={tuple(target_value.shape)}/{target_value.dtype}"
            )
        validated[key] = source_value
    if missing:
        phase_value = target_state.get(phase_key)
        if not isinstance(phase_value, torch.Tensor):
            raise ValueError(f"{namespace_name} phase adaptation target lacks tensor {phase_key!r}")
        validated[phase_key] = torch.zeros_like(phase_value)
    module.load_state_dict(validated, strict=True)
    return tuple(sorted(validated))


def inspect_actor_init_checkpoint(
    path: str | Path,
    *,
    target_arm: str,
    target_history_encoder: str,
    target_provider_identity: Mapping[str, Any],
    initialize_critic: bool = False,
    actor_init_sigma: float | None = None,
    target_sigma_mode: str = "global",
    target_recovery_sigma_floor: float | None = None,
    allow_recovery_exploration_adaptation: bool = False,
    target_phase_period_steps: int | None = None,
    allow_phase_clock_adaptation: bool = False,
) -> dict[str, Any]:
    'Handle inspect Actor init checkpoint.'

    if actor_init_sigma is not None and (not math.isfinite(actor_init_sigma) or actor_init_sigma <= 0.0):
        raise ValueError("actor-init sigma must be finite and positive")
    if target_recovery_sigma_floor is not None and (
        target_sigma_mode != "global"
        or not math.isfinite(target_recovery_sigma_floor)
        or target_recovery_sigma_floor <= 0.0
    ):
        raise ValueError("recovery exploration requires global sigma and a finite positive floor")
    target_phase_period = _validate_phase_period(
        target_phase_period_steps, field_name="target phase_period_steps"
    )
    resolved = Path(path).expanduser().resolve(strict=True)
    _document, identity, model = _checkpoint_document(resolved)
    policy = identity.get("policy")
    training = identity.get("training")
    source_provider = identity.get("geometry_provider")
    if not isinstance(policy, Mapping) or not isinstance(training, Mapping) or not isinstance(source_provider, Mapping):
        raise ValueError("actor-init checkpoint identity lacks policy/training/geometry provider")
    source_arm = str(policy.get("arm", ""))
    source_history = str(training.get("history_encoder", ""))
    source_phase_period = _validate_phase_period(
        training.get("phase_period_steps"), field_name="source phase_period_steps"
    )
    source_actor_phase_keys = {
        str(key)[len(ACTOR_CHECKPOINT_PREFIX) :]
        for key in model
        if str(key).startswith(ACTOR_CHECKPOINT_PREFIX)
        and str(key)[len(ACTOR_CHECKPOINT_PREFIX) :].startswith("phase_")
    }
    source_critic_prefix = "a2c_network.package.critic."
    source_critic_phase_keys = {
        str(key)[len(source_critic_prefix) :]
        for key in model
        if str(key).startswith(source_critic_prefix)
        and str(key)[len(source_critic_prefix) :].startswith("phase_")
    }
    expected_actor_phase_keys = {PHASE_CONTEXTUAL_ADAPTER_KEY}
    expected_critic_phase_keys = {PHASE_READOUT_ADAPTER_KEY}
    if source_phase_period is None and source_actor_phase_keys:
        raise ValueError("source phase clock is disabled but actor checkpoint contains phase parameters")
    if source_phase_period is None and source_critic_phase_keys:
        raise ValueError("source phase clock is disabled but critic checkpoint contains phase parameters")
    if source_phase_period is not None and source_actor_phase_keys != expected_actor_phase_keys:
        raise ValueError("source phase clock is enabled but Actor phase adapter keys are incomplete")
    if initialize_critic and source_phase_period is not None and source_critic_phase_keys != expected_critic_phase_keys:
        raise ValueError("source phase clock is enabled but Critic phase adapter keys are incomplete")
    if not initialize_critic and source_phase_period is not None and source_critic_phase_keys not in (
        set(),
        expected_critic_phase_keys,
    ):
        raise ValueError("source phase clock has malformed Critic phase adapter keys")
    phase_adaptation: dict[str, Any] | None = None
    if source_phase_period != target_phase_period:
        if source_phase_period is None and target_phase_period is not None and allow_phase_clock_adaptation:
            if source_arm not in {"direct", "direct_token"}:
                raise ValueError("phase clock adaptation requires a direct or direct_token Actor")
            phase_adaptation = {
                "source_period_steps": source_phase_period,
                "target_period_steps": target_phase_period,
                "source_phase_period_steps": source_phase_period,
                "target_phase_period_steps": target_phase_period,
                "new_actor_key": PHASE_CONTEXTUAL_ADAPTER_KEY,
                "new_critic_key": PHASE_READOUT_ADAPTER_KEY,
                "zero_initialized": True,
            }
        elif source_phase_period is None and target_phase_period is not None:
            raise ValueError("phase clock period changed from disabled to enabled; explicit adaptation is required")
        else:
            raise ValueError(
                "phase clock adaptation only supports disabled source to enabled target; removal or period changes "
                "are rejected"
            )
    if training.get("sigma_mode", "global") != target_sigma_mode:
        raise ValueError("actor-init sigma mode mismatch; a new distribution head needs an explicit migration")
    source_recovery_sigma_floor = training.get("recovery_sigma_floor")
    recovery_adaptation = {}
    if source_recovery_sigma_floor != target_recovery_sigma_floor:
        if not allow_recovery_exploration_adaptation:
            raise ValueError("recovery exploration rule changed; explicit distribution adaptation is required")
        source_log_std = model.get(f"{ACTOR_CHECKPOINT_PREFIX}global_log_std")
        if not isinstance(source_log_std, torch.Tensor) or source_log_std.numel() != 1:
            raise ValueError("recovery adaptation requires the same one-parameter global sigma namespace")
        source_sigma = float(source_log_std.detach().exp().item())
        if not math.isfinite(source_sigma) or source_sigma <= 0.0:
            raise ValueError("recovery adaptation source global sigma is invalid")
        recovery_adaptation = {
            "recovery_exploration_adaptation": {
                "source_sigma_floor": source_recovery_sigma_floor,
                "target_sigma_floor": target_recovery_sigma_floor,
                "source_base_sigma": source_sigma,
                "parameter_names_and_mean": "unchanged",
                "changed_component": "state-conditioned-effective-sigma-rule",
            }
        }
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

    actor_state = _namespace_state(model, prefix=ACTOR_CHECKPOINT_PREFIX, namespace_name="actor")
    actor_keys = sorted(actor_state)
    exploration = {}
    if actor_init_sigma is not None:
        source_log_std = model.get(f"{ACTOR_CHECKPOINT_PREFIX}global_log_std")
        if not isinstance(source_log_std, torch.Tensor) or source_log_std.numel() != 1:
            raise ValueError("actor-init sigma override requires one shared source log standard deviation")
        source_sigma = float(source_log_std.detach().exp().item())
        if not math.isfinite(source_sigma) or source_sigma <= 0.0:
            raise ValueError("actor-init source sigma is non-finite or non-positive")
        exploration = {"actor_init_sigma": float(actor_init_sigma), "source_actor_sigma": source_sigma}
    task_contract = identity.get("task_contract")
    task_contract = dict(task_contract) if isinstance(task_contract, Mapping) else {}
    if initialize_critic:

        if task_contract.get("critic_task_state") != "axis-goal-error-max-positive-net-and-current-net":
            raise ValueError("critic-init source privileged task semantics do not match the target")
        if not any(str(key).startswith(source_critic_prefix) for key in model):
            raise ValueError("critic-init checkpoint contains no critic tensors")
        if "value_mean_std.count" not in model:
            raise ValueError("critic-init requires source value normalization statistics")
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
        "source_phase_period_steps": source_phase_period,
        "target_phase_period_steps": target_phase_period,
        "phase_clock_adaptation": phase_adaptation,
        "initialize_critic": bool(initialize_critic),
        "value_normalizer_initial_count": float(model["value_mean_std.count"].item()) if initialize_critic else None,
        **exploration,
        **recovery_adaptation,
        "reset_components": [
            *([] if initialize_critic else ["critic", "value_normalizer"]),
            *(["actor_exploration"] if actor_init_sigma is not None else []),
            *(["phase_adapters"] if phase_adaptation is not None else []),
            "actor_optimizer",
            "critic_optimizer",
            "reward_curriculum",
            "adr_state",
            "diagnostics_recorder",
            "training_counters",
            "random_states",
        ],
    }


def inspect_resumed_actor_warm_start(path: str | Path) -> dict[str, Any] | None:
    'Handle inspect resumed Actor warm start.'

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
    if not isinstance(reset_components, list) or (
        not bool(evidence.get("initialize_critic", False)) and "critic" not in reset_components
    ):
        raise ValueError("resume actor-warm-start provenance lacks the reset boundary")
    return dict(evidence)


def should_load_actor_init_checkpoint(
    *, actor_init_path: str, warm_start: object, full_checkpoint_resume: bool
) -> bool:
    'Handle should load Actor init checkpoint.'

    has_actor_init_path = bool(actor_init_path.strip())
    has_warm_start_identity = isinstance(warm_start, Mapping)
    if full_checkpoint_resume:
        if has_actor_init_path:
            raise ValueError("full checkpoint resume cannot also load an actor-init checkpoint")
        return False
    if has_actor_init_path != has_warm_start_identity:
        raise ValueError("fresh actor-init path and runtime warm-start identity must be jointly present or absent")
    return has_actor_init_path


def load_actor_init_checkpoint(
    actor: nn.Module,
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    actor_init_sigma: float | None = None,
    allow_phase_clock_adaptation: bool = False,
) -> tuple[str, ...]:
    'Load Actor init checkpoint.'

    exploration_parameter = None
    target_log_std = None
    if actor_init_sigma is not None:
        if not math.isfinite(actor_init_sigma) or actor_init_sigma <= 0.0:
            raise ValueError("actor-init sigma must be finite and positive")
        parameter = getattr(actor, "global_log_std", None)
        if not isinstance(parameter, nn.Parameter) or parameter.numel() != 1:
            raise ValueError("actor-init sigma override requires one shared target log standard deviation")
        target_log_std = math.log(actor_init_sigma)
        ceiling = float(getattr(actor, "max_log_std", float("nan")))
        if not math.isfinite(ceiling) or target_log_std > ceiling:
            raise ValueError("actor-init sigma exceeds the target exploration ceiling")
        if actor_init_sigma < torch.finfo(parameter.dtype).tiny:
            raise ValueError("actor-init sigma is below the target dtype normal range")
        exploration_parameter = parameter

    resolved = Path(path).expanduser().resolve(strict=True)
    actual_sha = _sha256(resolved)
    if actual_sha != expected_checkpoint_sha256:
        raise ValueError(
            f"actor-init checkpoint bytes drifted after identity inspection: expected={expected_checkpoint_sha256}, "
            f"actual={actual_sha}"
        )
    _document, _identity, model = _checkpoint_document(resolved)
    actor_state = _namespace_state(model, prefix=ACTOR_CHECKPOINT_PREFIX, namespace_name="actor")
    _load_module_state_with_optional_phase(
        actor,
        actor_state,
        namespace_name="actor",
        phase_key=PHASE_CONTEXTUAL_ADAPTER_KEY,
        allow_phase_clock_adaptation=allow_phase_clock_adaptation,
    )
    if exploration_parameter is not None:
        assert target_log_std is not None
        with torch.no_grad():
            exploration_parameter.fill_(target_log_std)
    return tuple(sorted(actor_state))


def load_critic_init_checkpoint(
    model: Any,
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    allow_phase_clock_adaptation: bool = False,
) -> None:
    'Load Critic init checkpoint.'

    resolved = Path(path).expanduser().resolve(strict=True)
    if _sha256(resolved) != expected_checkpoint_sha256:
        raise ValueError("critic-init checkpoint changed after inspection")
    _, _, state = _checkpoint_document(resolved)
    prefix = "a2c_network.package.critic."
    critic_state = _namespace_state(state, prefix=prefix, namespace_name="critic")
    normalizer_prefix = "value_mean_std."
    normalizer_state = {
        key[len(normalizer_prefix) :]: value for key, value in state.items() if key.startswith(normalizer_prefix)
    }
    if not normalizer_state or not bool(getattr(model, "normalize_value", False)):
        raise ValueError("critic-init requires source and target value normalization")
    _load_module_state_with_optional_phase(
        model.a2c_network.package.critic,
        critic_state,
        namespace_name="critic",
        phase_key=PHASE_READOUT_ADAPTER_KEY,
        allow_phase_clock_adaptation=allow_phase_clock_adaptation,
    )
    _load_module_state_with_optional_phase(
        model.value_mean_std,
        normalizer_state,
        namespace_name="value normalizer",
        phase_key="__never_allow_phase__",
        allow_phase_clock_adaptation=False,
    )


__all__ = [
    "ACTOR_CHECKPOINT_PREFIX",
    "ACTOR_WARM_START_SCHEMA_VERSION",
    "PHASE_CONTEXTUAL_ADAPTER_KEY",
    "PHASE_PERIOD_MIN_STEPS",
    "PHASE_READOUT_ADAPTER_KEY",
    "inspect_actor_init_checkpoint",
    "inspect_resumed_actor_warm_start",
    "load_actor_init_checkpoint",
    "load_critic_init_checkpoint",
    "should_load_actor_init_checkpoint",
]
