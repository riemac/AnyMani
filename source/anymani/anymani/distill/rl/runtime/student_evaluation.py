'Run a frozen student through the canonical twelve-input TorchScript ABI.'

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from anymani.distill.il.family_student import (
    FAMILY_STUDENT_ACTOR_ABI,
    FAMILY_STUDENT_SCHEMA_VERSION,
    TORCHSCRIPT_INPUT_ABI,
    FamilyStudentConfig,
    FamilyStudentTorchScriptWrapper,
    build_family_student,
)
from anymani.distill.models.family_rotation_policy import (
    FamilyRotationStudentActor,
    FamilyRotationVariant,
)

from .palm_rotation_student import binding_joint_kinematics_bank



STUDENT_PPO_ACTOR_PREFIX = "a2c_network.package.actor."
STUDENT_PPO_ARTIFACT_TYPE = "anymani.family_student_ppo_checkpoint"
STUDENT_PPO_SCHEMA_VERSION = "1.0.0"
STUDENT_PPO_TORCHSCRIPT_ARTIFACT_TYPE = "anymani.family_student_ppo_actor_torchscript"
STUDENT_PPO_CONTINUATION_SCHEMA_VERSION = "1.0.0"
ACTION_AUTHORITY_RAD_PER_STEP = 1.0 / 24.0
ACTION_RANGE_EPS = 1.0e-5


TORCHSCRIPT_INPUT_SHAPES: tuple[tuple[object, ...], ...] = (
    ("B", 16, 5),
    ("B", 30, 16, 5),
    ("B", 16, 2),
    ("B", 21, 1),
    ("B", 16),
    ("B", 4),
    ("B", 21),
    ("B", 21, 128),
    ("B", 21, 21),
    ("B", 21, 21),
    ("B", 21, 21),
    ("B", 16, 15),
)
TORCHSCRIPT_INPUT_DTYPES: tuple[str, ...] = (
    "float32",
    "float32",
    "float32",
    "float32",
    "bool",
    "bool",
    "bool",
    "float32",
    "int64",
    "int64",
    "int64",
    "float32",
)


def _sha256(path: Path) -> str:
    'Handle sha256.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: Mapping[str, Any]) -> str:
    'Hash the stable digest.'

    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    'Handle JSON safe.'

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_safe(value.item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("student PPO metadata contains a non-finite float")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    'Handle mapping.'

    if not isinstance(value, Mapping):
        raise ValueError(f"student PPO {name} must be a mapping")
    return value


def _sha256_value(value: Any, name: str) -> str:
    'Handle sha256 value.'

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"student PPO {name} must be a concrete 64-hex SHA-256")
    return value.lower()


def _nonempty_string(value: Any, name: str) -> str:
    'Handle nonempty string.'

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"student PPO {name} must be a non-empty string")
    return value


def _identity_retained_sha(identity: Mapping[str, Any], name: str) -> str:
    'Handle identity retained sha.'

    provider = _mapping(identity.get("geometry_provider"), f"{name}.geometry_provider")
    retained = _mapping(provider.get("retained_artifact"), f"{name}.geometry_provider.retained_artifact")
    sha = _sha256_value(retained.get("sha256"), f"{name}.geometry_provider.retained_artifact.sha256")

    if "artifact_type" in retained and retained["artifact_type"] != "retained_geometry_encoder":
        raise ValueError(f"{name} retained artifact is not a retained_geometry_encoder")
    if "schema_version" in retained and retained["schema_version"] != "5.0.0":
        raise ValueError(f"{name} retained artifact schema must be 5.0.0")
    return sha


def _source_init(identity: Mapping[str, Any], training: Mapping[str, Any]) -> Mapping[str, Any]:
    'Handle source init.'

    policy = identity.get("policy")
    policy_init = policy.get("student_initialization") if isinstance(policy, Mapping) else None
    training_init = training.get("student_initialization")
    if (
        isinstance(policy_init, Mapping)
        and isinstance(training_init, Mapping)
        and dict(policy_init) != dict(training_init)
    ):
        raise ValueError("student PPO policy/training student_initialization identities disagree")
    candidate = policy_init if isinstance(policy_init, Mapping) else training_init
    if not isinstance(candidate, Mapping):
        raise ValueError("student PPO checkpoint is missing student_initialization source identity")
    return candidate


def _resolve_declared_path(raw: str, checkpoint_path: Path) -> Path:
    'Resolve declared path.'

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    roots = (checkpoint_path.parent, Path.cwd(), Path(__file__).resolve().parents[5])
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (checkpoint_path.parent / candidate).resolve()


def _load_payload(path: Path) -> Mapping[str, Any]:
    'Load payload.'

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError(f"cannot load student PPO checkpoint {path}: {error}") from error
    return _mapping(payload, "checkpoint root")


def _assert_finite_state(state: Mapping[str, Any], name: str) -> None:
    'Handle assert finite state.'

    if not state:
        raise ValueError(f"student PPO {name} is empty")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"student PPO {name} must map string names to tensors")
        if value.is_floating_point() or value.is_complex():
            if value.dtype != torch.float32:
                raise ValueError(f"student PPO {name}.{key} must be FP32, got {value.dtype}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"student PPO {name}.{key} contains non-finite values")


def _validate_common_identity(identity: Mapping[str, Any], name: str) -> str:
    'Validate common identity.'

    schema = identity.get("identity_schema_version")
    if schema not in {"3.0.0", "4.0.0"}:
        raise ValueError(f"{name} identity schema must be 3.0.0 or 4.0.0")
    digest = identity.get("identity_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError(f"{name} identity_digest must be a concrete 64-hex SHA-256")
    payload = {key: value for key, value in identity.items() if key != "identity_digest"}
    if _stable_digest(payload) != digest.lower():
        raise ValueError(f"{name} identity_digest disagrees with its payload")
    if identity.get("task_id") != "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0":
        raise ValueError(f"{name} task_id is not the palm-rotation MVP task")

    policy = _mapping(identity.get("policy"), f"{name}.policy")
    if policy.get("arm") != "direct_token":
        raise ValueError(f"{name}.policy.arm must be direct_token for student PPO")
    if policy.get("actor_contact") != "tip-only-binary":
        raise ValueError(f"{name}.policy.actor_contact must be tip-only-binary")
    authority = policy.get("action_authority_rad_per_policy_step")
    if not isinstance(authority, (int, float)) or abs(float(authority) - ACTION_AUTHORITY_RAD_PER_STEP) > 1.0e-12:
        raise ValueError(f"{name}.policy action authority must remain 1/24 rad per policy step")

    training = _mapping(identity.get("training"), f"{name}.training")
    if training.get("history_encoder") != "tcn":
        raise ValueError(f"{name}.training.history_encoder must be tcn")
    if training.get("phase_period_steps") is not None:
        raise ValueError(f"{name}.training.phase_period_steps must be null for student Actor")
    if training.get("sigma_mode", "global") != "global":
        raise ValueError(f"{name}.training.sigma_mode must be global")
    return _identity_retained_sha(identity, name)


def _validate_source_actor(
    source_path: Path,
    *,
    declared_sha: str,
    variant: str,
    n040_sha: str,
) -> dict[str, Any]:
    'Validate source Actor.'

    actual_sha = _sha256(source_path)
    if actual_sha != declared_sha:
        raise ValueError("student PPO initial source Actor SHA disagrees with the declared identity")
    payload = _load_payload(source_path)
    if payload.get("artifact_type") != "anymani.family_distilled_actor":
        raise ValueError("student PPO initial source Actor is not a family distilled Actor artifact")
    if payload.get("schema_version") != FAMILY_STUDENT_SCHEMA_VERSION:
        raise ValueError("student PPO initial source Actor schema is unsupported")
    if payload.get("variant") != variant or payload.get("representation") != variant:
        raise ValueError("student PPO initial source Actor variant identity disagrees")
    metadata = _mapping(payload.get("metadata"), "initial source Actor metadata")
    source_n040 = metadata.get("n040_sha256")
    source_n040_matches = n040_sha in source_n040 if isinstance(source_n040, (list, tuple)) else source_n040 == n040_sha
    if not source_n040_matches:
        raise ValueError("student PPO initial source Actor N040 SHA disagrees with retained artifact")
    source_abi = payload.get("actor_abi")
    if not isinstance(source_abi, Mapping) or dict(source_abi) != FAMILY_STUDENT_ACTOR_ABI:
        raise ValueError("student PPO initial source Actor ABI disagrees with canonical family student ABI")
    return {
        "path": str(source_path),
        "sha256": actual_sha,
        "artifact_type": payload.get("artifact_type"),
        "schema_version": payload.get("schema_version"),
        "variant": variant,
        "n040_sha256": n040_sha,
        "dataset_sha256": metadata.get("dataset_sha256"),
    }


@dataclass(frozen=True)
class FrozenStudentActorArtifact:
    'Contract for frozen student Actor artifact.'

    actor: FamilyRotationStudentActor
    checkpoint_path: Path
    checkpoint_sha256: str
    identity: dict[str, Any]
    continuation: dict[str, Any]
    source_initialization: dict[str, Any]
    actor_config: dict[str, object]
    checkpoint_progress: dict[str, Any]
    variant: FamilyRotationVariant

    @property
    def n040_sha256(self) -> str:
        'Handle N040 sha256.'

        return _identity_retained_sha(self.identity, "student PPO")

    @property
    def method_identity_digest(self) -> str:
        'Handle method identity digest.'

        return str(self.identity["identity_digest"])

    @property
    def source_actor_sha256(self) -> str:
        'Handle source Actor sha256.'

        return str(self.source_initialization["declared_sha256"])

    @property
    def source_actor_path(self) -> str:
        'Handle source Actor path.'

        return str(self.source_initialization["declared_path"])


def load_frozen_student_actor(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
    expected_variant: FamilyRotationVariant | None = None,
    expected_n040_sha256: str | None = None,
    expected_source_actor_sha256: str | None = None,
) -> FrozenStudentActorArtifact:
    'Load frozen student Actor.'

    checkpoint_path = Path(path).expanduser().resolve()
    payload = _load_payload(checkpoint_path)
    identity = cast(dict[str, Any], dict(_mapping(payload.get("anymani_identity"), "anymani_identity")))
    n040_sha = _validate_common_identity(identity, "student PPO")
    if expected_n040_sha256 is not None and n040_sha != _sha256_value(expected_n040_sha256, "expected_n040_sha256"):
        raise ValueError("student PPO retained N040 SHA disagrees with expected_n040_sha256")

    training = _mapping(identity.get("training"), "student PPO.training")
    source_init = _source_init(identity, training)
    variant_raw = source_init.get("variant")
    if variant_raw not in {"n040", "no_z", "fk"}:
        raise ValueError("student PPO source initialization must declare n040/no_z/fk variant")
    variant = cast(FamilyRotationVariant, variant_raw)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"student PPO variant mismatch: expected {expected_variant!r}, got {variant!r}")

    declared_source_sha = _sha256_value(
        source_init.get("checkpoint_sha256"), "student_initialization.checkpoint_sha256"
    )
    if expected_source_actor_sha256 is not None and declared_source_sha != _sha256_value(
        expected_source_actor_sha256, "expected_source_actor_sha256"
    ):
        raise ValueError("student PPO source Actor SHA disagrees with expected_source_actor_sha256")
    source_raw = _nonempty_string(source_init.get("checkpoint_path"), "student_initialization.checkpoint_path")
    source_path = _resolve_declared_path(source_raw, checkpoint_path)
    source_metadata: dict[str, Any] = {
        "declared_path": source_raw,
        "declared_sha256": declared_source_sha,
        "available": source_path.is_file(),
    }
    if source_path.is_file():
        source_metadata.update(
            _validate_source_actor(
                source_path,
                declared_sha=declared_source_sha,
                variant=variant,
                n040_sha=n040_sha,
            )
        )

    continuation_value = payload.get("anymani_student_continuation")
    continuation = cast(dict[str, Any], dict(_mapping(continuation_value, "anymani_student_continuation")))
    if continuation.get("schema_version") != STUDENT_PPO_CONTINUATION_SCHEMA_VERSION:
        raise ValueError("student PPO continuation schema must be 1.0.0")
    continuation_sha = _sha256_value(
        continuation.get("initial_actor_checkpoint_sha256"),
        "anymani_student_continuation.initial_actor_checkpoint_sha256",
    )
    if continuation_sha != declared_source_sha:
        raise ValueError("student PPO continuation source Actor SHA disagrees with method identity")
    continuation_path = _nonempty_string(
        continuation.get("initial_actor_checkpoint_path"),
        "anymani_student_continuation.initial_actor_checkpoint_path",
    )
    if continuation_path != source_raw:
        raise ValueError("student PPO continuation source Actor path disagrees with method identity")
    for key in ("warmup", "stats", "anchor_dataset_paths", "anchor_source_hashes", "anchor_sampler"):
        if key not in continuation:
            raise ValueError(f"student PPO continuation misses complete training field {key!r}")

    model = _mapping(payload.get("model"), "model")
    actor_state: dict[str, torch.Tensor] = {}
    for key, value in model.items():
        if isinstance(key, str) and key.startswith(STUDENT_PPO_ACTOR_PREFIX):
            actor_state[key[len(STUDENT_PPO_ACTOR_PREFIX) :]] = cast(torch.Tensor, value)
    if not actor_state:
        raise ValueError(
            "student PPO checkpoint has no a2c_network.package.actor state; teacher Actor or IL policy was supplied"
        )
    _assert_finite_state(actor_state, "Actor state")

    actor_config_value = source_init.get("actor_config") or payload.get("student_actor_config")
    if actor_config_value is None:
        actor_config = FamilyStudentConfig(
            variant=variant,
            initial_log_std=float(training.get("initial_log_std", -0.5)),
            max_log_std=float(training.get("max_log_std", -0.43)),
        )
    else:
        config_map = _mapping(actor_config_value, "student actor_config")
        required = {
            "variant",
            "initial_log_std",
            "max_log_std",
            "history_encoder",
            "history_length",
            "local_skip",
            "sigma_mode",
            "phase_clock_enabled",
            "joint_kinematics_width",
            "joint_origin_width",
            "link_length_m",
        }
        missing = sorted(required.difference(config_map))
        if missing:
            raise ValueError(f"student PPO actor_config misses keys {missing}")
        if config_map.get("variant") != variant or config_map.get("representation", variant) != variant:
            raise ValueError("student PPO actor_config variant/representation disagrees")
        try:
            actor_config = FamilyStudentConfig(
                variant=variant,
                initial_log_std=float(cast(Any, config_map["initial_log_std"])),
                max_log_std=float(cast(Any, config_map["max_log_std"])),
                history_encoder=str(config_map["history_encoder"]),  # type: ignore[arg-type]
                history_length=int(cast(Any, config_map["history_length"])),
                local_skip=bool(config_map["local_skip"]),
                sigma_mode=str(config_map["sigma_mode"]),  # type: ignore[arg-type]
                phase_clock_enabled=bool(config_map["phase_clock_enabled"]),
                joint_kinematics_width=int(cast(Any, config_map["joint_kinematics_width"])),
                joint_origin_width=int(cast(Any, config_map["joint_origin_width"])),
                link_length_m=float(cast(Any, config_map["link_length_m"])),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("student PPO actor_config is invalid") from error

    actor = build_family_student(variant, config=actor_config, device=device)
    try:
        actor.load_state_dict(actor_state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"student PPO Actor state/config mismatch: {error}") from error
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    if actor.global_log_std.ndim != 0 or not bool(torch.isfinite(actor.global_log_std).all().item()):
        raise ValueError("student PPO Actor global_log_std must be one finite scalar")


    candidate = source_init.get("candidate")
    if isinstance(candidate, Mapping) and "rl_log_std" in candidate:
        expected_log_std = float(cast(Any, candidate["rl_log_std"]))
        if abs(float(actor.global_log_std.item()) - expected_log_std) > 1.0e-5:
            raise ValueError("student PPO global_log_std disagrees with student candidate identity")


    actor_config_dict = cast(dict[str, object], actor_config.as_dict())
    checkpoint_progress = {key: payload[key] for key in ("epoch", "frame", "last_mean_rewards") if key in payload}
    return FrozenStudentActorArtifact(
        actor=actor,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_path),
        identity=cast(dict[str, Any], _json_safe(identity)),
        continuation=cast(dict[str, Any], _json_safe(continuation)),
        source_initialization=cast(dict[str, Any], _json_safe({**dict(source_init), **source_metadata})),
        actor_config=actor_config_dict,
        checkpoint_progress=cast(dict[str, Any], _json_safe(checkpoint_progress)),
        variant=variant,
    )


def _validate_observation_tensor(
    value: Any,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    'Validate observation tensor.'

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != expected {shape}")
    if value.device != device:
        raise ValueError(f"{name} device {value.device} != expected {device}")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype {value.dtype} != expected {dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _validate_graph_tensor(
    value: Any,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    'Validate graph tensor.'

    if not isinstance(value, torch.Tensor) or value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError(f"{name} must be an integer graph bucket tensor")
    _validate_observation_tensor(value, name, shape, device=device, dtype=value.dtype)
    if bool((value < 0).any().item()):
        raise ValueError(f"{name} graph bucket values must be non-negative")
    return value.to(dtype=torch.long)


def _exact_zero(value: torch.Tensor, name: str) -> None:
    'Handle exact zero.'

    if value.numel() and bool(torch.count_nonzero(value).item()):
        raise ValueError(f"{name} ghost/padding values must be exactly zero")


def _binary(value: torch.Tensor, name: str) -> None:
    'Handle binary.'

    if not bool(torch.isfinite(value).all().item()) or not bool(((value == 0.0) | (value == 1.0)).all().item()):
        raise ValueError(f"{name} must be finite binary FP32")


class FrozenPpoStudent:
    'Contract for frozen PPO student.'

    def __init__(
        self,
        student_checkpoint: str | os.PathLike[str],
        *,
        device: torch.device | str = "cpu",
        expected_variant: FamilyRotationVariant | None = None,
        expected_n040_sha256: str | None = None,
        expected_source_actor_sha256: str | None = None,
    ) -> None:
        'Initialize the instance.'

        artifact = load_frozen_student_actor(
            student_checkpoint,
            device=device,
            expected_variant=expected_variant,
            expected_n040_sha256=expected_n040_sha256,
            expected_source_actor_sha256=expected_source_actor_sha256,
        )
        self._artifact = artifact
        self._actor = artifact.actor
        self.student_checkpoint = artifact.checkpoint_path
        self._student_sha256 = artifact.checkpoint_sha256
        self._student_identity = artifact.identity
        self._continuation = artifact.continuation
        self._source_initialization = artifact.source_initialization
        self._actor_config = artifact.actor_config
        self.variant = artifact.variant
        self._device = torch.device(device)
        self._allow_device_migration = (
            self._device.type == "cpu"
        )  # direct evaluator may discover CUDA only after env reset
        self._wrapper = FamilyStudentTorchScriptWrapper(self._actor).to(device=self._device, dtype=torch.float32).eval()
        self._started = False
        self._finished = False
        self._step = 0
        self._executed_steps = 0
        self._steps = 0
        self._replicas = 0
        self._asset_count = 0
        self._kinematics_features: torch.Tensor | None = None
        self._kinematics_sha256: str | None = None
        self._module_snapshot: dict[str, torch.Tensor] = {}
        self._reference_checkpoint: Path | None = None
        self._reference_checkpoint_sha256: str | None = None
        self._reference_identity: dict[str, Any] = {}
        self._runtime_identity: dict[str, Any] = {}
        self._cohort_path: Path | None = None
        self._cohort_members: list[Any] = []
        self._export_metadata: dict[str, Any] | None = None
        self._helper_source_sha256 = _sha256(Path(__file__).resolve())
        self._metadata: dict[str, Any] = {
            "artifact_type": STUDENT_PPO_ARTIFACT_TYPE,
            "schema_version": STUDENT_PPO_SCHEMA_VERSION,
            "student_checkpoint_path": str(self.student_checkpoint),
            "student_checkpoint_sha256": self._student_sha256,
            "student_method_identity_digest": artifact.method_identity_digest,
            "method_identity_digest": artifact.method_identity_digest,
            "variant": self.variant,
            "representation": self.variant,
            "n040_sha256": artifact.n040_sha256,
            "source_actor": dict(self._source_initialization),
            "actor_config": dict(self._actor_config),
            "checkpoint_progress": dict(self._artifact.checkpoint_progress),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            "precision": {"dtype": "float32", "amp": False, "tf32": False},
            "deterministic": True,
            "teacher_actor_used": False,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        'Handle metadata.'

        return cast(dict[str, Any], json.loads(json.dumps(_json_safe(self._metadata), allow_nan=False)))

    @property
    def executed_steps(self) -> int:
        'Handle executed steps.'

        return self._executed_steps

    @property
    def actor(self) -> FamilyRotationStudentActor:
        'Handle Actor.'

        return self._actor

    def attach_export_metadata(self, sidecar: Mapping[str, Any]) -> None:
        'Handle attach export metadata.'

        if self._started:
            raise RuntimeError("cannot attach export metadata after student evaluation has started")
        payload = dict(sidecar)
        if payload.get("checkpoint_sha256") != self._student_sha256:
            raise ValueError("export sidecar checkpoint SHA disagrees with FrozenPpoStudent")
        if payload.get("variant") != self.variant or payload.get("n040_sha256") != self._artifact.n040_sha256:
            raise ValueError("export sidecar variant/N040 identity disagrees with FrozenPpoStudent")
        self._export_metadata = cast(dict[str, Any], _json_safe(payload))
        self._metadata["torchscript"] = dict(self._export_metadata)

    def _snapshot_actor(self) -> dict[str, torch.Tensor]:
        'Handle snapshot Actor.'

        snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in list(self._actor.named_parameters()) + list(self._actor.named_buffers()):
            snapshot[name] = tensor.detach().cpu().clone()
        return snapshot

    def _validate_reference_compatibility(
        self,
        checkpoint_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
    ) -> None:
        'Validate reference compatibility.'

        student_n040 = _identity_retained_sha(self._student_identity, "student PPO")
        teacher_n040 = _identity_retained_sha(checkpoint_identity, "reference teacher")
        runtime_n040 = _identity_retained_sha(runtime_identity, "runtime")
        if student_n040 != teacher_n040 or student_n040 != runtime_n040:
            raise ValueError("student/reference/runtime retained N040 SHA identities disagree")
        for name, identity in (("reference teacher", checkpoint_identity), ("runtime", runtime_identity)):
            _validate_common_identity(identity, name)
            if identity.get("task_id") != self._student_identity.get("task_id"):
                raise ValueError(f"student and {name} task_id identities disagree")
            policy = _mapping(identity.get("policy"), f"{name}.policy")
            if policy.get("arm") != "direct_token" or policy.get("actor_contact") != "tip-only-binary":
                raise ValueError(f"{name} does not expose the student direct-token TIP-only route")
            authority = policy.get("action_authority_rad_per_policy_step")
            if abs(float(cast(Any, authority)) - ACTION_AUTHORITY_RAD_PER_STEP) > 1.0e-12:
                raise ValueError(f"{name} action authority disagrees with student ABI")
        # student training manifest is deliberately 256 while strict evaluation may use 128; compare only
        # transport shapes that are common to both identities, leaving joint_kinematics as student-only input.
        student_transport = self._student_identity.get("transport_abi")
        reference_transport = checkpoint_identity.get("transport_abi")
        if isinstance(student_transport, Mapping) and isinstance(reference_transport, Mapping):
            for group in ("float_shapes", "bool_shapes", "int16_shapes"):
                student_group = student_transport.get(group)
                reference_group = reference_transport.get(group)
                if not isinstance(student_group, Mapping) or not isinstance(reference_group, Mapping):
                    continue
                for key, shape in reference_group.items():
                    if key in student_group and student_group[key] != shape:
                        raise ValueError(f"student/reference transport ABI disagrees at {group}.{key}")

    def _build_kinematics(self, binding: Any) -> torch.Tensor:
        'Build kinematics; shapes [A,16,15].'

        bank = binding_joint_kinematics_bank(binding, device=self._device)
        features = getattr(bank, "features", None)
        if not isinstance(features, torch.Tensor):
            raise ValueError("student kinematics bank must expose tensor features")
        if features.ndim != 3 or tuple(features.shape[1:]) != (16, 15):
            raise ValueError(f"student kinematics bank shape {tuple(features.shape)} != expected [A,16,15]")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("student kinematics bank contains non-finite values")
        result = features.to(device=self._device, dtype=torch.float32).contiguous()
        self._kinematics_sha256 = hashlib.sha256(result.cpu().numpy().tobytes()).hexdigest()
        return result

    def _prototype_index(self, observation: Mapping[str, Any], batch: int) -> torch.Tensor:
        'Handle prototype index.'

        value = observation.get("prototype_index")
        if value is None:
            if self._asset_count < 1:
                raise RuntimeError("student asset axis is not initialized")
            return torch.arange(batch, device=self._device, dtype=torch.long) % self._asset_count
        if not isinstance(value, torch.Tensor) or value.device != self._device:
            raise ValueError("prototype_index must be a tensor on the student device")
        if tuple(value.shape) not in {(batch,), (batch, 1)}:
            raise ValueError(f"prototype_index shape {tuple(value.shape)} != expected [{batch}] or [{batch},1]")
        if value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("prototype_index must use an integer storage dtype")
        result = value.reshape(-1).to(dtype=torch.long)
        if bool((result < 0).any().item()) or bool((result >= self._asset_count).any().item()):
            raise ValueError("prototype_index lies outside the static student asset bank")
        return result

    def _prepare_inputs(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
        'Prepare inputs.'

        if self._kinematics_features is None:
            raise RuntimeError("student kinematics bank is not initialized; call start first")
        current_raw = observation.get("actor_jnt_current")
        if not isinstance(current_raw, torch.Tensor) or current_raw.ndim != 3:
            raise ValueError("actor_jnt_current must have shape [B,16,5]")
        batch = int(current_raw.shape[0])
        if batch != self._asset_count * self._replicas:
            raise ValueError(
                f"student observation batch {batch} != asset_count*replicas {self._asset_count * self._replicas}"
            )
        device = self._device
        current = _validate_observation_tensor(
            current_raw, "actor_jnt_current", (batch, 16, 5), device=device, dtype=torch.float32
        )
        history = _validate_observation_tensor(
            observation.get("actor_jnt_history"),
            "actor_jnt_history",
            (batch, 30, 16, 5),
            device=device,
            dtype=torch.float32,
        )
        limits = _validate_observation_tensor(
            observation.get("actor_jnt_limits"), "actor_jnt_limits", (batch, 16, 2), device=device, dtype=torch.float32
        )
        owner_contact = _validate_observation_tensor(
            observation.get("actor_owner_contact"),
            "actor_owner_contact",
            (batch, 21, 1),
            device=device,
            dtype=torch.float32,
        )
        jnt_valid = _validate_observation_tensor(
            observation.get("jnt_valid"), "jnt_valid", (batch, 16), device=device, dtype=torch.bool
        )
        tip_valid = _validate_observation_tensor(
            observation.get("tip_valid"), "tip_valid", (batch, 4), device=device, dtype=torch.bool
        )
        owner_valid = _validate_observation_tensor(
            observation.get("owner_valid"), "owner_valid", (batch, 21), device=device, dtype=torch.bool
        )
        geometry_tokens = _validate_observation_tensor(
            observation.get("geometry_tokens"), "geometry_tokens", (batch, 21, 128), device=device, dtype=torch.float32
        )
        shortest_path = _validate_graph_tensor(
            observation.get("shortest_path"), "shortest_path", (batch, 21, 21), device=device
        )
        parent_direction = _validate_graph_tensor(
            observation.get("parent_direction"), "parent_direction", (batch, 21, 21), device=device
        )
        child_direction = _validate_graph_tensor(
            observation.get("child_direction"), "child_direction", (batch, 21, 21), device=device
        )

        expected_owner = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=device), jnt_valid, tip_valid), dim=-1
        )
        if not torch.equal(owner_valid, expected_owner):
            raise ValueError("owner_valid must equal PALM/JOINT/TIP validity concatenation")
        _binary(owner_contact, "actor_owner_contact")
        _binary(current[..., 3:], "actor_jnt_current contact channels")
        _binary(history[..., 3:], "actor_jnt_history contact channels")
        if bool((limits[..., 0] > limits[..., 1]).any().item()):
            raise ValueError("actor_jnt_limits lower bound must not exceed upper bound")
        invalid_joint = ~jnt_valid
        _exact_zero(current[invalid_joint], "actor_jnt_current")
        _exact_zero(history[invalid_joint[:, None, :, None].expand_as(history)], "actor_jnt_history")
        _exact_zero(limits[invalid_joint], "actor_jnt_limits")
        _exact_zero(owner_contact[~owner_valid], "actor_owner_contact")
        _exact_zero(geometry_tokens[~owner_valid], "geometry_tokens")
        if not bool(torch.allclose(history[:, -1], current, rtol=0.0, atol=1.0e-6)):
            raise ValueError("actor_jnt_history latest frame must equal actor_jnt_current")

        prototype = self._prototype_index(observation, batch)
        kinematics = self._kinematics_features.index_select(0, prototype)
        _exact_zero(kinematics[invalid_joint], "joint_kinematics")
        if not bool(torch.isfinite(kinematics).all().item()):
            raise ValueError("joint_kinematics input must be finite FP32")
        provided_kinematics = observation.get("joint_kinematics")
        if provided_kinematics is not None:
            supplied = _validate_observation_tensor(
                provided_kinematics, "joint_kinematics", (batch, 16, 15), device=device, dtype=torch.float32
            )
            if not torch.equal(supplied, kinematics):
                raise ValueError("observation joint_kinematics disagrees with prototype-index static evidence")
        return (
            current,
            history,
            limits,
            owner_contact,
            jnt_valid,
            tip_valid,
            owner_valid,
            geometry_tokens,
            shortest_path,
            parent_direction,
            child_direction,
            kinematics,
        )

    def start(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        binding: Any,
        observation: Mapping[str, Any],
        cohort_path: str | Path,
        cohort_members: Sequence[Any],
        steps: int,
        replicas: int,
        actor: Any = None,
    ) -> None:
        'Handle start.'

        del actor
        if self._started:
            raise RuntimeError("FrozenPpoStudent.start may only occur once")
        if not isinstance(checkpoint_identity, Mapping) or not isinstance(runtime_identity, Mapping):
            raise ValueError("checkpoint_identity and runtime_identity must be mappings")
        teacher = Path(checkpoint_path).expanduser().resolve()
        cohort = Path(cohort_path).expanduser().resolve()
        if not teacher.is_file() or not cohort.is_file():
            raise FileNotFoundError("student evaluation reference checkpoint/cohort is missing")
        try:
            step_count = operator.index(steps)
            replica_count = operator.index(replicas)
        except (TypeError, ValueError) as error:
            raise ValueError("student evaluation steps/replicas must be integers") from error
        if step_count < 1 or replica_count < 1:
            raise ValueError("student evaluation steps/replicas must be positive")
        self._validate_reference_compatibility(checkpoint_identity, runtime_identity)
        current = observation.get("actor_jnt_current")
        if not isinstance(current, torch.Tensor) or current.ndim < 1:
            raise ValueError("student start observation must expose actor_jnt_current batch")
        # CLI construction intentionally defaults to CPU so it remains Kit-free; once the fixed evaluator has
        # created its environment, migrate that explicitly CPU-loaded Actor to the actual observation device.
        observation_device = current.device
        if observation_device != self._device:
            if not self._allow_device_migration:
                raise ValueError(
                    f"student Actor device {self._device} disagrees with runtime observation {observation_device}"
                )
            self._device = observation_device
            self._actor.to(device=self._device, dtype=torch.float32)
            self._wrapper.to(device=self._device, dtype=torch.float32)
        self._kinematics_features = self._build_kinematics(binding)
        self._asset_count = int(self._kinematics_features.shape[0])
        batch = int(current.shape[0])
        if batch != self._asset_count * replica_count:
            raise ValueError(
                f"student start observation batch {batch} != asset_count*replicas {self._asset_count * replica_count}"
            )
        self._steps = step_count
        self._replicas = replica_count
        self._reference_checkpoint = teacher
        self._reference_checkpoint_sha256 = _sha256(teacher)
        self._reference_identity = cast(dict[str, Any], _json_safe(dict(checkpoint_identity)))
        self._runtime_identity = cast(dict[str, Any], _json_safe(dict(runtime_identity)))
        self._cohort_path = cohort
        self._cohort_members = cast(list[Any], _json_safe(list(cohort_members)))
        self._module_snapshot = self._snapshot_actor()
        self._prepare_inputs(observation)
        self._metadata.update({
            "started": True,
            "steps": step_count,
            "replicas": replica_count,
            "asset_count": self._asset_count,
            "device": str(self._device),
            "joint_kinematics_shape": list(self._kinematics_features.shape),
            "joint_kinematics_dtype": str(self._kinematics_features.dtype).replace("torch.", ""),
            "joint_kinematics_sha256": self._kinematics_sha256,
            "runtime_n040_sha256": _identity_retained_sha(runtime_identity, "runtime"),
            "reference_teacher_identity_digest": checkpoint_identity.get("identity_digest"),
            "runtime_identity_digest": runtime_identity.get("identity_digest"),
        })
        self._started = True

    def act(self, step: int, observation: Mapping[str, Any]) -> torch.Tensor:
        'Handle act; shapes [B,16].'

        if not self._started or self._finished:
            raise RuntimeError("FrozenPpoStudent.act requires a started, unfinished evaluator")
        if isinstance(step, (bool, torch.Tensor)):
            raise ValueError("step must be an integer")
        try:
            index = operator.index(step)
        except (TypeError, ValueError) as error:
            raise ValueError("step must be an integer") from error
        if index != self._step or index < 0 or index >= self._steps:
            raise ValueError(
                f"student act step must be contiguous in [0,{self._steps}), expected {self._step}, got {index}"
            )
        inputs = self._prepare_inputs(observation)
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad():
                output = self._wrapper(*inputs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn
        if output.shape != (inputs[0].shape[0], 16) or output.dtype != torch.float32:
            raise ValueError(f"student Actor output shape/dtype {tuple(output.shape)}/{output.dtype} != [B,16]/float32")
        if not bool(torch.isfinite(output).all().item()):
            raise ValueError("student Actor output contains non-finite values")
        if bool((output.abs() > 1.0 + ACTION_RANGE_EPS).any().item()):
            raise ValueError("student Actor output exceeds canonical action range [-1,1]")
        _exact_zero(output[~inputs[4]], "student action")
        self._step += 1
        return output.detach()

    def after_step(self) -> None:
        'Handle after step.'

        if not self._started or self._finished:
            raise RuntimeError("FrozenPpoStudent.after_step requires a started, unfinished evaluator")
        if self._executed_steps >= self._step:
            raise RuntimeError("FrozenPpoStudent.after_step has no pending action")
        self._executed_steps += 1
        self._metadata["executed_steps"] = self._executed_steps

    def finish(self) -> None:
        'Finish the declared contract.'

        if not self._started:
            raise RuntimeError("FrozenPpoStudent.finish requires start")
        if self._finished:
            raise RuntimeError("FrozenPpoStudent.finish may only occur once")
        if self._step > self._steps or self._executed_steps != self._step:
            raise RuntimeError(
                f"student finish requires after_step for every act: acted={self._step}, executed={self._executed_steps}"
            )
        current = self._snapshot_actor()
        if set(current) != set(self._module_snapshot):
            raise RuntimeError("student Actor parameter/buffer names changed during evaluation")
        for name, before in self._module_snapshot.items():
            if (
                current[name].dtype != before.dtype
                or current[name].shape != before.shape
                or not torch.equal(current[name], before)
            ):
                raise RuntimeError(f"student Actor parameter/buffer {name!r} changed during evaluation")
        if _sha256(self.student_checkpoint) != self._student_sha256:
            raise RuntimeError("student PPO checkpoint SHA changed during evaluation")
        if self._reference_checkpoint is not None and self._reference_checkpoint_sha256 is not None:
            if _sha256(self._reference_checkpoint) != self._reference_checkpoint_sha256:
                raise RuntimeError("reference teacher checkpoint SHA changed during student evaluation")
        self._metadata["finished"] = True
        self._metadata["executed_steps"] = self._executed_steps
        self._finished = True

    def identity_updates(self) -> dict[str, Any]:
        'Handle identity updates.'

        if not self._started:
            raise RuntimeError("FrozenPpoStudent.identity_updates requires start")
        runtime_training = self._runtime_identity.get("training", {})
        runtime_precision = self._runtime_identity.get("precision")
        if not isinstance(runtime_precision, Mapping):
            runtime_precision = {
                "dtype": runtime_training.get("dtype", "float32"),
                "tf32": bool(runtime_training.get("allow_tf32", False)),
                **({"device": runtime_training["device"]} if "device" in runtime_training else {}),
            }
        training_state = {
            key: self._student_identity.get("training", {}).get(key)
            for key in ("seed", "num_envs", "asset_count", "horizon_length", "mini_epochs", "max_updates")
            if key in self._student_identity.get("training", {})
        }
        training_state.update({
            key: self._continuation.get(key)
            for key in ("rollout_phase", "anchor_microbatch_size")
            if key in self._continuation
        })

        training_state.update(self._artifact.checkpoint_progress)
        student = {
            "artifact_type": STUDENT_PPO_ARTIFACT_TYPE,
            "schema_version": STUDENT_PPO_SCHEMA_VERSION,
            "checkpoint_path": str(self.student_checkpoint),
            "checkpoint_sha256": self._student_sha256,
            "method_identity_digest": self._student_identity["identity_digest"],
            "identity": _json_safe(self._student_identity),
            "variant": self.variant,
            "representation": self.variant,
            "n040_sha256": self._artifact.n040_sha256,
            "source_actor": _json_safe(self._source_initialization),
            "actor_config": _json_safe(self._actor_config),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "continuation": _json_safe(self._continuation),
            "checkpoint_progress": _json_safe(self._artifact.checkpoint_progress),
            "training_state": _json_safe(training_state),
        }
        torchscript = (
            _json_safe(self._export_metadata)
            if self._export_metadata is not None
            else {
                "artifact_type": STUDENT_PPO_TORCHSCRIPT_ARTIFACT_TYPE,
                "exported": False,
                "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            }
        )
        reference = {
            "checkpoint_path": str(self._reference_checkpoint) if self._reference_checkpoint else None,
            "checkpoint_sha256": self._reference_checkpoint_sha256,
            "identity": _json_safe(self._reference_identity),
            "role": "teacher_reference_mdp_source_only",
        }
        evaluated_cohort = {
            "path": str(self._cohort_path) if self._cohort_path else None,
            "sha256": _sha256(self._cohort_path) if self._cohort_path else None,
            "members": _json_safe(self._cohort_members),
            "steps": self._steps,
            "replicas": self._replicas,
            "asset_count": self._asset_count,
            "role": "student_evaluation_population",
        }
        runtime = {
            "identity": _json_safe(self._runtime_identity),
            "identity_digest": self._runtime_identity.get("identity_digest"),
            "n040_sha256": _identity_retained_sha(self._runtime_identity, "runtime"),
            "actor_precision": {"dtype": "float32", "amp": False, "tf32": False},
            "reference_runtime_precision": _json_safe(runtime_precision),
            "device": str(self._device),
            "action_authority_rad_per_policy_step": ACTION_AUTHORITY_RAD_PER_STEP,
            "phase_clock": False,
            "actor_contact": "tip-only-binary",
            "teacher_actor_used": False,
            "joint_kinematics_sha256": self._kinematics_sha256,
        }
        result: dict[str, Any] = {
            "student_checkpoint_sha256": self._student_sha256,
            "student_checkpoint_path": str(self.student_checkpoint),
            "student_method_identity_digest": self._student_identity["identity_digest"],
            "method_identity_digest": self._student_identity["identity_digest"],
            "variant": self.variant,
            "representation": self.variant,
            "n040_sha256": self._artifact.n040_sha256,
            "source_actor_checkpoint_sha256": self._source_initialization["declared_sha256"],
            "source_actor_checkpoint_path": self._source_initialization["declared_path"],
            "actor_config": _json_safe(self._actor_config),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "training_state": _json_safe(training_state),
            "executed_steps": self._executed_steps,
            "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            "input_shapes": [list(shape) for shape in TORCHSCRIPT_INPUT_SHAPES],
            "input_dtypes": list(TORCHSCRIPT_INPUT_DTYPES),
            "output_shape": ["B", 16],
            "precision": {"dtype": "float32", "amp": False, "tf32": False},
            "runtime_actor_precision": {"dtype": "float32", "amp": False, "tf32": False},
            "reference_runtime_precision": _json_safe(runtime_precision),
            "document_updates": {
                "student_checkpoint_path": str(self.student_checkpoint),
                "student_checkpoint_sha256": self._student_sha256,
                "student_method_identity_digest": self._student_identity["identity_digest"],
                "source_actor_checkpoint_sha256": self._source_initialization["declared_sha256"],
                "n040_sha256": self._artifact.n040_sha256,
                "executed_steps": self._executed_steps,
                "training_state": _json_safe(training_state),
            },
            "student": student,
            "torchscript": torchscript,
            "reference_teacher": reference,
            "evaluated_cohort": evaluated_cohort,
            "runtime": runtime,
            "helper_source_sha256": self._helper_source_sha256,
            "method": {
                "artifact_type": STUDENT_PPO_ARTIFACT_TYPE,
                "schema_version": STUDENT_PPO_SCHEMA_VERSION,
                "variant": self.variant,
                "representation": self.variant,
            },
        }
        return cast(dict[str, Any], json.loads(json.dumps(_json_safe(result), allow_nan=False)))


def _torchscript_examples(device: torch.device) -> tuple[tuple[torch.Tensor, ...], ...]:
    'Handle torchscript examples.'

    examples: list[tuple[torch.Tensor, ...]] = []
    for batch, joints, tips in ((2, 16, 4), (5, 12, 3), (17, 9, 2)):
        generator = torch.Generator(device="cpu").manual_seed(1700 + batch)
        joint_valid = torch.zeros(batch, 16, dtype=torch.bool, device=device)
        joint_valid[:, :joints] = True
        tip_valid = torch.zeros(batch, 4, dtype=torch.bool, device=device)
        tip_valid[:, :tips] = True
        owner_valid = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=device), joint_valid, tip_valid), dim=-1)
        current = torch.randn(batch, 16, 5, generator=generator, device="cpu").to(device=device, dtype=torch.float32)
        history = torch.randn(batch, 30, 16, 5, generator=generator, device="cpu").to(
            device=device, dtype=torch.float32
        )
        limits = torch.stack((-torch.ones(batch, 16, device=device), torch.ones(batch, 16, device=device)), dim=-1)
        contact = torch.randint(0, 2, (batch, 21, 1), generator=generator, device="cpu", dtype=torch.int64).to(
            device=device, dtype=torch.float32
        )
        tokens = torch.randn(batch, 21, 128, generator=generator, device="cpu").to(device=device, dtype=torch.float32)
        graph = torch.randint(0, 4, (batch, 21, 21), generator=generator, device="cpu", dtype=torch.int64).to(
            device=device
        )
        kinematics = torch.randn(batch, 16, 15, generator=generator, device="cpu").to(
            device=device, dtype=torch.float32
        )

        current[~joint_valid] = 0.0
        history[:, -1] = current
        history[~joint_valid[:, None, :, None].expand_as(history)] = 0.0
        limits[~joint_valid] = 0.0
        contact[~owner_valid] = 0.0
        tokens[~owner_valid] = 0.0
        kinematics[~joint_valid] = 0.0
        examples.append((
            current,
            history,
            limits,
            contact,
            joint_valid,
            tip_valid,
            owner_valid,
            tokens,
            graph,
            graph.clone(),
            graph.clone(),
            kinematics,
        ))
    return tuple(examples)


@contextmanager
def _tf32_disabled() -> Any:
    'Handle TF32 disabled.'

    if not torch.cuda.is_available():
        yield
        return
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def export_frozen_student_actor(
    checkpoint_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    metadata_path: str | os.PathLike[str] | None = None,
    device: torch.device | str = "cpu",
    expected_variant: FamilyRotationVariant | None = None,
    expected_n040_sha256: str | None = None,
    expected_source_actor_sha256: str | None = None,
    tolerance: float = 1.0e-5,
) -> dict[str, object]:
    'Export frozen student Actor.'

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("TorchScript parity tolerance must be positive and finite")
    artifact = load_frozen_student_actor(
        checkpoint_path,
        device=device,
        expected_variant=expected_variant,
        expected_n040_sha256=expected_n040_sha256,
        expected_source_actor_sha256=expected_source_actor_sha256,
    )
    output = Path(output_path).expanduser().resolve()
    sidecar = Path(metadata_path).expanduser().resolve() if metadata_path is not None else Path(f"{output}.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"student PPO export refuses to overwrite {output} or {sidecar}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary_ts = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_json = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:

        actor = artifact.actor
        actor.eval()
        actor.fk_head = None
        wrapper = FamilyStudentTorchScriptWrapper(actor).to(device=torch.device(device), dtype=torch.float32).eval()
        examples = _torchscript_examples(torch.device(device))
        with _tf32_disabled(), torch.no_grad():
            traced = cast(
                torch.jit.ScriptModule,
                torch.jit.trace(wrapper, examples[0], check_trace=True, check_inputs=list(examples[1:]), strict=False),
            ).eval()
            traced.save(str(temporary_ts))
            loaded = cast(
                torch.jit.ScriptModule, torch.jit.load(str(temporary_ts), map_location=torch.device(device))
            ).eval()
            parity_errors: list[float] = []
            ghost_errors: list[float] = []
            range_errors: list[float] = []
            for example in examples:
                python_output = wrapper(*example)
                loaded_output = loaded(*example)
                parity_errors.append(float((python_output - loaded_output).abs().max().item()))
                ghost = torch.cat(
                    (python_output.masked_select(~example[4]).abs(), loaded_output.masked_select(~example[4]).abs())
                )
                ghost_errors.append(float(ghost.max().item()) if ghost.numel() else 0.0)
                excess = torch.cat((
                    torch.clamp(python_output.abs() - 1.0, min=0.0).reshape(-1),
                    torch.clamp(loaded_output.abs() - 1.0, min=0.0).reshape(-1),
                ))
                range_errors.append(float(excess.max().item()) if excess.numel() else 0.0)
            no_z_invariance: float | None = None
            if artifact.variant in {"no_z", "fk"}:
                changed = list(examples[0])
                changed[7] = changed[7] + 3.0
                python_delta = (wrapper(*examples[0]) - wrapper(*tuple(changed))).abs().max()
                loaded_delta = (loaded(*examples[0]) - loaded(*tuple(changed))).abs().max()
                no_z_invariance = float(torch.maximum(python_delta, loaded_delta).item())
            max_parity = max(parity_errors)
            max_ghost = max(ghost_errors, default=0.0)
            max_range = max(range_errors, default=0.0)
            if max_parity > tolerance or max_ghost > tolerance or max_range > tolerance:
                raise RuntimeError(
                    f"student PPO TorchScript parity/ghost/range failed: parity={max_parity:.3g}, "
                    f"ghost={max_ghost:.3g}, range={max_range:.3g}, tolerance={tolerance:.3g}"
                )
            if no_z_invariance is not None and no_z_invariance > tolerance:
                raise RuntimeError("student PPO no_z/fk TorchScript remains sensitive to geometry token values")
        ts_sha = _sha256(temporary_ts)
        sidecar_payload: dict[str, object] = {
            "artifact_type": STUDENT_PPO_TORCHSCRIPT_ARTIFACT_TYPE,
            "schema": STUDENT_PPO_SCHEMA_VERSION,
            "schema_version": STUDENT_PPO_SCHEMA_VERSION,
            "checkpoint_sha256": artifact.checkpoint_sha256,
            "checkpoint_path": str(artifact.checkpoint_path),
            "torchscript_sha256": ts_sha,
            "ppo_identity_digest": artifact.method_identity_digest,
            "method_identity_digest": artifact.method_identity_digest,
            "variant": artifact.variant,
            "representation": artifact.variant,
            "n040_sha256": artifact.n040_sha256,
            "source_actor": _json_safe(artifact.source_initialization),
            "source_actor_checkpoint_sha256": artifact.source_initialization["declared_sha256"],
            "source_actor_checkpoint_path": artifact.source_initialization["declared_path"],
            "checkpoint_progress": _json_safe(artifact.checkpoint_progress),
            "actor_config": _json_safe(artifact.actor_config),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            "input_shapes": [list(shape) for shape in TORCHSCRIPT_INPUT_SHAPES],
            "input_dtypes": list(TORCHSCRIPT_INPUT_DTYPES),
            "output_shape": ["B", 16],
            "precision": {"dtype": "float32", "amp": False, "tf32": False},
            "deterministic": True,
            "teacher_actor_used": False,
            "validation_batch_sizes": [2, 5, 17],
            "parity_max_abs": max_parity,
            "parity_tolerance": tolerance,
            "ghost_max_abs": max_ghost,
            "range_excess_max_abs": max_range,
            "no_z_token_invariance_max_abs": no_z_invariance,
            "auxiliary_fk_head_deployed": False,
            "loaded_file_parity": {"passed": True, "parity_max_abs": max_parity, "tolerance": tolerance},
            "torch_version": torch.__version__,
        }
        with temporary_json.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(sidecar_payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_ts, output)
        os.replace(temporary_json, sidecar)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary_ts.unlink()
        with suppress(FileNotFoundError):
            temporary_json.unlink()
        raise
    sidecar_payload["torchscript_path"] = str(output)
    sidecar_payload["metadata_path"] = str(sidecar)
    return cast(dict[str, object], _json_safe(sidecar_payload))



load_student_ppo_checkpoint = load_frozen_student_actor
export_student_ppo_actor = export_frozen_student_actor
FrozenFamilyStudentPpo = FrozenPpoStudent
FrozenStudentPpo = FrozenPpoStudent


def _main() -> int:
    'Handle main.'

    parser = argparse.ArgumentParser(description="Export a frozen AnyMani student PPO Actor.", allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Complete student PPO .pth checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="New TorchScript Actor path.")
    parser.add_argument("--metadata", type=Path, default=None, help="New JSON sidecar path; defaults to <output>.json.")
    parser.add_argument("--variant", choices=("n040", "no_z", "fk"), default=None)
    parser.add_argument("--n040_sha256", default=None, help="Expected retained N040 encoder checkpoint SHA-256.")
    parser.add_argument("--source_actor_sha256", default=None, help="Expected initial BC Actor checkpoint SHA-256.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = export_frozen_student_actor(
        args.checkpoint,
        args.output,
        metadata_path=args.metadata,
        device=args.device,
        expected_variant=args.variant,
        expected_n040_sha256=args.n040_sha256,
        expected_source_actor_sha256=args.source_actor_sha256,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ACTION_AUTHORITY_RAD_PER_STEP",
    "FAMILY_STUDENT_ACTOR_ABI",
    "FrozenFamilyStudentPpo",
    "FrozenPpoStudent",
    "FrozenStudentActorArtifact",
    "FrozenStudentPpo",
    "STUDENT_PPO_ACTOR_PREFIX",
    "STUDENT_PPO_ARTIFACT_TYPE",
    "STUDENT_PPO_SCHEMA_VERSION",
    "STUDENT_PPO_TORCHSCRIPT_ARTIFACT_TYPE",
    "TORCHSCRIPT_INPUT_ABI",
    "TORCHSCRIPT_INPUT_DTYPES",
    "TORCHSCRIPT_INPUT_SHAPES",
    "export_frozen_student_actor",
    "export_student_ppo_actor",
    "load_frozen_student_actor",
    "load_student_ppo_checkpoint",
]
