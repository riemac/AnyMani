r"""三组 family student 的独立 IL artifact 边界。

学生模型在 ``models.family_rotation_policy`` 中复用 ``PalmRotationDirectActor`` 的 History30、TIP-only
contact、16-action direct-token 和 graph-biased owner backbone。本模块只负责三件事：固定 artifact 的
身份/config 合同、构造随机初始化 student、以及严格保存/加载 checkpoint。它不导入 Isaac、rl_games、
critic 或 teacher checkpoint。

``n040`` 使用采集时缓存的 FP32 N040 owner token；``no_z`` 与 ``fk`` 在模型 forward 内将同一 token 数值
置零，但保留 owner mask、graph、limits、current/history/contact 和 joint kinematics。三组的 FK 变体身份
独立保存，不能拿一个已训练 Ours checkpoint 在评价时临时清零来冒充 no_z/fk。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from anymani.distill.models.family_rotation_policy import (
    FAMILY_ROTATION_VARIANTS,
    JOINT_KINEMATICS_WIDTH,
    JOINT_ORIGIN_WIDTH,
    LINK_LENGTH_M,
    FamilyRotationActorOutput,
    FamilyRotationStudentActor,
    FamilyRotationVariant,
    build_family_rotation_policy,
)
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry

FAMILY_STUDENT_ARTIFACT_TYPE = "anymani.family_distilled_actor"
FAMILY_STUDENT_SCHEMA_VERSION = "1.0.0"
Representation = FamilyRotationVariant

# Python loader 需要知道模型语义闭包，而不只知道本文件的 SHA；这些依赖共同决定 state dict/forward ABI。
FAMILY_STUDENT_SOURCE_FILES: tuple[str, ...] = (
    "source/anymani/anymani/distill/models/family_rotation_policy.py",
    "source/anymani/anymani/distill/models/palm_rotation_policy.py",
    "source/anymani/anymani/distill/models/temporal_encoder.py",
    "source/anymani/anymani/distill/models/backbones/geometry_transformer.py",
    "source/anymani/anymani/distill/il/family_dataset.py",
    "source/anymani/anymani/distill/il/family_student.py",
    "source/anymani/anymani/distill/il/train_family.py",
)

# actor_abi 是 collector、dataset、student 和 evaluator 共用的稳定字段，不随 student 超参变化。
FAMILY_STUDENT_ACTOR_ABI: dict[str, object] = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": 30,
    "joint_count": 16,
    "owner_count": 21,
    "geometry_width": 128,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": 15,
}


def family_student_source_code_files() -> dict[str, str]:
    r"""返回 student Python 语义依赖闭包的逐文件 SHA-256。"""

    root = Path(__file__).resolve().parents[5]
    files: dict[str, str] = {}
    for relative in FAMILY_STUDENT_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"family student source dependency is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[relative] = digest.hexdigest()
    return files


def family_student_source_code_hash(files: Mapping[str, str] | None = None) -> str:
    r"""对逐文件 source closure 形成稳定 digest，供 artifact lineage 使用。"""

    payload = dict(files) if files is not None else family_student_source_code_files()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FamilyStudentConfig:
    r"""三组共享 actor 的可恢复配置。

    direct-token 动作路径为

    $$
    \mu_{t,j}=\tanh f_{direct}(H^a_{t,j}),
    \qquad H^a_{t,j}\in\mathbb{R}^{128}.
    $$

    ``joint_kinematics`` 的 15 维静态向量经过 ``Linear(15,128)`` 零初始化后加到 JOINT owner token；
    FK 变体的 origin 目标先除以 $L=0.1$ m。``global_log_std`` 取默认 $-0.5$、上限 $-0.43$，
    mean supervision 时冻结。
    """

    variant: Representation = "n040"
    initial_log_std: float = -0.5
    max_log_std: float = -0.43
    history_encoder: Literal["tcn"] = "tcn"
    history_length: int = 30
    local_skip: bool = False
    sigma_mode: Literal["global"] = "global"
    phase_clock_enabled: bool = False
    joint_kinematics_width: int = JOINT_KINEMATICS_WIDTH
    joint_origin_width: int = JOINT_ORIGIN_WIDTH
    link_length_m: float = LINK_LENGTH_M

    @property
    def representation(self) -> Representation:
        r"""返回与旧 IL 命名兼容的 variant 只读别名；不引入第二个可变身份字段。"""

        return self.variant

    def __post_init__(self) -> None:
        r"""在 config 构造点拒绝会改变三组 actor ABI 的值。"""

        if self.variant not in FAMILY_ROTATION_VARIANTS:
            raise ValueError(f"family student variant must be one of {FAMILY_ROTATION_VARIANTS}, got {self.variant!r}")
        if self.history_encoder != "tcn" or self.history_length != 30:
            raise ValueError("family student ABI requires History30 TCN")
        if self.local_skip:
            raise ValueError("family student ABI requires local_skip=False (direct_token)")
        if self.sigma_mode != "global" or self.phase_clock_enabled:
            raise ValueError("family student ABI requires global sigma and phase_clock_enabled=False")
        if self.joint_kinematics_width != 15 or self.joint_origin_width != 3:
            raise ValueError("family student ABI requires joint kinematics width 15 and FK target width 3")
        if not math.isfinite(self.initial_log_std) or not math.isfinite(self.max_log_std):
            raise ValueError("family student log_std values must be finite")
        if self.initial_log_std > self.max_log_std:
            raise ValueError("family student initial_log_std must not exceed max_log_std")
        if self.link_length_m != LINK_LENGTH_M:
            raise ValueError("family student link_length_m is fixed at 0.1 m")

    def as_dict(self) -> dict[str, object]:
        r"""返回 JSON-safe 的完整 actor config，供 checkpoint 和 resume identity 使用。"""

        return {
            "variant": self.variant,
            "representation": self.variant,
            "initial_log_std": float(self.initial_log_std),
            "max_log_std": float(self.max_log_std),
            "history_encoder": self.history_encoder,
            "history_length": self.history_length,
            "local_skip": self.local_skip,
            "sigma_mode": self.sigma_mode,
            "phase_clock_enabled": self.phase_clock_enabled,
            "joint_kinematics_width": self.joint_kinematics_width,
            "joint_origin_width": self.joint_origin_width,
            "link_length_m": float(self.link_length_m),
        }


def _config_from_mapping(value: Mapping[str, object], *, expected_variant: str | None = None) -> FamilyStudentConfig:
    r"""严格解析 checkpoint config，避免缺字段时悄悄构造另一种 student。"""

    keys = {
        "variant",
        "representation",
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
    missing = keys - set(value)
    if missing:
        raise ValueError(f"family student checkpoint actor_config misses keys {sorted(missing)}")
    variant = str(value["variant"])
    if str(value["representation"]) != variant:
        raise ValueError("family student actor_config variant/representation disagree")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"family student variant mismatch: expected {expected_variant!r}, got {variant!r}")
    try:
        return FamilyStudentConfig(
            variant=variant,  # type: ignore[arg-type]
            initial_log_std=float(cast(Any, value["initial_log_std"])),
            max_log_std=float(cast(Any, value["max_log_std"])),
            history_encoder=str(value["history_encoder"]),  # type: ignore[arg-type]
            history_length=int(cast(Any, value["history_length"])),
            local_skip=bool(value["local_skip"]),
            sigma_mode=str(value["sigma_mode"]),  # type: ignore[arg-type]
            phase_clock_enabled=bool(value["phase_clock_enabled"]),
            joint_kinematics_width=int(cast(Any, value["joint_kinematics_width"])),
            joint_origin_width=int(cast(Any, value["joint_origin_width"])),
            link_length_m=float(cast(Any, value["link_length_m"])),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid family student actor_config: {value!r}") from exc


def build_family_student(
    variant: Representation | None = None,
    *,
    representation: Representation | None = None,
    device: torch.device | str | None = None,
    config: FamilyStudentConfig | None = None,
) -> FamilyRotationStudentActor:
    r"""构造指定 variant 的随机初始化 FP32 student。

    ``variant`` 独立决定训练身份：``n040`` 保留 Z，``no_z``/``fk`` 在 forward 清零 Z，``fk`` 额外拥有
    origin/L 辅助 head。所有参数从当前随机流初始化，不从 accepted teacher Actor/Critic 继承。
    """

    if representation is not None:
        if variant is not None and variant != representation:
            raise ValueError(f"variant {variant!r} disagrees with representation {representation!r}")
        variant = representation
    if variant is None:
        variant = config.variant if config is not None else "n040"
    effective_config = config or FamilyStudentConfig(variant=variant)
    if effective_config.variant != variant:
        raise ValueError(f"config variant {effective_config.variant!r} disagrees with requested {variant!r}")
    actor = build_family_rotation_policy(
        effective_config.variant,
        device=device,
        initial_log_std=effective_config.initial_log_std,
        max_log_std=effective_config.max_log_std,
    )
    # 模型构造时记录完整 config，避免 optimizer/checkpoint 依赖外部可变对象。
    actor.family_actor_config = effective_config.as_dict()
    actor.global_log_std.requires_grad_(False)
    return actor


def _assert_finite_state_dict(state_dict: Mapping[str, object]) -> None:
    r"""严格检查 actor state 的 tensor 类型、FP32 dtype 与有限性。"""

    if not state_dict:
        raise ValueError("family student checkpoint actor_state_dict is empty")
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("family student actor_state_dict must map string keys to tensors")
        if value.is_floating_point() or value.is_complex():
            if value.dtype != torch.float32:
                raise ValueError(f"family student state {name!r} must be FP32, got {value.dtype}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"family student state {name!r} contains non-finite values")


def _state_dict_cpu(actor: FamilyRotationStudentActor) -> dict[str, torch.Tensor]:
    r"""复制 checkpoint state 到 CPU，保持已写 artifact 不受后续 optimizer 更新影响。"""

    state = actor.state_dict()
    _assert_finite_state_dict(state)
    return {key: value.detach().to(device="cpu", dtype=torch.float32).clone() for key, value in state.items()}


def save_family_student_checkpoint(
    path: str | os.PathLike[str],
    actor: FamilyRotationStudentActor,
    *,
    metadata: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object] | None = None,
    training_state: Mapping[str, object] | None = None,
    rng_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    r"""atomic 写出独立 ``anymani.family_distilled_actor`` artifact。

    metadata 由 trainer 提供 dataset/source/N040 SHA、quality、split、protocol、ordered asset 与源码 hash；
    training_state 保存 epoch/update/processed_samples，rng_state 同时保存 global RNG 与 sampling generator。
    本函数不写 PPO frame counter，也不把 sigma 描述为学习得到的探索状态。
    """

    if not isinstance(actor, FamilyRotationStudentActor):
        raise TypeError("family student checkpoint requires FamilyRotationStudentActor")
    if not isinstance(metadata, Mapping):
        raise TypeError("family student checkpoint metadata must be a mapping")
    config = _config_from_mapping(actor.family_actor_config)
    payload: dict[str, object] = {
        "artifact_type": FAMILY_STUDENT_ARTIFACT_TYPE,
        "schema": FAMILY_STUDENT_SCHEMA_VERSION,
        "schema_version": FAMILY_STUDENT_SCHEMA_VERSION,
        "actor_state_dict": _state_dict_cpu(actor),
        "actor_config": config.as_dict(),
        "variant": config.variant,
        "representation": config.variant,
        "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
        "metadata": dict(metadata),
        "optimizer_state_dict": dict(optimizer_state_dict or {}),
        "training_state": dict(training_state or {}),
        "rng_state": dict(rng_state or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件 + fsync + replace，last/best 覆盖仅触及本 run 目标。
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return payload


def _load_payload(path: Path) -> Mapping[str, object]:
    r"""读取 checkpoint 根 mapping；optimizer/RNG 容器属于受控恢复数据。"""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError(f"cannot load family student checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("family student checkpoint root must be a mapping")
    return payload


def load_family_student(
    path: str | os.PathLike[str],
    device: torch.device | str = "cpu",
    *,
    expected_schema: str = FAMILY_STUDENT_SCHEMA_VERSION,
    expected_variant: Representation | None = None,
    expected_representation: Representation | None = None,
    expected_dataset_sha256: str | None = None,
    expected_n040_sha256: str | None = None,
    expected_teacher_checkpoint_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[FamilyRotationStudentActor, dict[str, object]]:
    r"""严格加载并返回 ``(actor, metadata)``，供 evaluator 显式载入独立 IL artifact。

    loader 验证 artifact/schema、actor ABI、完整 config、strict state keys、FP32 finite 参数及可选 source
    identity。``global_log_std`` 在返回 actor 上冻结；FK 预测只作为 output 辅助字段，不进入 action API。
    """

    checkpoint_path = Path(path)
    if expected_representation is not None:
        if expected_variant is not None and expected_variant != expected_representation:
            raise ValueError("expected_variant and expected_representation disagree")
        expected_variant = expected_representation
    payload = _load_payload(checkpoint_path)
    required_keys = {
        "artifact_type",
        "schema",
        "schema_version",
        "actor_state_dict",
        "actor_config",
        "variant",
        "representation",
        "actor_abi",
        "metadata",
        "optimizer_state_dict",
        "training_state",
        "rng_state",
    }
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(f"family student checkpoint misses required keys {sorted(missing_keys)}")
    if payload.get("artifact_type") != FAMILY_STUDENT_ARTIFACT_TYPE:
        raise ValueError(f"family student checkpoint artifact_type mismatch: {payload.get('artifact_type')!r}")
    schema = payload.get("schema", payload.get("schema_version"))
    if "schema" in payload and "schema_version" in payload and payload["schema"] != payload["schema_version"]:
        raise ValueError("family student checkpoint schema and schema_version disagree")
    if schema != expected_schema:
        raise ValueError(f"family student checkpoint schema mismatch: expected {expected_schema!r}, got {schema!r}")
    if schema != FAMILY_STUDENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported family student checkpoint schema {schema!r}")
    abi = payload.get("actor_abi")
    if not isinstance(abi, Mapping) or dict(abi) != FAMILY_STUDENT_ACTOR_ABI:
        raise ValueError("family student checkpoint actor_abi mismatch")
    config_payload = payload.get("actor_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("family student checkpoint actor_config must be a mapping")
    config = _config_from_mapping(config_payload, expected_variant=expected_variant)
    if payload.get("variant") != config.variant or payload.get("representation") != config.variant:
        raise ValueError("family student checkpoint top-level variant/representation disagrees with actor_config")
    state = payload.get("actor_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("family student checkpoint misses actor_state_dict")
    _assert_finite_state_dict(state)
    metadata_payload = payload.get("metadata", {})
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("family student checkpoint metadata must be a mapping")
    metadata = dict(metadata_payload)
    metadata_abi = metadata.get("actor_abi")
    if metadata_abi is not None and (not isinstance(metadata_abi, Mapping) or dict(metadata_abi) != FAMILY_STUDENT_ACTOR_ABI):
        raise ValueError("family student checkpoint metadata actor_abi mismatch")
    metadata_variant = metadata.get("variant")
    if metadata_variant is not None and metadata_variant != config.variant:
        raise ValueError("family student checkpoint metadata variant mismatch")
    stored_source_files = metadata.get("source_code_files")
    if stored_source_files is not None:
        if not isinstance(stored_source_files, Mapping):
            raise ValueError("family student checkpoint source_code_files must be a mapping")
        current_source_files = family_student_source_code_files()
        if dict(stored_source_files) != current_source_files:
            raise ValueError("family student checkpoint source code dependency drift")
        stored_source_hash = metadata.get("source_code_hash")
        if stored_source_hash is not None and stored_source_hash != family_student_source_code_hash(current_source_files):
            raise ValueError("family student checkpoint source_code_hash disagrees with source_code_files")
    expected_identities = {
        "dataset_sha256": expected_dataset_sha256,
        "n040_sha256": expected_n040_sha256,
        "teacher_checkpoint_sha256": expected_teacher_checkpoint_sha256,
        "source_sha256": expected_source_sha256,
    }
    for key, expected in expected_identities.items():
        actual = metadata.get(key)
        matches = actual == expected or (isinstance(actual, (list, tuple)) and expected in actual)
        if expected is not None and not matches:
            raise ValueError(
                f"family student checkpoint {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    actor = build_family_student(config.variant, device=device, config=config)
    try:
        actor.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"family student checkpoint actor keys/config mismatch: {exc}") from exc
    actor.global_log_std.requires_grad_(False)
    actor.eval()
    return actor, metadata


class FamilyStudentTorchScriptWrapper(torch.nn.Module):
    r"""把 student 的 12 个 tensor 输入固定为可跨 runtime 部署的动作模块。

    输入顺序严格是 ``jnt_current, jnt_history, jnt_limits, owner_contact, jnt_valid, tip_valid,
    owner_valid, geometry_tokens, shortest_path, parent_direction, child_direction, joint_kinematics``；
    输出只有 `[B,16]` mean。family/asset ID、FK auxiliary head 和 optimizer 都不进入 TorchScript。
    """

    def __init__(self, actor: FamilyRotationStudentActor) -> None:
        r"""绑定已加载的 FP32/eval student，不创建第二套参数。"""

        super().__init__()
        self.actor = actor

    def forward(
        self,
        jnt_current: torch.Tensor,
        jnt_history: torch.Tensor,
        jnt_limits: torch.Tensor,
        owner_contact: torch.Tensor,
        jnt_valid: torch.Tensor,
        tip_valid: torch.Tensor,
        owner_valid: torch.Tensor,
        geometry_tokens: torch.Tensor,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        r"""将 12 tensor 组装为 actor dataclass，并只返回 deterministic action mean。"""

        observation = PalmRotationActorObservation(
            jnt_current=jnt_current,
            jnt_history=jnt_history,
            jnt_limits=jnt_limits,
            owner_contact=owner_contact,
            jnt_valid=jnt_valid,
            tip_valid=tip_valid,
            owner_valid=owner_valid,
        )
        geometry = PalmRotationGeometry(
            tokens=geometry_tokens,
            owner_valid=owner_valid,
            shortest_path=shortest_path,
            parent_direction=parent_direction,
            child_direction=child_direction,
        )
        return self.actor(observation, geometry, joint_kinematics=joint_kinematics).mean


TORCHSCRIPT_INPUT_ABI: tuple[str, ...] = (
    "jnt_current",
    "jnt_history",
    "jnt_limits",
    "owner_contact",
    "jnt_valid",
    "tip_valid",
    "owner_valid",
    "geometry_tokens",
    "shortest_path",
    "parent_direction",
    "child_direction",
    "joint_kinematics",
)


def _checkpoint_sha256(path: Path) -> str:
    r"""流式绑定待导出 family checkpoint 的内容 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torchscript_example_inputs(device: torch.device) -> tuple[tuple[torch.Tensor, ...], ...]:
    r"""生成 B=2/5/17、DoF/finger/graph 多模式的 trace 与独立 parity 输入。"""

    examples: list[tuple[torch.Tensor, ...]] = []
    for batch, valid_joint_count, valid_tip_count in ((2, 16, 4), (5, 12, 3), (17, 9, 2)):
        generator = torch.Generator(device="cpu").manual_seed(700 + batch)
        jnt_valid = torch.zeros(batch, 16, dtype=torch.bool)
        jnt_valid[:, :valid_joint_count] = True
        tip_valid = torch.zeros(batch, 4, dtype=torch.bool)
        tip_valid[:, :valid_tip_count] = True
        owner_valid = torch.cat((torch.ones(batch, 1, dtype=torch.bool), jnt_valid, tip_valid), dim=-1)
        current = torch.randn(batch, 16, 5, generator=generator).to(device=device)
        history = torch.randn(batch, 30, 16, 5, generator=generator).to(device=device)
        limits = torch.stack(
            (-torch.ones(batch, 16, device=device), torch.ones(batch, 16, device=device)), dim=-1
        )
        contact = torch.rand(batch, 21, 1, generator=generator).to(device=device)
        tokens = torch.randn(batch, 21, 128, generator=generator).to(device=device)
        graph = torch.randint(0, 4, (batch, 21, 21), generator=generator, dtype=torch.long).to(device=device)
        kinematics = torch.randn(batch, 16, 15, generator=generator).to(device=device)
        examples.append(
            (
                current.float(),
                history.float(),
                limits.float(),
                contact.float(),
                jnt_valid.to(device),
                tip_valid.to(device),
                owner_valid.to(device),
                tokens.float(),
                graph,
                graph.clone(),
                graph.clone(),
                kinematics.float(),
            )
        )
    return tuple(examples)


@contextmanager
def _tf32_disabled() -> Any:
    r"""在导出/验证期间显式关闭 TF32，防止继承 teacher/evaluator 的全局 matmul 设置。"""

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


def export_family_student_torchscript(
    checkpoint_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
    sidecar_path: str | os.PathLike[str] | None = None,
    tolerance: float = 1.0e-5,
) -> dict[str, object]:
    r"""从真实 student checkpoint 导出动态 batch TorchScript 动作模块与 parity sidecar。

    trace 使用 B=2/5/17 和不同 DoF/finger/graph mask 做独立 Python-vs-loaded-TS 验证；若 trace 将
    batch 维固定或数值不满足 FP32 tolerance，会显式抛出错误，不生成一个看似可部署的固定 batch 文件。
    输出路径和 sidecar 已存在时拒绝覆盖正式 artifact。
    """

    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("TorchScript parity tolerance must be positive and finite")
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    sidecar = Path(sidecar_path) if sidecar_path is not None else Path(f"{output}.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"TorchScript export refuses to overwrite {output} or {sidecar}")
    target_device = torch.device(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary_ts = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_json = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    actor, metadata = load_family_student(checkpoint, device=target_device)
    actor.eval()
    if actor.fk_head is not None:
        # FK 头只服务训练监督；先从部署副本移除，避免 TorchScript artifact 携带未使用的 auxiliary weights。
        actor.fk_head = None
    wrapper = FamilyStudentTorchScriptWrapper(actor).to(device=target_device, dtype=torch.float32).eval()
    examples = _torchscript_example_inputs(target_device)
    with _tf32_disabled():
        try:
            traced = cast(
                torch.jit.ScriptModule,
                torch.jit.trace(
                    wrapper,
                    examples[0],
                    check_trace=True,
                    check_inputs=list(examples[1:]),
                    strict=False,
                ),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "TorchScript trace failed dynamic B=2/5/17 validation; no fixed-batch export was produced: "
                f"{exc}"
            ) from exc
        traced = traced.eval()
        # 先写临时 TS，再从磁盘重新加载；后续所有 parity 都针对真正的 loaded module。
        traced.save(str(temporary_ts))
        loaded = cast(torch.jit.ScriptModule, torch.jit.load(str(temporary_ts), map_location=target_device)).eval()
        parity_errors: list[float] = []
        ghost_errors: list[float] = []
        range_errors: list[float] = []
        with torch.no_grad():
            for example in examples:
                python_mean = wrapper(*example)
                loaded_mean = loaded(*example)
                parity_errors.append(float((python_mean - loaded_mean).abs().max().item()))
                ghost_values = torch.cat(
                    (python_mean.masked_select(~example[4]).abs(), loaded_mean.masked_select(~example[4]).abs())
                )
                ghost_errors.append(float(ghost_values.max().item()) if ghost_values.numel() else 0.0)
                range_values = torch.cat(
                    (
                        torch.clamp(python_mean.abs() - 1.0, min=0.0).reshape(-1),
                        torch.clamp(loaded_mean.abs() - 1.0, min=0.0).reshape(-1),
                    )
                )
                range_errors.append(float(range_values.max().item()) if range_values.numel() else 0.0)
        max_parity = max(parity_errors)
        if max_parity > tolerance:
            raise RuntimeError(f"TorchScript parity max_abs={max_parity:.3g} exceeds tolerance={tolerance:.3g}")
        if max(ghost_errors, default=0.0) > tolerance or max(range_errors, default=0.0) > tolerance:
            raise RuntimeError("TorchScript output violates ghost-zero or action-range contract")
        no_z_invariance: float | None = None
        if actor.variant in {"no_z", "fk"}:
            changed = list(examples[0])
            changed[7] = changed[7] + 3.0
            with torch.no_grad():
                python_delta = (wrapper(*examples[0]) - wrapper(*tuple(changed))).abs().max()
                scripted_delta = (loaded(*examples[0]) - loaded(*tuple(changed))).abs().max()
                no_z_invariance = float(torch.maximum(python_delta, scripted_delta).item())
            if no_z_invariance > tolerance:
                raise RuntimeError("No-Z/FK TorchScript export remains sensitive to geometry token values")
    torchscript_sha256 = _checkpoint_sha256(temporary_ts)
    sidecar_payload: dict[str, object] = {
        "artifact_type": "anymani.family_distilled_actor_torchscript",
        "schema": FAMILY_STUDENT_SCHEMA_VERSION,
        "schema_version": FAMILY_STUDENT_SCHEMA_VERSION,
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "torchscript_sha256": torchscript_sha256,
        "checkpoint_path": str(checkpoint),
        "variant": actor.variant,
        "actor_config": dict(actor.family_actor_config),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "n040_sha256": metadata.get("n040_sha256"),
        "source_sha256": metadata.get("source_sha256"),
        "source_code_files": metadata.get("source_code_files"),
        "source_code_hash": metadata.get("source_code_hash"),
        "input_abi": list(TORCHSCRIPT_INPUT_ABI),
        "input_shapes": [
            ["B", 16, 5],
            ["B", 30, 16, 5],
            ["B", 16, 2],
            ["B", 21, 1],
            ["B", 16],
            ["B", 4],
            ["B", 21],
            ["B", 21, 128],
            ["B", 21, 21],
            ["B", 21, 21],
            ["B", 21, 21],
            ["B", 16, 15],
        ],
        "input_dtypes": ["float32", "float32", "float32", "float32", "bool", "bool", "bool", "float32", "int64", "int64", "int64", "float32"],
        "output_shape": ["B", 16],
        "precision": {"dtype": "float32", "amp": False, "tf32": False},
        "torch_version": torch.__version__,
        "validation_batch_sizes": [2, 5, 17],
        "parity_max_abs": max_parity,
        "parity_tolerance": tolerance,
        "ghost_max_abs": max(ghost_errors, default=0.0),
        "range_excess_max_abs": max(range_errors, default=0.0),
        "no_z_token_invariance_max_abs": no_z_invariance,
        "auxiliary_fk_head_deployed": False,
    }
    try:
        with temporary_json.open("w", encoding="utf-8") as stream:
            json.dump(sidecar_payload, stream, ensure_ascii=False, indent=2)
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
    return sidecar_payload


# 公开短名称供运行时脚本调用；两者严格共享同一导出实现和 sidecar 合同。
export_family_student = export_family_student_torchscript


# 下游 evaluator 可能按 IL 语义引用短名称；别名不复制模型或改变 checkpoint key。
FamilyStudentActor = FamilyRotationStudentActor
build_student = build_family_student


__all__ = [
    "FAMILY_STUDENT_ACTOR_ABI",
    "FAMILY_STUDENT_ARTIFACT_TYPE",
    "FAMILY_STUDENT_SCHEMA_VERSION",
    "FAMILY_STUDENT_SOURCE_FILES",
    "FamilyStudentTorchScriptWrapper",
    "FamilyRotationActorOutput",
    "FamilyRotationStudentActor",
    "FamilyStudentActor",
    "FamilyStudentConfig",
    "build_family_student",
    "build_student",
    "export_family_student_torchscript",
    "export_family_student",
    "family_student_source_code_files",
    "family_student_source_code_hash",
    "load_family_student",
    "save_family_student_checkpoint",
    "TORCHSCRIPT_INPUT_ABI",
]
