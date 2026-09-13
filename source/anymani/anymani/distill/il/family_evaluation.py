r"""跨 LEAP/Allegro runtime 复用冻结 family student TorchScript 的评价边界。

本模块故意不 import 当前 Python actor 或当前 policy model。学生动作只能
来自独立 IL checkpoint 导出的 ``.ts``；当前 runtime 只提供真实 task observation、静态
geometry token/graph 与 source asset binding。这样旧的 LEAP c4d7 runtime 与当前 Allegro
runtime 可以调用同一份动作文件，而不会因 Python 类版本漂移重新解释 checkpoint。

TorchScript 输入顺序固定为

$$
(jnt\_current,jnt\_history,jnt\_limits,owner\_contact,
 jnt\_valid,tip\_valid,owner\_valid,geometry\_tokens,
 shortest\_path,parent\_direction,child\_direction,joint\_kinematics).
$$

前四项来自 runtime 的 actor observation，三个 mask 保持 ``torch.bool``，三个 owner graph
保持 ``torch.long``，kinematics 是由 ``joint_frames.build_joint_kinematics_bank`` 以
``float64`` 物理 source 生成后转交的 FP32 ``[B,16,15]``。family/asset ID、object state、
critic state 与 FK target 都不进入 TorchScript；asset routing 只在 evaluator 内按
``env_id % asset_count`` gather。``act`` 在 no-grad 下临时关闭 TF32，并在异常与正常返回
两条路径都恢复全局 matmul/cudnn flags。
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import torch

from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1
from anymani.distill.representations.sources.joint_frames import build_joint_kinematics_bank

# 独立 IL 与 TorchScript artifact 的格式身份，不能用 PPO checkpoint 或旧 teacher 替代。
FAMILY_STUDENT_ARTIFACT_TYPE: Final[str] = "anymani.family_distilled_actor"
TORCHSCRIPT_ARTIFACT_TYPE: Final[str] = "anymani.family_distilled_actor_torchscript"
SCHEMA_VERSION: Final[str] = "1.0.0"

# 与导出 worker 和 TorchScript wrapper 完全同序；改变任一位置都改变动作函数的数学输入。
TORCHSCRIPT_INPUT_ABI: Final[tuple[str, ...]] = (
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

# 每个输入去掉 batch 轴后的 actor ABI shape；B 只由当前 vectorized runtime 决定。
TORCHSCRIPT_INPUT_SHAPES: Final[tuple[tuple[Any, ...], ...]] = (
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

# TorchScript sidecar 中的 dtype 字符串是 deployment boundary，不依赖 PyTorch repr。
TORCHSCRIPT_INPUT_DTYPES: Final[tuple[str, ...]] = (
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

# 这组字段与导出 worker 的稳定 ABI 同值，但此处独立复制以避免导入冲突版本。
FAMILY_STUDENT_ACTOR_ABI: Final[dict[str, object]] = {
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

# 环境动作 authority 是每个策略步的固定弧度尺度；evaluator 只返回无量纲 canonical mean。
ACTION_AUTHORITY_RAD_PER_STEP: Final[float] = 1.0 / 24.0
ACTION_RANGE_EPS: Final[float] = 1.0e-6


def _sha256(path: Path) -> str:
    r"""分块计算文件 SHA-256，不将 checkpoint 或 TorchScript 全部装入内存。"""

    digest = hashlib.sha256()  # 每次只读取固定大小 block，适用于数百 MB 权重。
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_identity_digest(student: Mapping[str, Any]) -> str:
    r"""返回 IL method identity；优先使用 checkpoint 声明，否则由真实 ABI/config 稳定派生。"""

    metadata = student.get("metadata")
    if isinstance(metadata, Mapping):
        declared = metadata.get("method_identity_digest")
        if isinstance(declared, str) and declared:
            return declared
    payload = {
        "artifact_type": student["artifact_type"],
        "schema_version": student["schema_version"],
        "variant": student["variant"],
        "actor_config": student["actor_config"],
        "actor_abi": student["actor_abi"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    r"""把 identity 中的 Path/Tensor/映射/序列递归规约为严格 JSON-safe 容器。"""

    if isinstance(value, Path):  # provenance 只记录路径文字，不嵌入文件内容。
        return str(value)
    if isinstance(value, torch.Tensor):  # identity 只应含小型状态，但仍明确搬到 CPU 标量/列表。
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):  # key 转 str 防止混合类型破坏排序后的 digest。
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    return str(value)  # 外部 identity 对象只作 provenance，禁止进入动作输入。


def _strict_json_mapping(value: Any, name: str) -> dict[str, Any]:
    r"""读取 JSON object 并拒绝 NaN/不可序列化 identity。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{name} JSON root must be an object")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    r"""验证不可为空的 identity 文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_index(value: Any, name: str) -> int:
    r"""读取严格正整数配置，拒绝 bool/浮点截断。"""

    if isinstance(value, (bool, torch.Tensor)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _load_plain_checkpoint(path: Path) -> dict[str, Any]:
    r"""读取 IL checkpoint 的 plain mapping，不构造 Python actor。"""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError(f"cannot load IL student checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("IL student checkpoint root must be a plain mapping")
    return dict(payload)


def _validate_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    r"""验证独立 IL artifact/schema/ABI/config/metadata，并返回不可变读取副本。"""

    required = {
        "artifact_type",
        "schema",
        "schema_version",
        "actor_state_dict",
        "actor_config",
        "variant",
        "representation",
        "actor_abi",
        "metadata",
        "training_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"IL student checkpoint misses required keys: {missing}")
    if payload["artifact_type"] != FAMILY_STUDENT_ARTIFACT_TYPE:
        raise ValueError(f"IL student checkpoint artifact_type mismatch: {payload['artifact_type']!r}")
    if payload["schema"] != SCHEMA_VERSION or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("IL student checkpoint schema must be 1.0.0")

    # actor ABI 每个固定维度/模式都参与比较，避免仅新增 kinematics 后误载旧 direct actor。
    actor_abi = payload["actor_abi"]
    if not isinstance(actor_abi, Mapping) or dict(actor_abi) != FAMILY_STUDENT_ACTOR_ABI:
        raise ValueError("IL student checkpoint actor_abi disagrees with canonical family ABI")

    variant = payload["variant"]
    if not isinstance(variant, str) or variant not in {"n040", "no_z", "fk"} or payload["representation"] != variant:
        raise ValueError("IL student checkpoint variant/representation must be n040, no_z, or fk")
    config = payload["actor_config"]
    if not isinstance(config, Mapping):
        raise ValueError("IL student checkpoint actor_config must be a mapping")
    if config.get("variant") != variant:
        raise ValueError("IL student checkpoint actor_config variant disagrees with top-level variant")

    # state dict 只做 plain finite/dtype 检查；具体 parameter keys 由导出 worker 和 TS parity 封闭。
    state = payload["actor_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("IL student checkpoint actor_state_dict must be a mapping")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("IL student checkpoint state must map string names to tensors")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"IL student checkpoint state tensor {name!r} is non-finite")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"IL student checkpoint state tensor {name!r} must be float32")

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("IL student checkpoint metadata must be a mapping")
    n040_sha = _nonempty_string(metadata.get("n040_sha256"), "IL metadata.n040_sha256")
    dataset_sha = _nonempty_string(metadata.get("dataset_sha256"), "IL metadata.dataset_sha256")
    training_state = payload["training_state"]
    if not isinstance(training_state, Mapping):
        raise ValueError("IL student checkpoint training_state must be a mapping")
    count_state: dict[str, int] = {}
    for key in ("epoch", "update", "processed_samples"):
        value = training_state.get(key, 0)  # 空 training_state 只表示未训练 dummy，身份中明确为零计数。
        if isinstance(value, bool):
            raise ValueError(f"IL training_state.{key} must be a nonnegative integer")
        try:
            count = operator.index(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"IL training_state.{key} must be a nonnegative integer") from error
        if count < 0:
            raise ValueError(f"IL training_state.{key} must be a nonnegative integer")
        count_state[key] = int(count)

    return {
        "artifact_type": FAMILY_STUDENT_ARTIFACT_TYPE,
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "variant": str(variant),
        "representation": str(variant),
        "actor_abi": dict(actor_abi),
        "actor_config": _json_safe(config),
        "metadata": _json_safe(metadata),
        "n040_sha256": n040_sha,
        "dataset_sha256": dataset_sha,
        "training_state": count_state,
    }


def _validate_sidecar(
    sidecar_path: Path,
    sidecar: Mapping[str, Any],
    *,
    student_sha256: str,
    student: Mapping[str, Any],
    torchscript_sha256: str,
) -> dict[str, Any]:
    r"""验证 TorchScript sidecar 的文件、IL、12-input ABI、precision 与 loaded parity。"""

    required = {
        "artifact_type",
        "schema",
        "schema_version",
        "torchscript_sha256",
        "checkpoint_sha256",
        "n040_sha256",
        "dataset_sha256",
        "variant",
        "actor_config",
        "input_abi",
        "precision",
    }
    missing = sorted(required.difference(sidecar))
    if missing:
        raise ValueError(f"TorchScript sidecar {sidecar_path} misses required keys: {missing}")
    if sidecar["artifact_type"] != TORCHSCRIPT_ARTIFACT_TYPE:
        raise ValueError("TorchScript sidecar artifact_type mismatch")
    if sidecar["schema"] != SCHEMA_VERSION or sidecar["schema_version"] != SCHEMA_VERSION:
        raise ValueError("TorchScript sidecar schema must be 1.0.0")
    if sidecar["torchscript_sha256"] != torchscript_sha256:
        raise ValueError("TorchScript sidecar torchscript_sha256 disagrees with loaded file")
    if sidecar["checkpoint_sha256"] != student_sha256:
        raise ValueError("TorchScript sidecar checkpoint_sha256 disagrees with IL checkpoint")
    if sidecar["n040_sha256"] != student["n040_sha256"]:
        raise ValueError("TorchScript sidecar n040_sha256 disagrees with IL metadata")
    if sidecar["dataset_sha256"] != student["dataset_sha256"]:
        raise ValueError("TorchScript sidecar dataset_sha256 disagrees with IL metadata")
    if sidecar["variant"] != student["variant"]:
        raise ValueError("TorchScript sidecar variant disagrees with IL checkpoint")

    side_config = sidecar["actor_config"]
    if not isinstance(side_config, Mapping):
        raise ValueError("TorchScript sidecar actor_config must be a mapping")
    student_config = student["actor_config"]
    if not isinstance(student_config, Mapping):
        raise ValueError("IL actor_config must be a mapping")
    for key, value in student_config.items():
        if side_config.get(key) != value:
            raise ValueError(f"TorchScript sidecar actor_config.{key} disagrees with IL checkpoint")
    for config_key in ("variant", "representation"):
        if config_key in side_config and side_config[config_key] != student["variant"]:
            raise ValueError(f"TorchScript sidecar actor_config.{config_key} disagrees with IL checkpoint")

    input_abi = sidecar["input_abi"]
    if not isinstance(input_abi, Sequence) or isinstance(input_abi, (str, bytes)) or tuple(input_abi) != TORCHSCRIPT_INPUT_ABI:
        raise ValueError("TorchScript sidecar input_abi disagrees with canonical 12-input ABI")
    if "input_shapes" in sidecar:
        input_shapes = sidecar["input_shapes"]
        if (
            not isinstance(input_shapes, Sequence)
            or isinstance(input_shapes, (str, bytes))
            or tuple(tuple(item) for item in input_shapes) != TORCHSCRIPT_INPUT_SHAPES
        ):
            raise ValueError("TorchScript sidecar input_shapes disagrees with canonical ABI")
    if "input_dtypes" in sidecar:
        input_dtypes = sidecar["input_dtypes"]
        if (
            not isinstance(input_dtypes, Sequence)
            or isinstance(input_dtypes, (str, bytes))
            or tuple(input_dtypes) != TORCHSCRIPT_INPUT_DTYPES
        ):
            raise ValueError("TorchScript sidecar input_dtypes disagrees with canonical ABI")
    if "output_shape" in sidecar:
        output_shape = sidecar["output_shape"]
        if not isinstance(output_shape, Sequence) or isinstance(output_shape, (str, bytes)) or tuple(output_shape) != ("B", 16):
            raise ValueError("TorchScript sidecar output_shape must be ['B',16]")

    precision = sidecar["precision"]
    if not isinstance(precision, Mapping):
        raise ValueError("TorchScript sidecar precision must be a mapping")
    if precision.get("dtype") != "float32" or precision.get("tf32") is not False:
        raise ValueError("TorchScript sidecar precision must be float32 with tf32=False")
    if "amp" in precision and precision["amp"] is not False:
        raise ValueError("TorchScript sidecar precision amp must be false")

    # 正式 sidecar 可用一个 passed 哨兵，也可直接保存 exporter 已计算的 max/tolerance 证据。
    if "loaded_file_parity" in sidecar:
        parity = sidecar["loaded_file_parity"]
        if isinstance(parity, Mapping):
            if parity.get("passed") is not True:
                raise ValueError("TorchScript sidecar loaded_file_parity is not passed")
        elif parity is not True:
            raise ValueError("TorchScript sidecar loaded_file_parity must be true")
    else:
        parity_keys = ("parity_max_abs", "parity_tolerance", "ghost_max_abs", "range_excess_max_abs")
        if any(key not in sidecar for key in parity_keys):
            raise ValueError("TorchScript sidecar lacks loaded-file parity evidence")
        try:
            parity_max = float(sidecar["parity_max_abs"])
            parity_tolerance = float(sidecar["parity_tolerance"])
            ghost_max = float(sidecar["ghost_max_abs"])
            range_excess = float(sidecar["range_excess_max_abs"])
        except (TypeError, ValueError) as error:
            raise ValueError("TorchScript sidecar parity evidence must be numeric") from error
        if not all(torch.isfinite(torch.tensor(value)).item() for value in (parity_max, parity_tolerance, ghost_max, range_excess)):
            raise ValueError("TorchScript sidecar parity evidence must be finite")
        if parity_tolerance <= 0.0 or max(parity_max, ghost_max, range_excess) > parity_tolerance:
            raise ValueError("TorchScript sidecar loaded-file parity evidence did not pass")
    if student["variant"] in {"no_z", "fk"} and "no_z_token_invariance_max_abs" in sidecar:
        try:
            invariance = float(sidecar["no_z_token_invariance_max_abs"])
            tolerance = float(sidecar.get("parity_tolerance", 1.0e-5))
        except (TypeError, ValueError) as error:
            raise ValueError("TorchScript sidecar No-Z invariance evidence must be numeric") from error
        if not all(torch.isfinite(torch.tensor(value)).item() for value in (invariance, tolerance)):
            raise ValueError("TorchScript sidecar No-Z invariance evidence must be finite")
        if tolerance <= 0.0 or invariance > tolerance:
            raise ValueError("TorchScript sidecar No-Z token invariance evidence did not pass")

    return _json_safe(sidecar)


def _loaded_file_parity(sidecar: Mapping[str, Any]) -> Any:
    r"""返回 sidecar 中统一的 loaded-file parity 证据形状供 identity_updates 使用。"""

    if "loaded_file_parity" in sidecar:
        return _json_safe(sidecar["loaded_file_parity"])
    return {
        "passed": True,
        "parity_max_abs": float(sidecar["parity_max_abs"]),
        "parity_tolerance": float(sidecar["parity_tolerance"]),
        "ghost_max_abs": float(sidecar["ghost_max_abs"]),
        "range_excess_max_abs": float(sidecar["range_excess_max_abs"]),
    }


def _reference_n040(identity: Mapping[str, Any], name: str) -> str:
    r"""读取现有 runtime identity 的 retained-artifact SHA，不读取 N040 权重内容。"""

    provider = identity.get("geometry_provider")
    if not isinstance(provider, Mapping):
        raise ValueError(f"{name} is missing geometry_provider identity")
    retained = provider.get("retained_artifact")
    if not isinstance(retained, Mapping):
        raise ValueError(f"{name} is missing geometry_provider.retained_artifact identity")
    return _nonempty_string(retained.get("sha256"), f"{name}.geometry_provider.retained_artifact.sha256")


def _validate_reference_identity(identity: Mapping[str, Any], name: str) -> None:
    r"""验证 teacher/runtime 的 DirectToken、TIP-only、History30、1/24 与 phase-free contract。"""

    policy = identity.get("policy")
    training = identity.get("training")
    if not isinstance(policy, Mapping) or not isinstance(training, Mapping):
        raise ValueError(f"{name} must contain policy and training mappings")
    if policy.get("arm") != "direct_token":
        raise ValueError(f"{name}.policy.arm must be direct_token")
    if policy.get("actor_contact") != "tip-only-binary":
        raise ValueError(f"{name}.policy.actor_contact must be tip-only-binary")
    authority = policy.get("action_authority_rad_per_policy_step")
    if not isinstance(authority, (int, float)) or not torch.isfinite(torch.tensor(float(authority))):
        raise ValueError(f"{name}.policy action authority must be finite")
    if abs(float(authority) - ACTION_AUTHORITY_RAD_PER_STEP) > 1.0e-12:
        raise ValueError(f"{name}.policy action authority must remain 1/24 rad per policy step")
    if training.get("history_encoder") != "tcn":
        raise ValueError(f"{name}.training.history_encoder must be tcn")
    if training.get("phase_period_steps") is not None:
        raise ValueError(f"{name}.training.phase_period_steps must be null")
    if policy.get("phase_clock") not in (None, False) or training.get("phase_clock_enabled") is True:
        raise ValueError(f"{name} must not carry a phase clock")
    if training.get("sigma_mode", "global") != "global":
        raise ValueError(f"{name}.training.sigma_mode must be global")
    transport = identity.get("transport_abi")
    if isinstance(transport, Mapping):
        float_shapes = transport.get("float_shapes")
        if isinstance(float_shapes, Mapping) and "phase_clock" in float_shapes:
            raise ValueError(f"{name}.transport_abi must not contain phase_clock")
        history_shape = float_shapes.get("actor_jnt_history") if isinstance(float_shapes, Mapping) else None
        if history_shape is not None and tuple(history_shape) != (30, 16, 5):
            raise ValueError(f"{name}.transport_abi actor_jnt_history must be [30,16,5]")


def _validate_observation_tensor(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""验证 runtime observation 单 tensor 的 shape/device/dtype/finite 边界。"""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != expected {expected_shape}")
    if value.device != device:
        raise ValueError(f"{name} device {value.device} disagrees with actor device {device}")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype {value.dtype} != expected {dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _validate_graph_tensor(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    r"""接受 runtime 为节省存储交付的有符号 graph bucket，并在 TS 边界规约为 int64。"""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} dtype {value.dtype} must be an integer graph bucket")
    validated = _validate_observation_tensor(
        value,
        name,
        expected_shape,
        device=device,
        dtype=value.dtype,
    )
    return validated.to(dtype=torch.long)  # sidecar ABI 固定 long；int16 runtime storage 无损提升。


def _exact_zero_tensor(value: torch.Tensor, name: str) -> None:
    r"""验证无效 joint/owner 的 ghost 值精确为 0。"""

    if value.numel() and bool(torch.count_nonzero(value).item()):
        raise ValueError(f"{name} ghost/padding values must be exactly zero")


class FrozenFamilyStudent:
    r"""跨现有 teacher runtime 部署同一份冻结 TorchScript student 的最窄 wrapper。

    Constructor 只读取独立 IL checkpoint、TorchScript 和 sidecar；它不创建或 import Python
    actor。``start`` 再绑定一个具体 teacher reference/runtime、source geometry semantics 与
    reset 后 observation，构造不随时间变化的 ``[A,16,15]`` kinematics bank。``act`` 只
    接收 actor-prefix observation，返回 ``float32 [B,16]`` canonical mean；``finish`` 用
    参数/buffer snapshot 与文件 SHA 证明本轮没有偷偷替换 student。
    """

    def __init__(
        self,
        student_checkpoint: str | Path,
        torchscript_path: str | Path,
        sidecar_path: str | Path | None = None,
    ) -> None:
        r"""加载并严格绑定 IL plain dict、TorchScript 文件和 sidecar identity。"""

        self.student_checkpoint = Path(student_checkpoint).expanduser().resolve()
        self.torchscript_path = Path(torchscript_path).expanduser().resolve()
        self.sidecar_path = (
            Path(sidecar_path).expanduser().resolve()
            if sidecar_path is not None
            else Path(f"{self.torchscript_path}.json")
        )
        for path in (self.student_checkpoint, self.torchscript_path, self.sidecar_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        # SHA 在加载前计算并由 sidecar 绑定；后续 start/finish 再计算闭合实际文件未漂移。
        self._student_sha256 = _sha256(self.student_checkpoint)
        self._torchscript_sha256 = _sha256(self.torchscript_path)
        payload = _load_plain_checkpoint(self.student_checkpoint)
        self._student = _validate_checkpoint(self.student_checkpoint, payload)
        try:
            sidecar_raw = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot parse TorchScript sidecar {self.sidecar_path}: {error}") from error
        sidecar = _strict_json_mapping(sidecar_raw, "TorchScript sidecar")
        self._sidecar = _validate_sidecar(
            self.sidecar_path,
            sidecar,
            student_sha256=self._student_sha256,
            student=self._student,
            torchscript_sha256=self._torchscript_sha256,
        )

        # 先在 CPU 载入实际 .ts，后续 start 才迁移到 runtime observation device。
        try:
            self._script = torch.jit.load(str(self.torchscript_path), map_location="cpu").eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"cannot load TorchScript student {self.torchscript_path}: {error}") from error

        self._started = False
        self._finished = False
        self._step = 0
        self._executed_steps = 0
        self._steps = 0
        self._replicas = 0
        self._asset_count = 0
        self._device = torch.device("cpu")
        self._kinematics_features: torch.Tensor | None = None
        self._module_snapshot: dict[str, torch.Tensor] = {}
        self._reference_checkpoint: Path | None = None
        self._reference_checkpoint_sha256: str | None = None
        self._reference_identity: dict[str, Any] = {}
        self._runtime_identity: dict[str, Any] = {}
        self._cohort_path: Path | None = None
        self._cohort_members: list[Any] = []
        self._helper_source_sha256 = _sha256(Path(__file__).resolve())
        self._metadata: dict[str, Any] = {
            "student_checkpoint_path": str(self.student_checkpoint),
            "student_checkpoint_sha256": self._student_sha256,
            "method_identity_digest": _method_identity_digest(self._student),
            "torchscript_sha256": self._torchscript_sha256,
            "sidecar": _json_safe(self._sidecar),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        r"""返回当前 evaluator metadata 的 JSON-safe 副本，不暴露可变内部 identity。"""

        return cast(dict[str, Any], json.loads(json.dumps(_json_safe(self._metadata), allow_nan=False)))

    @property
    def executed_steps(self) -> int:
        """返回已经成功完成的物理动作数，供异常路径记录实际交互成本。"""
        return self._executed_steps

    def _snapshot_script(self) -> dict[str, torch.Tensor]:
        r"""复制 TorchScript 的 parameters/buffers 到 CPU，供 finish 做逐值冻结检查。"""

        snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in list(self._script.named_parameters()) + list(self._script.named_buffers()):
            snapshot[name] = tensor.detach().cpu().clone()
        return snapshot

    def _build_kinematics(
        self,
        binding: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        r"""用 source geometry semantics + canonical routing 生成共享 [A,16,15] FP32 bank。"""

        source_assets = getattr(binding, "source_assets", None)
        canonical_artifacts = getattr(binding, "canonical_artifacts", None)
        if not isinstance(source_assets, Sequence) or not isinstance(canonical_artifacts, Sequence):
            raise ValueError("binding must expose source_assets and canonical_artifacts sequences")
        if not source_assets or len(source_assets) != len(canonical_artifacts):
            raise ValueError("binding source/canonical asset axes must be nonempty and aligned")
        slot_by_name = {
            name: index for index, name in enumerate(CANONICAL_HAND_SCHEMA_V1.joint_names)
        }  # canonical schema 是唯一 joint slot 顺序，不按 finger/name 猜位置。
        semantics: list[Any] = []
        mappings: list[dict[str, int]] = []
        for asset_index, (source, artifact) in enumerate(zip(source_assets, canonical_artifacts, strict=True)):
            semantic = getattr(source, "geometry_semantics", None)
            routing = getattr(artifact, "routing", None)
            pairs = getattr(routing, "source_to_canonical", None)
            if semantic is None or not isinstance(pairs, Sequence):
                raise ValueError(f"binding asset {asset_index} lacks geometry_semantics/source_to_canonical")
            mapping: dict[str, int] = {}
            for pair in pairs:
                if not isinstance(pair, Sequence) or len(pair) != 2:
                    raise ValueError(f"binding asset {asset_index} has malformed source_to_canonical pair")
                source_name, canonical_name = pair
                if not isinstance(source_name, str) or not isinstance(canonical_name, str):
                    raise ValueError("source_to_canonical names must be strings")
                if canonical_name not in slot_by_name:
                    raise ValueError(f"canonical joint name {canonical_name!r} is absent from schema v1")
                if source_name in mapping:
                    raise ValueError(f"binding asset {asset_index} repeats source joint {source_name!r}")
                canonical_slot = slot_by_name[canonical_name]
                if canonical_slot in mapping.values():
                    raise ValueError(f"binding asset {asset_index} repeats canonical joint slot {canonical_name!r}")
                mapping[source_name] = canonical_slot
            semantics.append(semantic)
            mappings.append(mapping)

        # float64 build 保留 source SE(3) 复合精度；只在交给 TS 前转成声明的 FP32 features。
        try:
            bank = build_joint_kinematics_bank(
                semantics,
                mappings,
                joint_count=16,
                dtype=torch.float64,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"cannot build joint kinematics bank from binding: {error}") from error
        features = getattr(bank, "features", None)
        if not isinstance(features, torch.Tensor):
            raise ValueError("joint kinematics bank must expose tensor features")
        expected = (len(source_assets), 16, 15)
        if tuple(features.shape) != expected:
            raise ValueError(f"joint kinematics bank shape {tuple(features.shape)} != expected {expected}")
        if features.dtype != torch.float64 or not bool(torch.isfinite(features).all().item()):
            raise ValueError("joint kinematics bank must be finite float64 before deployment cast")
        return features.float().to(device=device)  # 先 FP64→FP32，再搬到 runtime actor device。

    def start(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        binding: Any,
        observation: Mapping[str, torch.Tensor],
        cohort_path: str | Path,
        cohort_members: Sequence[Any],
        steps: int,
        replicas: int,
        actor: Any = None,
    ) -> None:
        r"""绑定一个 teacher reference 与当前 runtime，准备 TS student 的第一次 act。

        ``checkpoint_path``/``checkpoint_identity`` 是原 teacher reference，只定义相同的物理
        task、action authority 与来源 provenance；真正执行的 student weights 永远来自
        constructor 的独立 IL checkpoint。``actor`` 参数仅保留给调用方接口对齐，evaluator
        不读取其参数、不读取 Critic，也不把它传入 TorchScript。
        """

        del actor  # reference teacher actor 由原 evaluator 自己管理，不能污染 student 输入。
        if self._started:
            raise RuntimeError("FrozenFamilyStudent.start may only occur once")
        if not isinstance(checkpoint_identity, Mapping) or not isinstance(runtime_identity, Mapping):
            raise ValueError("checkpoint_identity and runtime_identity must be mappings")
        teacher_checkpoint = Path(checkpoint_path).expanduser().resolve()
        cohort = Path(cohort_path).expanduser().resolve()
        if not teacher_checkpoint.is_file():
            raise FileNotFoundError(teacher_checkpoint)
        if not cohort.is_file():
            raise FileNotFoundError(cohort)
        step_count = _positive_index(steps, "steps")
        replica_count = _positive_index(replicas, "replicas")
        _validate_reference_identity(checkpoint_identity, "checkpoint_identity")
        _validate_reference_identity(runtime_identity, "runtime_identity")

        # teacher 与 runtime 各自保留现有 MDP provenance；student N040 需与 runtime retained sha 相同。
        runtime_n040 = _reference_n040(runtime_identity, "runtime_identity")
        if runtime_n040 != self._student["n040_sha256"]:
            raise ValueError("runtime N040 sha256 disagrees with IL metadata.n040_sha256")
        if "geometry_provider" in checkpoint_identity:
            teacher_n040 = _reference_n040(checkpoint_identity, "checkpoint_identity")
            if teacher_n040 != self._student["n040_sha256"]:
                raise ValueError("teacher reference N040 sha256 disagrees with IL metadata.n040_sha256")

        source_assets = getattr(binding, "source_assets", None)
        canonical_artifacts = getattr(binding, "canonical_artifacts", None)
        if not isinstance(source_assets, Sequence) or not isinstance(canonical_artifacts, Sequence):
            raise ValueError("binding must expose source_assets and canonical_artifacts sequences")
        asset_count = len(source_assets)
        if asset_count < 1 or len(canonical_artifacts) != asset_count:
            raise ValueError("binding source/canonical assets must be nonempty and aligned")
        if not isinstance(cohort_members, Sequence) or len(cohort_members) != asset_count:
            raise ValueError("cohort_members must align exactly with binding source_assets")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a named actor mapping")
        current = observation.get("actor_jnt_current")
        if not isinstance(current, torch.Tensor) or current.ndim < 1:
            raise ValueError("observation.actor_jnt_current must define a nonempty batch")
        batch = int(current.shape[0])
        if batch != asset_count * replica_count:
            raise ValueError(
                f"observation env batch {batch} != asset_count {asset_count} * replicas {replica_count}"
            )
        device = current.device
        kinematics = self._build_kinematics(binding, device=device)

        # TS `.to(device).eval()` 是唯一 deployment model 操作；不构造或导入当前 Python actor。
        self._script = self._script.to(device=device).eval()
        self._device = device
        self._kinematics_features = kinematics
        self._asset_count = asset_count
        self._steps = step_count
        self._replicas = replica_count
        self._reference_checkpoint = teacher_checkpoint
        self._reference_checkpoint_sha256 = _sha256(teacher_checkpoint)
        declared_teacher_sha = checkpoint_identity.get("checkpoint_sha256")
        if declared_teacher_sha is not None and declared_teacher_sha != self._reference_checkpoint_sha256:
            raise ValueError("checkpoint_identity.checkpoint_sha256 disagrees with checkpoint_path")
        self._reference_identity = cast(dict[str, Any], _json_safe(checkpoint_identity))
        self._runtime_identity = cast(dict[str, Any], _json_safe(runtime_identity))
        self._cohort_path = cohort
        self._cohort_members = cast(list[Any], _json_safe(list(cohort_members)))
        self._module_snapshot = self._snapshot_script()

        # start 同时对 reset 后 H0 observation 做完整 12-input shape/ghost 预检；不执行动作。
        self._prepare_inputs(observation)
        self._metadata.update(
            {
                "started": True,
                "steps": step_count,
                "replicas": replica_count,
                "asset_count": asset_count,
                "device": str(device),
                "student_variant": self._student["variant"],
                "student_n040_sha256": self._student["n040_sha256"],
                "student_dataset_sha256": self._student["dataset_sha256"],
                "student_method_identity_digest": _method_identity_digest(self._student),
                "student_checkpoint_path": str(self.student_checkpoint),
                "variant": self._student["variant"],
                "n040_sha256": self._student["n040_sha256"],
                "dataset_sha256": self._student["dataset_sha256"],
                "actor_config": _json_safe(self._student["actor_config"]),
                "training_state": dict(self._student["training_state"]),
                "input_abi": list(TORCHSCRIPT_INPUT_ABI),
                "precision": _json_safe(self._sidecar["precision"]),
            }
        )
        self._started = True

    def _prepare_inputs(self, observation: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        r"""从 runtime named observation 组装严格 12-input tuple，并执行 actor ghost contract。"""

        if self._kinematics_features is None:
            raise RuntimeError("FrozenFamilyStudent kinematics bank is not initialized; call start first")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        current = observation.get("actor_jnt_current")
        if not isinstance(current, torch.Tensor) or current.ndim != 3:
            raise ValueError("actor_jnt_current must have shape [B,16,5]")
        batch = int(current.shape[0])
        if self._asset_count and batch != self._asset_count * self._replicas:
            raise ValueError(
                f"observation batch {batch} != started asset_count {self._asset_count} * replicas {self._replicas}"
            )
        device = current.device
        if self._kinematics_features is not None and device != self._device:
            raise ValueError(f"observation device {device} != started device {self._device}")
        current = _validate_observation_tensor(current, "actor_jnt_current", (batch, 16, 5), device=device, dtype=torch.float32)
        history = _validate_observation_tensor(
            observation.get("actor_jnt_history"),
            "actor_jnt_history",
            (batch, 30, 16, 5),
            device=device,
            dtype=torch.float32,
        )
        limits = _validate_observation_tensor(
            observation.get("actor_jnt_limits"),
            "actor_jnt_limits",
            (batch, 16, 2),
            device=device,
            dtype=torch.float32,
        )
        owner_contact = _validate_observation_tensor(
            observation.get("actor_owner_contact"),
            "actor_owner_contact",
            (batch, 21, 1),
            device=device,
            dtype=torch.float32,
        )
        joint_valid = _validate_observation_tensor(
            observation.get("jnt_valid"), "jnt_valid", (batch, 16), device=device, dtype=torch.bool
        )
        tip_valid = _validate_observation_tensor(
            observation.get("tip_valid"), "tip_valid", (batch, 4), device=device, dtype=torch.bool
        )
        owner_valid = _validate_observation_tensor(
            observation.get("owner_valid"), "owner_valid", (batch, 21), device=device, dtype=torch.bool
        )
        geometry_tokens = _validate_observation_tensor(
            observation.get("geometry_tokens"),
            "geometry_tokens",
            (batch, 21, 128),
            device=device,
            dtype=torch.float32,
        )
        shortest_path = _validate_graph_tensor(
            observation.get("shortest_path"),
            "shortest_path",
            (batch, 21, 21),
            device=device,
        )
        parent_direction = _validate_graph_tensor(
            observation.get("parent_direction"),
            "parent_direction",
            (batch, 21, 21),
            device=device,
        )
        child_direction = _validate_graph_tensor(
            observation.get("child_direction"),
            "child_direction",
            (batch, 21, 21),
            device=device,
        )

        # Binary contact/graph ranges属于 runtime ABI；不做 clamp，坏物理事实必须暴露。
        if not bool(torch.isfinite(owner_contact).all().item()) or not bool(
            ((owner_contact == 0.0) | (owner_contact == 1.0)).all().item()
        ):
            raise ValueError("actor_owner_contact must be finite binary FP32")
        if not bool(
            ((current[..., 3:] == 0.0) | (current[..., 3:] == 1.0)).all().item()
        ) or not bool(((history[..., 3:] == 0.0) | (history[..., 3:] == 1.0)).all().item()):
            raise ValueError("actor current/history contact channels must be binary")
        if any(bool((graph < 0).any().item()) for graph in (shortest_path, parent_direction, child_direction)):
            raise ValueError("owner graph relation buckets must be nonnegative int64")
        if bool((limits[..., 0] > limits[..., 1]).any().item()):
            raise ValueError("actor_jnt_limits lower bound must not exceed upper bound")

        # owner 轴固定 PALM + JOINT16 + TIP4；不容许 mask 之间发生静默错位。
        expected_owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=device), joint_valid, tip_valid), dim=-1)
        if not torch.equal(owner_valid, expected_owner):
            raise ValueError("owner_valid must equal PALM/JOINT/TIP validity concatenation")
        invalid_joint = ~joint_valid
        _exact_zero_tensor(current[invalid_joint], "actor_jnt_current")
        _exact_zero_tensor(history[invalid_joint[:, None, :, None].expand_as(history)], "actor_jnt_history")
        _exact_zero_tensor(limits[invalid_joint], "actor_jnt_limits")
        _exact_zero_tensor(owner_contact[~owner_valid], "actor_owner_contact")
        _exact_zero_tensor(geometry_tokens[~owner_valid], "geometry_tokens")
        if not bool(torch.allclose(history[:, -1], current, rtol=0.0, atol=1.0e-6)):
            raise ValueError("actor_jnt_history latest frame must equal actor_jnt_current")

        # env e→asset e%A 仅在 evaluator 内做静态 routing；asset index 不进入此 tuple。
        env_asset = torch.arange(batch, device=device, dtype=torch.long) % self._asset_count
        gathered_kinematics = self._kinematics_features.index_select(0, env_asset)
        _exact_zero_tensor(gathered_kinematics[invalid_joint], "joint_kinematics")
        if not bool(torch.isfinite(gathered_kinematics).all().item()):
            raise ValueError("joint_kinematics input must be finite FP32")
        return (
            current,
            history,
            limits,
            owner_contact,
            joint_valid,
            tip_valid,
            owner_valid,
            geometry_tokens,
            shortest_path,
            parent_direction,
            child_direction,
            gathered_kinematics,
        )

    def act(self, step: int, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        r"""对一个 runtime state 执行 TorchScript deterministic mean，返回 FP32 ``[B,16]``。

        `step` 只用于评价时钟审计，不拼入 TS 输入。TF32 flag 仅在这次 no-grad forward 的
        临界区临时关闭，`finally` 恢复调用方原值；因此不会改变旧 N040/teacher reference
        的全局 precision mode。
        """

        if not self._started or self._finished:
            raise RuntimeError("FrozenFamilyStudent.act requires a started, unfinished evaluator")
        if isinstance(step, (bool, torch.Tensor)):
            raise ValueError("step must be an integer")
        try:
            index = operator.index(step)
        except (TypeError, ValueError) as error:
            raise ValueError("step must be an integer") from error
        if index != self._step or index < 0 or index >= self._steps:
            raise ValueError(f"act step must be contiguous in [0,{self._steps}), expected {self._step}, got {index}")
        inputs = self._prepare_inputs(observation)

        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad():
                output = self._script(*inputs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("TorchScript student output must be a tensor")
        batch = int(inputs[0].shape[0])
        if output.shape != (batch, 16) or output.dtype != torch.float32:
            raise ValueError(f"TorchScript student output shape/dtype {tuple(output.shape)}/{output.dtype} != {(batch,16)}/float32")
        if not bool(torch.isfinite(output).all().item()):
            raise ValueError("TorchScript student output contains non-finite values")
        if bool((output.abs() > 1.0 + ACTION_RANGE_EPS).any().item()):
            raise ValueError("TorchScript student output exceeds canonical action range [-1,1]")
        joint_valid = inputs[4]
        _exact_zero_tensor(output[~joint_valid], "TorchScript student action")
        self._step += 1
        return output.detach()

    def after_step(self) -> None:
        r"""在主 evaluator 的物理 ``env.step`` 成功后登记一条真实完成动作。"""

        if not self._started or self._finished:
            raise RuntimeError("FrozenFamilyStudent.after_step requires a started, unfinished evaluator")
        if self._executed_steps >= self._step:
            raise RuntimeError("after_step has no pending action; call act before the physical step")
        self._executed_steps += 1  # 该计数只代表物理动作成功返回，不把 act 前向冒充已执行。
        self._metadata["executed_steps"] = self._executed_steps

    def finish(self) -> None:
        r"""验证 student TS 参数/buffer 与文件 SHA 未漂移，并结束 evaluator 生命周期。"""

        if not self._started:
            raise RuntimeError("FrozenFamilyStudent.finish requires start")
        if self._finished:
            raise RuntimeError("FrozenFamilyStudent.finish may only occur once")
        if self._step > self._steps:
            raise RuntimeError("FrozenFamilyStudent executed more steps than declared")
        if self._executed_steps != self._step:
            raise RuntimeError(
                f"finish requires after_step for every act: acted={self._step}, executed={self._executed_steps}"
            )
        current = self._snapshot_script()
        if set(current) != set(self._module_snapshot):
            raise RuntimeError("TorchScript parameter/buffer names changed during evaluation")
        for name, before in self._module_snapshot.items():
            if current[name].dtype != before.dtype or current[name].shape != before.shape or not torch.equal(current[name], before):
                raise RuntimeError(f"TorchScript parameter/buffer {name!r} changed during evaluation")
        student_sha = _sha256(self.student_checkpoint)
        script_sha = _sha256(self.torchscript_path)
        if student_sha != self._student_sha256:
            raise RuntimeError("IL student checkpoint SHA changed during evaluation")
        if script_sha != self._torchscript_sha256:
            raise RuntimeError("TorchScript file SHA changed during evaluation")
        if self._reference_checkpoint is not None and self._reference_checkpoint_sha256 is not None:
            if _sha256(self._reference_checkpoint) != self._reference_checkpoint_sha256:
                raise RuntimeError("teacher reference checkpoint SHA changed during evaluation")
        self._metadata["finished"] = True
        self._metadata["executed_steps"] = self._executed_steps
        self._finished = True

    def identity_updates(self) -> dict[str, Any]:
        r"""返回可合并入 evaluation_identity 的 JSON-safe provenance 更新。

        返回值明确区分实际执行的 student checkpoint/method 与 reference teacher 的 MDP 来源。
        `training_state` 只取 IL 的 epoch/update/processed_samples 计数，不伪装 PPO epoch/frame；
        12-input ABI、N040、TS SHA、runtime precision 与 helper source SHA 都可独立复核。
        """

        if not self._started:
            raise RuntimeError("FrozenFamilyStudent.identity_updates requires start")
        training = self._runtime_identity.get("training", {})
        runtime_precision = self._runtime_identity.get("precision")
        if runtime_precision is None:
            runtime_precision = {
                "dtype": training.get("dtype", "float32"),
                "tf32": bool(training.get("allow_tf32", False)),
                **({"device": training["device"]} if "device" in training else {}),
            }
        method_digest = _method_identity_digest(self._student)
        student = {
            "artifact_type": self._student["artifact_type"],
            "schema_version": self._student["schema_version"],
            "checkpoint_path": str(self.student_checkpoint),
            "checkpoint_sha256": self._student_sha256,
            "dataset_sha256": self._student["dataset_sha256"],
            "n040_sha256": self._student["n040_sha256"],
            "variant": self._student["variant"],
            "representation": self._student["representation"],
            "actor_config": _json_safe(self._student["actor_config"]),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "training_state": dict(self._student["training_state"]),
            "method_identity_digest": method_digest,
            "metadata": _json_safe(self._student["metadata"]),
        }
        torchscript = {
            "artifact_type": TORCHSCRIPT_ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "path": str(self.torchscript_path),
            "sha256": self._torchscript_sha256,
            "sidecar_path": str(self.sidecar_path),
            "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            "input_shapes": [list(shape) for shape in TORCHSCRIPT_INPUT_SHAPES],
            "input_dtypes": list(TORCHSCRIPT_INPUT_DTYPES),
            "output_shape": ["B", 16],
            "precision": _json_safe(self._sidecar["precision"]),
            "loaded_file_parity": _loaded_file_parity(self._sidecar),
            "variant": self._student["variant"],
        }
        reference = {
            "checkpoint_path": str(self._reference_checkpoint) if self._reference_checkpoint else None,
            "checkpoint_sha256": self._reference_checkpoint_sha256,
            "identity": _json_safe(self._reference_identity),
            "checkpoint_identity": _json_safe(self._reference_identity),
            "role": "teacher_reference_mdp_source_only",
        }
        # 当前评价成员属于 student rollout；与参考教师原训练集合分别记录，
        # 避免把未来未见手的测试声明误读成 teacher 的训练暴露。
        evaluated_cohort = {
            "path": str(self._cohort_path) if self._cohort_path else None,
            "sha256": _sha256(self._cohort_path) if self._cohort_path else None,
            "members": _json_safe(self._cohort_members),
            "steps": self._steps,
            "replicas": self._replicas,
            "role": "student_evaluation_population",
        }
        runtime = {
            "identity": _json_safe(self._runtime_identity),
            "identity_digest": self._runtime_identity.get("identity_digest"),
            "n040_sha256": _reference_n040(self._runtime_identity, "runtime_identity"),
            "actor_precision": _json_safe(self._sidecar["precision"]),
            "reference_runtime_precision": _json_safe(runtime_precision),
            "device": str(self._device),
            "action_authority_rad_per_policy_step": ACTION_AUTHORITY_RAD_PER_STEP,
            "phase_clock": False,
            "actor_contact": "tip-only-binary",
        }
        result: dict[str, Any] = {
            # direct fields are deliberate stable update keys for primary's evaluation identity merge。
            "student_checkpoint_sha256": self._student_sha256,
            "student_checkpoint_path": str(self.student_checkpoint),
            "torchscript_sha256": self._torchscript_sha256,
            "dataset_sha256": self._student["dataset_sha256"],
            "n040_sha256": self._student["n040_sha256"],
            "variant": self._student["variant"],
            "actor_config": _json_safe(self._student["actor_config"]),
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "training_state": dict(self._student["training_state"]),
            "executed_steps": self._executed_steps,
            "method_identity_digest": method_digest,
            "input_abi": list(TORCHSCRIPT_INPUT_ABI),
            "precision": _json_safe(self._sidecar["precision"]),
            "runtime_actor_precision": _json_safe(self._sidecar["precision"]),
            "reference_runtime_precision": _json_safe(runtime_precision),
            "student_method_identity_digest": method_digest,
            "document_updates": {
                "student_checkpoint_path": str(self.student_checkpoint),
                "student_checkpoint_sha256": self._student_sha256,
                "method_identity_digest": method_digest,
                "training_state": dict(self._student["training_state"]),
                "executed_steps": self._executed_steps,
            },
            "student": student,
            "torchscript": torchscript,
            "reference_teacher": reference,
            "evaluated_cohort": evaluated_cohort,
            "runtime": runtime,
            "helper_source_sha256": self._helper_source_sha256,
            "method": {
                "artifact_type": self._student["artifact_type"],
                "schema_version": self._student["schema_version"],
                "variant": self._student["variant"],
                "representation": self._student["representation"],
            },
        }
        return cast(dict[str, Any], json.loads(json.dumps(_json_safe(result), allow_nan=False)))


__all__ = [
    "ACTION_AUTHORITY_RAD_PER_STEP",
    "FAMILY_STUDENT_ACTOR_ABI",
    "FAMILY_STUDENT_ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "TORCHSCRIPT_ARTIFACT_TYPE",
    "TORCHSCRIPT_INPUT_ABI",
    "TORCHSCRIPT_INPUT_DTYPES",
    "TORCHSCRIPT_INPUT_SHAPES",
    "FrozenFamilyStudent",
]
