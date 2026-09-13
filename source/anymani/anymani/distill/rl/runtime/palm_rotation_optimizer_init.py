r"""掌旋 PPO 的名称驱动 Adam 初始化合同。

PyTorch ``Optimizer.state_dict`` 只保存整数参数 ID；这些 ID 会随参数组顺序和模型构造顺序变化，因而不能
作为跨 warm-start 分支的参数身份。本模块把可审计的相对参数名作为唯一坐标：source checkpoint 的整数 ID
只用于恢复 ``source_name_groups`` 所声明的 name→state 对应关系，target 端随后以真实 ``Parameter`` 对象
写入 state。这样即使 Actor 参数组或组内参数顺序改变，只要名字、shape、dtype和Adam语义仍一致，矩状态仍能
正确归属；同shape但不同名的参数不会被静默误配。

当前公共合同限定为标准 ``torch.optim.Adam`` 的非AMSGrad状态：每个已训练参数必须有标量 ``step``、与参数
同shape同dtype的有限 ``exp_avg`` 和 ``exp_avg_sq``。target 新增参数只能来自显式 ``allowed_new_names``，并
以 ``step=0``、零一阶矩和零二阶矩初始化；target 原有 ``lr`` 及其它组元数据留在原对象中，source 的学习率
只作可审计输入而不会倒灌到新实验配置。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import Parameter

OPTIMIZER_NAME_LEDGER_SCHEMA_VERSION = "1.0.0"
"""名称ledger的稳定schema版本；schema变化必须显式迁移而不能靠宽松解析。"""

_GROUP_SCHEMA_KEYS = frozenset({"name", "parameters"})
"""精简名称组只保留组名与相对参数名序列，避免把整数ID伪装成持久身份。"""

_STATE_SCHEMA_KEYS = frozenset({"state", "param_groups"})
"""当前Adam state_dict的唯一顶层字段；未知字段可能携带未审计的优化器语义。"""

_STATE_ENTRY_KEYS = frozenset({"step", "exp_avg", "exp_avg_sq"})
"""非AMSGrad参数状态的完整字段集合；出现max_exp_avg_sq即说明合同被改变。"""

_GROUP_META_EXCLUDED_KEYS = frozenset({"params", "lr", "name"})
"""比较组语义时排除参数ID、学习率和名称；其中学习率由target配置拥有。"""

_KNOWN_PARAMETER_PREFIXES = (
    "a2c_network.package.actor.",
    "a2c_network.package.critic.",
    "actor.",
    "critic.",
)
"""从完整package.named_parameters()中剥离到Actor/Critic相对坐标的已知前缀。"""


def _sha256(path: Path) -> str:
    r"""以有界内存流式计算文件的SHA-256字节身份。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)  # 每次最多1 MiB，避免把checkpoint/ledger复制为另一份大bytes
    return digest.hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    r"""验证外部hash是64位小写十六进制字符串，拒绝截断或隐式转换。"""

    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 hex string")
    return value


def _relative_parameter_name(name: str) -> str:
    r"""把完整Actor/Critic路径规约为当前优化器使用的相对参数名。"""

    for prefix in _KNOWN_PARAMETER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _validate_group_name(value: object, *, context: str) -> str | None:
    r"""验证optimizer组名；无名单组由``None``表示，不能用空字符串混淆。"""

    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} group name must be a non-empty string or None")
    return value


def _values_equal(left: object, right: object) -> bool:
    r"""比较Adam组元数据，同时支持tuple/list与零维tensor等state_dict值。"""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and bool(torch.equal(left, right))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return False
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _finite_scalar(value: object, *, context: str) -> float:
    r"""读取有限标量，用于学习率与Adam step的边界验证。"""

    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite numeric scalar, not bool")
    if isinstance(value, torch.Tensor):
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if value.numel() != 1 or value.is_complex() or not (value.is_floating_point() or value.dtype in integer_dtypes):
            raise ValueError(f"{context} must be a finite scalar tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{context} is non-finite")
        scalar = float(value.detach().cpu().item())
    elif isinstance(value, (int, float)):
        scalar = float(value)
    else:
        raise ValueError(f"{context} must be a finite numeric scalar")
    if not math.isfinite(scalar):
        raise ValueError(f"{context} is non-finite")
    return scalar


def _parameter_id(value: object, *, context: str) -> int:
    r"""把state_dict中的参数ID规约为非负Python int，并检测JSON字符串ID。"""

    if isinstance(value, bool):
        raise ValueError(f"{context} parameter id must be a non-negative integer")
    if isinstance(value, int):
        identifier = value
    elif isinstance(value, str) and value.isdecimal():
        identifier = int(value)
    else:
        raise ValueError(f"{context} parameter id must be a non-negative integer")
    if identifier < 0:
        raise ValueError(f"{context} parameter id must be non-negative")
    return identifier


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    r"""JSON object hook：重复字段会改变schema含义，必须在解析阶段拒绝。"""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"optimizer name ledger contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    r"""禁止NaN/Infinity进入名称ledger；这些值不是JSON数值身份。"""

    raise ValueError(f"optimizer name ledger contains non-standard JSON constant {value}")


def _validate_name_groups(
    groups: object,
    *,
    context: str,
) -> tuple[dict[str, Any], ...]:
    r"""验证名称组的字段、顺序、唯一性，并返回规约后的内部表示。"""

    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be a list of parameter groups")
    if not groups:
        raise ValueError(f"{context} must contain at least one parameter group")
    normalized: list[dict[str, Any]] = []
    group_names: list[str | None] = []
    seen_parameters: set[str] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != _GROUP_SCHEMA_KEYS:
            raise ValueError(f"{context}[{group_index}] must contain exactly name and parameters")
        group_name = _validate_group_name(group["name"], context=f"{context}[{group_index}]")
        parameters = group["parameters"]
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)) or not parameters:
            raise ValueError(f"{context}[{group_index}].parameters must be a non-empty list")
        normalized_parameters: list[str] = []
        for parameter_index, parameter_name in enumerate(parameters):
            if not isinstance(parameter_name, str) or not parameter_name:
                raise ValueError(
                    f"{context}[{group_index}].parameters[{parameter_index}] must be a non-empty string"
                )
            relative_name = _relative_parameter_name(parameter_name)
            if not relative_name:
                raise ValueError(f"{context}[{group_index}] contains an empty relative parameter name")
            if relative_name in seen_parameters:
                raise ValueError(f"duplicate optimizer parameter name {relative_name!r} in {context}")
            seen_parameters.add(relative_name)
            normalized_parameters.append(relative_name)
        if group_name in group_names:
            raise ValueError(f"duplicate optimizer group name {group_name!r} in {context}")
        group_names.append(group_name)
        normalized.append({"name": group_name, "parameters": normalized_parameters})
    return tuple(normalized)


def optimizer_parameter_names(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, Parameter],
) -> list[dict[str, Any]]:
    r"""导出optimizer各组的相对参数名，作为未来checkpoint的稳定坐标。

    Args:
        optimizer (torch.optim.Optimizer): 已构造的Actor或Critic optimizer；组内顺序按PyTorch实际配置保留。
        named_parameters (Mapping[str, Parameter]): 当前Actor/Critic的``name -> Parameter``映射；也可传入
            完整package映射，函数会剥离已知的``actor.``/``critic.``前缀。

    Returns:
        list[dict[str, Any]]: 每组一个``{"name": group.get("name"), "parameters": [relative names]}``字典。

    Raises:
        ValueError: 参数未被命名、跨组重复、组名非法或同一Parameter具有歧义别名。
    """

    if not isinstance(named_parameters, Mapping):
        raise TypeError("named_parameters must be a name-to-Parameter mapping")
    name_by_id: dict[int, str] = {}
    for raw_name, parameter in named_parameters.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("named_parameters keys must be non-empty strings")
        if not isinstance(parameter, Parameter):
            raise TypeError(f"named_parameters[{raw_name!r}] is not torch.nn.Parameter")
        parameter_id = id(parameter)
        if parameter_id in name_by_id:
            raise ValueError(f"Parameter object {parameter_id} has duplicate names in named_parameters")
        name_by_id[parameter_id] = _relative_parameter_name(raw_name)

    groups: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    seen_group_names: set[str | None] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = _validate_group_name(group.get("name"), context=f"optimizer.param_groups[{group_index}]")
        if group_name in seen_group_names:
            raise ValueError(f"duplicate optimizer group name {group_name!r}")
        seen_group_names.add(group_name)
        parameters = group.get("params")
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)) or not parameters:
            raise ValueError(f"optimizer.param_groups[{group_index}].params must be non-empty")
        names: list[str] = []
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("optimizer parameter groups must contain torch.nn.Parameter objects")
            parameter_id = id(parameter)
            if parameter_id not in name_by_id:
                raise ValueError(f"optimizer parameter {parameter_id} is absent from named_parameters")
            if parameter_id in seen_ids:
                raise ValueError(f"duplicate optimizer parameter {name_by_id[parameter_id]!r} across groups")
            relative_name = name_by_id[parameter_id]
            if relative_name in seen_names:
                raise ValueError(f"duplicate optimizer parameter name {relative_name!r} across groups")
            seen_ids.add(parameter_id)
            seen_names.add(relative_name)
            names.append(relative_name)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate optimizer parameter names in group {group_name!r}")
        groups.append({"name": group_name, "parameters": names})
    return groups


def load_optimizer_parameter_names(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    r"""读取并严格验证精简optimizer名称ledger。

    ledger本身的SHA绑定防止名称映射在inspect与实际加载之间被替换；其中的checkpoint SHA再绑定对应
    source权重。schema只接受两个完整optimizer入口和每组唯一相对参数名，不会忽略未知字段或重复JSON键。

    Args:
        path (str | Path): 精简名称ledger JSON文件。
        expected_checkpoint_sha256 (str): warm-start identity声明的source checkpoint SHA-256。
        expected_ledger_sha256 (str): 由调用方预先核验的ledger文件字节SHA-256。

    Returns:
        dict[str, Any]: 经过schema、hash和名称唯一性验证的直接optimizer名称映射；调用方可直接取
            ``result["optimizer"]``与``result["anymani_critic_optimizer"]``。

    Raises:
        ValueError: 文件hash、checkpoint hash、schema字段、组结构或唯一性不满足合同。
    """

    expected_checkpoint = _validate_sha256(expected_checkpoint_sha256, label="expected checkpoint SHA-256")
    expected_ledger = _validate_sha256(expected_ledger_sha256, label="expected ledger SHA-256")
    resolved = Path(path).expanduser().resolve(strict=True)
    actual_ledger = _sha256(resolved)
    if actual_ledger != expected_ledger:
        raise ValueError(
            f"optimizer name ledger SHA-256 mismatch: expected={expected_ledger}, actual={actual_ledger}"
        )
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            document = json.load(
                stream,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"optimizer name ledger is not valid JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "checkpoint_sha256", "optimizers"}:
        raise ValueError("optimizer name ledger must contain exactly schema_version, checkpoint_sha256, optimizers")
    if document["schema_version"] != OPTIMIZER_NAME_LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported optimizer name ledger schema: {document['schema_version']!r}; "
            f"expected {OPTIMIZER_NAME_LEDGER_SCHEMA_VERSION!r}"
        )
    checkpoint_sha = _validate_sha256(document["checkpoint_sha256"], label="ledger checkpoint SHA-256")
    if checkpoint_sha != expected_checkpoint:
        raise ValueError(
            f"optimizer name ledger checkpoint SHA-256 mismatch: expected={expected_checkpoint}, actual={checkpoint_sha}"
        )
    optimizers = document["optimizers"]
    if not isinstance(optimizers, Mapping) or set(optimizers) != {"optimizer", "anymani_critic_optimizer"}:
        raise ValueError("optimizer name ledger must contain exactly optimizer and anymani_critic_optimizer")
    validated_optimizers: dict[str, tuple[dict[str, Any], ...]] = {}
    for optimizer_name in ("optimizer", "anymani_critic_optimizer"):
        validated_optimizers[optimizer_name] = _validate_name_groups(
            optimizers[optimizer_name], context=f"optimizers.{optimizer_name}"
        )
    return {
        name: [
            {"name": group["name"], "parameters": list(group["parameters"])}
            for group in groups
        ]
        for name, groups in validated_optimizers.items()
    }


def _validate_source_state(
    source_state: Mapping[str, Any],
    source_name_groups: object,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    r"""核验source Adam state与名称ledger的组/整数ID对应关系，并生成name→state索引。"""

    if not isinstance(source_state, Mapping) or set(source_state) != _STATE_SCHEMA_KEYS:
        raise ValueError("source optimizer state must contain exactly state and param_groups")
    name_groups = _validate_name_groups(source_name_groups, context="source_name_groups")
    raw_state = source_state["state"]
    raw_groups = source_state["param_groups"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("source optimizer state.state must be a mapping")
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes, bytearray)) or not raw_groups:
        raise ValueError("source optimizer state.param_groups must be a non-empty list")
    source_groups: list[dict[str, Any]] = []
    source_group_by_name: dict[str | None, dict[str, Any]] = {}
    referenced_ids: list[int] = []
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"source optimizer param_groups[{group_index}] must be a mapping")
        group_name = _validate_group_name(raw_group.get("name"), context=f"source param_groups[{group_index}]")
        if group_name in source_group_by_name:
            raise ValueError(f"duplicate source optimizer group name {group_name!r}")
        raw_ids = raw_group.get("params")
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes, bytearray)) or not raw_ids:
            raise ValueError(f"source optimizer param_groups[{group_index}].params must be non-empty")
        ids: list[int] = []
        for parameter_index, raw_id in enumerate(raw_ids):
            identifier = _parameter_id(raw_id, context=f"source param_groups[{group_index}].params[{parameter_index}]")
            if identifier in referenced_ids:
                raise ValueError(f"duplicate source optimizer parameter id {identifier}")
            referenced_ids.append(identifier)
            ids.append(identifier)
        group = {"name": group_name, "params": ids, "raw": raw_group}
        source_groups.append(group)
        source_group_by_name[group_name] = group
    if len(source_groups) != len(name_groups):
        raise ValueError(
            f"source optimizer group count mismatch: state={len(source_groups)}, names={len(name_groups)}"
        )
    name_group_by_name = {group["name"]: group for group in name_groups}
    if set(source_group_by_name) != set(name_group_by_name):
        raise ValueError(
            f"source optimizer group names mismatch: state={sorted(source_group_by_name, key=str)}, "
            f"names={sorted(name_group_by_name, key=str)}"
        )

    state_by_id: dict[int, Mapping[str, Any]] = {}
    for raw_id, state_entry in raw_state.items():
        identifier = _parameter_id(raw_id, context="source optimizer state")
        if identifier in state_by_id:
            raise ValueError(f"duplicate source optimizer state id {identifier}")
        if not isinstance(state_entry, Mapping):
            raise ValueError(f"source optimizer state[{identifier}] must be a mapping")
        state_by_id[identifier] = state_entry
    referenced_set = set(referenced_ids)
    state_set = set(state_by_id)
    missing_ids = referenced_set - state_set
    orphan_ids = state_set - referenced_set
    if missing_ids:
        raise ValueError(f"source optimizer state is missing parameter ids: {sorted(missing_ids)}")
    if orphan_ids:
        raise ValueError(f"orphan optimizer state ids are not referenced by any parameter group: {sorted(orphan_ids)}")

    state_by_name: dict[str, dict[str, Any]] = {}
    ordered_groups: list[dict[str, Any]] = []
    for source_group in source_groups:
        name_group = name_group_by_name[source_group["name"]]
        if len(name_group["parameters"]) != len(source_group["params"]):
            raise ValueError(
                f"source optimizer group {source_group['name']!r} member count mismatch: "
                f"state={len(source_group['params'])}, names={len(name_group['parameters'])}"
            )
        parameters: list[str] = []
        for parameter_name, identifier in zip(name_group["parameters"], source_group["params"], strict=True):
            if parameter_name in state_by_name:
                raise ValueError(f"duplicate source optimizer parameter name {parameter_name!r}")
            state_by_name[parameter_name] = {
                "id": identifier,
                "state": state_by_id[identifier],
                "group": source_group,
            }
            parameters.append(parameter_name)
        ordered_groups.append({"name": source_group["name"], "parameters": parameters, "state_group": source_group})
    if len(state_by_name) != len(referenced_ids):
        raise ValueError("source optimizer name mapping does not cover each referenced parameter exactly once")
    return dict(source_state), state_by_name, tuple(ordered_groups)


def _target_groups(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, Parameter],
) -> tuple[dict[str, Parameter], tuple[dict[str, Any], ...]]:
    r"""以真实target Parameter对象构造optimizer组的相对名称索引。"""

    if not isinstance(named_parameters, Mapping):
        raise TypeError("named_parameters must be a name-to-Parameter mapping")
    name_by_id: dict[int, str] = {}
    for raw_name, parameter in named_parameters.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("named_parameters keys must be non-empty strings")
        if not isinstance(parameter, Parameter):
            raise TypeError(f"named_parameters[{raw_name!r}] is not torch.nn.Parameter")
        parameter_id = id(parameter)
        if parameter_id in name_by_id:
            raise ValueError(f"Parameter object {parameter_id} has duplicate names in named_parameters")
        relative_name = _relative_parameter_name(raw_name)
        name_by_id[parameter_id] = relative_name

    # 只从optimizer实际拥有的参数建立name索引；完整package映射同时含Actor/Critic时，
    # 两个分支可能有同名局部层，但它们不会被错误地互相覆盖。
    parameter_by_name: dict[str, Parameter] = {}
    groups: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    seen_group_names: set[str | None] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = _validate_group_name(group.get("name"), context=f"target param_groups[{group_index}]")
        if group_name in seen_group_names:
            raise ValueError(f"duplicate target optimizer group name {group_name!r}")
        seen_group_names.add(group_name)
        parameters = group.get("params")
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)) or not parameters:
            raise ValueError(f"target param_groups[{group_index}].params must be non-empty")
        names: list[str] = []
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("target optimizer parameter groups must contain torch.nn.Parameter objects")
            parameter_id = id(parameter)
            if parameter_id not in name_by_id:
                raise ValueError(f"target optimizer parameter {parameter_id} is absent from named_parameters")
            if parameter_id in seen_ids:
                raise ValueError(f"duplicate target optimizer parameter {name_by_id[parameter_id]!r}")
            relative_name = name_by_id[parameter_id]
            if relative_name in seen_names:
                raise ValueError(f"duplicate target optimizer parameter name {relative_name!r}")
            seen_ids.add(parameter_id)
            seen_names.add(relative_name)
            parameter_by_name[relative_name] = parameter
            names.append(relative_name)
        groups.append({"name": group_name, "parameters": names, "group": group})
    return parameter_by_name, tuple(groups)


def _validate_group_semantics(source_group: Mapping[str, Any], target_group: Mapping[str, Any], *, name: object) -> None:
    r"""比较source/target除学习率外的Adam语义；target的组对象保持原样。"""

    source_name = _validate_group_name(source_group.get("name"), context="source")
    target_name = _validate_group_name(target_group.get("name"), context="target")
    if source_name != target_name or source_name != name:
        raise ValueError(
            f"optimizer group metadata mismatch for {name!r}: source={source_name!r}, target={target_name!r}"
        )
    source_meta = set(source_group) - _GROUP_META_EXCLUDED_KEYS
    target_meta = set(target_group) - _GROUP_META_EXCLUDED_KEYS
    if source_meta != target_meta:
        raise ValueError(
            f"optimizer Adam metadata keys mismatch for group {name!r}: "
            f"source={sorted(source_meta)}, target={sorted(target_meta)}"
        )
    for key in sorted(source_meta):
        if not _values_equal(source_group[key], target_group[key]):
            raise ValueError(f"optimizer Adam semantic mismatch for group {name!r} field {key!r}")
    source_lr = _finite_scalar(source_group.get("lr"), context=f"source group {name!r} lr")
    target_lr = _finite_scalar(target_group.get("lr"), context=f"target group {name!r} lr")
    if target_lr < 0.0 or source_lr < 0.0:
        raise ValueError(f"optimizer learning rates must be non-negative for group {name!r}")
    if bool(source_group.get("amsgrad", False)) or bool(target_group.get("amsgrad", False)):
        raise ValueError("load_named_optimizer_state supports only Adam with amsgrad=False")


def _validated_state_entry(
    state_entry: Mapping[str, Any],
    parameter: Parameter,
    *,
    parameter_name: str,
) -> tuple[dict[str, Any], int]:
    r"""验证单个source Adam状态的字段、shape、dtype、有限性和step，并复制到target device。"""

    if set(state_entry) != _STATE_ENTRY_KEYS:
        raise ValueError(
            f"source optimizer state for {parameter_name!r} must contain exactly step, exp_avg, exp_avg_sq"
        )
    step_value = state_entry["step"]
    step_number = _finite_scalar(step_value, context=f"source optimizer {parameter_name!r} step")
    if step_number < 0.0 or not step_number.is_integer():
        raise ValueError(f"source optimizer {parameter_name!r} step must be a non-negative integer")
    staged: dict[str, Any] = {}
    for key in ("exp_avg", "exp_avg_sq"):
        value = state_entry[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"source optimizer {parameter_name!r} {key} must be a tensor")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"source optimizer {parameter_name!r} {key} shape mismatch: "
                f"source={tuple(value.shape)}, target={tuple(parameter.shape)}"
            )
        if value.dtype != parameter.dtype:
            raise ValueError(
                f"source optimizer {parameter_name!r} {key} dtype mismatch: "
                f"source={value.dtype}, target={parameter.dtype}"
            )
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"source optimizer {parameter_name!r} {key} is non-finite")
        else:
            raise ValueError(f"source optimizer {parameter_name!r} {key} must use a floating dtype")
        staged[key] = value.detach().clone().to(device=parameter.device)  # 目标Parameter拥有独立state存储
    if isinstance(step_value, torch.Tensor):
        staged_step = step_value.detach().clone().to(device=parameter.device)
    else:
        staged_step = torch.tensor(step_number, dtype=torch.float32, device=parameter.device)
    staged["step"] = staged_step
    return staged, int(step_number)


def _zero_state(parameter: Parameter) -> dict[str, torch.Tensor]:
    r"""为显式allowlist中的新参数构造Adam零状态，保持参数dtype/device。"""

    return {
        "step": torch.zeros((), dtype=torch.float32, device=parameter.device),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }


def load_named_optimizer_state(
    optimizer: torch.optim.Optimizer,
    source_state: Mapping[str, Any],
    source_name_groups: Sequence[Mapping[str, Any]],
    named_parameters: Mapping[str, Parameter],
    *,
    allowed_new_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    r"""按参数名恢复Adam moments，并保留target学习率与组配置。

    Args:
        optimizer (torch.optim.Optimizer): 已按target模型构造的Adam；其参数对象是最终写入state的owner。
        source_state (Mapping[str, Any]): source ``optimizer.state_dict()``，只接受非AMSGrad Adam合同。
        source_name_groups (Sequence[Mapping[str, Any]]): 与source参数组对应的精简名称组列表；每个成员是
            ``{"name": ..., "parameters": [...]}``，组内顺序用于解释source整数ID。
        named_parameters (Mapping[str, Parameter]): target相对或完整Actor/Critic name→Parameter映射。
        allowed_new_names (tuple[str, ...]): target相对source新增参数的显式allowlist；新增状态严格为零。

    Returns:
        dict[str, Any]: 加载/零初始化参数名、step范围、组名和target学习率的审计报告。

    Raises:
        ValueError: source/target名称、组、Adam语义、状态shape/dtype/finite性或allowlist不一致。
    """

    if not isinstance(allowed_new_names, tuple) or any(
        not isinstance(name, str) or not name for name in allowed_new_names
    ):
        raise TypeError("allowed_new_names must be a tuple of non-empty strings")
    normalized_allowed = tuple(_relative_parameter_name(name) for name in allowed_new_names)
    if len(set(normalized_allowed)) != len(normalized_allowed):
        raise ValueError("duplicate allowed_new_names are ambiguous")

    _raw_source, source_by_name, source_groups = _validate_source_state(source_state, source_name_groups)
    target_by_name, target_groups = _target_groups(optimizer, named_parameters)

    source_names = set(source_by_name)
    target_names = set(target_by_name)
    unknown_source = source_names - target_names
    if unknown_source:
        raise ValueError(f"source optimizer contains names unknown in target: {sorted(unknown_source)}")
    target_new = target_names - source_names
    allowed_set = set(normalized_allowed)
    if allowed_set - target_names:
        raise ValueError(f"allowed_new_names are absent from target optimizer: {sorted(allowed_set - target_names)}")
    if allowed_set & source_names:
        raise ValueError(f"allowed_new_names already exist in source optimizer: {sorted(allowed_set & source_names)}")
    if target_new != allowed_set:
        raise ValueError(
            f"target optimizer contains non-allowlisted new parameters: expected={sorted(allowed_set)}, "
            f"actual={sorted(target_new)}"
        )

    source_group_by_name = {group["name"]: group for group in source_groups}
    target_group_by_name = {group["name"]: group for group in target_groups}
    if set(source_group_by_name) != set(target_group_by_name):
        raise ValueError(
            f"optimizer group names mismatch: source={sorted(source_group_by_name, key=str)}, "
            f"target={sorted(target_group_by_name, key=str)}"
        )
    for group_name, source_group in source_group_by_name.items():
        target_group = target_group_by_name[group_name]
        _validate_group_semantics(source_group["state_group"]["raw"], target_group["group"], name=group_name)
        source_members = set(source_group["parameters"])
        target_members = set(target_group["parameters"])
        if not source_members <= target_members:
            raise ValueError(
                f"optimizer group membership mismatch for {group_name!r}: "
                f"source_only={sorted(source_members - target_members)}"
            )
        target_group_new = target_members - source_members
        if not target_group_new <= allowed_set:
            raise ValueError(
                f"target optimizer group {group_name!r} contains non-allowlisted new names: "
                f"{sorted(target_group_new - allowed_set)}"
            )
        for parameter_name in target_group["parameters"]:
            if parameter_name in source_names and source_by_name[parameter_name]["group"]["name"] != group_name:
                raise ValueError(
                    f"optimizer parameter {parameter_name!r} moved between groups; Adam semantics changed"
                )

    staged_state: dict[Parameter, dict[str, Any]] = {}
    loaded_names: list[str] = []
    initialized_names: list[str] = []
    source_steps: list[int] = []
    for target_group in target_groups:
        for parameter_name in target_group["parameters"]:
            parameter = target_by_name[parameter_name]
            if parameter_name in source_by_name:
                state_entry, step = _validated_state_entry(
                    source_by_name[parameter_name]["state"], parameter, parameter_name=parameter_name
                )
                staged_state[parameter] = state_entry
                loaded_names.append(parameter_name)
                source_steps.append(step)
            else:
                if parameter_name not in allowed_set:
                    raise ValueError(f"target optimizer parameter {parameter_name!r} lacks explicit new-name allowlist")
                staged_state[parameter] = _zero_state(parameter)
                initialized_names.append(parameter_name)

    if set(staged_state) != {target_by_name[name] for name in target_names}:
        raise ValueError("target optimizer state staging did not cover every target parameter exactly once")

    # 所有检查都已通过；这一处是唯一的可变写入点，失败的source不会污染target optimizer。
    optimizer.state.clear()
    for parameter, state in staged_state.items():
        optimizer.state[parameter] = state
    target_learning_rates = [float(group["group"]["lr"]) for group in target_groups]
    report: dict[str, Any] = {
        "schema_version": OPTIMIZER_NAME_LEDGER_SCHEMA_VERSION,
        "loaded_parameter_names": loaded_names,
        "initialized_parameter_names": initialized_names,
        "loaded_state_count": len(loaded_names),
        "initialized_state_count": len(initialized_names),
        "source_parameter_count": len(source_names),
        "target_parameter_count": len(target_names),
        "allowed_new_names": list(normalized_allowed),
        "source_steps": sorted(set(source_steps)),
        "optimizer_group_names": [group["name"] for group in target_groups],
        "target_learning_rates": target_learning_rates,
    }
    return report


__all__ = [
    "OPTIMIZER_NAME_LEDGER_SCHEMA_VERSION",
    "load_named_optimizer_state",
    "load_optimizer_parameter_names",
    "optimizer_parameter_names",
]
