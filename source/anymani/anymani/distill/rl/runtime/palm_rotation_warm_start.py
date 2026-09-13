r"""掌托旋转PPO的新方法分支初始化与来源边界。

``--checkpoint``仍由rl_games执行完整同identity续训；本模块为首次``--actor_init_checkpoint``加载Actor，并在随后
full resume时从target checkpoint恢复同一份只读迁移血缘。默认仅迁移Actor；显式``--init_critic``额外继承具有相同
privileged输入语义的Critic和value normalizer。两套optimizer、课程、ADR、诊断cursor与随机状态均重新初始化。Source与target可使用不同cohort/task
reward，但必须共享actor arm、History30 encoder与retained N040 artifact，否则张量即使shape碰巧相同也拒绝迁移。
"""

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
"""默认初始化模式加载的Actor参数空间；Critic迁移必须显式授权。"""

ACTOR_WARM_START_SCHEMA_VERSION = "1.0.0"
"""写入target method identity的Actor-only迁移证据schema。"""

PHASE_CONTEXTUAL_ADAPTER_KEY = "phase_contextual_adapter.weight"
"""Direct Actor新增的无bias相位适配器参数名；其输入是有界phase clock。"""

PHASE_READOUT_ADAPTER_KEY = "phase_readout_adapter.weight"
"""Structured Critic新增的无bias相位读出参数名；其输出只改变value读出。"""

PHASE_PERIOD_MIN_STEPS = 2
"""phase clock的最短周期；1步周期没有可辨识的相位变化，因而不属于本合同。"""


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


def _validate_phase_period(value: object, *, field_name: str) -> int | None:
    r"""把checkpoint/target中的phase周期收敛为`None`或不小于2的整数。

    ``None``表示关闭phase clock；正整数表示每个episode内的有界周期。这里拒绝`bool`和可转换字符串，
    因为它们会把identity中的声明错误静默改写成另一个运行语义。
    """

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
    r"""提取并验证一个checkpoint namespace中的tensor，不丢弃其余键。

    namespace内的每个值都必须是有限tensor；后续与target module的key、shape和dtype比较在写入前完成，
    使错误不会留下半份Actor/Critic迁移结果。
    """

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
    r"""严格复制module state，且只允许旧无phase到新phase的单一缺项迁移。

    旧参数、RMS及module metadata必须逐键匹配。适配时唯一允许的差集是target新增的`phase_key`；该键不
    从source猜测，而是在target dtype/shape上显式置零。启用到关闭、增加bias、漏掉其它权重和source多出
    未知参数均保持fail-closed。
    """

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
        validated[phase_key] = torch.zeros_like(phase_value)  # 新适配器的相位贡献从严格零函数开始
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
    r"""验证迁移兼容性并返回写入target identity的JSON-safe证据。

    Args:
        path (str | Path): 完整source PPO checkpoint。
        target_arm (str): 目标``base/residual/direct/direct_token``动作生成结构。
        target_history_encoder (str): 目标``tcn/raw_stack``时间编码器。
        target_provider_identity (Mapping[str, Any]): 目标N040 provider identity，必须含retained artifact SHA。
        initialize_critic (bool): 额外继承兼容Critic和value统计，不继承optimizer或课程。
        actor_init_sigma (float | None): 加载后显式设置的共享潜高斯标准差，无量纲；None继承原权重。
        target_phase_period_steps (int | None): 目标phase clock周期；``None``关闭，整数必须不小于2。
        allow_phase_clock_adaptation (bool): 仅显式允许旧无phase到新phase的Direct adapter零初始化。

    Returns:
        dict[str, Any]: Parent SHA、source arm/cohort/task、加载namespace与显式重置组件。
    """

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
    )  # target identity的有界时钟声明
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
    )  # 旧checkpoint缺省字段等价于明确关闭
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
        }  # 新方法初始化证据；无新增learned keys，不能冒充同identity完整resume。
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
        source_sigma = float(source_log_std.detach().exp().item())  # 权重中的真实潜标准差，不用构造默认值代替
        if not math.isfinite(source_sigma) or source_sigma <= 0.0:
            raise ValueError("actor-init source sigma is non-finite or non-positive")
        exploration = {"actor_init_sigma": float(actor_init_sigma), "source_actor_sigma": source_sigma}
    task_contract = identity.get("task_contract")
    task_contract = dict(task_contract) if isinstance(task_contract, Mapping) else {}
    if initialize_critic:
        # 维度相同不代表task变量含义相同；当前critic的最大正向净圈/净圈定义必须与source一致。
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
    if not isinstance(reset_components, list) or (
        not bool(evidence.get("initialize_critic", False)) and "critic" not in reset_components
    ):
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
    actor_init_sigma: float | None = None,
    allow_phase_clock_adaptation: bool = False,
) -> tuple[str, ...]:
    r"""只把source Actor namespace严格加载到目标Actor，并返回有序tensor keys。

    ``strict=True``同时拒绝跨arm结构、缺参数和多参数；调用方应在fresh optimizer构造前执行本函数。
    显式sigma覆盖只在source权重加载后、新采样段之前写共享``global_log_std``，其余Actor参数原样继承。
    对$a=\tanh(\operatorname{atanh}\mu+\sigma\epsilon)$，它保留确定性动作中心网络，但随机动作均值也可变化。
    旧无phase到新phase只在``allow_phase_clock_adaptation=True``时允许，并且仅把唯一
    ``phase_contextual_adapter.weight``缺项初始化为严格零；其它missing/unexpected权重一律拒绝。
    """

    exploration_parameter = None
    target_log_std = None
    if actor_init_sigma is not None:
        if not math.isfinite(actor_init_sigma) or actor_init_sigma <= 0.0:
            raise ValueError("actor-init sigma must be finite and positive")
        parameter = getattr(actor, "global_log_std", None)
        if not isinstance(parameter, nn.Parameter) or parameter.numel() != 1:
            raise ValueError("actor-init sigma override requires one shared target log standard deviation")
        target_log_std = math.log(actor_init_sigma)  # sigma无量纲，网络存储其自然对数
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
    )  # 先完整核对差集，再提交source权重与可审计的零adapter
    if exploration_parameter is not None:
        assert target_log_std is not None  # 参数与数值由同一显式覆盖分支共同验证
        with torch.no_grad():
            exploration_parameter.fill_(target_log_std)  # 均值网络不变，实际探索起点写在权重载入之后
    return tuple(sorted(actor_state))


def load_critic_init_checkpoint(
    model: Any,
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    allow_phase_clock_adaptation: bool = False,
) -> None:
    r"""新方法分支显式继承兼容critic和value统计，仍重置两套optimizer。

    Actor观察改变时，privileged critic输入仍可完全相同。保留其可用估计避免用随机critic破坏已学策略；
    是否适合新的reward/时长仍属实验假设，由warm-start identity明确记录，不称为同identity完整续训。
    旧无phase到新phase只在显式声明时允许唯一``phase_readout_adapter.weight``缺项，并在target
    dtype/shape上置零；value RMS仍以严格state_dict逐项复制。
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    if _sha256(resolved) != expected_checkpoint_sha256:
        raise ValueError("critic-init checkpoint changed after inspection")
    _, _, state = _checkpoint_document(resolved)
    prefix = "a2c_network.package.critic."
    critic_state = _namespace_state(state, prefix=prefix, namespace_name="critic")  # 不加载actor/optimizer
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
    )  # Critic phase readout与Actor adapter分开验证，避免混淆权限
    _load_module_state_with_optional_phase(
        model.value_mean_std,
        normalizer_state,
        namespace_name="value normalizer",
        phase_key="__never_allow_phase__",
        allow_phase_clock_adaptation=False,
    )  # RMS均值/方差/count无迁移缺项例外


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
