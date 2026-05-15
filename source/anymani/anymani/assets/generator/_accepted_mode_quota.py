r"""post-mutate 高层 mode 的 accepted/output quota 工具。

本模块把原先只服务 `mount_perturb` 的 accepted quota 逻辑推广成：

1. 任何 mutator term 只要声明 `self_mode=dict[str, float]`；
2. 且 runtime 提供 `sample_one_for_mode(target, resolved_mode=...)`；
3. generator 就可以把这组概率解释为 **accepted/output** 分布，而不是 proposal prior。

这里刻意把“mode 概率 -> 整数 quota”“强制某个 term 采指定 mode”“从结果里回读
resolved mode”三个动作抽出来，是为了让 `hand_generator.py` 继续只做流水线编排，
不在 façade 里硬编码某个具体算子的 mode 细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..asset_base import HandCfg
from ._generation_result import HandGenerationResult

try:
    from .mutate import HandMutator, HandMutatorCfg, LimitTweakCfg, MountPerturbCfg
except Exception:
    class MountPerturbCfg:  # type: ignore[no-redef]
        r"""mutate package 不可用时的占位类型，保持 generator fallback 可导入。"""

    class LimitTweakCfg:  # type: ignore[no-redef]
        r"""mutate package 不可用时的占位类型，保持 generator fallback 可导入。"""

    HandMutator = Any  # type: ignore[misc, assignment]
    HandMutatorCfg = Any  # type: ignore[misc, assignment]


MOUNT_MODE_ORDER = (
    "identity",
    "general",
    "index_ring_yaw_rot",
    "index_ring_x_pos",
    "index_ring",
)
"""`MountPerturbCfg.self_mode` 的固定 tie-break 顺序。"""

LIMIT_TWEAK_MODE_ORDER = (
    "identity",
    "disturb",
    "homologous_non_thumb",
)
"""`LimitTweakCfg.self_mode` 的固定 tie-break 顺序。"""


@dataclass(frozen=True)
class AcceptedModeTermSpec:
    r"""一项支持 accepted/output mode quota 的 mutator term 规格。

    Attributes:
        term_name (str): `HandMutatorCfg` 中的 term 名，如 `mount_perturb` / `limit_tweak`。
        mode_order (tuple[str, ...]): 当前 term 合法 mode 的稳定顺序，供 quota tie-break 使用。
        probabilities (dict[str, float]): 用户声明的 accepted/output 概率分布。
    """

    term_name: str
    mode_order: tuple[str, ...]
    probabilities: dict[str, float]


def allocate_accepted_mode_quota(
    probabilities: dict[str, float],
    n_samples: int,
    *,
    mode_order: tuple[str, ...] | None = None,
    label: str = "self_mode",
) -> dict[str, int]:
    r"""把 accepted/output mode 概率分布确定性转成整数 quota。

    算法是 largest remainder：

    1. $q_i=p_iN$；
    2. 取 floor 得到基础 quota；
    3. 剩余槽位按小数部分从大到小分配；
    4. 小数部分相等时使用稳定的 `mode_order`，不依赖 dict 插入顺序。

    Args:
        probabilities (dict[str, float]): mode 概率表。
        n_samples (int): 目标 accepted 样本总数。
        mode_order (tuple[str, ...] | None): 当前 term 的固定 mode 顺序。
        label (str): 报错时使用的字段名，便于指出是哪个高层配置字段出错。

    Returns:
        dict[str, int]: 只保留正 quota 项的整数配额表。
    """

    stable_order = tuple(mode_order or MOUNT_MODE_ORDER)  # mount 旧测试默认仍可直接复用
    target = max(int(n_samples), 0)  # accepted/output 的目标样本总数
    normalized = {mode: float(probabilities.get(mode, 0.0)) for mode in stable_order}  # 缺省 mode 视作 0 概率
    unknown_modes = set(probabilities) - set(stable_order)  # 非法 mode 立即报错，避免统计静默跑偏
    if unknown_modes:
        raise ValueError(f"unsupported {label} keys for quota: {sorted(unknown_modes)!r}")

    raw = {mode: normalized[mode] * target for mode in stable_order}  # 浮点期望配额 $p_iN$
    quota = {mode: int(raw[mode] // 1) for mode in stable_order}  # 先取 floor，得到基础整数配额
    remainder = target - sum(quota.values())  # 还剩多少个 accepted 槽位需要补分配
    ranked_modes = sorted(
        stable_order,
        key=lambda mode: (-(raw[mode] - quota[mode]), stable_order.index(mode)),
    )  # 小数部分越大越优先；平手时按固定顺序
    for mode in ranked_modes[:remainder]:
        quota[mode] += 1
    return {mode: count for mode, count in quota.items() if count > 0}


def mode_term_specs(cfg: HandMutatorCfg) -> dict[str, AcceptedModeTermSpec]:
    r"""收集当前 Mutate 中所有需要按 accepted/output quota 处理的 mode term。

    当前只显式支持 `mount_perturb` 与 `limit_tweak` 两类 term；未来若新增 mode 化
    mutator，只需在这里补上 cfg 类型与 `mode_order` 映射，不需要改 generator 主循环。
    """

    specs: dict[str, AcceptedModeTermSpec] = {}
    for term_name, term_cfg in cfg.ordered_terms():
        if isinstance(term_cfg, MountPerturbCfg) and isinstance(term_cfg.self_mode, dict):
            specs[term_name] = AcceptedModeTermSpec(
                term_name=term_name,
                mode_order=MOUNT_MODE_ORDER,
                probabilities={str(mode): float(probability) for mode, probability in term_cfg.self_mode.items()},
            )
            continue
        if isinstance(term_cfg, LimitTweakCfg) and isinstance(term_cfg.self_mode, dict):
            specs[term_name] = AcceptedModeTermSpec(
                term_name=term_name,
                mode_order=LIMIT_TWEAK_MODE_ORDER,
                probabilities={str(mode): float(probability) for mode, probability in term_cfg.self_mode.items()},
            )
    return specs


def expand_quota_schedule(quota: dict[str, int], *, mode_order: tuple[str, ...]) -> list[str]:
    r"""把整数 quota 展开成长度为 `N` 的稳定 mode 序列。

    这里故意不做额外打乱：accepted/output quota 的首要目标是统计确定性，而不是 proposal
    随机性。若后续需要“同 quota 但不同输出顺序”的随机化，可在 generator 层显式加洗牌。
    """

    return [mode for mode in mode_order for _ in range(int(quota.get(mode, 0)))]


def force_mode_terms(
    sampled_terms: dict[str, dict[str, Any]],
    *,
    mutator: HandMutator,
    target: HandCfg,
    forced_modes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    r"""把一组联合采样中的若干 mode term 改写为指定 mode。

    这里会调用各 runtime 的 `sample_one_for_mode()` 重新生成该 mode 所需的完整 payload，
    避免“只改 mode 名但缺少 mode 专属随机量”的伪样本。
    """

    patched = {term: dict(payload) for term, payload in sampled_terms.items()}
    for term_name, term_cfg in mutator.cfg.ordered_terms():
        if term_name not in forced_modes:
            continue
        runtime = mutator._make_runtime(term_cfg)
        if not hasattr(runtime, "sample_one_for_mode"):
            raise RuntimeError(
                f"mode-quota term {term_name!r} does not implement sample_one_for_mode(); "
                "accepted/output mode forcing would create pseudo-samples"
            )
        patched[term_name] = {
            "sample": runtime.sample_one_for_mode(target, resolved_mode=str(forced_modes[term_name])),
        }
    return patched


def resolved_term_mode(
    result: HandGenerationResult | None,
    sampled_terms: dict[str, dict[str, Any]],
    *,
    term_name: str,
) -> str | None:
    r"""从 result metadata 或 sampled_terms 中读取某个 term 的 resolved mode。"""

    if result is not None:
        samples = result.metadata.get("post_mutate_samples")
        if isinstance(samples, dict):
            term_payload = samples.get(term_name)
            if isinstance(term_payload, dict) and "resolved_self_mode" in term_payload:
                return str(term_payload["resolved_self_mode"])
    term_terms = sampled_terms.get(term_name)
    if isinstance(term_terms, dict):
        sample = term_terms.get("sample")
        if isinstance(sample, dict) and "resolved_self_mode" in sample:
            return str(sample["resolved_self_mode"])
        if "resolved_self_mode" in term_terms:
            return str(term_terms["resolved_self_mode"])
    return None


__all__ = [
    "AcceptedModeTermSpec",
    "LIMIT_TWEAK_MODE_ORDER",
    "MOUNT_MODE_ORDER",
    "allocate_accepted_mode_quota",
    "expand_quota_schedule",
    "force_mode_terms",
    "mode_term_specs",
    "resolved_term_mode",
]
