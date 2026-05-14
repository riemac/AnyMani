r"""post-mutate accepted/output self_mode quota 工具。

本模块从 `hand_generator.py` 拆出，是为了遵守 generator 目录的“瘦身原则”：
主 façade 只负责流水线编排，概率分配、mode 强制采样和 provenance 读取这类
helper 逻辑放在独立模块里。
"""

from __future__ import annotations

from typing import Any

from ..asset_base import HandCfg
from ._generation_result import HandGenerationResult

try:
    from .mutate import HandMutator, HandMutatorCfg, MountPerturbCfg
except Exception:
    class MountPerturbCfg:  # type: ignore[no-redef]
        r"""mutate package 不可用时的占位类型，保持 hand_generator fallback 可导入。"""

    HandMutator = Any  # type: ignore[misc, assignment]
    HandMutatorCfg = Any  # type: ignore[misc, assignment]


MOUNT_MODE_ORDER = (
    "identity",
    "general",
    "index_ring_yaw_rot",
    "index_ring_x_pos",
    "index_ring",
)
"""`MountPerturbCfg.self_mode` accepted quota 的固定 tie-break 顺序。"""


def allocate_accepted_mode_quota(probabilities: dict[str, float], n_samples: int) -> dict[str, int]:
    r"""把 accepted/output mode 概率分布确定性转成整数 quota。

    算法是 largest remainder：

    1. $q_i=p_iN$；
    2. 取 floor 得到基础 quota；
    3. 剩余槽位按小数部分从大到小分配；
    4. 小数部分相等时使用 `MOUNT_MODE_ORDER`，不依赖 dict 插入顺序。
    """

    target = max(int(n_samples), 0)
    normalized = {mode: float(probabilities.get(mode, 0.0)) for mode in MOUNT_MODE_ORDER}
    unknown_modes = set(probabilities) - set(MOUNT_MODE_ORDER)
    if unknown_modes:
        raise ValueError(f"unsupported mount_perturb self_mode keys for quota: {sorted(unknown_modes)!r}")
    raw = {mode: normalized[mode] * target for mode in MOUNT_MODE_ORDER}
    quota = {mode: int(raw[mode] // 1) for mode in MOUNT_MODE_ORDER}
    remainder = target - sum(quota.values())
    ranked_modes = sorted(
        MOUNT_MODE_ORDER,
        key=lambda mode: (-(raw[mode] - quota[mode]), MOUNT_MODE_ORDER.index(mode)),
    )
    for mode in ranked_modes[:remainder]:
        quota[mode] += 1
    return {mode: count for mode, count in quota.items() if count > 0}


def mount_perturb_mode_probabilities(cfg: HandMutatorCfg) -> dict[str, float] | None:
    r"""若当前 Mutate 含可 quota 化的 mount_perturb dict mode，则返回概率表。"""

    for _term_name, term_cfg in cfg.ordered_terms():
        if isinstance(term_cfg, MountPerturbCfg) and isinstance(term_cfg.self_mode, dict):
            return {str(mode): float(probability) for mode, probability in term_cfg.self_mode.items()}
    return None


def force_mount_perturb_mode(
    sampled_terms: dict[str, dict[str, Any]],
    *,
    mutator: HandMutator,
    target: HandCfg,
    mode: str,
) -> dict[str, dict[str, Any]]:
    r"""把一组联合采样中的 mount_perturb mode 改写为指定 mode。

    这里会调用 `MountPerturbMutator.sample_one_for_mode()` 重新生成该 mode
    所需的完整 payload，避免“只改 mode 名但缺少 mode 专属随机量”的伪样本。
    """

    patched = {term: dict(payload) for term, payload in sampled_terms.items()}
    for term_name, term_cfg in mutator.cfg.ordered_terms():
        if not isinstance(term_cfg, MountPerturbCfg):
            continue
        runtime = mutator._make_runtime(term_cfg)
        if hasattr(runtime, "sample_one_for_mode"):
            patched[term_name] = {
                "sample": runtime.sample_one_for_mode(target, resolved_mode=mode),
            }
            return patched
    patched["mount_perturb"] = {"resolved_self_mode": mode}
    return patched


def resolved_mount_mode(result: HandGenerationResult | None, sampled_terms: dict[str, dict[str, Any]]) -> str | None:
    r"""从 result metadata 或 sampled_terms 中读取 resolved self_mode。"""

    if result is not None:
        samples = result.metadata.get("post_mutate_samples")
        if isinstance(samples, dict):
            mount = samples.get("mount_perturb")
            if isinstance(mount, dict) and "resolved_self_mode" in mount:
                return str(mount["resolved_self_mode"])
    mount_terms = sampled_terms.get("mount_perturb")
    if isinstance(mount_terms, dict):
        sample = mount_terms.get("sample")
        if isinstance(sample, dict) and "resolved_self_mode" in sample:
            return str(sample["resolved_self_mode"])
        if "resolved_self_mode" in mount_terms:
            return str(mount_terms["resolved_self_mode"])
    return None


__all__ = [
    "MOUNT_MODE_ORDER",
    "allocate_accepted_mode_quota",
    "force_mount_perturb_mode",
    "mount_perturb_mode_probabilities",
    "resolved_mount_mode",
]
