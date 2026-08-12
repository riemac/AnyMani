r"""mutate-only 独立联合 proposal 与逐槽 rejection sampling。

`self_mode` 字典描述的是每个候选的 proposal 概率，不是最终 accepted 资产的
边缘配额。validator 会改变 accepted 分布；这种偏移是几何合法性筛选的真实结果，
必须通过 proposed/accepted 统计显式观察，不能用 forced mode 把它抹平。
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..mutate.tip_replace import iter_tip_types_from_sample
from ..result import HandGenerationResult

if TYPE_CHECKING:
    from ...asset_base import HandCfg
    from ..hand_generator import HandGenerator
    from ..mutate import HandMutator


def run_mutate_batch_with_independent_proposals(
    *,
    generator: HandGenerator,
    mutator: HandMutator,
    source_hand: HandCfg,
    target_count: int,
    attempts_per_variant: int,
    seed: int,
) -> Iterator[HandGenerationResult]:
    r"""逐槽生成 post-mutate variants，并保留真实 proposal/acceptance 证据。

    第 $j$ 个计划槽位最多消费 $K$ 个候选：

    $$
    \xi_{j,k}\sim\prod_t p_t(\text{mode}_t,\text{parameters}_t),\qquad
    k=1,\ldots,K.
    $$

    任一候选被拒绝后，下一次重新抽取完整 $\xi_{j,k+1}$。若 $K$ 次均失败，
    槽位 $j$ 记为 shortfall 后继续 $j+1$；因此一个困难槽位不会偷用其他槽位预算。
    """

    context = generator._ensure_run_context()
    summary = context.summary
    sampling_stats = {
        "seed": int(seed),
        "planned_variants": int(target_count),
        "attempts_per_variant": int(attempts_per_variant),
        "successful_variants": 0,
        "shortfall": int(target_count),
        "slots": [],
    }
    mode_stats: dict[str, dict[str, dict[str, int]]] = {}
    joint_mode_stats: dict[str, dict[str, int]] = {"proposed": {}, "accepted": {}}
    tip_type_stats: dict[str, dict[str, int]] = {"proposed": {}, "accepted": {}}
    summary["post_mutate_sampling"] = sampling_stats
    summary["post_mutate_mode_stats"] = mode_stats
    summary["post_mutate_joint_mode_stats"] = joint_mode_stats
    summary["post_mutate_tip_type_stats"] = tip_type_stats
    context.write_summary()

    caller_random_state = random.getstate()
    random.seed(seed)
    try:
        for slot_index in range(target_count):
            slot_stats: dict[str, Any] = {
                "slot_index": slot_index,
                "attempts": 0,
                "accepted": False,
                "rejections": [],
            }
            sampling_stats["slots"].append(slot_stats)

            for _ in range(attempts_per_variant):
                sampled_terms = _sample_joint_proposal(mutator, source_hand)
                slot_stats["attempts"] += 1
                _record_mode_counts(mode_stats, sampled_terms, bucket="proposed")
                _increment(joint_mode_stats["proposed"], _joint_mode_key(sampled_terms))
                _record_tip_type_counts(tip_type_stats, sampled_terms, bucket="proposed")

                result = generator._generate_once(
                    hand_preset_name=None,
                    connectivity_preset_name=None,
                    sampled_mutation_terms=sampled_terms,
                )
                if result is None:
                    _record_mode_counts(mode_stats, sampled_terms, bucket="rejected")
                    slot_stats["rejections"].append(
                        {
                            "stage": context.last_rejection_stage,
                            "error_codes": list(context.last_rejection_error_codes),
                        }
                    )
                    context.write_summary()
                    continue

                accepted_samples = result.metadata.get("post_mutate_samples", sampled_terms)
                _record_mode_counts(mode_stats, accepted_samples, bucket="accepted")
                _increment(joint_mode_stats["accepted"], _joint_mode_key(accepted_samples))
                _record_tip_type_counts(tip_type_stats, accepted_samples, bucket="accepted")
                slot_stats["accepted"] = True
                slot_stats["result_id"] = result.metadata.get("id")
                sampling_stats["successful_variants"] += 1
                sampling_stats["shortfall"] = target_count - sampling_stats["successful_variants"]
                context.write_summary()
                yield result
                break

            sampling_stats["shortfall"] = target_count - sampling_stats["successful_variants"]
            context.write_summary()
    finally:
        random.setstate(caller_random_state)
        context.write_summary()


def _sample_joint_proposal(
    mutator: HandMutator,
    source_hand: HandCfg,
) -> dict[str, dict[str, Any]]:
    r"""通过唯一联合采样入口抽取一个完整候选。"""

    sampled_batch = mutator.sample_batch(source_hand, batch_size=1)
    if not sampled_batch:
        raise RuntimeError("post-mutate joint sampler returned an empty batch for batch_size=1")
    return sampled_batch[0]


def _resolved_mode(payload: Any) -> str | None:
    r"""兼容采样期 `{"sample": ...}` 与结果期扁平 payload。"""

    if not isinstance(payload, dict):
        return None
    sample = payload.get("sample")
    if isinstance(sample, dict) and "resolved_self_mode" in sample:
        return str(sample["resolved_self_mode"])
    if "resolved_self_mode" in payload:
        return str(payload["resolved_self_mode"])
    return None


def _joint_mode_key(samples: Any) -> str:
    r"""把一次真实联合 mode realization 编成稳定、可读的 summary 键。"""

    if not isinstance(samples, dict):
        return "unresolved"
    resolved = [
        f"{term_name}={mode}"
        for term_name, payload in samples.items()
        if (mode := _resolved_mode(payload)) is not None
    ]
    return "|".join(resolved) if resolved else "unresolved"


def _record_mode_counts(stats: dict[str, dict[str, dict[str, int]]], samples: Any, *, bucket: str) -> None:
    r"""按 term/mode 记录边缘 proposal、accepted 与 rejected 次数。"""

    if not isinstance(samples, dict):
        return
    for term_name, payload in samples.items():
        mode = _resolved_mode(payload)
        if mode is None:
            continue
        mode_record = stats.setdefault(str(term_name), {}).setdefault(
            mode,
            {"proposed": 0, "accepted": 0, "rejected": 0},
        )
        mode_record[bucket] += 1


def _record_tip_type_counts(stats: dict[str, dict[str, int]], samples: Any, *, bucket: str) -> None:
    r"""记录候选实际包含的 per-finger tip type，观察 validator 的选择偏移。"""

    if not isinstance(samples, dict):
        return
    for tip_type in iter_tip_types_from_sample(samples.get("tip_replace")):
        _increment(stats[bucket], tip_type)


def _increment(stats: dict[str, int], key: str) -> None:
    r"""对 summary 中一个离散事件做可加和计数。"""

    stats[key] = int(stats.get(key, 0)) + 1


__all__ = ["run_mutate_batch_with_independent_proposals"]
