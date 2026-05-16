r"""mutate-only accepted/output quota 批量执行 helper。

这个模块只承载 mutate-only 的 accepted-mode batch orchestration，不承载：

- `HandGeneratorCfg` 的公开配置字段；
- 单样本 build / mutate / validate / export 主流程；
- 具体某个 mutator term 的几何 / 物理细节。

这样 `hand_generator.py` 可以继续作为最高 façade，而不是把 quota 统计、
forced-mode 重采、summary 回写和错误诊断全部揉在同一个大文件里。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..quota.accepted_mode import (
    AcceptedModeTermSpec,
    allocate_accepted_mode_quota,
    expand_quota_schedule,
    force_mode_terms,
    resolved_term_mode,
)
from ..result import HandGenerationResult
from ..mutate.tip_replace import iter_tip_types_from_sample

if TYPE_CHECKING:
    from ..hand_generator import HandGenerator
    from ..mutate import HandMutator
    from ...asset_base import HandCfg


def run_mutate_batch_with_accepted_mode_quota(
    *,
    generator: "HandGenerator",
    mutator: "HandMutator",
    source_hand: "HandCfg",
    accepted_mode_terms: dict[str, AcceptedModeTermSpec],
    target_count: int,
    max_attempts: int,
    sample_mutation_terms_fn: Callable[["HandMutator", "HandCfg"], dict[str, dict[str, float]]],
) -> Iterator[HandGenerationResult]:
    r"""按 accepted/output mode quota 执行 mutate-only 批量生成。

    这里兑现的是新的统计 contract：

    - 任一 `self_mode=dict` term 都不再是 proposal prior；
    - 它描述最终 accepted 样本的该 term 边缘 mode 分布；
    - 某个 slot 被指定为一组 forced modes 后，validator 拒绝时就在同组 forced modes 内补采；
    - 任何 shortfall 都 fail-hard，不允许其他 mode 静默填坑。
    """

    schedules = {
        term_name: expand_quota_schedule(
            allocate_accepted_mode_quota(
                spec.probabilities,
                target_count,
                mode_order=spec.mode_order,
                label=f"{term_name}.self_mode",
            ),
            mode_order=spec.mode_order,
        )
        for term_name, spec in accepted_mode_terms.items()
    }
    diagnostics = {
        term_name: {
            mode: {
                "target_quota": quota,
                "proposed": 0,
                "accepted": 0,
                "emitted": 0,
                "shortfall": quota,
                "rejected_by_validator": 0,
                "rejected_by_unsupported_geometry": 0,
                "rejected_by_incomplete_certificate": 0,
                "rejected_by_budget": 0,
                "representative_failures": [],
            }
            for mode, quota in allocate_accepted_mode_quota(
                spec.probabilities,
                target_count,
                mode_order=spec.mode_order,
                label=f"{term_name}.self_mode",
            ).items()
        }
        for term_name, spec in accepted_mode_terms.items()
    }
    generator._ensure_run_context().summary["post_mutate_mode_stats"] = diagnostics  # 运行时汇总按 term/mode 挂到 summary
    tip_type_stats = generator._ensure_run_context().summary.setdefault(
        "post_mutate_tip_type_stats",
        {"proposed": {}, "accepted": {}},
    )
    attempts_used = 0  # accepted quota 的全局尝试预算，跨所有目标样本共享

    for sample_index in range(target_count):
        forced_modes = {
            term_name: schedule[sample_index]
            for term_name, schedule in schedules.items()
        }  # 每个 accepted 槽位都先固定一组 term-level mode；失败时只在该组合内补采
        while True:
            if attempts_used >= max_attempts:
                for term_name, mode in forced_modes.items():
                    diagnostics[term_name][mode]["rejected_by_budget"] += 1
                generator._update_mode_quota_shortfall(diagnostics)  # fail-hard 前先把 shortfall 刷到 summary
                generator._write_run_summary()
                raise RuntimeError(
                    "post-mutate accepted self_mode quota shortfall; "
                    f"forced_modes={forced_modes!r}, accepted_slots={sample_index}, "
                    f"attempted={attempts_used}, budget={max_attempts}, diagnostics={diagnostics!r}"
                )

            try:
                sampled_terms = mutator.sample_batch(source_hand, batch_size=1)[0]  # 优先复用 mutator 的批量采样
            except Exception:
                sampled_terms = sample_mutation_terms_fn(mutator, source_hand)  # 保留单样本 fallback，避免批量路径异常时整轮失效
            sampled_terms = force_mode_terms(
                sampled_terms,
                mutator=mutator,
                target=source_hand,
                forced_modes=forced_modes,
            )
            for term_name, mode in forced_modes.items():
                diagnostics[term_name][mode]["proposed"] += 1
            _record_tip_type_counts(tip_type_stats, sampled_terms, bucket="proposed")
            attempts_used += 1

            result = generator._generate_once(
                hand_preset_name=None,
                connectivity_preset_name=None,
                sampled_mutation_terms=sampled_terms,
            )
            if result is None:
                reason = generator._classify_last_rejection()  # validator / unsupported / incomplete 三类粗分
                for term_name, mode in forced_modes.items():
                    diagnostics[term_name][mode][reason] += 1
                    failures = diagnostics[term_name][mode]["representative_failures"]
                    if len(failures) < 5:
                        failures.append({"reason": reason, "sampled_terms": sampled_terms})
                generator._update_mode_quota_shortfall(diagnostics)
                generator._ensure_run_context().summary["post_mutate_mode_stats"] = diagnostics
                continue

            for term_name, expected_mode in forced_modes.items():
                resolved_mode = resolved_term_mode(result, sampled_terms, term_name=term_name)
                if resolved_mode != expected_mode:
                    raise RuntimeError(
                        "accepted self_mode quota invariant broken; "
                        f"term={term_name!r}, expected mode={expected_mode!r}, got {resolved_mode!r}"
                    )

            for term_name, mode in forced_modes.items():
                diagnostics[term_name][mode]["accepted"] += 1
                diagnostics[term_name][mode]["emitted"] += 1
            _record_tip_type_counts(tip_type_stats, result.metadata.get("post_mutate_samples"), bucket="accepted")
            generator._update_mode_quota_shortfall(diagnostics)
            generator._ensure_run_context().summary["post_mutate_mode_stats"] = diagnostics
            generator._ensure_run_context().summary["post_mutate_tip_type_stats"] = tip_type_stats
            generator._write_run_summary()
            yield result
            break


def _record_tip_type_counts(stats: dict[str, dict[str, int]], samples: Any, *, bucket: str) -> None:
    r"""把 `tip_replace` 的 per-finger tip_type 计数写入 summary。

    `tip_range` 在 v1 中是 proposal 分布，而不是 accepted quota。因此这里同时
    记录 proposed 与 accepted，让研究者能直接观察 validator 是否对某些 tip_type
    产生偏置。
    """

    if not isinstance(samples, dict):
        return
    tip_replace_sample = samples.get("tip_replace")
    tip_types = iter_tip_types_from_sample(tip_replace_sample)
    if not tip_types:
        return
    bucket_stats = stats.setdefault(bucket, {})
    for tip_type in tip_types:
        bucket_stats[tip_type] = int(bucket_stats.get(tip_type, 0)) + 1


__all__ = ["run_mutate_batch_with_accepted_mode_quota"]
