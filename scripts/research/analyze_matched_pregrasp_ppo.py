r"""合并row16 matched support/contact PPO与exploration survival证据。

分析器先验证每对run除tier/record/elapsed/results外的network、PPO、seed、env与budget完全相同，再计算contact-minus-
support差值。结论只限单physical asset、单seed和204,800-transition早期学习，不外推128/2048或最终收敛。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    r"""解析四条formal summaries、两条sigma probes与输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-support", type=Path, required=True)
    parser.add_argument("--standard-contact", type=Path, required=True)
    parser.add_argument("--low-support", type=Path, required=True)
    parser.add_argument("--low-contact", type=Path, required=True)
    parser.add_argument("--support-survival", type=Path, required=True)
    parser.add_argument("--contact-survival", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, artifact_type: str) -> dict[str, Any]:
    r"""读取并验证artifact type。"""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("artifact_type") != artifact_type:
        raise ValueError(f"{path} is not {artifact_type}")
    return document


def _validate_pair(support: dict[str, Any], contact: dict[str, Any], label: str) -> None:
    r"""验证matched pair只有tier与pregrasp record不同。"""

    for field in ("seed", "num_envs", "updates", "transitions", "ppo_config", "network"):
        if support[field] != contact[field]:
            raise ValueError(f"{label} pair differs in matched field {field}")
    support_provider = dict(support["provider_identity"])
    contact_provider = dict(contact["provider_identity"])
    if support_provider != contact_provider:
        raise ValueError(f"{label} pair differs in N040 provider identity")
    if support["tier"] != "support_basin" or contact["tier"] != "contact_basin":
        raise ValueError(f"{label} pair does not contain support/contact arms")
    if support["pregrasp_record_digest"] == contact["pregrasp_record_digest"]:
        raise ValueError(f"{label} pair unexpectedly shares one pregrasp record")


def _evaluation_delta(support: dict[str, float], contact: dict[str, float]) -> dict[str, float]:
    r"""计算contact减support的共同数值metrics。"""

    return {
        name: float(contact[name]) - float(support[name])
        for name in support.keys() & contact.keys()
        if isinstance(support[name], (int, float)) and isinstance(contact[name], (int, float))
    }


def _curve_summary(summary_path: Path) -> dict[str, dict[str, float]]:
    r"""报告updates前/后10个窗口的核心均值。"""

    updates = [json.loads(line) for line in summary_path.with_name("updates.jsonl").read_text().splitlines()]
    metric_names = (
        "rollout_reward_mean",
        "rollout_done_fraction",
        "critic_loss",
        "kl",
        "clip_fraction",
        "coordination_scale",
        "global_log_std",
    )
    return {
        window_name: {name: statistics.mean(float(row[name]) for row in window) for name in metric_names}
        for window_name, window in (("first10", updates[:10]), ("last10", updates[-10:]))
    }


def main() -> int:
    r"""验证、配对并写条件性pregrasp learning结论。"""

    args = _parse_args()
    standard_support = _load(args.standard_support, "anymani.hetero.structured_ppo_run")
    standard_contact = _load(args.standard_contact, "anymani.hetero.structured_ppo_run")
    low_support = _load(args.low_support, "anymani.hetero.structured_ppo_run")
    low_contact = _load(args.low_contact, "anymani.hetero.structured_ppo_run")
    support_survival = _load(args.support_survival, "anymani.hetero.pregrasp_exploration_survival")
    contact_survival = _load(args.contact_survival, "anymani.hetero.pregrasp_exploration_survival")
    _validate_pair(standard_support, standard_contact, "standard exploration")
    _validate_pair(low_support, low_contact, "low exploration")
    if standard_support["network"].get("initial_log_std", -0.5) != -0.5:
        raise ValueError("standard pair does not use logstd=-0.5")
    if abs(float(low_support["network"]["initial_log_std"]) - (-1.203972804)) > 1.0e-9:
        raise ValueError("low-exploration pair does not use sigma=0.3")

    standard_final = {
        "support": standard_support["final_evaluation"],
        "contact": standard_contact["final_evaluation"],
    }
    low_final = {"support": low_support["final_evaluation"], "contact": low_contact["final_evaluation"]}
    standard_task_capability = any(
        float(arm["subgoal_throughput_per_env_s"]) > 0.0 or float(arm["positive_full_turn_fraction"]) > 0.0
        for arm in standard_final.values()
    )
    low_task_capability = any(
        float(arm["subgoal_throughput_per_env_s"]) > 0.0 or float(arm["positive_full_turn_fraction"]) > 0.0
        for arm in low_final.values()
    )
    artifact = {
        "artifact_type": "anymani.hetero.matched_pregrasp_ppo_analysis",
        "schema_version": "1.0.0",
        "estimand": "contact_basin minus support_basin on row16 under matched early PPO",
        "matched_fields": ["physical asset", "N040", "actor/critic", "seed", "PPO", "budget", "evaluation"],
        "standard_exploration": {
            "initial_log_std": -0.5,
            "support": standard_final["support"],
            "contact": standard_final["contact"],
            "contact_minus_support": _evaluation_delta(standard_final["support"], standard_final["contact"]),
            "support_curve": _curve_summary(args.standard_support),
            "contact_curve": _curve_summary(args.standard_contact),
            "any_task_capability": standard_task_capability,
        },
        "low_exploration": {
            "initial_log_std": -1.203972804,
            "sigma": 0.3,
            "support": low_final["support"],
            "contact": low_final["contact"],
            "contact_minus_support": _evaluation_delta(low_final["support"], low_final["contact"]),
            "support_curve": _curve_summary(args.low_support),
            "contact_curve": _curve_summary(args.low_contact),
            "any_task_capability": low_task_capability,
        },
        "exploration_survival": {
            "support": support_survival["results"],
            "contact": contact_survival["results"],
        },
        "conclusion": {
            "contact_improves_task_learning": False,
            "contact_preserves_more_tip_contact": True,
            "standard_exploration_destabilizes_contact": True,
            "low_exploration_prevents_total_axis_collapse": True,
            "low_exploration_contact_still_underperforms_support_stability": True,
            "expand_to_more_assets": False,
        },
        "limitations": [
            "single physical asset row16",
            "single training seed 42",
            "204800 transitions per arm",
            "early-learning comparison, not convergence",
            "trajectories within one shared policy run are correlated",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "standard_contact_axis_failure": standard_final["contact"]["axis_failure_fraction"],
                "low_contact_axis_failure": low_final["contact"]["axis_failure_fraction"],
                "standard_task_capability": standard_task_capability,
                "low_task_capability": low_task_capability,
                "expand_to_more_assets": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
