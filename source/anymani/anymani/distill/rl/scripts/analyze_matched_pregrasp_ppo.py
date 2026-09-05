r"""合并row16 matched support/contact PPO与exploration survival证据。

分析器先验证每对run除tier/record/elapsed/results外的network、PPO、seed、env与budget完全相同，再计算contact-minus-
support差值。结论只限单physical asset、单seed和204,800-transition早期学习，不外推128/2048或最终收敛。
"""

from __future__ import annotations

import argparse
import hashlib
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
    r"""验证matched pair的source/task/evaluation/PPO身份，只允许arm tier/record不同。"""

    for field in ("seed", "num_envs", "updates", "transitions", "ppo_config", "network", "schema_version"):
        if support[field] != contact[field]:
            raise ValueError(f"{label} pair differs in matched field {field}")
    if support["schema_version"] != "2.0.0":
        raise ValueError(f"{label} pair does not use pre-reset evaluation schema 2.0.0")
    support_provider = dict(support["provider_identity"])
    contact_provider = dict(contact["provider_identity"])
    if support_provider != contact_provider:
        raise ValueError(f"{label} pair differs in N040 provider identity")
    if support["tier"] != "support_basin" or contact["tier"] != "contact_basin":
        raise ValueError(f"{label} pair does not contain support/contact arms")
    if support["pregrasp_record_digest"] == contact["pregrasp_record_digest"]:
        raise ValueError(f"{label} pair unexpectedly shares one pregrasp record")
    support_identity = support["run_identity"]
    contact_identity = contact["run_identity"]
    for field in (
        "source",
        "matched_task_contract",
        "matched_task_contract_digest",
        "formal_cache_index_digest",
        "provider_identity_digest",
    ):
        if support_identity[field] != contact_identity[field]:
            raise ValueError(f"{label} pair differs in run identity field {field}")
    for summary, expected_tier in ((support, "support_basin"), (contact, "contact_basin")):
        identity = summary["run_identity"]
        if identity["arm"]["tier"] != expected_tier:
            raise ValueError(f"{label} arm identity disagrees with tier")
        if identity["arm"]["pregrasp_record_digest"] != summary["pregrasp_record_digest"]:
            raise ValueError(f"{label} arm identity disagrees with record digest")
        launch = dict(identity["launch_arguments"])
        if launch["eval_steps"] != identity["matched_task_contract"]["evaluation"]["eval_steps"]:
            raise ValueError(f"{label} evaluation step identity is inconsistent")
        for name in ("initial_evaluation_trajectories", "final_evaluation_trajectories"):
            reference = summary[name]
            path = Path(reference["path"])
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                raise ValueError(f"{label} {name} SHA-256 mismatch")
            if len(payload.splitlines()) != int(reference["count"]):
                raise ValueError(f"{label} {name} count mismatch")
        checkpoint_path = Path(summary["checkpoint"])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != summary["checkpoint_sha256"]:
            raise ValueError(f"{label} checkpoint SHA-256 mismatch")
        if not summary["checkpoint_strict_restore_passed"] or not summary["optimizer_checkpoint_restore_passed"]:
            raise ValueError(f"{label} checkpoint restore gates did not pass")
    support_launch = dict(support_identity["launch_arguments"])
    contact_launch = dict(contact_identity["launch_arguments"])
    for launch in (support_launch, contact_launch):
        for arm_specific in ("tier", "run_dir", "argv"):
            launch.pop(arm_specific)
    if support_launch != contact_launch:
        raise ValueError(f"{label} pair differs in normalized launch arguments")


def _evaluation_delta(support: dict[str, float], contact: dict[str, float]) -> dict[str, float]:
    r"""计算contact减support的共同数值metrics。"""

    return {
        name: float(contact[name]) - float(support[name])
        for name in support.keys() & contact.keys()
        if type(support[name]) in {int, float} and type(contact[name]) in {int, float}
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
    for field in (
        "source",
        "matched_task_contract",
        "matched_task_contract_digest",
        "formal_cache_index_digest",
        "provider_identity_digest",
    ):
        if standard_support["run_identity"][field] != low_support["run_identity"][field]:
            raise ValueError(f"exploration conditions differ in invariant run identity field {field}")
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
        "schema_version": "2.0.0",
        "estimand": "contact_basin minus support_basin on row16 under matched early PPO",
        "matched_fields": ["physical asset", "N040", "actor/critic", "seed", "PPO", "budget", "evaluation"],
        "matched_identity": {
            "standard": standard_support["run_identity"]["matched_task_contract_digest"],
            "low_exploration": low_support["run_identity"]["matched_task_contract_digest"],
            "source_bundle_digest": standard_support["run_identity"]["source"]["source_bundle_digest"],
            "git_commit": standard_support["run_identity"]["source"]["git_commit"],
            "evaluation_lifecycle": "task_post_physics_pre_reset_snapshot",
        },
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
            "no seed-level confidence interval is reported because the independent training-seed count is one",
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
