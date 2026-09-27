"""Check the paper criterion rather than the evaluator's broader diagnostic gate."""

import importlib.util
import json
from pathlib import Path


def test_paper_threshold_and_missing_hand_denominator(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("paper_evaluation_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "finite": True,
            "replica_count": 16,
            "net_turns_median": 1.0,
            "directional_consistency": 0.7,
            "safe_replica_fraction": 12 / 16,
        },
        {
            "finite": True,
            "replica_count": 16,
            "net_turns_median": 2.0,
            "directional_consistency": 0.9,
            "safe_replica_fraction": 11 / 16,
        },
        {
            "finite": True,
            "replica_count": 16,
            "net_turns_median": 0.99,
            "directional_consistency": 0.9,
            "safe_replica_fraction": 1.0,
        },
    ]
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"executed_steps": 600, "physical_rotation": {"asset_results": rows}}))
    result = module.aggregate_paper_metrics(
        [
            {
                "evaluation": str(evaluation),
                "cohort": "leap_right_variant",
                "nominal_asset_count": 4,
                "ready_asset_count": 3,
                "initialization_failure_count": 1,
            }
        ]
    )
    overall = result["groups"]["overall"]
    assert overall["successful_hands"] == 1
    assert overall["success_rate"] == 0.25
    assert overall["evaluated_hands"] == 3
    assert result["groups"]["leap"] == overall
