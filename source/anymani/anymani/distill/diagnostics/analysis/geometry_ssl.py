'Aggregate paired geometry-SSL validation evidence by asset and query.'

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_METRICS = ("density", "kappa", "derived_field")


def analyze_geometry_ssl_ablation_file(
    input_path: Path,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    'Analyze geometry SSL ablation file.'

    evidence = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("ablation evidence must be a YAML mapping")
    return analyze_geometry_ssl_ablation_evidence(
        evidence,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        input_label=str(input_path),
    )


def analyze_geometry_ssl_ablation_evidence(
    evidence: dict[str, Any],
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20260813,
    input_label: str = "in_memory_method_report",
) -> dict[str, Any]:
    'Analyze geometry SSL ablation evidence.'

    if evidence.get("pairing_key") != ["asset_id", "q_index"]:
        raise ValueError("ablation evidence must declare pairing_key=['asset_id','q_index']")
    raw_ablations = evidence.get("ablations")
    records = evidence.get("records")
    if not isinstance(raw_ablations, (tuple, list)) or not isinstance(records, list) or not records:
        raise ValueError("ablation evidence requires non-empty ablation names and records")
    ablations = tuple(str(name) for name in raw_ablations)
    if "full" not in ablations:
        raise ValueError("ablation evidence must contain the full reference")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")


    samples: dict[tuple[str, int], dict[str, dict[str, float | None]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each ablation record must be a mapping")
        asset_id = record.get("asset_id")
        q_index = record.get("q_index")
        metrics = record.get("metrics")
        if not isinstance(asset_id, str) or not isinstance(q_index, int) or not isinstance(metrics, dict):
            raise ValueError("ablation record must contain string asset_id, integer q_index and metrics mapping")
        key = (asset_id, q_index)
        if key in samples:
            raise ValueError(f"duplicate ablation pairing key={key!r}")
        samples[key] = _validate_sample_metrics(metrics, ablations)


    asset_ids = tuple(dict.fromkeys(asset_id for asset_id, _ in samples))
    summary: dict[str, Any] = {
        "input": input_label,
        "pairing_key": ["asset_id", "q_index"],
        "bootstrap": {
            "method": "hierarchical_asset_q_paired_resample",
            "samples": bootstrap_samples,
            "seed": int(seed),
        },
        "record_count": len(samples),
        "asset_count": len(asset_ids),
        "ablations": list(ablations),
        "metrics": {},
        "paired_differences": {},
    }
    for ablation in ablations:
        summary["metrics"][ablation] = {
            metric: _asset_balanced_metric(samples, asset_ids, ablation, metric) for metric in _METRICS
        }


    rng = np.random.default_rng(seed)
    for ablation in ablations:
        if ablation == "full":
            continue
        summary["paired_differences"][ablation] = {}
        for metric in _METRICS:
            cluster_values = _asset_q_paired_differences(samples, asset_ids, ablation, metric)
            summary["paired_differences"][ablation][metric] = _bootstrap_difference(
                cluster_values,
                rng=rng,
                bootstrap_samples=bootstrap_samples,
            )
    return summary


def write_geometry_ssl_ablation_analysis(
    input_path: Path,
    output_path: Path,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20260813,
) -> None:
    'Write geometry SSL ablation analysis.'

    analysis = analyze_geometry_ssl_ablation_file(
        input_path,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8")


def _validate_sample_metrics(
    metrics: dict[str, Any],
    ablations: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    'Validate sample metrics.'

    result: dict[str, dict[str, float | None]] = {}
    for ablation in ablations:
        value = metrics.get(ablation)
        if value is None:
            result[ablation] = {metric: None for metric in _METRICS}
            continue
        if not isinstance(value, dict):
            raise ValueError(f"metrics[{ablation!r}] must be a mapping or null")
        result[ablation] = {}
        for metric in _METRICS:
            raw = value.get(metric)
            if raw is not None and (not isinstance(raw, (int, float)) or not np.isfinite(raw) or raw < 0.0):
                raise ValueError(f"metrics[{ablation!r}][{metric!r}] must be finite non-negative or null")
            result[ablation][metric] = None if raw is None else float(raw)
    return result


def _asset_balanced_metric(
    samples: dict[tuple[str, int], dict[str, dict[str, float | None]]],
    asset_ids: tuple[str, ...],
    ablation: str,
    metric: str,
) -> dict[str, float | int | None]:
    'Handle asset balanced metric.'

    asset_means: list[float] = []
    for candidate in asset_ids:
        values = [
            value
            for (asset_id, _), row in samples.items()
            if asset_id == candidate
            for value in [row[ablation][metric]]
            if value is not None
        ]
        if values:
            asset_means.append(float(np.mean(values)))
    return {
        "asset_balanced_mean": float(np.mean(asset_means)) if asset_means else None,
        "asset_count_with_metric": len(asset_means),
        "record_count_with_metric": sum(
            samples[(asset_id, q)][ablation][metric] is not None for asset_id, q in samples
        ),
    }


def _asset_q_paired_differences(
    samples: dict[tuple[str, int], dict[str, dict[str, float | None]]],
    asset_ids: tuple[str, ...],
    ablation: str,
    metric: str,
) -> tuple[np.ndarray, ...]:
    'Handle asset q paired differences.'

    differences: list[np.ndarray] = []
    for asset_id in asset_ids:
        paired_differences: list[float] = []
        for (candidate, _), row in samples.items():
            if candidate != asset_id:
                continue
            ablation_value = row[ablation][metric]
            full_value = row["full"][metric]
            if ablation_value is not None and full_value is not None:
                paired_differences.append(ablation_value - full_value)
        if paired_differences:
            differences.append(np.asarray(paired_differences, dtype=np.float64))
    return tuple(differences)


def _bootstrap_difference(
    cluster_values: tuple[np.ndarray, ...],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | int | bool | None]:
    'Handle bootstrap difference.'

    if not cluster_values:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "asset_count": 0, "full_better": False}
    cluster_means = np.asarray([values.mean() for values in cluster_values], dtype=np.float64)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        selected_assets = rng.integers(0, len(cluster_values), size=len(cluster_values))
        selected_means = []
        for asset_index in selected_assets:
            q_values = cluster_values[int(asset_index)]
            selected_q = rng.integers(0, len(q_values), size=len(q_values))
            selected_means.append(float(q_values[selected_q].mean()))
        bootstrap_means[sample_index] = float(np.mean(selected_means))
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    estimate = float(cluster_means.mean())
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "asset_count": len(cluster_values),
        "full_better": bool(low > 0.0),
    }


def main() -> None:
    'Handle main.'

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="validation_ablations.yaml")
    parser.add_argument("output", type=Path, help="validation_ablation_analysis.yaml")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    write_geometry_ssl_ablation_analysis(
        args.input,
        args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "analyze_geometry_ssl_ablation_evidence",
    "analyze_geometry_ssl_ablation_file",
    "write_geometry_ssl_ablation_analysis",
]
