r"""Geometry SSL ablation evidence 的只读聚合与配对 bootstrap。

最终 validation ablation 文件保存每个 ``(asset_id,q_index)`` 的 density、$\kappa$ 与 derived-$g$
raw MSE。本模块不重新执行模型、不读取 teacher、不改变训练 metric 的数学定义；它只把同一
``asset_id`` 下的 q 样本先做等权平均，再以 asset 为 cluster 重采样，避免 64 个 q 把 4 个
held-out morphology 当成 256 个独立 morphology。置信区间使用两级 paired bootstrap：先重采样
asset cluster，再在每个被抽中的 asset 内重采样配对 q。

对于 ablation $a$ 与完整模型 $f$，定义误差增加：

$$
\Delta_{a,m}=\operatorname{MSE}_{a,m}-\operatorname{MSE}_{f,m},
$$

其中 $m\in\{\rho,\kappa,g\}$。$\Delta>0$ 表示该 ablation 误差更大，因而支持完整
morphology latent 对该 metric 的必要性；bootstrap 只对存在完整模型与 ablation 配对值的
asset cluster 重采样。
"""

from __future__ import annotations

import argparse  # 提供显式的 run-artifact analysis CLI
from pathlib import Path  # 输入 validation_ablations.yaml 与输出分析 YAML
from typing import Any  # YAML mapping 的递归基础类型

import numpy as np  # cluster bootstrap 与 asset-balanced 聚合
import yaml  # evidence schema 是人类可读 YAML

_METRICS = ("density", "kappa", "derived_field")  # held-out morphology 三个 raw MSE 轴


def analyze_geometry_ssl_ablation_file(
    input_path: Path,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    r"""读取固定 ablation evidence，返回 asset-balanced 与 cluster-bootstrap 分析结果。

    Args:
        input_path (Path): ``validation_ablations.yaml``，必须声明 `(asset_id,q_index)` pairing key。
        bootstrap_samples (int): cluster bootstrap 重采样次数；正式 pilot 默认 2000。
        seed (int): bootstrap 独立随机数种子，与训练 seed 分开记录。

    Returns:
        dict[str, Any]: 可直接写入 ``validation_ablation_analysis.yaml`` 的基础 mapping。

    Raises:
        ValueError: evidence pairing、metric、样本数或 bootstrap 配置不合法。
    """

    evidence = yaml.safe_load(input_path.read_text(encoding="utf-8"))  # 只读 frozen validation artifact
    if not isinstance(evidence, dict):  # 顶层 schema 必须是 mapping
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
    r"""直接分析 Method 返回的配对 ablation evidence，不要求中间 YAML 文件。"""

    if evidence.get("pairing_key") != ["asset_id", "q_index"]:  # 配对身份不能由路径猜测
        raise ValueError("ablation evidence must declare pairing_key=['asset_id','q_index']")
    raw_ablations = evidence.get("ablations")  # 预注册 ablation namespace
    records = evidence.get("records")  # 每个 q 的可配对 raw metrics
    if not isinstance(raw_ablations, (tuple, list)) or not isinstance(records, list) or not records:
        raise ValueError("ablation evidence requires non-empty ablation names and records")
    ablations = tuple(str(name) for name in raw_ablations)  # 稳定输出顺序
    if "full" not in ablations:  # 差值参考必须是完整 frozen model
        raise ValueError("ablation evidence must contain the full reference")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    # 按 `(asset_id,q_index)` 读取并拒绝重复 key，防止同一 q 被静默重复计权。
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

    # 每个 asset 内先等权平均 q，再把 asset 等权平均；这对应 morphology-level evidence。
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

    # 对每个 ablation/metric 做 `ablation - full` cluster bootstrap；None 保持缺测，不当作 0。
    rng = np.random.default_rng(seed)  # 独立、可复现且不消费训练/runtime RNG
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
    r"""生成固定 validation ablation 的 YAML 分析 artifact。"""

    analysis = analyze_geometry_ssl_ablation_file(
        input_path,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 只创建显式 analysis 目录
    output_path.write_text(yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8")  # 可审计基础类型


def _validate_sample_metrics(
    metrics: dict[str, Any],
    ablations: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """验证单 q 的 ablation/metric 数值域和缺测语义。"""

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
    """返回每 asset 先 q-平均、再 asset-平均的 metric。"""

    asset_means: list[float] = []  # 每项对应一个 morphology cluster 的 q-均值
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
    """返回每个 asset 内逐 q 的配对误差增加 `ablation-full`。"""

    differences: list[np.ndarray] = []
    for asset_id in asset_ids:
        paired_differences: list[float] = []  # 同一 q 上的 `ablation-full` 配对差
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
    """对 asset 与其内部 q 做两级 paired bootstrap，返回 percentile 95% CI。"""

    if not cluster_values:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "asset_count": 0, "full_better": False}
    cluster_means = np.asarray([values.mean() for values in cluster_values], dtype=np.float64)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)  # 每次两级重采样后的 morphology 等权均值
    for sample_index in range(bootstrap_samples):
        selected_assets = rng.integers(0, len(cluster_values), size=len(cluster_values))  # asset 有放回抽样
        selected_means = []
        for asset_index in selected_assets:
            q_values = cluster_values[int(asset_index)]  # 该 morphology 的完整配对 q 差值
            selected_q = rng.integers(0, len(q_values), size=len(q_values))  # asset 内 q 有放回抽样
            selected_means.append(float(q_values[selected_q].mean()))
        bootstrap_means[sample_index] = float(np.mean(selected_means))  # 被抽 morphology 继续等权
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    estimate = float(cluster_means.mean())  # 点估计：先 asset 内 q-平均，再跨 asset 等权
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "asset_count": len(cluster_values),
        "full_better": bool(low > 0.0),
    }


def main() -> None:
    r"""命令行读取 validation ablation YAML 并写入配对统计 artifact。"""

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
