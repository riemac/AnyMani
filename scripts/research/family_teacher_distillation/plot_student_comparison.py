"""从可追溯表格绘制共享学生学习曲线与配对覆盖比较。

学习曲线只使用三个条件共同完成的种子。覆盖图要求全部九组与八个人口
评价完整；不把缺失实验补成零。误差条是训练种子间样本标准差。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"n040": "Ours", "no_z": "No-Z", "fk": "FK"}
COLORS = {"n040": "#0072B2", "no_z": "#777777", "fk": "#D55E00"}


def save(figure, path: Path) -> None:
    """同时保存可编辑矢量与独立预览，所有导出使用同一画布。"""
    for extension in ("pdf", "svg", "png"):
        figure.savefig(path.with_suffix("." + extension), dpi=200, bbox_inches="tight")
    plt.close(figure)


def style_axis(axis) -> None:
    """保留坐标与单位的清晰层次；网格置于数据之后。"""
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)


def main() -> None:
    """先读正式种子交集，再决定可绘制的证据范围。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--learning-only", action="store_true")
    args = parser.parse_args()
    case = args.case.resolve()
    output = case / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    reports = {}
    sources = {}
    for variant in LABELS:
        for seed in (42, 43, 44):
            path = case / f"formal-v1-{variant}-s{seed}/model/training-report.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["status"] in {"completed", "time_limit"}:
                    reports[(variant, seed)] = report
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    seeds = [seed for seed in (42, 43, 44) if all((variant, seed) in reports for variant in LABELS)]
    if not seeds:
        raise ValueError("no completed paired seed across all three conditions")
    curve_data = []
    figure, axis = plt.subplots(figsize=(3.45, 2.35), layout="constrained")
    for variant in LABELS:
        arrays, axes = [], []
        for seed in seeds:
            epochs = reports[(variant, seed)]["epochs"]
            x = np.asarray([row["update"] for row in epochs])
            y = np.asarray([row["validation"]["balanced_mean_action_mse"] for row in epochs])
            axes.append(x)
            arrays.append(y)
            curve_data.extend({"variant": variant, "seed": seed, "optimizer_updates": int(a), "balanced_validation_action_mse": float(b)} for a, b in zip(x, y))
        if any(not np.array_equal(x, axes[0]) for x in axes[1:]):
            raise ValueError("paired curves require matched recorded update axes")
        values = np.stack(arrays)
        mean = values.mean(axis=0)
        axis.plot(axes[0], mean, label=LABELS[variant], color=COLORS[variant], linewidth=1.6)
        if len(seeds) > 1:
            sd = values.std(axis=0, ddof=1)
            axis.fill_between(axes[0], np.maximum(mean-sd, 0), mean+sd, color=COLORS[variant], alpha=0.15)
    axis.set(xlabel="Optimizer updates", ylabel="Balanced validation action MSE", ylim=(0, None))
    axis.set_title(f"Matched offline training (n={len(seeds)} seed{'s' if len(seeds)>1 else ''})")
    axis.legend(frameon=False)
    style_axis(axis)
    save(figure, output / "learning-curves")
    with (output / "learning-curves-data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_data[0]))
        writer.writeheader()
        writer.writerows(curve_data)
    manifest = {"common_seeds": seeds, "source_reports": sources,
                "uncertainty": "sample standard deviation across training seeds; none for n=1",
                "wall_time_not_compared": True, "coverage_status": "not-requested"}
    if not args.learning_only:
        completeness = json.loads((case / "comparison/completeness.json").read_text())
        if completeness["status"] != "complete" or seeds != [42, 43, 44]:
            raise ValueError("coverage figures require the complete nine-model comparison")
        with (case / "comparison/per-seed.csv").open() as stream:
            trials = list(csv.DictReader(stream))
        panels = [
            ("LEAP training", ["leap_training"], 128, 96),
            ("LEAP unseen variants", ["leap_right_variant"], 32, None),
            ("LEAP unseen base designs", ["leap_right_mother"], 32, None),
            ("Allegro training", ["allegro_training"], 128, 89),
            ("Allegro unseen variants", ["allegro_right_variant"], 32, None),
            ("Allegro unseen base designs", ["allegro_right_mother"], 32, None),
        ]
        # 两种留出性质分别展示，避免合并覆盖率掩盖几何变体与新基型的差异。
        figure, grid = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey=True, layout="constrained")
        for axis, (title, roles, denominator, teacher) in zip(grid.flat, panels):
            for index, variant in enumerate(LABELS):
                values = []
                for seed in seeds:
                    rows = [row for row in trials if row["variant"]==variant and int(row["seed"])==seed and row["population"] in roles]
                    if len(rows) != len(roles) or sum(int(row["nominal_assets"]) for row in rows) != denominator:
                        raise ValueError("plot population denominator changed")
                    values.append(100*sum(int(row["passed_assets"]) for row in rows)/denominator)
                axis.bar(index, np.mean(values), yerr=np.std(values, ddof=1), color=COLORS[variant], alpha=0.8, width=0.58, capsize=3)
                axis.scatter(index+np.asarray([-0.12, 0, 0.12]), values, color=COLORS[variant], edgecolors="white", linewidths=0.6, s=20, zorder=3)
            if teacher is not None:
                axis.axhline(100*teacher/denominator, linestyle="--", linewidth=0.8, color="#333333", label="Family teacher")
            axis.set(title=f"{title}\nN={denominator}", xticks=range(3), xticklabels=list(LABELS.values()), ylim=(0, 100), yticks=range(0, 101, 20))
            style_axis(axis)
        for axis in grid[:, 0]:
            axis.set_ylabel("Success rate (%)")
        grid[0, 0].legend(frameon=False, loc="upper left")
        save(figure, output / "coverage-comparison")
        manifest.update(coverage_status="complete", coverage_source_sha256=hashlib.sha256((case / "comparison/per-seed.csv").read_bytes()).hexdigest(),
                        initialization_failure_policy="Retained in original denominator; see per-asset outcome and admission records.",
                        success_gate={"seconds": 30, "replicas": 16, "net_turns_median_min": 1.0,
                                      "direction_min": 0.7, "safe_replicas_min": 12},
                        coverage_marks="Bars: mean; whiskers: sample standard deviation; dots: individual training seeds42/43/44.",
                        registry_sha256=hashlib.sha256((case / "evaluation-populations/registry.json").read_bytes()).hexdigest())
    (output / "figure-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "common_seeds": seeds, "coverage": manifest["coverage_status"]}))


if __name__ == "__main__":
    main()
