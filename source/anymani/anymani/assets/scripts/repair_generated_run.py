r"""审计并修补历史 pre-made generated run 中的 OBJ-only 半成品。

历史生成链要求 procedural `cs` fingertip 在 physics closure 与 validator 前物化。
旧实现没有在 validator rejection 后回滚新写 OBJ，因此 generated 树中同时出现：

- 完整 bundle：同一 topology 根持有 `hand.urdf`、`hand.yaml` 与 `meshes/`；
- 拒绝半成品：只持有 `meshes/*.obj`，不是可训练或可预览资产。

本工具默认只做 dry-run。`--apply` 必须同时给出人工核对过的完整数和半成品数；
任何歧义目录、summary 统计不闭合或预期计数不一致都会 fail closed，不执行删除。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_PALM_FAMILIES: tuple[str, ...] = ("allegro", "leap")
_THUMB_TOKEN_PATTERN = re.compile(r"^(?:left|right)_(allegro|leap)_t\d+(?:_|$)")
_NON_THUMB_DOF_PATTERN = re.compile(r"(?:^|_)([imrl])(\d+)(?=_|$)")


class RepairError(RuntimeError):
    r"""表示一次修补请求未满足安全前提，磁盘内容应保持不变。"""


@dataclass(frozen=True)
class GeneratedRunAudit:
    r"""一次历史 run 的确定性目录分类与拒绝原因重建结果。

    Attributes:
        complete_bundles: 同时具有 `hand.urdf` 与 `hand.yaml` 的 topology 根。
        incomplete_bundles: 只具有非空 `meshes/*.obj` 的拒绝候选根。
        ambiguous_bundles: 不满足上述任一严格目录形状的候选根。
        rejected_by_reason: 按稳定规则代码集合统计的历史拒绝原因。
    """

    complete_bundles: tuple[Path, ...]
    incomplete_bundles: tuple[Path, ...]
    ambiguous_bundles: tuple[Path, ...]
    rejected_by_reason: dict[str, int]


def _build_parser() -> argparse.ArgumentParser:
    r"""构造历史 run 修补 CLI；默认行为必须保持只读。"""

    parser = argparse.ArgumentParser(
        description="Audit and optionally remove OBJ-only rejected candidates from one generated pre-made run."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Generated pre-made run root containing summary.yaml.")
    parser.add_argument("--apply", action="store_true", help="Apply deletion and append maintenance metadata.")
    parser.add_argument(
        "--expect-complete",
        type=int,
        default=None,
        help="Human-verified complete bundle count; required with --apply.",
    )
    parser.add_argument(
        "--expect-incomplete",
        type=int,
        default=None,
        help="Human-verified OBJ-only candidate count; required with --apply.",
    )
    return parser


def _load_summary(run_root: Path) -> tuple[Path, dict[str, Any]]:
    r"""读取并校验修补所需的最小 summary contract。"""

    summary_path = run_root / "summary.yaml"
    if not run_root.is_dir():
        raise RepairError(f"run root does not exist or is not a directory: {run_root}")
    if not summary_path.is_file():
        raise RepairError(f"run root is missing summary.yaml: {run_root}")

    raw_summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw_summary, dict):
        raise RepairError("summary.yaml must contain a mapping document")

    run_cfg = raw_summary.get("run")
    stats = raw_summary.get("stats")
    if not isinstance(run_cfg, dict) or not isinstance(stats, dict):
        raise RepairError("summary.yaml must contain mapping-valued run and stats sections")
    if run_cfg.get("mode") != "made" or run_cfg.get("artifact_level") != "bundle":
        raise RepairError("repair only supports mode='made', artifact_level='bundle' generated runs")
    return summary_path, raw_summary


def _candidate_bundle_roots(run_root: Path) -> tuple[Path, ...]:
    r"""收集 pre-made 固定深度上出现 bundle 文件或 mesh 根的 topology 目录。

    pre-made 目录只有两种合法深度：

    - single/missing：`<run>/<group>/<topology>`；
    - mixed：`<run>/mixed/<composition>/<topology>`。

    一个成功 topology 根下还可能嵌套后续 post-mutate 时间戳 run。那些目录属于另一轮
    summary 与生命周期，不能被本工具计入当前 pre-made 的 complete/incomplete 统计。
    """

    candidates: set[Path] = set()
    for filename in ("hand.urdf", "hand.yaml"):
        candidates.update(path.parent for path in run_root.rglob(filename) if path.is_file())
    candidates.update(path.parent for path in run_root.rglob("meshes") if path.is_dir())
    premade_candidates = [
        path
        for path in candidates
        if _is_premade_topology_root(path, run_root=run_root)
    ]
    return tuple(sorted(premade_candidates, key=lambda path: path.relative_to(run_root).as_posix()))


def _is_premade_topology_root(candidate: Path, *, run_root: Path) -> bool:
    r"""按 run-relative 深度识别 pre-made topology 根，排除嵌套 mutate 子树。"""

    relative_parts = candidate.relative_to(run_root).parts
    if not relative_parts:
        return False
    if relative_parts[0] == "mixed":
        return len(relative_parts) == 3
    return len(relative_parts) == 2


def _is_strict_mesh_only_candidate(candidate: Path) -> bool:
    r"""判断候选是否严格等于“一个非空、只含 OBJ 的 meshes 目录”。"""

    mesh_root = candidate / "meshes"
    if candidate.is_symlink() or not mesh_root.is_dir() or mesh_root.is_symlink():
        return False

    # sample 根若还含 tree、sidecar、临时文件或其它目录，就不能当作纯拒绝半成品删除。
    if {entry.name for entry in candidate.iterdir()} != {"meshes"}:
        return False

    mesh_files = [path for path in mesh_root.rglob("*") if path.is_file()]
    return bool(mesh_files) and all(path.suffix.lower() == ".obj" for path in mesh_files)


def _non_thumb_threshold(summary: dict[str, Any]) -> int:
    r"""从历史配置读取 non-thumb 最小 revolute DOF 阈值。"""

    try:
        raw_threshold = summary["config"]["Validate"]["pre_made"]["require_non_thumb_with_min_revolute_dof"]
    except (KeyError, TypeError) as exc:
        raise RepairError("summary config is missing pre-made non-thumb revolute DOF threshold") from exc
    if raw_threshold is None:
        raise RepairError("cannot reconstruct low-DOF rejection when the historical threshold is disabled")
    return int(raw_threshold)


def _palm_thumb_family_mismatch(candidate: Path, *, run_root: Path) -> bool:
    r"""由 mixed 目录和 leaf identity 重建 palm-thumb family 是否错配。"""

    relative_parts = candidate.relative_to(run_root).parts
    if len(relative_parts) < 3 or relative_parts[0] != "mixed":
        return False

    topology_group = relative_parts[1]
    palm_family = next((family for family in _PALM_FAMILIES if topology_group.startswith(f"{family}_")), None)
    thumb_match = _THUMB_TOKEN_PATTERN.match(candidate.name)
    if palm_family is None or thumb_match is None:
        raise RepairError(f"cannot reconstruct mixed palm/thumb family from {candidate.relative_to(run_root)}")
    return thumb_match.group(1) != palm_family


def _all_non_thumb_dofs_below_threshold(candidate: Path, *, threshold: int) -> bool:
    r"""由 stable topology leaf 名重建 surviving non-thumb revolute DOF。"""

    non_thumb_dofs = [int(match.group(2)) for match in _NON_THUMB_DOF_PATTERN.finditer(candidate.name)]
    if not non_thumb_dofs:
        raise RepairError(f"cannot reconstruct non-thumb DOF values from topology leaf {candidate.name!r}")
    return not any(dof >= threshold for dof in non_thumb_dofs)


def _rejection_reason_key(candidate: Path, *, run_root: Path, threshold: int) -> str:
    r"""把历史目录 identity 映射成排序、去重后的 validator 规则组合键。"""

    error_codes: list[str] = []
    if _all_non_thumb_dofs_below_threshold(candidate, threshold=threshold):
        error_codes.append("hand.non_thumb_revolute_dof_below_min")
    if _palm_thumb_family_mismatch(candidate, run_root=run_root):
        error_codes.append("hand.palm_thumb_family_mismatch")
    return "+".join(sorted(set(error_codes))) if error_codes else "unclassified"


def _audit_generated_run(run_root: Path, summary: dict[str, Any]) -> GeneratedRunAudit:
    r"""严格分类候选目录，并从 stable identity 重建每个拒绝样本的原因集合。"""

    complete: list[Path] = []
    incomplete: list[Path] = []
    ambiguous: list[Path] = []
    for candidate in _candidate_bundle_roots(run_root):
        has_urdf = (candidate / "hand.urdf").is_file()
        has_yaml = (candidate / "hand.yaml").is_file()
        if has_urdf and has_yaml:
            complete.append(candidate)
        elif not has_urdf and not has_yaml and _is_strict_mesh_only_candidate(candidate):
            incomplete.append(candidate)
        else:
            ambiguous.append(candidate)

    threshold = _non_thumb_threshold(summary)
    reason_counts = Counter(
        _rejection_reason_key(candidate, run_root=run_root, threshold=threshold)
        for candidate in incomplete
    )
    return GeneratedRunAudit(
        complete_bundles=tuple(complete),
        incomplete_bundles=tuple(incomplete),
        ambiguous_bundles=tuple(ambiguous),
        rejected_by_reason=dict(sorted(reason_counts.items())),
    )


def _validate_audit_against_summary(audit: GeneratedRunAudit, summary: dict[str, Any]) -> None:
    r"""确认磁盘分类与历史 attempted/succeeded/rejected 三元组严格闭合。"""

    if audit.ambiguous_bundles:
        raise RepairError(
            f"refusing repair because ambiguous={len(audit.ambiguous_bundles)}; "
            f"first={audit.ambiguous_bundles[0]}"
        )

    stats = summary["stats"]
    complete_count = len(audit.complete_bundles)
    incomplete_count = len(audit.incomplete_bundles)
    attempted = int(stats.get("attempted", -1))
    succeeded = int(stats.get("succeeded", -1))
    rejected = int(stats.get("rejected", -1))
    if succeeded != complete_count:
        raise RepairError(f"summary succeeded={succeeded}, audited complete={complete_count}")
    if rejected != incomplete_count:
        raise RepairError(f"summary rejected={rejected}, audited incomplete={incomplete_count}")
    if attempted != complete_count + incomplete_count:
        raise RepairError(
            f"summary attempted={attempted}, audited complete+incomplete={complete_count + incomplete_count}"
        )
    if sum(audit.rejected_by_reason.values()) != incomplete_count:
        raise RepairError("reconstructed rejection reason counts do not sum to audited incomplete count")

    maintenance = summary.get("maintenance")
    if maintenance is not None and not isinstance(maintenance, list):
        raise RepairError("summary maintenance field must be a list when already present")


def _validate_apply_expectations(args: argparse.Namespace, audit: GeneratedRunAudit) -> None:
    r"""要求 destructive apply 显式复述 dry-run 得到的两个关键计数。"""

    if args.expect_complete is None or args.expect_incomplete is None:
        raise RepairError("--apply requires both --expect-complete and --expect-incomplete")
    if args.expect_complete != len(audit.complete_bundles):
        raise RepairError(
            f"expected complete={args.expect_complete}, audited complete={len(audit.complete_bundles)}"
        )
    if args.expect_incomplete != len(audit.incomplete_bundles):
        raise RepairError(
            f"expected incomplete={args.expect_incomplete}, audited incomplete={len(audit.incomplete_bundles)}"
        )


def _prune_empty_parents(start_dir: Path, *, run_root: Path) -> int:
    r"""删除半成品后向上清理空 group，但绝不删除 run root。"""

    removed_count = 0
    current = start_dir
    while current != run_root:
        current.relative_to(run_root)  # 越界时由 pathlib 直接抛错，删除不会继续
        try:
            current.rmdir()
        except OSError:
            break
        removed_count += 1
        current = current.parent
    return removed_count


def _write_summary_atomically(summary_path: Path, summary: dict[str, Any]) -> None:
    r"""在同一文件系统内原子替换 summary，避免半写 YAML。"""

    temporary_path = summary_path.with_name(f".{summary_path.name}.repair.tmp")
    temporary_path.write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary_path.replace(summary_path)


def _apply_repair(
    *,
    run_root: Path,
    summary_path: Path,
    summary: dict[str, Any],
    audit: GeneratedRunAudit,
) -> tuple[int, int]:
    r"""删除已审计的纯 mesh 候选，并把历史原因与维护事件写回 summary。"""

    removed_empty_parent_dirs = 0
    for candidate in audit.incomplete_bundles:
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(run_root)  # 删除前再次锁住 run 边界
        parent = resolved_candidate.parent
        shutil.rmtree(resolved_candidate)  # 候选已被严格证明只含 `meshes/*.obj`
        removed_empty_parent_dirs += _prune_empty_parents(parent, run_root=run_root)

    # 原始 attempted/succeeded/rejected 保持不变；新增字段只补充当时缺失的拒绝原因证据。
    summary["stats"]["rejected_by_reason"] = dict(audit.rejected_by_reason)
    maintenance = summary.setdefault("maintenance", [])
    maintenance.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "action": "prune_incomplete_rejected_artifacts",
            "tool": "repair_generated_run.py",
            "preserved_original_stats": True,
            "complete_bundles_preserved": len(audit.complete_bundles),
            "removed_incomplete_bundles": len(audit.incomplete_bundles),
            "removed_empty_parent_dirs": removed_empty_parent_dirs,
            "rejected_by_reason": dict(audit.rejected_by_reason),
        }
    )
    _write_summary_atomically(summary_path, summary)
    return len(audit.incomplete_bundles), removed_empty_parent_dirs


def _print_audit(*, mode: str, run_root: Path, audit: GeneratedRunAudit) -> None:
    r"""输出适合人工复述到 `--expect-*` 的紧凑审计摘要。"""

    print(f"mode={mode} run_root={run_root}")
    print(
        f"complete={len(audit.complete_bundles)} "
        f"incomplete={len(audit.incomplete_bundles)} "
        f"ambiguous={len(audit.ambiguous_bundles)}"
    )
    for reason_key, count in audit.rejected_by_reason.items():
        print(f"rejected_by_reason[{reason_key}]={count}")


def main(argv: list[str] | None = None) -> int:
    r"""执行 dry-run 或带双重计数护栏的历史 run 修补。"""

    args = _build_parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    try:
        summary_path, summary = _load_summary(run_root)
        audit = _audit_generated_run(run_root, summary)
        _print_audit(mode="apply" if args.apply else "dry-run", run_root=run_root, audit=audit)
        _validate_audit_against_summary(audit, summary)
        if not args.apply:
            return 0

        _validate_apply_expectations(args, audit)
        removed_bundles, removed_parents = _apply_repair(
            run_root=run_root,
            summary_path=summary_path,
            summary=summary,
            audit=audit,
        )
        print(f"removed_incomplete={removed_bundles} removed_empty_parent_dirs={removed_parents}")
        return 0
    except RepairError as exc:
        print(f"repair refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
