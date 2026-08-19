r"""分层 hand dataset 的 plan、build、resume 与 manifest 派生 CLI。

正式工作流先运行 ``plan`` 冻结 mother cohorts、variant 数和每母 mutation seed；
``build`` 只能消费内容一致的 lock，避免用户修改模板后继续向旧 cohort 写资产。
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..generator.dataset_build import (
    build_dataset_from_lock,
    build_dataset_selection_plan,
    derive_ppo_manifest_from_lock,
    load_dataset_build_template,
    recover_dataset_build,
    write_selection_lock,
)
from ..generator.hand_generator import HandGeneratorCfg
from ..generator.runtime.recipe_loader import RecipeLoader


def _build_parser() -> argparse.ArgumentParser:
    r"""构造显式 ``plan`` 优先的 dataset CLI。"""

    parser = argparse.ArgumentParser(description="Plan and build stratified AnyMani hand asset datasets.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser("plan", help="Freeze mother cohorts and mutation task seeds.")
    plan.add_argument("--template", required=True, help="Dataset build template YAML path.")
    plan.add_argument(
        "--config-module",
        required=True,
        help="Explicit Python module exposing the POST_MUTATE_CFG used to freeze this dataset plan.",
    )
    plan.add_argument(
        "--lock-path",
        default=None,
        help="Optional selection lock output; defaults to <template-dir>/selection.lock.yaml.",
    )
    build = subcommands.add_parser("build", help="Generate locked variant sets and publish SSL/PPO manifests.")
    build.add_argument("--template", required=True, help="Dataset build template YAML path.")
    build.add_argument(
        "--config-module",
        required=True,
        help="Explicit Python module exposing the POST_MUTATE_CFG required by the selection lock.",
    )
    build.add_argument(
        "--lock-path",
        default=None,
        help="Optional selection lock path; defaults to <template-dir>/selection.lock.yaml.",
    )
    build.add_argument("--workers", type=int, default=None, help="Mother-level post-mutate process workers.")
    build.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed tasks whose summary and fingerprints remain valid.",
    )
    derive = subcommands.add_parser("derive-ppo", help="Derive a PPO manifest without generating new assets.")
    derive.add_argument("--template", required=True, help="Dataset build template YAML path.")
    derive.add_argument("--lock-path", default=None, help="Selection lock path.")
    derive.add_argument("--state-path", default=None, help="Completed build state path.")
    derive.add_argument("--output", default=None, help="Derived PPO YAML output path.")
    derive.add_argument("--mother-count", type=int, default=128, help="PPO train mother count.")
    derive.add_argument("--selection-seed", type=int, default=None, help="Override derived PPO selection seed.")
    derive.add_argument(
        "--reuse-ssl-holdouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require both SSL seen-mother suites to remain in PPO train.",
    )
    recover = subcommands.add_parser("recover", help="Audit, adopt, or roll back an interrupted dataset build.")
    recover.add_argument("--template", required=True, help="Dataset build template YAML path.")
    recover.add_argument("--lock-path", default=None, help="Selection lock path.")
    recover.add_argument("--state-path", default=None, help="Interrupted build state path.")
    recover.add_argument("--strategy", choices=("adopt", "rollback"), required=True)
    recover.add_argument(
        "--apply",
        action="store_true",
        help="Execute the audited recovery action; without this flag only a dry-run report is written.",
    )
    return parser


def _load_post_mutate_cfg(module_name: str) -> tuple[HandGeneratorCfg, dict[str, Any]]:
    r"""读取正式 post-mutate cfg，并生成可写入 lock 的完整快照。"""

    module = importlib.import_module(module_name)
    cfg = getattr(module, "POST_MUTATE_CFG", None)
    if not isinstance(cfg, HandGeneratorCfg) or cfg.mode != "mutate":
        raise TypeError(f"{module_name} must expose POST_MUTATE_CFG with mode='mutate'")
    return cfg, RecipeLoader.dump(cfg)


def _git_provenance() -> tuple[str, bool]:
    r"""记录当前代码 commit 与 dirty bit，不把工作树文本复制进 selection lock。"""

    repo_root = Path(__file__).resolve().parents[5]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _run_plan(*, template_path: str, config_module: str, lock_path: str | None = None) -> int:
    r"""冻结 selection lock，并打印足够人工审阅的分布摘要。"""

    template_file = Path(template_path).expanduser().resolve()
    template, template_sha256 = load_dataset_build_template(template_file)
    _cfg, config_snapshot = _load_post_mutate_cfg(config_module)
    git_commit, git_dirty = _git_provenance()
    plan = build_dataset_selection_plan(
        template,
        template_sha256=template_sha256,
        generator_config_module=config_module,
        generator_config_snapshot=config_snapshot,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    resolved_lock_path = Path(lock_path).expanduser().resolve() if lock_path else template_file.parent / "selection.lock.yaml"
    write_selection_lock(plan, resolved_lock_path)

    print(f"template_id      = {template.template_id}")
    print(f"inventory        = {plan.inventory_run_dir}")
    print(f"inventory_pairs  = {plan.quota_report['inventory']['pair_count']}")
    print(f"inventory_mothers= {plan.quota_report['inventory']['mother_count']}")
    for role, distribution in plan.quota_report["selected"].items():
        print(
            f"{role:35s} mothers={distribution['mother_count']:4d} "
            f"pairs={distribution['pair_count']:4d} shape={distribution['topology_shape']}"
        )
    print(f"planned_variants = {plan.quota_report['planned_variants']}")
    print(f"planned_assets   = {plan.quota_report['planned_assets']}")
    print(f"selection_lock   = {resolved_lock_path}")
    print(f"git_dirty        = {plan.git_dirty}")
    return 0


def _run_build(
    *,
    template_path: str,
    config_module: str,
    lock_path: str | None,
    workers: int | None,
    resume: bool,
) -> int:
    r"""消费已审阅 selection lock，并行生成后发布最终 manifests。"""

    template_file = Path(template_path).expanduser().resolve()
    template, template_sha256 = load_dataset_build_template(template_file)
    cfg, _snapshot = _load_post_mutate_cfg(config_module)
    resolved_lock = Path(lock_path).expanduser().resolve() if lock_path else template_file.parent / "selection.lock.yaml"
    report = build_dataset_from_lock(
        template,
        template_sha256=template_sha256,
        lock_path=resolved_lock,
        post_mutate_cfg=cfg,
        workers=workers,
        resume=resume,
    )
    print(f"published         = {report['published']}")
    print(f"status_counts     = {report['status_counts']}")
    print(f"build_report      = {resolved_lock.parent / 'build_report.yaml'}")
    print(f"ssl_manifest      = {resolved_lock.parent / 'ssl.yaml'}")
    if template.manifests.ppo.enabled:
        print(f"ppo_manifest      = {resolved_lock.parent / 'ppo.yaml'}")
    return 0


def _run_derive_ppo(
    *,
    template_path: str,
    lock_path: str | None,
    state_path: str | None,
    output_path: str | None,
    mother_count: int,
    selection_seed: int | None,
    reuse_ssl_holdouts: bool,
) -> int:
    r"""从 completed SSL build state 重新分层派生 PPO manifest。"""

    template_file = Path(template_path).expanduser().resolve()
    template, _template_sha = load_dataset_build_template(template_file)
    resolved_lock = Path(lock_path).expanduser().resolve() if lock_path else template_file.parent / "selection.lock.yaml"
    resolved_state = Path(state_path).expanduser().resolve() if state_path else resolved_lock.parent / ".build_state.yaml"
    lock = yaml.safe_load(resolved_lock.read_text(encoding="utf-8")) or {}
    state = yaml.safe_load(resolved_state.read_text(encoding="utf-8")) or {}
    if not isinstance(lock, dict) or not isinstance(state, dict):
        raise TypeError("selection lock and build state must be mappings")
    seed = template.seeds.selection if selection_seed is None else selection_seed
    manifest = derive_ppo_manifest_from_lock(
        template,
        lock=lock,
        state=state,
        mother_count=mother_count,
        selection_seed=seed,
        reuse_ssl_holdouts=reuse_ssl_holdouts,
    )
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_lock.parent / f"ppo_{mother_count}_seed_{seed}.yaml"
    )
    resolved_output.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"ppo_manifest      = {resolved_output}")
    print(f"train_mothers     = {mother_count}")
    print(f"selection_seed    = {seed}")
    print(f"reuse_holdouts    = {reuse_ssl_holdouts}")
    return 0


def _run_recover(
    *,
    template_path: str,
    lock_path: str | None,
    state_path: str | None,
    strategy: str,
    apply: bool,
) -> int:
    r"""审计并可选执行中断 build 的 adopt/rollback；该命令不重新加载 generator cfg。"""

    template_file = Path(template_path).expanduser().resolve()
    template, _template_sha = load_dataset_build_template(template_file)
    resolved_lock = Path(lock_path).expanduser().resolve() if lock_path else template_file.parent / "selection.lock.yaml"
    report = recover_dataset_build(
        template,
        lock_path=resolved_lock,
        state_path=state_path,
        strategy=strategy,  # type: ignore[arg-type]
        apply=apply,
    )
    counts = report["counts"]
    print(f"strategy          = {strategy}")
    print(f"dry_run           = {report['dry_run']}")
    print(f"run_roots         = {counts['run_roots']}")
    print(f"complete          = {counts['complete']}")
    print(f"partial           = {counts['partial']}")
    print(f"variant_sidecars  = {counts['variant_sidecars']}")
    print(f"recovery_report   = {resolved_lock.parent / 'recovery_report.yaml'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    r"""解析命令并执行无生成副作用的 plan 阶段。"""

    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        return _run_plan(
            template_path=args.template,
            config_module=args.config_module,
            lock_path=args.lock_path,
        )
    if args.command == "build":
        return _run_build(
            template_path=args.template,
            config_module=args.config_module,
            lock_path=args.lock_path,
            workers=args.workers,
            resume=args.resume,
        )
    if args.command == "derive-ppo":
        return _run_derive_ppo(
            template_path=args.template,
            lock_path=args.lock_path,
            state_path=args.state_path,
            output_path=args.output,
            mother_count=args.mother_count,
            selection_seed=args.selection_seed,
            reuse_ssl_holdouts=args.reuse_ssl_holdouts,
        )
    if args.command == "recover":
        return _run_recover(
            template_path=args.template,
            lock_path=args.lock_path,
            state_path=args.state_path,
            strategy=args.strategy,
            apply=args.apply,
        )
    raise AssertionError(f"unhandled dataset command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
