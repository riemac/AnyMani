"Prepares stages from typed configs without importing one fixed recipe module, keeping argument help independent from config loading."

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
from ..presets.connectivity_presets import list_finger_connectivity_preset_names


EditablePath = str | Path
"String or Path accepted by the source-root resolver."


def _print_physics_summary(run_cfg: HandGeneratorCfg) -> None:

    physics_cfg = run_cfg.Physics
    print(f"physics_on         = {physics_cfg is not None and physics_cfg.enabled}")
    if physics_cfg is None:
        return

    density = physics_cfg.density
    print(f"physics.density.default     = {density.default}")
    print(f"physics.density.palm        = {density.palm}")
    print(f"physics.density.finger_link = {density.finger_link}")
    print(f"physics.density.fingertip   = {density.fingertip}")
    print(f"physics.density.custom_tip  = {density.custom_tip}")


def print_premade_registry_summary(run_cfg: HandGeneratorCfg) -> None:
    "Prints available connectivity recipes and effective pre-made settings."


    print("=== actual finger-level connectivity recipes ===")
    for family in ("allegro", "leap"):
        for finger_kind in ("thumb", "non_thumb"):
            recipe_names = list_finger_connectivity_preset_names(family=family, finger_kind=finger_kind)
            print(f"{family}:{finger_kind} -> {list(recipe_names)}")
    print()


    print("=== effective pre-made knobs ===")
    print(f"hand_presets       = {run_cfg.hand_presets}")
    print(f"handedness         = {run_cfg.handedness}")
    print(f"mixed              = {run_cfg.mixed}")
    print(f"missing            = {run_cfg.missing}")
    print(f"recolored          = {run_cfg.recolored}")
    print(f"artifact_level     = {run_cfg.artifact_level}")
    print(f"output_dir         = {run_cfg.output_dir}")
    print(f"max_enumerate      = {run_cfg.max_enumerate}")
    print(f"premade_parallel   = {run_cfg.premade_parallel}")
    print(f"parallel_workers   = {run_cfg.premade_parallel_workers}")
    print(f"parallel_fallback  = {run_cfg.premade_parallel_fallback}")
    print(f"connectivity_cfg   = {run_cfg.connectivity_presets}")
    _print_physics_summary(run_cfg)
    print(f"validator_on       = {run_cfg.Validate is not None}")
    if run_cfg.Validate is not None:
        print(f"pre_made.finger_count_min = {run_cfg.Validate.pre_made.finger_count_min}")
        print(
            "pre_made.require_non_thumb_with_min_revolute_dof = "
            f"{run_cfg.Validate.pre_made.require_non_thumb_with_min_revolute_dof}"
        )
        print(f"pre_made.check_palm_thumb_binding = {run_cfg.Validate.pre_made.check_palm_thumb_binding}")
    print()


def enumerate_premade_bundles(run_cfg: HandGeneratorCfg) -> list[Any]:
    'Builds premade bundles.'

    if run_cfg.mode != "made":
        raise ValueError("premade runner requires run_cfg.mode='made'")
    return list(HandGenerator(run_cfg).generate_batch())


def print_premade_result_summary(results: list[Any], run_cfg: HandGeneratorCfg, *, print_limit: int | None) -> None:
    "Prints accepted pre-made topology IDs and output bundle paths."

    topology_counter = Counter(str(result.metadata.get("topology_kind", "unknown")) for result in results)
    base_hand_counter = Counter(str(result.metadata.get("base_hand_preset", "-")) for result in results)

    print(f"generated {len(results)} bundles under {run_cfg.output_dir}")
    print(f"topology counts: {dict(topology_counter)}")
    print(f"base-hand counts: {dict(base_hand_counter)}")

    if print_limit == 0:
        return

    preview_results = results if print_limit is None else results[:print_limit]
    print("=== result preview ===")
    for index, result in enumerate(preview_results, start=1):
        topology_kind = str(result.metadata.get("topology_kind", "unknown"))
        topology_name = str(result.metadata.get("topology_name", "-"))
        connectivity_name = str(result.metadata.get("connectivity_preset", "-"))
        urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"
        print(f"[{index:04d}] {topology_kind} | {topology_name} | {connectivity_name} | {urdf_path}")
    if print_limit is not None and len(results) > len(preview_results):
        print(f"... {len(results) - len(preview_results)} more results omitted from terminal preview")


def _repo_root_from_runner_file() -> Path:

    return Path(__file__).resolve().parents[5]


def _resolve_editable_path(path_like: EditablePath) -> Path:

    repo_root = _repo_root_from_runner_file()
    raw_path = Path(path_like).expanduser()
    if raw_path.is_absolute():
        return raw_path
    workspace_path = repo_root.parent / raw_path
    if workspace_path.exists():
        return workspace_path
    return repo_root / raw_path


def resolve_source_topology_dir(source_path: EditablePath) -> Path:
    'Resolves source topology dir.'

    resolved_path = _resolve_editable_path(source_path)
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"source topology path does not exist or is not a directory: {resolved_path}")
    if not (resolved_path / "hand.yaml").is_file():
        raise FileNotFoundError(
            "Independent post-mutate now requires a topology-root sidecar; "
            f"missing {resolved_path / 'hand.yaml'}"
        )
    return resolved_path


def plan_post_mutate_run_dir(source_topology_dir: Path) -> Path:
    "Plans a collision-free run directory beneath the selected source topology."

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = source_topology_dir / timestamp
    collision_index = 2
    while run_dir.exists():
        run_dir = source_topology_dir / f"{timestamp}_{collision_index:02d}"
        collision_index += 1
    return run_dir


def prepare_post_mutate_run_cfg(
    run_cfg: HandGeneratorCfg,
    *,
    source_path: EditablePath,
) -> tuple[HandGeneratorCfg, Path, Path]:
    "Resolves one explicit source topology and derives its independent mutation run config."

    source_topology_dir = resolve_source_topology_dir(source_path)
    planned_run_dir = plan_post_mutate_run_dir(source_topology_dir)
    return (
        run_cfg.replace(
            source_topology_dir=source_topology_dir,
            output_dir=source_topology_dir.parent,
        ),
        source_topology_dir,
        planned_run_dir,
    )


def print_post_mutate_summary(
    run_cfg: HandGeneratorCfg,
    *,
    source_path: EditablePath,
    source_topology_dir: Path,
    planned_run_dir: Path,
) -> None:
    "Prints the source topology, mutator config, validation gates, and planned output root."

    print("=== independent post-mutate knobs ===")
    print(f"source_topology_path = {source_path}")
    print(f"source_topology_dir  = {source_topology_dir}")
    print(f"planned_run_dir      = {planned_run_dir}")
    print(f"source_topology_cfg  = {run_cfg.source_topology_dir}")
    print(f"n_samples            = {run_cfg.n_samples}")
    print(f"artifact_level       = {run_cfg.artifact_level}")
    print(f"recolored            = {run_cfg.recolored}")
    _print_physics_summary(run_cfg)
    print(f"validator_on         = {run_cfg.Validate is not None}")
    print(f"mutator_terms        = {[name for name, _ in run_cfg.Mutate.ordered_terms()]}")
    if run_cfg.Validate is not None:
        print(f"post_mutate.finger_count_min = {run_cfg.Validate.post_mutate.finger_count_min}")
        print(
            "post_mutate.require_non_thumb_with_min_revolute_dof = "
            f"{run_cfg.Validate.post_mutate.require_non_thumb_with_min_revolute_dof}"
        )
        print(f"post_mutate.check_finger_spacing = {run_cfg.Validate.post_mutate.check_finger_spacing}")
        print(f"post_mutate.min_finger_spacing = {run_cfg.Validate.post_mutate.min_finger_spacing}")
    print()


def enumerate_post_mutate_bundles(run_cfg: HandGeneratorCfg) -> list[HandGenerationResult]:
    'Builds post mutate bundles.'

    return list(HandGenerator(run_cfg).generate_batch())


def print_post_mutate_result_summary(results: list[HandGenerationResult], *, print_limit: int | None) -> None:
    "Prints post-mutate acceptance counts and output paths."

    print("=== independent post-mutate summary ===")
    print(f"generated variants = {len(results)}")
    if not results:
        print("(no result)")
        print()
        return

    topology_counter = Counter(str(result.metadata.get("topology_name", "-")) for result in results)
    for topology_name, count in sorted(topology_counter.items()):
        print(f"{topology_name}: {count}")
    print()

    if print_limit == 0:
        return

    preview_limit = len(results) if print_limit is None else min(len(results), print_limit)
    print("=== result preview ===")
    for index, result in enumerate(results[:preview_limit], start=1):
        sample_id = str(result.metadata.get("id", "-"))
        origin_id = str(result.metadata.get("source_origin_sample_id", "-"))
        topology_name = str(result.metadata.get("topology_name", result.metadata.get("source_topology_dir", "-")))
        term_names = ",".join(sorted(result.metadata.get("post_mutate_samples", {}).keys()))
        urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"
        print(f"[{index:03d}] {sample_id} <= {origin_id} | {topology_name} | terms={term_names} | {urdf_path}")
    if preview_limit < len(results):
        print(f"... ({len(results) - preview_limit} more results omitted)")
    print()


__all__ = [
    "enumerate_post_mutate_bundles",
    "enumerate_premade_bundles",
    "plan_post_mutate_run_dir",
    "prepare_post_mutate_run_cfg",
    "print_post_mutate_result_summary",
    "print_post_mutate_summary",
    "print_premade_registry_summary",
    "print_premade_result_summary",
    "resolve_source_topology_dir",
]
