r"""统一资产生成 runner helper。

本文件集中放置原先 `quick_pre_made.py` / `quick_post_mutate.py`
里那些“不属于配置模块、也不该塞进 CLI 主入口”的中间逻辑。

它的职责刻意收敛成两类：

1. pre-made 方向：
   - registry summary 打印；
   - 结果枚举与 preview；
2. post-mutate 方向：
   - 来源 topology 路径解析；
   - mutate run 时间戳规划；
   - 结果 summary 与 preview。

# NOTE:
这里不反向依赖具体的 config 模块常量，而只接受：

- `HandGeneratorCfg`
- runner 传下来的少量路径参数

这样做的原因是减少导入链长度，避免“只想 import helper 或看 `--help`，
却因为某个配置模块 import 失败而整体不可用”。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
from ..presets.connectivity_presets import list_finger_connectivity_preset_names


EditablePath = str | Path
"""允许 helper 接收绝对路径、相对路径或 `Path` 对象。"""


def _print_physics_summary(run_cfg: HandGeneratorCfg) -> None:
    r"""打印当前生效的 physics closure 与密度配置。

    Args:
        run_cfg (HandGeneratorCfg): 已经完成 dataclass 规范化的正式运行配置。
    """

    physics_cfg = run_cfg.Physics  # generator 最高 façade 中的物理闭包配置
    print(f"physics_on         = {physics_cfg is not None and physics_cfg.enabled}")
    if physics_cfg is None:
        return

    density = physics_cfg.density  # `DensityProfileCfg`，按部位表达均匀密度假设
    print(f"physics.density.default     = {density.default}")
    print(f"physics.density.palm        = {density.palm}")
    print(f"physics.density.finger_link = {density.finger_link}")
    print(f"physics.density.fingertip   = {density.fingertip}")
    print(f"physics.density.custom_tip  = {density.custom_tip}")


def print_premade_registry_summary(run_cfg: HandGeneratorCfg) -> None:
    r"""打印 pre-made registry 与当前有效运行参数。

    Args:
        run_cfg (HandGeneratorCfg): 当前 pre-made 正式配置。
    """

    # 先打印 finger-level connectivity registry，便于核对 registry 本身的可用 recipe。
    print("=== actual finger-level connectivity recipes ===")
    for family in ("allegro", "leap"):
        for finger_kind in ("thumb", "non_thumb"):
            recipe_names = list_finger_connectivity_preset_names(family=family, finger_kind=finger_kind)  # 当前 family/kind 对应的全部合法 recipe
            print(f"{family}:{finger_kind} -> {list(recipe_names)}")
    print()

    # 再打印当前实际生效的 `HandGeneratorCfg` 关键字段，避免“以为自己改了配置，其实没生效”。
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
    r"""执行 pre-made 正式枚举，并把结果收成列表。

    Args:
        run_cfg (HandGeneratorCfg): 必须是 `mode="made"` 的正式生成配置。

    Returns:
        list[Any]: `HandGenerator.generate_batch()` 产出的结果包列表。

    Raises:
        ValueError: 当传入 cfg 不是 pre-made 模式时抛出。
    """

    if run_cfg.mode != "made":
        raise ValueError("premade runner requires run_cfg.mode='made'")
    return list(HandGenerator(run_cfg).generate_batch())  # pre-made 离散空间通常需要完整 materialize 以便打印 summary


def print_premade_result_summary(results: list[Any], run_cfg: HandGeneratorCfg, *, print_limit: int | None) -> None:
    r"""打印 pre-made 结果统计与 preview。

    Args:
        results (list[Any]): pre-made 生成结果列表。
        run_cfg (HandGeneratorCfg): 当前生效的 pre-made 配置。
        print_limit (int | None): 终端 preview 上限；`None` 表示全打印。
    """

    topology_counter = Counter(str(result.metadata.get("topology_kind", "unknown")) for result in results)  # topology family 统计
    base_hand_counter = Counter(str(result.metadata.get("base_hand_preset", "-")) for result in results)  # base hand preset 统计

    print(f"generated {len(results)} bundles under {run_cfg.output_dir}")
    print(f"topology counts: {dict(topology_counter)}")
    print(f"base-hand counts: {dict(base_hand_counter)}")

    if print_limit == 0:
        return

    preview_results = results if print_limit is None else results[:print_limit]  # preview 不影响真实结果列表
    print("=== result preview ===")
    for index, result in enumerate(preview_results, start=1):
        topology_kind = str(result.metadata.get("topology_kind", "unknown"))  # 例如 single_family / mixed_family
        topology_name = str(result.metadata.get("topology_name", "-"))  # 目录语义上的 topology 名
        connectivity_name = str(result.metadata.get("connectivity_preset", "-"))  # 当前显式 connectivity recipe
        urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"  # 如果只导出 hand_cfg，则这里明确提示
        print(f"[{index:04d}] {topology_kind} | {topology_name} | {connectivity_name} | {urdf_path}")
    if print_limit is not None and len(results) > len(preview_results):
        print(f"... {len(results) - len(preview_results)} more results omitted from terminal preview")


def _repo_root_from_runner_file() -> Path:
    r"""根据当前 helper 文件位置回推 AnyMani 仓库根目录。"""

    return Path(__file__).resolve().parents[5]  # `.../source/anymani/anymani/assets/scripts/_asset_generate_runner.py` → repo root


def _resolve_editable_path(path_like: EditablePath) -> Path:
    r"""把配置模块中可编辑路径规范化成绝对路径。

    解析顺序是：

    1. 若本身已是绝对路径，则直接使用；
    2. 若是相对路径，优先解释成“工作区根目录下的相对路径”；
    3. 若工作区解释失败，再回退为“AnyMani 仓库根下的相对路径”。

    Args:
        path_like (EditablePath): 研究者在配置模块里填写的路径。

    Returns:
        Path: 绝对路径形式的规范化结果。
    """

    repo_root = _repo_root_from_runner_file()  # AnyMani 仓库根
    raw_path = Path(path_like).expanduser()  # 先处理 `~` 这类用户路径写法
    if raw_path.is_absolute():
        return raw_path  # 绝对路径不再做额外解释
    workspace_path = repo_root.parent / raw_path  # 工作区根通常是 `/home/hac/isaac`
    if workspace_path.exists():
        return workspace_path  # 优先允许配置直接从工作区根写相对路径
    return repo_root / raw_path  # 最后回退到 AnyMani 仓库根


def resolve_source_topology_dir(source_path: EditablePath) -> Path:
    r"""把用户填写的来源路径解析成 pre-made topology 根目录。

    新 contract 下，独立 post-mutate 只接受 topology 根目录作为来源输入。
    topology 根必须直接包含 `hand.yaml`。

    Args:
        source_path (EditablePath): 来源 topology 路径。

    Returns:
        Path: 规范化后的 topology 根目录。
    """

    resolved_path = _resolve_editable_path(source_path)  # 先规范化相对/绝对路径
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"source topology path does not exist or is not a directory: {resolved_path}")
    if not (resolved_path / "hand.yaml").is_file():
        raise FileNotFoundError(
            "Independent post-mutate now requires a topology-root sidecar; "
            f"missing {resolved_path / 'hand.yaml'}"
        )
    return resolved_path


def plan_post_mutate_run_dir(source_topology_dir: Path) -> Path:
    r"""为一次独立 post-mutate 规划新的时间戳 run 根目录。

    Args:
        source_topology_dir (Path): pre-made topology 根目录。

    Returns:
        Path: 形如 `<topology>/<mutate_timestamp>/` 的新 run 根。
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # mutate run 时间戳采用与 premade 一致的格式
    run_dir = source_topology_dir / timestamp
    collision_index = 2
    while run_dir.exists():
        run_dir = source_topology_dir / f"{timestamp}_{collision_index:02d}"  # 同秒重跑时继续追加后缀
        collision_index += 1
    return run_dir


def prepare_post_mutate_run_cfg(
    run_cfg: HandGeneratorCfg,
    *,
    source_path: EditablePath,
) -> tuple[HandGeneratorCfg, Path, Path]:
    r"""把 mutate-only 所需的 topology 路径 lower 回正式 `HandGeneratorCfg`。

    Args:
        run_cfg (HandGeneratorCfg): 原始 `mode="mutate"` 配置模板。
        source_path (EditablePath): 用户填写的来源 topology 路径。

    Returns:
        tuple[HandGeneratorCfg, Path, Path]:
        1. 已替换 `source_topology_dir` 的正式运行 cfg；
        2. 解析出的来源 topology 根目录；
        3. 计划使用的 mutate run 根目录。
    """

    source_topology_dir = resolve_source_topology_dir(source_path)  # mutate-only 来源恢复现在只认 topology 根
    planned_run_dir = plan_post_mutate_run_dir(source_topology_dir)  # 这里只做预览/打印，不实际 mkdir
    return (
        run_cfg.replace(
            source_topology_dir=source_topology_dir,  # 运行时会由 `HandGenerator` 自己在 topology 根下创建时间戳 run
            output_dir=source_topology_dir.parent,  # mutate-only 新 contract 下该字段不再主导路径，只保留成兼容占位
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
    r"""打印独立 post-mutate 当前实际生效的运行参数。

    Args:
        run_cfg (HandGeneratorCfg): 已 lower 完 topology 路径后的正式运行配置。
        source_path (EditablePath): 用户原始填写的来源路径。
        source_topology_dir (Path): 最终解析出的来源 topology 根目录。
        planned_run_dir (Path): 本轮计划使用的 mutate 时间戳 run 根。
    """

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
    r"""执行独立 post-mutate 正式路径，并把结果收成列表。"""

    return list(HandGenerator(run_cfg).generate_batch())  # mutate-only 常需要一次性打印 summary，因此先 materialize


def print_post_mutate_result_summary(results: list[HandGenerationResult], *, print_limit: int | None) -> None:
    r"""打印独立 post-mutate 的结果统计与 preview。

    Args:
        results (list[HandGenerationResult]): mutate-only 结果列表。
        print_limit (int | None): 终端 preview 上限；`None` 表示全打印。
    """

    print("=== independent post-mutate summary ===")
    print(f"generated variants = {len(results)}")
    if not results:
        print("(no result)")
        print()
        return

    topology_counter = Counter(str(result.metadata.get("topology_name", "-")) for result in results)  # 每个来源 topology 下成功生成的样本数
    for topology_name, count in sorted(topology_counter.items()):
        print(f"{topology_name}: {count}")
    print()

    if print_limit == 0:
        return

    preview_limit = len(results) if print_limit is None else min(len(results), print_limit)  # preview 只限制终端输出，不裁剪真实结果
    print("=== result preview ===")
    for index, result in enumerate(results[:preview_limit], start=1):
        sample_id = str(result.metadata.get("id", "-"))  # 当前新生成的变体 sample id
        origin_id = str(result.metadata.get("source_origin_sample_id", "-"))  # 来源 pre-made 逻辑样本 id
        topology_name = str(result.metadata.get("topology_name", result.metadata.get("source_topology_dir", "-")))  # 当前来源 topology 语义名
        term_names = ",".join(sorted(result.metadata.get("post_mutate_samples", {}).keys()))  # 本轮实际涉及的 mutator term 名集合
        urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"  # 如未导出 URDF，则显式提示
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
