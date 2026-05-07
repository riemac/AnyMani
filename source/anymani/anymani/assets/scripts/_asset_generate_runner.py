r"""统一资产生成 runner helper。

本文件集中放置原先 `quick_pre_made.py` / `quick_post_mutate.py`
里那些“不属于配置模块、也不该塞进 CLI 主入口”的中间逻辑。

它的职责刻意收敛成两类：

1. pre-made 方向：
   - registry summary 打印；
   - 结果枚举与 preview；
2. post-mutate 方向：
   - 来源 sample / topology 路径解析；
   - staging 目录准备；
   - run 目录命名策略；
   - 结果 summary 与 preview。

# NOTE:
这里不反向依赖具体的 config 模块常量，而只接受：

- `HandGeneratorCfg`
- runner 传下来的少量路径 / layout / policy 参数

这样做的原因是减少导入链长度，避免“只想 import helper 或看 `--help`，
却因为某个配置模块 import 失败而整体不可用”。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
from typing import Any, Literal

from ..generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
from ..presets.connectivity_presets import list_finger_connectivity_preset_names


EditablePath = str | Path
"""允许 helper 接收绝对路径、相对路径或 `Path` 对象。"""

PostMutateLayout = Literal["nested", "sibling"]
"""独立 post-mutate 的 run 目录布局模式。"""

PostMutateRunPolicy = Literal["overwrite", "new", "reuse"]
"""独立 post-mutate 的 run 目录分配策略。"""


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
    print(f"output_layout      = {run_cfg.output_layout}")
    print(f"output_dir         = {run_cfg.output_dir}")
    print(f"max_enumerate      = {run_cfg.max_enumerate}")
    print(f"premade_parallel   = {run_cfg.premade_parallel}")
    print(f"parallel_workers   = {run_cfg.premade_parallel_workers}")
    print(f"parallel_fallback  = {run_cfg.premade_parallel_fallback}")
    print(f"connectivity_cfg   = {run_cfg.connectivity_presets}")
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


def _sample_id_from_dir(sample_dir: Path) -> str:
    r"""从 sample 目录名恢复 sample id。

    Args:
        sample_dir (Path): 形如 `.../<sample_id>/` 的目录。

    Returns:
        str: 去掉尾部 `/` 后的 sample id 字符串。
    """

    return sample_dir.name.rstrip("/")  # 保持和目录名一一对应，避免额外引入 rename 规则


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


def resolve_source_premade_sample_dir(
    source_path: EditablePath,
    *,
    sample_id: str | None = None,
) -> Path:
    r"""把用户填写的来源路径解析成唯一的 pre-made sample 目录。

    允许两种入口：

    1. 直接给 sample 目录；
    2. 给 topology 目录，再配合 `sample_id` 精确选择其中某个 sample。

    Args:
        source_path (EditablePath): 来源 sample 或 topology 路径。
        sample_id (str | None): 当 `source_path` 是 topology 目录时，用于锁定来源样本。

    Returns:
        Path: 唯一确定的来源 sample 目录。
    """

    resolved_path = _resolve_editable_path(source_path)  # 先规范化路径解释语义
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"source pre-made path does not exist or is not a directory: {resolved_path}")

    # 如果路径本身就是 sample 目录，则它必须直接包含 `hand.yaml`。
    if (resolved_path / "hand.yaml").is_file():
        if sample_id is not None and resolved_path.name != sample_id:
            raise ValueError(
                "SOURCE_PREMADE_SAMPLE_ID conflicts with sample path: "
                f"path sample={resolved_path.name!r}, requested={sample_id!r}"
            )
        return resolved_path

    sample_dirs = sorted(path for path in resolved_path.iterdir() if path.is_dir() and (path / "hand.yaml").is_file())  # topology 目录下所有候选 sample
    if sample_id is not None:
        selected_dir = resolved_path / sample_id  # 用户显式指定的目标 sample 子目录
        if selected_dir in sample_dirs:
            return selected_dir
        raise FileNotFoundError(f"sample_id {sample_id!r} was not found under topology directory {resolved_path}")

    normal_sample_dirs = [path for path in sample_dirs if not path.name.endswith("_origin")]  # mutate-only 回放时要忽略 `_origin` 备份目录
    if len(normal_sample_dirs) == 1:
        return normal_sample_dirs[0]

    raise ValueError(
        "SOURCE_PREMADE_PATH points to a topology directory with multiple candidate samples; "
        "please set SOURCE_PREMADE_SAMPLE_ID explicitly. "
        f"candidates={[path.name for path in normal_sample_dirs]}"
    )


def planned_post_mutate_topology_dir(
    *,
    source_sample_dir: Path,
    layout: PostMutateLayout,
    run_name: str,
) -> Path:
    r"""根据 layout 规则推导 post-mutate run 目录。

    Args:
        source_sample_dir (Path): 原始 pre-made sample 目录。
        layout (PostMutateLayout): `nested` 或 `sibling`。
        run_name (str): 当前 mutate 调试轮次名。

    Returns:
        Path: 计划使用的 run 目录路径。
    """

    sample_id = _sample_id_from_dir(source_sample_dir)  # 当前来源 sample 的稳定语义 id
    if layout == "nested":
        return source_sample_dir / run_name  # 在 sample 目录内部再嵌套一个 run 子目录
    if layout == "sibling":
        return source_sample_dir.parent / f"{sample_id}_post_mutate" / run_name  # 在 sample 同级新开一个 mutate-only topology 目录
    raise ValueError(f"unknown post-mutate layout: {layout!r}")


def _choose_run_dir(base_run_dir: Path, *, run_policy: PostMutateRunPolicy) -> Path:
    r"""按 run policy 为 post-mutate 分配最终 run 目录。

    Args:
        base_run_dir (Path): 由 layout 推导出的理论 run 目录。
        run_policy (PostMutateRunPolicy): `overwrite` / `reuse` / `new`。

    Returns:
        Path: 最终应使用的 run 目录路径。
    """

    if run_policy == "overwrite":
        if base_run_dir.exists():
            shutil.rmtree(base_run_dir)  # 仅删除当前目标 run 目录，不碰其它 run 痕迹
        return base_run_dir
    if run_policy == "reuse":
        return base_run_dir  # 明确允许复用已有目录，适合只想重复恢复已有 staging 的情况
    if run_policy != "new":
        raise ValueError(f"unknown post-mutate run policy: {run_policy!r}")

    if not base_run_dir.exists():
        return base_run_dir
    for index in range(1, 1000):
        candidate = base_run_dir.with_name(f"{base_run_dir.name}_{index:02d}")  # 自动在 run 名后追加递增后缀
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate a new post-mutate run directory near {base_run_dir}")


def _copy_source_sample_bundle(source_sample_dir: Path, staged_sample_dir: Path) -> None:
    r"""把来源 sample 的 bundle 文件复制到 staging sample 目录。

    Args:
        source_sample_dir (Path): 原始 pre-made sample 目录。
        staged_sample_dir (Path): 将要创建的 staging sample 目录。
    """

    staged_sample_dir.mkdir(parents=True, exist_ok=False)  # staging sample 必须是全新目录，防止混入旧文件
    for child in source_sample_dir.iterdir():
        target = staged_sample_dir / child.name  # 保留 bundle 内原始文件名，便于后续恢复逻辑使用
        if child.is_file():
            shutil.copy2(child, target)  # 保留时间戳和基本 metadata，方便人工比对


def _copy_source_sample_for_staging(source_sample_dir: Path, run_dir: Path) -> None:
    r"""在 run 目录下创建 mutate-only staging sample。

    Args:
        source_sample_dir (Path): 原始 pre-made sample 目录。
        run_dir (Path): 当前 mutate run 根目录。
    """

    if not source_sample_dir.is_dir():
        raise FileNotFoundError(f"source pre-made sample dir does not exist: {source_sample_dir}")
    if not (source_sample_dir / "hand.yaml").is_file():
        raise FileNotFoundError(f"source pre-made sample dir must contain hand.yaml: {source_sample_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)  # run 根目录允许由 overwrite/reuse/new 逻辑预先分配
    staged_sample_dir = run_dir / _sample_id_from_dir(source_sample_dir)  # mutate-only 生成前的工作副本目录
    origin_sample_dir = run_dir / f"{_sample_id_from_dir(source_sample_dir)}_origin"  # 生成后原始样本会被重命名到这里
    if staged_sample_dir.exists() or origin_sample_dir.exists():
        return  # 若 staging 或 `_origin` 已存在，说明当前 run 已做过初始化，避免重复复制
    _copy_source_sample_bundle(source_sample_dir, staged_sample_dir)


def prepare_post_mutate_source_topology(
    *,
    source_sample_dir: Path,
    layout: PostMutateLayout,
    run_name: str,
    run_policy: PostMutateRunPolicy,
) -> Path:
    r"""为独立 post-mutate 准备正式的 staging topology 目录。

    Args:
        source_sample_dir (Path): 来源 pre-made sample 目录。
        layout (PostMutateLayout): run 布局模式。
        run_name (str): 当前 mutate 调试轮次名。
        run_policy (PostMutateRunPolicy): run 目录策略。

    Returns:
        Path: 已准备好的 mutate-only run 目录。
    """

    base_run_dir = planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout=layout,
        run_name=run_name,
    )
    run_dir = _choose_run_dir(base_run_dir, run_policy=run_policy)  # 先按策略分配最终 run 目录
    _copy_source_sample_for_staging(source_sample_dir, run_dir)  # 再在该 run 下放入 staging sample 副本
    return run_dir


def prepare_post_mutate_run_cfg(
    run_cfg: HandGeneratorCfg,
    *,
    source_path: EditablePath,
    sample_id: str | None,
    layout: PostMutateLayout,
    run_name: str,
    run_policy: PostMutateRunPolicy,
) -> tuple[HandGeneratorCfg, Path, Path]:
    r"""把 mutate-only 运行所需的 staging 路径 lower 回正式 `HandGeneratorCfg`。

    Args:
        run_cfg (HandGeneratorCfg): 原始 `mode="mutate"` 配置模板。
        source_path (EditablePath): 用户填写的来源 sample/topology 路径。
        sample_id (str | None): 当来源是 topology 目录时需要的 sample id。
        layout (PostMutateLayout): run 目录布局模式。
        run_name (str): 当前 mutate 调试轮次名。
        run_policy (PostMutateRunPolicy): run 目录分配策略。

    Returns:
        tuple[HandGeneratorCfg, Path, Path]:
        1. 已替换 `source_topology_dir` 与 `output_dir` 的正式运行 cfg；
        2. 解析出的来源 sample 目录；
        3. 已准备好的 mutate-only run 目录。
    """

    source_sample_dir = resolve_source_premade_sample_dir(source_path, sample_id=sample_id)  # 把人类友好的路径入口解析成唯一 sample
    prepared_topology_dir = prepare_post_mutate_source_topology(
        source_sample_dir=source_sample_dir,
        layout=layout,
        run_name=run_name,
        run_policy=run_policy,
    )
    return (
        run_cfg.replace(
            source_topology_dir=prepared_topology_dir,  # mutate-only 运行时真正读取的是 staging topology
            output_dir=prepared_topology_dir.parent,  # 新样本 summary / bundle 默认写到 run 目录的父层
        ),
        source_sample_dir,
        prepared_topology_dir,
    )


def print_post_mutate_summary(
    run_cfg: HandGeneratorCfg,
    *,
    source_path: EditablePath,
    source_sample_id: str | None,
    source_sample_dir: Path,
    layout: PostMutateLayout,
    run_name: str,
    run_policy: PostMutateRunPolicy,
) -> None:
    r"""打印独立 post-mutate 当前实际生效的运行参数。

    Args:
        run_cfg (HandGeneratorCfg): 已 lower 完 staging 路径后的正式运行配置。
        source_path (EditablePath): 用户原始填写的来源路径。
        source_sample_id (str | None): 用户原始填写的来源 sample id。
        source_sample_dir (Path): 最终解析出的来源 sample 目录。
        layout (PostMutateLayout): 当前 run 布局模式。
        run_name (str): 当前 mutate 调试轮次名。
        run_policy (PostMutateRunPolicy): 当前 run 目录策略。
    """

    print("=== independent post-mutate knobs ===")
    print(f"source_premade_path = {source_path}")
    print(f"source_sample_id    = {source_sample_id}")
    print(f"source_sample_dir   = {source_sample_dir}")
    print(f"layout              = {layout}")
    print(f"run_name            = {run_name}")
    print(f"run_policy          = {run_policy}")
    print(f"source_topology_dir = {run_cfg.source_topology_dir}")
    print(f"n_samples           = {run_cfg.n_samples}")
    print(f"artifact_level      = {run_cfg.artifact_level}")
    print(f"recolored           = {run_cfg.recolored}")
    print(f"validator_on        = {run_cfg.Validate is not None}")
    print(f"mutator_terms       = {[name for name, _ in run_cfg.Mutate.ordered_terms()]}")
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
        origin_id = str(result.metadata.get("source_origin_sample_id", "-"))  # 来源 pre-made 原样本 id
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
    "planned_post_mutate_topology_dir",
    "prepare_post_mutate_run_cfg",
    "prepare_post_mutate_source_topology",
    "print_post_mutate_result_summary",
    "print_post_mutate_summary",
    "print_premade_registry_summary",
    "print_premade_result_summary",
    "resolve_source_premade_sample_dir",
]
