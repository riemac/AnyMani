r"""post-mutate 独立入口 quick façade。

这个脚本的定位与 `quick_pre_made.py` 完全一致：

1. 不再包一层额外 runner；
2. 顶部直接把研究者最常改的字段摊开；
3. 真正唯一的运行入口仍然只有一个 `RUN_CFG = HandGeneratorCfg(...)`。

当前脚本只服务“对某个已生成的 pre-made sample 做独立 post-mutate”这条工作流：

- 用户只需要改一行 `SOURCE_PREMADE_PATH`，可以填 topology 目录，也可以填 sample 目录；
- 默认在该 sample 下创建 `<run_name>/` 运行目录
- 每次运行先把 source sample 复制成本轮 staging origin，再交给正式 generator
- `RUN_POLICY="overwrite"` / `"new"` / `"reuse"` 控制这轮是否覆盖旧目录

# NOTE:
这里的“source sample 内部 run 目录”方案是默认策略。它的科研好处是：

1. `f5d8c069` 这个 pre-made 原点永远不被重命名、不被覆盖；
2. post-mutate 每轮调参都有独立目录，可以直接对比；
3. 当某轮参数不通过 validator，只需覆盖该轮 run，不污染原始 topology。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
import shutil
from typing import Any, Literal


# ============================================================================
#  Python 路径 bootstrap
# ============================================================================


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPO_ROOT / "source" / "anymani"


def _bootstrap_python_path() -> None:
    r"""确保脚本以文件路径直跑时，也稳定命中当前工作区源码。"""

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


if __package__ in {None, ""}:
    _bootstrap_python_path()

    from anymani.assets.generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
    from anymani.assets.generator.mutate import (
        HandMutatorCfg,
        LimitTweakCfg,
        LinkScaleCfg,
        MountPerturbCfg,
        MutatorTerm,
        ScalarDistributionCfg,
        TipReplaceCfg,
    )
    from anymani.assets.validator.hand_rules import HandValidatorCfg
else:
    from ..generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
    from ..generator.mutate import (
        HandMutatorCfg,
        LimitTweakCfg,
        LinkScaleCfg,
        MountPerturbCfg,
        MutatorTerm,
        ScalarDistributionCfg,
        TipReplaceCfg,
    )
    from ..validator.hand_rules import HandValidatorCfg


# ============================================================================
#  类型别名
# ============================================================================


ArtifactLevel = Literal["hand_cfg", "urdf", "bundle"]
PostMutateLayout = Literal["nested", "sibling"]
PostMutateRunPolicy = Literal["overwrite", "new", "reuse"]
RecolorFacade = str | dict[str, tuple[float, float, float, float]] | bool | None
EditablePath = str | Path


# ============================================================================
#  用户可编辑区
# ============================================================================


SOURCE_PREMADE_PATH: EditablePath = "AnyMani/source/anymani/anymani/assets/generated/2026-05-03_09-45-45/single_palm_leap/right_t4_i4_m4_r4/f5d8c069"  # 直接粘贴 topology/sample 路径即可；相对路径默认从 `/home/hac/isaac` 解析

SOURCE_PREMADE_SAMPLE_ID: str | None = "f5d8c069"
"""当 `SOURCE_PREMADE_PATH` 指向 topology 目录且目录下有多个 sample 时，用这里指定 sample ID。"""

POST_MUTATE_LAYOUT: PostMutateLayout = "nested"
r"""产物布局。

- `"nested"`：默认方案，写到 `SOURCE_PREMADE_SAMPLE_DIR/<RUN_NAME>/`
- `"sibling"`：备用方案，写到同一 topology 下的 `<sample_id>_post_mutate/<RUN_NAME>/`

二者都会先复制 source sample，再让 generator 在复制出的 staging 目录中标记
`*_origin`，因此不会破坏 `SOURCE_PREMADE_SAMPLE_DIR` 原始 pre-made 产物。
"""

POST_MUTATE_RUN_NAME: str = "try_001"
"""当前这轮 post-mutate 调参运行名；建议手动改成 `try_002` / `limit_small` 等可读名字。"""

POST_MUTATE_RUN_POLICY: PostMutateRunPolicy = "overwrite"
"""当目标 run 目录已存在时如何处理。

- `"overwrite"`：删除旧 run 后重建，适合反复调同一组参数；
- `"new"`：自动追加 `_01/_02/...`，适合保留每次尝试；
- `"reuse"`：复用已有 run，继续向其中补样本。
"""

N_SAMPLES: int = 20  # post-mutate 固定走 Monte Carlo 联合采样；这里控制对同一 pre-made 原点采几次
ARTIFACT_LEVEL: ArtifactLevel = "bundle"  # 独立 post-mutate 默认保留完整 bundle，方便人工逐个检查
RECOLORED: RecolorFacade = "anatomy_soft_v1"  # 默认沿用当前 pre-made 的柔和 anatomy palette；若想关掉，直接改成 None/False
OUTPUT_LAYOUT: Literal["flat", "recursive"] = "recursive"  # mutate-only 不再新建 topology 层，但 metadata 里仍保留 layout 语义

# NOTE:
# 你已经明确要求“首版默认四个 term 都启用”，所以这里把四种工具全部直接写在顶部。
# 若后续实验只想关掉某一项，直接把 `MUTATE_CFG` 里的对应 term 删除即可。
LINK_SCALE_TERM_CFG = LinkScaleCfg(
    scale_mode="relative",
    delta_distribution=ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.03),
    clip_ratio=0.2,
)  # 连杆长度做相对缩放，默认 $\sigma=3\%$
MOUNT_PERTURB_TERM_CFG = MountPerturbCfg(
    perturb_rotation=True,
    translation_distribution=ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.002),
    rotation_distribution=ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.03),
    clip_translation=0.01,
    clip_rotation=0.12,
)  # finger 挂载点同时做平移/姿态小扰动
LIMIT_TWEAK_TERM_CFG = LimitTweakCfg(
    mode="absolute",
    symmetric=False,
    delta_distribution=ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.03),
    clip=0.12,
)  # 关节限位直接做弧度级小扰动
TIP_REPLACE_TERM_CFG = TipReplaceCfg(
    mode="geometry_swap",
    target_geometry=None,
    size_distribution=ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.0015),
)  # 指尖几何在 box/cylinder 间互换，并叠加毫米级尺寸扰动

POST_MUTATE_PATCH_ORDER = ("link_scale", "mount_perturb", "limit_tweak", "tip_replace")
r"""联合采样后的 patch 合成顺序，不是“逐算子串行采样”。

post-mutate 的并行/工程优化语义仍然是：

1. 四个算子的随机变量先从各自分布中**联合批量采样**；
2. 每个 term 基于同一个 source `HandCfg` 生成 deferred patch；
3. pipeline 按这里的顺序检查 patch 冲突并 `ApplyOnce`。

因此这个顺序只负责 deterministic compose / conflict reporting。它保留在 quick
脚本顶部，是为了后续若两个 term 都想触碰同一路径时，研究者能显式决定优先级。
"""

POST_MUTATE_CFG = HandMutatorCfg(
    terms={
        "link_scale": MutatorTerm(cfg=LINK_SCALE_TERM_CFG),
        "mount_perturb": MutatorTerm(cfg=MOUNT_PERTURB_TERM_CFG),
        "limit_tweak": MutatorTerm(cfg=LIMIT_TWEAK_TERM_CFG),
        "tip_replace": MutatorTerm(cfg=TIP_REPLACE_TERM_CFG),
    },
    order=POST_MUTATE_PATCH_ORDER,
    on_reject="abort",
    step_validate=False,
)  # `terms` 声明启用哪些算子；`order` 仅声明联合采样后的 patch 合成顺序

POST_MUTATE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    post_mutate=HandValidatorCfg.PostMutateCfg(
        finger_count_min=3,
        require_non_thumb_with_min_revolute_dof=3,
        check_finger_spacing=True,
        min_finger_spacing=0.01,
        check_mount_consistency=True,
    )
)  # quick 顶部显式写出 post-mutate validator；若想完全关闭，直接改成 None

_PRINT_RESULT_LIMIT: int | None = 20  # 终端最多 preview 多少个新变体；`0` = 只看 summary


# ============================================================================
#  post-mutate staging 目录辅助
# ============================================================================


def _sample_id_from_dir(sample_dir: Path) -> str:
    r"""从 pre-made sample 目录名取稳定样本 ID。"""

    return sample_dir.name.rstrip("/")  # 目录名就是 pre-made sample ID，例如 `f5d8c069`


def _resolve_editable_path(path_like: EditablePath) -> Path:
    r"""把用户粘贴的一行路径解析成绝对路径。

    Args:
        path_like (EditablePath): 用户在 `SOURCE_PREMADE_PATH` 中填写的路径，可以是
            `AnyMani/...` 这样的 workspace 相对路径，也可以是 `/home/...` 绝对路径。

    Returns:
        Path: 解析后的绝对路径。
    """

    raw_path = Path(path_like).expanduser()  # 支持 `~/...`，方便临时复制路径
    if raw_path.is_absolute():
        return raw_path  # 绝对路径不再猜测 root，避免误改用户意图

    workspace_path = REPO_ROOT.parent / raw_path  # `/home/hac/isaac` 视角，最符合 IDE 里复制的 `AnyMani/...`
    if workspace_path.exists():
        return workspace_path
    return REPO_ROOT / raw_path  # 兼容从 AnyMani repo root 视角写的 `source/anymani/...`


def resolve_source_premade_sample_dir(
    source_path: EditablePath,
    *,
    sample_id: str | None = None,
) -> Path:
    r"""从用户可编辑路径解析出真正的 pre-made sample 目录。

    `SOURCE_PREMADE_PATH` 可以有两种形状：

    1. sample 目录：`.../right_t4_i4_m4_r4/f5d8c069`，自身包含 `hand.yaml`；
    2. topology 目录：`.../right_t4_i4_m4_r4`，其下有一个或多个 sample 目录。

    Args:
        source_path (EditablePath): 用户填写的 topology 或 sample 路径。
        sample_id (str | None): 当 `source_path` 是 topology 且不止一个 sample 时的选择锚点。

    Returns:
        Path: 真正用于 staging copy 的 pre-made sample 目录。
    """

    resolved_path = _resolve_editable_path(source_path)
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"source pre-made path does not exist or is not a directory: {resolved_path}")

    if (resolved_path / "hand.yaml").is_file():
        if sample_id is not None and resolved_path.name != sample_id:
            raise ValueError(
                "SOURCE_PREMADE_SAMPLE_ID conflicts with sample path: "
                f"path sample={resolved_path.name!r}, requested={sample_id!r}"
            )
        return resolved_path  # 用户已直接填到 sample 目录，这是最轻量的路径

    sample_dirs = sorted(path for path in resolved_path.iterdir() if path.is_dir() and (path / "hand.yaml").is_file())
    if sample_id is not None:
        selected_dir = resolved_path / sample_id
        if selected_dir in sample_dirs:
            return selected_dir
        raise FileNotFoundError(f"sample_id {sample_id!r} was not found under topology directory {resolved_path}")

    normal_sample_dirs = [path for path in sample_dirs if not path.name.endswith("_origin")]
    if len(normal_sample_dirs) == 1:
        return normal_sample_dirs[0]  # topology 下只有一个普通 pre-made sample，可自动推断

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
    r"""计算 quick façade 计划使用的 post-mutate staging topology 目录。

    Args:
        source_sample_dir (Path): 原始 pre-made sample 目录。
        layout (PostMutateLayout): `nested` 或 `sibling`。
        run_name (str): 当前 post-mutate 运行名。

    Returns:
        Path: 传给 `HandGeneratorCfg.source_topology_dir` 的 staging topology 目录。
    """

    sample_id = _sample_id_from_dir(source_sample_dir)  # 用 source sample ID 作为 run 目录命名锚点
    if layout == "nested":
        return source_sample_dir / run_name  # source sample 内部 run：路径最短，且不再重复出现 `post_mutate`
    if layout == "sibling":
        return source_sample_dir.parent / f"{sample_id}_post_mutate" / run_name  # 平级 run：便于同 topology 下集中浏览
    raise ValueError(f"unknown post-mutate layout: {layout!r}")


def _choose_run_dir(base_run_dir: Path, *, run_policy: PostMutateRunPolicy) -> Path:
    r"""根据覆盖策略选择本次实际使用的 run 目录。

    `new` 会在已有目录后追加 `_01/_02/...`，而 `overwrite` / `reuse` 都保持
    原始 `base_run_dir` 名字，差别只在是否清空目录。
    """

    if run_policy == "overwrite":
        if base_run_dir.exists():
            shutil.rmtree(base_run_dir)  # 明确覆盖整轮 post-mutate run，不触碰 source sample
        return base_run_dir
    if run_policy == "reuse":
        return base_run_dir
    if run_policy != "new":
        raise ValueError(f"unknown post-mutate run policy: {run_policy!r}")

    if not base_run_dir.exists():
        return base_run_dir
    for index in range(1, 1000):
        candidate = base_run_dir.with_name(f"{base_run_dir.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate a new post-mutate run directory near {base_run_dir}")


def _copy_source_sample_for_staging(source_sample_dir: Path, run_dir: Path) -> None:
    r"""把 pre-made sample 复制到 staging topology 目录下。

    复制目标形状是：

    ```text
    <run_dir>/
      <sample_id>/
        hand.yaml
        hand.urdf
        ...
    ```

    随后现有 `load_post_mutate_source(...)` 会把 `<sample_id>/` 自动改名成
    `<sample_id>_origin/`，并把新 post-mutate 样本写在同一个 `<run_dir>/` 下。
    """

    if not source_sample_dir.is_dir():
        raise FileNotFoundError(f"source pre-made sample dir does not exist: {source_sample_dir}")
    if not (source_sample_dir / "hand.yaml").is_file():
        raise FileNotFoundError(f"source pre-made sample dir must contain hand.yaml: {source_sample_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)  # run_dir 本身是 generator 眼里的 topology 目录
    staged_sample_dir = run_dir / _sample_id_from_dir(source_sample_dir)
    origin_sample_dir = run_dir / f"{_sample_id_from_dir(source_sample_dir)}_origin"
    if staged_sample_dir.exists() or origin_sample_dir.exists():
        return  # `reuse` 路径下已有 staging/origin 时，不重复复制，避免覆盖已生成结果

    _copy_source_sample_bundle(source_sample_dir, staged_sample_dir)


def _copy_source_sample_bundle(source_sample_dir: Path, staged_sample_dir: Path) -> None:
    r"""只复制 source sample 的 bundle 文件，不递归复制历史 run 目录。

    nested 布局现在写到 `SOURCE_PREMADE_SAMPLE_DIR/<RUN_NAME>/`。因此 source sample
    自身会逐渐包含 `try_001/`、`try_002/` 等目录；若继续整目录 `copytree`，这些
    历史 run 会被复制进本轮 `<sample_id>_origin/`，既浪费空间，也会让人工浏览误以为
    origin 本身包含 post-mutate 产物。

    Args:
        source_sample_dir (Path): 原始 pre-made sample 目录。
        staged_sample_dir (Path): 本轮 run 内的临时 source sample 复制件。
    """

    staged_sample_dir.mkdir(parents=True, exist_ok=False)  # staging sample 必须是全新目录，避免混入旧 origin
    for child in source_sample_dir.iterdir():
        target = staged_sample_dir / child.name
        if child.is_file():
            shutil.copy2(child, target)  # `hand.yaml` / `hand.urdf` / `tree.txt` 等 bundle 文件逐个复制


def prepare_post_mutate_source_topology(
    *,
    source_sample_dir: Path,
    layout: PostMutateLayout,
    run_name: str,
    run_policy: PostMutateRunPolicy,
) -> Path:
    r"""准备本轮 post-mutate 使用的 staging topology 目录。

    这个函数是 quick façade 和正式 generator 之间的唯一“目录适配层”：

    - quick 层吃用户最直观的 `SOURCE_PREMADE_SAMPLE_DIR`；
    - generator 层继续吃它已经实现好的 `source_topology_dir`；
    - 中间只做一次复制，不修改 pre-made 原始样本。
    """

    base_run_dir = planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout=layout,
        run_name=run_name,
    )
    run_dir = _choose_run_dir(base_run_dir, run_policy=run_policy)
    _copy_source_sample_for_staging(source_sample_dir, run_dir)
    return run_dir


SOURCE_PREMADE_SAMPLE_DIR: Path = resolve_source_premade_sample_dir(
    SOURCE_PREMADE_PATH,
    sample_id=SOURCE_PREMADE_SAMPLE_ID,
)  # 解析后的内部 sample 目录；后续逻辑统一只消费 sample 语义

SOURCE_TOPOLOGY_DIR: Path = planned_post_mutate_topology_dir(
    source_sample_dir=SOURCE_PREMADE_SAMPLE_DIR,
    layout=POST_MUTATE_LAYOUT,
    run_name=POST_MUTATE_RUN_NAME,
)  # 静态预览用路径；真正运行时会先经过 `prepare_post_mutate_source_topology(...)`


# 这是 quick_post_mutate.py 唯一正式运行入口。
RUN_CFG = HandGeneratorCfg(
    mode="mutate",
    artifact_level=ARTIFACT_LEVEL,
    source_topology_dir=SOURCE_TOPOLOGY_DIR,
    output_dir=SOURCE_TOPOLOGY_DIR.parent,  # mutate-only 实际落盘根由 source_topology_dir 决定；这里仅保留 cfg 一致性
    n_samples=N_SAMPLES,
    Mutate=POST_MUTATE_CFG,
    Validate=POST_MUTATE_VALIDATOR_CFG,
    recolored=RECOLORED,
    output_layout=OUTPUT_LAYOUT,
)


# ============================================================================
#  摘要 / 打印辅助
# ============================================================================


def print_post_mutate_summary(run_cfg: HandGeneratorCfg) -> None:
    r"""打印当前 mutate-only quick façade 的关键配置。"""

    print("=== independent post-mutate quick knobs ===")
    print(f"source_premade_path= {SOURCE_PREMADE_PATH}")  # 用户手改的一行路径，可能是 topology，也可能是 sample
    print(f"source_sample_id   = {SOURCE_PREMADE_SAMPLE_ID}")  # topology 多 sample 时的显式选择
    print(f"source_sample_dir   = {SOURCE_PREMADE_SAMPLE_DIR}")  # 原始 pre-made sample，quick 层的真正用户入口
    print(f"layout             = {POST_MUTATE_LAYOUT}")  # nested / sibling
    print(f"run_name           = {POST_MUTATE_RUN_NAME}")  # 本轮 post-mutate run 名
    print(f"run_policy         = {POST_MUTATE_RUN_POLICY}")  # overwrite / new / reuse
    print(f"source_topology_dir= {run_cfg.source_topology_dir}")  # generator 实际消费的 staging topology 目录
    print(f"n_samples          = {run_cfg.n_samples}")  # 当前对该 topology 原点采样多少个变体
    print(f"artifact_level     = {run_cfg.artifact_level}")  # hand_cfg / urdf / bundle
    print(f"recolored          = {run_cfg.recolored}")  # 当前导出时是否继续做 visual recolor
    print(f"validator_on       = {run_cfg.Validate is not None}")  # 让研究者一眼看出 post-mutate validator 是否启用
    print(f"mutator_order      = {list(run_cfg.Mutate.order)}")  # 联合采样后按什么顺序 lower 成确定性变换
    print(f"mutator_terms      = {list(run_cfg.Mutate.terms)}")  # 当前真正启用的工具名
    if run_cfg.Validate is not None:
        print(f"post_mutate.finger_count_min = {run_cfg.Validate.post_mutate.finger_count_min}")  # 后序几何扰动后仍至少保留 3 根手指
        print(
            "post_mutate.require_non_thumb_with_min_revolute_dof = "
            f"{run_cfg.Validate.post_mutate.require_non_thumb_with_min_revolute_dof}"
        )  # 保持至少一根 non-thumb 仍然是有效操作手指
        print(f"post_mutate.check_finger_spacing = {run_cfg.Validate.post_mutate.check_finger_spacing}")  # 是否检查 finger 挂载点最小间距
        print(f"post_mutate.min_finger_spacing = {run_cfg.Validate.post_mutate.min_finger_spacing}")  # 最小挂载间距阈值（meter）
    print()


def _result_preview_line(index: int, result: HandGenerationResult) -> str:
    r"""把一条 mutate-only 结果压成便于终端扫读的一行。"""

    sample_id = str(result.metadata.get("id", "-"))  # 当前新生成的 post-mutate sample ID
    origin_id = str(result.metadata.get("source_origin_sample_id", "-"))  # 当前变体来自哪个 pre-made 原点
    topology_name = str(result.metadata.get("topology_name", result.metadata.get("source_topology_dir", "-")))
    term_names = ",".join(sorted(result.metadata.get("post_mutate_samples", {}).keys()))
    urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"
    return f"[{index:03d}] {sample_id} <= {origin_id} | {topology_name} | terms={term_names} | {urdf_path}"


def print_result_summary(results: list[HandGenerationResult]) -> None:
    r"""打印 mutate-only 批运行的结果摘要。"""

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

    if _PRINT_RESULT_LIMIT == 0:
        return

    preview_limit = len(results) if _PRINT_RESULT_LIMIT is None else min(len(results), _PRINT_RESULT_LIMIT)
    print("=== result preview ===")
    for index, result in enumerate(results[:preview_limit], start=1):
        print(_result_preview_line(index, result))
    if preview_limit < len(results):
        print(f"... ({len(results) - preview_limit} more results omitted)")
    print()


def enumerate_post_mutate_bundles(run_cfg: HandGeneratorCfg) -> list[HandGenerationResult]:
    r"""执行一轮独立 post-mutate 批量生成，并把成功结果收成列表。"""

    return list(HandGenerator(run_cfg).generate_batch())


def prepare_run_cfg(run_cfg: HandGeneratorCfg) -> HandGeneratorCfg:
    r"""准备 staging 目录，并返回真正要交给 generator 的 cfg。

    `RUN_CFG` 在模块 import 时只是一个“可读配置快照”；真正运行前需要根据
    `POST_MUTATE_RUN_POLICY` 选择或清理目录，并复制 source sample。
    """

    prepared_topology_dir = prepare_post_mutate_source_topology(
        source_sample_dir=SOURCE_PREMADE_SAMPLE_DIR,
        layout=POST_MUTATE_LAYOUT,
        run_name=POST_MUTATE_RUN_NAME,
        run_policy=POST_MUTATE_RUN_POLICY,
    )
    return run_cfg.replace(
        source_topology_dir=prepared_topology_dir,
        output_dir=prepared_topology_dir.parent,
    )


def main(run_cfg: HandGeneratorCfg | None = None) -> int:
    r"""运行独立 post-mutate quick façade。"""

    effective_cfg = prepare_run_cfg(run_cfg or RUN_CFG)
    print_post_mutate_summary(effective_cfg)
    results = enumerate_post_mutate_bundles(effective_cfg)
    print_result_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
