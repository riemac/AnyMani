"""post-mutate 独立入口 quick façade。

这个脚本的定位与 `quick_pre_made.py` 完全一致：

1. 不再包一层额外 runner；
2. 顶部直接把研究者最常改的字段摊开；
3. 真正唯一的运行入口仍然只有一个 `RUN_CFG = HandGeneratorCfg(...)`。

当前脚本只服务“对已有 pre-made topology 做独立 post-mutate”这条工作流：

- 输入是一个 **topology 目录**
- 首次运行时自动把唯一 pre-made 原始样本重命名为 `*_origin`
- 新的 post-mutate 变体作为兄弟 sample 目录继续写在该 topology 下
- topology 目录下同步写本轮 `summary.yaml`
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
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
RecolorFacade = str | dict[str, tuple[float, float, float, float]] | bool | None


# ============================================================================
#  用户可编辑区
# ============================================================================


# NOTE:
# 这里的输入语义已经固定为 **topology 目录**，而不是 sample 目录。
# generator 会自动：
#
# 1. 找到该 topology 下唯一 pre-made 原始 sample；
# 2. 首次运行时把它改名为 `*_origin`；
# 3. 从 `hand.yaml.hand_cfg` 恢复出完整 `HandCfg`；
# 4. 把新的 post-mutate 样本继续写回同一个 topology 目录。
SOURCE_TOPOLOGY_DIR: Path = (
    REPO_ROOT
    / "source"
    / "anymani"
    / "anymani"
    / "assets"
    / "generated"
    / "2026-04-29_21-38-48"
    / "single_palm_leap"
    / "left_t4_i4_m4_r4"
)  # 本轮验收默认指向 Leap full topology；重新跑 quick_pre_made.py 后按新 timestamp 更新即可

N_SAMPLES: int = 20  # post-mutate 固定走 Monte Carlo 联合采样；这里控制对同一 pre-made 原点采几次
ARTIFACT_LEVEL: ArtifactLevel = "bundle"  # 独立 post-mutate 默认保留完整 bundle，方便人工逐个检查
RECOLORED: RecolorFacade = "anatomy_v1"  # 默认沿用 pre-made 常用 anatomy palette；若想关掉，直接改成 None/False
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

POST_MUTATE_CFG = HandMutatorCfg(
    terms={
        "link_scale": MutatorTerm(cfg=LINK_SCALE_TERM_CFG),
        "mount_perturb": MutatorTerm(cfg=MOUNT_PERTURB_TERM_CFG),
        "limit_tweak": MutatorTerm(cfg=LIMIT_TWEAK_TERM_CFG),
        "tip_replace": MutatorTerm(cfg=TIP_REPLACE_TERM_CFG),
    },
    order=("link_scale", "mount_perturb", "limit_tweak", "tip_replace"),
    on_reject="abort",
    step_validate=False,
)  # 这里直接显式列出 term container，避免再引入一层 QuickPostMutateCfg 包装

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
    print(f"source_topology_dir = {run_cfg.source_topology_dir}")  # 当前 post-mutate 操作的 topology 目录
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
    return f"[{index:03d}] {sample_id} <= {origin_id} | {topology_name} | terms={term_names}"


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


def main(run_cfg: HandGeneratorCfg | None = None) -> int:
    r"""运行独立 post-mutate quick façade。"""

    effective_cfg = run_cfg or RUN_CFG
    print_post_mutate_summary(effective_cfg)
    results = enumerate_post_mutate_bundles(effective_cfg)
    print_result_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
