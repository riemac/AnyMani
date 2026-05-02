"""pre-made 全量生成 quick façade。

这个脚本的定位，不是再发明一层新的 generator 框架，而是把你现在最常做的那条
科研工作流，收成一个**单文件、直接可运行、顶部少量字段可改**的便利入口：

1. 默认直接枚举 Leap / Allegro 当前全部合法 pre-made 变体；
2. 这个“全部合法变体”包含：
   - single-family canonical topology
   - missing-finger topology
   - mixed-family topology
3. 默认同时启用 URDF visual recolor；
4. 若你只想缩小空间巡检，只需要改顶部那几行大写变量，而不用再往下翻着找包装层。

# NOTE:
当前这条 quick façade 的设计目标，不是“参数最多”，而是“研究者最少改动就能跑”。
因此脚本对外只暴露少量最关键的 knobs：

- `hand_presets`
- `connectivity_presets`
- `mixed`
- `missing`
- `recolored`
- `max_enumerate`
- `output_dir`

其余复杂 lowering 语义，继续交给正式的：

- `HandGeneratorCfg`
- `HandGenerator`

来负责，避免 quick 脚本自己偷偷分叉一套逻辑。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal


# ============================================================================
#  Python 路径 bootstrap
# ============================================================================


# 当前文件位于：
#   `.../AnyMani/source/anymani/anymani/assets/generator/quick.py`
# 因此：
# - `parents[5]` 是 AnyMani 仓库根目录；
# - `source/anymani/` 才是应加进 `sys.path` 的源码根。
REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPO_ROOT / "source" / "anymani"


def _bootstrap_python_path() -> None:
    r"""确保脚本以文件路径直跑时，也能稳定导入当前工作区源码。

    这里故意不用更“框架式”的入口，是因为 quick façade 的核心诉求就是：
    你在终端里直接 `python quick.py`，也必须稳定落到当前工作区这份源码，
    而不是不小心吃到别的 Python 环境里的旧安装包。
    """

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))  # 把当前源码根插到最前面，显式压过其它环境缓存包


if __package__ in {None, ""}:
    _bootstrap_python_path()

    from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
    from anymani.assets.presets.connectivity_presets import (
        list_finger_connectivity_preset_names,
    )
    from anymani.assets.validator.hand_rules import HandValidatorCfg
else:
    from .hand_generator import HandGenerator, HandGeneratorCfg
    from ..presets.connectivity_presets import (
        list_finger_connectivity_preset_names,
    )
    from ..validator.hand_rules import HandValidatorCfg


# ============================================================================
#  类型别名
# ============================================================================


ConnectivityFacade = dict[str, dict[str, list[str]]] | None
RecolorFacade = str | dict[str, tuple[float, float, float, float]] | bool | None


# ============================================================================
#  用户可编辑区
# ============================================================================

HAND_PRESETS: list[str] = ["single_palm_allegro", "single_palm_leap"]  # 当前纳入 pre-made 枚举的 canonical palm anchor
# CONNECTIVITY_PRESETS: ConnectivityFacade = {
#     "single_palm_allegro": {
#         "thumb": ["allegro_thumb_full"],  # thumb 必须与 palm family 绑定，因此这里只给 allegro thumb
#         "index": ["allegro_non_thumb_full", "leap_non_thumb_full"],  # non-thumb 允许跨 family，直接把 mixed candidate pool 写死在顶部
#         "middle": ["allegro_non_thumb_full", "leap_non_thumb_full"],
#         "ring": ["allegro_non_thumb_full", "leap_non_thumb_full"],
#     },
#     "single_palm_leap": {
#         "thumb": ["leap_thumb_full"],  # leap palm 同理只配 leap thumb
#         "index": ["allegro_non_thumb_full", "leap_non_thumb_full"],
#         "middle": ["allegro_non_thumb_full", "leap_non_thumb_full"],
#         "ring": ["allegro_non_thumb_full", "leap_non_thumb_full"],
#     },
# }  # 若真想炸完整 registry，再手动改回 `None`
CONNECTIVITY_PRESETS = None  # `None` = 自动展开 registry 里所有合法 slot-level recipe；否则只枚举这里指定的子集
HANDEDNESS: Literal["left", "right", "all"] = "all"  # `all` = 同时生成左右手；后续目录命名会显式带 `left_` / `right_`
MIXED = True  # 是否把 mixed-family topology 纳入 pre-made 主线
MISSING = True  # 是否把“缺失一根 non-thumb”的 topology 也纳入 pre-made 主线
RECOLORED: RecolorFacade = "anatomy_soft_v1"  # URDF visual recolor façade；默认使用低饱和 anatomy palette，避免 RGB 原色过艳
MAX_ENUMERATE: int | None = None  # `None` = 真正跑完整合法空间；小整数 = 先做 smoke-test / 局部巡检
ARTIFACT_LEVEL: Literal["hand_cfg", "urdf", "bundle"] = "bundle"  # quick façade 默认直接导出完整 bundle 便于人工巡检
OUTPUT_LAYOUT: Literal["flat", "recursive"] = "recursive"  # mixed / missing / connectivity 回溯时，递归布局更适合人工浏览
OUTPUT_DIR: Path = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "generated"  # 产物根目录仍沿用项目自己的 generated/
PRE_MADE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    pre_made=HandValidatorCfg.PreMadeCfg(
        finger_count_min=3,  # 当前 pre-made 主线允许 missing topology，因此最少保留 3 根手指
        require_non_thumb_with_min_revolute_dof=3,  # 至少保留 1 根 non-thumb finger 仍具有 >=3 个 revolute DOF
        check_palm_thumb_binding=True,  # mixed 时 thumb family 必须与 palm family 绑定
    )
)  # 显式写 `None` = 本次 quick 运行完全禁用 hand-level validator
PREMADE_PARALLEL = True  # pre-made 默认样本级并行：每个 worker 独立 build / validate / export
PREMADE_PARALLEL_WORKERS: int | None = None  # `None` = 按 CPU 自动推断；小整数可用于限制本机负载
PREMADE_PARALLEL_FALLBACK: Literal["serial", "raise"] = "serial"  # 并行环境异常时默认回退串行，优先保证科研产物落盘
_SHOW_REGISTRY = True  # 是否在真正生成前先打印 connectivity registry 摘要
_PRINT_RESULT_LIMIT: int | None = 40  # 终端最多 preview 多少条结果；`0` = 只看 summary，`None` = 打印全部


# 这是 quick.py 真正唯一的运行配置入口。
# `mode="made"` 是 quick.py 这条脚本入口的边界：它服务 pre-made，不在这里接 post-mutate。
RUN_CFG = HandGeneratorCfg(
    mode="made",  # quick.py 当前定位就是 pre-made 直接入口，不承担 post-mutate 编排
    artifact_level=ARTIFACT_LEVEL,  # hand_cfg / urdf / bundle 的导出粒度
    output_dir=OUTPUT_DIR,  # 产物根目录；后续 run-level 时间戳目录会在 generator 层继续展开
    handedness=HANDEDNESS,  # 当前要生成哪种 handedness；目录命名逻辑会在 generator 主线里真正消费它
    hand_presets=list(HAND_PRESETS),  # palm anchor 离散空间
    connectivity_presets=CONNECTIVITY_PRESETS,  # `None` 表示自动展开全部合法 slot-level connectivity
    mixed=MIXED,  # 是否允许 mixed-family topology
    missing=MISSING,  # 是否允许 missing-finger topology
    Validate=PRE_MADE_VALIDATOR_CFG,  # quick.py 顶部显式声明 pre-made validator；避免研究者看不出当前是否启用
    recolored=RECOLORED,  # visual recolor façade，透传给正式 generator/exporter
    output_layout=OUTPUT_LAYOUT,  # recursive / flat
    max_enumerate=MAX_ENUMERATE,  # 若不为 None，则用于快速 smoke-test
    premade_parallel=PREMADE_PARALLEL,  # pre-made 样本级并行开关；默认开启
    premade_parallel_workers=PREMADE_PARALLEL_WORKERS,  # 并行 worker 数；None 表示自动推断
    premade_parallel_fallback=PREMADE_PARALLEL_FALLBACK,  # 并行失败时的回退策略
)


# ============================================================================
#  摘要 / 打印辅助
# ============================================================================


def print_registry_summary(run_cfg: HandGeneratorCfg) -> None:
    r"""打印当前 quick façade 对应的 registry 摘要。

    quick.py 现在只面向 slot-level connectivity façade，因此这里直接打印：

    - 各 family / finger_kind 已注册的 finger-level recipe
    - 当前 quick.py 顶部大写变量 lower 后的有效值
    """

    print("=== actual finger-level connectivity recipes ===")
    for family in ("allegro", "leap"):
        for finger_kind in ("thumb", "non_thumb"):
            recipe_names = list_finger_connectivity_preset_names(family=family, finger_kind=finger_kind)
            print(f"{family}:{finger_kind} -> {list(recipe_names)}")
    print()

    print("=== effective quick façade knobs ===")
    print(f"hand_presets      = {run_cfg.hand_presets}")  # 当前 palm anchor 离散空间
    print(f"handedness       = {run_cfg.handedness}")  # 当前要生成哪种 handedness
    print(f"mixed            = {run_cfg.mixed}")  # 是否允许 mixed-family topology
    print(f"missing          = {run_cfg.missing}")  # 是否允许 missing-finger topology
    print(f"recolored        = {run_cfg.recolored}")  # 当前 visual recolor façade
    print(f"artifact_level   = {run_cfg.artifact_level}")  # hand_cfg / urdf / bundle
    print(f"output_layout    = {run_cfg.output_layout}")  # recursive / flat
    print(f"output_dir       = {run_cfg.output_dir}")  # 导出根目录
    print(f"max_enumerate    = {run_cfg.max_enumerate}")  # 若非 None，则是 smoke-test 上限
    print(f"premade_parallel = {run_cfg.premade_parallel}")  # 是否启用样本级并行
    print(f"parallel_workers = {run_cfg.premade_parallel_workers}")  # None 表示 generator 自动推断
    print(f"parallel_fallback= {run_cfg.premade_parallel_fallback}")  # 并行异常时 serial / raise
    print(f"connectivity_cfg = {run_cfg.connectivity_presets}")  # `None` 表示自动展开全部合法 slot-level recipe
    print(f"validator_on     = {run_cfg.Validate is not None}")  # 让研究者一眼看出 hand-level validator 是否启用
    if run_cfg.Validate is not None:
        print(f"pre_made.finger_count_min = {run_cfg.Validate.pre_made.finger_count_min}")  # pre-made 最少手指数
        print(
            "pre_made.require_non_thumb_with_min_revolute_dof = "
            f"{run_cfg.Validate.pre_made.require_non_thumb_with_min_revolute_dof}"
        )  # pre-made 至少一根 non-thumb 的最小剩余 revolute DOF
        print(f"pre_made.check_palm_thumb_binding = {run_cfg.Validate.pre_made.check_palm_thumb_binding}")  # mixed 中 thumb/palm family 绑定
    print()


def _result_preview_line(index: int, result: Any) -> str:
    r"""把一条 `HandGenerationResult` 压成适合终端快速扫读的一行。"""

    topology_kind = str(result.metadata.get("topology_kind", "unknown"))  # single / missing / mixed
    topology_name = str(result.metadata.get("topology_name", "-"))  # 当前 topology 的显式 provenance 名
    connectivity_name = str(result.metadata.get("connectivity_preset", "-"))  # 当前 connectivity 选择名
    urdf_path = str(result.urdf_path) if result.urdf_path is not None else "(hand_cfg only)"
    return f"[{index:04d}] {topology_kind} | {topology_name} | {connectivity_name} | {urdf_path}"


def print_result_summary(results: list[Any], run_cfg: HandGeneratorCfg) -> None:
    r"""打印本次 quick 运行的摘要与结果 preview。"""

    topology_counter = Counter(str(result.metadata.get("topology_kind", "unknown")) for result in results)
    base_hand_counter = Counter(str(result.metadata.get("base_hand_preset", "-")) for result in results)

    print(f"generated {len(results)} bundles under {run_cfg.output_dir}")
    print(f"topology counts: {dict(topology_counter)}")  # 统计 single / missing / mixed 各自产物数量
    print(f"base-hand counts: {dict(base_hand_counter)}")  # 统计每个 palm anchor 对应产物数量

    if _PRINT_RESULT_LIMIT == 0:
        return  # 用户显式要求只看 summary，不逐条 preview

    preview_results = results
    if _PRINT_RESULT_LIMIT is not None:
        preview_results = results[:_PRINT_RESULT_LIMIT]

    print("=== result preview ===")
    for index, result in enumerate(preview_results, start=1):
        print(_result_preview_line(index, result))

    if _PRINT_RESULT_LIMIT is not None and len(results) > len(preview_results):
        print(f"... {len(results) - len(preview_results)} more results omitted from terminal preview")


# ============================================================================
#  运行主线
# ============================================================================


def enumerate_premade_bundles(run_cfg: HandGeneratorCfg) -> list[Any]:
    r"""执行 quick façade 对应的正式 pre-made 批量生成。

    # NOTE:
    这个函数名保留了历史上的 `enumerate_premade_bundles`，因为 quick.py 当前主用法
    仍然是 pre-made 枚举巡检；但这里不再偷偷 lower 一份新的 cfg，而是直接消费用户给的
    `HandGeneratorCfg`。
    """

    if run_cfg.mode != "made":
        raise ValueError("quick.py currently only supports pre-made generation; set run_cfg.mode='made'.")

    generator = HandGenerator(run_cfg)  # 真正执行完全委托正式 `HandGenerator`，不再引入第二层包装 cfg
    return list(generator.generate_batch())  # quick 场景下直接收成列表，更方便 notebook / 人工检查


def main(run_cfg: HandGeneratorCfg | None = None) -> int:
    r"""脚本主入口。

    Args:
        run_cfg (HandGeneratorCfg | None): 若为 `None`，则使用顶部全局 `RUN_CFG`；
            这保证了两种使用方式都顺手：

            1. 终端直接 `python quick.py`
            2. notebook / 测试里显式构造一个更小的 `HandGeneratorCfg`

    Returns:
        int: Unix 风格退出码；成功时返回 0。
    """

    effective_cfg = run_cfg or RUN_CFG  # direct-run 时默认吃顶部可编辑全局配置
    if _SHOW_REGISTRY:
        print_registry_summary(effective_cfg)  # 先打印 registry 和关键开关，避免生成后才发现空间开大了

    results = enumerate_premade_bundles(effective_cfg)  # 真正执行 pre-made 全量 / 子集枚举
    print_result_summary(results, effective_cfg)  # 末尾给出摘要与少量结果 preview
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
