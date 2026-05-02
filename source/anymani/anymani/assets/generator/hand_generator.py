r"""手部资产生成器主入口草案。

本文件是你想要的“用户真正直接面对的主接口”草案：

- 默认语义是完整流程：`made -> mutate -> validate -> export`
- 但每个阶段都必须可选，用户可以只做 `made`、只做 `mutate`
- 产物粒度也必须可控：可以只拿 `HandCfg`，也可以一步到位导出 `URDF`

设计说明
--------

### 核心定位

`HandGenerator` 不是另一个 builder，而是一个面向用户的 façade。
它负责调度前序生成、后序变异、校验和导出，并把各阶段产物组织成
一个轻量结果包，方便交互式使用和批量实验。

### 与 builder / mutator 的关系

- `made` 阶段优先复用 `builder` 体系，负责“造骨架”
- `mutate` 阶段使用 `mutate/` 子包里的工具，负责“在已有手上派生”
- `validate` 与 `export` 仍保留独立职责，不内嵌到变异工具里

### 对用户的使用体验

用户应该可以把这层当作主要入口，只配置少量参数和 recipe，就能：

1. 生成一个初始 `HandCfg`
2. 可选地对其做局部变异
3. 可选地立即导出 URDF / sidecar 产物
4. 也可以先停在轻量产物阶段，之后再接人工微调
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

import yaml

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..exporter import HandExporter, HandExporterCfg
from ..validator import HandValidator, HandValidatorCfg
from ._generation_result import HandGenerationResult
from ._post_mutate_restore import PostMutateSource, load_post_mutate_source
from ._premade import (
    apply_connectivity_preset as _apply_premade_connectivity_preset,
    build_base_hand as _build_premade_base_hand,
    candidate_hand_preset_names as _candidate_premade_hand_preset_names,
    connectivity_names_for_hand_preset as _connectivity_names_for_premade_hand_preset,
    normalize_connectivity_mapping,
    normalize_name_list,
    resolve_export_root as _resolve_premade_export_root,
    resolve_single_premade_selection as _resolve_single_premade_selection,
    stable_premade_id,
)
from ._recolor import RecolorSpec, describe_recolor_spec, normalize_recolor_spec, resolve_visual_recolor_materials

try:
    from .mutate import HandMutator, HandMutatorCfg, sample_scalar_distribution
except Exception:
    @dataclass
    class HandMutatorCfg(AssetCfgBase):
        r"""Fallback mutate cfg used when the mutate package is unavailable.

        The first implementation slice does not execute post-mutate logic, but the
        generator cfg still keeps the field so the public interface remains stable.
        """

        order: tuple[str, ...] = ()
        on_reject: Literal["abort", "skip"] = "abort"
        step_validate: bool = False
        terms: dict[str, object] = field(default_factory=dict)

        def has_terms(self) -> bool:
            return bool(self.terms)

    class HandMutator:
        r"""Fallback mutator used when the mutate package is unavailable."""

        def __init__(self, cfg: HandMutatorCfg):
            self.cfg = cfg

        def describe_sampling(self, target: HandCfg) -> dict[str, dict[str, Any]]:
            return {}

        def mutate(self, target: HandCfg, *, sampled_params: dict[str, dict[str, Any]] | None = None) -> HandCfg | None:
            raise NotImplementedError("mutate runtime is unavailable in the current environment")

    def sample_scalar_distribution(cfg, *, rng=None):
        raise NotImplementedError("mutate sampling runtime is unavailable in the current environment")


def _has_enabled_mutation(cfg: HandMutatorCfg) -> bool:
    r"""Check whether any post-mutate tool is enabled in the cfg."""

    return cfg.has_terms()


def _sample_mutation_terms(mutator: HandMutator, target: HandCfg) -> dict[str, dict[str, float]]:
    r"""按 mutator 描述的独立联合分布为当前 hand 采样一组 term 参数。"""

    sampled_terms: dict[str, dict[str, float]] = {}
    for term_name, distribution_map in mutator.describe_sampling(target).items():
        sampled_terms[term_name] = {
            local_name: sample_scalar_distribution(distribution_cfg)
            for local_name, distribution_cfg in distribution_map.items()
        }
    return sampled_terms


@dataclass(frozen=True)
class _PremadeTask:
    r"""一个可独立并行执行的 pre-made 离散样本任务。

    pre-made 的第一性原理是离散组合枚举，而不是连续参数采样：

    $$
    \mathcal{D}_{\text{pre}} =
    \{(\text{topology}_i,\ \text{connectivity}_j)\}
    $$

    因而每个元素天然互不依赖，可以交给一个 worker 完整执行
    `build -> validate -> export`。这就是工程上的 task/data parallelism，
    不是把单个 hand 内部的 validator 规则拆碎并行。
    """

    hand_preset_name: str
    """内部 premade topology registry key。"""

    connectivity_preset_name: str
    """当前 topology 下的 connectivity selection 名。"""

    enumerated: bool
    """是否使用稳定 sample id；纯 pre-made 枚举应为 True。"""


@dataclass
class _PremadeWorkerResult:
    r"""worker 返回给主进程的最小结果包。

    worker 可以写 URDF / sidecar / tree，但不能写 run-level `summary.yaml`。
    summary 是全局统计文件，必须由主进程单点串行维护，否则并行写 YAML 会产生
    竞争条件，也会让拒绝统计难以复现。
    """

    result: HandGenerationResult | None
    """成功时返回完整生成结果；失败/拒绝时为 None。"""

    rejection_stage: str | None = None
    """validator / mutate 拒绝阶段；成功时为 None。"""


def _generate_premade_worker(
    cfg: "HandGeneratorCfg",
    run_root: Path | str,
    task: _PremadeTask,
) -> _PremadeWorkerResult:
    r"""在独立 worker 中执行一个 pre-made 样本任务。

    Args:
        cfg (HandGeneratorCfg): 主进程传入的生成配置快照。
        run_root (Path | str): 主进程已经创建好的 run 根目录。
        task (_PremadeTask): 当前离散 topology/connectivity 样本。

    Returns:
        _PremadeWorkerResult: 成功结果或拒绝阶段。
    """

    worker_generator = HandGenerator(cfg)  # worker 内部独立持有 runtime façade，避免共享可变状态
    worker_generator._run_root = Path(run_root)  # 导出目录必须与主 run 对齐
    worker_generator._run_summary = {"stats": {"by_topology": {}}}  # 占位 summary；worker 不会写全局 YAML
    worker_generator._last_rejection_stage = None  # 记录本样本被哪个阶段拒绝，供主进程汇总
    result = worker_generator._generate_once(
        hand_preset_name=task.hand_preset_name,
        connectivity_preset_name=task.connectivity_preset_name,
        enumerated=task.enumerated,
        record_summary=False,
    )
    return _PremadeWorkerResult(result=result, rejection_stage=worker_generator._last_rejection_stage)


# ============================================================================
#  生成器配置
# ============================================================================


@dataclass
class HandGeneratorCfg(AssetCfgBase):
    r"""整手生成器配置。

    这个 cfg 面向用户主入口，采用“默认一整套流程，但各阶段可选”的原则。
    默认情况下，用户可以把它理解为一个完整 recipe；如果想只做某一段，
    也可以通过 `mode` 与 `artifact_level` 显式收缩工作范围。

    # NOTE:
    关于手部 preset 资产的 pre-made façade，这里按最新讨论收敛为**仅两个字段**：

    1. `hand_presets: list[str]`
       指定需要参与 pre-made 生成的 base hand preset 名列表，例如：
       `["single_palm_allegro", "single_palm_leap"]`。
    2. `connectivity_presets: dict[str, list[str]] | None`
       指定“某个 hand preset 允许搭配哪些 connectivity preset”。

    这比上一版 `hand_preset / connectivity_preset / hand_preset_names /
    connectivity_preset_names` 的四字段 façade 更直率，也更贴近你的心流：

    - 先选 canonical hand preset；
    - 再为它列出允许的 connectivity 变体；
    - `sample` 时从这个离散空间随机抽；
    - `enumerate` 时显式遍历它的笛卡尔积。

    **默认行为：**
    若 `connectivity_presets is None`，或者字典里缺少某个 `hand_preset` 的键，
    则自动回退为“该 hand 所属 family 下全部已注册的合法 connectivity preset”。
    """

    class_type: type["HandGenerator"] | None = None
    """关联的运行时类。"""

    mode: Literal["made", "mutate", "full"] = "full"
    """执行模式。`made` 只做前序生成，`mutate` 只做后序变异，`full` 走整套流程。"""

    artifact_level: Literal["hand_cfg", "urdf", "bundle"] = "bundle"
    """产物粒度。`hand_cfg` 只返回轻量结构，`urdf` 侧重落盘，`bundle` 同时保留多种产物。"""

    n_samples: int = 1
    """post-mutate 阶段的 Monte Carlo 采样预算。

    # NOTE:
    当前批处理语义已经固定，不再暴露单独的 `sampling_strategy`：

    - pre-made：离散笛卡尔展开
    - post-mutate：对每个 pre-made topology 采样 `n_samples` 个后序变体
    """

    max_enumerate: int | None = None
    """pre-made 笛卡尔展开的最大产物数上限；为 ``None`` 时不截断。"""

    premade_parallel: bool = True
    """是否默认对 pre-made 离散样本启用样本级并行。

    这里的“并行”特指 task/data parallelism：

    - 一个 worker 负责一个 `(topology, connectivity)` 样本；
    - worker 内部顺序完成 `build -> pre-made validator -> export`；
    - 主进程只做任务分发与 `summary.yaml` 汇总。

    # NOTE:
    这不是 GPU 并行。pre-made 主耗时来自 Python 对象构建、离散规则检查和文件写出，
    并不是可以一次性搬到 CUDA 上的连续张量计算；GPU 更适合 post-mutate 的联合
    Monte Carlo 参数采样。
    """

    premade_parallel_workers: int | None = None
    """pre-made 样本级并行 worker 数。

    为 ``None`` 时按 CPU 数自动推断，但会保留一个核心给系统/交互环境，避免全量生成时
    把本机完全打满导致 IDE 或仿真环境卡死。
    """

    premade_parallel_fallback: Literal["serial", "raise"] = "serial"
    """pre-made 并行路径失败时的处理策略。

    - ``"serial"``：回退到原串行枚举路径，优先保证科研流程能产出；
    - ``"raise"``：直接抛出并行异常，便于专门排查 worker / pickle / 导出错误。
    """

    post_mutate_max_attempt_factor: int = 10
    """post-mutate validator 补采预算系数。

    若目标成功数为 $N$，最多允许尝试 $N\times f$ 个候选。这样可以实现
    “目标 100、首轮成功 94、下一轮补 6” 的缺口补采，同时避免 validator
    拒绝率异常时无限循环。
    """

    Made: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """前序生成配置入口；主要负责关节拓扑维度的变体，把生成空间中的选择落到一个初始 `HandCfg`。"""

    Mutate: HandMutatorCfg = field(default_factory=HandMutatorCfg)
    """后序变异配置入口；主要负责非关节拓扑维度的变体，可为空操作，也可串联多个局部工具。"""

    Validate: HandValidatorCfg | None = None
    """手级验证配置入口；为 ``None`` 时显式禁用 hand-level validator。

    # NOTE:
    这里把 validator 语义从“总是隐式开启”改成“显式声明才启用”，原因不是弱化合法性，
    而是把控制权重新交还给研究者：

    - quick.py 这类科研入口应在顶部直接写出“本次到底启没启 validator”
    - 若用户正在做 topology / exporter / schema 的局部排查，也需要一条完全跳过
      hand-level validator 的直通路径
    """

    Export: HandExporterCfg = field(default_factory=HandExporterCfg)
    """手级导出器配置入口；用于把 HandCfg 导出为 URDF / sidecar / tree 文件等产物。"""

    output_dir: Path | str = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "generated")
    """产物落盘根目录。

    默认写到 `assets/generated/`，与当前子项目的目录约定保持一致；
    测试或批量脚本也可以显式覆盖成临时目录。
    """

    source_topology_dir: Path | str | None = None
    """独立 post-mutate 的来源 topology 目录。

    只在 `mode="mutate"` 时生效，约定输入形状固定为：

    `.../generated/<timestamp>/<group>/<topology_name>/`

    # NOTE:
    首版 mutate-only 入口故意不接受“整个时间戳目录”或“单个样本目录”，
    因为用户已经把工作流钉死为：

    - 传入 topology 目录；
    - 自动找到唯一 pre-made 原始样本；
    - 首次运行时把它改名为 `*_origin`；
    - 新 post-mutate 样本作为同级兄弟目录继续写入。
    """

    handedness: Literal["left", "right", "all"] = "all"
    """生成哪种 handedness 的手。`all` 表示同时生成左右手；`left` / `right` 则只生成单一 handedness。"""

    hand_presets: list[str] = field(default_factory=list)
    """pre-made 阶段参与生成的 base hand preset 名列表。

    # NOTE:
    这里显式保留 `list[str]` 形状，而不是转成 tuple，
    因为它本身就是“用户手写离散列表”的语义对象。
    """

    connectivity_presets: dict[str, dict[str, list[str]]] | None = None
    """pre-made connectivity façade。

    当前只支持唯一一种直观形状：

    `hand_preset -> {slot -> [finger_connectivity_preset_name, ...]}`

    这里的科研语义就是：

    - 给定一只 base hand；
    - 直接声明它每个 slot 允许使用哪些**已注册手指 connectivity 资产**。

    例如：

    ```python
    connectivity_presets = {
        "single_palm_allegro": {
            "thumb": ["allegro_thumb_full"],
            "index": ["allegro_non_thumb_full", "allegro_non_thumb_drop_j3"],
            "middle": ["allegro_non_thumb_full"],
            "ring": ["allegro_non_thumb_full"],
        }
    }
    ```

    若为 ``None``，则每个 surviving slot 自动展开该 slot family 下全部已注册合法 recipe。
    """

    mixed: bool = True
    """是否混合不同 family 的手指拓扑。如果为 True，则在 pre-made 阶段允许在同一只手上组合 leap/allegro 的手指变体；如果为 False，则默认每只手只能选一个 family 的 preset 进行派生。"""

    missing: bool = True
    """是否把“缺失一根 non-thumb”的 topology 纳入 pre-made。

    你已明确要求 missing topology 属于 pre-made 主线的一部分，因此这里默认开启；
    若旧脚本只想保留 canonical single-family 空间，可显式写 `missing=False`。
    """

    recolored: RecolorSpec = None
    """控制 URDF visual recolor 的 façade 字段。

    已确认的 contract 如下：

    - `None` / `False`：关闭 recolor
    - `str`：命名 palette，名称来自 `assets/presets/color_presets.py`
    - `dict[child_link_name, rgba]`：按 child link 名做局部覆盖

    注意两点：

    1. recolor 只作用于 `<visual>`，不会改 `<collision>`；
    2. 命名 palette 的 anatomy 规则为：
       - palm / LEAP `root_fixed_link`：红
       - CMC1 / MCP1：黄
       - CMC2 / MCP2：青
       - PIP：绿
       - DIP：蓝
       - TIP：紫
    """

    output_layout: Literal["flat", "recursive"] = "recursive"
    """pre-made 产物的目录组织模式。

    - ``recursive``：`generated/<timestamp>/{group}/{topology}/{sample_id}/`
    - ``flat``：`generated/<timestamp>/flat/{sample_id}/`

    # NOTE:
    这个字段仍然收口在 `HandGeneratorCfg`，因为用户已经明确要求：
    `HandGeneratorCfg` 才是生成资产时的唯一 façade，不再额外包装新的 runner。
    """

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandGenerator
        self.output_dir = Path(self.output_dir)  # 统一在 cfg 边界内把路径收口为 `Path`
        if self.source_topology_dir is not None:
            self.source_topology_dir = Path(self.source_topology_dir)
        self.hand_presets = normalize_name_list(self.hand_presets, field_name="hand_presets")
        self.connectivity_presets = normalize_connectivity_mapping(self.connectivity_presets)
        self.recolored = normalize_recolor_spec(self.recolored)
        if self.premade_parallel_workers is not None and self.premade_parallel_workers < 1:
            raise ValueError("premade_parallel_workers must be >= 1 when provided")
        if self.premade_parallel_fallback not in {"serial", "raise"}:
            raise ValueError("premade_parallel_fallback must be either 'serial' or 'raise'")

        # pre-made façade 一旦显式给出 `connectivity_presets`，就必须同时给出 hand preset 列表；
        # 否则运行时连“这张映射是给谁的”都无法确定。
        if self.connectivity_presets is not None and not self.hand_presets:
            raise ValueError("connectivity_presets requires hand_presets to be provided together")

        # `Made` 作为 concrete builder cfg 的 override 能力仍保留给单样本 preview / 局部实验，
        # 但若 hand_presets 本身要枚举多个 canonical hand，就不应再让同一个 concrete `Made`
        # 同时伪装成多个不同 base hand。
        if len(self.hand_presets) > 1 and self.Made.class_type is not HandBuilder:
            raise ValueError(
                "When hand_presets contains multiple base hand presets, Made must stay abstract; "
                "otherwise one concrete builder cfg would be incorrectly reused for all preset anchors."
            )
        if self.mode == "mutate" and self.source_topology_dir is None:
            raise ValueError("mode='mutate' requires 'source_topology_dir' to point at one topology directory")
        if self.mode != "mutate" and self.source_topology_dir is not None:
            raise ValueError("'source_topology_dir' is only valid when mode='mutate'")
        if self.mode == "mutate" and not _has_enabled_mutation(self.Mutate):
            raise ValueError("mode='mutate' requires at least one enabled mutator term")


# ============================================================================
#  生成器运行时壳
# ============================================================================


class HandGenerator:
    r"""整手生成器主入口。

    这里的职责是把 `Made`、`Mutate`、`Validate`、`Export` 按用户指定的
    模式串起来，并把结果组织为一个可交互、可回写的轻量结果包。
    """

    cfg: HandGeneratorCfg

    def __init__(self, cfg: HandGeneratorCfg):
        self.cfg = cfg
        self._run_root: Path | None = None
        self._run_summary: dict[str, Any] | None = None
        self._mutate_source: PostMutateSource | None = None
        self._last_rejection_stage: str | None = None

    def _ensure_run_context(self) -> tuple[Path, dict[str, Any]]:
        r"""为当前 generator 实例懒创建一次 run 根目录与 summary 文档。

        pre-made / full 与 mutate-only 的根目录语义不同：

        - pre-made / full：每次运行都新建 `generated/<timestamp>/`
        - mutate-only：直接复用用户给定的 topology 目录，在该目录下写 summary
        """

        if self._run_root is not None and self._run_summary is not None:
            return self._run_root, self._run_summary

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.cfg.mode == "mutate":
            if self.cfg.source_topology_dir is None:  # 防御式分支；正常情况下已在 cfg 期校验掉
                raise ValueError("mode='mutate' requires 'source_topology_dir'")
            run_root = Path(self.cfg.source_topology_dir)
            run_root.mkdir(parents=True, exist_ok=True)
        else:
            output_root = Path(self.cfg.output_dir)
            run_root = output_root / timestamp

            collision_index = 2
            while run_root.exists():
                run_root = output_root / f"{timestamp}_{collision_index:02d}"
                collision_index += 1  # 同秒重复启动时追加后缀，避免不同 run 相互覆盖

            run_root.mkdir(parents=True, exist_ok=False)

        from ._recipe_loader import RecipeLoader

        pre_made_enabled = self.cfg.mode in {"made", "full"}
        post_mutate_enabled = self.cfg.mode in {"mutate", "full"} and _has_enabled_mutation(self.cfg.Mutate)
        self._run_root = run_root
        self._run_summary = {
            "run": {
                "timestamp": timestamp,
                "root_dir": str(run_root),
                "mode": self.cfg.mode,
                "artifact_level": self.cfg.artifact_level,
                "phases": {
                    "pre_made": pre_made_enabled,
                    "post_mutate": post_mutate_enabled,
                    "combined": pre_made_enabled and post_mutate_enabled,
                },
            },
            "config": RecipeLoader.dump(self.cfg),
            "stats": {
                "attempted": 0,
                "succeeded": 0,
                "rejected": 0,
                "rejected_by_stage": {},
                "by_topology": {},
            },
        }
        self._write_run_summary()
        return self._run_root, self._run_summary

    def _load_mutate_source(self) -> PostMutateSource:
        r"""懒加载独立 post-mutate 的来源样本。

        这里缓存的是“同一个 topology 目录里的唯一 pre-made 原点”，这样一轮
        `n_samples=20` 的 Monte Carlo 采样不会重复做 20 次磁盘扫描与 YAML 解析。
        """

        if self._mutate_source is not None:
            return self._mutate_source
        if self.cfg.source_topology_dir is None:
            raise ValueError("Independent post-mutate requires 'source_topology_dir'")
        self._mutate_source = load_post_mutate_source(self.cfg.source_topology_dir)
        return self._mutate_source

    def _write_run_summary(self) -> None:
        r"""把当前 run summary 刷到 `<run_root>/summary.yaml`。"""

        if self._run_root is None or self._run_summary is None:
            return

        stats = self._run_summary["stats"]
        stats["topology_count"] = len(stats["by_topology"])
        summary_path = self._run_root / "summary.yaml"
        summary_path.write_text(
            yaml.safe_dump(self._run_summary, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _result_topology_key(self, result: HandGenerationResult) -> str:
        r"""把单个结果映射成 summary 里的 topology 路径键。"""

        topology_name = str(result.metadata.get("topology_name") or result.metadata.get("family") or "unknown_topology")
        topology_group_name = str(
            result.metadata.get("topology_group_name")
            or result.metadata.get("base_hand_preset")
            or result.metadata.get("family")
            or "ungrouped"
        )
        topology_kind = str(result.metadata.get("topology_kind") or "single_family")
        if topology_kind == "mixed":
            return f"mixed/{topology_group_name}/{topology_name}"
        return f"{topology_group_name}/{topology_name}"

    def _record_generation_rejection(self, *, stage: str, write_summary: bool = True) -> None:
        r"""把一次被拒绝的样本尝试写入 run summary。"""

        self._last_rejection_stage = stage

        _, summary = self._ensure_run_context()
        stats = summary["stats"]
        stats["attempted"] += 1
        stats["rejected"] += 1
        rejected_by_stage = dict(stats.get("rejected_by_stage") or {})
        rejected_by_stage[stage] = int(rejected_by_stage.get(stage, 0)) + 1
        stats["rejected_by_stage"] = rejected_by_stage
        if write_summary:
            self._write_run_summary()

    def _record_generation_success(self, result: HandGenerationResult, *, write_summary: bool = True) -> None:
        r"""把一次成功样本写入 run summary。"""

        self._last_rejection_stage = None

        _, summary = self._ensure_run_context()
        stats = summary["stats"]
        stats["attempted"] += 1
        stats["succeeded"] += 1

        topology_key = self._result_topology_key(result)
        by_topology = dict(stats.get("by_topology") or {})
        by_topology[topology_key] = int(by_topology.get(topology_key, 0)) + 1
        stats["by_topology"] = by_topology
        if write_summary:
            self._write_run_summary()

    def _candidate_hand_preset_names(self) -> tuple[str, ...]:
        r"""返回当前 generator 可见的 premade topology registry key 集合。

        # NOTE:
        用户在 cfg 里仍然只写 base `hand_presets`，但运行时真正被枚举的是：

        - base hand preset
        - handedness
        - missing / mixed topology 扩展

        共同形成的内部 topology registry key。这样 `enumerate` 才能稳定覆盖
        左右手与各类 pre-made 拓扑，而不把它们错误压扁成同一个名字。
        """

        return _candidate_premade_hand_preset_names(self.cfg)

    def _connectivity_names_for_hand_preset(self, *, hand_preset_name: str) -> tuple[str, ...]:
        r"""返回某个 premade topology registry key 允许搭配的 connectivity 名集合。

        规则直接对应 `HandGeneratorCfg.connectivity_presets` 的科研语义：

        1. 若字典为 `None`，则默认展开该 family 下全部已注册 connectivity；
        2. 若字典缺少当前 hand preset 的键，同样默认展开该 family 全部 connectivity；
        3. 若字典显式给了该 hand 的列表，则严格采用该列表；
        4. 无论是哪条路径，最终都要再次校验 family 一致性，避免手 preset 与
           connectivity preset 的跨 family 错配。
        """

        return _connectivity_names_for_premade_hand_preset(self.cfg, hand_preset_name=hand_preset_name)

    def _resolve_single_premade_selection(self) -> tuple[str | None, str | None] | None:
        r"""为 `generate()` 的单样本路径解析本次要使用的 pre-made 选择。

        这一版的 pre-made façade 已经非常明确：

        - `hand_presets` 给出 canonical base hand 离散空间；
        - `connectivity_presets` 给出每个 base hand 允许搭配的 connectivity 列表。

        因而这里的 sample 语义也非常直接：

        1. 先从 `hand_presets` 中随机抽一个 base hand；
        2. 再从这个 hand 对应的 connectivity 列表中随机抽一条；
        3. 两者组成一次 pre-made 样本。
        """

        return _resolve_single_premade_selection(self.cfg)

    def _build_base_hand(self, *, hand_preset_name: str | None) -> tuple[HandCfg, str]:
        r"""构建本次样本的 canonical base hand。

        base hand 的来源按以下优先级收敛：

        1. 若 `Made` 已经是具体 builder cfg，则优先使用它；
        2. 否则若给了 `hand_preset_name`，就从 hand preset 解析出 builder cfg；
        3. 两者都没有时，说明当前 cfg 既没有具体 `Made`，也没有 pre-made hand preset，
           这在运行时应视为无效输入。

        这样做的动机，是同时支持两条工作流：

        - 正式 pre-made：`hand_preset -> canonical hand`
        - 科研局部实验：当 `hand_presets` 只有一个锚点时，`Made` 仍可作为局部覆写后的
          concrete builder cfg，帮助你在不改 hand preset 名称的前提下快速调试
        """

        return _build_premade_base_hand(self.cfg, hand_preset_name=hand_preset_name)

    def _apply_connectivity_preset(
        self,
        hand_cfg: HandCfg,
        *,
        connectivity_preset_name: str,
        hand_preset_name: str | None,
    ) -> tuple[HandCfg, dict[str, Any]]:
        r"""把 hand-level connectivity preset lower 成显式的 joint delete + regroup 结果。

        这里刻意采用“两层语义分离”：

        - 合法 recipe 在 `assets/presets/connectivity_presets.py`
        - 真正执行删除/重连的 runtime 在 `mutate/joint_delete.py`

        也就是说，本函数本质上做的是：

        $$\text{legal connectivity preset} \xrightarrow{\text{lower}} \text{per-finger deleted joint set} \xrightarrow{\text{JointDeleteMutator(drop)}} \text{new HandCfg}$$

        # NOTE:
        对当前 joint-centric 图建模而言，这里的 delete 语义必须理解成：
        **删 joint = 删这个 joint 对应的 child-link 几何节点。**
        因而 pre-made connectivity 主线默认使用 `drop`，
        而不是把被删段 mesh merge 回上游节点。
        
        """

        return _apply_premade_connectivity_preset(
            self.cfg,
            hand_cfg,
            connectivity_preset_name=connectivity_preset_name,
            hand_preset_name=hand_preset_name,
        )

    def _resolve_export_root(self, *, result: HandGenerationResult) -> Path:
        r"""根据 pre-made provenance 与 `output_layout` 计算本次导出的根目录。

        当前导出器仍保持它一贯的职责边界：

        - `HandExporter` 负责在传入目录下再补一层 `{sample_id}/`
        - `HandGenerator` 负责决定这个“传入目录”到底应该是平铺还是递归层级

        这样可以在不破坏现有 exporter 结构的前提下，把目录语义仍然收口到
        `HandGeneratorCfg` 这个唯一 façade。
        """

        run_root, _ = self._ensure_run_context()
        if self.cfg.mode == "mutate":
            return run_root
        return _resolve_premade_export_root(self.cfg, result=result, run_root=run_root)

    def _generate_once(
        self,
        *,
        hand_preset_name: str | None,
        connectivity_preset_name: str | None,
        enumerated: bool = False,
        sampled_mutation_terms: dict[str, dict[str, float]] | None = None,
        record_summary: bool = True,
    ) -> HandGenerationResult | None:
        r"""执行一次单样本生成；供 `generate()` 与 `generate_batch()` 共同复用。

        这个内部 helper 的价值，是把：

        - 单样本 `generate()`
        - 枚举式 `generate_batch()`

        这两条路径共享到同一套 build / connectivity / mutate / validate / export
        语义上，而不是各写一份相似但悄悄分叉的实现。
        """

        self._ensure_run_context()  # 一次 generator 实例对应一次 run；首次尝试时就固定时间戳根目录

        validator = HandValidator(self.cfg.Validate) if self.cfg.Validate is not None else None
        validation_warnings: list[str] = []

        if self.cfg.mode == "mutate":
            mutate_source = self._load_mutate_source()
            hand_cfg = mutate_source.hand_cfg.copy()  # 每个 post-mutate 样本都从同一 pre-made 原点重新起步
            builder_cfg_name = str(mutate_source.metadata.get("builder_cfg", "restored_hand_cfg"))
            premade_metadata = dict(mutate_source.metadata)
        else:
            hand_cfg, builder_cfg_name = self._build_base_hand(hand_preset_name=hand_preset_name)

            premade_metadata: dict[str, Any] = {}
            if connectivity_preset_name is not None:
                hand_cfg, premade_metadata = self._apply_connectivity_preset(
                    hand_cfg,
                    connectivity_preset_name=connectivity_preset_name,
                    hand_preset_name=hand_preset_name,
                )

            # pre-made 结构闸门是“可选的显式闸门”：
            # - `Validate is None`：完全跳过 hand-level validator；
            # - 否则：在 connectivity lower 之后、mutate / export 之前执行。
            if validator is not None:
                pre_made_validation = validator.validate_pre_made(hand_cfg)
                if not pre_made_validation:
                    self._record_generation_rejection(stage="pre_made_validate", write_summary=record_summary)
                    return None  # pre-made 结构闸门拒绝后，不再允许继续进入 mutate / export
                validation_warnings.extend(pre_made_validation.warnings)
        sampled_terms: dict[str, dict[str, float]] | None = None

        # 后序派生的两种合法入口：
        #
        # 1. `mode="full"`：在 pre-made 基座上继续采样后序变体；
        # 2. `mode="mutate"`：从 sidecar `hand_cfg` 快照恢复后再采样。
        if self.cfg.mode in {"full", "mutate"} and _has_enabled_mutation(self.cfg.Mutate):
            mutator = HandMutator(self.cfg.Mutate)
            sampled_terms = sampled_mutation_terms or _sample_mutation_terms(mutator, hand_cfg)
            hand_cfg = mutator.mutate(hand_cfg, sampled_params=sampled_terms)
            if hand_cfg is None:
                self._record_generation_rejection(stage="mutate", write_summary=record_summary)
                return None  # 变异被拒绝；拒绝语义统一表现为“本次样本无结果”
            if validator is not None:
                post_mutate_validation = validator.validate_post_mutate(hand_cfg)
                if not post_mutate_validation:
                    self._record_generation_rejection(stage="post_mutate_validate", write_summary=record_summary)
                    return None
                validation_warnings.extend(post_mutate_validation.warnings)

        sample_id = uuid4().hex[:8]
        if self.cfg.mode != "mutate" and connectivity_preset_name is not None and enumerated:
            sample_id = stable_premade_id(
                hand_preset_name or hand_cfg.family,
                connectivity_preset_name,
            )

        metadata = {
            "id": sample_id,  # 8 位短 ID，sample 路径默认随机，enumerate 路径按 recipe 稳定化
            "builder_cfg": builder_cfg_name,  # 记录 base hand 最终使用的 builder cfg 类型；mutate-only 则标记为恢复来源
            "warnings": validation_warnings,  # 汇总 pre-made / post-mutate 两阶段 warning，保留给 sidecar / 调试消费
            "family": hand_cfg.family,
            "handedness": hand_cfg.handedness,
        }
        metadata.update({key: value for key, value in premade_metadata.items() if value is not None})
        if sampled_terms:
            metadata["post_mutate_samples"] = sampled_terms
        metadata["output_layout"] = self.cfg.output_layout
        recolor_metadata = describe_recolor_spec(self.cfg.recolored)
        if recolor_metadata is not None:
            metadata["recolored"] = recolor_metadata

        result = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata=metadata,
        )

        # `artifact_level="hand_cfg"` 表示用户只想拿内存中的 hand schema；
        # 其余两档则交给 exporter 负责落盘。
        if self.cfg.artifact_level != "hand_cfg":
            resolved_recolor_materials = resolve_visual_recolor_materials(hand_cfg, self.cfg.recolored)
            export_cfg = self.cfg.Export.replace(
                artifact_level=self.cfg.artifact_level,  # 把主入口的粒度选择下传给 exporter
                Urdf=self.cfg.Export.Urdf.replace(
                    recolored_materials=resolved_recolor_materials,
                ),
            )
            exporter = HandExporter(export_cfg)  # 导出器负责 URDF / sidecar / tree 文件
            exporter.export(result, output_dir=self._resolve_export_root(result=result))  # 目录布局仍由 HandGenerator façade 决定

        self._record_generation_success(result, write_summary=record_summary)
        return result

    def generate(self) -> HandGenerationResult | None:
        r"""执行一次整手资产生成。

        当前这条主路径已经实现的是：

        1. `mode="made"`：执行 `builder -> validator -> export`
        2. `mode="full"`：执行 `builder -> mutate -> validator -> export`
        3. `mode="mutate"`：从已有 topology 目录里的 `hand.yaml.hand_cfg`
           快照恢复，再执行 `mutate -> validator -> export`
        4. `artifact_level="hand_cfg"`：只保留内存中的 `HandCfg`
        5. `artifact_level="urdf" / "bundle"`：落盘导出由 `HandExporter` 负责

        当前仍然保留的边界是：

        - mutate-only 只支持从 pre-made sidecar 的 `hand_cfg` 快照恢复
        - 更细的 mode 分支统计 / provenance 记录还可以继续加密

        因此这里应把算法规格放在活代码前面，而不是留在 `return` 后面变成
        死注释；死注释既破坏可读性，也会让读者误判“这段到底做没做”。

        Returns:
            HandGenerationResult: 一次生成调用的轻量结果包。

        Raises:
            ValueError: 当 `Made` 仍是抽象 `HandBuilderCfg` 而非具体 builder cfg 时抛出。
        """

        # 算法规格：mode-aware generation pipeline
        # ────────────────────────────────────────
        # 输入
        #   cfg.mode: `made` / `mutate` / `full`
        #   cfg.artifact_level: `hand_cfg` / `urdf` / `bundle`
        #   cfg.Made: 前序生成配置
        #   cfg.Mutate: 后序变异配置
        #   cfg.Validate: 生成后验证配置
        #   cfg.Export: 导出器入口
        #
        # 输出：`HandGenerationResult`
        #
        # ── 当前已落地部分 ──
        #   1. `mode=made`：执行 made -> validate -> export。
        #   2. `mode=full`：执行 made -> mutate -> validate -> export。
        #   3. `mode=mutate`：从 sidecar `hand_cfg` 快照恢复后执行 mutate。
        #   4. `artifact_level=hand_cfg`：不强迫用户落盘 URDF。
        #   5. `artifact_level=bundle`：`HandCfg` 与导出物可同时保留。
        #
        # ── 当前未完全落地部分 ──
        #   TODO: 更广义的“URDF 反向恢复 HandCfg”尚未提供。
        #   TODO: 更细粒度的 provenance / rejection 统计仍可继续扩充。
        #
        # IDEA：主入口的价值不是把每一步都做满，而是把默认路径做顺，
        # 同时给用户足够多的“中间停靠点”。

        if self.cfg.mode == "mutate":
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        selection = self._resolve_single_premade_selection()
        if selection is None:
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        return self._generate_once(hand_preset_name=selection[0], connectivity_preset_name=selection[1])

    def _premade_tasks(self, *, mutate_samples_per_topology: int) -> list[_PremadeTask]:
        r"""把 pre-made 离散空间展开成可调度任务列表。

        这里不做任何 build / validate / export，只把组合空间写成显式任务：

        $$
        \mathcal{T} =
        \{(h_i,\ c_j,\ k)\mid h_i\in\mathcal{H}, c_j\in\mathcal{C}(h_i)\}
        $$

        Args:
            mutate_samples_per_topology (int): 每个 topology/connectivity 基座需要派生的样本数。

        Returns:
            list[_PremadeTask]: 按原串行枚举顺序排列的任务表。
        """

        tasks: list[_PremadeTask] = []
        for hand_preset_name in self._candidate_hand_preset_names():
            connectivity_names = self._connectivity_names_for_hand_preset(hand_preset_name=hand_preset_name)
            for connectivity_preset_name in connectivity_names:
                for _ in range(mutate_samples_per_topology):
                    tasks.append(
                        _PremadeTask(
                            hand_preset_name=hand_preset_name,
                            connectivity_preset_name=connectivity_preset_name,
                            enumerated=mutate_samples_per_topology == 1,
                        )
                    )
        return tasks

    def _premade_parallel_worker_count(self, *, task_count: int) -> int:
        r"""计算 pre-made 样本级并行 worker 数。

        worker 数本质上是吞吐和交互性的折中：

        - 太少：无法覆盖 build / validate / export 的 CPU 等待时间；
        - 太多：文件系统写入和 Python 进程调度会开始互相争抢。

        Returns:
            int: 至少为 1、至多为任务数的 worker 数。
        """

        if task_count <= 0:
            return 1
        if self.cfg.premade_parallel_workers is not None:
            return max(1, min(int(self.cfg.premade_parallel_workers), task_count))
        cpu_count = os.cpu_count() or 2
        inferred_workers = max(cpu_count - 1, 1)  # 留一个核心给 IDE / shell / 仿真环境，避免全机无响应
        return max(1, min(inferred_workers, task_count))

    def _record_premade_worker_result(self, worker_result: _PremadeWorkerResult) -> HandGenerationResult | None:
        r"""把 worker 返回值并入主进程 summary，并返回可 yield 的成功结果。

        Args:
            worker_result (_PremadeWorkerResult): worker 侧成功结果或拒绝阶段。

        Returns:
            HandGenerationResult | None: 成功样本返回结果；拒绝样本返回 None。
        """

        if worker_result.result is not None:
            self._record_generation_success(worker_result.result, write_summary=False)
            return worker_result.result
        self._record_generation_rejection(
            stage=worker_result.rejection_stage or "premade_worker_rejected",
            write_summary=False,
        )
        return None

    def _generate_premade_serial(self, *, tasks: list[_PremadeTask]) -> list[HandGenerationResult]:
        r"""沿用原有串行路径执行 pre-made 任务表。

        这条路径承担两个职责：

        1. 用户显式关闭 `premade_parallel` 时的确定性基线；
        2. 进程池 / pickle / worker 环境异常时的 fallback。
        """

        results: list[HandGenerationResult] = []
        success_limit = self.cfg.max_enumerate
        for task in tasks:
            if success_limit is not None and len(results) >= success_limit:
                break
            result = self._generate_once(
                hand_preset_name=task.hand_preset_name,
                connectivity_preset_name=task.connectivity_preset_name,
                enumerated=task.enumerated,
            )
            if result is not None:
                results.append(result)
        return results

    def _generate_premade_parallel(self, *, tasks: list[_PremadeTask]) -> list[HandGenerationResult]:
        r"""用进程池执行 pre-made 样本级并行。

        worker 粒度是一个完整样本，而不是 validator 的单条规则：

        - worker 内：`build -> pre-made validator -> export`
        - 主进程：按原任务顺序汇总 success / rejection 到 `summary.yaml`

        这样可以最大化复用现有单样本语义，同时避免引入 producer-consumer pipeline
        所需的跨阶段队列、回滚协议和 summary 竞争。
        """

        if not tasks:
            return []

        run_root, _ = self._ensure_run_context()
        success_limit = self.cfg.max_enumerate
        worker_count = self._premade_parallel_worker_count(task_count=len(tasks))
        if worker_count <= 1:
            return self._generate_premade_serial(tasks=tasks)

        ordered_results: list[HandGenerationResult] = []
        task_cursor = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            while task_cursor < len(tasks):
                if success_limit is None:
                    batch_size = len(tasks) - task_cursor
                else:
                    remaining_success = success_limit - len(ordered_results)
                    if remaining_success <= 0:
                        break
                    # `max_enumerate` 的语义是“成功产物数上限”。因此并行路径按缺口提交：
                    # 先提交缺口数量的候选；若 validator 拒绝了一部分，再继续补同等缺口。
                    # 这避免 smoke test 为了拿 512 个成功样本而提前跑完整 5788 个候选。
                    batch_size = min(len(tasks) - task_cursor, max(remaining_success, worker_count))

                batch_tasks = tasks[task_cursor : task_cursor + batch_size]
                task_cursor += batch_size

                indexed_results: list[tuple[int, _PremadeWorkerResult]] = []
                future_to_index = {
                    executor.submit(_generate_premade_worker, self.cfg, run_root, task): index
                    for index, task in enumerate(batch_tasks)
                }
                for future in as_completed(future_to_index):
                    indexed_results.append((future_to_index[future], future.result()))

                for _, worker_result in sorted(indexed_results, key=lambda item: item[0]):
                    if success_limit is not None and len(ordered_results) >= success_limit:
                        break
                    result = self._record_premade_worker_result(worker_result)
                    if result is not None:
                        ordered_results.append(result)
        self._write_run_summary()
        return ordered_results

    def generate_batch(self) -> Iterator[HandGenerationResult]:
        r"""批量生成整手资产。

        这是面向批量数据集生成的主接口。与 ``generate()`` 的区别在于它
        返回一个迭代器，支持 lazy 消费（边生成边落盘），不需要把所有结果
        同时塞进内存。

        当前 contract 已明确固定为两层：

        1. pre-made：显式遍历离散 topology × connectivity 空间
        2. post-mutate：对每个 pre-made 基座采样 `n_samples` 个后序样本

        Yields:
            HandGenerationResult: 每次成功生成的轻量结果包。

        Raises:
            RuntimeError: 当拒绝样本过多，超过最大尝试次数时抛出。
        """

        if self.cfg.mode == "mutate":
            target_count = max(int(self.cfg.n_samples), 0)
            success_count = 0
            attempt_count = 0
            max_attempts = max(target_count * int(self.cfg.post_mutate_max_attempt_factor), 10)
            mutator = HandMutator(self.cfg.Mutate)
            source_hand = self._load_mutate_source().hand_cfg

            while success_count < target_count:
                remaining = target_count - success_count
                batch_budget = min(remaining, max_attempts - attempt_count)
                if batch_budget <= 0:
                    raise RuntimeError(
                        "too many rejected samples during mutate-only generate_batch(); "
                        f"succeeded={success_count}, target={target_count}, attempted={attempt_count}, "
                        f"budget={max_attempts}"
                    )

                # 联合采样在这里按“缺口”批量完成：若目标 100 已成功 94，
                # 下一轮只采 6 组联合参数，而不是重新采满 100。
                try:
                    sampled_batch = mutator.sample_batch(source_hand, batch_size=batch_budget)
                except Exception:
                    sampled_batch = [_sample_mutation_terms(mutator, source_hand) for _ in range(batch_budget)]

                for sampled_terms in sampled_batch:
                    attempt_count += 1
                    result = self._generate_once(
                        hand_preset_name=None,
                        connectivity_preset_name=None,
                        sampled_mutation_terms=sampled_terms,
                    )
                    if result is None:
                        continue
                    yield result
                    success_count += 1
                    if success_count >= target_count:
                        break
            return

        hand_preset_names = self._candidate_hand_preset_names()
        if hand_preset_names:
            mutate_samples_per_topology = (
                max(int(self.cfg.n_samples), 0)
                if self.cfg.mode == "full" and _has_enabled_mutation(self.cfg.Mutate)
                else 1
            )
            tasks = self._premade_tasks(mutate_samples_per_topology=mutate_samples_per_topology)

            # pre-made 的默认并行只覆盖纯 `mode="made"` 枚举。
            # `mode="full"` 还会叠加 post-mutate 随机采样与拒绝补采，先保留串行路径，
            # 避免把两类完全不同的并行语义揉在同一个控制流里。
            if self.cfg.mode == "made" and self.cfg.premade_parallel:
                try:
                    results = self._generate_premade_parallel(tasks=tasks)
                except Exception:
                    if self.cfg.premade_parallel_fallback == "raise":
                        raise
                    results = self._generate_premade_serial(tasks=tasks)
            else:
                results = self._generate_premade_serial(tasks=tasks)

            for result in results:
                yield result
            return

        target_count = max(int(self.cfg.n_samples), 0)
        success_count = 0
        attempt_count = 0
        max_attempts = max(target_count * 10, 10)

        while success_count < target_count:
            attempt_count += 1
            if attempt_count > max_attempts:
                raise RuntimeError("too many rejected samples during generate_batch()")
            result = self.generate()
            if result is None:
                continue
            yield result
            success_count += 1

__all__ = [
    "HandGenerationResult",
    "HandGeneratorCfg",
    "HandGenerator",
]
