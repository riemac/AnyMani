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

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..asset_physics import AssetPhysicsCfg, close_hand_physics
from ..exporter import HandExporter, HandExporterCfg
from ..geometry_identity import geometry_fingerprint_from_hand
from ..procedural_meshes import materialize_hand_procedural_meshes
from ..validator import HandValidator, HandValidatorCfg
from .mutate import HandMutator, HandMutatorCfg
from .premade.batch import (
    build_premade_tasks,
    run_premade_parallel,
    run_premade_serial,
)
from .premade.connectivity import (
    apply_connectivity_preset as _apply_premade_connectivity_preset,
)
from .premade.connectivity import (
    connectivity_names_for_hand_preset as _connectivity_names_for_premade_hand_preset,
)
from .premade.connectivity import (
    resolve_single_premade_selection as _resolve_single_premade_selection,
)
from .premade.identity import resolve_export_root as _resolve_premade_export_root
from .premade.identity import stable_premade_id
from .premade.normalize import normalize_connectivity_mapping, normalize_name_list
from .premade.topology import (
    build_base_hand as _build_premade_base_hand,
)
from .premade.topology import (
    candidate_hand_preset_names as _candidate_premade_hand_preset_names,
)
from .presentation.recolor import (
    RecolorSpec,
    describe_recolor_spec,
    normalize_recolor_spec,
    resolve_visual_recolor_materials,
)
from .result import HandGenerationResult
from .runtime.artifact_lifecycle import rollback_created_directory, rollback_written_artifacts
from .runtime.mutate_batch import PostMutateVariantSetResult, run_post_mutate_source_batch
from .runtime.mutate_sampling import run_mutate_batch_with_independent_proposals
from .runtime.restore import PostMutateSource, load_post_mutate_source
from .runtime.run_context import GenerationRunContext


def _has_enabled_mutation(cfg: HandMutatorCfg) -> bool:
    r"""Check whether any post-mutate tool is enabled in the cfg."""

    return cfg.has_terms()


def _sample_mutation_terms(mutator: HandMutator, target: HandCfg) -> dict[str, dict[str, Any]]:
    r"""按 mutator 描述的独立联合分布为当前 hand 采样一组 term 参数。"""

    batch = mutator.sample_batch(target, batch_size=1)
    return batch[0] if batch else {}


# ============================================================================
#  生成器配置
# ============================================================================


@dataclass(frozen=True)
class PostMutateSourceCfg:
    r"""一个可独立并行调度的 mother variant-set 生成任务。

    ``n_samples`` 只统计 post-mutate variants，不包含 mother 本体；dataset planner
    负责由 ``assets_per_lineage`` 扣除 ``include_mother`` 后给出该值。
    """

    task_id: str
    source_topology_dir: Path | str
    n_samples: int
    seed: int

    def __post_init__(self) -> None:
        r"""规范路径并拒绝空任务、负 variant 数和非法 seed。"""

        object.__setattr__(self, "source_topology_dir", Path(self.source_topology_dir))
        if not self.task_id.strip():
            raise ValueError("post-mutate source task_id cannot be empty")
        if self.n_samples < 0:
            raise ValueError("post-mutate source n_samples must be non-negative")
        if self.seed < 0:
            raise ValueError("post-mutate source seed must be non-negative")


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

    class_type: type[HandGenerator] | None = None
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

    post_mutate_seed: int = 20260813
    """post-mutate 联合 proposal 的随机种子。

    同一来源 topology、Mutate 配置和 seed 必须重放同一串 mode 与连续参数。
    该 seed 只在当前 mutate batch 内生效，运行结束后恢复调用者的 Python
    随机状态，避免资产生成改变后续实验的随机序列。
    """

    post_mutate_attempts_per_variant: int = 10
    """每个计划 variant 槽位允许的完整联合 proposal 次数。

    validator 拒绝后，当前候选的全部 mode 与连续参数共同作废；下一次尝试从
    四个 mutator 的原始 proposal 分布重新独立抽样。预算耗尽只造成当前槽位
    shortfall，不在其他 mode 内补采，也不占用后续槽位的预算。
    """

    post_mutate_require_unique_geometry: bool = False
    r"""是否要求每个 post-mutate variant set 内的静态几何严格唯一。

    - ``False``：保留早期“identity 可作为正/负权重样本”的采样语义；validator
      通过即可计为成功，因此允许与 mother 或同 set 其它 variant 几何相同。
    - ``True``：把 mother 与本 set 已接受 variants 的 geometry fingerprints 作为
      禁集；重复候选不导出、不计成功，并在当前槽位剩余 proposal 预算内重新抽样。

    唯一性只比较静态运动链、$q_{home}$ 与 collision geometry；joint limits、
    mass/inertia、ID 和 metadata 单独变化不构成新的几何样本。跨 mother 与跨
    dataset role 的重复继续由 dataset builder 的全局闸门处理。
    """

    post_mutate_sources: list[PostMutateSourceCfg] = field(default_factory=list)
    r"""多个 mother 的 source-level variant-set tasks。

    该字段只供 :meth:`HandGenerator.generate_variant_sets` 使用，并与单源
    ``source_topology_dir`` 互斥。每项 task 可独立覆盖 variant 数与 seed。
    """

    post_mutate_parallel: bool = True
    """是否在 mother variant-set 粒度启用进程并行。"""

    post_mutate_parallel_workers: int | None = None
    """post-mutate worker 数；``None`` 使用不超过 8 的 conservative 自动值。"""

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

    Physics: AssetPhysicsCfg | None = field(default_factory=AssetPhysicsCfg)
    r"""物理闭包配置入口。

    这层与 builder / mutator 的职责不同：它不负责生成几何，而是负责在
    最终 collision 几何确定之后，统一闭合整手的 `mass / inertial`。

    - `None`：显式关闭 physics closure；
    - `AssetPhysicsCfg(...)`：按声明式密度与 mesh backend 设置执行闭包。
    """

    output_dir: Path | str = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "generated")
    """产物落盘根目录。

    默认写到 `assets/generated/`，与当前子项目的目录约定保持一致；
    测试或批量脚本也可以显式覆盖成临时目录。
    """

    source_topology_dir: Path | str | None = None
    """独立 post-mutate 的来源 pre-made topology 根目录。

    只在 `mode="mutate"` 时生效，输入形状固定为：

    `.../generated/<premade_run_timestamp>/<group>/<topology_name>/`

    新 contract 下，topology 根目录本身就持有 pre-made 的 `hand.yaml`，
    因而 mutate-only 不再接受 sample 子目录，也不再引入 `*_origin`
    这种过渡性目录语义。
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
    """是否混合不同 family 的 non-thumb 手指拓扑。

    为 ``True`` 时，index / middle / ring / little 可以跨 LEAP / Allegro family
    组合；thumb 始终绑定 base palm family，因为 thumb mount 位姿属于 palm 的
    canonical 装配语义。为 ``False`` 时，所有 surviving finger 都沿用 base family。
    """

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

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandGenerator
        if self.Physics is not None and not isinstance(self.Physics, AssetPhysicsCfg):
            self.Physics = AssetPhysicsCfg(**dict(self.Physics))
        self.output_dir = Path(self.output_dir)  # 统一在 cfg 边界内把路径收口为 `Path`
        if self.source_topology_dir is not None:
            self.source_topology_dir = Path(self.source_topology_dir)
        self.post_mutate_sources = [
            source if isinstance(source, PostMutateSourceCfg) else PostMutateSourceCfg(**source)
            for source in self.post_mutate_sources
        ]
        self.hand_presets = normalize_name_list(self.hand_presets, field_name="hand_presets")
        self.connectivity_presets = normalize_connectivity_mapping(self.connectivity_presets)
        self.recolored = normalize_recolor_spec(self.recolored)
        if self.premade_parallel_workers is not None and self.premade_parallel_workers < 1:
            raise ValueError("premade_parallel_workers must be >= 1 when provided")
        if self.premade_parallel_fallback not in {"serial", "raise"}:
            raise ValueError("premade_parallel_fallback must be either 'serial' or 'raise'")
        if int(self.post_mutate_attempts_per_variant) < 1:
            raise ValueError("post_mutate_attempts_per_variant must be >= 1")
        if not isinstance(self.post_mutate_require_unique_geometry, bool):
            raise TypeError("post_mutate_require_unique_geometry must be bool")
        if self.post_mutate_parallel_workers is not None and self.post_mutate_parallel_workers < 1:
            raise ValueError("post_mutate_parallel_workers must be >= 1 when provided")
        task_ids = [source.task_id for source in self.post_mutate_sources]
        source_paths = [Path(source.source_topology_dir) for source in self.post_mutate_sources]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("post_mutate_sources task_id values must be unique")
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("post_mutate_sources source topology paths must be unique within one batch")

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
        if self.source_topology_dir is not None and self.post_mutate_sources:
            raise ValueError("source_topology_dir and post_mutate_sources are mutually exclusive")
        if self.mode == "mutate" and self.source_topology_dir is None and not self.post_mutate_sources:
            raise ValueError("mode='mutate' requires one source_topology_dir or non-empty post_mutate_sources")
        if self.mode != "mutate" and (self.source_topology_dir is not None or self.post_mutate_sources):
            raise ValueError("post-mutate source fields are only valid when mode='mutate'")
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
        self._run_context: GenerationRunContext | None = None
        self._mutate_source: PostMutateSource | None = None
        self._last_rejection_detail: dict[str, Any] | None = None
        self._post_mutate_geometry_registry: dict[str, str] | None = None

    def _ensure_run_context(self) -> GenerationRunContext:
        r"""懒创建当前 generator 对应的 run 生命周期对象。"""

        if self._run_context is not None:
            return self._run_context
        from .runtime.recipe_loader import RecipeLoader

        self._run_context = GenerationRunContext.create(
            self.cfg,
            config_dump=RecipeLoader.dump(self.cfg),
        )
        return self._run_context

    def _make_worker_run_context(self, run_root: Path) -> GenerationRunContext:
        r"""为 pre-made worker 构造一个不写磁盘 summary 的占位 run context。"""

        self._run_context = GenerationRunContext(
            root_dir=Path(run_root),
            summary={"stats": {"by_topology": {}}},
        )
        return self._run_context

    def _load_mutate_source(self) -> PostMutateSource:
        r"""懒加载独立 post-mutate 的来源样本。

        这里缓存的是“同一个 topology 根目录里的唯一 pre-made 原点”，这样一轮
        `n_samples=20` 的 Monte Carlo 采样不会重复做 20 次磁盘扫描与 YAML 解析。
        """

        if self._mutate_source is not None:
            return self._mutate_source
        if self.cfg.source_topology_dir is None:
            raise ValueError("Independent post-mutate requires 'source_topology_dir'")
        self._mutate_source = load_post_mutate_source(self.cfg.source_topology_dir)
        return self._mutate_source

    def _ensure_post_mutate_geometry_registry(self) -> dict[str, str]:
        r"""建立当前 mother variant set 独占的静态几何禁集。

        mother fingerprint 在第一次候选比较时从已恢复 ``HandCfg`` 计算；后续只有
        成功导出的 variant 才加入集合。每个 multi-source worker 都创建独立 generator，
        因而该字典不跨进程共享，也不改变 mother-level 并行模型。
        """

        if self._post_mutate_geometry_registry is None:
            source = self._load_mutate_source()
            mother_fingerprint = geometry_fingerprint_from_hand(source.hand_cfg)
            self._post_mutate_geometry_registry = {
                mother_fingerprint: f"mother:{source.origin_sample_id}",
            }
        return self._post_mutate_geometry_registry

    def _write_run_summary(self) -> None:
        r"""把当前 run summary 刷到 `<run_root>/summary.yaml`。"""

        if self._run_context is None:
            return
        self._run_context.write_summary()

    def _record_generation_rejection(
        self,
        *,
        stage: str,
        error_codes: tuple[str, ...] = (),
        write_summary: bool = True,
    ) -> None:
        r"""把一次被拒绝的样本尝试及规则代码写入 run summary。"""

        self._ensure_run_context().record_rejection(
            stage=stage,
            error_codes=error_codes,
            write_summary=write_summary,
        )

    def _record_generation_success(self, result: HandGenerationResult, *, write_summary: bool = True) -> None:
        r"""把一次成功样本写入 run summary。"""

        self._ensure_run_context().record_success(result, write_summary=write_summary)

    def _close_physics_if_enabled(
        self,
        hand_cfg: HandCfg,
        *,
        stage: str,
        path_metadata: dict[str, Any] | None = None,
    ) -> tuple[HandCfg, tuple[Path, ...]]:
        r"""在 generator 主链中执行 mesh materialization 与可选物理闭包。

        这层 helper 故意保持很薄：

        - `hand_generator.py` 只知道“什么时候应该把程序化 mesh 固化、什么时候应该闭包”；
        - `procedural_meshes.py` 负责“如何把参数化 `cs` 写成 OBJ”；
        - `asset_physics.py` 负责“如何从 collision 几何算出 inertial”。

        # NOTE:
        `cs` single-mesh contract 要求 validator / exporter / physics closure 都消费真实
        OBJ，而不是 builder 阶段的 `procedural://...` URI。因此 materialization 必须排在
        physics closure 之前；即便 `Physics=None`，validator 或 exporter 仍需要这一步。
        """

        path_probe = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata=dict(path_metadata or {}),
        )  # 用当前阶段已知 provenance 解析 topology/run 级共享 mesh 根目录
        materialized, written_paths = materialize_hand_procedural_meshes(
            hand_cfg,
            mesh_root_dir=self._resolve_mesh_root(result=path_probe),
        )  # procedural `cs` 与 legacy two-primitive `cs` 在这里统一迁移成真实 OBJ mesh
        try:
            closed = close_hand_physics(materialized, self.cfg.Physics, stage=stage)
        except Exception:
            # physics backend 已能看见 OBJ 后才可能失败；异常候选必须回滚本次新建 mesh。
            rollback_written_artifacts(written_paths, boundary_dir=self._ensure_run_context().root_dir)
            raise
        return closed, tuple(written_paths)

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
        - 真正执行删除/重连的 runtime 在 `generator/premade/connectivity_lowering.py`

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
        r"""解析当前样本真正应直写到哪里的导出根目录。

        - pre-made：返回 topology 根目录
        - mutate-only：返回本轮 `<topology>/<mutate_timestamp>/` run 根目录
        """

        run_root = self._ensure_run_context().root_dir
        if self.cfg.mode == "mutate":
            return run_root
        return _resolve_premade_export_root(self.cfg, result=result, run_root=run_root)

    def _resolve_mesh_root(self, *, result: HandGenerationResult) -> Path:
        r"""解析当前样本所属导出边界共享的 mesh 根目录。

        contract:

        - pre-made：以 topology 根目录作为自包含边界，共享 `topology_root/meshes/`
        - mutate-only：以 mutate run 根目录作为自包含边界，共享 `run_root/meshes/`
        """

        export_root = self._resolve_export_root(result=result)
        if self.cfg.mode == "mutate":
            return export_root / self.cfg.Export.Urdf.canonical_mesh_dirname
        return export_root / self.cfg.Export.Urdf.canonical_mesh_dirname

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

        self._ensure_run_context()  # 一次 generator 实例对应一次 run；首次尝试时就固定 run 根与 summary

        validator = HandValidator(self.cfg.Validate) if self.cfg.Validate is not None else None
        validation_warnings: list[str] = []
        validation_metadata: dict[str, Any] = {}
        written_mesh_paths: tuple[Path, ...] = ()  # 仅记录当前候选新写 mesh；成功后成为 bundle 的正式组成部分
        candidate_export_root: Path | None = None  # pre-made topology 根由当前离散任务独占
        candidate_export_root_preexisted = False  # 只有本候选新建的目录才允许在 export 异常时整体回滚
        candidate_geometry_fingerprint: str | None = None  # strict mutate 模式下导出前冻结的静态几何身份

        if self.cfg.mode == "mutate":
            mutate_source = self._load_mutate_source()
            hand_cfg = mutate_source.hand_cfg.copy()  # 每个 post-mutate 样本都从同一份 pre-made topology 根 sidecar 重新起步
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
            path_probe = HandGenerationResult(hand_cfg=hand_cfg, metadata=dict(premade_metadata))
            candidate_export_root = self._resolve_export_root(result=path_probe)
            candidate_export_root_preexisted = candidate_export_root.exists()
            hand_cfg, written_mesh_paths = self._close_physics_if_enabled(
                hand_cfg,
                stage="pre_made",
                path_metadata=premade_metadata,
            )

            # pre-made 结构闸门是“可选的显式闸门”：
            # - `Validate is None`：完全跳过 hand-level validator；
            # - 否则：在 connectivity lower 之后、mutate / export 之前执行。
            if validator is not None:
                try:
                    pre_made_validation = validator.validate_pre_made(hand_cfg)
                except Exception:
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    raise
                if not pre_made_validation:
                    self._last_rejection_detail = {
                        "stage": "pre_made_validate",
                        "errors": list(pre_made_validation.errors),
                        "error_codes": list(pre_made_validation.error_codes),
                        "metadata": dict(pre_made_validation.metadata),
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="pre_made_validate",
                        error_codes=tuple(pre_made_validation.error_codes),
                        write_summary=record_summary,
                    )
                    return None  # pre-made 结构闸门拒绝后，不再允许继续进入 mutate / export
                validation_warnings.extend(pre_made_validation.warnings)
                validation_metadata["pre_made"] = dict(pre_made_validation.metadata)
        sampled_terms: dict[str, dict[str, float]] | None = None

        # 当前后序派生只保留 mutate-only 入口：
        #
        # 1. 输入 pre-made topology 根目录；
        # 2. 从 topology 根 `hand.yaml.hand_cfg` 快照恢复；
        # 3. 在 `<topology>/<mutate_timestamp>/<hash>/` 下写出新样本。
        if self.cfg.mode == "mutate" and _has_enabled_mutation(self.cfg.Mutate):
            mutator = HandMutator(self.cfg.Mutate)
            sampled_terms = sampled_mutation_terms or _sample_mutation_terms(mutator, hand_cfg)
            hand_cfg = mutator.mutate(hand_cfg, sampled_params=sampled_terms)
            if hand_cfg is None:
                self._last_rejection_detail = {
                    "stage": "mutate",
                    "errors": ["mutator returned None"],
                    "error_codes": ["mutate.returned_none"],
                    "metadata": {},
                }
                self._record_generation_rejection(
                    stage="mutate",
                    error_codes=("mutate.returned_none",),
                    write_summary=record_summary,
                )
                return None  # 变异被拒绝；拒绝语义统一表现为“本次样本无结果”
            hand_cfg, written_mesh_paths = self._close_physics_if_enabled(
                hand_cfg,
                stage="post_mutate",
                path_metadata=premade_metadata,
            )
            if validator is not None:
                try:
                    post_mutate_validation = validator.validate_post_mutate(hand_cfg)
                except Exception:
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    raise
                if not post_mutate_validation:
                    self._last_rejection_detail = {
                        "stage": "post_mutate_validate",
                        "errors": list(post_mutate_validation.errors),
                        "error_codes": list(post_mutate_validation.error_codes),
                        "metadata": dict(post_mutate_validation.metadata),
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="post_mutate_validate",
                        error_codes=tuple(post_mutate_validation.error_codes),
                        write_summary=record_summary,
                    )
                    return None
                validation_warnings.extend(post_mutate_validation.warnings)
                validation_metadata["post_mutate"] = dict(post_mutate_validation.metadata)

            # strict variant-set 模式在 sample ID 与 bundle 目录创建前拒绝 mother no-op 和 set 内重复。
            # 候选已经完成 mesh materialization、physics closure 与 validator，因而此处看到的正是
            # exporter/sidecar 将交付的最终 collision geometry；limits/dynamics 不进入该身份。
            if self.cfg.post_mutate_require_unique_geometry:
                candidate_geometry_fingerprint = geometry_fingerprint_from_hand(hand_cfg)
                registry = self._ensure_post_mutate_geometry_registry()
                previous = registry.get(candidate_geometry_fingerprint)
                if previous is not None:
                    duplicate_kind = "mother" if previous.startswith("mother:") else "variant"
                    error_code = f"post_mutate.duplicate_{duplicate_kind}_geometry"
                    self._last_rejection_detail = {
                        "stage": "post_mutate_unique_geometry",
                        "errors": [f"static geometry duplicates {previous}"],
                        "error_codes": [error_code],
                        "metadata": {
                            "geometry_fingerprint": candidate_geometry_fingerprint,
                            "duplicate_of": previous,
                        },
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="post_mutate_unique_geometry",
                        error_codes=(error_code,),
                        write_summary=record_summary,
                    )
                    return None

        sample_id = uuid4().hex[:8]  # mutate-only 变体目录仍以 8 位短哈希为样本根
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
        if validation_metadata:
            metadata["validation"] = validation_metadata
        if isinstance(hand_cfg.metadata.get("post_mutate_samples"), dict):
            merged_samples = dict(metadata.get("post_mutate_samples", {}))
            merged_samples.update(hand_cfg.metadata["post_mutate_samples"])
            metadata["post_mutate_samples"] = merged_samples
        recolor_metadata = describe_recolor_spec(self.cfg.recolored)
        if recolor_metadata is not None:
            metadata["recolored"] = recolor_metadata
        if candidate_geometry_fingerprint is not None:
            metadata["geometry_fingerprint"] = candidate_geometry_fingerprint

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
            try:
                exporter.export(
                    result,
                    output_dir=self._resolve_export_root(result=result),  # pre-made 直写 topology 根；mutate-only 写到 mutate run 根
                    nest_sample_dir=self.cfg.mode == "mutate",  # 只有 mutate-only 仍需要 `<hash>/` 这一层
                    mesh_root_dir=self._resolve_mesh_root(result=result),
                )
            except Exception:
                run_root = self._ensure_run_context().root_dir
                if (
                    self.cfg.mode != "mutate"
                    and candidate_export_root is not None
                    and not candidate_export_root_preexisted
                ):
                    # exporter 可能已写出 URDF 后才在 sidecar/tree 阶段失败；新 topology 根应整体撤销。
                    rollback_created_directory(candidate_export_root, boundary_dir=run_root)
                else:
                    rollback_written_artifacts(written_mesh_paths, boundary_dir=run_root)
                raise

        self._record_generation_success(result, write_summary=record_summary)
        if candidate_geometry_fingerprint is not None:
            self._ensure_post_mutate_geometry_registry()[candidate_geometry_fingerprint] = f"variant:{sample_id}"
        self._last_rejection_detail = None
        return result

    def generate(self) -> HandGenerationResult | None:
        r"""执行一次整手资产生成。

        当前这条主路径已经实现的是：

        1. `mode="made"`：执行 `builder -> validator -> export`
        2. `mode="mutate"`：从已有 topology 目录里的 `hand.yaml.hand_cfg`
           快照恢复，再执行 `mutate -> validator -> export`
        3. `artifact_level="hand_cfg"`：只保留内存中的 `HandCfg`
        4. `artifact_level="urdf" / "bundle"`：落盘导出由 `HandExporter` 负责

        当前仍然保留的边界是：

        - mutate-only 只支持从 pre-made sidecar 的 `hand_cfg` 快照恢复
        - `mode="full"` 这轮被显式暂停，避免沿用旧目录语义
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
        #   2. `mode=mutate`：从 sidecar `hand_cfg` 快照恢复后执行 mutate。
        #   3. `artifact_level=hand_cfg`：不强迫用户落盘 URDF。
        #   4. `artifact_level=bundle`：`HandCfg` 与导出物可同时保留。
        #   5. `mode=full`：当前显式报不支持，等待后续目录语义迁移完成。
        #
        # ── 当前未完全落地部分 ──
        #   TODO: 更广义的“URDF 反向恢复 HandCfg”尚未提供。
        #   TODO: 更细粒度的 provenance / rejection 统计仍可继续扩充。
        #
        # IDEA：主入口的价值不是把每一步都做满，而是把默认路径做顺，
        # 同时给用户足够多的“中间停靠点”。

        if self.cfg.mode == "full":
            raise NotImplementedError(
                "mode='full' is temporarily unsupported. "
                "This migration only covers mode='made' and independent mode='mutate'; "
                "the full pipeline has not been adapted to topology-root export semantics yet."
            )
        if self.cfg.mode == "mutate":
            if self.cfg.post_mutate_sources:
                raise ValueError("multi-source mutate cfg must use generate_variant_sets(), not generate()")
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        selection = self._resolve_single_premade_selection()
        if selection is None:
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        return self._generate_once(hand_preset_name=selection[0], connectivity_preset_name=selection[1])

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

        if self.cfg.mode == "full":
            raise NotImplementedError(
                "mode='full' is temporarily unsupported. "
                "This migration only covers mode='made' and independent mode='mutate'; "
                "the full pipeline has not been adapted to topology-root export semantics yet."
            )

        if self.cfg.mode == "mutate":
            if self.cfg.post_mutate_sources:
                raise ValueError("multi-source mutate cfg must use generate_variant_sets(), not generate_batch()")
            target_count = max(int(self.cfg.n_samples), 0)
            mutator = HandMutator(self.cfg.Mutate)
            source_hand = self._load_mutate_source().hand_cfg
            yield from run_mutate_batch_with_independent_proposals(
                generator=self,
                mutator=mutator,
                source_hand=source_hand,
                target_count=target_count,
                attempts_per_variant=int(self.cfg.post_mutate_attempts_per_variant),
                seed=int(self.cfg.post_mutate_seed),
            )
            return

        hand_preset_names = self._candidate_hand_preset_names()
        if hand_preset_names:
            tasks = build_premade_tasks(self)  # made 模式下每个 topology 只导出一次 topology 根基座

            # pre-made 的默认并行只覆盖离散 topology 枚举；新 contract 下这里已经不再承担 full 的复合语义。
            if self.cfg.mode == "made" and self.cfg.premade_parallel:
                try:
                    results = run_premade_parallel(self, tasks=tasks)
                except Exception:
                    if self.cfg.premade_parallel_fallback == "raise":
                        raise
                    results = run_premade_serial(self, tasks=tasks)
            else:
                results = run_premade_serial(self, tasks=tasks)

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

    def generate_variant_sets(self) -> Iterator[PostMutateVariantSetResult]:
        r"""为多个 mother source 生成相互独立的 variant-set runs。

        每个 source task 在独立 ``HandGenerator`` 中恢复 mother、设置自己的 seed，
        顺序完成全部 variants；source tasks 可进程并行，返回顺序仍与 cfg 声明一致。
        """

        if self.cfg.mode != "mutate" or not self.cfg.post_mutate_sources:
            raise ValueError("generate_variant_sets() requires mode='mutate' and non-empty post_mutate_sources")
        yield from run_post_mutate_source_batch(self, tasks=tuple(self.cfg.post_mutate_sources))

__all__ = [
    "HandGenerationResult",
    "HandGeneratorCfg",
    "HandGenerator",
    "PostMutateSourceCfg",
    "PostMutateVariantSetResult",
]
