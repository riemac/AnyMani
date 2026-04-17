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

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

import yaml

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..exporter import HandExporter, HandExporterCfg
from ..validator import HandValidator, HandValidatorCfg
from ._generation_result import HandGenerationResult
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
from ._tree_render import render_hand_tree_mermaid, render_hand_tree_txt

try:
    from .mutate import HandMutator, HandMutatorCfg
except Exception:
    @dataclass
    class HandMutatorCfg(AssetCfgBase):
        r"""Fallback mutate cfg used when the mutate package is unavailable.

        The first implementation slice does not execute post-mutate logic, but the
        generator cfg still keeps the field so the public interface remains stable.
        """

        joint_delete: object | None = None
        link_scale: object | None = None
        tip_replace: object | None = None
        limit_tweak: object | None = None
        mount_perturb: object | None = None
        finger_replace: object | None = None

    class HandMutator:
        r"""Fallback mutator used when the mutate package is unavailable."""

        def __init__(self, cfg: HandMutatorCfg):
            self.cfg = cfg

        def mutate(self, target: HandCfg) -> HandCfg | None:
            raise NotImplementedError("mutate runtime is unavailable in the current environment")


def _has_enabled_mutation(cfg: HandMutatorCfg) -> bool:
    r"""Check whether any post-mutate tool is enabled in the cfg."""

    return any(
        getattr(cfg, key) is not None
        for key in ("joint_delete", "link_scale", "tip_replace", "limit_tweak", "mount_perturb", "finger_replace")
    )


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

    sampling_strategy: Literal["sample", "enumerate"] = "sample"
    """批量生成时的采样策略。

    - ``sample``：先确定总预算 ``n_samples``，每次从生成空间联合采样
      (pre-made 参数 × post-mutate 参数)，产物数量严格等于 ``n_samples``。
      适合大规模多样化训练数据集，不会产生笛卡尔爆炸。

    - ``enumerate``：遍历 pre-made 配置的离散组合，对每个再遍历 post-mutate
      的离散选项（如关节删除方案）。产物数量 = |pre-made 离散空间| ×
      |post-mutate 离散空间|，可用 ``max_enumerate`` 做硬上限截断。
      适合对照实验和可复现小规模数据集，使用不当会产生爆炸数量。
    """

    n_samples: int = 1
    """``sampling_strategy="sample"`` 时的总产物预算；``generate_batch()`` 将
    循环采样直到累计 ``n_samples`` 个成功通过 validator 的产物。"""

    max_enumerate: int | None = None
    """``sampling_strategy="enumerate"`` 时的最大产物数上限；为 ``None`` 时不截断。
    强烈建议在实验前先预估枚举空间大小，避免无意触发笛卡尔爆炸。"""

    Made: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """前序生成配置入口；主要负责关节拓扑维度的变体，把生成空间中的选择落到一个初始 `HandCfg`。"""

    Mutate: HandMutatorCfg = field(default_factory=HandMutatorCfg)
    """后序变异配置入口；主要负责非关节拓扑维度的变体，可为空操作，也可串联多个局部工具。"""

    Validate: HandValidatorCfg = field(default_factory=HandValidatorCfg)
    """手级验证配置入口；用于生成后校验结构和语义约束。"""

    Export: HandExporterCfg = field(default_factory=HandExporterCfg)
    """手级导出器配置入口；用于把 HandCfg 导出为 URDF / sidecar / tree 文件等产物。"""

    output_dir: Path | str = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "generated")
    """产物落盘根目录。

    默认写到 `assets/generated/`，与当前子项目的目录约定保持一致；
    测试或批量脚本也可以显式覆盖成临时目录。
    """

    handedness: Literal["left", "right", "all"] = "all"
    """TODO: 生成哪种 handedness 的手。`all` 表示同时生成左右手；`left` / `right` 则只生成单一 handedness。"""

    hand_presets: list[str] = field(default_factory=list)
    """pre-made 阶段参与生成的 base hand preset 名列表。

    # NOTE:
    这里显式保留 `list[str]` 形状，而不是转成 tuple，
    因为它本身就是“用户手写离散列表”的语义对象。
    """

    connectivity_presets: dict[str, list[str] | dict[str, list[str]]] | None = None
    """pre-made connectivity façade。

    # FIXME：只支持1套形状：

    **slot-level 主语义**
       `hand_preset -> {slot -> [finger_connectivity_preset_name, ...]}`

    第二种才是 mixed / missing topology 真正依赖的 candidate-pool 语义。
    """

    mixed: bool = True
    """TODO:是否混合不同 family 的手指拓扑。如果为 True，则在 pre-made 阶段允许在同一只手上组合 leap/allegro 的手指变体；如果为 False，则默认每只手只能选一个 family 的 preset 进行派生。"""

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
        self.hand_presets = normalize_name_list(self.hand_presets, field_name="hand_presets")
        self.connectivity_presets = normalize_connectivity_mapping(self.connectivity_presets)
        self.recolored = normalize_recolor_spec(self.recolored)

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

    def _ensure_run_context(self) -> tuple[Path, dict[str, Any]]:
        r"""为当前 generator 实例懒创建一次时间戳 run 根目录与 summary 文档。"""

        if self._run_root is not None and self._run_summary is not None:
            return self._run_root, self._run_summary

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = Path(self.cfg.output_dir)
        run_root = output_root / timestamp

        collision_index = 2
        while run_root.exists():
            run_root = output_root / f"{timestamp}_{collision_index:02d}"
            collision_index += 1  # 同秒重复启动时追加后缀，避免不同 run 相互覆盖

        run_root.mkdir(parents=True, exist_ok=False)

        from ..tool.recipe_loader import RecipeLoader

        pre_made_enabled = self.cfg.mode in {"made", "full"}
        post_mutate_enabled = self.cfg.mode == "full" and _has_enabled_mutation(self.cfg.Mutate)
        self._run_root = run_root
        self._run_summary = {
            "run": {
                "timestamp": run_root.name,
                "root_dir": str(run_root),
                "mode": self.cfg.mode,
                "artifact_level": self.cfg.artifact_level,
                "sampling_strategy": self.cfg.sampling_strategy,
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

    def _record_generation_rejection(self, *, stage: str) -> None:
        r"""把一次被拒绝的样本尝试写入 run summary。"""

        _, summary = self._ensure_run_context()
        stats = summary["stats"]
        stats["attempted"] += 1
        stats["rejected"] += 1
        rejected_by_stage = dict(stats.get("rejected_by_stage") or {})
        rejected_by_stage[stage] = int(rejected_by_stage.get(stage, 0)) + 1
        stats["rejected_by_stage"] = rejected_by_stage
        self._write_run_summary()

    def _record_generation_success(self, result: HandGenerationResult) -> None:
        r"""把一次成功样本写入 run summary。"""

        _, summary = self._ensure_run_context()
        stats = summary["stats"]
        stats["attempted"] += 1
        stats["succeeded"] += 1

        topology_key = self._result_topology_key(result)
        by_topology = dict(stats.get("by_topology") or {})
        by_topology[topology_key] = int(by_topology.get(topology_key, 0)) + 1
        stats["by_topology"] = by_topology
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
        return _resolve_premade_export_root(self.cfg, result=result, run_root=run_root)

    def _generate_once(
        self,
        *,
        hand_preset_name: str | None,
        connectivity_preset_name: str | None,
    ) -> HandGenerationResult | None:
        r"""执行一次单样本生成；供 `generate()` 与 `generate_batch()` 共同复用。

        这个内部 helper 的价值，是把：

        - 单样本 `generate()`
        - 枚举式 `generate_batch()`

        这两条路径共享到同一套 build / connectivity / mutate / validate / export
        语义上，而不是各写一份相似但悄悄分叉的实现。
        """

        # `mode="mutate"` 的语义要求调用方先提供一份现成 `HandCfg`。
        # 当前 `HandGeneratorCfg` 还没有这个输入槽位，因此这里显式拒绝，
        # 避免伪装成“支持 mutate-only”。
        if self.cfg.mode == "mutate":
            raise NotImplementedError("mode='mutate' is intentionally deferred in the first pre-made slice.")

        self._ensure_run_context()  # 一次 generator 实例对应一次 run；首次尝试时就固定时间戳根目录

        hand_cfg, builder_cfg_name = self._build_base_hand(hand_preset_name=hand_preset_name)

        premade_metadata: dict[str, Any] = {}
        if connectivity_preset_name is not None:
            hand_cfg, premade_metadata = self._apply_connectivity_preset(
                hand_cfg,
                connectivity_preset_name=connectivity_preset_name,
                hand_preset_name=hand_preset_name,
            )

        validator = HandValidator(self.cfg.Validate)
        pre_made_validation = validator.validate_pre_made(hand_cfg)
        if not pre_made_validation:
            self._record_generation_rejection(stage="pre_made_validate")
            return None  # pre-made 结构闸门拒绝后，不再允许继续进入 mutate / export

        validation_warnings = list(pre_made_validation.warnings)

        # 后序派生：只有在 `mode="full"` 且至少启用一个 mutate 工具时才进入。
        # 这样 `mode="made"` 不会因为空 mutate cfg 产生额外语义分支。
        if self.cfg.mode == "full" and _has_enabled_mutation(self.cfg.Mutate):
            hand_cfg = HandMutator(self.cfg.Mutate).mutate(hand_cfg)  # `HandCfg -> HandCfg | None`
            if hand_cfg is None:
                self._record_generation_rejection(stage="mutate")
                return None  # 变异被拒绝；拒绝语义统一表现为“本次样本无结果”
            post_mutate_validation = validator.validate_post_mutate(hand_cfg)
            if not post_mutate_validation:
                self._record_generation_rejection(stage="post_mutate_validate")
                return None
            validation_warnings.extend(post_mutate_validation.warnings)

        sample_id = uuid4().hex[:8]
        if connectivity_preset_name is not None and self.cfg.sampling_strategy == "enumerate":
            sample_id = stable_premade_id(
                hand_preset_name or hand_cfg.family,
                connectivity_preset_name,
            )

        metadata = {
            "id": sample_id,  # 8 位短 ID，sample 路径默认随机，enumerate 路径按 recipe 稳定化
            "builder_cfg": builder_cfg_name,  # 记录 base hand 最终使用的 builder cfg 类型
            "warnings": validation_warnings,  # 汇总 pre-made / post-mutate 两阶段 warning，保留给 sidecar / 调试消费
            "family": hand_cfg.family,
            "handedness": hand_cfg.handedness,
        }
        metadata.update({key: value for key, value in premade_metadata.items() if value is not None})
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

        self._record_generation_success(result)
        return result

    def generate(self) -> HandGenerationResult | None:
        r"""执行一次整手资产生成。

        当前这条主路径已经实现的是：

        1. `mode="made"`：执行 `builder -> validator -> export`
        2. `mode="full"`：执行 `builder -> mutate -> validator -> export`
        3. `artifact_level="hand_cfg"`：只保留内存中的 `HandCfg`
        4. `artifact_level="urdf" / "bundle"`：落盘导出由 `HandExporter` 负责

        你原先写在函数尾部的 `# TODO:算法之一（mode-aware generation pipeline）`
        并不是“完全没做”，而是**规格已部分落地**。真正还没有落地的是：

        - `mode="mutate"` 的“只做后序、外部输入 HandCfg”入口
        - 更细的 mode 分支统计 / provenance 记录

        因此这里应把算法规格放在活代码前面，而不是留在 `return` 后面变成
        死注释；死注释既破坏可读性，也会让读者误判“这段到底做没做”。

        Returns:
            HandGenerationResult: 一次生成调用的轻量结果包。

        Raises:
            NotImplementedError: 当请求 `mode="mutate"` 时抛出；该分支仍待接入
                “外部给定 HandCfg -> 后序变异 -> 校验/导出”的独立入口。
            ValueError: 当 `Made` 仍是抽象 `HandBuilderCfg` 而非具体 builder cfg 时抛出。
        """

        # TODO:算法之一（mode-aware generation pipeline）
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
        #   3. `artifact_level=hand_cfg`：不强迫用户落盘 URDF。
        #   4. `artifact_level=bundle`：`HandCfg` 与导出物可同时保留。
        #
        # ── 当前未落地部分 ──
        #   1. `mode=mutate`：尚未提供“外部输入 HandCfg 后仅做后序工具”的入口。
        #   2. 更细粒度的 provenance / rejection 统计仍可继续扩充。
        #
        # IDEA：主入口的价值不是把每一步都做满，而是把默认路径做顺，
        # 同时给用户足够多的“中间停靠点”。

        selection = self._resolve_single_premade_selection()
        if selection is None:
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        return self._generate_once(hand_preset_name=selection[0], connectivity_preset_name=selection[1])

    def generate_batch(self) -> Iterator[HandGenerationResult]:
        r"""批量生成整手资产，按 ``cfg.sampling_strategy`` 路由到不同策略。

        这是面向批量数据集生成的主接口。与 ``generate()`` 的区别在于它
        返回一个迭代器，支持 lazy 消费（边生成边落盘），不需要把所有结果
        同时塞进内存。

        你原先写在函数尾部的两段 TODO，其实对应两种非常不同的批处理语义：

        1. `sample`：从联合分布 $(\text{pre-made} \times \text{post-mutate})$ 反复采样，
           总产物数由 `n_samples` 严格控制
        2. `enumerate`：显式遍历离散空间，理论总产物数近似为
           $|\mathcal{P}| \times |\mathcal{M}|$

        当前这两条路线都已经落地：

        1. `sample`：不断调用 `generate()`，直到得到 `n_samples` 个通过 validator 的样本
        2. `enumerate`：显式遍历 `hand_presets × connectivity presets` 的 pre-made 离散空间，
           并在每个 canonical 组合上再按需叠加 post-mutate

        Yields:
            HandGenerationResult: 每次成功生成的轻量结果包。

        Raises:
            RuntimeError: 当拒绝样本过多，超过最大尝试次数时抛出。
        """

        # TODO:算法之一（batch orchestration — sample 策略）
        # ────────────────────────────────────────
        # 触发条件：cfg.sampling_strategy == "sample"
        #
        # 输入
        #   cfg.n_samples: 目标产物总数 N
        #   cfg.Made / cfg.Mutate / cfg.Validate / cfg.Export: 各阶段配置
        #
        # 输出：yield HandGenerationResult，共 N 个（不含被 validator 拒绝者）
        #
        # ── 当前已落地部分 ──
        #   1. 反复调用 `self.generate()` 进行单次联合采样。
        #   2. `result is not None` 才计入成功样本数。
        #   3. 用 `max_attempts` 抑制 rejection 过多导致的无限循环。
        #
        # ── 关键性质 ──
        #   每次 `generate()` 独立从联合分布采样 $(\text{pre-made} \times \text{post-mutate})$，
        #   不做笛卡尔展开，产物数量严格由 $N$ 控制。
        #
        # TODO:算法之二（batch orchestration — enumerate 策略）
        # ────────────────────────────────────────
        # 触发条件：cfg.sampling_strategy == "enumerate"
        #
        # 输入
        #   cfg.Made: 前序离散生成空间（palm_type × finger_preset 组合列表）
        #   cfg.Mutate: 后序离散选项（joint_delete 方案列表、finger_replace preset 列表）
        #   cfg.max_enumerate: 硬上限（None = 不截断，危险！）
        #
        # 输出：yield HandGenerationResult，最多 max_enumerate 个
        #
        # ── 当前未落地部分 ──
        #   1. `cfg.Made.enumerate()` 的离散 builder 空间接口
        #   2. `cfg.Mutate.enumerate(hand)` 的离散后序方案接口
        #   3. `P × M` 爆炸下的更细粒度预算控制
        #
        # IDEA：两种策略的 API 对调用者完全透明（都是 yield 迭代器），
        # 切换只需修改 `cfg.sampling_strategy`，不需要改调用代码。

        # `enumerate` 不是“循环多跑几次 sample”，而是显式遍历离散组合空间。
        # 当前这条路优先为 pre-made façade 落地：也就是显式遍历
        # `base hand preset × connectivity preset`。
        if self.cfg.sampling_strategy == "enumerate":
            hand_preset_names = self._candidate_hand_preset_names()
            if not hand_preset_names:
                raise NotImplementedError(
                    "enumerate batch generation currently requires hand_presets in the HandGenerator pre-made facade."
                )

            emitted = 0
            max_enumerate = self.cfg.max_enumerate
            for hand_preset_name in hand_preset_names:
                connectivity_names = self._connectivity_names_for_hand_preset(hand_preset_name=hand_preset_name)
                for connectivity_preset_name in connectivity_names:
                    if max_enumerate is not None and emitted >= max_enumerate:
                        return
                    result = self._generate_once(
                        hand_preset_name=hand_preset_name,
                        connectivity_preset_name=connectivity_preset_name,
                    )
                    if result is None:
                        continue
                    yield result
                    emitted += 1
            return

        # `target_count` 是用户要求的成功样本数 $N$，而不是尝试次数。
        # 失败样本（被 mutate / validator 拒绝）不会计入这个预算。
        target_count = max(int(self.cfg.n_samples), 0)  # 目标成功样本数 $N$
        success_count = 0  # 已经产出的有效样本数
        attempt_count = 0  # 总尝试次数（含失败）
        max_attempts = max(target_count * 10, 10)  # 保守上限：默认允许最多约 $10N$ 次尝试

        # sample 批处理的核心循环：直到成功样本数达到 $N$ 才停止。
        while success_count < target_count:
            attempt_count += 1  # 每次循环都代表一次独立联合采样尝试
            if attempt_count > max_attempts:
                raise RuntimeError("too many rejected samples during generate_batch()")
            result = self.generate()  # 复用单样本主路径，避免 batch 和 single 两套语义分叉
            if result is None:
                continue  # 被拒绝样本只消耗尝试次数，不消耗成功预算
            yield result  # lazy 产出，支持边生成边落盘/边消费
            success_count += 1  # 只有成功样本才推进批次完成度

__all__ = [
    "HandGenerationResult",
    "HandGeneratorCfg",
    "HandGenerator",
    "render_hand_tree_txt",
    "render_hand_tree_mermaid",
]
