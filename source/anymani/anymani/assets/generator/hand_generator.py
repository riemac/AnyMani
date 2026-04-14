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

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..exporter import HandExporter, HandExporterCfg
from ..presets.connectivity_presets import (
    get_finger_connectivity_preset_data,
    get_hand_connectivity_preset_data,
    list_hand_connectivity_preset_names,
)
from ..presets.hand_presets import get_hand_builder_preset_data, make_human_like_builder_cfg_from_preset
from ..validator import HandValidator, HandValidatorCfg

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


def _normalize_name_list(values: list[str] | tuple[str, ...] | None, *, field_name: str) -> list[str]:
    r"""把 recipe / YAML 侧的名称列表统一规约为 `list[str]`。

    这次 pre-made façade 的显式设计就是：

    - `hand_presets: list[str]`
    - `connectivity_presets: dict[str, list[str]] | None`

    因而这里不再像上一版那样往 tuple 收，而是显式回到 list。
    这样科研侧在读配置时，看到的形状会和 `FIXME` 里的契约完全一致。
    """

    if values is None:
        return []
    if isinstance(values, str):
        return [str(values)]
    if isinstance(values, tuple):
        return [str(item) for item in values]
    if isinstance(values, list):
        return [str(item) for item in values]
    raise TypeError(f"{field_name} must be a list/tuple of str or None, got {values!r}")


def _normalize_connectivity_mapping(
    values: dict[str, list[str] | tuple[str, ...]] | None,
) -> dict[str, list[str]] | None:
    r"""把 `connectivity_presets` 统一规约为 `dict[str, list[str]] | None`。

    这里的规约目标不是“做复杂兼容层”，而是把 YAML / Python 两侧常见写法收拢成
    同一形状，避免后续 pre-made 主循环到处判断 tuple / list / None。
    """

    if values is None:
        return None
    if not isinstance(values, dict):
        raise TypeError(f"connectivity_presets must be a mapping or None, got {values!r}")

    normalized: dict[str, list[str]] = {}
    for hand_preset_name, preset_names in values.items():
        normalized[str(hand_preset_name)] = _normalize_name_list(
            preset_names,
            field_name=f"connectivity_presets[{hand_preset_name!r}]",
        )
    return normalized


def _stable_premade_id(*parts: str) -> str:
    r"""为 pre-made 的离散 recipe 组合生成稳定短 ID。

    在 `enumerate` 路径里，我们希望：

    - 同一个 `(base_hand_preset, connectivity_preset)` 组合，多次生成时 ID 稳定；
    - sidecar / output path 能直接靠这组 provenance 回溯；
    - 但目录名又不要长到影响人工浏览。

    因而这里对 provenance 字符串做 md5，再取前 8 位十六进制作为稳定短签名。
    """

    payload = "::".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


def _resolve_deleted_joint_names(finger, *, deleted_joint_suffixes: tuple[str, ...]) -> tuple[str, ...]:
    r"""把 slot-agnostic 的 delete recipe 展开成当前 finger 上的真实 joint 名。

    例如，当 finger 为 `index`，而 recipe 写的是 `("j2", "j3")` 时，这里会解析成：

    - `index_j2`
    - `index_j3`

    这样 connectivity preset 的科学语义仍然是“显式删除哪些 joint”，
    只是为了避免对 index / middle / ring 重复抄写，允许 recipe 在注册层使用后缀表达。
    """

    joint_name_set = {joint.name for joint in finger.joints}  # 当前 finger 真实存在的 joint 名全集
    resolved: list[str] = []
    for suffix in deleted_joint_suffixes:
        candidate = str(suffix)  # 允许 recipe 直接写完整 joint 名，也允许只写后缀
        if candidate not in joint_name_set and not candidate.startswith(f"{finger.name}_"):
            candidate = f"{finger.name}_{candidate}"  # 把 `j2` 展开成 `index_j2` / `thumb_j2` 这类真实名字
        if candidate not in joint_name_set:
            raise ValueError(
                f"Deleted joint token {suffix!r} cannot be resolved on finger {finger.name!r}; "
                f"available joints are {[joint.name for joint in finger.joints]!r}"
            )
        resolved.append(candidate)
    return tuple(resolved)


# ============================================================================
#  生成结果包
# ============================================================================


@dataclass
class HandGenerationResult:
    r"""一次生成调用的轻量结果包。

    这个结果包的设计目标是“按需承载产物”，而不是强迫每次都生成完整
    产物链。若用户只想看结构，可以只填 `hand_cfg`；若用户想落盘，则可以
    同时填 `urdf_path` 与 `sidecar_path`。
    """

    hand_cfg: HandCfg | None = None
    """内存中的手部配置；轻量模式下可直接返回。"""

    urdf_path: Path | None = None
    """导出的 URDF 路径；若未请求导出则为 `None`。"""

    sidecar_path: Path | None = None
    """附带元数据文件路径；例如 yaml / json sidecar。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """生成过程的辅助信息，例如 preset 名、随机种子、拒绝原因统计等。"""

    tree_txt: str | None = None
    """ASCII 树状可视化；通过 `render_trees()` 填充，也可落盘为 `.txt` 文件。"""

    tree_mermaid: str | None = None
    """Mermaid 树状可视化；通过 `render_trees()` 填充，可直接嵌入 Markdown。"""

    def render_trees(self) -> "HandGenerationResult":
        """从 `self.hand_cfg` 就地生成 txt 和 Mermaid 两种树状可视化，并返回自身。

        若 `hand_cfg` 为 `None` 则无操作。
        """

        if self.hand_cfg is not None:
            self.tree_txt = render_hand_tree_txt(self.hand_cfg)
            self.tree_mermaid = render_hand_tree_mermaid(self.hand_cfg)
        return self


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
    """前序生成配置入口；负责把生成空间中的选择落到一个初始 `HandCfg`。"""

    Mutate: HandMutatorCfg = field(default_factory=HandMutatorCfg)
    """后序变异配置入口；可为空操作，也可串联多个局部工具。"""

    Validate: HandValidatorCfg = field(default_factory=HandValidatorCfg)
    """手级验证配置入口；用于生成后校验结构和语义约束。"""

    Export: HandExporterCfg = field(default_factory=HandExporterCfg)
    """手级导出器配置入口；用于把 HandCfg 导出为 URDF / sidecar / tree 文件等产物。"""

    output_dir: Path | str = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "generated")
    """产物落盘根目录。

    默认写到 `assets/generated/`，与当前子项目的目录约定保持一致；
    测试或批量脚本也可以显式覆盖成临时目录。
    """

    hand_presets: list[str] = field(default_factory=list)
    """pre-made 阶段参与生成的 base hand preset 名列表。

    # NOTE:
    这里显式保留 `list[str]` 形状，而不是转成 tuple，
    因为它本身就是“用户手写离散列表”的语义对象。
    """

    connectivity_presets: dict[str, list[str]] | None = None
    """base hand preset -> connectivity preset 名列表的映射。

    - key: `hand_preset` 名，例如 `single_palm_allegro`
    - value: 该 hand 允许搭配的 `HandConnectivityPreset` 名列表

    若为 `None`，或缺少某个 hand preset 的键，则运行时回退到该 family 全部合法 preset。
    """

    output_layout: Literal["flat", "recursive"] = "recursive"
    """pre-made 产物的目录组织模式。

    - ``recursive``：`generated/pre_made/{hand_preset}/{connectivity_preset}/{sample_id}/`
    - ``flat``：`generated/pre_made_flat/{sample_id}/`

    # NOTE:
    这个字段仍然收口在 `HandGeneratorCfg`，因为用户已经明确要求：
    `HandGeneratorCfg` 才是生成资产时的唯一 façade，不再额外包装新的 runner。
    """

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandGenerator
        self.output_dir = Path(self.output_dir)  # 统一在 cfg 边界内把路径收口为 `Path`
        self.hand_presets = _normalize_name_list(self.hand_presets, field_name="hand_presets")
        self.connectivity_presets = _normalize_connectivity_mapping(self.connectivity_presets)

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

    def _candidate_hand_preset_names(self) -> tuple[str, ...]:
        r"""返回当前 generator 可见的 base hand preset 名集合。

        pre-made façade 现在只保留 `hand_presets` 这一条 hand 入口，
        因而这里的职责也相应收敛为：把 list 规整成稳定 tuple，供 sample / enumerate 共用。
        """

        return tuple(self.cfg.hand_presets)

    def _connectivity_names_for_hand_preset(self, *, hand_preset_name: str) -> tuple[str, ...]:
        r"""返回某个 base hand preset 允许搭配的 connectivity preset 名集合。

        规则直接对应 `HandGeneratorCfg.connectivity_presets` 的科研语义：

        1. 若字典为 `None`，则默认展开该 family 下全部已注册 connectivity；
        2. 若字典缺少当前 hand preset 的键，同样默认展开该 family 全部 connectivity；
        3. 若字典显式给了该 hand 的列表，则严格采用该列表；
        4. 无论是哪条路径，最终都要再次校验 family 一致性，避免手 preset 与
           connectivity preset 的跨 family 错配。
        """

        hand_preset_data = get_hand_builder_preset_data(hand_preset_name)  # hand preset 是 family 真源
        family = str(hand_preset_data["family"])

        if self.cfg.connectivity_presets is None or hand_preset_name not in self.cfg.connectivity_presets:
            names = list_hand_connectivity_preset_names(family)  # 缺省时自动展开该 family 全部合法 connectivity
        else:
            configured = tuple(self.cfg.connectivity_presets[hand_preset_name])
            if not configured:
                raise ValueError(
                    f"connectivity_presets[{hand_preset_name!r}] is empty; "
                    "please provide at least one connectivity preset name or remove the key to use family defaults."
                )
            names = configured

        compatible: list[str] = []
        for name in names:
            preset = get_hand_connectivity_preset_data(name)
            if preset.family != family:
                raise ValueError(
                    f"Connectivity preset {name!r} belongs to family {preset.family!r}, "
                    f"but hand preset {hand_preset_name!r} belongs to family {family!r}"
                )
            compatible.append(name)
        return tuple(compatible)

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

        hand_candidates = self._candidate_hand_preset_names()
        if not hand_candidates:
            return None

        hand_preset_name = random.choice(hand_candidates)  # pre-made sample 第一步：抽 base hand preset
        connectivity_names = self._connectivity_names_for_hand_preset(hand_preset_name=hand_preset_name)
        connectivity_preset_name = random.choice(connectivity_names)  # pre-made sample 第二步：抽该 hand 允许的 connectivity
        return hand_preset_name, connectivity_preset_name

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

        if self.cfg.Made.class_type is not HandBuilder:
            builder_cfg = self.cfg.Made  # 显式 `Made` 一旦具体化，就说明用户要以它作为真实基座
        elif hand_preset_name is not None:
            builder_cfg = make_human_like_builder_cfg_from_preset(hand_preset_name)
        else:
            raise ValueError("HandGenerator requires a concrete Made cfg or at least one hand preset when using the pre-made facade")

        builder = builder_cfg.class_type(builder_cfg)
        return builder.build(), builder_cfg.__class__.__name__

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

        connectivity_preset = get_hand_connectivity_preset_data(connectivity_preset_name)
        if connectivity_preset.family != hand_cfg.family:
            raise ValueError(
                f"Connectivity preset {connectivity_preset.name!r} belongs to family {connectivity_preset.family!r}, "
                f"but the built hand belongs to {hand_cfg.family!r}"
            )

        # 这里局部导入 `JointDeleteMutator`，是为了保留当前模块原先的 fallback 结构：
        # 没有 mutate 子包的环境下，`HandGenerator` 其它轻量路径仍可 import。
        from .mutate import JointDeleteCfg, JointDeleteMutator

        mutated = hand_cfg.copy()
        per_finger_connectivity: dict[str, Any] = {}

        # 按 slot recipe 顺序逐根手指 lower。
        # 每一步都只改当前 finger，其余 finger 保持不动，便于 sidecar 回溯。
        for finger_name, finger_recipe_name in connectivity_preset.finger_slots.items():
            current_finger = next((finger for finger in mutated.fingers if finger.name == finger_name), None)
            if current_finger is None:
                continue  # 当前 hand 若没有这个 slot，就跳过，不对未来 little-finger 扩展设死约束

            finger_recipe = get_finger_connectivity_preset_data(finger_recipe_name)
            deleted_joint_names = _resolve_deleted_joint_names(
                current_finger,
                deleted_joint_suffixes=finger_recipe.deleted_joint_suffixes,
            )
            deleted_joint_set = set(deleted_joint_names)  # 便于同时回溯被删 joint 与被删 child-link
            deleted_child_links = [
                str(joint.child)
                for joint in current_finger.joints
                if joint.name in deleted_joint_set
            ]
            remaining_revolute = sum(
                1
                for joint in current_finger.joints
                if joint.joint_type == "revolute" and joint.name not in deleted_joint_set
            )
            per_finger_connectivity[finger_name] = {
                "finger_connectivity_preset": finger_recipe.name,
                "deleted_joint_suffixes": list(finger_recipe.deleted_joint_suffixes),
                "deleted_joints": list(deleted_joint_names),
                "deleted_child_links": deleted_child_links,
                "remaining_revolute": remaining_revolute,
                "regroup_strategy": finger_recipe.regroup_strategy,
            }

            if not deleted_joint_names:
                continue  # full chain 这类 recipe 不需要真正执行 delete

            lowered = JointDeleteMutator(
                JointDeleteCfg(
                    target_finger=finger_name,
                    deleted_joints=deleted_joint_names,
                    regroup_strategy=finger_recipe.regroup_strategy,
                    respect_preset=False,  # legality 已由 connectivity registry 定义，这里不再让 generic mutator 额外裁决
                    keep_terminal_joint=True,
                )
            ).mutate(mutated)
            if lowered is None:
                raise ValueError(
                    f"Failed to lower connectivity preset {connectivity_preset_name!r} on finger {finger_name!r}"
                )
            mutated = lowered

        hand_metadata = dict(mutated.metadata)
        hand_metadata["premade_connectivity"] = {
            "base_hand_preset": hand_preset_name,
            "connectivity_preset": connectivity_preset_name,
            "per_finger": per_finger_connectivity,
        }
        mutated = mutated.replace(metadata=hand_metadata)
        return mutated, {
            "base_hand_preset": hand_preset_name,
            "connectivity_preset": connectivity_preset_name,
            "per_finger_connectivity": per_finger_connectivity,
        }

    def _resolve_export_root(self, *, result: HandGenerationResult) -> Path:
        r"""根据 pre-made provenance 与 `output_layout` 计算本次导出的根目录。

        当前导出器仍保持它一贯的职责边界：

        - `HandExporter` 负责在传入目录下再补一层 `{sample_id}/`
        - `HandGenerator` 负责决定这个“传入目录”到底应该是平铺还是递归层级

        这样可以在不破坏现有 exporter 结构的前提下，把目录语义仍然收口到
        `HandGeneratorCfg` 这个唯一 façade。
        """

        if result.hand_cfg is None or "connectivity_preset" not in result.metadata:
            return self.cfg.output_dir

        connectivity_preset_name = str(result.metadata["connectivity_preset"])
        if self.cfg.output_layout == "flat":
            return self.cfg.output_dir / "pre_made_flat"

        hand_anchor = str(result.metadata.get("base_hand_preset") or result.hand_cfg.family)
        return self.cfg.output_dir / "pre_made" / hand_anchor / connectivity_preset_name

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

        hand_cfg, builder_cfg_name = self._build_base_hand(hand_preset_name=hand_preset_name)

        premade_metadata: dict[str, Any] = {}
        if connectivity_preset_name is not None:
            hand_cfg, premade_metadata = self._apply_connectivity_preset(
                hand_cfg,
                connectivity_preset_name=connectivity_preset_name,
                hand_preset_name=hand_preset_name,
            )

        # 后序派生：只有在 `mode="full"` 且至少启用一个 mutate 工具时才进入。
        # 这样 `mode="made"` 不会因为空 mutate cfg 产生额外语义分支。
        if self.cfg.mode == "full" and _has_enabled_mutation(self.cfg.Mutate):
            hand_cfg = HandMutator(self.cfg.Mutate).mutate(hand_cfg)  # `HandCfg -> HandCfg | None`
            if hand_cfg is None:
                return None  # 变异被拒绝；拒绝语义统一表现为“本次样本无结果”

        # 统一在 made / mutate 之后做手级 validator，保证输出侧永远消费的是
        # 同一种“已通过当前约束”的 `HandCfg`。
        validator = HandValidator(self.cfg.Validate)
        validation = validator.validate(hand_cfg)  # 结构、命名、链式一致性等检查结果
        if not validation:
            return None  # validator 拒绝时不抛异常，而是返回空样本给上层批处理逻辑

        sample_id = uuid4().hex[:8]
        if connectivity_preset_name is not None and self.cfg.sampling_strategy == "enumerate":
            sample_id = _stable_premade_id(
                hand_preset_name or hand_cfg.family,
                connectivity_preset_name,
            )

        metadata = {
            "id": sample_id,  # 8 位短 ID，sample 路径默认随机，enumerate 路径按 recipe 稳定化
            "builder_cfg": builder_cfg_name,  # 记录 base hand 最终使用的 builder cfg 类型
            "warnings": validation.warnings,  # validator 的非致命 warning，保留给 sidecar / 调试消费
            "family": hand_cfg.family,
        }
        metadata.update({key: value for key, value in premade_metadata.items() if value is not None})
        metadata["output_layout"] = self.cfg.output_layout

        result = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata=metadata,
        )

        # `artifact_level="hand_cfg"` 表示用户只想拿内存中的 hand schema；
        # 其余两档则交给 exporter 负责落盘。
        if self.cfg.artifact_level != "hand_cfg":
            export_cfg = self.cfg.Export.replace(artifact_level=self.cfg.artifact_level)  # 把主入口的粒度选择下传给 exporter
            exporter = HandExporter(export_cfg)  # 导出器负责 URDF / sidecar / tree 文件
            exporter.export(result, output_dir=self._resolve_export_root(result=result))  # 目录布局仍由 HandGenerator façade 决定

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

        当前已经落地的是 `sample` 路线的最小可用实现：不断调用 `generate()`
        直到得到 `n_samples` 个通过 validator 的样本；`enumerate` 仍明确后延。

        Yields:
            HandGenerationResult: 每次成功生成的轻量结果包。

        Raises:
            NotImplementedError: 当请求 `sampling_strategy="enumerate"` 时抛出。
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

# ============================================================================
#  树状渲染工具
# ============================================================================


def _axis_label(axis: tuple[float, float, float]) -> str:
    """把旋转轴向量压缩成 '+X' / '-Y' / '+Z' 这样的简短标签。"""

    labels = ("X", "Y", "Z")
    idx = max(range(3), key=lambda i: abs(axis[i]))
    sign = "-" if axis[idx] < 0 else "+"
    return f"{sign}{labels[idx]}"


def _link_length(origin: Any) -> float:
    """从 PoseCfg.pos 计算子 link 相对父 link 的平移距离（米）。"""

    if origin is None:
        return 0.0
    x, y, z = origin.pos
    return math.sqrt(x * x + y * y + z * z)


def _fmt_vec(v: tuple[float, float, float]) -> str:
    x, y, z = v
    return f"({x:+.3f}, {y:+.3f}, {z:+.3f})"


def render_hand_tree_txt(hand_cfg: "HandCfg") -> str:
    r"""把 `HandCfg` 渲染为富信息 ASCII 树字符串。

    每条 joint 行包含：joint 名、child link 名、关节类型、旋转轴、
    两岸距离（link length）、关节限位、指尖标记。
    """

    lines: list[str] = []

    # ── 顶层 palm 行 ──────────────────────────────────────────────────────
    dof = hand_cfg.dof_count
    lines.append(
        f"{hand_cfg.palm.name}"
        f"  [family={hand_cfg.family} · {hand_cfg.handedness} · dof={dof}]"
    )

    n_fingers = len(hand_cfg.fingers)
    for f_idx, finger in enumerate(hand_cfg.fingers):
        is_last_finger = f_idx == n_fingers - 1
        f_branch = "└── " if is_last_finger else "├── "
        f_cont = "    " if is_last_finger else "│   "

        # ── finger 挂载行 ─────────────────────────────────────────────────
        mount_pos = _fmt_vec(finger.mount.pos) if finger.mount else "(+0.000, +0.000, +0.000)"
        mount_rpy = _fmt_vec(finger.mount.rpy) if finger.mount else "(+0.000, +0.000, +0.000)"
        lines.append(f"{f_branch}[{finger.name}]  mount={mount_pos} m  rpy={mount_rpy} rad")

        n_joints = len(finger.joints)
        for j_idx, joint in enumerate(finger.joints):
            is_last = j_idx == n_joints - 1
            j_prefix = f"{f_cont}{'└── ' if is_last else '├── '}"

            # 旋转轴与距离
            axis_str = _axis_label(joint.axis) if joint.joint_type != "fixed" else "fixed"
            length = _link_length(joint.origin)

            # 关节限位
            limit_str = ""
            if joint.limit is not None and joint.joint_type == "revolute":
                lo = joint.limit.lower
                hi = joint.limit.upper
                limit_str = f"  [{lo:+.2f}, {hi:+.2f}] rad"

            tip_str = "  ★ TIP" if joint.is_tip else ""

            lines.append(
                f"{j_prefix}{joint.name}  →  {joint.child}"
                f"  {joint.joint_type}  axis={axis_str}  len={length:.4f} m"
                f"{limit_str}{tip_str}"
            )

    return "\n".join(lines)


def render_hand_tree_mermaid(hand_cfg: "HandCfg") -> str:
    r"""把 `HandCfg` 渲染为 Mermaid ``graph TD`` 代码块字符串。

    节点标签包含 joint 名、child link 名、关节类型、旋转轴、link length、
    关节限位；指尖节点使用圆角双圆括号区分。返回值可直接嵌入 Markdown
    三反引号代码块中渲染。
    """

    def node_id(name: str) -> str:
        """把任意名称转为合法 Mermaid 节点 ID。"""
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    lines: list[str] = ["```mermaid", "graph TD"]

    # ── palm 节点 ─────────────────────────────────────────────────────────
    dof = hand_cfg.dof_count
    palm_id = node_id(hand_cfg.palm.name)
    lines.append(
        f'    {palm_id}["{hand_cfg.palm.name}'
        f"<br/>family={hand_cfg.family} · {hand_cfg.handedness} · dof={dof}\"]"
    )

    for finger in hand_cfg.fingers:
        prev_id = palm_id

        for j_idx, joint in enumerate(finger.joints):
            child_id = node_id(joint.child)

            # ── 节点标签 ──────────────────────────────────────────────────
            axis_str = _axis_label(joint.axis) if joint.joint_type != "fixed" else "fixed"
            length = _link_length(joint.origin)

            limit_part = ""
            if joint.limit is not None and joint.joint_type == "revolute":
                lo = joint.limit.lower
                hi = joint.limit.upper
                limit_part = f"<br/>[{lo:+.2f}, {hi:+.2f}] rad"

            tip_part = "<br/>★ TIP" if joint.is_tip else ""

            label = (
                f"{joint.name} → {joint.child}"
                f"<br/>{joint.joint_type} · axis={axis_str} · len={length:.3f} m"
                f"{limit_part}{tip_part}"
            )

            # 指尖用双圆括号，普通节点用方括号
            if joint.is_tip:
                lines.append(f'    {child_id}(("{label}"))')
            else:
                lines.append(f'    {child_id}["{label}"]')

            # ── 边标签 ────────────────────────────────────────────────────
            if j_idx == 0:
                # 第一段边：标注 finger 名称和挂载位置
                mount_pos = _fmt_vec(finger.mount.pos) if finger.mount else "+0.000,+0.000,+0.000"
                edge_lbl = f'|"[{finger.name}] mount={mount_pos}"|'
            else:
                edge_lbl = ""

            lines.append(f"    {prev_id} -->{edge_lbl} {child_id}")
            prev_id = child_id

    lines.append("```")
    return "\n".join(lines)


__all__ = [
    "HandGenerationResult",
    "HandGeneratorCfg",
    "HandGenerator",
    "render_hand_tree_txt",
    "render_hand_tree_mermaid",
]
