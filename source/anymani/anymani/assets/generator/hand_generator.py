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

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..exporter import HandExporter, HandExporterCfg
from ..validator import HandValidator, HandValidatorCfg

try:
    from .mutate import HandMutatorCfg
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


def _has_enabled_mutation(cfg: HandMutatorCfg) -> bool:
    r"""Check whether any post-mutate tool is enabled in the cfg."""

    return any(
        getattr(cfg, key) is not None
        for key in ("joint_delete", "link_scale", "tip_replace", "limit_tweak", "mount_perturb", "finger_replace")
    )


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

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandGenerator


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

    def generate(self) -> HandGenerationResult | None:
        r"""执行一次整手资产生成。

        Returns:
            HandGenerationResult: 一次生成调用的轻量结果包。
        """

        if self.cfg.mode == "mutate":
            raise NotImplementedError("mode='mutate' is intentionally deferred in the first pre-made slice.")

        if self.cfg.Made.class_type is HandBuilder:
            raise ValueError("HandGeneratorCfg.Made must be a concrete HandBuilderCfg subclass, not bare HandBuilderCfg")

        builder = self.cfg.Made.class_type(self.cfg.Made)
        hand_cfg = builder.build()

        if self.cfg.mode == "full" and _has_enabled_mutation(self.cfg.Mutate):
            raise NotImplementedError("post-mutate remains out of scope for the first implementation slice.")

        validator = HandValidator(self.cfg.Validate)
        validation = validator.validate(hand_cfg)
        if not validation:
            return None

        result = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata={
                "id": uuid4().hex[:8],
                "builder_cfg": self.cfg.Made.__class__.__name__,
                "warnings": validation.warnings,
            },
        )

        if self.cfg.artifact_level != "hand_cfg":
            export_cfg = self.cfg.Export.replace(artifact_level=self.cfg.artifact_level)
            exporter = HandExporter(export_cfg)
            exporter.export(result, output_dir=Path("outputs"))

        return result

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
        # ── 计算步骤 ──
        #   1. 根据 mode 决定是否执行前序生成（made）
        #   2. 若启用 mutate，则在 HandCfg 上执行后序变异
        #   3. 按 Validate 约束校验当前 HandCfg
        #   4. 按 artifact_level 决定是否导出 URDF / sidecar
        #   5. 将请求的轻量/完整产物写入结果包并返回
        #
        # ── 与 preset 的交叉验证 ──
        #   1. `mode=made` 时，结果应能与 builder preset 一一对应。
        #   2. `mode=mutate` 时，输入必须是可被后序工具消费的 HandCfg。
        #   3. `artifact_level=hand_cfg` 时，不应强迫用户落盘 URDF。
        #   4. `artifact_level=bundle` 时，HandCfg 与导出物应可同时保留。
        #
        # IDEA：主入口的价值不是把每一步都做满，而是把默认路径做顺，
        # 同时给用户足够多的“中间停靠点”。
    def generate_batch(self) -> Iterator[HandGenerationResult]:
        r"""批量生成整手资产，按 ``cfg.sampling_strategy`` 路由到不同策略。

        这是面向批量数据集生成的主接口。与 ``generate()`` 的区别在于它
        返回一个迭代器，支持 lazy 消费（边生成边落盘），不需要把所有结果
        同时塞进内存。

        Yields:
            HandGenerationResult: 每次成功生成的轻量结果包。
        """

        if self.cfg.sampling_strategy == "enumerate":
            raise NotImplementedError("enumerate batch generation is deferred beyond the first pre-made slice.")

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
        # ── 循环体 ──
        #   success_count = 0
        #   attempt_count = 0
        #   while success_count < N:
        #     attempt_count += 1
        #     result = self.generate()              # 单次联合采样
        #     if result is not None:
        #       yield result
        #       success_count += 1
        #     if attempt_count > N × max_attempt_ratio:  # 防止无限循环
        #       raise RuntimeError("too many rejections")
        #
        # ── 关键性质 ──
        #   每次 generate() 独立从联合分布采样 (pre-made × post-mutate)，
        #   不存在笛卡尔展开，产物数量严格由 N 控制。
        #   连续工具（link_scale、mount_perturb、limit_tweak）的随机性来自
        #   其 sigma 参数，不会重复；离散工具（joint_delete、finger_replace）
        #   每次随机选一种操作，而不是枚举所有可能。
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
        # ── 枚举体 ──
        #   pre_made_space = cfg.Made.enumerate()      # 返回所有离散 HandBuilderCfg 列表
        #   count = 0
        #   for builder_cfg in pre_made_space:
        #     hand = HandBuilder(builder_cfg).build()
        #     if hand is None: continue
        #     mutate_options = cfg.Mutate.enumerate(hand)  # 返回所有离散 mutate 方案
        #     for mutated_hand in mutate_options:
        #       if max_enumerate and count >= max_enumerate: return
        #       result = HandGenerationResult(hand_cfg=mutated_hand)
        #       if cfg.Validate: ...validate...
        #       yield result
        #       count += 1
        #
        # ── 爆炸风险说明 ──
        #   若 pre-made 有 P 个离散组合，post-mutate 有 M 个离散方案，
        #   总产物数 = P × M，在大空间下极易爆炸。
        #   建议：enumerate 策略仅用于 P × M ≤ 几百 的小规模对照实验；
        #   大规模数据集请使用 sample 策略。
        #
        # IDEA：两种策略的 API 对调用者完全透明（都是 yield 迭代器），
        # 切换只需修改 cfg.sampling_strategy，不需要改调用代码。

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
