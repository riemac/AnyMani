r"""后序变异流水线：把多个单体工具按顺序串联调度。

本模块是 `mutate/` 子包的顶层调度者，对应 `资产生产概略.png` 中
`pre-made → validator → HandCfgs → post-mutate → HandCfgs` 的 post-mutate 阶段。

分类说明
--------

- **结构类工具**（拓扑变化）：`joint_delete`、`finger_replace`
- **参数类工具**（纯参数修改）：`link_scale`、`limit_tweak`、`mount_perturb`
- **几何类工具**（末端几何替换）：`tip_replace`

设计说明
--------

### 流水线语义

`HandMutator` 按 `order` 字段声明的工具名顺序依次执行。每一步都传入上一步的
输出 `HandCfg`；若某步返回 `None`，流水线按 `on_reject` 策略处理：

- ``"abort"``：立即终止，整个 mutate 调用返回 `None`
- ``"skip"``：跳过该步，继续下一个工具

### 可选性

所有子工具配置字段默认为 `None`，表示"不启用该工具"。运行时只实例化非 `None`
的工具，并按 `order` 中出现的顺序执行（若 `order` 中列出了某个工具名但对应 cfg
为 `None`，则该名称被忽略）。

### 轻量校验钩子

流水线内置一个可选的"步间轻量校验钩子"（`step_validate`），在每步工具执行后
调用，用于提前拦截明显违反结构约束的中间态，避免让错误状态传递到后续工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...validator import HandValidator, HandValidatorCfg
from ._base import MutatorBase
from .finger_replace import FingerReplaceCfg, FingerReplaceMutator
from .joint_delete import JointDeleteCfg, JointDeleteMutator
from .limit_tweak import LimitTweakCfg, LimitTweakMutator
from .link_scale import LinkScaleCfg, LinkScaleMutator
from .mount_perturb import MountPerturbCfg, MountPerturbMutator
from .tip_replace import TipReplaceCfg, TipReplaceMutator

# 所有合法工具名（与 HandMutatorCfg 字段名一一对应）
_TOOL_KEYS = Literal[
    "joint_delete",
    "link_scale",
    "tip_replace",
    "limit_tweak",
    "mount_perturb",
    "finger_replace",
]


# ============================================================================
#  流水线配置类
# ============================================================================


@dataclass
class HandMutatorCfg(AssetCfgBase):
    r"""整手后序变异流水线配置。

    把多个单体工具按顺序串起来。未配置（保持 ``None``）的工具在执行时
    自动跳过；`order` 决定执行顺序。
    """

    class_type: type["HandMutator"] | None = None
    """关联的整手后序变异运行时类。"""

    joint_delete: JointDeleteCfg | None = None
    """关节删除工具配置；为 ``None`` 时不执行。"""

    link_scale: LinkScaleCfg | None = None
    """连杆长度缩放工具配置；为 ``None`` 时不执行。"""

    tip_replace: TipReplaceCfg | None = None
    """指尖替换工具配置；为 ``None`` 时不执行。"""

    limit_tweak: LimitTweakCfg | None = None
    """关节限位微调工具配置；为 ``None`` 时不执行。"""

    mount_perturb: MountPerturbCfg | None = None
    """挂载点扰动工具配置；为 ``None`` 时不执行。"""

    finger_replace: FingerReplaceCfg | None = None
    """整根手指替换工具配置；为 ``None`` 时不执行。"""

    order: tuple[str, ...] = (
        "joint_delete",
        "finger_replace",
        "link_scale",
        "tip_replace",
        "limit_tweak",
        "mount_perturb",
    )
    """工具执行顺序。默认先做结构类（拓扑），再做几何类，最后做参数类。
    若某工具名对应 cfg 为 ``None``，则该名称在执行时被自动忽略。"""

    on_reject: Literal["abort", "skip"] = "abort"
    """某步工具返回 ``None`` 时的处理策略。``abort`` 立即返回 ``None``；
    ``skip`` 跳过该步继续下一个工具。"""

    step_validate: bool = False
    """是否在每步工具执行后插入轻量步间校验。开启可提前拦截中间态错误，
    但会增加调用开销。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandMutator


# ============================================================================
#  流水线运行时壳
# ============================================================================


class HandMutator(MutatorBase):
    r"""整手后序变异流水线运行时类。

    按 `cfg.order` 依次实例化并执行各单体工具，把 `HandCfg` 逐步传递并更新。
    """

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""按流水线配置串联执行后序变异工具。

        Args:
            target (HandCfg): 待变异的整手配置（通常来自 pre-made 阶段产物）。

        Returns:
            HandCfg | None: 变异后的整手配置；若流水线在 ``on_reject="abort"`` 时
            中途遇到拒绝，则返回 ``None``。
        """

        current = target  # 流水线始终把上一步输出显式传给下一步，不在层间隐藏状态
        validator = HandValidator(HandValidatorCfg()) if self.cfg.step_validate else None

        for tool_key, tool in self._build_tools():
            result = tool.mutate(current)
            if result is None:
                if self.cfg.on_reject == "abort":
                    return None
                continue  # `skip` 语义：保留当前状态，直接进入下一工具

            if validator is not None:
                validation = validator.validate(result)  # 这里用默认 validator 做轻量结构闸门
                if not validation:
                    if self.cfg.on_reject == "abort":
                        return None
                    continue

            current = result  # 只有通过 mutate + 可选 step_validate 后才推进流水线状态

        return current

        # TODO:算法之一（pipeline orchestration）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.order: 工具执行顺序元组（字符串名列表）
        #   cfg.{tool_key}: 各工具的配置（None 则跳过）
        #   cfg.on_reject: "abort" | "skip"
        #   cfg.step_validate: 是否启用步间校验
        #
        # 输出：HandCfg | None
        #
        # ── 工具实例化 ──
        #   _TOOL_MAP = {
        #     "joint_delete":  (JointDeleteMutator,  cfg.joint_delete),
        #     "link_scale":    (LinkScaleMutator,    cfg.link_scale),
        #     "tip_replace":   (TipReplaceMutator,   cfg.tip_replace),
        #     "limit_tweak":   (LimitTweakMutator,   cfg.limit_tweak),
        #     "mount_perturb": (MountPerturbMutator, cfg.mount_perturb),
        #     "finger_replace":(FingerReplaceMutator,cfg.finger_replace),
        #   }
        #   active_tools = [(key, MutClass(tool_cfg))
        #                   for key in order
        #                   if _TOOL_MAP[key][1] is not None]
        #
        # ── 流水线执行 ──
        #   current = target
        #   for key, tool in active_tools:
        #     result = tool.mutate(current)
        #     if result is None:
        #       if on_reject == "abort": return None
        #       else:  # "skip"
        #         continue  # current 不更新，继续下一步
        #     current = result
        #     if step_validate:
        #       # 步间轻量校验（如检查全局 joint/link 唯一性）
        #       # 若校验失败：按 on_reject 处理
        #       pass
        #   return current
        #
        # ── 与 preset 的交叉验证 ──
        #   流水线层不直接校验 preset 约束，但建议在 step_validate 里插入
        #   "结构改变类工具后的轻量一致性检查"；参数类工具的 preset 约束
        #   由 validator 阶段统一检查。
        #
        # IDEA：流水线层应尽量薄，只负责编排顺序和拒绝策略，不吞掉底层工具
        # 的语义。单体工具的逻辑不应泄露进流水线层。

    def _build_tools(self) -> list[tuple[str, MutatorBase]]:
        r"""根据 cfg.order 和各工具配置，构造有序的 (key, mutator) 列表。

        Returns:
            list[tuple[str, MutatorBase]]: 按执行顺序排列的 (工具名, 工具实例) 对。
        """

        tool_table = {
            "joint_delete": (JointDeleteMutator, self.cfg.joint_delete),
            "link_scale": (LinkScaleMutator, self.cfg.link_scale),
            "tip_replace": (TipReplaceMutator, self.cfg.tip_replace),
            "limit_tweak": (LimitTweakMutator, self.cfg.limit_tweak),
            "mount_perturb": (MountPerturbMutator, self.cfg.mount_perturb),
            "finger_replace": (FingerReplaceMutator, self.cfg.finger_replace),
        }

        tools: list[tuple[str, MutatorBase]] = []
        for key in self.cfg.order:
            if key not in tool_table:
                raise ValueError(f"Unknown mutator key in order: {key!r}")
            tool_class, tool_cfg = tool_table[key]
            if tool_cfg is None:
                continue  # 未启用的工具不进入执行列表
            tools.append((key, tool_class(tool_cfg)))
        return tools

        # TODO:算法之二（tool instantiation）
        # ────────────────────────────────────────
        # 依据 cfg.order 顺序，跳过 cfg.{key} 为 None 的工具，
        # 构造 (key, ToolClass(tool_cfg)) 列表并返回。


__all__ = ["HandMutatorCfg", "HandMutator"]
