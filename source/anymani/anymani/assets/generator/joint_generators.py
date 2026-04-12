r"""关节级生成器的声明式配置类和运行时类。

本文件服务于生成层中的**叶级局部操作**，职责边界刻意收窄：

- **Pre**：在进入 finger-level delete / regroup 之前，统一 preset 中的 joint role 语义；
- **Post**：在 canonical `JointCfg` 上做局部参数派生，如 limit 扰动或几何尺度派生。

设计说明
--------

### 为什么 joint-level 不直接承接整根 finger 的 delete/regroup

`joint delete + regroup` 的真正组合爆炸发生在 finger/hand 两层，因此 joint-level
更适合作为 leaf helper：它只负责“单个 joint 应该如何被理解、如何被轻量派生”。

### 与 builder 的边界

joint-level generator 不负责重新定义 joint 的几何构造公式；那些仍然留在
`builder/joint_builders_primitive.py` 与 `builder/joint_builders_custom.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, JointCfg
from ..asset_builders import JointBuilderCfg
from ..asset_generator import Generator, GeneratorCfg


# ============================================================================
#  关节级前序生成配置
# ============================================================================


@dataclass
class JointPreGeneratorCfg(GeneratorCfg):
    r"""关节级前序生成器配置。

    主要用于把各 family preset 中不完全一致的 joint 命名/角色，规整到一个
    canonical role 词汇表下，方便 finger-level delete pattern 复用。
    """

    class_type: type["JointPreGenerator"] | None = None
    """关联的关节级前序生成器类。"""

    canonical_role_aliases: dict[str, str] = field(default_factory=dict)
    """别名到 canonical role 的映射。

    例如可把 `mcp_joint`、`pip_4` 之类 family-specific 名称，归并到更稳定的
    role 名词上；若为空，则默认退回 preset 自己声明的顺序。"""

    preserve_fixed_tip_joint: bool = True
    """是否保留固定指尖 joint 的角色语义。默认保留。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointPreGenerator


@dataclass
class JointPostGeneratorCfg(GeneratorCfg):
    r"""关节级后序生成器配置。"""

    class_type: type["JointPostGenerator"] | None = None
    """关联的关节级后序生成器类。"""

    allow_limit_mutation: bool = True
    """是否允许对关节限位做局部派生。"""

    geometry_scale_range: tuple[float, float] | None = None
    """局部几何尺度扰动范围。

    为 `None` 时表示不在 joint-level 直接改几何尺度；若给出 `(s_min, s_max)`，
    则表示以后实现时可对局部 primitive 参数做比例派生。"""

    skip_fixed_joints: bool = True
    """后序派生时是否默认跳过 fixed joint。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointPostGenerator


# ============================================================================
#  关节级前序 / 后序运行时类
# ============================================================================


class JointPreGenerator(Generator):
    r"""关节级前序生成器。"""

    cfg: JointPreGeneratorCfg

    def __init__(self, cfg: JointPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[JointBuilderCfg] | None:
        r"""生成关节级前序候选。

        Args:
            target (AssetCfgBase | None): 预期应为一个 joint recipe 或它的更上层 recipe。

        Returns:
            list[JointBuilderCfg] | None: 规整后的关节级 recipe 列表。
        """
        pass

        # TODO:算法之一（joint role 规范化）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 当前 preset 中的 joint recipe 或由上层拆下来的局部 joint 描述。
        #   `canonical_role_aliases` — family-specific 名称到 canonical role 的别名映射。
        #
        # 输出：角色语义规整后的 joint recipe 列表。
        #
        # ── 角色对齐 ──
        #   1. 若 cfg 提供了别名映射，则先把 joint 名称投影到 canonical role 空间。
        #   2. 若未提供，则以后实现时退回 preset 自身的 canonical joint order。
        #
        # ── 指尖语义保留 ──
        #   1. 若 `preserve_fixed_tip_joint=True`，则 tip 的 role 不因 `joint_type=fixed` 而被丢弃。
        #   2. 这样 finger-level delete pattern 可以显式写“是否保留 tip”。
        #
        # IDEA：joint-level pre 只是 helper，不在这里直接做 delete pattern 的组合枚举。


class JointPostGenerator(Generator):
    r"""关节级后序生成器。"""

    cfg: JointPostGeneratorCfg

    def __init__(self, cfg: JointPostGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[JointCfg] | None:
        r"""在 canonical `JointCfg` 上生成后序派生候选。"""
        pass

        # TODO:算法之一（joint-level 局部派生）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为一个 canonical `JointCfg`。
        #   可调参数：
        #     `allow_limit_mutation` — 是否允许调整 `q_min, q_max`
        #     `geometry_scale_range=(s_min, s_max)` — 局部几何尺度派生区间
        #     `skip_fixed_joints` — 是否跳过 fixed joint
        #
        # 输出：一组局部派生后的 `JointCfg`。
        #
        # ── 限位派生 ──
        #   1. 若允许 limit mutation，则以后实现时围绕原始区间做对称/非对称缩放。
        #   2. 需要保证 $q_{\min} \le q_{\max}$，且不破坏 fixed joint 的零自由度语义。
        #
        # ── 几何派生 ──
        #   1. 若给出 `geometry_scale_range`，则以后实现时对 child link 的 primitive 参数做比例扰动。
        #   2. 派生后应继续满足 collision-first 和 child-link 局部系可并集查询约束。
        #
        # IDEA：joint-level post 更像局部 leaf mutation，不应在这里重写整条 finger chain。


__all__ = [
    "JointPreGeneratorCfg",
    "JointPostGeneratorCfg",
    "JointPreGenerator",
    "JointPostGenerator",
]