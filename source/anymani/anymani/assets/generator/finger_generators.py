r"""手指级生成器的声明式配置类和运行时类。

这是 generator 体系中最关键的一层之一，因为你当前明确希望把：

- preset finger 的扩展
- joint delete
- delete 后的 finger regroup

放到 **Pre** 阶段，而不是拖到 `HandCfg` mutator 里再做。

设计说明
--------

### Pre 与 Post 的边界

- **Pre**：操作 `FingerBuilderCfg` 一类的 recipe，对 preset finger 做结构枚举；
- **Post**：操作 canonical `FingerCfg`，对已建好的 finger 做 tip / 几何 / limit 派生。

### 对 Get-Zero 的吸收方式

参考 `get_zero/get_zero/rl/scripts/gen_leap_assets.py` 的思想，finger-level pre 会把
“可保留的 joint role 子序列”当作候选空间来源；但这里不直接用 URDF link-name 字符串做主 IR，
而是围绕 preset recipe 的 canonical joint role 表达 delete pattern。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, FingerCfg
from ..asset_builders import FingerBuilderCfg
from ..asset_generator import Generator, GeneratorCfg


# ============================================================================
#  手指级前序 / 后序配置
# ============================================================================


@dataclass
class FingerPreGeneratorCfg(GeneratorCfg):
    r"""手指级前序生成器配置。"""

    class_type: type["FingerPreGenerator"] | None = None
    """关联的手指级前序生成器类。"""

    canonical_joint_roles: tuple[str, ...] = ()
    """canonical joint role 顺序。

    若为空，则以后实现时退回当前 preset 自己声明的顺序；若给出，则用它作为
    delete pattern 的统一参照系。"""

    delete_patterns: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    """preset family 到 delete pattern 列表的映射。

    每个 tuple 表示“保留的 canonical role 子序列”，例如可表达：
    - 完整链
    - 删除 distal joint 后的链
    - 仅保留末端两段的链
    但它刻意不绑定具体 URDF link 名。"""

    preserve_tip_role: bool = True
    """delete 之后是否强制保留 tip role。默认保留。"""

    max_candidates_per_preset: int | None = None
    """每个 preset 最多保留多少 finger-level 候选。`None` 表示不截断。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerPreGenerator


@dataclass
class FingerPostGeneratorCfg(GeneratorCfg):
    r"""手指级后序生成器配置。"""

    class_type: type["FingerPostGenerator"] | None = None
    """关联的手指级后序生成器类。"""

    allow_tip_replace: bool = True
    """是否允许在 finger-level 替换 tip 方案。"""

    allow_joint_limit_mutation: bool = True
    """是否允许在 finger 内对关节限位做派生。"""

    max_mutations_per_finger: int = 2
    """单根 finger 最多允许的后序派生数。默认 2。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerPostGenerator


# ============================================================================
#  手指级前序 / 后序运行时类
# ============================================================================


class FingerPreGenerator(Generator):
    r"""手指级前序生成器。"""

    cfg: FingerPreGeneratorCfg

    def __init__(self, cfg: FingerPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[FingerBuilderCfg] | None:
        r"""从 preset finger recipe 生成一组前序候选。"""
        pass

        # TODO:算法之一（preset finger 的 joint delete + regroup）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为一个 preset `FingerBuilderCfg`，或其更上层拆出的 finger recipe。
        #   `canonical_joint_roles` — 统一的 delete pattern 参考系。
        #   `delete_patterns` — 针对 family/preset 的保留子序列集合。
        #
        # 输出：一组新的 `FingerBuilderCfg` 候选。
        #
        # ── 角色对齐 ──
        #   1. 若 cfg 显式给出了 `canonical_joint_roles`，则以后实现时按该顺序解释 preset recipe。
        #   2. 若为空，则退回 preset 的 canonical order。
        #
        # ── delete pattern 展开 ──
        #   1. 对当前 preset family 取出所有允许的保留子序列。
        #   2. 每个保留子序列都对应一个新的 finger recipe；被删掉的 role 不再进入 build。
        #   3. 若 `preserve_tip_role=True`，则以后实现时自动过滤掉没有 tip 的子序列。
        #
        # ── regroup 为 finger recipe ──
        #   1. 把保留下来的 role 子序列重新压缩成单根 finger 的 recipe。
        #   2. regroup 的结果仍然是 `FingerBuilderCfg`，而不是 `FingerCfg`。
        #
        # ── 与 Get-Zero 的交叉验证 ──
        #   参考 LEAP non-thumb 的启发式候选数：
        #   $$
        #   N_{cand} = \{\text{full},\ \text{drop-distal},\ \text{distal-only},\ \text{tip-only},\ \emptyset\}
        #   $$
        #   这里借鉴的是“每指 topology 候选”的思想，而不是其基于 link-name 的 XML 重写实现。
        #
        # IDEA：finger-level pre 的产物，正好可以被 hand-level pre 当成非拇指/拇指的候选池来 regroup。


class FingerPostGenerator(Generator):
    r"""手指级后序生成器。"""

    cfg: FingerPostGeneratorCfg

    def __init__(self, cfg: FingerPostGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[FingerCfg] | None:
        r"""在 canonical `FingerCfg` 上生成后序派生候选。"""
        pass

        # TODO:算法之一（finger-level 后序派生）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 canonical `FingerCfg`。
        #   可调参数：
        #     `allow_tip_replace` — 是否允许替换 tip 方案
        #     `allow_joint_limit_mutation` — 是否允许 finger 内局部 limit 派生
        #     `max_mutations_per_finger` — mutation budget
        #
        # 输出：一组派生后的 `FingerCfg`。
        #
        # ── tip 派生 ──
        #   1. 若允许 tip 替换，则以后实现时围绕末端 joint/link 的 tip 语义做有限集合替换。
        #   2. tip 派生应保持 parent/child chain 合法，不破坏 `FingerCfg` 的连续性约束。
        #
        # ── limit 派生 ──
        #   1. 对 finger 内多个关节的 limit 做协同调整。
        #   2. 调整后的 chain 应仍满足手指级解剖/运动学直觉。
        #
        # IDEA：finger-level post 只在已建好的 `FingerCfg` 上派生，不再回到 preset delete 空间。


__all__ = [
    "FingerPreGeneratorCfg",
    "FingerPostGeneratorCfg",
    "FingerPreGenerator",
    "FingerPostGenerator",
]