r"""关节限位微调工具：在已有 HandCfg 上对 joint limit 做小范围参数修改。

对应 `前后序.png` 中归属后序的 `关节限位微调`，理由是"纯参数修改"。

这是后序工具里侵入性最小的操作：不改变拓扑，不改变几何，只修改 `JointCfg.limit`
的 `lower` / `upper` 数值。

设计说明
--------

### 调整模式

提供两种模式：

- ``"absolute"``：直接在 lower/upper 上加一个绝对偏移 $\delta$（弧度）
- ``"relative"``：以现有限位范围的 $\pm p\%$ 施加比例扰动

### 对称 vs 非对称扰动

默认允许 lower/upper 独立采样（非对称），适合非对称关节（如拇指）。
可通过 `symmetric` 参数强制 lower 和 upper 做大小相等、方向相反的扰动。

### 裁剪保护

最终结果必须满足 `lower < upper`；若扰动后违反此约束，回退至原值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ._base import MutatorBase
from ._distribution import ScalarDistributionCfg


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class LimitTweakCfg(AssetCfgBase):
    r"""关节限位微调工具配置。"""

    class_type: type["LimitTweakMutator"] | None = None
    """关联的运行时类。"""

    target_joints: tuple[str, ...] = ()
    """需要调整限位的关节名称集合；空元组表示作用于全部非固定关节。"""

    mode: Literal["absolute", "relative"] = "absolute"
    """调整模式。``absolute`` 在 lower/upper 上叠加绝对偏移（rad）；
    ``relative`` 以限位范围的比例施加扰动。"""

    delta_distribution: ScalarDistributionCfg = field(
        default_factory=lambda: ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.05)
    )
    """单个限位扰动的基础分布。"""

    symmetric: bool = False
    """是否强制对称扰动（lower 和 upper 采样大小相等、方向相反）。
    对非对称关节（如拇指）推荐关闭（False）。"""

    clip: float | None = 0.5
    """扰动幅度裁剪上限（rad）；为 ``None`` 时不额外裁剪。防止单步扰动过大。"""

    per_joint_delta_distribution: dict[str, ScalarDistributionCfg] = field(default_factory=dict)
    """可选的每 joint 单独分布覆盖；键为 joint 名。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = LimitTweakMutator
class LimitTweakMutator(MutatorBase):
    r"""关节限位微调运行时壳。

    在已构建好的 `HandCfg` 上对目标关节的 `limit.lower` / `limit.upper`
    做小范围参数修改，不改变拓扑和几何。
    """

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, ScalarDistributionCfg]:
        r"""给出当前 hand 上每个目标 joint 的限位扰动分布。"""

        target_names = set(self.cfg.target_joints)
        distribution_plan: dict[str, ScalarDistributionCfg] = {}
        for joint in target.iter_joints():
            if joint.joint_type == "fixed" or joint.limit is None:
                continue
            if target_names and joint.name not in target_names:
                continue

            joint_range = float(joint.limit.upper) - float(joint.limit.lower)
            distribution_cfg = self.cfg.per_joint_delta_distribution.get(joint.name, self.cfg.delta_distribution).copy()
            if self.cfg.mode == "relative":
                if distribution_cfg.kind == "normal":
                    distribution_cfg.mean *= joint_range
                    distribution_cfg.sigma *= joint_range
                elif distribution_cfg.kind == "uniform":
                    distribution_cfg.low *= joint_range
                    distribution_cfg.high *= joint_range
                else:
                    distribution_cfg.value *= joint_range
            if self.cfg.clip is not None:
                clip = float(self.cfg.clip)
                distribution_cfg.clip_min = -clip if distribution_cfg.clip_min is None else distribution_cfg.clip_min
                distribution_cfg.clip_max = clip if distribution_cfg.clip_max is None else distribution_cfg.clip_max

            if self.cfg.symmetric:
                distribution_plan[f"{joint.name}::delta"] = distribution_cfg
            else:
                distribution_plan[f"{joint.name}::lower"] = distribution_cfg.copy()
                distribution_plan[f"{joint.name}::upper"] = distribution_cfg.copy()
        return distribution_plan

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行关节限位微调。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 限位微调后的整手配置。
        """

        mutated = target.copy()
        target_names = set(self.cfg.target_joints)
        sampled = sampled_params or {}

        for joint in mutated.iter_joints():
            # fixed joint 没有限位语义；limit 缺失时也不做“自动补限位”的越权操作。
            if joint.joint_type == "fixed" or joint.limit is None:
                continue
            if target_names and joint.name not in target_names:
                continue

            lower = float(joint.limit.lower)
            upper = float(joint.limit.upper)
            if self.cfg.symmetric:
                delta_lower = float(sampled.get(f"{joint.name}::delta", 0.0))
                delta_upper = -delta_lower
            else:
                delta_lower = float(sampled.get(f"{joint.name}::lower", 0.0))
                delta_upper = float(sampled.get(f"{joint.name}::upper", 0.0))

            lower_new = lower + delta_lower  # $l' = l + \delta_l$
            upper_new = upper + delta_upper  # $u' = u + \delta_u$

            # 保护约束：若新限位区间塌缩或反向，则回退到原值。
            if lower_new >= upper_new:
                continue

            joint.limit = joint.limit.replace(lower=lower_new, upper=upper_new)

        return mutated

        # TODO:算法之一（joint limit perturbation）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.target_joints: 目标关节名集合（空 = 全部非固定关节）
        #   cfg.mode: "absolute" | "relative"
        #   cfg.sigma: 全局扰动强度
        #   cfg.symmetric: 是否强制对称
        #   cfg.clip: 单步扰动幅度裁剪
        #   cfg.per_joint_sigma: 每关节 sigma 覆盖
        #
        # 输出：HandCfg（深拷贝 + 修改 joint.limit）
        #
        # ── 对每个目标 joint j ──
        #   σ_j = per_joint_sigma.get(j.name, sigma)
        #
        #   [absolute 模式]
        #     δ_lower ~ N(0, σ_j)，裁剪到 [-clip, clip]
        #     δ_upper ~ N(0, σ_j)，裁剪到 [-clip, clip]
        #     若 symmetric=True：δ_upper = -δ_lower
        #     lower_new = j.limit.lower + δ_lower
        #     upper_new = j.limit.upper + δ_upper
        #
        #   [relative 模式]
        #     range_j = j.limit.upper - j.limit.lower
        #     δ_lower ~ N(0, σ_j × range_j)，裁剪
        #     δ_upper ~ N(0, σ_j × range_j)，裁剪
        #     若 symmetric=True：δ_upper = -δ_lower
        #     lower_new = j.limit.lower + δ_lower
        #     upper_new = j.limit.upper + δ_upper
        #
        # ── 约束保护 ──
        #   若 lower_new >= upper_new，回退至原值（不修改该关节）
        #   若 j.joint_type == "fixed"，跳过（固定关节无 limit 语义）
        #
        # ── 重建 HandCfg ──
        #   深拷贝 target，修改目标 joint 的 limit，返回新对象
        #
        # ── 与 preset 的交叉验证 ──
        #   微调后的限位应在 preset 规定的允许范围内；
        #   当前草案不强制校验，留给 validator 阶段检查。
        #
        # IDEA：关节限位微调是完全参数级的操作，对仿真稳定性影响较小；
        # 适合作为批量枚举实验时的"末尾扰动层"叠加使用。


__all__ = ["LimitTweakCfg", "LimitTweakMutator"]
