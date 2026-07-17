r"""GM tactile rotation 的 ADR 数值端点。

本模块只保存跨 config/curriculum 共享的不可变实验常数，避免 startup nominal material
与 curriculum 第 0 档分别手写后发生漂移。材料三元组始终按
`(static friction, dynamic friction, restitution)` 解释；yaw half-width 单位为 rad。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialADRRanges:
    r"""单个接触角色的材料范围，三个字段均为闭区间 `(lower, upper)`。"""

    static: tuple[float, float]  # 静摩擦系数 $\mu_s$，无量纲
    dynamic: tuple[float, float]  # 动摩擦系数 $\mu_d$，无量纲，采样后约束 $\mu_d\leq\mu_s$
    restitution: tuple[float, float]  # 恢复系数 $e$，无量纲

    def as_dict(self) -> dict[str, tuple[float, float]]:
        r"""返回 curriculum interpolation 使用的独立字典副本。"""

        return {
            "static": self.static,
            "dynamic": self.dynamic,
            "restitution": self.restitution,
        }


# 第 0 档是高摩擦、完全非弹性的稳定接触盆地；startup 与 curriculum 必须引用同一对象。
GM_ADR_OBJECT_MATERIAL_INITIAL = MaterialADRRanges((1.0, 1.0), (1.0, 1.0), (0.0, 0.0))
GM_ADR_HAND_MATERIAL_INITIAL = MaterialADRRanges((1.0, 1.0), (1.0, 1.0), (0.0, 0.0))

# 第 25 档沿用 LEAP endpoint；它表达 robustness stress range，不宣称是实物材料标定分布。
GM_ADR_OBJECT_MATERIAL_FINAL = MaterialADRRanges((0.3, 1.5), (0.3, 1.5), (0.0, 0.5))
GM_ADR_HAND_MATERIAL_FINAL = MaterialADRRanges((1.0, 1.0), (1.0, 1.0), (0.0, 0.5))

# Body yaw 在 25 档从 0 线性展开到完整 $[-\pi,\pi]$；roll/pitch 不进入本任务初态分布。
GM_ADR_OBJECT_BODY_YAW_FINAL = math.pi


__all__ = [
    "GM_ADR_HAND_MATERIAL_FINAL",
    "GM_ADR_HAND_MATERIAL_INITIAL",
    "GM_ADR_OBJECT_BODY_YAW_FINAL",
    "GM_ADR_OBJECT_MATERIAL_FINAL",
    "GM_ADR_OBJECT_MATERIAL_INITIAL",
    "MaterialADRRanges",
]
