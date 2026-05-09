"""assets 子系统统一单位辅助。

本模块承载 `assets/` 子系统的人手录入单位 contract：

1. 裸 `float` 一律按 SI 解释；
2. 若作者希望在配置 / preset 中显式标注量纲，可使用这里的纯函数 helper；
3. helper 只在 authoring 阶段提升可读性，运行时下游模块统一只消费已经归一化后的 `float`。

当前首批统一的量有四类：

- 长度：米 / 厘米 / 毫米；
- 角度：弧度 / 角度；
- 质量：千克 / 克；
- 密度：`kg/m^3` / `g/cm^3`。

本轮暂不提供 helper、但仍在统一 contract 中明确按 SI 或无量纲解释的字段包括：

- `effort`：按国际单位制力矩/力解释；
- `velocity`：按国际单位制角速度/线速度解释；
- `friction`：沿用下游 schema 的 plain `float` 语义；
- RGBA / scale ratio：无量纲。

# NOTE:
这里故意不引入 `Quantity` 一类对象。原因不是“做不出来”，而是当前 `assets`
流水线已经明确收敛为：schema / builder / generator / validator / exporter
内部统一处理 plain `float`。只要 authoring 侧能清楚、显式地做单位换算，
就不需要把 runtime contract 搞复杂；`assets.units` 因而成为 authoring 侧
唯一推荐的单位真源。
"""

from __future__ import annotations

import math


def m(value: float) -> float:
    """米制直写。

    当一组长度参数里同时出现 `cm(...)` / `mm(...)` / `m(...)` 时，
    显式写 `m(...)` 能让读者一眼看出“这里不是忘了写单位，而是故意按 SI 直写”。
    """

    return float(value)


def cm(value: float) -> float:
    """厘米转米。"""

    return float(value) / 100.0


def mm(value: float) -> float:
    """毫米转米。"""

    return float(value) / 1000.0


def rad(value: float) -> float:
    """弧度直写。"""

    return float(value)


def deg(value: float) -> float:
    """角度转弧度。"""

    return math.radians(float(value))


def kg(value: float) -> float:
    """千克直写。"""

    return float(value)


def g(value: float) -> float:
    """克转千克。"""

    return float(value) / 1000.0


def kg_m3(value: float) -> float:
    """`kg/m^3` 直写。"""

    return float(value)


def g_cm3(value: float) -> float:
    r"""`g/cm^3` 转 `kg/m^3`。

    换算关系：
    $$
    1\ \text{g/cm}^3 = 1000\ \text{kg/m}^3.
    $$
    """

    return float(value) * 1000.0


__all__ = [
    "m",
    "cm",
    "mm",
    "rad",
    "deg",
    "kg",
    "g",
    "kg_m3",
    "g_cm3",
]
