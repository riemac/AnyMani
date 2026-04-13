"""preset 层单位辅助。

用户手工记录 preset 时，距离常常以 cm/mm 记，角度常常以 deg 记；
而 schema / URDF 最终都要求米制与弧度制。

这里提供一组极轻量的单位辅助函数，让 preset 文件既保留科研记录时的可读性，
又不会把单位换算散落在各个常量定义里。
"""

from __future__ import annotations

import math


def m(value: float) -> float:
    """米制直写；主要用于与 `cm/mm` 并列时保持语义对齐。"""

    return float(value)


def cm(value: float) -> float:
    """厘米 -> 米。"""

    return float(value) / 100.0


def mm(value: float) -> float:
    """毫米 -> 米。"""

    return float(value) / 1000.0


def deg(value: float) -> float:
    """角度 -> 弧度。"""

    return math.radians(float(value))


def rad(value: float) -> float:
    """弧度直写；与 `deg(...)` 并列时更直观。"""

    return float(value)


__all__ = ["m", "cm", "mm", "deg", "rad"]
