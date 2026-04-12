r"""Builder 内部共用的小工具函数。

本模块只放“多种 builder 都可能复用、且不携带具体 hand family 语义”的轻量辅助：

- 单位转换
- 偏移/位姿规范化
- 逐关节限位规范化
- primitive recipe 的轻量构造
- 从 primitive recipe 里读取长度与横截面信息

之所以放在 `assets/builder/_utils.py`，而不是 `assets/tool/`，是因为这些函数
属于 builder 内部实现细节，不是面向用户的 recipe/runner 工具层接口。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..asset_schema_core import JointLimitCfg, Vector3, Vector6, _ensure_tuple


def _to_si(value: float | int) -> float:
    r"""把“疑似厘米输入”转换为米制。

    当前很多 preset 仍沿用注释里的 cm 直觉值，因此这里做一个轻量兼容：

    - 若量级明显更像 cm，则除以 100；
    - 若本来就是 m，则原样保留。

    这是一个工程约定，不是严格的单位推断器。其目的只是让 preset 与参数化输入
    在首轮实现里更顺滑地共存。
    """

    value = float(value)  # 统一先压成 float，避免 dataclass 里混入 int 语义
    return value / 100.0 if abs(value) > 0.5 else value  # 经验规则：大于 0.5 更像 cm 量纲


def _normalize_pose_value(value: float | Sequence[float] | None, *, field_name: str) -> Vector6:
    r"""把偏移输入统一解析为 6D pose。

    支持三种常用写法：

    - `float`：只写沿生长方向的 $y$ 偏移
    - `xyz`
    - `xyzrpy`

    对 thumb 的 `CMC1` 特例，还额外接受 `(y, z)` 二元写法。
    """

    if value is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 空值视作零位姿
    if isinstance(value, (int, float)):
        return (0.0, _to_si(value), 0.0, 0.0, 0.0, 0.0)  # 标量默认只表达沿 $y$ 的偏移
    packed = _ensure_tuple(value, length=len(value), field_name=field_name)  # 接受 tuple/list/Vector 等宽松写法
    if len(packed) == 2:
        return (0.0, _to_si(packed[0]), _to_si(packed[1]), 0.0, 0.0, 0.0)  # CMC1 常用 $(y, z)$ 二元偏移
    if len(packed) == 3:
        return (_to_si(packed[0]), _to_si(packed[1]), _to_si(packed[2]), 0.0, 0.0, 0.0)
    if len(packed) == 6:
        return (
            _to_si(packed[0]),
            _to_si(packed[1]),
            _to_si(packed[2]),
            float(packed[3]),
            float(packed[4]),
            float(packed[5]),
        )
    raise ValueError(f"{field_name} must be scalar / yz / xyz / xyzrpy, got {value!r}")


def _normalize_pose_list(values: Sequence[Any], *, count: int, field_name: str) -> list[Vector6]:
    r"""把多段 mesh 偏移规范为定长 `list[Vector6]`。"""

    if not values:
        return [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(count)]  # 空输入等价于所有 joint mesh 都贴标准位
    if len(values) != count:
        raise ValueError(f"{field_name} length must be {count}, got {len(values)}")
    return [_normalize_pose_value(value, field_name=f"{field_name}[{idx}]") for idx, value in enumerate(values)]


def _normalize_joint_limits(values: Sequence[Any] | None, *, count: int) -> list[JointLimitCfg | None]:
    r"""把逐关节限位输入规范化。"""

    if not values:
        return [(-3.141592653589793, 3.141592653589793) for _ in range(count)]  # 首轮用对称大范围限位兜底
    if len(values) != count:
        raise ValueError(f"joint_limits length must be {count}, got {len(values)}")
    limits: list[JointLimitCfg | None] = []
    for value in values:
        if value is None:
            limits.append(None)  # 允许单个关节显式写成“无额外限位”
        elif isinstance(value, JointLimitCfg):
            limits.append(value.copy())  # 避免共享同一个可变对象
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            low, high = _ensure_tuple(value, length=2, field_name="joint_limits")
            limits.append(JointLimitCfg(lower=float(low), upper=float(high)))
        else:
            raise TypeError(f"Unsupported joint limit value: {value!r}")
    return limits


def _mesh_length(mesh: dict[str, Any]) -> float:
    r"""从 primitive recipe 中读取沿生长方向的长度。"""

    if mesh["type"] == "box":
        return float(mesh["length"])  # box 的主长度
    if mesh["type"] == "cylinder":
        return float(mesh["length"])  # cylinder 的轴向长度
    raise ValueError(f"Unsupported mesh type for length inference: {mesh['type']}")


def _mesh_cross_section(mesh: dict[str, Any]) -> tuple[float, float]:
    r"""从 primitive recipe 中读取横截面尺寸 $(w, h)$。"""

    if mesh["type"] == "box":
        return float(mesh["width"]), float(mesh["height"])  # box 直接读宽高
    radius = float(mesh["radius"])
    diameter = radius * 2.0
    return diameter, diameter  # cylinder 用直径近似横截面宽高


def _build_box_mesh(*, length: float, width: float, height: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:
    r"""构造 box primitive recipe。

    这里只生成 canonical recipe，不直接生成 `JointCfg`。真正的 geometry / inertia
    lower 交给 `PrimJointBuilder` 处理。
    """

    return {
        "type": "box",
        "length": length,
        "width": width,
        "height": height,
        "offset": offset,
        "center_on_joint": center_on_joint,
    }


def _build_cylinder_mesh(*, length: float, radius: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:
    r"""构造 cylinder primitive recipe。"""

    return {
        "type": "cylinder",
        "length": length,
        "radius": radius,
        "offset": offset,
        "center_on_joint": center_on_joint,
    }


__all__ = [
    "_to_si",
    "_normalize_pose_value",
    "_normalize_pose_list",
    "_normalize_joint_limits",
    "_mesh_length",
    "_mesh_cross_section",
    "_build_box_mesh",
    "_build_cylinder_mesh",
]
