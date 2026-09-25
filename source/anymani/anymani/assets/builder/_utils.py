"Normalizes builder inputs, all of which are already in SI units. Lengths are meters; the unit helpers perform authoring-side conversions, and builders never infer centimeters from value magnitude."

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from ..asset_schema_core import JointLimitCfg, JointPropertiesCfg, Vector6, _ensure_tuple


def _to_si(value: float | int) -> float:

    return float(value)


def _normalize_pose_value(value: float | Sequence[float] | None, *, field_name: str) -> Vector6:

    if value is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(value, (int, float)):
        return (0.0, _to_si(value), 0.0, 0.0, 0.0, 0.0)
    packed = _ensure_tuple(value, length=len(value), field_name=field_name)
    if len(packed) == 2:
        return (0.0, _to_si(packed[0]), _to_si(packed[1]), 0.0, 0.0, 0.0)
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

    if not values:
        return [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"{field_name} length must be {count}, got {len(values)}")
    return [_normalize_pose_value(value, field_name=f"{field_name}[{idx}]") for idx, value in enumerate(values)]


def _normalize_joint_limits(values: Sequence[Any] | None, *, count: int) -> list[JointLimitCfg | None]:

    if not values:
        return [(-3.141592653589793, 3.141592653589793) for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"joint_limits length must be {count}, got {len(values)}")
    limits: list[JointLimitCfg | None] = []
    for value in values:
        if value is None:
            limits.append(None)
        elif isinstance(value, JointLimitCfg):
            limits.append(value.copy())
        elif isinstance(value, Mapping):
            limits.append(JointLimitCfg(**dict(value)))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            low, high = _ensure_tuple(value, length=2, field_name="joint_limits")
            limits.append(JointLimitCfg(lower=float(low), upper=float(high)))
        else:
            raise TypeError(f"Unsupported joint limit value: {value!r}")
    return limits


def _normalize_joint_properties(values: Sequence[Any] | None, *, count: int) -> list[JointPropertiesCfg | None]:

    if not values:
        return [None for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"joint_properties length must be {count}, got {len(values)}")
    properties: list[JointPropertiesCfg | None] = []
    for value in values:
        if value is None:
            properties.append(None)
        elif isinstance(value, JointPropertiesCfg):
            properties.append(value.copy())
        elif isinstance(value, Mapping):
            properties.append(JointPropertiesCfg(**dict(value)))
        else:
            raise TypeError(f"Unsupported joint_properties value: {value!r}")
    return properties


def _mesh_length(mesh: dict[str, Any]) -> float:

    if mesh["type"] == "box":
        return float(mesh["length"])
    if mesh["type"] == "cylinder":
        return float(mesh["length"])
    raise ValueError(f"Unsupported mesh type for length inference: {mesh['type']}")


def _mesh_cross_section(mesh: dict[str, Any]) -> tuple[float, float]:

    if mesh["type"] == "box":
        return float(mesh["width"]), float(mesh["height"])
    radius = float(mesh["radius"])
    diameter = radius * 2.0
    return diameter, diameter


def _build_box_mesh(*, length: float, width: float, height: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:

    return {
        "type": "box",
        "length": length,
        "width": width,
        "height": height,
        "offset": offset,
        "center_on_joint": center_on_joint,
    }


def _build_cylinder_mesh(*, length: float, radius: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:

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
    "_normalize_joint_properties",
    "_mesh_length",
    "_mesh_cross_section",
    "_build_box_mesh",
    "_build_cylinder_mesh",
]
