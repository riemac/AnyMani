r"""Canonical heterogeneous scene的per-link object-filtered ContactSensor装配与force读取。"""

from __future__ import annotations

from typing import Any

import torch

from .contact_layout import HeterogeneousContactLayout


def make_contact_sensor_cfg(
    link_name: str,
    *,
    robot_prim_path: str = "{ENV_REGEX_NS}/Robot",
    object_prim_path: str = "{ENV_REGEX_NS}/object",
):
    r"""构造一个robot link对DexCube的一对多filtered contact sensor。

    Physics-rate history与air-time关闭；任务只维护20 Hz共享EMA。Friction开启，使force magnitude包含normal与
    tangential分量；每prim保留最多64条contact records，与N000容量一致，避免复杂掌托接触截断pair reduction。
    """

    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(
        prim_path=f"{robot_prim_path}/{link_name}",
        filter_prim_paths_expr=[object_prim_path],
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=False,
    )


def install_contact_sensors(scene_cfg: Any, layout: HeterogeneousContactLayout) -> None:
    r"""把固定24个single-link sensors安装到InteractiveSceneCfg实例。"""

    for sensor_name, link_name in layout.scene_sensor_link_pairs:
        setattr(scene_cfg, sensor_name, make_contact_sensor_cfg(link_name))


def sensor_contact_magnitude(env: Any, sensor_name: str) -> torch.Tensor:
    r"""对单sensor的body/filter pairs先取$\|F\|_2$再amax，避免向量相消。"""

    sensor = env.scene[sensor_name]
    force_w = getattr(sensor.data, "force_matrix_w", None)
    if force_w is None:
        force_w = getattr(sensor.data, "net_forces_w", None)
    if force_w is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} exposes no force tensor")
    total_force_w = torch.nan_to_num(force_w, nan=0.0)
    friction_w = getattr(sensor.data, "friction_forces_w", None)
    if friction_w is not None:
        total_force_w = total_force_w + torch.nan_to_num(friction_w, nan=0.0)
    magnitude = torch.linalg.vector_norm(total_force_w, dim=-1)
    if magnitude.ndim > 1:
        magnitude = magnitude.amax(dim=tuple(range(1, magnitude.ndim)))
    return magnitude


__all__ = ["install_contact_sensors", "make_contact_sensor_cfg", "sensor_contact_magnitude"]
