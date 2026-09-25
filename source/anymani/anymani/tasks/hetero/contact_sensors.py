'Configure per-link object-filtered ContactSensors for canonical heterogeneous hands and expose force reductions.'

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
    'Create one filtered ContactSensor for a robot link against DexCube. Disable physics-rate history and air-time; the task keeps one shared 20 Hz EMA. Enable friction so force includes normal and tangential components, and retain up to 64 contact records per prim to match N000 capacity and avoid truncating palm contact pairs.'

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
    'Install the fixed set of 24 single-link sensors into the scene config.'

    for sensor_name, link_name in layout.scene_sensor_link_pairs:
        setattr(scene_cfg, sensor_name, make_contact_sensor_cfg(link_name))


def sensor_contact_magnitude(env: Any, sensor_name: str) -> torch.Tensor:
    'Take the vector norm per body/filter pair, then max across pairs so opposing forces do not cancel.'

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
