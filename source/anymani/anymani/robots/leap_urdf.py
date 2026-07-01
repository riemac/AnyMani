r"""Official LEAP hand URDF articulation cfg.

本模块提供一条与 `robots.leap.LEAP_HAND_CFG` 并列的 official LEAP runtime 路线：

- `robots.leap.LEAP_HAND_CFG`：历史 USD / edited-USD 路线；
- `robots.leap_urdf.LEAP_HAND_URDF_CFG`：直接从 official URDF 通过 IsaacLab URDF importer
  转换并 spawn。

这两条路线必须分开命名，避免训练配置里看不出当前使用的是 USD 还是 URDF。当前
GM LEAP 对照实验希望以 URDF 为真源，因此 task cfg 应显式 import
`LEAP_HAND_URDF_CFG`。
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

LEAP_HAND_URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "hands" / "leap_hand" / "leap_hand_right.urdf"
r"""Official LEAP right-hand URDF 路径；该文件保留原始 link / joint 名称与 collision boxes。"""


LEAP_HAND_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(LEAP_HAND_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=False,
        make_instanceable=True,
        collision_from_visuals=False,
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=3.0,
                damping=0.1,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1.0e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.5, 0.5, -0.5, 0.5),
        joint_pos={"a_.*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
r"""Official LEAP hand articulation cfg backed by URDF importer.

关键 importer 语义：

- `merge_fixed_joints=False`：保留 URDF 中 `palm_lower`、`fingertip`、`thumb_fingertip`
  等 link 名，保证 `GmContactSensorLayout` 的 per-link sensor 路径仍可解析；
- `activate_contact_sensors=True`：在 importer 生成的 rigid bodies 上挂
  `PhysxContactReportAPI`，否则 IsaacLab `ContactSensorCfg` 会在初始化时报
  “could not find any bodies with contact reporter API”；
- `fix_base=True` / `fix_root_link=True`：保持手掌基座固定，符合当前 hand-in-place
  in-hand reorientation 对照实验。
"""


__all__ = ["LEAP_HAND_URDF_CFG", "LEAP_HAND_URDF_PATH"]
