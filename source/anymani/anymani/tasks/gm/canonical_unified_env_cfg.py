r"""AnyMani canonical unified in-hand environment configuration。

该 variant 固定使用五个分层 mother：7-DOF right LEAP、9-DOF left Allegro、12-DOF
right mixed、14-DOF left LEAP 与 16-DOF right Allegro。HandSpawnAdapter 在 assets
真源上把它们 lower 为同一 16-DOF / 25-body PhysX articulation；本文件只组装 scene、
observation、reset 与 reward manager，不重新解析 URDF 或生成 geometry。

canonical actor flat observation 顺序固定为：

$$
[q_{16},\dot q_{16},\Delta a_{16},q^{min}_{16},q^{max}_{16},
  c_{6},p_o^h{}_{3},R_{ho}^{6},f_{tip}^{4},asset\_row_{1},m_{16}],
$$

总维度为 $116$。``asset_row`` 是离散 evidence routing，``m`` 是 `[env,joint]` active
mask；rl_games YAML 关闭 global input RMS，custom model 在边界处读取这两项。
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from anymani.assets.bank import HandBankCfg
from anymani.robots.hand_spawn import (
    CanonicalRuntimeCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)

from . import mdp as gm_mdp
from .contact_sensors import build_contact_sensor_layout_from_hand_spawn, install_contact_sensors
from .inhand_env_cfg import (
    GmActionsCfg,
    GmCommandsCfg,
    GmCurriculumCfg,
    GmInHandEnvCfg,
    GmInHandSceneCfg,
    GmRewardsCfg,
    GmTerminationsCfg,
    build_gm_hand_articulation_cfg,
)

CANONICAL_MOTHER_PATHS = (
    Path("source/anymani/anymani/assets/generated/2026-08-16_14-55-19/single_palm_leap/right_t3_i1_m3"),
    Path("source/anymani/anymani/assets/generated/2026-08-16_14-55-19/single_palm_allegro/left_t3_i2_m4"),
    Path(
        "source/anymani/anymani/assets/generated/2026-08-16_14-55-19/mixed/"
        "allegro_single_palm_allegro_thumb_index_leap_middle_allegro_ring/right_allegro_t3_i2_leap_m3_allegro_r4"
    ),
    Path("source/anymani/anymani/assets/generated/2026-08-16_14-55-19/single_palm_leap/left_t3_i3_m4_r4"),
    Path("source/anymani/anymani/assets/generated/2026-08-16_14-55-19/single_palm_allegro/right_t4_i4_m4_r4"),
)
"""固定五个 formal acceptance mother 的 source topology bundle 路径。"""

CANONICAL_ENVS_PER_MOTHER = 32
"""每个 mother 的默认 env replication；5 mother 共 160 env。"""


CANONICAL_HAND_SPAWN_CFG = HandSpawnCfg(
    bank=HandBankCfg(
        source_mode="mixed",
        selection_mode="explicit",
        containers=tuple(CANONICAL_MOTHER_PATHS),
        validate_mesh_relpaths=True,
        parse_visual_rgba=True,
        require_geometry_semantics=True,
    ),
    urdf=HandUrdfSpawnCfg(activate_contact_sensors=True),
    canonical_runtime=CanonicalRuntimeCfg(enabled=True, output_root="outputs"),
    asset_routing="round_robin",
    restore_visual_materials=False,
    validate_same_schema=True,
)
"""canonical single PhysX batch 的唯一 hand spawn source。"""

_CANONICAL_ADAPTER = HandSpawnAdapter(CANONICAL_HAND_SPAWN_CFG)
CANONICAL_ARTIFACTS = _CANONICAL_ADAPTER.canonical_artifacts
CANONICAL_ACTIVE_MASK_ROWS = tuple(artifact.routing.active_joint_mask for artifact in CANONICAL_ARTIFACTS)
CANONICAL_ASSET_ROWS = tuple(range(len(CANONICAL_ARTIFACTS)))
CANONICAL_Q_HOME_ROWS = tuple(artifact.routing.q_home for artifact in CANONICAL_ARTIFACTS)
CANONICAL_CONTACT_LAYOUT = build_contact_sensor_layout_from_hand_spawn(
    CANONICAL_HAND_SPAWN_CFG,
    validate_all_assets=True,
)


@configclass
class CanonicalUnifiedSceneCfg(GmInHandSceneCfg):
    r"""使用 canonical 16-DOF articulation 的 GM scene。"""

    robot = build_gm_hand_articulation_cfg(CANONICAL_HAND_SPAWN_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self) -> None:
        r"""先完成 GM scene base，再把 contact sensor links 切换为 canonical child names。"""

        InteractiveSceneCfg.__post_init__(self)  # type: ignore[attr-defined]  # 跳过旧 same-topology layout
        install_contact_sensors(self, CANONICAL_CONTACT_LAYOUT)


@configclass
class CanonicalUnifiedObservationsCfg:
    r"""canonical flat actor/critic observation，policy 与 symmetric critic 共用。"""

    @configclass
    class PolicyCfg(ObsGroup):
        r"""顺序化 `[B,116]` policy observation terms。"""

        joint_pos = ObsTerm(func=gm_mdp.joint_pos_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=gm_mdp.joint_vel_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=gm_mdp.last_processed_action, params={"action_name": "hand_joint_pos"})
        joint_limits = ObsTerm(func=gm_mdp.joint_soft_pos_limits, params={"asset_cfg": SceneEntityCfg("robot")})
        command = ObsTerm(func=gm_mdp.reorient_command, params={"command_name": "goal_pose"})
        object_pos = ObsTerm(
            func=gm_mdp.object_pos,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                "semantic_p_ha": (0.0, 0.0, 0.0),
                "frame": "h",
                "reference": "hand",
            },
        )
        object_orientation = ObsTerm(
            func=gm_mdp.object_orientation,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                "frame": "h",
                "representation": "rot6d",
            },
        )
        fingertip_contact = ObsTerm(
            func=gm_mdp.fingertip_contact_binary,
            params={"sensor_names": CANONICAL_CONTACT_LAYOUT.fingertip_sensor_names, "force_threshold": 0.2},
        )
        asset_row = ObsTerm(func=gm_mdp.canonical_asset_row)
        active_joint_mask = ObsTerm(func=gm_mdp.canonical_active_joint_mask)

        def __post_init__(self) -> None:
            r"""关闭 corruption，并保持 flat concatenation 顺序。"""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = PolicyCfg(history_length=1)


@configclass
class CanonicalUnifiedEventsCfg:
    r"""canonical startup routing/ghost lock 与 in-hand reset。"""

    initialize_runtime = EventTerm(
        func=gm_mdp.initialize_canonical_runtime_state,
        mode="startup",
        params={
            "active_joint_mask": CANONICAL_ACTIVE_MASK_ROWS,
            "asset_rows": CANONICAL_ASSET_ROWS,
            "q_home": CANONICAL_Q_HOME_ROWS,
            "routing_mode": "round_robin",
        },
    )
    lock_ghost_limits = EventTerm(
        func=gm_mdp.lock_canonical_ghost_joint_limits,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )
    reset_robot_joints = EventTerm(
        func=gm_mdp.reset_canonical_robot_joints,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )
    reset_object = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.005, 0.005),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CanonicalUnifiedRewardsCfg(GmRewardsCfg):
    r"""沿用 GM object reward，但把 contact terms 路由到 canonical link names。"""

    good_contact = RewTerm(
        func=gm_mdp.good_fingertip_contact,
        weight=0.5,
        params={
            "sensor_names": CANONICAL_CONTACT_LAYOUT.fingertip_sensor_names,
            "min_contacts": 2,
            "force_threshold": 0.2,
            "lambda_floor": 0.05,
        },
    )
    bad_non_tip_contact = RewTerm(
        func=gm_mdp.bad_non_tip_contact,
        weight=-0.2,
        params={
            "sensor_names": CANONICAL_CONTACT_LAYOUT.finger_non_tip_sensor_names,
            "force_threshold": 0.2,
            "lambda_floor": 0.0,
        },
    )


@configclass
class CanonicalUnifiedInHandEnvCfg(GmInHandEnvCfg):
    r"""formal canonical unified PPO task configuration。"""

    scene: CanonicalUnifiedSceneCfg = CanonicalUnifiedSceneCfg(
        num_envs=len(CANONICAL_MOTHER_PATHS) * CANONICAL_ENVS_PER_MOTHER,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: CanonicalUnifiedObservationsCfg = CanonicalUnifiedObservationsCfg()
    actions: GmActionsCfg = GmActionsCfg()
    commands: GmCommandsCfg = GmCommandsCfg()
    rewards: CanonicalUnifiedRewardsCfg = CanonicalUnifiedRewardsCfg()
    terminations: GmTerminationsCfg = GmTerminationsCfg()
    events: CanonicalUnifiedEventsCfg = CanonicalUnifiedEventsCfg()
    curriculum: GmCurriculumCfg = GmCurriculumCfg()


@configclass
class CanonicalUnifiedInHandEnvCfg_PLAY(CanonicalUnifiedInHandEnvCfg):
    r"""canonical task 的小规模 visual play variant。"""


__all__ = [
    "CANONICAL_ACTIVE_MASK_ROWS",
    "CANONICAL_ARTIFACTS",
    "CANONICAL_CONTACT_LAYOUT",
    "CANONICAL_HAND_SPAWN_CFG",
    "CANONICAL_MOTHER_PATHS",
    "CANONICAL_Q_HOME_ROWS",
    "CanonicalUnifiedInHandEnvCfg",
    "CanonicalUnifiedInHandEnvCfg_PLAY",
    "CanonicalUnifiedObservationsCfg",
    "CanonicalUnifiedRewardsCfg",
    "CanonicalUnifiedSceneCfg",
]
