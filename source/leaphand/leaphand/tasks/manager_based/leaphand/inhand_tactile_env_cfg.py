"""LeapHand连续旋转任务 + 触觉传感器环境配置

基于inhand_base_env_cfg，增加触觉传感器功能用于：
1. 触觉观测：指尖接触力作为策略输入
2. 接触奖励：Good Contact (多指接触) 和 Bad Contact (手掌接触惩罚)
3. 接触稳定性：接触力平滑性和均匀性奖励

Usage:
    创建该环境时会自动集成触觉传感器，无需修改基础配置
"""

import math

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 导入基础环境配置
from .inhand_base_env_cfg import (
    InHandSceneCfg,
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
    InHandObjectEnvCfg,
)

from . import mdp as leap_mdp
import isaaclab.envs.mdp as mdp


# =====================
# Tactile hyper-params
# =====================
# Keep tactile thresholds consistent across observations and rewards.
TACTILE_FORCE_THRESHOLD = 0.2

# Reward shaping mode:
# - "binary": keep legacy 0/1 logic
# - "count":  reward/penalty scales with number of contacts
TACTILE_CONTACT_REWARD_TYPE = "binary"

# Adaptive reward curriculum (AnyRotate Appendix B.3)
TACTILE_USE_REWARD_CURRICULUM = True
# NOTE:
# - metric_key="consecutive_success": g_eval is the number of reached goals in the current episode.
# - metric_key="cumulative_rotation": g_eval is the cumulative rotation angle (radians) in the current episode.
TACTILE_CURRICULUM_METRIC_KEY = "consecutive_success"  # or "cumulative_rotation"

# Defaults for different curriculum metrics.
_TACTILE_G_MIN_SUCCESS = 0.0
_TACTILE_G_MAX_SUCCESS = 8.0
_TACTILE_G_MIN_RAD = 0.0
_TACTILE_G_MAX_RAD = math.pi

# Final G range used by reward terms.
if TACTILE_CURRICULUM_METRIC_KEY == "cumulative_rotation":
    TACTILE_G_MIN = _TACTILE_G_MIN_RAD
    TACTILE_G_MAX = _TACTILE_G_MAX_RAD
else:
    TACTILE_G_MIN = _TACTILE_G_MIN_SUCCESS
    TACTILE_G_MAX = _TACTILE_G_MAX_SUCCESS


##
# 场景配置 - 添加触觉传感器
##

@configclass
class InHandTactileSceneCfg(InHandSceneCfg):
    """LeapHand场景 + 触觉传感器配置"""
    
    # 禁用物理复制以支持域随机化（如物体尺寸随机化）
    replicate_physics = False
    # ContactSensor 的 contact buffer 太小导致 contact_data.index_select 越
    # 越界会导致 contact_data.index_select 越界，导致 contact_data 全为0
    # 因此需要增加 contact buffer 的大小
    
    # ===== 指尖触觉传感器 =====
    # 食指指尖
    contact_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,  # 需要切向力以计算总合力
        # NOTE: contact-rich scenes (many envs / many contacts per body) need a larger buffer.
        # Otherwise, PhysX contact buffers can overflow and trigger CUDA index out-of-bounds.
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    # 中指指尖
    contact_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    # 无名指指尖
    contact_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    # 拇指指尖
    contact_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_fingertip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    # ===== 非指尖刚体传感器（用于惩罚非期望接触）=====
    # 手掌基座
    contact_palm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        debug_vis=False,
    )
    
    # ===== 食指关节（非指尖）=====
    contact_index_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_index_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_index_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 中指关节（非指尖）=====
    contact_middle_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_middle_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_middle_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 无名指关节（非指尖）=====
    contact_ring_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_ring_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_ring_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 拇指关节（非指尖）=====
    contact_thumb_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_temp_base",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_thumb_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_pip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    contact_thumb_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_dip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )


##
# 观测配置 - 添加触觉观测
##

@configclass
class TactileObservationsCfg(ObservationsCfg):
    """扩展观测配置，添加触觉信息"""
    
    @configclass
    class PolicyPropTactileCfg(ObservationsCfg.ProprioceptionObsCfg):
        """策略观测（Student Policy）：本体感受 + 二值化触觉信号"""
        
        # ===== 触觉观测：0-1二值接触信号 =====
        # 检测哪些指尖在接触物体（适合sim2real）
        fingertip_contact_binary = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": [
                    "contact_index",   # 食指
                    "contact_middle",  # 中指
                    "contact_ring",    # 无名指
                    "contact_thumb",   # 拇指
                ],
                "output_type": "binary",
                "force_threshold": TACTILE_FORCE_THRESHOLD,
            },
        )

    @configclass
    class PolicyPrivTactileCfg(ObservationsCfg.PrivilegedObsCfg):
        """策略观测（Student Policy）：本体感受 + 特权信息 + 二值化触觉信号"""
        
        # ===== 触觉观测：0-1二值接触信号 =====
        # 检测哪些指尖在接触物体（适合sim2real）
        fingertip_contact_binary = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": [
                    "contact_index",   # 食指
                    "contact_middle",  # 中指
                    "contact_ring",    # 无名指
                    "contact_thumb",   # 拇指
                ],
                "output_type": "binary",
                "force_threshold": TACTILE_FORCE_THRESHOLD,
            },
        )

    @configclass
    class CriticTactileCfg(ObservationsCfg.PrivilegedObsCfg):
        """Critic观测（Teacher Policy）：特权信息 + 精确力信号"""
        
        # ===== 触觉观测：指尖接触合力（法向+切向）=====
        # 每个指尖的总接触力矢量，用于精确力控制
        fingertip_contact_force = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": [
                    "contact_index",
                    "contact_middle",
                    "contact_ring",
                    "contact_thumb",
                ],
                "output_type": "force",  # 返回 (num_envs, 12) 的力矢量
            },
            clip=(-50.0, 50.0),
            scale=0.1,  # 归一化到合理范围
        )

    policy: ObsGroup = PolicyPropTactileCfg(history_length=1)
    critic: ObsGroup = CriticTactileCfg(history_length=1)

##
# 奖励配置 - 添加触觉奖励
##

@configclass
class TactileRewardsCfg(RewardsCfg):
    """扩展奖励配置，添加触觉奖励项"""
    
    # ===== 负载分配奖励 =====
    # 鼓励手指承担垂直载荷占比趋近于1（手掌承担趋近于0）
    load_distribution = RewTerm(
        func=leap_mdp.load_distribution_reward,
        weight=1,  # λ_load
        params={
            "fingertip_sensor_names": [
                "contact_index",
                "contact_middle",
                "contact_ring",
                "contact_thumb",
            ],
            "palm_sensor_names": [
                "contact_palm",
                "contact_index_mcp", "contact_index_pip", "contact_index_dip",
                "contact_middle_mcp", "contact_middle_pip", "contact_middle_dip",
                "contact_ring_mcp", "contact_ring_pip", "contact_ring_dip",
                "contact_thumb_base", "contact_thumb_pip", "contact_thumb_dip",
            ],
            "gravity_axis": 2,  # z轴向上
            "epsilon": 1e-3,
        },
    )
    
    # ===== Good Contact Reward (AnyRotate 公式6) =====
    # r_gc = 1 if n_tip_contact >= 2 else 0
    good_fingertip_contact = RewTerm(
        func=leap_mdp.good_fingertip_contact,
        weight=1.0,  # λ_contact
        params={
            "sensor_names": [
                "contact_index",
                "contact_middle",
                "contact_ring",
                "contact_thumb",
            ],
            "min_contacts": 2,  # 至少2个指尖接触
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )
    
    # ===== Bad Contact Penalty (AnyRotate 公式7) =====
    # r_bc = 1 if n_non_tip_contact > 0 else 0 (函数返回正值，权重为负)
    bad_palm_contact = RewTerm(
        func=leap_mdp.bad_palm_contact,
        weight=-1.0,  # 负权重实现惩罚
        params={
            "sensor_names": [
                "contact_palm",
                "contact_index_mcp", "contact_index_pip", "contact_index_dip",
                "contact_middle_mcp", "contact_middle_pip", "contact_middle_dip",
                "contact_ring_mcp", "contact_ring_pip", "contact_ring_dip",
                "contact_thumb_base", "contact_thumb_pip", "contact_thumb_dip",
            ],
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )


##
# 主环境配置
##

@configclass
class InHandTactileEnvCfg(InHandObjectEnvCfg):
    """LeapHand连续旋转 + 触觉传感器环境配置"""
    
    # 使用触觉增强的场景
    scene: InHandTactileSceneCfg = InHandTactileSceneCfg(num_envs=4096, env_spacing=0.6)
    
    # 使用触觉增强的观测
    observations: TactileObservationsCfg = TactileObservationsCfg()
    
    # 使用触觉增强的奖励
    rewards: TactileRewardsCfg = TactileRewardsCfg()
    
    # 其他配置继承自InHandObjectEnvCfg
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        """后初始化"""
        super().__post_init__()
        # 可在此处调整触觉环境特定的参数
