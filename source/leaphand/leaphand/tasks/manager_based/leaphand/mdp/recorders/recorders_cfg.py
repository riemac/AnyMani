"""LeapHand 录制配置（与 IsaacLab 官方目录风格一致）。

参考 `AnyRotate/source/leaphand/leaphand/ideas/learning.ipynb`.

该模块提供专门用于行为克隆(BC)数据采集的录制器配置，支持：
1. 时序数据记录：关节动作、SE(3)旋量、雅可比矩阵等
2. 固定参数metadata：伴随矩阵、阻尼系数、affine参数等

使用方法：在env_cfg.py中直接配置所有参数
    ```python
        @configclass
        class CommandsCfg: 
            recorder = leap_mdp.LeapHandBCRecorderManagerCfg(
                finger_body_names=["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],
                finger_xform_names=["index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head"],
                dataset_export_dir_path="./outputs/datasets",
                dataset_filename="leaphand_bc_joint_to_se3",
                dataset_export_mode=leap_mdp.DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
            )
    ```

HDF5数据结构：
    demo_0/
      ├── actions: (T, action_dim)           # 关节空间动作
      ├── observations: (T, obs_dim)         # 观测数据
      ├── actions_se3_twist: (T, 24)         # SE(3)旋量标签
      ├── jacobian: (T, 4, 6, 4)             # 雅可比矩阵
      └── attrs:                             # Episode级别metadata
          ├── adjoint_matrices: (4, 6, 6)    # 伴随矩阵(4指)
          ├── damping: float                 # DLS阻尼系数
          ├── dt: float                      # 控制周期
          └── ...
"""

from collections.abc import Sequence

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.datasets import DatasetExportMode
from isaaclab.utils import configclass

from .recorders import FingerTipSe3TwistRecorder, BCMetadataRecorder


@configclass
class LeapHandBCRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """用于 BC 数据采集的录制管理器 - 扁平化配置设计。
    
    该管理器继承IsaacLab的ActionStateRecorderManagerCfg，自动记录：
    - 观测数据 (observations)
    - 关节动作 (actions)
    - 状态信息 (states)
    
    额外添加：
    - SE(3)旋量标签 + 雅可比矩阵（通过内部FingerTipSe3TwistRecorder）
    - 固定参数metadata（通过内部BCMetadataRecorder）
    
    所有参数都可以在env_cfg.py中直接配置，无需创建嵌套的配置对象。
    
    Args:
        finger_body_names: 实际指尖刚体名称列表，用于读取刚体速度
        finger_xform_names: 虚拟指尖Xform名称列表（与se3Action的target对齐），用于伴随变换
        fingers: 手指关节名称映射字典，键为手指名称，值为对应的关节名称列表
        preserve_order: 是否保持关节名称列表的顺序，默认为True
        dataset_export_mode: 数据导出模式（继承自ActionStateRecorderManagerCfg）
        dataset_export_dir_path: 数据集导出目录（继承自ActionStateRecorderManagerCfg）
        dataset_filename: 数据集文件名，不含扩展名（继承自ActionStateRecorderManagerCfg）
    
    Example:
        在 inhand_base_env_cfg.py 中使用：
        
        ```python
        @configclass
            recorders = leap_mdp.LeapHandBCRecorderManagerCfg(
                # 指尖配置
                finger_body_names=["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],
                finger_xform_names=["index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head"],
                # 导出配置
                dataset_export_dir_path="./outputs/datasets",
                dataset_filename="leaphand_bc_joint_to_se3",
                dataset_export_mode=leap_mdp.DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
            )
        ```
    """
    
    # ============ 指尖配置参数（扁平化暴露） ============
    finger_body_names: Sequence[str] = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")
    """实际指尖刚体名称列表，默认沿用URDF中的名称。"""

    finger_xform_names: Sequence[str] | None = ("index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head")
    """可选：虚拟指尖Xform名称列表（与动作项的 target 对齐）。
    提供后会将旋量从刚体帧 {b} 变换到虚拟帧 {b'}。
    """

    fingers: dict[str, list[str]] = {
        "index": ["a_1", "a_0", "a_2", "a_3"],
        "middle": ["a_5", "a_4", "a_6", "a_7"],
        "ring": ["a_9", "a_8", "a_10", "a_11"],
        "thumb": ["a_12", "a_13", "a_14", "a_15"]
    }
    """手指关节名称映射字典，键为手指名称，值为对应的关节名称列表。"""
              
    preserve_order: bool = True
    """是否保持关节名称列表的顺序，默认为True。用于确保关节索引与配置中的顺序一致。"""
    
    # ============ 数据导出配置（继承自父类，提供默认值） ============
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    """数据导出模式，默认仅导出成功的episode。"""
    
    dataset_export_dir_path: str = "./outputs/datasets"
    """数据集导出目录。"""
    
    dataset_filename: str = "leaphand_bc_joint_to_se3"
    """数据集文件名（不含扩展名）。"""
    
    # ============ 内部Recorder配置（自动构建，用户无需关心） ============
    def __post_init__(self):
        """后初始化：根据用户参数自动构建内部recorder terms。"""
        super().__post_init__()
        
        # 动态创建内部 RecorderTermCfg 实例
        # 这样用户无需在 env_cfg.py 中手动配置嵌套的 FingerTipSe3TwistRecorderCfg
        @configclass
        class _InternalFingerTipRecorderCfg(RecorderTermCfg):
            class_type: type[RecorderTerm] = FingerTipSe3TwistRecorder
            finger_body_names: Sequence[str] = self.finger_body_names
            finger_xform_names: Sequence[str] | None = self.finger_xform_names
            finger_parent_names: Sequence[str] | None = None
            fingers: dict[str, list[str]] = self.fingers
            preserve_order: bool = self.preserve_order
        
        @configclass
        class _InternalBCMetadataRecorderCfg(RecorderTermCfg):
            class_type: type[RecorderTerm] = BCMetadataRecorder
        
        # 将内部recorder terms添加到管理器
        self.record_post_step_fingertip_se3_twist = _InternalFingerTipRecorderCfg()
        self.record_post_reset_bc_metadata = _InternalBCMetadataRecorderCfg()
