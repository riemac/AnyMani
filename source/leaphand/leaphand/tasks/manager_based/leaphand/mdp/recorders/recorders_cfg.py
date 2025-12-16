"""LeapHand 录制配置（与 IsaacLab 官方目录风格一致）。

用法：
- 创建环境前，将 env_cfg.recorders 指向这里的 LeapHandBCRecorderManagerCfg。
- 可按需覆盖导出目录/文件名。例如：
    env_cfg.recorders = LeapHandBCRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = "./outputs/datasets/leaphand_bc"
    env_cfg.recorders.dataset_filename = "joint_to_se3_bc"

录制内容：
- 保留基础动作/状态/观测条目（与 ActionStateRecorderManagerCfg 相同）。
- 额外输出四指指尖的刚体系 SE(3) 旋量 actions_se3_twist，展平为 24 维。
"""

from collections.abc import Sequence

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.datasets import DatasetExportMode
from isaaclab.utils import configclass

from .recorders import FingerTipSe3TwistRecorder


@configclass
class FingerTipSe3TwistRecorderCfg(RecorderTermCfg):
    """配置：记录四指指尖的刚体系 SE(3) 旋量。"""

    class_type: type[RecorderTerm] = FingerTipSe3TwistRecorder

    finger_body_names: Sequence[str] | None = None
    """实际指尖刚体名称列表，默认沿用URDF中的 ['fingertip','fingertip_2','fingertip_3','thumb_fingertip']。"""

    finger_xform_names: Sequence[str] | None = None
    """可选：虚拟指尖Xform名称列表（与动作项的 target 对齐）。提供后会将旋量从刚体帧 {b} 变换到虚拟帧 {b'}。"""

    finger_parent_names: Sequence[str] | None = None
    """可选：对应虚拟指尖的父刚体名称列表。默认使用 ``finger_body_names`` 中的刚体作为父节点。"""


@configclass
class LeapHandBCRecorderManagerCfg(ActionStateRecorderManagerCfg):
    # REVIEW：目前主要是语法架构方面有疑惑。它这里继承了ActionStateRecorderManagerCfg，而FingerTipSe3TwistRecorderCfg作为其中一个成员。这导致在 env_cfg.py 里配置不太方便
    # 我更希望所有可配置的参数可以在 env_cfg.py 里直接进行声明式配置，而不是在这个文件里配置好、然后在 env_cfg.py 里直接调用
    # 有一个想法是 FingerTipSe3TwistRecorderCfg 和 LeapHandBCRecorderManagerCfg 干脆转为1个类实现？
    # 但这样的话，ActionStateRecorderManagerCfg 里的 recorder_terms 就不能直接用了
    # 还有一个想法是双重继承，但我不知道python有这个语法没，而且会产生怎样的行为特性
    # 或者还有一个想法是 FingerTipSe3TwistRecorderCfg 继承 ActionStateRecorderManagerCfg 来单独实现会不会更好？
    """用于 BC 数据采集的录制管理器。
    
    录制指尖SE(3)旋量时，使用虚拟指尖坐标系{b'}（与se3Action的target对齐），
    确保录制的旋量标签与SE(3)环境的动作输入坐标系一致。
    """

    # 额外添加指尖 SE3 旋量，配置虚拟指尖坐标系
    record_post_step_fingertip_se3_twist = FingerTipSe3TwistRecorderCfg(
        finger_body_names=["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],
        finger_xform_names=["index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head"],
    )

    # 默认仅导出成功 episode，可按需覆盖
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    dataset_export_dir_path: str = "./outputs/datasets"
    dataset_filename: str = "leaphand_joint_to_se3"
