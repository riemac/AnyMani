# 该文件为 se(3) 动作项提供配置类。
# 指尖末端的虚拟坐标系 {b'} 需要通过伴随变换将父刚体 {b} 的雅可比矩阵转换得到。

from collections.abc import Sequence

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

from . import se3_actions as se3


@configclass
class se3ActionCfg(ActionTermCfg):
    r"""se(3) 动作项配置类。

    该类用于定义和管理 se(3) 动作项的配置参数。
    """

    class_type: type[ActionTerm] = se3.se3Action

    joint_names: str | Sequence[str] = ".*"
    r"""控制的关节名称列表或正则表达式。

    默认为 ".*"，表示控制所有关节。可以指定具体的关节名称列表，
    如 ["finger_1_joint_1", "finger_1_joint_2"] 来控制特定关节。
    """

    preserve_order: bool = True
    r"""是否保持关节索引的顺序。

    如果为 True，关节索引将按照 joint_names 中指定的顺序排列。
    如果为 False，关节索引将按照它们在机器人模型中的默认顺序排列。
    """

    is_xform: bool = False
    r"""指示末端是否为虚拟的Xform。"""  
    # 如果是False，则直接使用真实刚体的雅可比矩阵J_b
    # True的话，需要在动作项类初始化时，获取该Xform相对于真实刚体的位姿变换T_bb'，储存起来，为计算伴随变换提供支持。

    use_xform_jacobian: bool = False
    r"""虚拟 Xform 模式下的技术路线选择（仅当 is_xform=True 时有效）。
    
    当 is_xform=True 时，存在两种等价的技术路线：
    
    - **False (默认)**：变换旋量策略
      计算流程：V_b' (输入) → V_b = Ad_{T_bb'} @ V_b' → dθ = J_b^+ @ V_b
      优点：数值稳定，符合 AnyRotate-temp 的实现
    
    - **True**：变换雅可比策略  
      计算流程：V_b' (输入) → J_b' = Ad_{T_b'b} @ J_b → dθ = J_b'^+ @ V_b'
      优点：雅可比在虚拟帧表示，便于加权操作度计算
    
    数学上两者等价（dθ 相同），但数值特性和计算效率略有差异。
    """

    target: str = MISSING
    r"""末端名称。

    is_xform为True时，target指向虚拟Xform的名称
    
    is_xform为False时，target指向真实刚体的名称
    """

    parent: str = None
    r"""末端的父节点名称，为实际的末端刚体。

    如果is_xform为False，则parent应为None

    如果is_xform为True，可通过指定该prim名称，显示指定parent和target之间的父子关系，T_bb'即为parent到target的相对位姿变换。
    也可不指定该项，则在动作项类初始化时，通过解析环境中的target的上一级prim名称，作为parent。

    这种情况可应付target和parent隔了2个及以上prim层级的情况。
    """

    use_pd: bool = False
    r"""是否启用 PD 控制（位置+速度前馈）。

    如果为 True，将同时设置关节的位置目标和速度目标（前馈速度）。
    此时底层控制器执行：tau = Kp * (pos_target - pos) + Kd * (vel_target - vel)。

    如果为 False，仅设置关节的位置目标，速度目标强制设为 0。
    此时底层控制器执行：tau = Kp * (pos_target - pos) - Kd * vel。
    """

    angular_limits: None | float | tuple[float, float, float] = None
    r"""角速度限制。如果为None，则不设置限制；如果为单个k，则代表对所有分量都有不超过|pi/k|；如果提供三个分量(kx, ky, kz)，代表各自分量不超过|pi/ki|。
    
    这里设置pi，主要是轴角切向量在pi处有奇异的缘故
    """

    linear_limits: None | float | tuple[float, float, float] = None
    r"""线速度限制。如果为None，则不设置限制；如果为单个v，则代表对所有分量都有不超过sqrt(3)|v|；如果提供三个分量(vx, vy, vz)，代表各自分量不超过sqrt(3)|vi|。
    
    这里设置sqrt(3)，主要是v是在{s}上观察，而实际旋量的线速度分量是在{b}上，它们之间的关系由矩阵范数和向量范数的相容性所推导出
    """

    use_joint_limits: bool = True
    r"""是否启用关节限位。

    如果为 True，将在计算出目标关节位置后，将其限制在机器人的软关节限位范围内。
    这可以防止动作指令超出物理可行范围。
    """

    use_body_frame: bool = True
    r"""参考系选择（reference frame）。

    该开关同时决定：
    1) se(3) 动作 :math:`\mathcal{V}=[\omega, v]` 的坐标表达参考系；
    2) 雅可比 :math:`J` 的输出参考系（从而保证 :math:`\dot{\theta} = J^{\dagger} \mathcal{V}` 的帧一致）。

    - True: 参考系为末端刚体坐标系 {b}
        - is_xform=False: 动作为 :math:`\mathcal{V}_b`，参考点为 {b}
        - is_xform=True : 动作为 :math:`\mathcal{V}_{b'}`，参考点为 {b'}
    - False: 参考系为机器人根/基坐标系 {s}
        - is_xform=False: 动作为 :math:`\mathcal{V}_s`，参考点为 {b}
        - is_xform=True : 动作为 :math:`\mathcal{V}_s^{b'}`，参考点为 {b'}

    Notes:
        - 这里的 {s} 取自 articulation 的 root link 姿态（``root_quat_w``），并不等同于世界坐标系 {w}。
        - 在 is_xform=True 且 use_body_frame=False 时，动作在 {s} 下表达，因此 ``use_xform_jacobian``
          仅对 use_body_frame=True 的分支有意义。
    """

    def __post_init__(self):
        if self.target is MISSING:
            raise ValueError("se3ActionCfg.target 必须提供末端名称。")


@configclass
class se3dlsActionsCfg(se3ActionCfg):
    r"""se(3) 动作项 DLS（Damped Least Squares）配置类。

    该类用于定义和管理 se(3) 动作项 DLS 的配置参数。
    """

    class_type: type[ActionTerm] = se3.se3dlsAction

    damping: float = 0.01
    r"""阻尼系数 :math:`\lambda`。

    该参数控制 DLS 方法的数值稳定性。较大的值会使解更稳定但可能降低精度，
    较小的值会提高精度但可能在奇异位形附近不稳定。
    
    典型取值范围为 0.01 到 0.1，具体值需根据雅可比矩阵的量级调整。
    对于灵巧手操作，建议从 0.01 开始调试。
    """

    def __post_init__(self):
        super().__post_init__()
        if self.damping <= 0:
            raise ValueError("se3dlsActionsCfg.damping 必须为正值。")


@configclass
class se3dlsEmaActionsCfg(se3dlsActionsCfg):
    r"""se(3) 动作项 DLS + EMA（Exponential Moving Average）配置类。

    该配置用于对 se(3) 旋量命令（处理后的物理量级 twist）施加 EMA 平滑：

    .. math::

        \mathcal{V}_{\mathrm{ema},t} = \alpha\,\mathcal{V}_t + (1-\alpha)\,\mathcal{V}_{\mathrm{ema},t-1}

    Notes:
        - :math:`\alpha=1` 时等价于不平滑。
        - EMA 在 twist 层面平滑（策略输出的高频抖动），与 :math:`\Delta t` 在关节增量层面的平滑互补。
    """

    class_type: type[ActionTerm] = se3.se3dlsEmaAction

    alpha: float = 1.0
    """EMA 平滑系数 $\alpha \in (0, 1]$。"""

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("se3dlsEmaActionsCfg.alpha 必须在 (0, 1] 范围内。")

@configclass
class se3wdlsActionsCfg(se3dlsActionsCfg):
    r"""se(3) 动作项 WDLS（Weighted Damped Least Squares）配置类。"""

    class_type: type[ActionTerm] = se3.se3wdlsAction

    W_q: list | tuple | None = None
    r"""关节空间权重矩阵。
    
    如果为 None，则使用单位矩阵，维度可根据实际关节数自动推断。
    若提供具体矩阵，则应为方阵，维度与关节数相匹配，否则将引发报错。
    
    """

    W_x: list | tuple | None = None
    r"""任务空间权重矩阵。
    
    如果为 None，则使用单位矩阵，维度为6乘6。
    若提供具体矩阵，则应为方阵，维度为6乘6，否则将引发报错。

    """

    def __post_init__(self):
        super().__post_init__()
        if self.W_x is not None:
            if not isinstance(self.W_x, (list, tuple)):
                raise ValueError("se3wdlsActionsCfg.W_x 需为长度为6的列表或元组。")
            if len(self.W_x) != 6:
                raise ValueError(f"se3wdlsActionsCfg.W_x 长度应为6，实际为 {len(self.W_x)}。")
        if self.W_q is not None:
            if not isinstance(self.W_q, (list, tuple)):
                raise ValueError("se3wdlsActionsCfg.W_q 需为对角权重列表或元组。")
            if len(self.W_q) == 0:
                raise ValueError("se3wdlsActionsCfg.W_q 不能为空列表。")

@configclass
class se3adlsActionsCfg(se3dlsActionsCfg):
    r"""se(3) 动作项 ADLS（Adaptive Damped Least Squares）配置类。
    
    引入选择性阻尼机制，仅对接近奇异的方向施加阻尼。
    """
    class_type: type[ActionTerm] = se3.se3adlsAction

    singular_threshold: float = 0.05
    r"""最小奇异值阈值 :math:`\epsilon`。
    
    当雅可比矩阵的奇异值低于此阈值时，对该方向施加选择性阻尼。
    
    建议值范围：0.01 - 0.1。
    
    Note:
        使用选择性阻尼机制 (Selective Damping)：
        - 若 :math:`\sigma_i \ge \epsilon`: :math:`\lambda_i = 0`
        - 若 :math:`\sigma_i < \epsilon`: :math:`\lambda_i = \lambda_{max}(1-\sigma_i/\epsilon)^2`
    """

    def __post_init__(self):
        super().__post_init__()
        if self.singular_threshold <= 0:
            raise ValueError("se3adlsActionsCfg.singular_threshold 必须为正值。")

@configclass
class se3awdlsActionsCfg(se3wdlsActionsCfg):
    r"""se(3) 动作项 AWDLS（Adaptive Weighted Damped Least Squares）配置类。
    
    结合加权DLS的量纲归一化能力和自适应阻尼的奇异回避能力。
    """
    class_type: type[ActionTerm] = se3.se3awdlsAction

    singular_threshold: float = 0.05
    r"""最小奇异值阈值 :math:`\epsilon`。
    
    当加权雅可比矩阵 :math:`\tilde{J}_b` 的奇异值低于此阈值时，
    对该方向施加选择性阻尼。健康方向保持无阻尼状态。
    
    建议值范围：0.01 - 0.1。
    
    Note:
        使用选择性阻尼机制 (Selective Damping)：
        - 若 :math:`\sigma_i \ge \epsilon`: :math:`\lambda_i = 0`
        - 若 :math:`\sigma_i < \epsilon`: :math:`\lambda_i = \lambda_{max}(1-\sigma_i/\epsilon)^2`
    """
    
    def __post_init__(self):
        super().__post_init__()
        if self.singular_threshold <= 0:
            raise ValueError("se3awdlsActionsCfg.singular_threshold 必须为正值。")

            raise ValueError("se3awdlsActionsCfg.singular_threshold 必须为正值。")