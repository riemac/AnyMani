# TODO:该文件为se(3)动作项的配置类文件，用于定义和管理se(3)动作项的配置参数
# 有一个需注意的地方，指尖末端设置的坐标系{b'}是人为设置的Xform，它是虚拟的而非真实的刚体，因此无法获取关于该坐标系的雅可比矩阵J_b'
# 如leaphand的leap_hand_right/fingertip_2/middle_tip_head，middle_tip_head为虚拟设置的Xform，fingertip_2才是真实的刚体，其所在坐标系为{b}
# 好在这个T_bb'是固定不变的，可以预先计算好并存储下来，在计算J_b'^+时利用伴随变换转换为J_b^+即可

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

@configclass
class se3wdlsActionsCfg(se3dlsActionsCfg):
    r"""se(3) 动作项 WDLS（Weighted Damped Least Squares）配置类。

    该类用于定义和管理 se(3) 动作项 WDLS 的配置参数。
    """

    class_type: type[ActionTerm] = se3.se3wdlsAction

    W_q: list | tuple | None = None
    r"""TODO：关节空间权重矩阵。
    
    如果为 None，则使用单位矩阵，维度可根据实际关节数自动推断。
    若提供具体矩阵，则应为方阵，维度与关节数相匹配，否则将引发报错。
    
    """

    W_x: list | tuple | None = None
    r"""TODO：任务空间权重矩阵。
    
    如果为 None，则使用单位矩阵，维度为6乘6。
    若提供具体矩阵，则应为方阵，维度为6乘6，否则将引发报错。

    """

@configclass
class se3adlsActionsCfg(se3dlsActionsCfg):
    r"""se(3) 动作项 ADLS（Adaptive Damped Least Squares）配置类。

    该类用于定义和管理 se(3) 动作项 ADLS 的配置参数。
    """

@configclass
class se3awdlsActionsCfg(se3wdlsActionsCfg):
    r"""se(3) 动作项 AWDLS（Adaptive Weighted Damped Least Squares）配置类。

    该类用于定义和管理 se(3) 动作项 AWDLS 的配置参数。
    """