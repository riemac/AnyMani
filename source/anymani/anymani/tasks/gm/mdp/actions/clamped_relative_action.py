r"""TODO: 相对关节位置动作——带 soft joint limits clamp 的 raw rad delta action。

该模块是 `gm` 任务线「方案 C」的实现。它与 IsaacLab 原生
`RelativeJointPositionAction` 的**唯一差异**是：在 `apply_actions` 中将
增量目标显式 clamp 到各关节的 soft limits，弥补原生实现缺失的关节限位保护。

== 设计上下文 ==

问题来源:
    IsaacLab `RelativeJointPositionAction.apply_actions`（`joint_actions.py:228-232`）
    直接下发 $q_t + \Delta_t$ 作为 target，不 clamp 到关节限位。它依赖 PhysX 底层
    actuator 去兜底 clip，行为不够干净。用户在代码中标注了该疑问（
    `joint_actions.py:231 # 这里没有关节限位吗？`），本模块是对此的修复。

为何不做 EMA（已与用户对齐）:
    - EMA 的设计语境是**绝对位置**动作：policy 输出一个大绝对值，
      EMA 充当惯性平滑 $\bar q^{\text{target}}_t = \alpha q^{\text{target}}_t + (1-\alpha) \bar q^{\text{target}}_{t-1}$，
      让指令「悠着点走」。
    - 本模块是**相对增量**动作：$\Delta_t$ 已被 `scale` 压到约 $0.1$ rad/步，
      一阶差分限幅自带平滑。再叠 EMA 是二阶低通滤波，对接触密集任务增加无益延迟。
    - EMA 留作未来消融对照。
    - ref: IsaacLab 官方 inhand 使用 EMA（`inhand_env_cfg.py:96`），但它用绝对位置
      语义，不可直接搬来—相对增量下 EMA 纯多余。

与 obs 侧的一致性:
    - 动作空间：raw rad delta $\Delta_t \in \mathbb{R}^d$，(rad)。
    - state obs：$q_t$（raw rad），同量纲。
    - `last_action`：应为上一步 `processed_actions`（即上一帧实际下发的 rad delta），
      而非 IsaacLab `last_action` 返回的 `raw_actions`（policy 原始输出，
      被 scale 变换前的值）。见 `observations.py` state obs 段的对应 TODO。

== 动作空间建模 ==

策略输出 $a_t^{\text{raw}} \in \mathbb{R}^d$（policy NN 原始输出，通常 $\tanh$ 后
$\in[-1,1]$ 或高斯采样无界），经过 JointAction 基类的 affine 变换：

$$
a_t^{\text{proc}} = \text{clip}\big(a_t^{\text{raw}} \cdot s + b,\ [c_{\text{lo}}, c_{\text{hi}}]\big) \qquad (\text{rad})
$$

其中 $s$（scale）、$b$（offset）、$[c_{\text{lo}}, c_{\text{hi}}]$（per-step clip）
由 cfg 配置。默认 $s=0.1,\ b=0,\ c=\text{None}$。

然后下发 PD 目标：

$$
q_{t+1}^{\text{cmd}} = \text{clamp}\big(q_t + a_t^{\text{proc}},\ q^{\min},\ q^{\max}\big) \qquad (\text{rad})
$$

其中 $q_t$ 为当前实际关节角，$q^{\min}, q^{\max}$ 为 soft limits。
`clamp(·)` 是本模块对父类的唯一覆盖——父类原生缺失此行。

PD 控制器执行：$\tau = K_p (q_{t+1}^{\text{cmd}} - q) + K_d (\dot q^{\text{cmd}} - \dot q)$。

== 配置预设（preset / 超参） ==

每步最大增量预设为 $\pm 0.1$ rad，通过两条机制协作实现：
    - `scale = 0.1`：将 NN 输出 $\in[-1,1]$ 映射到 $\Delta_t \in[-0.1, 0.1]$。
    - `clip = {".*": (-0.1, 0.1)}`（opend）：可选安全网，防止高斯尾部采样出 $|\Delta_t| \gg 0.1$。
      若策略行为已够稳定，可省略对此条。

$0.1$ rad 的取法依据：
    IsaacLab inhand 绝对位置 EMA 在 rescale 后每步有效增量约 $\approx 0.026$ rad。
    本项目采用相对增量，无 EMA 拖慢，取 $0.1$ rad 作为起点（约 4 倍），
    后续根据训练收敛速度和动作抖动程度调参。

`preserve_order = True`（与 IsaacLab 默认 `False` 不同）：
    generated assets 的关节顺序来自 same-topology contract，
    joint-centric token 设计要求 token 索引与关节一一对应。
    不允许 `find_joints` 重排顺序。

TOAGENT:
    注释不可删，可重述、修改、补充和润色。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class ClampedRelativeJointActionCfg(RelativeJointPositionActionCfg):
    r"""相对关节位置动作的配置类——默认启用 joint limits clamp。

    继承自 IsaacLab `RelativeJointPositionActionCfg`，只覆写：
        - `class_type`：指向本模块的 `ClampedRelativeJointPositionAction`。
        - `scale`：默认 $0.1$（每步最大 raw rad 增量）。
        - `preserve_order`：默认 `True`（配合 generated asset contract 的关节顺序）。
        - `clip`：可选项，用于限幅 per-step raw delta 极端值，
          建议 `{".*": (-0.1, 0.1)}`；默认 `None`（暂不额外 clip，靠 scale 限幅）。
        - `use_zero_offset`：`True`（相对动作下 offset 必须为 0，继承自父类）。

    其余字段（`joint_names`, `offset` 语义等）保持父类不变。
    """

    class_type: type[ActionTerm] = None  # type: ignore[assignment]
    """启动时将在此处注入 `ClampedRelativeJointPositionAction` 类引用。
    写在 `__init_subclass__` 或构造后赋值；配置解析系统按 `class_type` 实例化。"""

    scale: float = 0.1
    r"""缩放系数 $s$，将 policy raw output 映射到 raw rad delta。

    预设 $s = 0.1$：NN 输出 $\in[-1,1]$ 时每步增量 $\le \pm 0.1$ rad。
    """

    preserve_order: bool = True
    """保持 articulation asset 中的关节顺序 (True)。

    原因：generated assets 的关节顺序来自 same-topology contract，
    joint-centric token 设计要求 token index 与关节一一对应。
    `False`（IsaacLab 默认）会对 `find_joints` 返回排序后的列表，破坏 contract。

    NOTE(已通过 codebase 校验): 关顺序在所有 same-topology variant 间绝对稳定。
    ref: `hand_builders.py:39` `NON_THUMB_FINGER_NAMES = ("index", "middle", "ring", "little")`
    （拇指永远 append 在最后）；`connectivity_lowering.py:219-241`
    （pre-made 删关节后幸存关节保持原始链序，不重排）；
    post-mutate 只改尺度/指尖几何，不改 schema。
    因此 `preserve_order=True` 下，同一拓扑所有 variant 的
    joint index → 语义关节 映射完全一致，specialist RL 阶段无需额外校验。
    """

    clip: dict[str, tuple[float, float]] | None = None
    r"""每步 delta 的硬限幅（opend 安全网）。

    若配置如 `{".*": (-0.1, 0.1)}`，则无论 NN 输出多大，
    $|a_t^{\text{proc}}| \le 0.1$ rad。默认 `None`（依赖 scale 本身限幅）。
    推荐：teacher RL 阶段先不设，待观察到动作抖动后按需开启。
    """


class ClampedRelativeJointPositionAction(RelativeJointPositionAction):
    r"""带 joint limits clamp 的相对关节位置动作。

    与父类 `RelativeJointPositionAction` 的唯一区别：
    在 `apply_actions` 中将 $q_t + \Delta_t$ clamp 到
    `asset.data.soft_joint_pos_limits` 后，才下发 PD 目标指令。

    FIX: 父类缺失此 clamp（见 `joint_actions.py:231` 用户标注的疑问）。
    该修复确保动作空间与 obs 侧的 margin-to-limit 信号语义自洽：
    obs 说「距限位还有 margin」，action 就不会下发越界 target。

    动作流水线（与父类差异标 ☆）：

    1. 父类 `process_actions`：$a_t^{\text{proc}} = \text{clip}(a_t^{\text{raw}} \cdot s + b)$
    2. **本类 `apply_actions`**（☆覆写）：
       $q_{t+1}^{\text{cmd}} = \text{clamp}(q_t + a_t^{\text{proc}},\ q^{\min},\ q^{\max})$
    3. 父类 `reset`：$a^{\text{raw}} \gets 0$
    """

    cfg: ClampedRelativeJointActionCfg
    """类型收窄后的配置，确保 IDE 能识别本类的额外字段。"""

    def __init__(self, cfg: ClampedRelativeJointActionCfg, env: ManagerBasedEnv):
        r"""初始化相对关节位置动作，继承父类的 scale/clip/offset 解析逻辑。

        Args:
            cfg: 动作配置（含 scale, clip, preserve_order, joint_names 等）。
            env: Isaac Lab 环境实例。

        NOTE: 父类 `RelativeJointPositionAction.__init__` 在 `cfg.use_zero_offset=True`
        时设置 `_offset = 0.0`，这正是相对增量动作需要的语义（无偏置）。
        不需在此覆写任何初始化逻辑。
        """
        # 直接委托父类：解析 joint_names / scale / offset / clip，设置 _offset=0
        super().__init__(cfg, env)

    def apply_actions(self):
        r"""下发 PD 目标，并对关节限位做 clamp（☆本类的唯一覆写点）。

        父类实现（`joint_actions.py:228-232`）：
        ```python
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
        ```
        缺失限位 clamp，依赖 PhysX 底层兜底。本覆写补上 `torch.clamp`。

        公式：
        $$
        q^{\text{cmd}} = \text{clamp}\big(q + a^{\text{proc}},\ q^{\min},\ q^{\max}\big)
        $$

        其中 $q^{\min}, q^{\max}$ 来自 `asset.data.soft_joint_pos_limits`，
        形状 $[B, d, 2]$，d 为 joint_ids 数量。用 soft limits 而非 hard：
        soft 才是 actuator 实际 clamp 的边界（行为相关），hard limits 不是。
        """
        # 当前关节角 $q_t$ (rad)，形状 $[B, d]$
        q_current = self._asset.data.joint_pos[:, self._joint_ids]  # $q_t$

        # 增量目标（未 clamp limit）：$q_t + a_t^{\text{proc}}$
        target_uncapped = q_current + self.processed_actions  # $q_t + \Delta_t$

        # ☆ 核心修复：clamp 到 soft joint limits
        # $q^{\min}, q^{\max}$ 形状 $[B, d]$，通过索引 `[:,:,0]` / `[:,:,1]` 取出
        q_min = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]  # $q^{\min}_i$，(rad)
        q_max = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]  # $q^{\max}_i$，(rad)

        target = torch.clamp(target_uncapped, min=q_min, max=q_max)  # $q^{\text{cmd}}_{t+1}$

        # 下发 PD 目标
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


# ---------------------------------------------------------------------------
# 注入 class_type 引用（绕过 dataclass 不允许 forward reference 直接赋值）
# ---------------------------------------------------------------------------
# 因为 `ClampedRelativeJointPositionAction` 在 `ClampedRelativeJointActionCfg`
# 之后才定义，无法在 cfg 的 class body 中直接引用。此处手动注入。
ClampedRelativeJointActionCfg.class_type = ClampedRelativeJointPositionAction

__all__ = [
    "ClampedRelativeJointActionCfg",
    "ClampedRelativeJointPositionAction",
]
