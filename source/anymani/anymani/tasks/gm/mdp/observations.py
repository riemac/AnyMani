r"""TODO:Observation terms for `tasks.gm`.

IsaacLab RL 既然是服务于层次通才专家训练阶段，用于训练 specialist policy / teacher 的，而该 policy 本身不用于 sim2real。
那么可以尽可能使用更有用的观测信息，包括各种特权信息，来帮助训练。

这里的观察项适合运行时状态、命令、接触等，对于更复杂的几何信息（如 mesh feature）和 token 表征（如 joint-centric unified representation），
不应直接塞进这个 cfg，而应由 `distill/` 与训练 wrapper 明确接管，但可以由它临时携带已经整理好的 geometry tensor，
这样好处是 rl_games 训练时所有输入都在 obs dict 里，最稳。

TOAGENT:
    注释不可删，但可重述、精炼、修改、补充和润色。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__: list[str] = ["reorient_command"]

# ==================
# state obs
# ==================

r"""TODO(state obs): 关节本体感受 (proprioception)，逐步变化的动态量，属于 obs mdp。

符号约定：$q_i$ 为第 $i$ 个关节角 (rad)，$\dot q_i$ 为关节角速度 (rad/s)，
$q_i^{\min}, q_i^{\max}$ 为该关节的 soft 限位 (rad)；下标 $i$ 遍历 surviving revolute joints。
坐标系语义见 `gm/AGENTS.md` 的 `{a} -> {h}` 约定。

NOTE(设计决策，已与用户对齐): 关节位置统一采用 **raw rad 表征** $q_i$，
而非 IsaacLab 默认的 `joint_pos_limit_normalized`（即 $q_i^{\text{norm}}$）。
该决策依据 `Research/总体/层次通才策略训练.md` 的 state obs 小组划分。

其中归一化变换定义为（IsaacLab `scale_transform` 语义，本项目刻意不采用）：
$$
q_i^{\text{norm}} = \frac{2\,(q_i - q_i^{\min})}{q_i^{\max} - q_i^{\min}} - 1 \in [-1, 1]
$$

为何用 $q_i$ 而非 $q_i^{\text{norm}}$（四条论证）:
    1. 跨 variant 语义不变性：资产由同一建模约定生成，且 home position 近似共面，
       故同一关节的 raw rad 在不同 post-mutate variant、乃至真实 leap/allegro
       URDF（对齐到 $\{h\}$ 后）语义一致：$q_i = 0.3$ 恒表示同一物理弯曲构型，
       与该关节的 limit 无关。
    2. post-mutate 只变 joint limit $[q_i^{\min}, q_i^{\max}]$，不变零位/轴向语义。
       $q_i$ 因此是跨 variant / sim2sim 的不变量；$q_i^{\text{norm}}$ 会把不同 raw pose
       压成同值（limit 不同但归一化后相同），抹掉接触任务关心的真实指尖构型。
    3. 恢复代价非对称：$q_i$ 与 limits 组合时，物理构型 $q_i$ 本身直接可得，
       距限位余量 $\text{margin}_i = q_i - q_i^{\min}$ 是线性可得量；反之若用
       $q_i^{\text{norm}}$ 还原 $q_i = q_i^{\min} + \tfrac{q_i^{\text{norm}}+1}{2}(q_i^{\max}-q_i^{\min})$
       需要输入维度间的**乘法** $q_i^{\text{norm}}\cdot(q_i^{\max}-q_i^{\min})$，
       是 MLP / attention 线性层难以精确拟合的乘性算子。
    4. 数值尺度本就友好：关节角 raw 值落在温和有界区间（约 $[-0.8, 1.5]$ rad），
       scale 已适合 PPO，无需为数值稳定而归一化。

本段应实现的项（仅 state obs，不含静态形态量）：
    - q_raw   : asset.data.joint_pos[:, joint_ids]                  $q_i$，(rad)
    - dq_raw  : asset.data.joint_vel[:, joint_ids]                  $\dot q_i$，(rad/s)
    - last_action : 上一步实际下发的 raw rad delta $\Delta_{t-1}$，(rad)。
      必须与动作空间 `ClampedRelativeJointPositionAction` 同量纲，详见下方。

NOTE(last_action 与动作空间的耦合): 动作空间已确定为 raw rad delta（方案 C，
`ClampedRelativeJointPositionAction`），故 last_action 也应在 raw rad 空间。

但 IsaacLab 内置 `isaac_mdp.last_action` 返回 `action_manager.action`
即 `raw_actions`（policy NN 输出的原始值，在 scale/clip 变换**前**），
不是实际下发的 rad delta `processed_actions`。若直接用，last_action 与
state obs 的 $q_i$ 不在同物理空间——策略需要在内部把「NN 输出值」映射回
「rad 增量」，徒增不必要的线性层负担。

TODO: 在 `observations.py` 中实现一个轻量 wrapper：
```python
def last_processed_action(env, action_name="hand_joint_pos") -> torch.Tensor:
    return env.action_manager.get_term(action_name).processed_actions
```
返回的是 `processed_actions`（$\Delta_{t-1}$，rad），与 $q_i$ 同量纲。

边界:
    - $q_i^{\min} / q_i^{\max}$ 是时间常量，属于形态 / geometry 量，不放这里
      （见 geometry obs 段），以免 history_length $H > 1$ 时被时间历史窗口
      重复堆叠 $H$ 次造成冗余。
    - 动作空间与 state obs 均在 raw rad 空间，量纲统一，不存在乘性还原问题。

ref: `Research/总体/层次通才策略训练.md` (obs 分组); `gm/AGENTS.md` (坐标系语义)。

TOAGENT:
    注释不可删，可重述、修改、补充和润色。
"""

# ==================
# contact obs
# ==================

r"""TODO(contact obs): 指尖触觉观测——embodiment-centric，对标 AnyRotate teacher 设计。

== 设计定位 ==

本项目主线是**手型泛化**（embodiment generalization），不是 gravity-invariant
multi-axis rotation。contact obs 应服务 embodiment-centric 表征，而非 world-centric。
具体来说：接触信息应表达在指尖/传感器的局部坐标系下，使接触语义绑定到 embodiment
自身，避免世界系手部姿态变化污染 contact 表征。

若未来引入任意手姿态的 multi-axis rotation（AnyRotate 路线），届时追加
gravity direction in hand frame 作为额外输入即可，不在当前阶段引入该复杂度。

ref: AnyRotate Sec. 3.1 "Simulated Touch"（接触位置 + 净接触力，teacher 用）；
本次设计讨论中与用户对齐的决策记录。

== 传感器配置前置条件 ==

需要 per-fingertip `ContactSensorCfg`（在 hand asset 或 scene 层面挂载），配置：
    - `track_pose = True`（拿传感器姿态 quat_w，用于力矢量和接触点的坐标系转换）
    - `track_contact_points = True`（拿 contact_pos_w）
    - `filter_prim_paths_expr = ["{ENV_REGEX_NS}/object"]`
      （只报与物体的接触，过滤手指间自碰）

NOTE: 传感器 prim_path 必须指向 generated asset 的指尖 mesh primitive。
由于资产是指尖形状 post-mutated 的，prim_path 应为 per-finger 动态绑定，
不能硬编码。该问题是 asset 侧工程问题，此处仅标记依赖。

== 输出规格 ==

每个指尖 $k \in \{1,\dots,K\}$ 输出 7 维，全部在传感器局部坐标系 $\{S_k\}$ 下：

$$
\mathbf{c}^{\text{contact}}_k =
\big[\,c_x,\ c_y,\ c_z,\ F_x,\ F_y,\ F_z,\ \|F\|\,\big]^{\{S_k\}}
\in \mathbb{R}^7
$$

- $c_x, c_y, c_z$：接触点局部坐标（取力最大的接触点；无接触时填 $(0,0,0)$）
- $F_x, F_y, F_z$：净接触力矢量（局部系，从 net_forces_w 经 quat_w 旋转）
- $\|F\|$：力幅值（frame-invariant，冗余但便宜）

Why 局部系而非世界系:
    - 接触点：AnyRotate 直接用局部坐标 $(c_x,c_y,c_z)$，对齐。
    - 力矢量：世界系力随手的整体朝向变化，即使真实接触物理不变，
      引入不必要的表征偏移。转局部系后力矢量绑定到指尖自身朝向，
      消除这一成分（旋转变换的偏移）。重力的动力学偏移（掌朝上/朝下
      时接触力真实物理不同）转局部系也消除不了，留到未来若需要时
      追加 gravity direction obs 处理。
    - 力幅值 $\|F\|$：frame-invariant，放在哪个系都一样。

接触点聚合策略:
    `contact_pos_w` 可能每个指尖有 $M \ge 1$ 个接触点（形状 N×B×M×3）。
    取 $\|F\|$ 最大的那个接触点——该点通常是主接触，对标 AnyRotate
    隐含的单接触点假设。无接触时（全为 NaN）填 $(0,0,0)$。

数据来源映射:
    - c_local  ← contact_pos_w → 选出最大力接触点 → quat_w.inverse*(contact_pos_w - pos_w)
    - F_local  ← net_forces_w → quat_w.inverse * net_forces_w
    - ||F||    ← net_forces_w → ||net_forces_w||₂

NOTE(teacher vs 未来 student): 当前 specialist teacher RL 阶段不承担 sim2real，
暂不引入 AnyRotate 的 EMA 模拟弹性延迟、saturation/rescale 仿真传感器、
binary contact 等 student 侧设计。这些可在 student distill 阶段回流。

TOAGENT:
    注释不可删，可重述、修改、补充和润色。
"""



# ==================
# command obs
# ==================


def reorient_command(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    r"""读取 `ReorientCommand` 生成的 policy-facing command。

    `ReorientCommand` 的 `command` property 已固定为：

    $$
    \mathbf{c}_t =
    \left[\hat\omega^{\{h\}},\ \phi_e^{\{h\}}\right]
    \in \mathbb{R}^{6}
    $$

    其中：
        - $\hat\omega^{\{h\}}$：hand semantic frame `{h}` 下的有向单位旋转轴；
        - $\phi_e^{\{h\}}$：space error
          $\log(R_{goal}R_{current}^{-1})$ 表达到 `{h}` 后的 so(3) 向量。

    DONE(与 command/reward 合同对齐):
        - 默认 `axis_resample_mode="subgoal"`，所以 axis 不再承诺整个 episode
          固定；每次 subgoal 成功后可重新采样 axis + theta。
        - policy 只看到 `{h}` 表达，保持 hand-centric 任务语义；reward / termination
          / curriculum 若需要 `{e}` 轴或 goal quaternion，应从 command term 内部
          buffer 读取 `axis_e`、`error_so3_e`、`goal_quat_w` 等，不从 obs 反推。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的重定向 command 名称。

    Returns:
        torch.Tensor: command tensor，形状 `[num_envs, 6]`。
    """

    return env.command_manager.get_command(command_name)  # `[B,6]`，即 `[axis_h, error_so3_h]`

# ==================
# geometry obs
# ==================

r"""TODO(geometry obs - joint limits): 关节限位作为静态形态量 (morphology feature)。

符号沿用 state obs 段：$q_i^{\min}, q_i^{\max}$ 为第 $i$ 个关节的 soft 限位 (rad)。

NOTE(路线 B 决策，已与用户对齐): limits 归 geometry / 形态，而非 state obs。
因为 $q_i^{\min}, q_i^{\max}$ 是时间常量，若混进 state obs 并启用 history_length
$H > 1$，会在时间窗口里被无意义地重复堆叠 $H$ 次。geometry 小组本身倾向交由
distill 接管（见 `Research/总体/层次通才策略训练.md:29` 的浅契约约定），故 limits
在这里以「不进时间历史的静态特征」形式提供，最终挂到 joint-centric token 上。

本项应提供（cheap，无历史）:
    - q_min, q_max : asset.data.soft_joint_pos_limits[:, joint_ids, 0/1]
      $q_i^{\min}, q_i^{\max}$，(rad)
    - 可选派生 margin（便宜且对接触任务有用，让「接近限位」trivially 可得）:
      $$
      \text{margin}_i^{\text{lo}} = q_i - q_i^{\min}, \qquad
      \text{margin}_i^{\text{hi}} = q_i^{\max} - q_i
      $$

NOTE(soft vs hard limits): 必须用 **soft** limits 而非 hard。soft 才是 actuator
实际 clamp、策略真正会撞到的边界（行为相关）；hard limits 不是行为相关边界。
IsaacLab 未把 limits 暴露为 observation（仅内部 clamp 用），故必须自写，
不能复用 `isaac_mdp`。

最终每个 joint-centric token 的目标形态:
    $\big[\,q_i,\ \dot q_i,\ q_i^{\min},\ q_i^{\max}\ (+\ \text{margin}_i)\,\big]$
    \ +\ last_action（raw rad delta，与 `ClampedRelativeJointPositionAction.processed_actions` 同量纲）。

NOTE(limits 接口，已决策): joint limits 作为静态 ObsTerm 进 obs mdp
（不进时间历史），使 teacher RL 可直接使用。distill 侧如需更复杂的
形态编码，可从 `asset.data` 直接读，不依赖 obs mdp 的 limits 项。

== 边界：其他静态几何特征 → distill/rl/geo_obs.py ==

以下几何特征同样是静态量（时间不变），但其提取和编码复杂度远高于
joint limits，由 `distill/rl/geo_obs.py` 接管：
    - link lengths / mount poses（相对父连杆的 6D 或标量）
    - tip mesh 描述符（BPS / 低维 shape descriptor / offline embedding）
    - 连杆 mesh 几何特征
    - palm / global scale 等全局形态量

observations.py 仅保留浅契约：若 distill 侧需要 observations.py
代为携带已整理好的 geometry tensor（如 rl_games 训练时统一走 obs dict），
可在本段另加一个不参与时间历史的 ObsTerm 转发。
具体选用什么特征、什么数学表征，由 `distill/rl/geo_obs.py` 决定。

ref: `Research/总体/层次通才策略训练.md:27-29` (geometry obs 小组 + 浅契约);
本次对话中与用户对齐的边界决策。

TOAGENT:
    注释不可删，可重述、修改、补充和润色。
"""

# ==================
# priv obs（物体特权信息，仅 teacher 可用）
# ==================

r"""TODO(privileged obs): 物体物理属性——teacher-only，sim2real 不可得。

== 设计定位 ==

本项目主线是手型泛化，但保留物体泛化接口。物体泛化必须在 teacher
训练阶段引入多元物体资产并通过 privileged info 显式条件化策略。
本段定义从仿真器提取哪些 raw physical values，交给 `distill/models`
侧的 object token encoder 投影为 `[OBJ]` 全局 token。

物体表征路线（已与用户对齐）:
    当前阶段采用 **扩展 HORA 路线**：
        - teacher 显式喂 raw physical properties（mass, scale, friction, COM等），
          不做 HORA 风格的压缩嵌入（避免嵌入与具体手型策略耦合）。
        - 策略侧由 `distill/models` 将 raw values 投影为 `[OBJ]` token，
          在 self-attention 池中与 joint tokens 交互。Specialist 阶段不引入
          cross-attention（架构文档 Section 2/6 均列为待定）。
        - `[OBJ]` token 的 projection 模块可额外接收手形态特征（joint limits,
          link lengths 等）作为条件，使物体表征对当前 hand embodiment 有感知——
          这是对 HORA 的扩展：HORA 的隐向量 z_t 仅从本体感受序列估计，
          天然与训练手型绑定；加入手形态条件后解除该绑定。
    未来扩展口（TRO-Grasp 路线）:
        - 若几何形状多样性成为瓶颈，`[OBJ]` token 可升级为多个静态 mesh token
          （BPS/几何描述符，离线预计算），通过 cross-attention 与 joint tokens
          交互。接口设计为 `[OBJ]` token 的可替换 adapter 模块，不影响
          self-attention 主链。

ref: `Research/总体/科研背景说明.md`; HORA Sec 3.1; TRO-Grasp GraphDenoiser;
`Research/总体/整体网络架构讨论.md` Section 2.1, 6;
本次对话中与用户对齐的路线决策。

== 输出规格 ==

privileged obs 应从仿真器中提取以下 raw physical values（共约 21 维）:

    - object_mass       : float                          $m$ (kg)
    - object_scale       : (s_x, s_y, s_z)               bounding box 或 scale (m)
    - object_friction    : float                          $\mu$ (0-1, 静态摩擦系数)
    - object_com_offset  : (dx, dy, dz)                  相对物体 body frame 的 COM 偏置 (m)
    - object_pose        : (x, y, z, qw, qx, qy, qz)    物体 $\{o\}$ 在世界系 $\{w\}$ 下的位姿
    - object_velocity    : (v_x, v_y, v_z, ω_x, ω_y, ω_z) 线速度 + 角速度（world frame）

数据来源（IsaacLab RigidObject API）:
    - mass:     `object.root_physx_view.get_masses()` 或 `object.mass`
    - friction: `object.root_physx_view` 的 material properties
    - pose:     `object.data.root_pos_w`, `object.data.root_quat_w`
    - velocity: `object.data.root_lin_vel_w`, `object.data.root_ang_vel_w`
    - scale:    从 object cfg / USD metadata 获取（非运行时量）
    - COM:      从 rigid body properties 获取（`object.root_physx_view`）

NOTE(embodiment-agnostic): 所有值均为 SI 物理量（kg, m, rad, rad/s），
不依赖手构型。同一物体对不同手型（Leap/Allegro/gen variants）的 raw values
完全一致，满足跨手型泛化的表征解耦需求。

== 与 distill 侧的接口 ==

observations.py 仅负责从 env 提取 raw values 并放入 obs dict:
    def object_privileged_properties(env, object_name="object") -> torch.Tensor:
        # 返回 (num_envs, 21) 或拆分为多个 ObsTerm 的 raw object properties.
    ```

distill/models 侧负责:
    - `ObjectTokenEncoder`: raw physical values → 投影为 `[OBJ]` token 嵌入
    - 可选的形态条件注入（从 hand asset metadata 读取 joint limits / link lengths）
    - mesh token adapter（未来扩展口，预计算静态物体 mesh descriptors）

具体投影维度、是否拆分为多个 token、是否加 position embedding，
由 `distill/models` 的 Specialist Policy Transformer 设计决定，
observations.py 不对此做假设。

TOAGENT:
    注释不可删，可重述、修改、补充和润色。
"""
