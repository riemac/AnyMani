r"""TODO: GM 单资产 palm-supported tactile rotation 的共享 MDP 契约。

本文件是下一代 GM single-asset baseline 的设计 scaffold。它承载的是同一套任务语义，
不是 GRU / TCN 网络实现，也不是 N052 的局部修补。后续实现应在本文件组装 scene、action、
command、observation、reward、reset、termination、reward curriculum 与 ADR；网络结构和
rl_games 训练配置归 `anymani.distill`。

当前阶段允许 `tasks.gm` 单向复用 `tasks.inhand` 中已经验证过的 policy-step target action
与 LEAP ADR 低层原件。禁止反向依赖：`tasks.inhand` 不得 import `tasks.gm`。GM 不复用旧
official reward、command 或 env cfg；这些任务语义由 GM 自己重新定义。待新基线验证后，
action 与 ADR 原件再完整迁入 GM，届时出清临时的单向依赖。

== 物理任务边界 ==

任务使用 generated `right_t4_i4_m4_r4` 与 DexCube，手掌朝上并允许掌面承托物体。
目标是形成平缓、连续、有向的绕手掌语义法向旋转，以及主要由 fingertip 承担的
release-recontact finger gait。掌面接触是合法支撑，不得照搬 AnyRotate 的无掌面
precision-grasp 假设而把 palm contact 当作坏接触。

任务几何统一定义在 hand semantic frame `{h}`。第一版固定旋转轴为：

$$
\hat{k}^{\{h\}}=(0,0,1).
$$

运行时由 hand pose 把该轴变换到 `{e}` / `{w}`，供姿态增量、物体角速度和 termination
计算使用。当前单资产 palm-up 配置可使 `{h}` 与 `{e}` 的轴看起来重合，但实现不能依赖
这个偶然关系，否则后续 hand orientation 或 morphology 变化时任务语义会漂移。

== 控制频率与 action contract ==

physics frequency 保持 120 Hz，policy frequency 固定为 20 Hz：

$$
\Delta t_{physics}=\frac{1}{120}\ \mathrm{s},
\qquad
d=6,
\qquad
\Delta t_{policy}=\frac{6}{120}=0.05\ \mathrm{s}.
$$

策略输出是无量纲 joint command：

$$
a_t^{policy}\in[-1,1]^{16}.
$$

ADR 依次施加 action noise、最多 3 个 policy step 的 latency 与裁剪，得到同样无量纲的
执行 command：

$$
a_t^{exec}
=
\operatorname{clip}
\left(
\operatorname{delay}
\left(a_t^{policy}+\epsilon_t\right),
-1,
1
\right).
$$

target buffer 每个 policy step 只更新一次，六个 physics substep 只 hold 同一 target：

$$
u_{t+1}
=
\operatorname{clip}
\left(
u_t+\frac{1}{24}a_t^{exec},
q_{min},
q_{max}
\right).
$$

第一版不使用 action EMA。平缓性由物体速度、机械功率、torque、pose、action 与
action-rate 等 stable reward 塑造。ADR 的 3-step latency 在 20 Hz 下等于 150 ms；
日志必须同时记录 step 与毫秒语义。

== Moving subgoal 与真实旋转进度 ==

每个 episode reset 后记录 object 初始位置作为固定位置 anchor：

$$
p_{anchor}^{\{h\}}=p_{o,0}^{\{h\}}.
$$

首个 goal 与每次 success 后的新 goal 都从当时的 object orientation 生成，而不是从
上一个 goal 继续累乘：

$$
R_{g,k+1}
=
\operatorname{Exp}
\left(
[\hat{k}]_\times\frac{\pi}{6}
\right)
R_{o,t_k}.
$$

位置 goal 在整个 episode 内保持为 reset anchor：

$$
p_{g,k}=p_{anchor}.
$$

command term 是 goal、axis 与旋转进度的唯一状态 owner。它每个 policy step 从相邻 object
orientation 计算未裁剪的有向轴向增量：

$$
\Delta\psi_t
=
\operatorname{Log}
\left(
R_{wo,t}R_{wo,t-1}^{-1}
\right)^{\vee\mathsf{T}}
\hat{k}^{\{w\}}.
$$

episode 净旋转使用未裁剪增量：

$$
\Psi_t
=
\sum_{j=1}^{t}\Delta\psi_j.
$$

reward 可以对单步增量裁剪，metric / curriculum / ADR 不得复用该裁剪值。command 应同时
暴露 signed `net_rotation_rad`、用于课程的 `net_rotation_turns=max(Psi,0)/(2*pi)`、
未裁剪瞬时轴向角速度、目标姿态、位置 anchor 与 success metrics。反向旋转先与正向旋转
相消；最终净值小于零时，课程进度裁为零而不是写入负 competence。

ManagerBasedRLEnv 在当前 step 中先计算 termination / reward，之后才调用普通
`command_manager.compute()`。因此 actual-rotation state 不能只在普通 command hook 中刷新，
否则 reward 会固定滞后一帧。未来 command 应提供按 `env.common_step_counter` 加戳的幂等刷新：
termination、reward 或 command hook 中第一个 consumer 读取进度时完成一次 post-physics 更新；
同一 policy step 的后续 consumer 必须 no-op。curriculum 在 command reset 前读取刚结束 episode
的累计值，随后 command partial reset 再清对应 env。该生命周期需要独立 contract test。

== Keypoint pose reward 与 success ==

在 object frame `{o}` 定义半径 5 cm 的六个轴向 keypoints：

$$
\mathcal{P}_o
=
\left\{
\pm r e_x,
\pm r e_y,
\pm r e_z
\right\},
\qquad
r=0.05\ \mathrm{m}.
$$

full-pose keypoint distance 同时包含位置和姿态误差：

$$
x_i^{\{h\}}
=
\left\|
p_o^{\{h\}}+R_{ho}r_i^{\{o\}}-
\left(p_{anchor}^{\{h\}}+R_{hg}r_i^{\{o\}}\right)
\right\|_2.
$$

稠密 reward 使用归一化到 1 的 AnyRotate logistic kernel：

$$
r_{kp}
=
\frac{1}{6}
\sum_{i=1}^{6}
\frac{4}
{\exp(ax_i^{\{h\}})+b+\exp(-ax_i^{\{h\}})},
\qquad
a=50,
\quad
b=2.
$$

success 不直接对 full-pose 平均距离设一个阈值。姿态与位置分别使用中心对齐 keypoint
距离和显式位置门，避免同一个米制预算让姿态精度补偿过多滑移：

$$
d_{kp}^{rot}
=
\frac{1}{6}
\sum_{i=1}^{6}
\left\|
R_{ho}r_i^{\{o\}}-R_{hg}r_i^{\{o\}}
\right\|_2,
$$

$$
d_{pos}
=
\left\|
p_o^{\{h\}}-p_{anchor}^{\{h\}}
\right\|_2,
$$

$$
success
=
\left[d_{kp}^{rot}<0.005\ \mathrm{m}\right]
\land
\left[d_{pos}<0.025\ \mathrm{m}\right].
$$

5 mm orientation-only keypoint threshold在 5 cm 半径下约对应 8.6 degree 姿态残差。该数值与
25 mm 位置门是第一版锚点，build 时必须用 reset-state 与旧 checkpoint rollout 分布核验，
不能复制 AnyRotate 论文中缩放不明确的 `d_tol=0.15/0.25`。

== Object-filtered contact state ==

contact topology 由 generated hand sidecar 推导，不能按字符串排序或 action slot 顺序硬编码。
`right_t4_i4_m4_r4` 的四个 actor-facing fingertip bits 按 sidecar finger 顺序排列；实现和
contract test 必须记录该顺序，并显式处理它与 official joint slot order 的差异。

每个 link 对 object 的有效接触力先在 body/filter contact pair 上取最大幅值，避免向量相消：

$$
f_{i,t}
=
\max_j
\left\|
F_{i,j,t}
\right\|_2.
$$

tip 与 finger-non-tip 共用 policy-rate EMA 和 0.25 N threshold：

$$
\bar f_{i,t}
=
0.5\bar f_{i,t-1}
+
0.5f_{i,t},
$$

$$
c_{i,t}
=
\mathbf{1}
\left[
\bar f_{i,t}>0.25\ \mathrm{N}
\right].
$$

同一个 stateful contact buffer 必须同时服务 actor observation 与 contact reward；不得让
observation 和 reward 各自更新 EMA 或使用不同 force reduction。EMA 每个 policy step 只更新
一次，partial reset 只清对应 env。掌面传感器保留用于 critic 与诊断，但 palm contact 是合法、
中性的支撑。bad-contact 只统计 19 个 finger non-tip links，不包含 palm。

== Actor observations: 两个薄 variant ==

共享的单帧部署观察为：

$$\large
x_t=
\left[
q_t/\pi,
u_t/\pi,
a_{t-1}^{policy},
c_t^{tip}
\right]
\in\mathbb{R}^{52}.
$$

`a_{t-1}^{policy}` 是 rl_games 上一步采样并经过 Isaac Lab wrapper 限幅后送入 ActionManager 的
无量纲 command；它不是 ADR-processed command、rad 增量、target position 或真实 joint motion。
`u_t` 是 rad target buffer，`q_t` 是测得的 rad joint position。

CurrentObs variant 只返回当前 52D frame，供 GRU actor 使用。GRU 的 1.5 s 训练上下文由
rl_games `seq_length=30` 表达，不由 ObservationTerm history 表达。

History30Obs variant 返回 causal 30-frame window：

$$\large
X_t=
\left[
x_{t-29},\ldots,x_t
\right]
\in\mathbb{R}^{30\times52}.
$$

episode reset 后不足 30 帧的前缀用 reset 后当前 frame 重复填充，不使用全零伪关节状态。
两种 variant 的 scene、action、command、reward、critic、curriculum、ADR 与 termination 必须是
同一个 base assembly；只允许 policy observation shape 不同。

环境层未来注册两个语义 ID：`CurrentObs` 与 `History30Obs`。`GRU` / `TCN` 名称只出现在
`distill.rl` 的训练 alias，不写入 tasks 层环境 ID。

== Privileged central critic ==

两个 actor 共享同一份 critic observation schema 和独立同构的 feed-forward critic。critic 每个
policy step 读取当前完整 state，不需要 30-frame history：

$$\large
s_t^V=
\left[
q,
\dot q,
u,
a_{t-1}^{policy},
p_o^h-p_{anchor}^h,
\operatorname{rot6d}\left(R_g^{-1}R_o\right),
v_o^h,
\omega_o^h,
f_{tip},
f_{palm},
c_{finger-non-tip},
\xi_{ADR},
\lambda_{rew}
\right].
$$

critic 使用 4 个 tip force magnitudes、1 个 palm support force magnitude 和 19 个 finger
non-tip bits。ADR state 应包含当前实际生效的 mass、COM、friction/restitution bucket、PD gains、
action noise、latency、wrench gate/scale 与 ADR fraction，而不是只放 scheduler 档位。

critic MLP 第一版使用 `[512,256,128]`，通过 rl_games `central_value_config` 与 actor 完全分参。
主 network 的 `separate=true` 不是 asymmetric critic 开关，不得用它代替 central value path。
部署只导出 actor，不携带 privileged critic。

== 组合式 reward 与时间量纲 ==

reward 分为四组：

$$\large
r=
r_{rotation}+
\lambda_{rew}
\left(
r_{contact}+r_{stable}
\right)+
r_{terminate}.
$$

rotation group：

$$\large
r_{rotation}=
\lambda_{kp}r_{kp}+
\lambda_{rot}r_{rot}+
\lambda_{goal}r_{goal}.
$$

`r_rot` 使用有向实际转角并在 0.025 rad 裁剪；为抵消 RewardManager 的 `dt`，term callable
返回 clipped delta angle 除以 `step_dt`，使积分贡献等于本步转角。goal bonus 是离散 impulse，
term callable 返回 success indicator 除以 `step_dt`，保证一次 goal 的总贡献不随 20/30 Hz 改变。

contact group：

$$\large
r_{contact}=
\lambda_{gc}
\mathbf{1}
\left[
n_{tip}\ge2
\right]-
\lambda_{bc}
\mathbf{1}
\left[
n_{finger-non-tip}>0
\right].
$$

palm contact 不进入 good 或 bad indicator。另记录排除 palm 的 fingertip force share：

$$
D_{tip}=
\frac{
\sum_{i\in tip}f_i
}{
\sum_{i\in tip}f_i+
\sum_{j\in finger-non-tip}f_j+
\epsilon
}.
$$

stable group：

$$ \large
\begin{aligned}
r_{stable}={}&
\lambda_{speed}r_{speed}+
\lambda_{jitter}r_{jitter}+
\lambda_{axis}r_{axis}+
\lambda_{linvel}r_{linvel}\\&+
\lambda_{pose}r_{pose}+
\lambda_{work}r_{work}+
\lambda_{torque}r_{torque}\\&+
\lambda_{action}r_{action}+
\lambda_{rate}r_{action-rate}.
\end{aligned}
$$

目标轴角速度为：

$$\large
\omega_{\parallel,t}
=
\hat{k}^{\{h\}\mathsf{T}}\omega_{o,t}^{\{h\}}.
$$

速度 EMA 的时间常数配置为 0.25 s，按实际 policy dt 精确离散：

$$
\alpha_{\omega}=
1-\exp
\left(
-\frac{\Delta t_{policy}}{0.25\ \mathrm{s}}
\right),
$$

$$\large
\bar\omega_t=
\left(1-\alpha_{\omega}\right)\bar\omega_{t-1}+
\alpha_{\omega}\omega_{\parallel,t}.
$$

20 Hz 下该系数约为 0.181。速度带与抖动项为：

$$
d_{band}=
\operatorname{ReLU}
\left(0.6-\bar\omega_t\right)+
\operatorname{ReLU}
\left(\bar\omega_t-0.833\right),
$$

$$
r_{speed}=-d_{band}^2,
\qquad
r_{jitter}=
-\left(\omega_{\parallel,t}-\bar\omega_t\right)^2.
$$

非目标轴摆动与物体平移为：

$$
r_{axis}=
-\left\|
\omega_o^{\{h\}}-
\omega_{\parallel}\hat{k}^{\{h\}}
\right\|_2^2,
\qquad
r_{linvel}=
-\left\|v_o^{\{h\}}\right\|_1.
$$

reset grasp pose、机械功率和 torque 采用物理闭合定义：

$$
r_{pose}=
-\left\|
q_t-q_{anchor}
\right\|_2,
$$

$$
r_{work}=
-\sum_i
\left|
\tau_i\dot q_i
\right|,
$$

$$
r_{torque}=
-\sum_i\tau_i^2.
$$

`q_anchor` 是本 episode reset 后实际采用的抓取姿态，不是永久固定 YAML target。work 是功率 rate，
RewardManager 乘 `dt` 后近似机械能量；不得使用依赖关节零位的 `tau^T u`。action L2 与
action-rate 都对无量纲 policy command 计算，并随 reward curriculum 释放。

持续状态项保持 rate 语义，由 RewardManager 乘 `dt` 积分。termination penalty 与 goal bonus
是 impulse，需在 term 内除以 `step_dt`。禁止再次把整个 combined reward 除以 `step_dt`。

第一版数值锚点从 AnyRotate 出发：`lambda_kp=1`、`lambda_rot=5`、`lambda_goal=10`、
`lambda_gc=0.1`、`lambda_bc=0.2`、`lambda_pose=0.5`、`lambda_work=0.1`、
`lambda_torque=0.05`、`lambda_terminate=50`。由于 `r_kp` 已归一化到 1，且本任务允许 palm
支撑，以上只是 probe 初值，不是论文绝对尺度复刻。新增 speed/jitter/axis/linvel/action-rate
权重必须先用 N030/N051/N052 checkpoint rollout 统计 raw rate 与 weighted episode contribution
再决定。

每个 reward term 都必须日志化 raw mean per step、weighted mean per step、episode integral 与
group episode integral。零权重 term 会被 RewardManager 跳过，诊断 metric 不得伪装成零权重 reward。

== Reward curriculum 与 ADR ==

contact 与 stable 两组共享独立的 reward-release curriculum。它读取刚结束 episode 的真实净
旋转圈数，而不是 subgoal count：

$$
G_{k+1}
=
\left(1-\beta\right)G_k
+
\beta
\operatorname{mean}_{i\in\mathcal{E}_k}
\left(
\max
\left(
\frac{\Psi_i}{2\pi},0
\right)
\right),
$$

$$
\lambda_{rew}
=
\operatorname{clip}
\left(
\frac{G_k-1}{2-1},
0,
1
\right).
$$

新增 net-rotation reward curriculum，不修改现有 goal-success curriculum 的含义。
新 tactile rotation env 只配置 net-rotation 版本；两个 curriculum 不叠加。旧 goal-success
版本继续服务已有 GM probe，待新基线验证和迁移完成后再决定是否出清。

ADR 继续复用 LEAP 的 mass、friction、restitution、PD、reset noise、action noise/latency 和 wrench
ranges，但新增明确的 net-rotation-rate scheduler，不改变旧 `LeapADRGlobalScheduler`：

$$
C_{k+1}^{turn}
=
\left(1-\beta_{ADR}\right)C_k^{turn}
+
\beta_{ADR}
\operatorname{mean}_{i\in\mathcal{E}_k}
\left(
\max
\left(
\frac{\Psi_i}{2\pi},0
\right)
\right),
$$

$$
R_k^{turn}
=
\frac{C_k^{turn}}{\bar T_{sampled}}
\ge
0.08\ \mathrm{turns/s}.
$$

分母使用 sampled full horizon，不使用提前掉落时的实际存活时间，防止“高速转一会儿立即掉落”
获得晋级。配置和日志必须直接使用 `turns/s` 命名，不能继续叫含糊的 `ADRScore` 或
`min_rot_adr_coeff`。

== Termination ==

object 相对 episode reset anchor 偏移 7 cm 时终止：

$$
\left\|
p_o^h-p_{anchor}^h
\right\|_2
\ge
0.07\ \mathrm{m}.
$$

object 轴与 goal 轴有符号夹角超过 45 degree 时终止：

$$
z_o^{\{h\}\mathsf{T}}z_g^{\{h\}}
<
\cos
\left(45^\circ\right).
$$

不得使用 absolute dot。绕目标 z 轴的任意 yaw（包括 180 degree）仍保持法向对齐，是合法的任务
旋转；这里拒绝的是绕横轴翻面后 object 法向反向或偏离超过 45 degree。timeout 继续使用 LEAP
sampled horizon。success 只推进 subgoal，不结束 episode。

== Distill 训练对照 ==

GRU 与 TCN 使用相同 seed、4096 env、MDP、critic、PPO optimizer 与 transition budget：

$$
N_{batch}
=
4096\times30
=
122880.
$$

每轮仍分为 4 个 minibatch：

$$
N_{minibatch}
=
\frac{122880}{4}
=
30720.
$$

GRU 的 `seq_length=30` 因而使每个 minibatch 包含 1024 条完整 30-step sequence。TCN 使用
相同的 `horizon_length=30` 与 `minibatch_size=30720`，保证 PPO 采样和更新预算可比。
central value config 也必须显式设置 `minibatch_size=30720`；rl_games 不会自动继承 actor
config 的 minibatch 字段。

== 实现前 acceptance checks ==

1. 纯 tensor：full-pose keypoint 对平移/旋转单调；orientation-only success 与位置门解耦。
2. 纯 tensor：left-increment 轴向进度的正转、反转、往返和 quaternion 符号不变性。
3. 纯 tensor：reward rate / impulse 在 20 Hz 与 30 Hz 离散下 episode integral 一致。
4. config contract：base MDP 完全相同，两个 variant 只改变 policy observation shape。
5. runtime smoke：target 每 policy step 只累计一次 `a/24`，六个 physics substep hold。
6. runtime smoke：tip/palm/19 finger-non-tip sensor 都可独立翻转，不全零、不全饱和。
7. runtime smoke：contact EMA 每 policy step 只更新一次，partial reset 只清指定 env。
8. actor leakage test：52D/30x52 输入不含 object/goal/non-tip/ADR privileged state。
9. central critic test：rl_games 确实消费 `states`，而不是仅由 env 计算后丢弃。
10. TCN reset test：不足 30 帧前缀重复 reset frame，且 history 最后一帧严格等于当前 frame。
11. 训练前 reward probe：N052 应被 keypoint/speed 压低，N030 应被 finger-non-tip 压低，
    N051 应保留 rotation 优势但在 speed/jitter 上受约束。

本 scaffold 不授权提前声明未来 cfg class、MDP callable、network builder、Gym ID 或 YAML 字段。
进入 build 后应先以 contract tests 固化上述公式、shape、frame 与时间量纲，再添加 executable symbols。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp as gm_mdp
from .single_asset_env_cfg import (
    GM_SINGLE_ASSET_CONTACT_LAYOUT,
    GM_SINGLE_ASSET_GRASP_PRESET,
    GM_SINGLE_ASSET_HAND_SPAWN_CFG,
    GmSingleAssetSceneCfg,
)

TACTILE_JOINT_CFG = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)
r"""Action、$q$、$u$、critic gains 共用的 canonical articulation joint order。"""

TACTILE_TIP_SENSOR_NAMES = GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_sensor_names
TACTILE_FINGER_NON_TIP_SENSOR_NAMES = GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_non_tip_sensor_names
TACTILE_PALM_SENSOR_NAME = GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_sensor_name
r"""Sidecar role order：4 tips、19 finger non-tips 与独立 neutral palm support。"""


def _contact_params() -> dict[str, object]:
    r"""返回每个 consumer 独立持有的 shared contact-state 参数 dict。"""

    return {
        "fingertip_sensor_names": TACTILE_TIP_SENSOR_NAMES,
        "finger_non_tip_sensor_names": TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
        "palm_sensor_name": TACTILE_PALM_SENSOR_NAME,
        "ema_alpha": 0.5,
        "force_threshold": 0.25,
    }


@configclass
class GmTactileRotationSceneCfg(GmSingleAssetSceneCfg):
    r"""Generated `right_t4_i4_m4_r4` + DexCube palm-supported scene。"""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),  # N052 DexCube preset；prestartup event 随后逐 env 写 U(1.1,1.25)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GM_SINGLE_ASSET_GRASP_PRESET.object_pos_cfg,
            rot=GM_SINGLE_ASSET_GRASP_PRESET.object_rot_wxyz,
        ),
    )


@configclass
class GmTactileRotationCommandsCfg:
    r"""固定 hand `+z` 轴、30 degree moving-subgoal command。"""

    goal_pose: gm_mdp.TactileRotationCommandCfg = gm_mdp.TactileRotationCommandCfg(
        asset_name="object",
        robot_asset_name="robot",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
        subgoal_angle=3.141592653589793 / 6.0,
        keypoint_radius=0.05,
        orientation_keypoint_success_threshold=0.005,
        position_success_threshold=0.025,
        speed_ema_time_constant_s=0.25,
        diagnostics_action_name="hand_joint_pos",
        diagnostics_fingertip_sensor_names=TACTILE_TIP_SENSOR_NAMES,
        diagnostics_finger_non_tip_sensor_names=TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
        diagnostics_palm_sensor_name=TACTILE_PALM_SENSOR_NAME,
        diagnostics_contact_ema_alpha=0.5,
        diagnostics_contact_force_threshold=0.25,
        resampling_time_range=(1.0e6, 1.0e6),
    )


@configclass
class GmTactileRotationActionsCfg:
    r"""20 Hz policy-step-once target update；6 个 120 Hz substeps 只 hold。"""

    hand_joint_pos = gm_mdp.PolicyStepADRTargetJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=1.0 / 24.0,
        use_zero_offset=True,
        use_adr=True,
        pregrasp_joint_pos=(),  # action term 从 articulation default_joint_pos 读取 canonical-order preset
    )


def _policy_term(
    func,
    *,
    params: dict[str, object] | None = None,
    scale: float | None = None,
    history_length: int = 1,
    flatten_history_dim: bool = True,
) -> ObsTerm:
    r"""构造 actor semantic term，并让 Current/History routes 只改变统一 history 参数。"""

    return ObsTerm(
        func=func,
        params={} if params is None else params,
        scale=scale,
        history_length=history_length,
        flatten_history_dim=flatten_history_dim,
    )


@configclass
class GmTactileRotationCurrentPolicyObsCfg(ObsGroup):
    r"""GRU route 的 `[B,52]` 当前 deployment frame，按字段声明顺序拼接。"""

    joint_pos = _policy_term(
        gm_mdp.tactile_joint_position,
        params={"robot_cfg": TACTILE_JOINT_CFG},
        scale=1.0 / math.pi,
    )  # `[B,16]`，$q_t/\pi$
    joint_target = _policy_term(
        gm_mdp.tactile_joint_target,
        params={"action_name": "hand_joint_pos"},
        scale=1.0 / math.pi,
    )  # `[B,16]`，$u_t/\pi$
    last_policy_action = _policy_term(
        gm_mdp.tactile_last_policy_action,
        params={"action_name": "hand_joint_pos"},
    )  # `[B,16]`，$a_{t-1}^{policy}$
    tip_contact_bits = _policy_term(gm_mdp.tactile_tip_contact_bits, params=_contact_params())  # `[B,4]`

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GmTactileRotationHistory30PolicyObsCfg(ObsGroup):
    r"""TCN route 的 `[B,30,52]` oldest-to-latest semantic-term histories。"""

    joint_pos = _policy_term(
        gm_mdp.tactile_joint_position,
        params={"robot_cfg": TACTILE_JOINT_CFG},
        scale=1.0 / math.pi,
        history_length=30,
        flatten_history_dim=False,
    )
    joint_target = _policy_term(
        gm_mdp.tactile_joint_target,
        params={"action_name": "hand_joint_pos"},
        scale=1.0 / math.pi,
        history_length=30,
        flatten_history_dim=False,
    )
    last_policy_action = _policy_term(
        gm_mdp.tactile_last_policy_action,
        params={"action_name": "hand_joint_pos"},
        history_length=30,
        flatten_history_dim=False,
    )
    tip_contact_bits = _policy_term(
        gm_mdp.tactile_tip_contact_bits,
        params=_contact_params(),
        history_length=30,
        flatten_history_dim=False,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GmTactileRotationCriticObsCfg(ObsGroup):
    r"""当前 152D central critic；shared fields 与 actor 采用完全相同的数值口径。"""

    joint_pos = ObsTerm(
        func=gm_mdp.tactile_joint_position,
        params={"robot_cfg": TACTILE_JOINT_CFG},
        scale=1.0 / math.pi,
    )  # 16D $q_t/\pi$
    joint_velocity = ObsTerm(
        func=gm_mdp.tactile_joint_velocity,
        params={"robot_cfg": TACTILE_JOINT_CFG},
    )  # 16D raw rad/s
    joint_target = ObsTerm(
        func=gm_mdp.tactile_joint_target,
        params={"action_name": "hand_joint_pos"},
        scale=1.0 / math.pi,
    )  # 16D $u_t/\pi$
    last_policy_action = ObsTerm(
        func=gm_mdp.tactile_last_policy_action,
        params={"action_name": "hand_joint_pos"},
    )  # 16D
    object_task_state = ObsTerm(
        func=gm_mdp.tactile_object_task_state,
        params={
            "command_name": "goal_pose",
            "semantic_R_ha": GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
            "robot_cfg": TACTILE_JOINT_CFG,
            "object_cfg": SceneEntityCfg("object"),
        },
    )  # 15D pose/velocity state
    tip_force_ema = ObsTerm(func=gm_mdp.tactile_tip_force_ema, params=_contact_params())  # 4D N
    palm_force_ema = ObsTerm(func=gm_mdp.tactile_palm_force_ema, params=_contact_params())  # 1D N
    finger_non_tip_bits = ObsTerm(func=gm_mdp.tactile_finger_non_tip_bits, params=_contact_params())  # 19D
    adr_actual = ObsTerm(func=gm_mdp.gm_adr_state_observation, params={"action_dim": 16})  # 48D
    reward_release = ObsTerm(func=gm_mdp.tactile_reward_release_coefficient)  # 1D $\lambda_{rew}$

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GmTactileRotationCurrentObservationsCfg:
    r"""CurrentObs actor + shared critic groups。"""

    policy: ObsGroup = GmTactileRotationCurrentPolicyObsCfg()
    critic: ObsGroup = GmTactileRotationCriticObsCfg()


@configclass
class GmTactileRotationHistory30ObservationsCfg:
    r"""History30Obs actor + shared critic groups。"""

    policy: ObsGroup = GmTactileRotationHistory30PolicyObsCfg()
    critic: ObsGroup = GmTactileRotationCriticObsCfg()


@configclass
class GmTactileRotationRewardsCfg:
    r"""Rotation + curriculum-gated contact/stable + failure impulse reward。

    Isaac Lab `Episode_Reward/*` tags 表示各 weighted term 的 episode integral，再除以配置的
    `max_episode_length_s=120 s`；它不是 raw reward，也不是按实际提前终止时长归一化的 rate。
    实际 episode duration 由 `Metrics/goal_pose/task/episode_duration_s` 单独记录。
    """

    pose_keypoint = RewTerm(
        func=gm_mdp.tactile_full_pose_keypoint_reward,
        weight=1.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "keypoint_radius": 0.05},
    )
    rotation_progress = RewTerm(
        func=gm_mdp.tactile_axis_delta_rotation_rate,
        weight=5.0,
        params={"command_name": "goal_pose", "clip_value": 0.025},
    )
    goal_success = RewTerm(
        func=gm_mdp.tactile_goal_success_impulse,
        weight=10.0,
        params={"command_name": "goal_pose"},
    )
    good_tip_contact = RewTerm(
        func=gm_mdp.tactile_good_tip_contact,
        weight=0.1,
        params={**_contact_params(), "min_contacts": 2},
    )
    bad_finger_non_tip_contact = RewTerm(
        func=gm_mdp.tactile_bad_finger_non_tip_contact,
        weight=-0.2,
        params=_contact_params(),
    )
    speed_band = RewTerm(
        func=gm_mdp.object_axis_speed_band_curriculum,
        weight=-0.5,
        params={"command_name": "goal_pose", "speed_min": 0.6, "speed_max": 0.833},
    )
    speed_jitter = RewTerm(
        func=gm_mdp.object_axis_speed_jitter_curriculum,
        weight=-0.05,
        params={"command_name": "goal_pose"},
    )
    off_axis_angular_velocity = RewTerm(
        func=gm_mdp.object_off_axis_ang_vel_curriculum,
        weight=-0.05,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object")},
    )
    object_linear_velocity = RewTerm(
        func=gm_mdp.object_lin_vel_l2_curriculum,
        weight=-0.2,
        params={"object_cfg": SceneEntityCfg("object")},
    )
    joint_pose_anchor = RewTerm(
        func=gm_mdp.joint_pose_anchor_l2_curriculum,
        weight=-0.5,
        params={"asset_cfg": TACTILE_JOINT_CFG},
    )
    mechanical_power = RewTerm(
        func=gm_mdp.joint_mechanical_power_curriculum,
        weight=-0.1,
        params={"asset_cfg": TACTILE_JOINT_CFG},
    )
    torque_l2 = RewTerm(
        func=gm_mdp.torque_l2_curriculum,
        weight=-0.05,
        params={"asset_cfg": TACTILE_JOINT_CFG, "lambda_floor": 0.0},
    )
    action_l2 = RewTerm(func=gm_mdp.action_l2_curriculum, weight=-1.0e-4, params={"lambda_floor": 0.0})
    action_rate_l2 = RewTerm(
        func=gm_mdp.action_rate_l2_curriculum, weight=-1.0e-2, params={"lambda_floor": 0.0}
    )
    failure = RewTerm(
        func=gm_mdp.failure_termination_impulse,
        weight=-50.0,
        params={"termination_term_names": ("object_out_of_anchor", "goal_axis_misaligned")},
    )


@configclass
class GmTactileRotationTerminationsCfg:
    r"""7 cm anchor、signed 45 degree normal alignment 与 sampled full-horizon timeout。"""

    object_out_of_anchor = DoneTerm(
        func=gm_mdp.tactile_object_out_of_anchor,
        params={"command_name": "goal_pose", "fall_dist": 0.07},
    )
    goal_axis_misaligned = DoneTerm(
        func=gm_mdp.tactile_goal_axis_misaligned,
        params={"command_name": "goal_pose", "max_angle_deg": 45.0},
    )
    time_out = DoneTerm(func=gm_mdp.adr_randomized_time_out, time_out=True)


@configclass
class GmTactileRotationEventsCfg:
    r"""N052 ranges + AnyRotate default-relative COM + shared-state reset events。"""

    apply_structural_collision_filter = EventTerm(
        func=gm_mdp.apply_generated_structural_collision_filter,
        mode="prestartup",
        params={
            "robot_prim_path": "{ENV_REGEX_NS}/Robot",
            "palm_link_name": GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
            "finger_link_chains": GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains,
            "filter_palm_finger": True,
            "filter_same_finger": True,
        },
    )
    randomized_object_scale = EventTerm(
        func=gm_mdp.randomize_object_scale_and_record,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("object"), "scale_range": (1.1, 1.25)},
    )
    initialize_object_material = EventTerm(
        func=gm_mdp.RandomizeRigidBodyMaterialAndRecord,  # pyright: ignore[reportArgumentType]  # class-term runtime contract
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=[".*"]),
            "static_friction_range": gm_mdp.GM_ADR_OBJECT_MATERIAL_INITIAL.static,
            "dynamic_friction_range": gm_mdp.GM_ADR_OBJECT_MATERIAL_INITIAL.dynamic,
            "restitution_range": gm_mdp.GM_ADR_OBJECT_MATERIAL_INITIAL.restitution,
            "num_buckets": 250,
            "make_consistent": True,
            "adr_state_field": "object_material",
        },
    )
    initialize_hand_contact_material = EventTerm(
        func=gm_mdp.RandomizeRigidBodyMaterialAndRecord,  # pyright: ignore[reportArgumentType]  # class-term runtime contract
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(GM_SINGLE_ASSET_CONTACT_LAYOUT.all_link_names)),
            "static_friction_range": gm_mdp.GM_ADR_HAND_MATERIAL_INITIAL.static,
            "dynamic_friction_range": gm_mdp.GM_ADR_HAND_MATERIAL_INITIAL.dynamic,
            "restitution_range": gm_mdp.GM_ADR_HAND_MATERIAL_INITIAL.restitution,
            "num_buckets": 250,
            "make_consistent": True,
            "adr_state_field": "hand_contact_material",
        },
    )
    randomized_object_mass = EventTerm(
        func=gm_mdp.RandomizeRigidBodyMassAndRecord,  # pyright: ignore[reportArgumentType]  # class-term runtime contract
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomized_actuator_gains = EventTerm(
        func=gm_mdp.RandomizeActuatorGainsAndRecord,  # pyright: ignore[reportArgumentType]  # class-term runtime contract
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": TACTILE_JOINT_CFG,
            "stiffness_distribution_params": (3.0, 3.0),
            "damping_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    resample_object_material_from_adr = EventTerm(
        func=gm_mdp.resample_adr_material_buckets,
        mode="reset",
        min_step_count_between_reset=720,
        params={"term_name": "initialize_object_material", "range_attr": "leap_adr_object_material_ranges"},
    )
    resample_hand_contact_material_from_adr = EventTerm(
        func=gm_mdp.resample_adr_material_buckets,
        mode="reset",
        min_step_count_between_reset=720,
        params={"term_name": "initialize_hand_contact_material", "range_attr": "leap_adr_robot_material_ranges"},
    )
    randomized_object_com = EventTerm(
        func=gm_mdp.randomize_object_com_from_default_and_record,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    reset_episode_length = EventTerm(
        func=gm_mdp.reset_adr_episode_length,
        mode="reset",
        params={"min_episode_length_s": 20.0},
    )
    reset_object_pose_from_adr = EventTerm(
        func=gm_mdp.reset_adr_object_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    reset_hand_joints_from_adr = EventTerm(
        func=gm_mdp.reset_adr_robot_joints,
        mode="reset",
        params={"asset_cfg": TACTILE_JOINT_CFG},
    )
    reset_wrench_gate = EventTerm(
        func=gm_mdp.reset_adr_wrench_state,
        mode="reset",
        params={"probability": 0.5, "asset_cfg": SceneEntityCfg("object")},
    )
    record_object_anchor = EventTerm(
        func=gm_mdp.record_object_reset_anchor,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("object")},
    )
    reset_contact_state = EventTerm(
        func=gm_mdp.reset_tactile_contact_state,
        mode="reset",
        params=_contact_params(),
    )
    object_wrench = EventTerm(
        func=gm_mdp.apply_adr_object_wrench,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={"asset_cfg": SceneEntityCfg("object"), "torsional_radius": 0.0},
    )


@configclass
class GmTactileRotationCurriculumCfg:
    r"""Actual net turns 独立驱动 reward release 与 LEAP ADR progression。"""

    reward_release = CurrTerm(
        func=gm_mdp.RewardCurriculumByNetRotation,  # pyright: ignore[reportArgumentType]  # ManagerTermBase class
        params={
            "command_name": "goal_pose",
            "release_start_turns": 1.0,
            "release_end_turns": 2.0,
            "ema_alpha": 0.05,
        },
    )
    adr = CurrTerm(
        func=gm_mdp.LeapADRByNetRotationRate,  # pyright: ignore[reportArgumentType]  # ManagerTermBase class
        params={
            "command_name": "goal_pose",
            "num_increments": 25,
            "threshold_turns_per_s": 0.08,
            "min_reset_checks_for_increase": 960,
            "ema_alpha": 0.1,
            "min_episode_length_s": 20.0,
            "episode_length_s": 120.0,
        },
    )


@configclass
class GmTactileRotationCurrentEnvCfg(ManagerBasedRLEnvCfg):
    r"""CurrentObs tactile rotation base assembly；GRU architecture 仍归 distill。"""

    is_finite_horizon: bool = True
    seed: int | None = 42
    scene: GmTactileRotationSceneCfg = GmTactileRotationSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    observations: GmTactileRotationCurrentObservationsCfg = GmTactileRotationCurrentObservationsCfg()
    actions: GmTactileRotationActionsCfg = GmTactileRotationActionsCfg()
    commands: GmTactileRotationCommandsCfg = GmTactileRotationCommandsCfg()
    rewards: GmTactileRotationRewardsCfg = GmTactileRotationRewardsCfg()
    terminations: GmTactileRotationTerminationsCfg = GmTactileRotationTerminationsCfg()
    events: GmTactileRotationEventsCfg = GmTactileRotationEventsCfg()
    curriculum: GmTactileRotationCurriculumCfg = GmTactileRotationCurriculumCfg()

    def __post_init__(self):
        r"""锁定 120 Hz physics、20 Hz policy 与 20--120 s sampled full horizon 上限。"""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]  # configclass injects base hook
        self.decimation = 6
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class GmTactileRotationHistory30EnvCfg(GmTactileRotationCurrentEnvCfg):
    r"""History30Obs variant；除 policy observation shape 外与 CurrentObs 完全相同。"""

    observations: GmTactileRotationHistory30ObservationsCfg = GmTactileRotationHistory30ObservationsCfg()


@configclass
class GmTactileRotationCurrentEnvCfg_PLAY(GmTactileRotationCurrentEnvCfg):
    r"""CurrentObs checkpoint 的少量环境 GUI/play variant。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


@configclass
class GmTactileRotationHistory30EnvCfg_PLAY(GmTactileRotationHistory30EnvCfg):
    r"""History30Obs checkpoint 的少量环境 GUI/play variant。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


__all__ = [
    "GmTactileRotationActionsCfg",
    "GmTactileRotationCommandsCfg",
    "GmTactileRotationCurrentEnvCfg",
    "GmTactileRotationCurrentEnvCfg_PLAY",
    "GmTactileRotationCurrentObservationsCfg",
    "GmTactileRotationCurriculumCfg",
    "GmTactileRotationEventsCfg",
    "GmTactileRotationHistory30EnvCfg",
    "GmTactileRotationHistory30EnvCfg_PLAY",
    "GmTactileRotationHistory30ObservationsCfg",
    "GmTactileRotationRewardsCfg",
    "GmTactileRotationSceneCfg",
    "GmTactileRotationTerminationsCfg",
]
