# AGENTS.md

`gm` means generalized manipulation in embodiments. 本子项目保留single-asset generated-hand probe、LEAP对照及其共享MDP原件；跨拓扑generated-hand任务属于并列的`tasks/hetero`。

## 边界

`gm` 定义环境语义：scene、obs、action、reward、reset、termination、command，以及 Gym 注册入口。

不要把训练算法、checkpoint、rollout dataset 或网络结构塞进这里。这些属于 `distill`。

不要把资产生成、validator、mesh 物理闭包、post-mutate 逻辑塞进这里。这些属于 `assets`。

环境variant可以声明single-asset或LEAP所需的可复现asset binding。跨拓扑dataset selection、canonical masks与
pregrasp provider由`tasks/hetero`装配；资产生成、bank构建和train/validation split仍不属于`gm`。

reset 初始状态分布仍属 `gm` 任务语义：例如 hand joint reset、object pose reset、
object reset anchor 记录、hand orientation reset scaffold。能直接复用 IsaacLab
官方 MDP event 的 reset 不要再包一层；只有 AnyMani 专属的物理/坐标系语义才放在
`gm/mdp/events.py`。

## 设计风格

保持浅目录。任务差异优先通过 MDP 组件组合表达，不提前拆 `manipulation/`、`grasp/` 等深目录。

`gm`、`tasks/inhand`与`tasks/hetero`是并列任务族：前两者保留single-hand/LEAP-style对照，后者拥有跨拓扑
generated-hand组合面。跨手型policy、teacher-student distillation和网络结构仍由`distill`承接。

### 声明式配置驱动

ManagerBasedRLEnv 本身就是一个高度声明式、配置驱动的环境框架，鼓励通过配置项组合出不同的任务和环境变体，而不是通过写新的 Python 代码。对于需要新代码的情况，也应优先考虑在现有组件基础上添加可配置选项，而不是直接写一个新的组件。

在设计 MDP 项时，也可从算法第一性原理考虑哪些是可以从流程里拆下来、可配置的组建、超参，而不是写死在流程逻辑里。

### Config 变体结构

`config/<variant>/`只放single asset或LEAP环境构型。每个variant可自包含scene/sim/command/action/
observation/reward/reset/termination/curriculum；不要把跨拓扑routing重新塞回GM兼容壳。

single-asset generated probe 位于 `config/single_asset/`；真实 LEAP 对照位于 `config/leap/`。根目录不保留
旧式 variant 兼容壳，避免废弃路径继续污染实验语义。

### 测试重点

仓库级 contract/runtime 分层遵守根 `AGENTS.md`，本文件只补充 GM 的高风险命题：frame algebra、command
reference、rotation representation、reset anchor、contact filter 与 reward buffer ownership 优先用纯
tensor/config contract；generated articulation、ContactSensor、USD collision pairs 和完整 reset/step 必须用
`source/anymani/anymani/smokes/isaacsim/` 的显式 smoke。改变 reset 分布、success 判据或 frame 语义时，
至少补一个能够直接证伪该变化的测试。

## 数学偏好

机器人学算法、变量与注释使用明确的 $SO(3)/SE(3)$、frame 和 reference 语义，但**不存在脱离用途的
统一表示优先级**。矩阵、rot6d、quaternion 与局部 log 各自解决不同问题：

- **几何复合与标定**：用 $R\in SO(3)$、$T\in SE(3)$ 写清 composition、inverse 和 frame chain。
  矩阵是群元素的自然实现，但其 9/16 个元素有约束且冗余，不能描述成无约束欧氏空间中的唯一双射。
- **局部姿态残差**：可用 $\phi=\log(R)^{\vee}\in\mathbb R^3$，但必须写明 principal branch。
  它只适合局部误差；接近 $\theta=\pi$ 时存在不可避免的分支不连续。
- **Simulator/runtime state**：Isaac Lab 的 `(w,x,y,z)` quaternion 可以作为 canonical buffer，并可直接做
  compose/inverse。不要因为旧偏好强制“读取后立即转走”。
- **Policy observation / command**：rot6d、matrix、quaternion 或 local log 都是实验 contract。必须声明
  frame、absolute/relative reference、reference 更新律、维度与连续性处理。Quaternion 进入网络时必须固定
  `(w,x,y,z)`，说明 `quat_unique` 或时间连续符号策略，并测试 $q\sim -q$ 的符号边界。
- **Markov 性**：相对量本身不天然非平稳；若 reference/goal 的移动状态对 policy 隐藏，才会造成部分可观测
  或表观非平稳。使用 relative log 时必须让 reference 状态或其确定更新律可推断，并检查 goal jump。
- **Euler/RPY**：只用于明确命名的 URDF、配置或 Isaac Lab API 字段；同时写明旋转顺序和参考 frame。

不要依赖全项目 `Log/log/Exp/exp` 大小写惯例；每个公式或函数在局部定义其输入、输出与 hat/vee 语义。

## Jargon

一些个人约定俗成，非官方的行话。

### 坐标系语义表述

`{w}`：Isaac Lab 全局世界坐标系。
`{e}`：每个 cloned env 的局部环境坐标系。它通常只相对 `{w}` 平移，姿态多与 `{w}` 对齐，因此不能表达“手自身朝向变化后的任务语义”。
`{a}`：raw asset/root frame，即 URDF/USD 被 Isaac Lab 加载后的资产根坐标系。它是资产文件天然存在的根 frame，反映文件作者或 importer 的坐标约定，不必然等于 AnyMani 的手部建模语义；对官方 LEAP / Allegro 这类真实资产，默认不通过改 URDF/USD 来“修正” `{a}`。训练配置中的 `ArticulationCfg.InitialStateCfg.pos/rot` 表达 $T_{ea}^{init}$，负责把这个 raw asset frame `{a}` 摆到任务需要的 env 姿态，例如官方 LEAP 对照中的 `(0,0,0.5)` 与 `(0.5,0.5,-0.5,0.5)`。
`{h}`：hand semantic frame。它是 `gm` 任务语义真正依赖的手坐标系，由 AnyMani 资产建模约定定义：手处于 home position 时，手掌平面、手指展开方向、手心法向应与 `AnyMani/source/anymani/anymani/assets/doc/平面示意-右手.png` 的语义一致。`{h}` 不负责把手摆到正确姿态；它只是固定附着在 `{a}` 上的语义锚点，供 command axis、object pose、contact force 等 MDP 项获得跨资产一致的手部语义。对官方 LEAP / Allegro / 其他真实 URDF，允许通过人工视觉校准或配置项给出 `{a} -> {h}` 的固定对齐变换，这是 sim2sim embodiment transfer 的先行语义对齐工作。
`{o}`：object body frame。它随物体自身旋转剧烈变化，不适合作为默认 command frame，但可用于计算物体当前姿态或局部几何观测。

`gm` 的 command 语义默认应锚定在 `{h}`，而不是 `{w}` / `{e}` / raw `{a}`。例如“绕 z 轴手内旋转”应解释为 `k^{h} = [0, 0, 1]`，即绕手心语义法向轴旋转物体；运行时再把该轴变换到 `{e}` 或 `{w}` 中供 reward、goal update 和 visualization 使用。

### Hand frame / orientation reset 约定

配置和文档用 $(R,p)$ 或 $T\in SE(3)$ 写清 frame chain；实现可以在矩阵与 `(w,x,y,z)` quaternion
之间选择最贴近当前 API 的 canonical state，只要组合方向和转换边界明确。全 $SO(3)$ 随机采样可使用
quaternion 算法，科研语义仍应说明采样的是 Haar-uniform 群分布还是受限局部扰动。

`robots.hand_spawn.HandFrameCfg` 是 generated-hand spawn 路径的静态装配锚点，记录：

- $T_{ha}$：raw asset/root frame `{a}` 到 hand semantic frame `{h}` 的固定校准；
- $T_{eh}^{anchor}$：hand semantic frame `{h}` 在 env frame `{e}` 中的默认参考 pose。

对 generated hand，spawn 层可以把已选资产按 hand semantic anchor 装配进 scene，不负责 episode 级随机朝向。若需要任意 hand orientation 训练，reset event 应采样 hand semantic pose $T_{eh}$，再写入 raw root pose：

$$
T_{ea}=T_{eh}T_{ha}.
$$

对 official LEAP / Allegro 这类对照资产，也允许暂时不走 `HandSpawnCfg` 装配公式，而是在对应 env cfg 里直接保留人工确认过的 `InitialStateCfg` root pose $T_{ea}^{init}$。此时 $T_{ea}^{init}$ 负责摆手，$T_{ah}$ / $T_{ha}$ 只负责描述 `{h}` 相对 `{a}` 的语义标定；不要把二者混成同一个“修正姿态”。若用户先用 VSCode URDF viewer 或 Isaac Sim viewer 标定出 $T_{ah}$，代码侧应按

$$
R_{ha}=R_{ah}^{\top},\qquad p_{ha}=-R_{ha}p_{ah}
$$

反算给 MDP 配置消费，同时保持 official asset 文件本身不变。

orientation domain randomization 目前只是 declarative scaffold，尚未由 active reset event 写入 sim。未来实现的
默认语义是 hand-frame body/right 扰动：从 anchor 右乘 $\Delta R_h$，即
$R'_{eh}=R_{eh}^{anchor}\Delta R_h$。`anchor` 表示 i.i.d. reset；`current` 只允许用于显式 continual
perturbation/curriculum，不能静默变成 episode 间随机游走。
