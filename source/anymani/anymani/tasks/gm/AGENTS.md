# AGENTS.md

`gm` means generalized manipulation in embodiments. 本子项目主要负责“手型泛化的手内操作”中的 Isaac Lab 任务环境部分。

## 边界

`gm` 只定义环境语义：scene、obs、action、reward、reset、termination、command，以及少量 Gym 注册入口。

不要把训练算法、checkpoint、rollout dataset、asset-bank split、asset 采样策略、网络结构塞进这里。这些属于 `distill`。

不要把资产生成、validator、mesh 物理闭包、post-mutate 逻辑塞进这里。这些属于 `assets`。

`gm` 可以消费一个已经选好的 hand asset，并声明它对 `hand.urdf` / `hand.yaml` 的最低 contract；但不拥有整个 asset bank。

reset 初始状态分布仍属 `gm` 任务语义：例如 hand joint reset、object pose reset、
object reset anchor 记录、hand orientation reset scaffold。能直接复用 IsaacLab
官方 MDP event 的 reset 不要再包一层；只有 AnyMani 专属的物理/坐标系语义才放在
`gm/mdp/events.py`。

## 设计风格

保持浅目录。任务差异优先通过 MDP 组件组合表达，不提前拆 `manipulation/`、`grasp/` 等深目录。

当前主线是 same-topology post-mutated hand assets 的层次通才 RL 环境；跨拓扑 unified policy、mesh feature learning、teacher-student distillation 暂由 `distill` 后续承接。

### 声明式配置驱动

ManagerBasedRLEnv 本身就是一个高度声明式、配置驱动的环境框架，鼓励通过配置项组合出不同的任务和环境变体，而不是通过写新的 Python 代码。对于需要新代码的情况，也应优先考虑在现有组件基础上添加可配置选项，而不是直接写一个新的组件。

在设计 MDP 项时，也可从算法第一性原理考虑哪些是可以从流程里拆下来、可配置的组建、超参，而不是写死在流程逻辑里。

### 测试策略

`gm` 的纯 MDP 逻辑应尽量 TDD：obs/action 量纲、SO(3) command、reward 曲线、termination anchor、reset event 拆分语义，都应能用 fake env / 小 tensor 做 contract test。需要 Isaac Sim 的 articulation loading、contact sensor、PhysX step、完整 reset/step，则用 headless smoke / integration test，不要求像纯函数一样严格 TDD。

凡实现会改变 reset 初始状态分布、reward 成功判据或坐标系语义，至少补一个能失败的最小测试或 smoke 记录，避免训练跑很久才发现数学方向错了。

GM 的测试应围绕“科研命题能在哪里被证伪”来选层级：配置组合、坐标系公式、reward 曲线和 reset 参数分布优先用 contract test；只在 Isaac Sim runtime 才存在的事实，例如 stage authoring、importer 结果、scene clone 后的实体、传感器读数、PhysX 初始化后的 handle 与 reset/step 生命周期，必须补 `source/anymani/anymani/smokes/isaacsim/` 下的显式 headless smoke。contract test 只能证明“我们声明了什么”，runtime smoke 才能证明“仿真实际看见了什么”。

IsaacSim smoke 不放入 `tasks/gm/tests/`，也不加入默认 `pytest.ini testpaths`。运行时必须显式指定路径，并用 `timeout --kill-after` 防止 Kit 卡住。单资产 generated structural collision filter 的当前 smoke 是一个具体例子：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

## 数学偏好

机器人学相关的算法、脚本、变量与注释，统一仿照《Modern Robotics》的群 / 李代数体系表达，保持学术严谨，不要临时拼凑角度约定。

**表示优先级**：

1. **首选位姿矩阵 / 旋转矩阵**（$T \in SE(3)$、$R \in SO(3)$）。它们与群同构，完备、无奇异、是唯一无歧义的双射，应作为位姿存储、复合、求逆、传递的默认载体。
2. **需要线性 / 向量形式时，优先轴角与旋量**（$\bm{\omega}\theta \in \mathbb{R}^3$、$\mathcal{S}\theta \in \mathbb{R}^6$，分别与 $so(3)$、$se(3)$ 同构）。它们是李代数上的线性量，便于做残差、插值、雅可比与特征；代价是 $\theta \to \pi$ 附近存在奇异，使用处须显式处理或回避。当前 command 即采用 axis + so(3) 形式。
3. **回避欧拉角 / RPY 与裸四元数作为内部表示**。仅在与外部接口对接的边界上换算：URDF、可视化、以及 Isaac Lab 官方 API（查询 body 位姿时它必然返回四元数）——拿到后尽快换算回矩阵 / 旋量再进入自己的逻辑。

记号示例，大小写区分向量与矩阵：
- $\text{Log}: SO(3) \to so(3)$ — 大写，返回**矩阵**（skew-symmetric）
- $\text{log}: SO(3) \to \mathbb{R}^3$ — 小写，返回**向量**

反过来：
- $\text{exp}: so(3) \to SO(3)$ — 矩阵指数
- $\text{Exp}: \mathbb{R}^3 \to SO(3)$

## Jargon

一些个人约定俗成，非官方的行话。

### 坐标系语义表述

`{w}`：Isaac Lab 全局世界坐标系。
`{e}`：每个 cloned env 的局部环境坐标系。它通常只相对 `{w}` 平移，姿态多与 `{w}` 对齐，因此不能表达“手自身朝向变化后的任务语义”。
`{a}`：raw asset/root frame，即 URDF/USD 被 Isaac Lab 加载后的资产根坐标系。它反映文件作者或 importer 的坐标约定，不必然等于 AnyMani 的手部建模语义。
`{h}`：hand semantic frame。它是 `gm` 任务语义真正依赖的手坐标系，由 AnyMani 资产建模约定定义：手处于 home position 时，手掌平面、手指展开方向、手心法向应与 `AnyMani/source/anymani/anymani/assets/doc/平面示意-右手.png` 的语义一致。对官方 LEAP / Allegro / 其他真实 URDF，允许通过人工视觉校准或配置项给出 `{a} -> {h}` 的固定对齐变换。
`{o}`：object body frame。它随物体自身旋转剧烈变化，不适合作为默认 command frame，但可用于计算物体当前姿态或局部几何观测。

`gm` 的 command 语义默认应锚定在 `{h}`，而不是 `{w}` / `{e}` / raw `{a}`。例如“绕 z 轴手内旋转”应解释为 `k^{h} = [0, 0, 1]`，即绕手心语义法向轴旋转物体；运行时再把该轴变换到 `{e}` 或 `{w}` 中供 reward、goal update 和 visualization 使用。

### Hand frame / orientation reset 约定

配置层优先使用旋转矩阵和平移向量 $(R,p)$，实现层应组合为 $T\in SE(3)$ 进行复合、求逆和传递；只有在 Isaac Lab 边界（如 `ArticulationCfg.InitialStateCfg.rot` 或 `write_root_pose_to_sim`）才转换为 `(w,x,y,z)` 四元数。`so3` / 全 $SO(3)$ 随机采样实现时可以使用四元数算法，但文档和中间语义仍以 $SO(3)$ / $SE(3)$ 表达。

`robots.hand_spawn.HandFrameCfg` 是静态装配锚点，记录：

- $T_{ha}$：raw asset/root frame `{a}` 到 hand semantic frame `{h}` 的固定校准；
- $T_{eh}^{anchor}$：hand semantic frame `{h}` 在 env frame `{e}` 中的默认参考 pose。

spawn 层只负责把已选资产按 anchor 装配进 scene，不负责 episode 级随机朝向。若需要任意 hand orientation 训练，reset event 应采样 hand semantic pose $T_{eh}$，再写入 raw root pose：

$$
T_{ea}=T_{eh}T_{ha}.
$$

orientation domain randomization 的默认语义是 hand-frame body/right 扰动：从 anchor 出发右乘 $\Delta R_h$，即 $R'_{eh}=R_{eh}^{anchor}\Delta R_h$。默认 reference mode 应为 `anchor`，保证 reset 初态是 i.i.d. 分布；`current` 随机游走只作为未来 continual perturbation / curriculum 预留。

## Progress

当前正在进行单资产 MLP 训练，来排查资产合理性和核验 MDP 模块正确性，有两个成功案例值得当前对比和借鉴

- LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py
  > LEAP Hand 官方的 IsaacLab 手内操作任务 demo，训练效果既快又好，只训练绕 z 轴的旋转
- IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/allegro_env_cfg.py
  > IsaacLab 官方的随机重定向任务，command 偏向当前我这版，训练效果也不错
