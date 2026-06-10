# AGENTS.md

`gm` means generalized manipulation in embodiments. 本子项目主要负责“手型泛化的手内操作”中的 Isaac Lab 任务环境部分。

## 边界

`gm` 只定义环境语义：scene、obs、action、reward、reset、termination、command，以及少量 Gym 注册入口。

不要把训练算法、checkpoint、rollout dataset、asset-bank split、asset 采样策略、网络结构塞进这里。这些属于 `distill`。

不要把资产生成、validator、mesh 物理闭包、post-mutate 逻辑塞进这里。这些属于 `assets`。

`gm` 可以消费一个已经选好的 hand asset，并声明它对 `hand.urdf` / `hand.yaml` 的最低 contract；但不拥有整个 asset bank。

## 设计风格

保持浅目录。任务差异优先通过 MDP 组件组合表达，不提前拆 `manipulation/`、`grasp/` 等深目录。

当前主线是 same-topology post-mutated hand assets 的层次通才 RL 环境；跨拓扑 unified policy、mesh feature learning、teacher-student distillation 暂由 `distill` 后续承接。

## 数学习惯

关于机器人学相关的变量、注释，仿照《Modern robotics》风格，保持学术严谨。
例如
- $\text{Log}: SO(3) \to so(3)$ — 大写，返回**向量**
- $\text{log}: SO(3) \to \mathbb{R}^3$ — 小写，返回**矩阵**（skew-symmetric）

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
