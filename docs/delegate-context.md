# Delegate Context: Graph-Structured Per-Finger Adaptation Idea Discussion

## 背景

用户是做灵巧手手内操作（in-hand manipulation）的科研人员，平台是 LeapHand + Isaac Lab。已有一个可训练的手内旋转环境（AnyMani 项目）。

### 导师反馈要点
1. paper 应该是**方法导向**，不是问题导向
2. 核心策略：选一个你最熟悉的方法，深入理解其核心优势，**平移到没人用过该方法的新问题/新组合**
3. 在新场景中发挥出方法的独特优势，就足够发表
4. finger gait 方向已被否定（"学到步态本身不是创新"）

### 用户当前状态
- finger gait 方向已放弃
- 最感兴趣的方法：**图注意力 + RMA + joint-wise dynamics**
- 任务：继续做 in-hand manipulation
- 希望最终有 sim-to-real
- 目标会议：CoRL / ICRA

### 用户初步想法
> "RMA 是否可以是单根手指的？设计 4 个节点（代表手指）和 1 个节点（代表物体）。"

## 已读文献（共 7 篇，Agent 已全部读完正文）

### 1. Qi 等 2022 - HORA (In-Hand Object Rotation via Rapid Motor Adaptation)
- **方法**：RMA 范式——先训练带特权信息的 base policy，再训练 adaptation module 从本体感受历史估计 extrinsics（8 维 latent，编码质量/尺寸等）
- **核心**：extrinsics 是**全局**的，整个手共享一个 adaptation latent
- **Finger gait**：自然涌现，不是显式设计
- **局限**：只做 z 轴旋转，只用 proprioception

### 2. Qi 等 2023 - RotateIt (General In-Hand Object Rotation with Vision and Touch)
- **方法**：在 HORA 基础上加入视觉 + 触觉，用 Visuotactile Transformer 替代 MLP adaptation module
- **核心**：显式编码物体形状（PointNet），多模态感知提升性能
- **Extrinsics**：包含 shape encoding + physics encoding，仍然是**全局**的
- **多轴**：通过蒸馏实现单策略多轴旋转

### 3. Tao 等 2023 - MAGCLA (Multi-Agent for Finger Cooperation)
- **方法**：MARL，每根手指是独立 agent，CTDE 框架
- **核心**：actor 可观察相邻 agent 的动作，critic 全局观察
- **发现**：MAGCLA 涌现出 gaiting（保守稳定），DDPG 涌现出 tossing（激进快速）
- **局限**：MuJoCo + Shadow Hand，没有 sim-to-real，没有图结构

### 4. Yang 等 2024 - AnyRotate (Gravity-Invariant In-Hand Object Rotation)
- **方法**：dense featured tactile + 辅助目标公式 + 自适应课程
- **核心**：统一策略实现任意轴 + 任意手朝向旋转
- **Sim-to-real**：通过 CNN 从触觉图像预测 contact pose + contact force
- **Finger gait**：涌现；rich tactile 让策略能检测不稳定抓握并做出 reactive 调整

### 5. Patel & Song 2024 - GET-Zero (Graph Embodiment Transformer)
- **方法**：Graph attention bias in transformer encoder，编码机器人运动学图
- **核心**：用 SPD（最短路径距离）和 parent-child 关系作为 attention bias
- **应用**：跨构型 LEAP Hand 零样本控制（删/加关节+链接）
- **关键细节**：
  - 每个关节是一个 token，包含 local obs + fixed embodiment info
  - Graph encoding 是在 attention score 上加 learned bias
  - Self-modeling loss（预测 FK）显著提升泛化
- **局限**：只编码了 robot 图，没有 object 节点；只做 in-hand rotation

### 6. Liu 等 2025 - DexNDM (Joint-Wise Neural Dynamics Model)
- **方法**：将整手动力学因式分解为 per-joint dynamics，每个关节只从自身历史预测下一状态
- **核心**：
  - information bottleneck——丢弃不相关的高维系统信息
  - 样本效率极高，泛化性强
  - autonomous data collection（"Chaos Box"——随机载荷，不需要人工干预）
- **理论**：通过 KL 散度的 data processing inequality 证明因式分解改善泛化
- **成果**：在 LEAP Hand 上实现了前所未有的物体多样性（长物体、复杂形状、小物体）
- **关键设计**：residual policy 而非直接 fine-tune

### 7. Fei 等 2025 - T(R,O) Grasp (Graph Diffusion for Cross-Embodiment Grasping)
- **方法**：T(R,O) Graph——将手-物交互表示为节点（物体 patch + 手 link）+ 边（相对 SE(3) 变换）
- **核心**：
  - 比 D(R,O) 内存效率高很多（patch-to-link 而非 point-to-point）
  - Graph diffusion model，支持无条件 + 有条件抓取生成
  - Lie group representation for SE(3)
- **应用**：静态抓取生成，support closed-loop
- **局限**：只做抓取，不做动态操作

## 讨论任务

基于以上背景，和用户讨论以下核心问题：

1. **Per-finger RMA 的合理性**
   - 标准 RMA（如 Hora）的 extrinsics 是全局的（一个 latent 代表整个物体）
   - DexNDM 的 joint-wise factorization 证明了"per-joint 信息足够预测自身下一状态"
   - 那 per-finger adaptation latent 是否合理？每根手指感受到的物体属性确实不同（局部曲率、摩擦、载荷）
   - 和全局 RMA 的对比：什么场景下 per-finger 比全局更好？

2. **图结构设计**
   - 用户想法：4 finger nodes + 1 object node
   - 可能的变体：finger nodes + object node + palm node + contact edge
   - 和 GET-Zero 的区别：GET-Zero 只编码 robot 图（joint 级别），不包含 object
   - 和 T(R,O) 的区别：T(R,O) 是 link-patch 级别，用于静态抓取
   - 这里是 finger 级别，用于动态策略

3. **方法的独特优势在哪？（导师的核心要求）**
   - 需要明确：这个方法在什么场景下比现有方法（全局 RMA、纯 DR、无图结构策略）更好？
   - 可能的场景：不对称物体、部分接触丢失、跨物体泛化、多轴旋转时不同手指的角色差异

4. **是否足以支撑一篇 paper？**
   - 需要评估：per-finger RMA + graph attention 是否只是"又一种策略网络设计"，还是有更深的洞见
   - 和 MAGCLA 的对比：MAGCLA 也是 per-finger 思路，但用 MARL 而非图

## 委派要求

- 使用 `askQuestion` 和用户保持循环反馈
- 长段分析写入 `/home/hac/isaac/AnyMani/source/anymani/ideas/graph/discusion.ipynb`（已创建，有初始内容）
- 用户是科研人员，不需要解释基础概念，但需要帮助理清方法定位
- 讨论应聚焦在"方法导向"的框架下：方法的核心优势 → 在什么新场景/新组合下发挥独特价值
- 不要让讨论发散到太多方向，帮用户收敛
- 在用户明确说"结束讨论"之前，持续通过 askQuestion 互动

## 第一轮讨论记录（已完成）

### 已收敛的共识
1. **核心方法**：Per-finger RMA + Graph Attention
2. **应用靶点**：跨形态（Cross-Embodiment）手内操作
3. **Paper Story**：将 RMA 因式分解到 finger 级别，通过图注意力动态路由。Transformer 天然支持变长 token，实现跨手型零样本迁移。
4. **训练范式初步共识**：单体 PPO + 图结构（而非 MARL），但 RL Expert → IL 蒸馏 vs 端到端 RL 仍有分歧
5. **前向传播设计草案**：N个 finger token + 1个 object token → Transformer Encoder → shared action head

### 悬留分歧
1. **Object 节点用什么表征？** Agent 建议纯 [CLS] 虚拟节点（无几何输入），用户倾向参考 TRO-Grasp 的有界几何编码
2. **训练范式**：RL Expert → IL 蒸馏（类似 GET-Zero）vs 端到端 RL 的可能性
3. **用户要求重新讨论两个方面**：(a) 图网络结构的具体设计 (b) 任务问题的选取

### 详细讨论纪要
见 `/home/hac/isaac/AnyMani/source/anymani/ideas/graph/discusion.ipynb` 的 Cell 1-7

## 第二轮讨论记录（已完成）

第二轮重新讨论了图网络结构设计和任务问题选取。

### 新收敛的共识
1. **Token 粒度**：从 per-finger 调整为 joint-level（更接近 GET-Zero）
2. **Adaptation 粒度与 Token 粒度解耦**：可以 joint-level token + per-finger adaptation latent 共存
3. **Object 表征**：non-[CLS] also non-patch-graph → per-finger local latent 聚合出 global memory
4. **Hand-generalization 核心机制**：dynamic joint stream × static embodiment stream 的 cross-attention（URDF/关节上下限/link 几何 → static tokens，关节状态/动作历史 → dynamic tokens）
5. **Local 主路 + Global 残差**：$a_j = \pi_{\text{local}}(o_j, z_{f(j)}, e_j^{\text{URDF}}) + \Delta\pi_{\text{global}}(o_j, z_g)$
6. **先单手型验证再抬多手型**：避免 object latent 被单手型绑死

### 悬留问题
1. **Object memory 是 1 个向量还是多个 memory tokens？**
2. **Per-finger latent 的监督目标**：local 预测什么？global 预测什么？（初步建议：local 预测本指交互如 contact/slip/force，global 预测物体 extrinsics）
3. **是否需要显式 finger summary layer 作为跨手型的对齐层？**
4. **训练路线是否需要 multi-hand × multi-object 的对齐阶段？**

### 新增的工程计划
见 `/home/hac/isaac/AnyMani/source/anymani/ideas/graph/plan.ipynb`（Phase 0-3 分阶段方案）

### 详细讨论纪要
见 `discusion.ipynb` 的 Cell 8-17

## 历史记录

| 轮次 | 说明 |
| --- | --- |
| 1 | 主代理盘点了所有环境，用户确认保留 Joint/Tactile/RoundTip/Direct，删除 SE3/Affine 及关联文件，彻底清理（含 MDP 组件、agent configs、floating_base_kinematic、skrl/rsl_rl configs）。 |
