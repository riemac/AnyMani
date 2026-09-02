# Heterogeneous generated-hand task family

`hetero`固定表示：同一个object-manipulation任务下，并行实例化多种generated hand embodiments；这些手可以具有不同topology、active DoF、handedness、TIP数和几何参数。它不是multi-agent、多物体或通用multi-task命名空间。

本目录已建立独立Python package、pregrasp partial-reset sidecar、fail-closed reset event与preload-aware action term；尚未装配scene、完整MDP或Gym注册，不得把局部runtime合同解释为任务已经可训练。

## 任务族边界

- 首版只覆盖generated canonical assets；official LEAP/Allegro后续作为held-out或zero-shot suites，不进入首版训练支持域。
- 正式配置默认消费2048-asset generated partition。16/128资产只用于contract、physics、pregrasp、runtime和短训练门禁；具体canary规模可按证据调整。
- `hetero`独立装配ManagerBasedRLEnv的scene、observation、action、command、reward、reset、termination、ADR state和diagnostics。
- 不import`anymani.tasks.gm`或`anymani.tasks.inhand`。N000 single-asset command/reward/ADR实现只能作为经过验证的科学参照；复制或适配后必须重新通过heterogeneous mask、frame、partial-reset和variable-cardinality contracts。
- `assets`负责dataset/physical identity与typed geometry semantics；`robots`负责canonical articulation lowering；`pregrasp`负责自动搜索、cache与identity；`tasks/hetero`只消费这些接口并定义任务MDP。
- 网络、TCN、attention backbone、actor/critic heads与PPO算法不进入本目录，继续属于`distill/models`和`distill/rl`。

## 符号语法

数据阶段固定使用：

- $O$：某网络域在learned transform前收到的raw structured input；
- $Z$：局部encoder产生的latent representation；
- $X$：input adapter融合后送入跨实体backbone的tokens；
- $H$：跨实体backbone输出的contextual hidden tokens；
- $\mu,\sigma,V$：actor mean、actor standard deviation与critic scalar value。

一级上标固定为$e/a/c$，分别表示geometry encoder、actor和critic；可选第二级上标使用完整语义词，例如$Z_{t,\mathrm{jnt}}^{a,\mathrm{hist}}$。实体/token role放在下标，使用`palm`、`jnt`、`tip`、`obj`和`task`，不把所有分类维度堆入上标。

参数层级为：

$$
\theta^e=(\theta^{eb},\theta^{ed},\theta^{eg}),\qquad \theta^a=(\theta^{ai},\theta^{at},\theta^{ab},\theta^{ah},\theta^{av}),\qquad \theta^c=(\theta^{ci},\theta^{cb},\theta^{ch}).
$$

其中$\theta_\star^{eb}$是RL消费的冻结geometry/FK backbone；actor与critic完全分参，且冻结geometry不属于两者。完整数学约定的Research权威记录位于`Research/总体/rl/异构策略符号与数据流.md`；源码不能依赖或解析Research vault，本文保留实现所需的自包含摘要。

## Structured observation contract

Actor raw observation由`tasks/hetero`交付：

$$
O_t^a=\left(O_{t,\mathrm{palm}}^a,O_{t,\mathrm{jnt}}^a,O_{t,\mathrm{tip}}^a\right).
$$

Policy observation terms必须保留joint/history/role/mask axes；task层不拥有`1969D`等手工flat切片知识。ManagerBasedRLEnv可以为后端使用固定padding tensor，但flatten只能发生在明确的training adapter/transport边界。

History30 raw window属于task observation。Trainable TCN/stack encoder输出$Z_{t,\mathrm{jnt}}^{a,\mathrm{hist}}$，属于`distill/models`；不得把trainable latent伪装成MDP observation term。

Critic raw privileged input可包含PALM/JOINT/TIP、object与task blocks：

$$
O_t^c=\left(O_{t,\mathrm{palm}}^c,O_{t,\mathrm{jnt}}^c,O_{t,\mathrm{tip}}^c,O_{t,\mathrm{obj}}^c,O_{t,\mathrm{task}}^c\right).
$$

All-link contact、object state、goal/anchor error、actuator state与actual ADR state只进入critic/diagnostics或显式oracle actor ablation。Morphology cell ID只服务分层诊断，不作为主actor/critic输入。

## Variable-cardinality与canonical transport

论文与任务语义在真实集合$\mathcal J_{\mathfrak m}$和$\mathcal E_{\mathfrak m}$上定义。Canonical-v1实现当前padding到16 JOINT、4 TIP和21 physical owners，并通过validity masks表示逻辑可变数量。Ghost只属于storage ABI；不得进入attention有效集合、动作执行、Normal log-prob、entropy、KL、bounds、reward或统计分母。

Asset row只允许作为evidence/cache lookup certificate，不作为连续policy feature。Runtime active mask必须与canonical evidence和physical identity一致；不一致时fail closed。

## Action与actor语义

ManagerBasedRLEnv action transport可以是`[B,16]`，但逻辑动作空间只覆盖有效关节：

$$
\mathcal A_{\mathfrak m}=\prod_{j\in\mathcal J_{\mathfrak m}}[-1,1].
$$

Shared per-joint head读取整手attention后的contextual joint token；它不是独立joint actor。首个科学基线使用一个global shared log-standard-deviation$\theta^{av}$。若后续比较state-dependent exploration，只允许shared contextual variance head，不建立16个absolute-slot variance参数。

## Critic语义

Critic使用与actor完全分参的structured token backbone，每个environment输出一个hand-level scalar value。主线不使用127D flat MLP或8-cell one-hot作为最终形态表示。Critic通过有效tokens、冻结geometry、graph和privileged object/contact state理解形态；TASK token readout与显式mask-aware pooling尚待后续合意，不能在实现中静默选择。

## Pregrasp与reset门禁

Pregrasp provider必须按physical geometry hash、cube identity/hash、scale interval、support mode、physics identity和search identity查询认证cache；无覆盖候选时fail closed。Dataset row JSON不是cache identity。

Palm support合法，但support-only reset与contact-basin pregrasp是不同等级。正式contact basin必须保留至少两个TIP的硬门、限制finger non-tip，并记录drop、penetration、drift/twist、joint margin、effort、scale stress和局部扰动成功率。不得通过把TIP threshold降为0或把non-tip上限放宽为1来宣布门禁通过。

## ADR边界

首版可以建立mask-aware actual ADR state、diagnostics以及global/group/asset/env scope接口，但第一个科学baseline关闭ADR。只有固定easy-tier能力与分层证据建立后，才引入group、asset residual、per-env或hierarchical scheduler。ADR状态必须记录实际采样值、scope、level、升降级事件与固定tier评估，不把curriculum变化误写成策略进步。

## 迁移与出清

新`hetero`任务通过contract、runtime、pregrasp与短训练验证后：

- 出清`tasks/gm`中的same-topology multi-asset、canonical unified、heterogeneous config/test和canonical-only MDP残留；
- 删除旧`AnyMani-GM-Heterogeneous*` Gym IDs，不保留deprecated alias；
- 历史checkpoint使用其原始commit复现；
- 只移动明确属于heterogeneous canonical ABI的实现。被`tasks/inhand`或single-asset GM消费的contact/action/event原件不得粗暴搬走。

## 测试优先级

1. 符号对应的structured shapes、units、frame和role axes；
2. owner/joint permutation、ghost mask、action probability与scalar critic invariance；
3. physical-hash pregrasp cache、scale interval和fail-closed lookup；
4. left/right semantic frame与action-sign counterfactual；
5. 2/16/128资产Isaac reset/step/contact/checkpoint smokes；
6. 正式2048资产capacity与固定逐资产evaluation；
7. ADR或TOPPO组件只在baseline证据之后进入。

任何改变reset分布、success、rotation axis、action authority、canonical schema或critic privilege的修改，都必须同时更新数学合同、任务配置、证伪测试和Research实验记录。
