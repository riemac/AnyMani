# Heterogeneous generated-hand task family

`hetero`固定表示：同一个object-manipulation任务下，并行实例化多种generated hand embodiments；这些手可以具有不同topology、active DoF、handedness、TIP数和几何参数。它不是multi-agent、多物体或通用multi-task命名空间。

本目录已建立独立Python package、generated canonical scene、good-pregrasp partial reset、structured observations/History30、contact state、fixed-axis command、reward、termination、preload-aware action与Gym注册。学习能力由`distill`侧固定评估判定；可运行、TIP contact或absolute speed本身不表示连续有向旋转。

## 任务族边界

- 首版只覆盖generated canonical assets；official LEAP/Allegro后续作为held-out或zero-shot suites，不进入首版训练支持域。
- 通用配置可消费generated partition；掌托旋转入口显式消费版本化母manifest或member-level canonical cohort lock。MVP80是历史默认支持集，实际规模由冻结集合决定；小规模运行检查与固定能力评价分别记录。
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
- $\mu,\sigma,V$：确定性动作中心、变换前高斯标准差与整手价值；有界随机动作的期望和标准差另由分布决定。

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

主位置通道固定为共同规范零构型下的`q/pi`、`u/pi`，运行时软限位单独传入；不能按每资产限位中点/宽度重定义并替换主角度通道。局部网络可另外使用限位内相对位置辅助量。`q_home=0`是坐标参考，预抓取回合初态通常非零且满足`q_0=u_0`。

Critic raw privileged input可包含PALM/JOINT/TIP、object与task blocks：

$$
O_t^c=\left(O_{t,\mathrm{palm}}^c,O_{t,\mathrm{jnt}}^c,O_{t,\mathrm{tip}}^c,O_{t,\mathrm{obj}}^c,O_{t,\mathrm{task}}^c\right).
$$

MVP观察producer保留全owner binary对照，RL训练主入口显式选择TIP-only：同时清除当前关节帧、History30与独立owner token的非TIP触觉，保留五通道存储、本体状态、TIP接触和实体mask。All-link force、object state、goal/anchor error、actuator state与actual ADR state只进入critic/diagnostics或显式oracle actor ablation。全触觉checkpoint的冻结遮蔽与TIP-only重新训练分别评价。Morphology cell ID只服务分层诊断，不作为主actor/critic输入。

## Variable-cardinality与canonical transport

论文与任务语义在真实集合$\mathcal J_{\mathfrak m}$和$\mathcal E_{\mathfrak m}$上定义。Canonical-v1实现当前padding到16 JOINT、4 TIP和21 physical owners，并通过validity masks表示逻辑可变数量。Ghost只属于storage ABI；不得进入attention有效集合、动作执行、Normal log-prob、entropy、KL、bounds、reward或统计分母。

Asset row只允许作为evidence/cache lookup certificate，不作为连续policy feature。Runtime active mask必须与canonical evidence和physical identity一致；不一致时fail closed。

## Action与actor语义

ManagerBasedRLEnv action transport可以是`[B,16]`，但逻辑动作空间只覆盖有效关节：

$$
\mathcal A_{\mathfrak m}=\prod_{j\in\mathcal J_{\mathfrak m}}[-1,1].
$$

掌托旋转可用逐JOINT TCN压缩History30，或把30×5固定lag交给local MLP；低维控制状态先编码，冻结$Z_j^e$通过有界FiLM调制。Residual分支使用基础动作加0.2残差；Direct与direct_token分别从上下文加局部旁路、纯上下文产生完整动作中心。分支由运行身份明确指定，不能用Residual默认描述所有模型。当前共享潜高斯标准差由一个可学习标量表示，不建立16个absolute-slot variance参数。

## Critic语义

Critic使用与actor完全分参的两层structured graph backbone，每个environment输出一个hand-level scalar value。主线不使用flat MLP或8-cell one-hot作为形态表示。Critic通过有效tokens、冻结geometry、graph、privileged object/contact/task state和显式mask-aware PALM/mean/max readout理解形态。

## Pregrasp与reset门禁

掌托旋转MVP provider必须按physical geometry hash、cube identity/hash、exact scale、physics identity和generation identity查询schema-3 Top-8 good-pregrasp catalog；无覆盖候选时fail closed。Dataset row不是cache identity。

Good pregrasp表示hand-object-scale耦合的cold-reset安全准备态。PALM/JOINT/TIP contact均是metadata，TIP数量不定义查询tier；硬门关注联合指间包络、joint reserve、穿透、位移/倾斜、速度峰值、palm support及训练同路径1 s hold。MVP固定消费rank-0并要求$q_0=u_0$、object upright和零速度。

cohort可通过`selection.pregrasp_generation_identity`声明旧Top-8候选的独立重验协议；binding必须验证physics与strict gate摘要保持，再构造新generation key。默认未声明时继续使用原Sobol/CEM。重验必须真实执行相同物理门，不能把旧entry改key；候选、投影与parent-rank顺序由`pregrasp.revalidation`合同管理，不进入Actor特征。

## ADR边界

首版可以建立mask-aware actual ADR state、diagnostics以及global/group/asset/env scope接口，但第一个科学baseline关闭ADR。只有固定easy-tier能力与分层证据建立后，才引入group、asset residual、per-env或hierarchical scheduler。ADR状态必须记录实际采样值、scope、level、升降级事件与固定tier评估，不把curriculum变化误写成策略进步。

逐关节stable reward以N000的16-DoF数值为参考：sum类项使用$\frac{16}{n_i}\sum_{j\in\mathcal J_i}$，pose $L_2$使用$\sqrt{\frac{16}{n_i}\sum_j(\cdot)^2}$。该归约在16-DoF资产上逐值恢复N000，在少DoF资产上保持相同per-joint量级；ghost不进入分子或分母。

## 迁移与出清

新`hetero`任务完成contract、runtime、pregrasp与短训练验证后已执行：

- `tasks/gm`只保留single-asset/LEAP共享原件，不再新增same-topology或canonical multi-asset配置；
- 不重新引入旧GM heterogeneous Gym IDs或deprecated alias；
- 历史checkpoint使用其原始commit复现；
- 只移动明确属于heterogeneous canonical ABI的实现。被`tasks/inhand`或single-asset GM消费的contact/action/event原件不得粗暴搬走。

## 测试优先级

1. 符号对应的structured shapes、units、frame和role axes；
2. owner/joint permutation、ghost mask、action probability与scalar critic invariance；
3. physical-hash pregrasp cache、scale interval和fail-closed lookup；
4. left/right semantic frame与action-sign counterfactual；
5. 2/16/80资产Isaac reset/step/contact/checkpoint smokes；
6. 正式2048资产capacity与固定逐资产evaluation；
7. ADR或TOPPO组件只在baseline证据之后进入。

任何改变reset分布、success、rotation axis、action authority、canonical schema或critic privilege的修改，都必须同时更新数学合同、任务配置、证伪测试和Research实验记录。
