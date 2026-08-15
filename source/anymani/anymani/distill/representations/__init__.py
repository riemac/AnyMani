r"""跨手型训练所需的物理表征真值层。

本包回答“要让网络保真什么物理对象”，不拥有任何可学习网络。当前研究对象是：给定
hand semantic frame ``{h}``、关节构型 $q$、有序 screw chain 与 home collision
geometry，构造 PALM / JOINT / TIP physical entity 在当前构型下的真实 posed geometry。
第 $g$ 个实体同时是第 $g$ 个碰撞表面归属体；实体索引、归属轴与 SSL 解码轴同索引。
首版监督由逐归属体 unsigned distance $d_g$ 派生多尺度 $\rho_{\sigma,g}$、距离灵敏度
$\kappa_g=\nabla_qd_g$ 与场灵敏度
$g_{\sigma,g}=-(d_g/\sigma^2)\rho_{\sigma,g}\kappa_g$。

候选分解保持正交：

```text
collision / kinematic source
    -> physical field definition
        -> fixed BPS or sampled spatial queries
            -> candidate-neutral target batch
```

因此 posed BPS、continuous density field 与 conditional implicit field 不是互斥的
单体实现。posed BPS 是 ``distance field + fixed ordered queries``；conditional
implicit field 是 ``任意 field + sampled queries + disposable query decoder``；
parametric Gaussian field 则让 decoder 输出有限个 Gaussian components，再在 query
space 中比较它们诱导出的物理场。

边界：

- ``representations`` 只拥有物理 source、field、query 与 target，不 import
  ``torch.nn``，不持有 policy checkpoint；
- 所有 learnable input adapter、backbone 与 decoder 位于 ``distill.models``；
- loss 位于 ``distill.objectives``；SSL / RL / IL 只负责阶段编排；
- 本包不得 import、解析或要求 ``Research`` vault 存在；Research 只能下游引用代码证据；
- pretraining-only target generator 可以较重；PPO 保留的是 adapter + backbone +
  $z^{(0)}/z^{(1)}$ heads。当前自动性能门槛只覆盖 RTX 5070 Ti、$B=4096$、单结构组的
  GPU-resident $X\rightarrow(Z^{(0)},Z^{(1)})$，50 次计时 p95 不超过 40 ms。
"""
