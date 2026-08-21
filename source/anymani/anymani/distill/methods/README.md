# Embodiment Methods

直接把物理场、可学习编码器和损失公式散落在 trainer 里，会让采样日程、checkpoint 与科学对象互相污染。`methods` 把一份可训练的几何方法写成聚合根：对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口。当前唯一 concrete method 是多锚点条件 Gaussian 隐式场。

## 研究对象

记手型静态定义为 $\mathfrak m$，当前活动关节角为 $q\in\mathbb R^{N_J}$，单位 rad。方法要编码的不是某个固定 mesh 或 BPS 网格，而是 PALM/JOINT/TIP owner 的条件邻近场。对固定于手部语义坐标系 `{h}` 的 query $x$（单位 m）和实际带宽 $\sigma$（单位 m），teacher 给出 unsigned distance $d_g(x;q)$ 与无量纲密度

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

一阶监督先取距离灵敏度 $\kappa_{g,i}=\partial d_g/\partial q_i$，单位 m/rad，再由链式法则得到场灵敏度 $g_{\sigma,g,i}=-(d_g/\sigma^2)\rho_{\sigma,g}\kappa_{g,i}$，单位 rad$^{-1}$。retained encoder 只读当前物理 $q$ 与静态证据；query、sigma、最近点、Jacobian 和 teacher 只属于 SSL。$z_i^{(1)}$ 是整手场 Jacobian 第 $i$ 列的固定宽度表示，不是对自身 $z_i^{(0)}$ 求导。

## 方法内部耦合

```text
μ_q                完整 joint-limit 超矩形上的 scrambled Sobol
R                  物理 source / field / query / target
f_θ                GeometrySSLModel
L                  五项比较公式
A                  单 JOINT 符号改写
```

`GeometryRepresentation` 只交付未 padding 的物理 teacher。method 的 batch 适配层选择 $A^{(k)}$、构造 `StaticGeometryEvidence`，并把异构 $N_J/G/E$ 填进稠密容器。一次 batch 在逻辑上分成三块：模型输入、读出条件与物理真值。模型不得读取 truth。padding 上限由 resolved 资产的实际最大 JOINT/TIP 和 backbone 图距离推导；超出则失败。

每资产独立生成 8 套 physical anchor bank。同资产 q-block 内共享一套并均衡轮换；validation、independent q-bank 与 PPO 固定 $A^{(0)}$。home-surface 每 owner 64 个 boundary 点。query 保持 workspace/shell/adjacent = 50/25/25；一阶边只从 shell 抽。训练 sigma 中心为 4/16/64 mm，log-space ±10% jitter；validation 关闭 jitter，仍用同一组。每个有效 JOINT：train 1 条 active + 1 条 structure-zero，validation 为 4+4。

## 五项约束

主损失是 density、κ、derived-field、Sobolev 与 chain。paired parity 不是主损失。joint-sign rewrite 是输入增强：每个 $(asset,q)$ 以 0.20 概率恰好翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；density/distance 不变，对应 $\kappa/g$ 翻号。validation 另做双前向 parity audit。

归约按 $(asset,q)$ 等权。一阶 active/zero 先分别平均，再 1:1 合并，避免最近点 mask 丢掉全部 active 后被 zero 主导。Trainer 不解释 owner、query 或 edge 轴，也不读取 `method.representation`。

`calibrate_objectives` 与 `pretrain` 共用同一 façade。calibration 前向计算全部五项，不更新参数，也不自动改权重。pretrain 必须核对 artifact 的数据集 hash、公式身份、method/model/representation、采样语义和代码 revision；声明权重以当时的 `OBJECTIVES_CFG` 为准。

## 阅读边界

物理真值见 [`../representations/README.md`](../representations/README.md)，retained encoder 见 [`../models/README.md`](../models/README.md)，训练生命周期见 [`../ssl/README.md`](../ssl/README.md)。当前证据覆盖配置、归约、padding 与 calibration 身份；正式 8192-asset 训练与跨手型泛化尚未成立。
