# Embodiment Methods

直接把物理场、可学习编码器和损失公式散落在 trainer 里，会让采样日程、checkpoint 与科学对象互相污染。`methods` 把一份可训练的几何方法写成聚合根：对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口。当前主线是 Gaussian density + anchor-relational Material-point Jacobian；旧 density/κ method 保留为 v0.7.5 对照。

## 研究对象

记手型静态定义为 $\mathfrak m$，当前活动关节角为 $q\in\mathbb R^{N_J}$，单位 rad。方法要编码的不是某个固定 mesh 或 BPS 网格，而是 PALM/JOINT/TIP owner 的条件邻近场。对固定于手部语义坐标系 `{h}` 的 query $x$（单位 m）和实际带宽 $\sigma$（单位 m），teacher 给出 unsigned distance $d_g(x;q)$ 与无量纲密度

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

当前一阶监督固定 owner-local collision material identity，并把 material-point Jacobian 投影到 PALM anchor constellation 定义的 height、radius、dot、chirality 四类关系，记为 $\Gamma_{gmki}$，单位 rad$^{-1}$。Retained encoder 只读当前物理 $q$ 与静态证据；query、sigma、current material point、Jacobian、ancestor mask 和 teacher 都只属于 SSL。JOINT view 从统一 owner-token $Z$ gather，不产生独立一阶 latent 包。

## 方法内部耦合

```text
μ_q                完整 joint-limit 超矩形上的 scrambled Sobol
R                  物理 source / field / query / target
f_θ                DensityMaterialJacobianSSLModel
L                  rho/Gamma、teacher baseline 与固定通道尺度
A                  单 JOINT 符号改写
```

`GeometryRepresentation` 交付未 padding 的物理 teacher。method 的 batch 适配层选择 $A^{(k)}$、构造 `StaticGeometryEvidence`，并把异构 $N_J/G/E$ 填进稠密容器。一次 batch 在逻辑上分成模型输入、读出条件与物理真值：模型前向消费前两块，truth 进入 objective。padding 上限由 resolved 资产的实际最大 JOINT/TIP 和 backbone 图距离推导，输入越界时直接报告对应资产与维度。

每资产独立生成 8 套 physical anchor bank。同资产 q-block 内共享一套并均衡轮换；evaluation、independent q-bank 与 PPO 固定 $A^{(0)}$。home-surface 每 owner 64 个 boundary 点。query 保持 workspace/shell/adjacent = 50/25/25；一阶边只从 shell 抽。训练 sigma 中心为 4/16/64 mm，log-space ±10% jitter；evaluation 关闭 jitter，仍用同一组。每个有效 JOINT：train 2 条 active + 1 条 structure-zero，evaluation 为 4+4。

## Density + Gamma 双项约束

Density 使用完整 owner/query/sigma raw MSE；Gamma 四通道先除以固定全数据集尺度，再在每个 `(asset,q)` 内按 active/structural-zero 2:1 归约。Shared encoder 的两项梯度进入 FairGrad，private readers 各自只接收本任务梯度。Joint-sign rewrite 以 0.20 概率翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；density 保持不变，对应 selected Gamma column 变号。

归约按 $(asset,q)$ 等权。一阶 active/zero 先分别平均，再按 2:1 合并，使监督质量与每 joint 的采样预算一致。owner、query、material 和 edge 轴由 Method 解释，Trainer 只接收归约后的充分统计。

`pretrain` 使用同一 façade 和 `max_epochs / num_minibatches / mini_epochs / microbatch_size / sampling` 配置接口。Method 按完整 minibatch denominator 把 8 个 64-pair stream units 合成为一次精确更新；Trainer 不跨 minibatch 累积梯度。run-local baseline 在新 teacher unit 上累计且不运行 learned model；schema-9 checkpoint 核对数据集、公式、method、代码 lineage、precision 与 source identity。

Method session 封装 source、Sobol cursor、resident window 和具体 batch。固定 evaluation 测度也由 Method 执行：$A^{(0)}$、4/16/64 mm、关闭 jitter/rewrite、每 JOINT 4+4 edges。evaluation 只消费显式 checkpoint，不选择 checkpoint；训练完成时可从独立 train q-bank 发布压缩 basis。

## 阅读边界

物理真值见 [`../representations/README.md`](../representations/README.md)，retained encoder 见 [`../models/README.md`](../models/README.md)，训练生命周期见 [`../ssl/README.md`](../ssl/README.md)。N031 已完成 8192-asset、12-cycle 正式训练与两条 1024-asset × 64-q held-out suites；density/Gamma skill 为约 `72%/86%`，schema-5 encoder-only artifact 是后续 PPO transfer 的当前主线。
