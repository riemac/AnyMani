# Self-Supervised Geometry Pretraining

策略监督把几何、接触和任务目标纠缠在同一回报中，很难判断网络究竟学会了手的物理结构，还是只记住了某个任务与某种手型的统计捷径。Geometry SSL 因而先提出一个更窄的命题：不启动环境、不观察物体，只凭当前关节构型与静态手型证据，能否恢复整只手的局部空间占据及其对关节运动的一阶响应？

## 学习问题

每项手资产被表示为 PALM、JOINT 与 TIP owners 的有序集合。对 owner $g$、固定 hand-frame query $x$ 和当前构型 $q$，teacher 从真实碰撞表面计算 unsigned distance $d_g(x;q)$。训练场不是离散 occupancy class，而是由物理尺度 $\sigma$ 参数化的连续邻近函数：

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

其构型导数由距离灵敏度

$$
\kappa_{g,i}(x;q)=\frac{\partial d_g(x;q)}{\partial q_i}
$$

和链式法则共同决定：

$$
g_{\sigma,g,i}(x;q)
=\frac{\partial\rho_{\sigma,g}}{\partial q_i}
=-\frac{d_g}{\sigma^2}\rho_{\sigma,g}\kappa_{g,i}.
$$

$d$ 与 $\kappa$ 的单位分别为 m 与 m/rad，$\rho$ 无量纲，$g$ 的单位为 rad$^{-1}$。保留这些单位使模型能区分真实几何尺度；它不会把每只手独立归一化到相同大小。

## 在线构造监督

`GeometrySource` 先从资产 sidecar 的 typed semantics 构造 q-independent physical oracle：float64 POE 规格、owner ancestry、严格 collision union、boundary-only home points、palm-seed anchors、physical identity 与可显式释放的 Warp BVH。它不包含当前 q、query 或 label。

训练时，每项资产拥有独立 scrambled Sobol 序列。对每个 q realization，`GeometryRepresentation` 在线生成三种互补 query：一半来自固定 physical anchors 周围的 5 cm 工作空间球，四分之一来自 current owner surface 两侧 0.5--4 mm 壳层，四分之一来自运动学图相邻 owners 之间的局部间隙。三种来源只改变采样测度，不作为 decoder 输入。

每个 owner 使用 64 个 query。训练 sigma centers 为 4、16、64 mm，并作 log-space ±10% jitter；同一资产的两项 q 共享一次实际 sigma realization。validation 则固定使用 4、8、16、32、64 mm。Warp teacher 返回距离、closest-source provenance 与局部三角面 margin；$\kappa$ 只在 sampled owner-query-JOINT edges 上物化，非祖先边保留为精确结构零。UDF 在 $d\approx0$ 或最近源切换处不可微，这些位置通过显式 mask 排除一阶损失，但仍保留零阶 density 与诊断记录。

## 表示与读取器

retained encoder 的输入只有当前物理 q 与静态 evidence。多锚点前端把 home surface points 与空间旋量都表达成相对完整 anchor constellation 的关系，并对 anchor permutation 聚合；hand frame 绕 palm normal 的面内旋转被视为 $SO(2)$ gauge，而 reflection 被保留为物理差异。

这些 owner tokens 进入两层、四头、hidden width 128 的 encoder-only Transformer。attention 在全部有效 entities 间保持全连接，运动学无向最短路径、parent distance 与 child distance 仅形成每头可学习的加性 bias，不是 hard mask，也不输入 current all-pairs $SE(3)$ 答案。输出 heads 产生逐 owner 的 $Z^{(0)}\in\mathbb R^{128}$ 与逐 JOINT 的 $z_i^{(1)}\in\mathbb R^{64}$；后者由 owner contextual latent 与唯一的 residual screw carrier 共同形成。

训练期 density reader 对每个 `(owner, query, sigma)` 输出一个 scalar。query feature 与 $\log(\sigma/\sigma_{ref})$ 走主路径，owner latent 在三个 residual blocks 的每一层通过

$$
\widetilde h=(1+\gamma(z_g^{(0)}))\operatorname{LN}(h)+\beta(z_g^{(0)})
$$

调制；$\sigma_{ref}=16$ mm。sensitivity reader 先从 $z_g^{(0)}$ 与 query 产生偶 coefficient，再与 $z_i^{(1)}$ 作无偏置内积并除以 $\sqrt{D_1}$，从结构上保证 joint-sign rewrite 下输出严格为奇。两个 readers 都只服务 SSL，retained checkpoint 只保存 `encoder.` namespace。

## 为什么联合六项目标

仅重建 density 允许 decoder 绕开 morphology latent，或让 latent 对 q 的局部变化缺乏可用结构。当前目标因此同时约束：density 重建、显式 $\kappa$、由预测 density/$\kappa$ 派生的 $g$、同一个 density predictor 对物理 q 的 Sobolev/JVP、自导数与显式链式路径的一致性，以及 joint-axis sign 成对改写下 $Z^{(0)}$ 偶/$z_i^{(1)}$ 奇的 parity。

六项权重首先在 8 个固定 train microbatches 上按共享 encoder 梯度范数校准。校准结果是 runtime evidence，单独进入 `loss_calibration.yaml` 与 checkpoint metadata；它不会改写实验声明中的 objective。由此可以区分“研究者声明了哪些项”与“本次数值运行采用了什么校准权重”。

## 实验协议

canonical dataset 由 assets 层 manifest 冻结多个 mother lineages：45 train、16 validation、16 unseen-variant-set 与 17 unseen-mother assets。每个 epoch 每项训练资产消费 256 个新 q，共 20 epochs。2 assets × 2 q 构成一个 microbatch，累积 4 次后更新；resident windows 的尾组不伪造 padding asset，因此 45 项训练集对应每 epoch 2944 microbatches、736 updates，完整预算为 14720 updates 与 230400 q samples。30000 只是异常 safety limit。

validation bank 每项资产固定 64 q，每 250 updates 评估。checkpoint score 以 initialization density、$\kappa$ 与 derived-field 误差归一化。训练结束后固定执行 query-only、same-asset q shuffle、cross-asset shuffle、first-order zero、JOINT shuffle 与 sign flip，并对 asset/q 配对差异做 2000 次 bootstrap。evaluation suites 本 patch 只完成资产选择与 physical-identity 审计，不进入 checkpoint selection 或训练 forward。

整个实验由 schema 2.0.0 的 `CanonicalResidualFamilyCfg` 声明；Hydra 只注入 `single_gpu_16gb` trainer preset。`GeometrySSLExperiment` 构造阶段无 IO，`run()` 才拥有 materialization、training、validation、checkpoint 与 lease release。schema/checkpoint 1.x 被明确拒绝。

## 已有证据与尚未回答的问题

当前 contracts 已覆盖 source realization、query/sigma 重放、graph-bias 公式、全连接 attention、padding masks、$SO(2)$、joint-sign parity、FiLM 条件、variable sigma、sigma detach、checkpoint key 与跨结构输出/梯度；synthetic 和真实 LEAP integration 都闭合到 backward。

在 RTX 5070 Ti 上，canonical retained encoder 的 $B=4096$ p95 为 20.13 ms，满足 40 ms 子预算。这个测试从 GPU-resident q/static evidence 开始，刻意排除 teacher、decoder、policy 和环境，因此不能回答完整 PPO 是否达到 20 Hz。正式 20-epoch pilot 尚未启动；unseen-variant-set/unseen-mother 的模型评估、official zero-shot、Isaac pose parity 与 PPO transfer 也仍是开放问题。

## 复现入口

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain

# Hydra override 示例
python -m anymani.distill.ssl.pretrain trainer.learning_rate=1e-4 representation.query.query_count=128
```

输出目录保存 resolved config、asset manifest、calibration、分层 metrics、q-bank digest、ablation、selection history、完整 resume checkpoint 与 retained-only encoder state。具体统计语义见 [`../diagnostics/README.md`](../diagnostics/README.md)。
