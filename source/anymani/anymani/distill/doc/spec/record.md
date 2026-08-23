# 多锚点 Gaussian 隐式场 SSL 审计记录

记录日期：2026-08-20。

## 0. 本记录的用途与证据地位

本文用于在会话 compact 前完整保存本轮关于大规模 task-free robot pretrain、多锚点 Gaussian 隐式场、采样生命周期、场 Jacobian、一阶表征、method 组织和 objective 权重标定的讨论上下文。它是后续继续研讨和制定实施计划的恢复记录，不自动取代 `Research/总体/ssl/当前设计/` 中的 canonical 科研合同，也不表示所有未决项已经获得实现授权。

本轮用户明确的工作优先级是：第一，科研语义、物理对象、采样测度和 lifecycle；第二，项目中 `method -> representation / model / objectives` 的结构组织；第三，才是 resident window、日志、Hydra、registry、checkpoint 接线等 infra 工程。`source/anymani/anymani/distill/doc/spec/question.md` 作为旧重构遗留的工程问题清单读取，但不能反过来主导科研路线。

2026-08-21 后续实现已完成本文所述结构收口：根配置改为四角色，Method session 封装 source/sampler/model-specific evaluation/checkpoint，Trainer 显式统筹 calibration/pretrain/validation/final evaluation。本文早期段落仍保留讨论形成过程；当前执行事实以本节更新、源码和 tests 为准。

## 1. 当前 Git 与资产配置事实

### 1.1 已完成的资产数据集配置收回

资产数据集构建已经结束，正式 SSL manifest 为：

```text
source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml
```

其原始 YAML bytes SHA-256 为：

```text
f1398417888e7c237cbb2583dcf8e9cd10bef7fee792b307c67dfa74fb6e0698
```

资产配置已从二层 Hydra data YAML 收回最高实验 façade。当前语义提交为：

```text
742fbf7 refactor(ssl): own asset config in experiment facade
```

当前 `source/anymani/anymani/distill/ssl/experiments/multi_anchor_gaussion_implicit_field.py` 逐项声明 `DATA_CFG / METHOD_CFG / TRAINER_CFG / RUN_CFG`；Hydra 只从 ConfigStore 读取这一完整 Python 实验，不再组合 distill YAML groups。

原 `distill/presets/ssl/data/hand_asset_catalog.yaml` 与该 data group 的 `__init__.py` 已删除；canonical Hydra root 改为从已含 `data` 的 Python experiment schema 开始组合。正式 `ssl.yaml` 已纳入上述提交，否则干净 checkout 会产生悬空配置。没有做 VERSION 或 CHANGELOG bump。

用户希望最终完整实验配置保持下列风格，而不是把科研数值分散到很多小 YAML：

```python
DATA_CFG = HandAssetCatalogCfg(...)
REPRESENTATION_CFG = GeometryRepresentationCfg(...)
MODEL_CFG = GeometrySSLModelCfg(...)
OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(...)
METHOD_CFG = MultiAnchorGaussianMethodCfg(
    representation=REPRESENTATION_CFG,
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    ...,
)
EXPERIMENT = EmbodimentPretrainCfg(
    data=DATA_CFG,
    method=METHOD_CFG,
    trainer=TRAINER_CFG,
    run=RUN_CFG,
)
```

当前确定使用 frozen dataclass + 单一 Python experiment 主配置；后续只有复杂度实际上升时，才考虑迁移为一份完整 YAML，不恢复分片 presets。

### 1.2 正式数据集规模已改变旧训练预算前提

新数据集不再是 Research 文档和旧 canonical preset 中的 45 train assets。`build_report.yaml` 与 template 给出的正式规模为：

| Partition | Mother 数 | 每 lineage 资产数 | 最终资产数 |
| --- | ---: | ---: | ---: |
| train | 512 | 16 | 8192 |
| validation.unseen_variant_set | 64 | 8 | 512 |
| validation.unseen_mother | 64 | 8 | 512 |
| evaluation.unseen_variant_set | 64 | 16 | 1024 |
| evaluation.unseen_mother | 64 | 16 | 1024 |
| official_zero_shot | 0 | 0 | 0 |

正式数据集合计 11264 assets，其中 10624 是新 post-mutate variants，其余为纳入 manifest 的 mothers。train cohort 覆盖 4 个 macro family、full/missing topology 与 7 至 16 DOF。

旧配置若继续使用 `256 q/asset/epoch x 20 epochs`，会产生 41,943,040 train q realizations、10,485,760 minibatches 和约 2,621,440 optimizer updates，并被当前 `run_safety_step_limit=30000` 在启动前拒绝。因此正式训练预算必须按 8192 assets 重新讨论，不能机械复用 45-asset 数值。该预算问题尚未在本轮最终拍板。

## 2. 已恢复的科研文档上下文

本轮按 `Research/总体/ssl/AGENTS.md` 的来源优先级，只读了以下 current 文档：

- `SSL 特权几何预训练索引.md`；
- `当前设计/基础合同/问题定义与研究边界.md`；
- `当前设计/基础合同/规范变换与物理不变量.md`；
- `当前设计/基础合同/训练—部署信息边界.md`；
- `当前设计/几何表征/全手物理实体与锚点关系.md`；
- `当前设计/几何表征/零阶与一阶几何包.md`；
- `当前设计/几何表征/两条几何编码路线.md`；
- `当前设计/特权预训练/条件隐式场与连续密度.md`；
- `当前设计/特权预训练/场灵敏度与一阶监督.md`；
- `当前设计/特权预训练/训练样本与目标生成.md`；
- `当前设计/特权预训练/在线采样与参数更新.md`；
- `当前设计/特权预训练/网络、损失与模块生命周期.md`；
- `实现脚手架/特权预训练实现地图.md`。

Research 仓库自身存在用户未提交改动，本轮没有修改这些笔记。archive 默认不是当前事实源，本轮没有用 archive 覆盖 current 文档。

当前稳定研究对象仍是：整手 PALM/JOINT/TIP collision-surface owners；隐式 retained encoder 只读取静态手型证据和当前 q；current mesh、distance、closest point、Jacobian、query、sigma 和 field targets 只属于 SSL；输出为 owner/entity 轴的零阶表征 `Z^(0)` 与逐 JOINT 的一阶表征 `z_i^(1)`；PPO 前删除 query/teacher/decoder/JVP graph。

当前正式路线仍同时保留 multi-anchor conditional implicit geometry 与 analytic direct geometry compression，不能因隐式路线方法更漂亮就预设其下游性能更好。本轮只审计当前 multi-anchor Gaussian implicit method，不实现 analytic direct 对照。

## 3. Method 的当前边界与项目组织目标

最高实验 façade 是 schema 4 `EmbodimentPretrainCfg`，公开组合 `data / method / trainer / run`。representation、model、objectives、固定评估测度、ablation 与 retained artifact 均由 concrete method 建立和封装。

本轮形成的目标边界是：

- `representation` 定义部署可见的物理输入 realization，以及 SSL-only query、sigma、edge 和 privileged target realization；
- `model` 定义 retained encoder、zero/first-order outputs 和 SSL-only density/sensitivity decoders；
- `objectives` 定义 prediction 与 target 的比较、mask、reduction 和诊断；
- `method` 组合三者，拥有 shared derived-field、Sobolev/JVP、joint-sign rewrite 等跨模块计算图语义，并执行兼容性检查；
- `trainer` 负责 phase epochs、minibatch/accumulation、反向、optimizer、validation promotion、final-evaluation 编排与通用 checkpoint，不知道 Gaussian field、owner/query axis 或 edge sampling；
- `run` 负责执行阶段、输出、resume 和 artifact lineage。

当前 lifecycle 只通过 Method/session 窄接口访问 split 数量、opaque batch、五项 update、固定评估报告与 state/artifact；具体 GeometrySSL 模型、padding batch、sources、Sobol sampler、sigma 和 ablation 已移出 Trainer/checkpoint。

用户认可未来在出现第二个真实 concrete method 后，再从共同外部行为中提取窄的 task-free robot pretrain method Protocol；当前不应提前建立万能 registry、`Any + MISSING` parser 或字段很多的抽象基类。

## 4. 当前代码中的逐量 lifecycle 事实

本轮直接追踪了 `representations/sources/geometry_source.py`、`collision_geometry.py`、`queries/spatial_sampling.py`、`representations/geometry.py`、`targets/geometry_field.py`、`ssl/runtime/lifecycle.py` 和 `methods/multi_anchor_gaussian.py`。当前真实行为如下：

| 量 | 当前代码行为 |
| --- | --- |
| owner union surface/solid | 每资产在 CPU source materialization 时构造一次，整个实验固定 |
| home-surface samples | 每资产按 owner 面积 proposal + FPS 生成一次，整个实验固定 |
| physical anchors | 每资产按 mount seed 邻域 surface/interior proposal + radial rejection + FPS 生成一套，整个实验固定 |
| GPU triangle sampling table | 资产进入 resident window 时上传，驱逐时释放；不改变物理 realization |
| Warp owner BVH | 资产进入 resident window 时取得 lease，驱逐时释放；不改变物理 realization |
| q | 每资产独立 scrambled Sobol 序列；cursor 跨 q round 和 epoch 连续，不重置 |
| 同资产构型块 | 当前一个 minibatch 中同一 asset 的 `q_per_asset_per_minibatch=2` 个 q |
| workspace query | 每同资产构型块重采一次；块内全部 q 和全部 owners 共享同一 `{h}` query realization |
| owner-shell query | 每个 q、每个 owner 独立重采 current surface face/barycentric/offset |
| owner-adjacent query | 每个 q、每个 owner 独立重采一跳 graph neighbor 与 surface candidate pair |
| sigma | 每同资产构型块重采一次；块内 q 共享实际 sigma realization |
| sensitivity edge selectors | 每同资产构型块重采；当前 owner/query-slot/joint selector 在块内 q 间共享，但 slot 对应的 shell 物理点随 q 独立 |
| train teacher | 每个新 q block 在线生成 distance/density/closest point/kappa/g；不落盘为离线 target dataset |
| validation bank | 正式训练开始前生成一次，q/query/sigma/teacher 冻结并在训练全程复用 |
| independent train-morphology q bank | 使用固定 seed 单独生成 initial/final evidence，不参与 optimizer sampling cursor |

具体专题 `训练样本与目标生成.md`、`条件隐式场与连续密度.md` 与当前代码在大部分 lifecycle 上一致；较上层的 `在线采样与参数更新.md` 仍把若干刷新周期写成“尚未冻结”，因此后续应以本轮确认结果更新 canonical 文档，消除层级冲突。

## 5. 已确认的采样与生命周期合意

### 5.1 Physical anchor bank

原实现每资产只有一套固定 anchors。本轮确认改为有限 anchor bank：每个 asset 在训练启动/materialization 阶段确定性生成可配置数量的 anchor constellations，首轮 preset 为 8 套：

$$
\mathcal B_A(\mathfrak m)=\left\{A_{\mathfrak m}^{(0)},A_{\mathfrak m}^{(1)},\ldots,A_{\mathfrak m}^{(7)}\right\}.
$$

每套仍使用当前已确认的生产测度：每 finger 10 anchors、5 cm seed 支持球、2.5 cm 径向衰减尺度、surface/interior 50/50、确定性 farthest-point/minimum-spacing selection。数字 8 是首轮可配置 preset，不是普适算法常数。

训练时以同资产构型块为单位在 8 套 bank 中均衡轮换；块内 2 个 q 使用同一套 anchors。选择由 asset identity、q-block cursor 与配置 seed 确定性导出，resume 不增加隐藏 RNG 状态。validation、independent q-bank 和 PPO 固定使用 canonical `A^(0)`。训练时不在每个 minibatch 重新执行昂贵的 Trimesh surface/interior sampling 和 FPS。

Anchor bank 是 retained-input realization augmentation：它要求编码器对同一 mount-conditioned support distribution 的有限 Monte-Carlo realization 稳健，同时保留不同资产的 mount layout、尺度与 chirality。它不是 target augmentation。

用户明确拒绝 left/right mirror pair 的逐套 anchor 镜像耦合要求。每个资产独立生成自己的 8 套 anchors，不为 mirror pair 建立额外采样对应关系。本轮只读 probe 曾发现相同 seed 下 left/right anchor 点集并不严格镜像，但该现象不再作为实施阻断项。

### 5.2 Home-surface samples

每资产、每 owner 的 retained home-surface samples 保持固定 64 点，当前 area proposal + oversample factor 8 + FPS realization 在整个 SSL/PPO 生命周期内不重采。用户判断 64 点足以表达 owner 的独特性，训练时重采会给 retained identity 增加不必要 variance。

不同 sampling seed、remeshing 和 point density 的稳健性只作为独立压力测试，不进入正式训练热路径。Anchor bank 轮换时，同一固定 home point 相对当前 `A^(k)` 的关系会自然变化，无需重新采 mesh。

### 5.3 q 的物理测度

首轮保持当前每资产完整 joint-limit hyperrectangle 上的连续 scrambled Sobol，不做在线自碰撞检测、不建立 feasible/boundary q bank，也不加入 home-local 混合分支：

$$
q\sim\operatorname{Sobol}\left(\prod_{i=1}^{N_J}[q_i^{min},q_i^{max}]\right).
$$

讨论中一度提出“无碰撞可行域 + 首次碰撞边界”或 global/local 1:1，但用户从 modern robotics 的 PoE 视角指出：owner 的当前 mesh state 只由该 owner 的祖先 screw、home geometry 和对应 q 决定；cross-owner 自碰撞不会使逐 owner forward geometry、distance 或 Jacobian target 变成数学错误。Per-joint/owner task-free self-geometry pretrain 与后续任务可达分布不是同一问题。

最终判断是：自碰撞 q 可能与下游访问分布不一致，但不是非法 geometry sample；引入任意 q mesh collision checker 会把首轮方法变成高成本几何规划问题。当前资产 validator 也只证明 home-pose sampled inter-finger clearance，sidecar 中 `all_pose_collision_free` 实际列在 `not_certified`，不能冒充全构型无碰撞证明。首轮不让碰撞检测阻塞预训练；以后若 pilot 显示预算效率或迁移有问题，再做小规模 q collision distribution audit。

### 5.4 Query mixture 与刷新周期

首轮维持每 owner `N_Q=64` 和 workspace/shell/adjacent = 50%/25%/25%，即当前 preset 的 32/16/16。配置层表达比例和总数，具体整数由严格校验得到，不把 32/16/16 写成不可替换算法常数。

Workspace 32 个 query 全部继续按同资产构型块重采；块内两个 q 和全部 owners 共享同一组绝对 `{h}` workspace queries，因此同一次 model batch 已经提供“同一空间点面对不同 q”的低方差比较。曾讨论 16 fixed + 16 resampled，最终不采用。Validation bank 单独固定全部 query realization。

Owner-shell 16 个 query 继续对每个 q、每个 owner 从 current area measure 独立重采，并使用 `[0.5,4]` mm 内外法向 offset。Shell 的采样分布随 owner 共动，因此它只承担 boundary profile 与局部一阶速度运动学监督，不独自证明零阶 latent 的 q-sensitive posedness。

Owner-adjacent 16 个 query 继续使用当前 owner graph one-hop neighbor 和 current surface candidate interpolation。讨论过缺少 thumb-index/cross-finger 纯空间邻近，但用户强调当前是 task-free self-geometry pretrain，不应提前把灵巧抓取先验硬编码进 query measure。Cross-owner 关系可由逐 owner density、whole-hand `Z^(0)` 和后续 PPO 空间读取学习；首轮不增加跨 finger nearest-neighbor query 类型。

### 5.5 Sigma 测度

训练继续使用 4/16/64 mm 三个中心，并对每个中心做 log-space 正负 10% jitter；同资产构型块共享实际 sigma realization，下一 block 重采。Sigma 仍作为 decoder 显式连续输入，不是固定输出 channel identity。

Validation 改为关闭 jitter 后只使用固定 4/16/64 mm，与训练中心测度一致。当前代码和测试中的 4/8/16/32/64 mm validation grid 没有得到用户批准，8/32 mm 位于训练 jitter 支持区间之外，不应参与 checkpoint selection。若以后需要 sigma interpolation probe，应作为独立诊断，不混进主 validation score。

### 5.6 Joint-sign rewrite augmentation

逐 JOINT coordinate sign 是同一物理机构的坐标规范。首轮不再把 paired parity 作为每个 minibatch 必算的第六个 latent MSE；改为 method-specific `JointSignRewriteCfg` 控制的训练输入 augmentation。

每个 `(asset,q)` realization 先以可配置概率决定是否 rewrite，首轮 `probability=0.20`。若选中，恰好改写一个有效 JOINT；不使用“每个 joint 独立 20% Bernoulli”，因为那会使高 DOF 手更容易同时翻转多个关节。JOINT 选择采用可配置、可恢复的 balanced cycle，按 asset identity、q cursor 和 seed 均衡覆盖。

对被选中的 JOINT `i` 执行：

$$
q_i'=-q_i,\qquad q_i^{home\prime}=-q_i^{home},\qquad \mathcal S_i'=-\mathcal S_i.
$$

Rewrite 后样本仍只做一次正常主 forward，并计算与普通样本相同的 density、kappa、derived-field、Sobolev 和 chain losses。物理 surface/query/closest point 不变，因此 density/distance target 不变；对应 JOINT 的 kappa/g target 随坐标改写翻号；其他 JOINT target 不变。若未来 rewrite 扩展到 q sampler、PPO control 或含 limits/action label 的边界，必须同步变换 `[q_min,q_max] -> [-q_max,-q_min]` 与动作坐标。

`JointSignRewriteCfg` 是 `MultiAnchorGaussianMethodCfg` 的方法专属配置，不是所有 task-free methods 的强制标配。固定 validation bank 仍需做双前向 parity audit，检查实际 zero-order even、first-order odd 数值误差。

用户原则上选择“结构保证 + rewrite 训练”，但对具体 even-axis / odd-carrier 前端拆分方案尚未理解和最终批准。当前网络把 even/odd screw 标量混入带 bias 的 MLP、GELU、attention 和普通 first-order MLP，奇偶性仍主要依赖学习。后续必须继续讨论严格结构的实际网络形式，不能把此前提出的 even/odd 拆分误记为已经锁定实现。

## 6. 从经典速度运动学到场 Jacobian

### 6.1 早期手写笔记的正确性

用户提供了一张早期手写推导图。该图不是纯 infra 参考，而是当前一阶表征科研解释的重要来源。图中主推导与当前代码符号一致。

设 `y_g^*(x,q)` 是固定 query `x` 到 owner `g` 当前碰撞表面的唯一最近点，Gaussian 邻近场为：

$$
\rho_{\sigma,g}(x;q)=\exp\left(-\frac{\|x-y_g^*(x,q)\|_2^2}{2\sigma^2}\right).
$$

在最近点唯一、最近投影位于光滑面片内部时，Envelope theorem 允许对最优距离值求导，而不显式展开最近点在表面参数上的重新优化项。对 JOINT `i`：

$$
\frac{\partial\rho_{\sigma,g}}{\partial q_i}=\frac{\rho_{\sigma,g}}{\sigma^2}(x-y_g^*)^T J_{p,g,i}(q,y_g^*).
$$

经典点 Jacobian column 为：

$$
J_{p,g,i}(q,y)=\frac{\partial y}{\partial q_i}=\omega_i^h(q)\times y+v_i^h(q)=\omega_i^h(q)\times\left(y-p_i^h(q)\right),
$$

前提是 JOINT `i` 位于 owner `g` 的祖先链；若不是祖先，该 column 为结构零。其单位为 m/rad。

定义 query 到 surface 的 outward radial direction `n_hat` 与 distance sensitivity：

$$
\widehat n_g=\frac{x-y_g^*}{d_g},\qquad \kappa_{g,i}=\frac{\partial d_g}{\partial q_i}=-\widehat n_g^T J_{p,g,i}.
$$

则 Gaussian field sensitivity 为：

$$
g_{\sigma,g,i}=\frac{\partial\rho_{\sigma,g}}{\partial q_i}=-\frac{d_g}{\sigma^2}\rho_{\sigma,g}\kappa_{g,i}.
$$

当前源码与该推导同号：`selected_point_jacobian()` 计算 `omega x y + v`；target 计算 `kappa = -(radial_direction * point_jacobian).sum()`；`field_sensitivity_from_distance()` 计算 `-(d/sigma^2) rho kappa`。现有 float64 finite-difference contract 已验证 sampled point Jacobian 和 chain/scale laws。

图中“刚体？”疑问在当前 owner 语义中已经闭合：每个 owner 是刚性共动 collision union，fixed descendants 已被吸收到 owner-local geometry。若同一个 owner 内 components 不能由一个 `T_hg(q)` 刚性移动，说明资产 owner 语义错误，不能由模型或 target backend 补救。

Envelope theorem 的简单形式只在最近点局部唯一且光滑时适用。Medial axis、最近三角面/基本体切换、triangle edge/vertex 和精确 `d=0` 仍需 mask 或分层诊断。当前 `feature_margin_m` 只排除当前 triangle 内靠近边界的投影，不是全局 second-nearest/medial-axis 证明。

### 6.2 `z_i^(1)` 的物理本质

本轮最重要的语义修正是：`z_i^(1)` 不应理解为“JOINT i 对自身 `z_i^(0)` 求导”，也不只描述 JOINT owner 自身。它应表示 **JOINT i 正方向微小变化对整手所有 owners 的场影响，也就是整手场 Jacobian 第 i 列的固定宽度学习式表示**。

对 owner `g` 上物质点的经典点 Jacobian：

$$
J_{p,g}(q,y)\in\mathbb R^{3\times N_J},\qquad \dot y=J_{p,g}(q,y)\dot q.
$$

距离灵敏度把该三维速度 Jacobian 沿 query-surface 法向投影。将 owner/query 合并为行索引 `alpha=(g,r)`，得到整手 field Jacobian：

$$
K(q)\in\mathbb R^{(G N_Q)\times N_J},\qquad K_{\alpha i}=\kappa_{g,r,i}.
$$

矩阵行描述某个 owner/query 的距离响应，矩阵列描述某个 JOINT 对整手全部 owner/query 场的影响。对 JOINT `i`：

- 它会影响自身 JOINT owner；
- 它会影响同一 finger chain 上所有 downstream/descendant JOINT/TIP owners；
- 它不会直接移动 PALM、upstream owners 或其他 fingers 的 owners，这些项为结构零；
- 某个 active edge 的法向投影偶然为零仍是正确物理值，不能按 target magnitude 拒绝。

这与 PoE 一致。对 owner `g`：

$$
T_{hg}(q)=\left(\prod_{j\in\operatorname{Anc}(g)}\exp\left(\widehat{\mathcal S}_j(q_j-q_j^{home})\right)\right)T_{hg}^{home}.
$$

因此一根 finger 上 proximal JOINT 的 `z_i^(1)` 应概括其对整条 descendant chain 的影响，不是只描述一个局部 owner。

### 6.3 当前 sensitivity decoder 的矩阵分解解释

当前 `D_kappa` 近似使用：

$$
\widehat K_{\alpha i}=a_\alpha^T z_i^{(1)},\qquad a_\alpha=a\left(z_g^{(0)},\Psi_C(x_{g,r})\right).
$$

`a_alpha` 是 owner/query 的条件行特征，`z_i^(1)` 是 JOINT `i` 的列特征。该形式可以解释为随资产与 q 条件变化的 field Jacobian factorization。`D_1` 不需要等于经典 twist 的 3 或 6；它是 learned column representation width。

但不能过度声称“低秩瓶颈已经成立”。当前 dataset 最大 `N_J=16`，而 `first_order_width=64`，所以 `D_1=64` 并没有相对 column count 形成严格低秩约束。本轮决定首轮仍保持 64，避免同时修改太多变量；后续正式比较 16/32/64，再根据误差、迁移、延迟和显存选择。

### 6.4 不物化完整 Jacobian，只采样矩阵元素

完整 `K(q)` 不需要预测或落盘。对 `G=21, N_Q=64, N_J=16`，每个 q 有 21504 个 owner-query-joint 元素；完整逻辑 target `[B,G,N_Q,N_J]` 和多 sigma `[B,G,N_Q,N_sigma,N_J]` 不应成为默认显式存储。

训练只在分层样本集合 `Omega(q)` 上监督：

$$
\mathcal L_\kappa=\frac{1}{|\Omega(q)|}\sum_{(\alpha,i)\in\Omega(q)}\left|\widehat K_{\alpha i}-K_{\alpha i}\right|^2.
$$

随机采样能近似覆盖完整 field Jacobian，但必须满足列覆盖、active/zero 分层和固定 validation bank；不能依赖无约束随机性自动保证合理分布。

## 7. 一阶 edge 采样合意

### 7.1 `edge` 的定义

这里的 edge 不是 collision edge，也不是 Transformer message edge。一个 sensitivity edge 是三元组 `(owner g, query r, joint i)`，表示在固定 query 上监督 `partial d_g / partial q_i` 及对应各 sigma 的 `partial rho / partial q_i`。

Current code 是 owner-first：每 owner 从 shell queries 取 `edges_per_owner=2`，交替选择 ancestor 与 non-ancestor JOINT。本轮决定改成 joint-first，因为 `z_i^(1)` 是 field Jacobian 第 `i` 列表示，应保证每个有效 JOINT column 在每个 q 都获得监督。

### 7.2 训练 edge budget

首轮每个有效 JOINT、每个 q 使用：

- 1 条 active edge：从该 JOINT 的 descendant owner 与该 owner 的 shell query 中选择；
- 1 条 structure-zero edge：从 non-descendant owner 与其 shell query 中选择，target 精确为零。

因此每个 q 约有 `2 N_J` 条一阶 edges。对 16 DOF 手为 32 条，仍小于当前 `2G` 级别预算，并远小于完整 Jacobian。

Active owner 采用可恢复的四步均衡周期：`self owner -> same-finger TIP -> other descendant -> other descendant`，长期测度为 self 25%、tip 25%、其余 descendants 50%。当 distal JOINT 的 descendant 类别较少时，可以复用合法 owner 但更换 shell query，不要求 owner 唯一。

TIP active stratum 不等于把抓取任务标签偷渡进 SSL。TIP 是资产定义的物理 owner，same-finger tip response 对应经典 chain-end point Jacobian，是 task-free velocity kinematics 的自然诊断。训练也不能只看 TIP，否则 `z_i^(1)` 会退化为 tip-response token，失去完整 column 语义。

Structure-zero owner 采用长期分层轮换：PALM、same-finger upstream owner、other-finger JOINT owner、other-finger TIP owner。不存在某类时从其余合法类别补足，并显式记录实际 stratum。

### 7.3 拒绝 hard ancestor mask

`owner_ancestor_mask[g,i]` 当前只存在于 typed `EmbodimentGeometrySpec`，用于 PoE/FK、解析 point Jacobian、active/zero edge 选择和 target 生成。它不在 retained encoder 的 `StaticGeometryEvidence` raw input 中，不进入 PPO，也不进入当前 `D_kappa`。

讨论中曾提出让 `D_kappa` 输出直接乘 `owner_ancestor_mask`，随后用户明确拒绝。当前 backbone 是 graph-biased Transformer：全连接 attention 保留全手交互，`shortest_path / parent_direction / child_direction` 只作为 soft bias，不使用 hard graph mask。

最终合意是：不把 ancestor mask 喂给模型，也不在 prediction 端硬乘 mask。Zero samples 的科研作用是让模型从 soft topology relation 与物理监督中学习“上下文相关不等于运动学因果相关”。这样才能检查 `z_i^(1)` 是否本身携带 Jacobian column 的支撑语义，而不是训练期 decoder 借 hard mask 得到答案。

### 7.4 一阶 query 来源

Kappa/g active 与 zero edges 首轮都只从 owner-shell queries 抽样。Shell query 在采样阶段随 owner surface 共动，但 query coordinates 在 target/model derivative 前停止梯度；Sobolev/JVP 求导时固定本次 query，因此得到的是固定 `{h}` 点上的局部法向速度响应，不含 query 共动项。

Shell query 离明确 owner surface 近，最贴近经典 point Jacobian 的法向投影；workspace 可能远离表面并发生 closest-source switch，adjacent 位于 owner 间隙且 source 切换更复杂。首轮 density 使用三种 query mixture，一阶监督只使用 shell。

Active edge 需要 distance epsilon、triangle feature margin 和后续可能的 global source stability mask。Structure-zero edge 的导数由拓扑严格为零，理论上不依赖最近点光滑性；当前代码把 active/zero 共用 smoothness mask，会不必要地丢弃 zero supervision，实施时需修正并保留分开有效率统计。

### 7.5 Validation edge bank

固定 validation bank 对每个有效 JOINT 使用 4 active + 4 zero edges，比训练的 1+1 更密，用于稳定评估稀疏训练是否学到整列语义，而不是增加训练预算。

四类 active 至少固定报告：self owner、same-finger tip、other descendant，以及另一条 descendant owner/query realization。四类 zero 至少固定报告：PALM、same-finger upstream、other-finger JOINT、other-finger TIP。某类不存在时补足，但必须记录实际类别。

Validation 除总指标外，必须分层报告 self、tip、other-descendant 与各类结构零的 kappa/g error、edge valid rate 和 mask 原因。不能只报告一个合并 MSE。

### 7.6 PPO 消费边界

用户确认首版下游思路仍是 per-joint action，而不是整手固定 action vector。Action `i` 读取自己的 `z_i^(1)`；跨关节协调暂由 whole-hand `Z^(0)` 的空间读取、当前控制量和原始历史承担，不在本轮设计新的普通 `z^(1)` cross-attention。

Per-joint action 不等于关节独立决策。每个 `z_i^(0)` 已经过 whole-hand Transformer 上下文化，PPO 后续还可让 JOINT `i` 用浅层 cross-attention 查询全部 `Z^(0)`。直接把所有 sign-odd `z_j^(1)` 放入普通 attention 容易破坏独立 `Z_2^(N_J)` 变换律；typed/equivariant first-order message passing 是未来候选，不属于本轮实施范围。

## 8. Objectives 的科研语义与配置组织

### 8.1 首轮五项主 objective

本轮确认首轮继续协同开启以下五项：

1. density reconstruction：`rho_hat` 对齐逐 owner/query/实际 sigma 的 `rho`；
2. distance sensitivity：`kappa_hat` 对齐 sampled field-Jacobian elements `kappa`，直接塑造 `z_i^(1)`；
3. derived field：由预测 density、预测 kappa 和 teacher distance 组成的 `g_hat^(kappa)` 对齐 teacher `g`；
4. Sobolev/JVP：同一 neural density predictor 对固定 query/sigma 的 q 导数对齐 teacher `g`；
5. chain consistency：`g_hat^(kappa)` 与 neural density q derivative 彼此对齐。

总关系仍可写为：

$$
\mathcal L_{SSL}=\lambda_\rho\mathcal L_{density}+\lambda_\kappa\mathcal L_\kappa+\lambda_g\mathcal L_g^{(\kappa)}+\lambda_{Sob}\mathcal L_{Sob}+\lambda_{chain}\mathcal L_{chain}.
$$

原第六项 paired latent parity loss 不再常驻主 loss；joint-sign 通过 20% coordinate rewrite 进入普通五项物理监督，并以 fixed validation dual-forward audit 检查 parity。

`derived_field` 和 `chain` 在 density/kappa/Sobolev 全正确时具有一定冗余，但首轮仍保留，用于把显式一阶 column representation 与同一 density function 的 q derivative 绑在一条计算图中。后续 ablation 必须能分别关闭五项，判断每项收益是否超过计算和优化成本。

### 8.2 每 asset-q 等权 reduction

当前代码把各 term 的所有有效 owner/query/sigma/edge scalars 跨 minibatch 全局相加，再除以总 scalar count。这样 owner 更多、DOF/edge 更多的手自然权重更大，与 dataset/trainer 声明的“所有 resolved assets 平等”冲突。

本轮确认跨异构手型使用每 `(asset,q)` realization 等权的科研测度：

```text
term 内先归约 owner/query/sigma/edge 等物理轴
-> 每个 (asset,q) 得到 term statistic
-> 同一 asset 的 q 等权
-> minibatch 中 assets 等权
-> accumulation 按真实 asset-q numerator/denominator 合并
```

每个 objective 仍负责解释自身 owner/query/edge axes，并返回 additive numerator/denominator；Trainer 不应理解这些内部轴。用户认为类似的 morphology/effective-sample equal weighting 将来可能也适用于 heterogeneous PPO，但 PPO reduction 不属于本轮任务。

### 8.3 IsaacLab 风格 Callable TermCfg

用户认可把当前六个空壳 objective runtime classes 收敛为 IsaacLab manager-term 风格的无状态 callable 配置。概念形态为：

```python
OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(
    density=ObjectiveTermCfg(func=density_objective, weight=..., ...),
    kappa=ObjectiveTermCfg(func=kappa_objective, weight=..., ...),
    derived_field=ObjectiveTermCfg(func=derived_field_objective, weight=..., ...),
    sobolev=ObjectiveTermCfg(func=sobolev_objective, weight=..., ...),
    chain=ObjectiveTermCfg(func=chain_objective, weight=..., ...),
)
```

`func` 接收 method 提供的 typed lazy context，返回 `ObjectiveTermResult`，其中保存 additive statistics 与诊断。Callable 必须是模块级可导入函数，artifact/checkpoint 记录完整限定名，不能使用 lambda 或进程对象地址。字段为 `None` 时表示显式关闭该 objective。

用户提到可以先定义 objective 配置基类，再由具体项继承。当前倾向是：所有无 term-specific 超参的项直接使用共同 `ObjectiveTermCfg(func, weight, ...)`；只有未来确实拥有独立 loss 超参时才增加 typed subclass，避免退回 `params: dict[str, Any]`。精确 class/interface 尚未实施或最终批准。

Method 继续拥有 lazy context：derived field、density q-JVP 和 joint-sign transformation 等共享节点在一个 method step 内至多计算一次；objective callable 不自行重新运行 encoder、decoder 或 JVP。

## 9. Objective 权重与 calibration 讨论

### 9.1 已识别的问题

当前五项 loss 的数值尺度和单位不同。Density error 无量纲；kappa error 基于 m/rad；field-sensitivity error 基于 `1/rad`。此前提出用固定 reference scale 先无量纲化，用户明确质疑：任意除以一个“一厢情愿”的参考值可能只是形式上消除量纲，不能代替科学调参。因此本轮没有锁定固定 kappa/g reference scale。

历史自动梯度 multiplier 路线已放弃。当前 calibration 只在真实训练采样分布上流式记录五项 additive statistics 和逐 batch trace，由研究者依据 artifact 人工修改显式权重。

### 9.2 当前决定：前向预实验与人工权重选择

正式训练前先运行 `calibrate_objectives`：完整复用 train partition、每 epoch 的 q coverage、minibatch shape、query/sigma jitter、anchor 轮换与 joint-sign rewrite，只减少显式 epoch 数，并取消参数 backward 和 optimizer update。Artifact 记录五项真实量级，研究者人工选择正式权重。

首轮不引入 GradNorm、MGDA、PCGrad、CAGrad、自动 multiplier、参数梯度范数或梯度余弦。若预实验或正式 pilot 暴露持续冲突，再单独设计对照。

此前术语 `reader` 已向用户澄清为“训练专用解码器”，不再单独引入奇怪概念。模型可按人话分为：

1. 共享几何网络：点-anchor、home surface、screw、entity assembly 和 graph-biased Transformer；
2. retained zero/first-order 输出头；
3. SSL-only density decoder 与 sensitivity decoder，训练后删除。

当前预实验只回答“五项在真实训练采样下各自多大、分层误差在哪里”，不把一次初始化的参数梯度解释成自动权重。

### 9.3 同一 EmbodimentPretrain façade 下的 calibration phase

用户明确不希望为权重预实验新建第二套 façade。当前合意是复用同一个完整 `EmbodimentPretrainCfg`，通过 `run.phase` 区分：

```text
phase = calibrate_objectives
phase = pretrain
```

`calibrate_objectives` 与正式训练使用同一个 `data/method/trainer/run` 根配置和同一资产数据集语义，不构造第二套资产 YAML。`OnlineSamplingCfg` 只声明共同的单 epoch 采样轴；Trainer 分别声明 `calibration_epochs` 与 `pretrain_epochs`。

`pretrain` 显式加载 calibration artifact，核对 dataset hash、method/model/representation/augmentation、objective callable、共同 sampling、phase epochs、Git HEAD 与 dirty 内容指纹；只有 objective 权重、run phase 和 artifact 定位允许变化。Artifact 记录实际资产数、epoch、样本/minibatch 数、五项精确均值和 traces，不产生 multiplier。

当前 calibration 使用 run seed 对应的一次初始化；多 seed 统计不是首轮执行合同。

### 9.4 Calibration 与 pretrain 的覆盖关系

Calibration 与 pretrain 使用相同完整 train partition，以及相同的 `q_per_asset_per_epoch / assets_per_minibatch / q_per_asset_per_minibatch / shuffle / seed`。两阶段只允许显式 epoch 数和参数更新行为不同。

当前 Python façade 显式写出 `calibration_epochs=1`、`pretrain_epochs=20` 和 `q_per_asset_per_epoch=256`，但这些仍是待预实验前确认的 8192-asset 脚手架预算，不是正式数值。

Sobolev/JVP 仍需要对物理输入 $q$ 的局部 autograd；calibration 不对模型参数执行 `backward()` 或 `autograd.grad`，参数 `.grad` 保持为空。

## 10. Method nested config 的当前目标形态

本轮认可的概念结构为：

```python
STATE_MEASURE_CFG = JointConfigurationMeasureCfg(...)
REPRESENTATION_CFG = GeometryRepresentationCfg(...)
MODEL_CFG = GeometrySSLModelCfg(...)
OBJECTIVES_CFG = MultiAnchorGaussianObjectivesCfg(...)
JOINT_SIGN_REWRITE_CFG = JointSignRewriteCfg(...)

METHOD_CFG = MultiAnchorGaussianMethodCfg(
    state_measure=STATE_MEASURE_CFG,
    representation=REPRESENTATION_CFG,
    model=MODEL_CFG,
    objectives=OBJECTIVES_CFG,
    joint_sign_rewrite=JOINT_SIGN_REWRITE_CFG,
)
```

`state_measure` 放在 method 而不是 Trainer，是因为 Trainer 只声明每资产需要多少 q 和何时更新；具体 `q ~ mu(q|asset)` 是 task-free method 的物理采样测度。首轮 concrete measure 仍是完整 limit Sobol。用户偏向该解耦方向，但精确类型尚未实现。

`JointSignRewriteCfg` 是 multi-anchor Gaussian method-specific augmentation；anchor bank 数量/生成测度属于 representation/source；objective callable aggregate 属于 method objectives；calibration/pretrain epochs、validation/final evaluation、training budget 和 optimizer 属于 Trainer。

### 10.1 Representation 内部组织

当前 `GeometryRepresentationCfg` 同时包含 source、field/sigma、query、target sampling 和 padding layout。前四项都是科研样本/teacher realization，可以继续由 representation 拥有；`layout` 是跨结构 dense batching 容器，与表征语义混在一起并重复配置 `max_graph_distance`。

用户偏向把 padding layout 从科研 representation config 移出并自动推导：joint/tip/owner 上限从 resolved dataset 的实际结构得到，graph bucket 上限由 model backbone config 得到，避免实验配置重复手写 `20 joints / 5 tips / graph distance 8`。用户同时表示对此尚未完全理解，因此实施前需要解释自动推导的 fail-closed 边界，例如 dataset 出现超出模型支持的结构时如何拒绝，而不是静默扩容。

### 10.2 Model 内部组织

`GeometrySSLModelCfg` 继续显式组合 retained encoder 与 SSL-only density/sensitivity decoders；decoder 输入宽度从 encoder output type 派生，不在实验配置复制 `D_0/D_1/D_q`。

当前 model 的 joint-sign parity 只是软约束。曾提出把 screw frontend 拆成 even axis geometry 与 odd directional carrier：zero-order 只消费 even 量，current displacement 与 odd carrier 相乘形成 even motion，一阶输出用 even 条件无偏置读取 odd carrier。用户表示暂不了解并要求先保留讨论，因此该结构尚未锁定。实施计划必须把它作为核心未决模型问题，而不是直接修改网络。

## 11. 正式训练预算的未决问题

新 train partition 有 8192 assets，旧 45-asset recipe 不能直接延用。需要同时考虑：资产覆盖、每资产 q coverage、anchor bank 8 套轮换、20% sign rewrite、5 项 objective、validation cadence、GPU 显存和实际墙钟。

一个尚未确认但值得计算的预算候选是：每 asset 总共约 32 个 q，使用 `assets_per_minibatch=2`、`q_per_asset_per_minibatch=2`、一次 coverage epoch。这会产生约 262144 个 train q realizations、65536 minibatches，若 gradient accumulation 为 4 则约 16384 optimizer updates，数量级接近旧 45-asset recipe 的 230400 q 和 14720 updates，同时每资产只覆盖一轮 32 q。它只是预算候选，不是已达成合意。

另一个等价组织是 2 个 epoch、每 asset 每 epoch 16 q，使 8 套 anchor bank 在每个 epoch 中各出现一次；这会改变 epoch/validation 语义但不改变总 q 数。需要以后从 checkpoint cadence、anchor bank rotation 和研究报告习惯选择，而不是沿用旧 `20 epochs` 作为心理默认。

Validation 以 density、κ、derived-field 三项初始化归一化后等权，并对 suites 等权选择 best；Sobolev/chain 不参与选点。训练结束后加载冻结 best，独立运行 evaluation unseen-variant/unseen-mother 的固定测度、分层指标、六项 ablation 和 bootstrap；official-zero-shot 为空时显式报告空集。

## 12. `question.md` 中的工程债与科研边界

以下来自 `source/anymani/anymani/distill/doc/spec/question.md`，主要是 infra 工程，不应在没有科研授权时改写物理测度。

### S0，正式 pilot 前需要处理

1. DONE：window-major schedule 和 Method session 让同一 resident window 完成 q coverage 后再切换。
2. DONE：calibration/validation/final evaluation 流式消费 batch，各 session 在 `finally` 中释放 resident state。
3. DONE：真实 generated-asset smoke 走通 calibration、update、validation、best reload、retained export 与 final evaluation。
4. DONE：checkpoint/calibration 同时记录 HEAD、dirty 状态和 tracked diff/untracked 内容摘要；正式实验仍应在干净提交上运行。

### S1，科研结论前需要处理

5. DONE（执行路径）：冻结 best 后运行 evaluation suites；official-zero-shot 空集显式报告。
6. diagnostics logger 尚未接入正式 lifecycle。需要记录五项 raw numerator/denominator、按 asset/owner/query stratum/sigma/distance shell/ancestor 分层的 error 与 valid rate、gradient norm、resident telemetry、q/query/teacher digest、validation 与 independent q-bank provenance。
7. DONE：Method 返回 morphology/bin/axis 分层统计，Trainer 只使用三项重建指标并对 suites 等权选点。
8. 当前最近面 mask 只有 face validity、owner-shell、distance epsilon 和 triangle feature margin，没有 global second-nearest/medial-axis margin。首轮至少要报告按 owner/query stratum/distance shell 的 valid rate、mask 原因和 source stability，不能把 local feature margin 写成全局唯一性证明。

### S2，后续清理

9. canonical compile gate 当前只拒绝未知 objective 名称，没有决定是否强制精确五项集合；本轮既要支持五项全开，也要允许显式 ablation。
10. `ssl/calibration.py` 与 lifecycle 内 calibration、`ssl/runtime/objective.py` 与 active method、`WindowedOnlineGeometryBatcher` 与正式 schedule 存在重复/孤立实现。科研合同稳定后再判断哪些出清，不要为了清理而改变生命周期。
11. input adapter 当前把 `palm_normal=(0,0,1)` 作为 hard assumption。资产 semantic/lowering 应显式交付和验证 directed palm normal，不能只靠代码默认值。
12. parametric Gaussian components、旧 attention-bias、temporal candidate scaffold 不属于当前 canonical implicit method 调用图，不应被记录为已验证能力。

## 13. 当前科研未决项总表

### 高优先级

- 8192 train assets 下的总 q budget、epoch 定义、每 asset coverage 和 validation cadence；
- 现有普通 MLP screw path 是否改成严格 even/odd typed architecture，还是首轮继续软 parity 并用 20% rewrite 训练。
- diagnostics logger、resident telemetry 和最近 source 非光滑区域的正式分层证据。

### 中优先级

- active edge 是否继续只从 shell 采样，以及 global source stability mask 是否首轮只报告不物化；
- home surface 64 点的 remesh/seed/point-density 稳健性；
- first_order_width 的 16/32/64 容量消融，首轮保持 64；
- anchor workspace 5 cm mount-centered sphere 是否覆盖足够的 palm/finger spatial region，首轮不回到 enclosing box；
- owner-adjacent 是否长期保持 one-hop graph-only，首轮不增加 cross-finger pure spatial query；

### 明确不属于本轮

- PPO 的完整 cross-attention、per-joint policy head 和 MDP 结构落地；
- analytic direct geometry route 的实现和公平 pilot；
- object/task/contact/action-conditioned field；
- dynamic GradNorm/MGDA/PCGrad/CAGrad；
- full Jacobian materialization；
- PPO runtime mesh distance、closest point、完整 Jacobian 或学习 latent history cache；
- 在没有证据的情况下声称 cross-topology、cross-DOF、official zero-shot 或 policy generalization 成立。

## 14. 下一步实验顺序

1. 确认 8192-asset calibration/pretrain 的 epoch 与每资产 q 预算。
2. 运行 `calibrate_objectives`，检查五项均值、trace 和已有分层指标，人工确定 `OBJECTIVES_CFG` 权重。
3. 用 CLI override 启动正式 pretrain；validation 只选 best，冻结后 evaluation 只报告 unseen suites。
4. 根据预实验暴露的问题再处理 logger、source stability 或模型结构，不在实验前继续泛化框架。

## 15. 本轮修改边界

原始讨论记录只用于恢复合意；2026-08-21 后续实现已修改 method/trainer/checkpoint/tests，并按用户要求执行 patch bump。Research vault 未由本次实现改写；现有无关 dirty worktree 保持不动，未创建或移动 tag。

## 16. 2026-08-23 显式 minibatch 预算决议

此前按 phase epoch 和每资产 q 配额反推训练长度的方案不再是当前合同。训练配置只保留 `num_minibatches`、`assets_per_minibatch`、`q_per_asset_per_minibatch` 与 `mini_epochs`：前三者决定新生成的数据量，`mini_epochs` 决定同一批 q/query/teacher realization 的循环利用次数。预实验与正式实验复用同一配置类型和运行管线，但每次运行允许使用不同 preset；预实验产物用于人工判断五项 objective 权重，不自动改权重，也不要求正式 preset 与预实验 preset 完全相同。

首个 8192-asset preset 为 `128 × 64 × 8 × 5`。它生成 65536 个不同 `(asset,q)` 样本；每 4 个新 minibatch 组成一个梯度累积组，同一组循环使用 5 次，每次重新抽 joint-sign rewrite。由此执行 640 次 minibatch forward 和 160 次 optimizer update。checkpoint 只在一组完成全部五次复用后保存，恢复点不需要持久化临时 teacher batch。
