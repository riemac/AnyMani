# Embodiment Geometry Pretraining

直接从任务回报学习跨手型策略，会把手的几何、接触、动作坐标与任务目标同时压进同一表示。即使 reward 上升，也难以判断网络恢复了可迁移的物理结构，还是只记住某一手型和任务的统计捷径。本模块先研究一个更窄、可证伪的问题：不启动 Isaac Sim、不观察物体，只给定手型 $\mathfrak m$、当前物理关节构型 $q$ 和静态几何证据，能否编码整只手的局部空间占据及其对关节运动的一阶响应？

这不是离线生成完毕的数据集训练，也不是 on-policy reinforcement learning。资产 YAML 与静态 catalog 预先存在；训练时在线采样 $q\sim\mu(q\mid\mathfrak m)$ 和方法随机条件 $\xi\sim\nu(\xi\mid\mathfrak m,q)$，再由无梯度 physical oracle 计算 $y=T(\mathfrak m,q,\xi)$。因此它是 online procedural supervised pretraining：生命周期具有 collect/realize、objective、backward 和 update，但没有 transition、reward、return、GAE、importance ratio 或 PPO clipping。

## 物理监督

资产 sidecar 把手组织为 PALM、JOINT 与 TIP owners。对 owner $g$、固定 hand-frame query $x$ 和构型 $q$，teacher 从真实 collision union 计算 unsigned distance $d_g(x;q)$，并以实际米制带宽定义 Gaussian 邻近场：

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

一阶监督首先计算距离灵敏度

$$
\kappa_{g,i}(x;q)=\frac{\partial d_g(x;q)}{\partial q_i},
$$

再由链式法则得到场灵敏度

$$
g_{\sigma,g,i}(x;q)=\frac{\partial\rho_{\sigma,g}}{\partial q_i}=-\frac{d_g}{\sigma^2}\rho_{\sigma,g}\kappa_{g,i}.
$$

$d$、$\kappa$ 与 $g$ 的单位分别为 m、m/rad 与 rad$^{-1}$，$\rho$ 无量纲。模型不会把每只手单独缩放到相同尺寸；query、geometry、anchor 和 bandwidth 共同保留绝对物理尺度。

`GeometrySource` 每项资产只物化一次 q-independent oracle：CPU float64 POE 规格、owner ancestry、严格 collision union、boundary-only home points、palm-seed anchors、physical identity 和可释放的 Warp BVH lease。`GeometryRepresentation` 随 minibatch 在线产生 q-conditioned query、实际 sigma、sampled edges 与 privileged targets。每个 owner 使用 64 个 query，其中 workspace、owner shell 与 adjacent gap 的比例为 50:25:25；stratum 只记录采样测度，不进入 decoder。训练 sigma centers 为 4、16、64 mm，并做 log-space ±10% jitter；validation 固定使用 4、8、16、32、64 mm。

## 保留表示与训练读取器

retained encoder 只读取当前物理 q 与静态 evidence。多锚点前端把 home surface points 和空间旋量表示为相对完整 anchor constellation 的关系，并对 anchor permutation 聚合；hand frame 绕 palm normal 的面内旋转是 $SO(2)$ gauge，reflection/chirality 仍是物理差异。

owner tokens 进入两层、四头、hidden width 128 的 graph-biased encoder-only Transformer。运动学无向距离、parent distance 和 child distance 只提供每头可学习的加性 bias，不是 hard attention mask。模型输出逐 owner 的 $Z^{(0)}\in\mathbb R^{128}$ 与逐 JOINT 的 $z_i^{(1)}\in\mathbb R^{64}$，一阶 head 的真实上游关系是

$$
z_i^{(1)}=H_1\!\left([z_i^{(0)}\Vert f_i^{screw}]\right).
$$

$f_i^{screw}$ 不直接送入当前 κ reader。κ reader读取 owner $z_g^{(0)}$、query feature、对应 $z_i^{(1)}$ 和结构化 joint selector；density reader读取 owner $z_g^{(0)}$、query feature 与显式 $\sigma$。两个 readers、query backend 和 target backend 都只存在于预训练阶段，standalone retained artifact 只保留 encoder 配置、`encoder.*` state、输入合同、`FeatureSpec` 与代码/资产 lineage。

## 六个可独立审计的目标

当前 method 组合六个独立 objective term：density reconstruction、distance sensitivity、derived field、Sobolev/JVP、chain consistency 与 paired parity。每项返回 additive numerator/denominator；Trainer 只跨一个 optimizer update 的 minibatches 合并同名统计量，不理解 owner、query、edge 或 latent channel 轴。Method 提供 typed lazy context，使基础 encoder forward、density q-JVP、derived field 和 paired second forward 在一个 minibatch 内各至多计算一次。

joint-sign rewrite 同步变换 $(q_i,q_{home,i},\mathcal S_i)\mapsto(-q_i,-q_{home,i},-\mathcal S_i)$，表示同一物理运动的另一坐标记号。因此 paired term 约束 $Z^{(0)}$ 为偶、$z_i^{(1)}$ 为奇；网络不接收裸 sign bit，也不构造 `screw_even/screw_odd` 双支。六项权重先在 8 个固定 train minibatches 上测量 shared-encoder gradient median；校准结果写入 `loss_calibration.yaml` 和 full checkpoint metadata，但不改写 frozen YAML 声明。

## 声明式运行架构

schema 3 的 `EmbodimentPretrainCfg` 只组合五个 concrete roles：`data` 解析平等 asset catalog；`method` 串联 representation、model 和 objectives；`trainer` 拥有 asset permutation、Sobol q、coverage、minibatch、gradient accumulation 与 optimizer；`evaluation` 拥有 fixed validation bank、selection metrics、q-bank replay 与 ablation；`run` 拥有输出位置、resume、随机性和 lineage。每个 concrete cfg 以不序列化的 `ClassVar runtime_type` 绑定本地 runtime；最高层只有统一构造 helper，没有 `_target_`、字段 parser 或万能 registry。

正式实验由 [`../../presets/ssl/canonical_multi_anchor_gaussian.yaml`](../../presets/ssl/canonical_multi_anchor_gaussian.yaml) 组合 data、method、representation、model、六个 term、trainer、evaluation 和 run。Hydra composition 直接恢复 concrete dataclasses；schema 1/2 配置和 checkpoint 均 fail-closed。

## 在线日程与恢复

每个 q round 都以 `(seed, epoch, q_round)` 确定性打乱全部 train assets，再按 `assets_per_minibatch` 分组。每项选中资产获得 `q_per_asset_per_minibatch` 个连续 Sobol q，直到达到该 epoch 的 per-asset coverage。尾资产组、尾 q block 和尾 accumulation group保持真实较小长度，不填充虚假资产，也不重复样本。

canonical dataset 固定 45 train、16 validation、16 unseen-variant-set 和 17 unseen-mother assets。每个 epoch 每项 train asset 消费 256 个新 q，共 20 epochs；2 assets × 2 q 构成一个 minibatch，累积 4 个 minibatches 后更新。因此每 epoch 为 2944 minibatches、736 updates，完整预算为 14720 updates 与 230400 q realizations；30000 只是 safety limit，不是训练预算。

schema 3 full checkpoint 保存完整 model/readers、optimizer、runtime calibration、当前 asset permutation、epoch/q-round/group cursor、每资产 Sobol cursor、validation selection history、CPU/CUDA RNG、resolved config 和 expanded physical manifest。resume 后下一资产组与下一 q 必须和不中断运行一致。full checkpoint 只服务恢复与审计；训练结束后加载 validation-best immutable checkpoint，再导出独立 `retained_artifact.pt` 供 IL/PPO 使用。

validation 每项资产固定 64 q，每 250 updates 计算 density、κ 与 derived-field 的 initialization-normalized score。冻结后执行 query-only、same-asset q shuffle、cross-asset shuffle、first-order zero、JOINT shuffle 和 sign flip，并进行 2000 次 paired bootstrap。training-morphology independent-q bank 在初始化和最终模型上重放完全相同的 q/query/teacher digest。unseen-variant-set、unseen-mother 和 official-zero-shot 当前只完成 asset/physical identity 审计，尚未形成模型效果结论。

## 证据边界

当前 deterministic contracts 覆盖 schema 3 concrete composition、asset shuffle/coverage/tails/resume、source realization、query/sigma replay、graph bias、padding masks、$SO(2)$、joint-sign parity、拆分前后六项数值与梯度等价、full checkpoint 和 standalone artifact。distill contracts 当前 117 项通过；synthetic integration 通过，real integration 按环境条件执行。RTX 5070 Ti 上既有 retained encoder $B=4096$ p95 为 20.13 ms，但该计时排除 teacher、readers、policy、Isaac Sim 与 `env.step`，不能外推为完整 20 Hz 控制系统。

正式 20-epoch pilot、unseen suites 模型评估、official zero-shot、Isaac pose parity 和 PPO transfer 尚未运行。目前证据支持实现与物理合同闭合，不支持跨手型泛化已经成立。

## 运行入口

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain

# Hydra override 示例
python -m anymani.distill.ssl.pretrain \
  trainer.optimizer.learning_rate=1e-4 \
  method.representation.query.query_count=128
```

产物位于 `logs/ssl/<experiment>/<UTC timestamp>/`，包括 resolved config、dataset/expanded manifest、calibration、JSONL metrics、fixed validation、independent-q replay、ablation、selection history、full checkpoints 和 standalone retained artifact。实验统计语义见 [`../diagnostics/README.md`](../diagnostics/README.md)。
