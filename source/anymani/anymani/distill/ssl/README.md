# Embodiment Geometry Pretraining

直接从任务回报学习跨手型策略，会把手的几何、接触、动作坐标与任务目标同时压进同一表示。即使 reward 上升，也难以判断网络恢复了可迁移的物理结构，还是只记住某一手型和任务的统计捷径。本模块先研究一个更窄、可证伪的问题：不启动 Isaac Sim、不观察物体，只给定手型 $\mathfrak m$、当前物理关节构型 $q$ 和静态几何证据，能否编码整只手的局部空间占据及其对关节运动的一阶响应？

这不是离线生成完毕的数据集训练，也不是 on-policy reinforcement learning。资产 YAML 与静态 catalog 预先存在；训练时在线采样 $q\sim\mu(q\mid\mathfrak m)$ 和方法随机条件 $\xi\sim\nu(\xi\mid\mathfrak m,q)$，再由无梯度 physical oracle 计算 $y=T(\mathfrak m,q,\xi)$。因此它是 online procedural supervised pretraining：生命周期具有 collect/realize、objective、backward 和 update，但没有 transition、reward、return、GAE、importance ratio 或 PPO clipping。

## 物理监督

资产 sidecar 把手组织为 PALM、JOINT 与 TIP owners。对 owner $g$、固定 hand-frame query $x$ 和构型 $q$，teacher 从真实 collision union 计算 unsigned distance $d_g(x;q)$，并以实际米制带宽定义 Gaussian 邻近场：

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

一阶监督首先计算距离灵敏度 $\kappa_{g,i}=\partial d_g/\partial q_i$，再由链式法则得到场灵敏度 $g_{\sigma,g,i}=-(d_g/\sigma^2)\rho_{\sigma,g}\kappa_{g,i}$。$d$、$\kappa$ 与 $g$ 的单位分别为 m、m/rad 与 rad$^{-1}$，$\rho$ 无量纲。

`GeometrySource` 每项资产只物化一次 q-independent oracle，并独立生成 8 套 physical anchor bank；同资产 q-block 内共享一套并均衡轮换，validation、独立 q-bank 和 PPO 固定 $A^{(0)}$。每个 owner 使用 64 个 query，workspace / owner-shell / adjacent 的比例为 50:25:25。训练 sigma 中心为 4、16、64 mm，并做 log-space ±10% jitter；validation 关闭 jitter，仍用同一组 4、16、64 mm。每个有效 JOINT、每个 q 固定抽取 2 条 active 与 1 条 structural-zero，并轮换 owner 类别、query stratum 及 fallback provenance。

## 统一保留表示与双目标

retained encoder 只读取当前物理 q 与静态 evidence。多锚点前端把 home surface 和带符号空间旋量表示为相对完整 anchor constellation 的关系；graph-biased Transformer 保持全连接 attention，拓扑只作 soft bias。唯一 retained 表示直接取 final-norm tokens：

$$
Z=E_\theta(q,q_{home},\mathcal S,\text{home geometry},\text{anchors},\text{topology})
\in\mathbb R^{B\times G\times128}.
$$

JOINT view 通过 `joint_entity_index` 从同一 $Z$ gather，不产生第二 latent 或 post-backbone screw bypass。density reader 与 kappa reader 各使用 2 个 FiLM residual blocks；kappa row 与 JOINT token 经无 bias 的 rank-64 双线性读取，并以固定 $0.1\,\mathrm{m/rad}$ 恢复物理尺度。两个 SSL-only readers、query backend 和 target backend 在 schema-5 retained artifact 中全部删除。

科学方法聚合根位于 `distill/methods/multi_anchor_gaussian_implicit_field/`。训练同时累计当前 run 的 teacher-only naive baseline，但该统计不反传、不改变 optimizer trajectory；训练结束才生成 normalized curves 与 skill。shared encoder 使用两任务解析 FairGrad，两个 readers 使用各自 private gradient，三个参数组分别裁剪。derived-field $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$ 与真实 density JVP 只作显式事后诊断。joint-sign rewrite 以 0.20 概率翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；只验收 observable density 不变和对应 $\kappa$ 变号，不增加 latent parity loss。

正式 pretrain 不再依赖外部 calibration artifact。每个已 realization minibatch 同时累计 run-local baseline 充分统计；checkpoint 保存统计状态，resume 继续累加。训练结束生成不可变 raw `metrics.jsonl`、`run_teacher_baselines.yaml`、`metrics_finalized.jsonl` 与 `training_summary.yaml`。validation/evaluation suite 另行累计自身 teacher baseline，不复用 train baseline。

## 声明式运行架构

schema 8 的 `EmbodimentPretrainCfg` 只组合 `data`、`method`、`trainer` 和 `run`。每个完整实验由 `experiments/` 下一个版本化 Python 快照装配；当前主实验是 [`geometry_ssl_multitask_representation_v0_7_3.py`](experiments/geometry_ssl_multitask_representation_v0_7_3.py)。registry 负责实验发现，Hydra 只负责 structured compose。validation/evaluation 如有需要，也从同一快照派生独立配置。

## 在线日程与恢复

正式 run 执行 256 epochs × 4 minibatches，即 1024 updates 与 8 个 catalog cycles。每个 minibatch 固定包含 64 个互异资产、每项生成 8 个 Sobol q，即 512 pairs；它被切成 8 个 64-pair microbatches，以完整 minibatch denominator 精确形成 shared/private gradients，再执行一次三参数组 AdamW update。每 4 epochs 保存 full checkpoint。

schema 8 pure pretrain 在首次 update 前保存 `epoch_000000.pt`，之后按配置在完整 epoch 的 optimizer boundary 保存 immutable checkpoint；`last.pt` 指向最终 epoch。checkpoint 记录 epoch、optimizer update、新 pair、每资产 Sobol cursor、CPU/CUDA RNG、run-local baseline 统计、resolved config、source artifact identity 与引用日志 prefix。epoch 内中断时从上一个边界确定性重放，teacher buffer 不进入 checkpoint。训练进程不生成 best 或 retained artifact。

Method session 封装 source、Sobol cursor、resident state 和具体 batch。训练进程不自动调用 validation/evaluation。显式 validation 只以 teacher-baseline-normalized density 与 κ 选择 checkpoint；derived-field、query-only、same-asset cross-q、cross-morphology、JOINT-token shuffle、完整 coordinate rewrite、density JVP 和 selected-parameter gradient Gram 属于手动事后证据。

## 证据边界

contracts 覆盖 unified shape/mask、SO(2)、anchor/entity permutation、graph/routing、双 reader 依赖、run-local baseline、FairGrad、microbatch 等价、source artifact、独立 post-training 与 PPO feature routing。RTX 5070 Ti 上当前 retained encoder 在 $B=4096$、20 warmups + 50 CUDA Events 下 p95 为 34.77 ms，582,343 parameters、峰值约 812.43 MiB；disposable readers p95 约 1.20 ms。正式 256-epoch run 尚未启动。

正式 8192-asset pilot、完整 unseen suites 统计、Isaac pose parity 和 PPO transfer 尚未运行。目前只证明执行与物理合同闭合，不支持跨手型泛化已经成立。

## 运行入口

日常实验使用配置驱动的 `pretrain` CLI。当前实验快照使用 `source_cache_mode=auto`：已有且 identity 匹配的 source artifact 直接复用，缺失或不完整时由训练 runtime 自动补建，然后以 readonly 方式进入训练。训练 CLI 不自动调用 validation、evaluation、PCA 或 retained export。

```bash
source /home/hac/isaac/env_isaaclab/bin/activate

# 运行正式训练；cache 检查、复用或补建由 pretrain runtime 内部完成
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.pretrain --config geometry_ssl_multitask_representation_v0_7_3 --device cuda:0 --seed 20260813

# 训练脚本的等价快捷入口
./source/anymani/anymani/distill/ssl/backup.sh

# 从指定 checkpoint 恢复；resume 不改变实验快照
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.pretrain --config geometry_ssl_multitask_representation_v0_7_3 --resume_checkpoint <last.pt> --device cuda:0

# 论文需要时，显式运行独立 validation
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.validate --config geometry_ssl_multitask_representation_v0_7_3 --baseline_checkpoint <epoch_000000.pt> --checkpoint <epoch_000032.pt>

# 论文需要时，显式运行独立 evaluation
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.evaluate --config geometry_ssl_multitask_representation_v0_7_3 --checkpoint <checkpoint.pt>
```

需要缩短开发试跑时，只覆盖预算参数，例如 `/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.pretrain --config geometry_ssl_multitask_representation_v0_7_3 --max_epochs 2 --num_minibatches 1 --assets_per_minibatch 2 --q_per_asset_per_minibatch 2 --microbatch_size 2`；这类结果不能替代 D019 的完整 256-epoch/8-cycle formal run。
