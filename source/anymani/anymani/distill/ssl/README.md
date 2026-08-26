# Embodiment Geometry Pretraining

直接从任务回报学习跨手型策略，会把手的几何、接触、动作坐标与任务目标同时压进同一表示。即使 reward 上升，也难以判断网络恢复了可迁移的物理结构，还是只记住某一手型和任务的统计捷径。本模块先研究一个更窄、可证伪的问题：不启动 Isaac Sim、不观察物体，只给定手型 $\mathfrak m$、当前物理关节构型 $q$ 和静态几何证据，能否编码整只手的局部空间占据及其对关节运动的一阶响应？

这不是离线生成完毕的数据集训练，也不是 on-policy reinforcement learning。资产 YAML 与静态 catalog 预先存在；训练时在线采样 $q\sim\mu(q\mid\mathfrak m)$ 和方法随机条件 $\xi\sim\nu(\xi\mid\mathfrak m,q)$，再由无梯度 physical oracle 计算 $y=T(\mathfrak m,q,\xi)$。因此它是 online procedural supervised pretraining：生命周期具有 collect/realize、objective、backward 和 update，但没有 transition、reward、return、GAE、importance ratio 或 PPO clipping。

## 物理监督

资产 sidecar 把手组织为 PALM、JOINT 与 TIP owners。对 owner $g$、固定 hand-frame query $x$ 和构型 $q$，teacher 从真实 collision union 计算 unsigned distance $d_g(x;q)$，并以实际米制带宽定义 Gaussian 邻近场：

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

一阶监督首先计算距离灵敏度 $\kappa_{g,i}=\partial d_g/\partial q_i$，再由链式法则得到场灵敏度 $g_{\sigma,g,i}=-(d_g/\sigma^2)\rho_{\sigma,g}\kappa_{g,i}$。$d$、$\kappa$ 与 $g$ 的单位分别为 m、m/rad 与 rad$^{-1}$，$\rho$ 无量纲。

`GeometrySource` 每项资产只物化一次 q-independent oracle，并独立生成 8 套 physical anchor bank；同资产 q-block 内共享一套并均衡轮换，validation、独立 q-bank 和 PPO 固定 $A^{(0)}$。每个 owner 使用 64 个 query，workspace / owner-shell / adjacent 的比例为 50:25:25。训练 sigma 中心为 4、16、64 mm，并做 log-space ±10% jitter；validation 关闭 jitter，仍用同一组 4、16、64 mm。一阶边只从 owner-shell 抽取：每个有效 JOINT 在训练期 1 条 active + 1 条 structure-zero，validation 为 4+4。

## 统一保留表示与双目标

retained encoder 只读取当前物理 q 与静态 evidence。多锚点前端把 home surface 和带符号空间旋量表示为相对完整 anchor constellation 的关系；graph-biased Transformer 保持全连接 attention，拓扑只作 soft bias。唯一 retained 表示直接取 final-norm tokens：

$$
Z=E_\theta(q,q_{home},\mathcal S,\text{home geometry},\text{anchors},\text{topology})
\in\mathbb R^{B\times G\times128}.
$$

JOINT view 通过 `joint_entity_index` 从同一 $Z$ gather，不产生第二 latent 或 post-backbone screw bypass。density reader 保留 query+sigma 主路径和 owner-token FiLM；kappa reader 以 query 为主路径，三个 128-width residual blocks 分别由 $[z_o\Vert z_i]$ 生成 FiLM 参数，最终输出无界 signed scalar。两个 SSL-only readers、query backend 和 target backend 在 schema-5 retained artifact 中全部删除。

科学方法聚合根位于 `distill/methods/multi_anchor_gaussian_implicit_field/`。active objective 固定为 $L=L_\rho/B_\rho+L_\kappa/B_\kappa$：$B_\rho$ 是完整训练 teacher distribution 上逐 bandwidth-slot 的 constant-density baseline，$B_\kappa$ 是严格复用 active/structural-zero 1:1 归约的 zero predictor。derived-field $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$ 与真实 density JVP 只作显式事后诊断。joint-sign rewrite 以 0.20 概率翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；只验收 observable density 不变和对应 $\kappa$ 变号，不增加 latent parity loss。

`calibrate_objectives` 与 `pretrain` 共用同一 `EmbodimentPretrainCfg` 类型、正式 `ssl.yaml` 和 teacher realization。baseline pass 必须恰好覆盖全部 8192 train assets，每项 8 个 q，共 65,536 pairs；单遍累计一/二阶矩、$B_\rho/B_\kappa$、query-stratum 诊断与 provenance，不运行 learned model。schema-7 `teacher_baselines.yaml` 明确将 `teacher_baselines` 与可选 `random_model_preflight` 分开；正式 pretrain 缺少该 artifact、hash 或 identity 不匹配时直接拒绝。

## 声明式运行架构

schema 7 的 `EmbodimentPretrainCfg` 只组合 `data`、`method`、`trainer` 和 `run`。完整实验由 [`experiments/multi_anchor_gaussion_implicit_field.py`](experiments/multi_anchor_gaussion_implicit_field.py) 用 Python 装配，Hydra 只从 ConfigStore 加载。Trainer 声明 `max_epochs / num_minibatches / mini_epochs / microbatch_size / sampling`，不存在 validation/evaluation 字段。独立 `EmbodimentValidationCfg` 与 `EmbodimentEvaluationCfg` 分别组合同一 data/method preset 和自己的 stage/run 配置。

## 在线日程与恢复

baseline pass 执行 32 epochs × 4 minibatches；正式 run 执行 256 epochs × 4 minibatches，即 8 个 catalog cycles。每个 minibatch 固定包含 64 个互异资产、每项生成 8 个 Sobol q，即 512 pairs；它被切成 8 个 64-pair microbatches，用完整 minibatch denominator 累积后执行一次 vanilla AdamW update。每 4 epochs 的最后一个 update 在 `optimizer.step()` 前额外累计 rho/kappa 对统一 $Z$ 的 norm、dot/cosine、Gram condition 与联合方向投影，只记录证据，不修改 update。

schema 7 pure pretrain 在首次 update 前保存 `epoch_000000.pt`，之后默认每个完整 epoch 保存 immutable checkpoint；`last.pt` 硬链接最终 epoch。checkpoint 记录 epoch、optimizer update、新 pair、pair use、teacher realization、全局 minibatch cursor、下一轮 permutation、每资产 Sobol cursor、CPU/CUDA RNG、resolved config 与轻量 dataset identity。epoch 内中断时从上一个边界确定性重放，teacher buffer 不进入 checkpoint。训练进程不生成 best 或 retained artifact。

Method session 封装 source、Sobol cursor、resident state 和具体 batch。训练进程不自动调用 validation/evaluation。显式 validation 只以 teacher-baseline-normalized density 与 κ 选择 checkpoint；derived-field、query-only、same-asset cross-q、cross-morphology、JOINT-token shuffle、完整 coordinate rewrite、density JVP 和 selected-parameter gradient Gram 属于手动事后证据。

## 证据边界

contracts 覆盖 unified shape/mask、SO(2)、anchor/entity permutation、graph/routing、双 reader 依赖、teacher baseline、microbatch 等价、artifact identity、4-epoch gradient cadence、独立 post-training 与 PPO feature routing。RTX 5070 Ti 上 unified retained encoder 在 $B=4096$、20 warmups + 50 CUDA Events 下 p95 为 18.275 ms，317,383 parameters、峰值 811.42 MiB；disposable decoders p95 为 1.461 ms。正式 256-epoch run 尚未启动。

正式 8192-asset pilot、完整 unseen suites 统计、Isaac pose parity 和 PPO transfer 尚未运行。目前只证明执行与物理合同闭合，不支持跨手型泛化已经成立。

## 运行入口

```bash
source /home/hac/isaac/env_isaaclab/bin/activate

# 完整 teacher-only baseline pass
python -m anymani.distill.ssl.pretrain \
  --phase calibrate_objectives \
  --max_epochs 32 \
  --num_minibatches 4 \
  --assets_per_minibatch 64 \
  --q_per_asset_per_minibatch 8 \
  --mini_epochs 1 \
  --microbatch_size 64 \
  --max_resident_assets 64 \
  --seed 20260813 \
  --device cuda:0 \
  --experiment_name canonical_unified_geometry_teacher_baselines_v0_7_2
```

Pure pretrain 通过 `--phase pretrain --calibration_artifact <teacher_baselines.yaml>` 显式启动；正式预算为 256 epochs、8 catalog cycles、checkpoint cadence 4。SSL 是 task-free PyTorch/Warp 进程，没有 Isaac Sim 窗口或 `--headless`。产物位于 `logs/ssl/<experiment>/<UTC timestamp>/`；JSONL/TensorBoard 同步保存 raw/normalized loss、skill、RMS、valid ratio、总梯度、Z-gradient proxy、预算、吞吐、显存与诊断耗时，NPZ 保存 prediction/target/mask/selectors/strata/$Z$。

```bash
BASELINE_ARTIFACT=/absolute/path/to/teacher_baselines.yaml
python -m anymani.distill.ssl.pretrain \
  --phase pretrain \
  --max_epochs 256 \
  --num_minibatches 4 \
  --assets_per_minibatch 64 \
  --q_per_asset_per_minibatch 8 \
  --mini_epochs 1 \
  --microbatch_size 64 \
  --max_resident_assets 64 \
  --checkpoint_every_epochs 4 \
  --calibration_artifact "$BASELINE_ARTIFACT" \
  --seed 20260813 \
  --device cuda:0 \
  --output_dir logs/ssl \
  --experiment_name canonical_unified_geometry_pretrain_8cycles_v0_7_2
```
