# Self-Supervised Geometry Pretraining

`ssl` 是可运行的 task-free physical-representation pretraining stage。它组合 assets/robots 提供的静态手型与运动学、在线 Warp target、retained geometry encoder、SSL-only decoder 和六项联合目标；不拥有资产生成、Isaac Lab MDP、PPO update 或 policy action semantics。

## 运行入口

在仓库根目录激活 AnyMani/Isaac Lab Python 环境后运行；该进程不启动 Isaac Sim：

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain \
  'assets.train_paths=[/absolute/path/to/generated/hand_bundle]'
```

正式 21-asset canonical residual pilot 使用自包含 experiment：

```bash
python -m anymani.distill.ssl.pretrain --config-name geometry_ssl_canonical_residual_family
```

`assets.train_paths` 与 `assets.validation_paths` 接受显式 generated bundle 列表，可同时包含 pre-made mother、post-mutate variants、跨 family 与不同 DOF。`HandBank` 自身支持 `pre_made` discovery 和 `mixed` manifest；当前 SSL CLI 尚未把 collection discovery 作为独立字段暴露。`assets.official_evaluation_paths` 只解析并写入隔离 manifest，不进入 cache、optimizer、损失校准或 checkpoint 选择。

## 当前生命周期

```text
HandBank geometry semantics
    -> CPU POE spec + strict owner Manifold union
    -> boundary-only home samples + radial-decay palm anchors + owner triangle sampling tables
    -> CPU catalog + bounded GPU resident asset window + reusable Warp BVH leases
    -> scrambled Sobol q within joint limits
    -> online 50/25/25 anchor-workspace / current owner-shell / owner-adjacent queries
    -> explicit sigma-conditioned online d / rho / kappa / g teacher
    -> optional 20-JOINT / 26-owner heterogeneous padding
    -> retained encoder + SSL-only density/kappa decoders
    -> fixed train-only gradient calibration
    -> density + kappa + derived-g + Sobolev + chain + paired parity loss
    -> deterministic full resume + retained-only encoder state
```

静态 cache 每项资产只物化一次；physical anchors 固定，workspace offsets、current-surface shell/adjacent 与 sigma 在每个同资产 q 子批次在线重采。在线主路径必须使用 GPU Warp，失败即报错，不自动回退 trimesh。CPU trimesh/manifold3d 只负责离线 Boolean、闭合性验证、surface/anchor sampling table 和 reference truth。

## Input 与 Target 分离

retained encoder 只读取部署可获得的当前物理 $q$ 与静态手型证据：显式 $q_{home}$、ordered screws、topology graph、PALM/JOINT/TIP home boundary samples、palm normal 与完整无序 physical anchors。joint limits 只用于 Sobol q 采样，不进入 encoder。

以下信息不得进入 retained encoder：current distance、closest point、surface Jacobian、query stratum、target field、contact、action、history、object state 或 future state。training-only decoder 读取 detached `{h}` query coordinates、显式 sigma 与 retained latent，SSL 后整体删除。

## 隐式 Gaussian 主线

逐 owner unsigned distance 为 $d_g(x;q)$，物理带宽为 $\sigma_\ell$，零阶 target 为：

$$
\rho_{\sigma_\ell,g}(x;q)=\exp\left[-\frac{d_g(x;q)^2}{2\sigma_\ell^2}\right].
$$

一阶距离灵敏度为 $\kappa_{g,i}=\partial d_g/\partial q_i$，单位 m/rad；链式 field target 为：

$$
g_{\sigma_\ell,g,i}=-\frac{d_g}{\sigma_\ell^2}\rho_{\sigma_\ell,g}\kappa_{g,i},
$$

单位 $\mathrm{rad}^{-1}$。联合目标同时训练 density、显式 $\kappa$、由预测 $\rho/\kappa$ 派生的 $g$、同一 density predictor 对物理 q 的 Sobolev 自导数，以及两条预测灵敏度路径的 chain consistency。

训练 sigma 中心为 4/16/64 mm，并施加 log-space 有界 ±10% jitter；validation 固定使用 4/8/16/32/64 mm。decoder 对每个 `(owner,query,sigma)` 输出一个 scalar，$N_Q$ 与 $N_\sigma$ 都只是数据轴。每 owner query 默认 64 个，按 32 workspace、16 owner-shell、16 adjacent 分解。模型默认 padding 上限为 20 JOINT、5 TIP、26 owner；无效槽由 entity/joint/field/edge masks 屏蔽，不具有可学习 identity。

## Validation 与诊断

validation generated assets 必须与 train 按 `physical_geometry_hash` 整组隔离；limit-only configuration domains 不得跨 split。held-out morphology 使用固定 Sobol q/query/teacher bank，并按 morphology、bin、axis、metric 依次等权选择 best checkpoint；训练形态另有独立 q bank，在初始化和最终模型上确定性流式重放并核对完整 teacher SHA-256。固定 ablation 包含 query-only、同手跨 q/跨手 latent shuffle，以及 $z^{(1)}$ 置零、跨 JOINT 打乱和符号翻转。

## Checkpoint 与运行证据

默认输出：

```text
logs/geometry_ssl/<experiment>/<UTC timestamp>/
├── resolved_config.yaml
├── asset_manifest.yaml
├── tensorboard/
├── metrics.jsonl
├── runtime.jsonl
├── loss_calibration.yaml
├── checkpoint_selection.yaml
├── training_morphology_q_bank.yaml
├── validation_ablations.yaml
├── validation_ablation_analysis.yaml
├── train_dense_step_*.npz
├── validation_dense_step_*.npz
└── checkpoints/{step_*.pt,best_step_*.pt,best.pt,last.pt}
```

checkpoint 保存完整模型、optimizer、step、代码/包/资产 schema、resolved config 与 split manifest；`retained_state` 只含 `encoder.` namespace。迁入 PPO/IL 时严格报告 missing/unexpected keys，density/sensitivity decoders 不进入部署图。

`metrics.jsonl` 保存 asset/q provenance、六项 loss 及 numerator/denominator；`runtime.jsonl` 保存 resident asset/owner/triangle、load/release 时间、设备 memory delta、训练 peak memory 与 q/s。NPZ 保存 $Z^{(0)}$、$Z^{(1)}$、query stratum、owner role、bandwidth、distance shell、ancestor selectors 与 dense error。selection evidence 按 owner role、50:25:25 stratum、bandwidth、distance shell、ancestor/non-ancestor 分层；ablation analysis 使用 asset/q 两级 paired bootstrap。

## 当前边界

- `pretrain.py` 只保留 Hydra CLI façade；`runtime/assets.py`、`scheduler.py`、`objective.py`、`validation.py`、`checkpointing.py`、`trainer.py` 分别拥有运行时职责，`runtime/__init__.py` 只导出稳定接口。
- `experiments/canonical_residual_family.py` 是正式 21-asset same-topology pilot 的声明式配置；20-epoch sustained run 尚未执行，当前 artifact 只证明 runner 与诊断闭环。
- 正式 generated 广混合 manifest、topology/family/handedness 分层抽样、官方 sidecar 实例和 importer pose smoke 尚未闭合。
- AMP/distributed 尚未接线；resume CLI 已严格校验 scientific config/manifest，并恢复 epoch/window/Sobol/RNG/initial baseline/historical best lineage。
- Isaac Sim 相关证据未来放在 `source/anymani/anymani/smokes/distill/`，不进入默认 pytest。

## 相关目录

- 物理 target：[`../representations/`](../representations/README.md)
- learnable components：[`../models/`](../models/README.md)
- reconstruction loss：`../objectives/representations/`
- recording/evaluation：[`../diagnostics/`](../diagnostics/README.md)
- PPO 生命周期：沿用当前 tracked `anymani.distill.train/play` 与 task/agent 配置；本目录不拥有 RL 入口迁移
