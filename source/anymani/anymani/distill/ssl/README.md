# Self-Supervised Geometry Pretraining

`ssl` 是可运行的 task-free physical-representation pretraining stage。它组合 assets/robots 提供的静态手型与运动学、在线 Warp target、retained geometry encoder、SSL-only decoder 和五项联合目标；不拥有资产生成、Isaac Lab MDP、PPO update 或 policy action semantics。

## 运行入口

在仓库根目录激活 AnyMani/Isaac Lab Python 环境后运行；该进程不启动 Isaac Sim：

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain \
  'assets.train_paths=[/absolute/path/to/generated/hand_bundle]'
```

`assets.train_paths` 与 `assets.validation_paths` 接受显式 generated bundle 列表，可同时包含 pre-made mother、post-mutate variants、跨 family 与不同 DOF。`HandBank` 自身支持 `pre_made` discovery 和 `mixed` manifest；当前 SSL CLI 尚未把 collection discovery 作为独立字段暴露。`assets.official_evaluation_paths` 只解析并写入隔离 manifest，不进入 cache、optimizer、损失校准或 checkpoint 选择。

## 当前生命周期

```text
HandBank geometry semantics
    -> CPU POE spec + strict owner Manifold union
    -> boundary-only home samples + palm anchors + fixed workspace bank
    -> GPU spec/evidence + reusable Warp BVHs
    -> scrambled Sobol q within joint limits
    -> 50/25/25 workspace / owner-shell / adjacent queries
    -> online d / rho / kappa / g teacher
    -> optional 20-JOINT / 26-owner heterogeneous padding
    -> retained encoder + SSL-only density/kappa decoders
    -> density + kappa + derived-g + Sobolev + chain loss
    -> full resume checkpoint + retained-only encoder state
```

静态 cache 每项资产只物化一次；每个训练 step 只采 q、当前 query 与 target。在线主路径必须使用 GPU Warp，失败即报错，不自动回退 trimesh。CPU trimesh/manifold3d 只负责离线 Boolean、闭合性验证、surface/anchor sampling 和 reference truth。

## Input 与 Target 分离

retained encoder 只读取部署可获得的当前物理 $q$ 与静态手型证据：显式 $q_{home}$、ordered screws、topology graph、PALM/JOINT/TIP home boundary samples、palm normal 与完整无序 physical anchors。joint limits 只用于 Sobol q 采样，不进入 encoder。

以下信息不得进入 retained encoder：current distance、closest point、surface Jacobian、query stratum、target field、contact、action、history、object state 或 future state。training-only decoder 可以读取固定 `{h}` query coordinates 与 retained latent，SSL 后默认整体删除。

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

默认带宽为 4/12/32/64 mm；每 owner query 默认 64 个，按 32 workspace、16 owner-shell、16 adjacent 分解。模型默认 padding 上限为 20 JOINT、5 TIP、26 owner；无效槽由 entity/joint/field/edge masks 屏蔽，不具有可学习 identity。

## Validation 与诊断

validation generated assets 必须与 train 按 geometry semantics SHA-256 内容哈希隔离。启动时固定一份 validation Sobol q/query/teacher bank，后续重复评估，避免曲线混入采样方差。已有显式诊断包括 query-only、latent-shuffle、真实 mother fixed-batch tiny-overfit，以及可选的 Warp/Kaolin 0.18.0 unsigned-distance 数值/吞吐对照；Kaolin 不属于核心依赖。

## Checkpoint 与运行证据

默认输出：

```text
logs/geometry_ssl/<experiment>/<UTC timestamp>/
├── resolved_config.yaml
├── asset_manifest.yaml
├── tensorboard/
├── metrics.jsonl
├── train_dense_step_*.npz
├── validation_dense_step_*.npz
└── checkpoints/step_*.pt
```

checkpoint 保存完整模型、optimizer、step、代码/包/资产 schema、resolved config 与 split manifest；`retained_state` 只含 `encoder.` namespace。迁入 PPO/IL 时严格报告 missing/unexpected keys，density/sensitivity decoders 不进入部署图。

当前 JSONL 保存 step、split、asset IDs、五项损失、总损失与训练梯度范数。NPZ 保存 $Z^{(0)}$、$Z^{(1)}$、entity/joint/field/edge masks 及 density/$\kappa$ dense error；更细的 owner/stratum/shell/bandwidth 分层 analysis 仍待实现。

## 当前边界

- `config/geometry_ssl.py`、`dataset.py`、`pretrain.py` 与 `checkpoint.py` 已可运行；`experiments/`、`runtime/` 仍是未来 ownership。
- 正式 generated 广混合 manifest、topology/family/handedness 分层抽样、官方 sidecar 实例和 importer pose smoke 尚未闭合。
- AMP/distributed 与 resume CLI 尚未接线；checkpoint load API 已有严格合同测试。
- Isaac Sim 相关证据未来放在 `source/anymani/anymani/smokes/distill/`，不进入默认 pytest。

## 相关目录

- 物理 target：[`../representations/`](../representations/README.md)
- learnable components：[`../models/`](../models/README.md)
- reconstruction loss：`../objectives/representations/`
- recording/evaluation：[`../diagnostics/`](../diagnostics/README.md)
- PPO 生命周期：沿用当前 tracked `anymani.distill.train/play` 与 task/agent 配置；本目录不拥有 RL 入口迁移
