# AGENTS.md

`methods/` 是 task-free 科学方法聚合根，对内组合 representation、model 与 objectives，对外向 SSL trainer 提供封闭接口。catalog、optimizer、resident window 和 MDP 分别由 Data、Trainer、runtime 与 task 层维护。

## Project Structure

```text
methods/
├── contracts.py
│   EmbodimentMethod、MethodSplitSession、MethodEvaluationReport
└── multi_anchor_gaussian_implicit_field/
    ├── __init__.py          绑定 rho/kappa 两项 ClassVar func
    ├── config.py            方法装配与固定双 objective
    ├── method.py            stable method façade 与 prepare/realize/forward/export
    ├── source_runtime.py    lazy sources、artifact/preflight、session 与 physical audit
    ├── training.py          microbatch backward、FairGrad、private gradients、diagnostics
    ├── evaluation.py        fixed evaluation、bank digest、ablation、PCA replay
    ├── artifact.py          retained encoder 的严格下游 loader
    ├── provenance.py        physical realization 与 split-isolation 证据
    ├── batch.py             选 A^(k)、evidence、跨结构 padding、三块视图
    ├── context.py           density/kappa typed objective context
    ├── objectives.py        双项公式、teacher baseline 与固定归一化
    ├── augmentation.py      20% 单 JOINT 符号改写
    └── state_measure.py     完整 joint-limit scrambled Sobol
```

当前 concrete method 是多锚点 Gaussian 隐式场。新方法以各自聚合根实现，只共享 `contracts.py` 中的窄协议。

## Development Style And Conventions

### 封装

Trainer 只调用 `prepare`、`open_session`、`forward_objectives`/可选流式 `backward_update`、`reduce_update`、`evaluate_session`、完整 state 保存/恢复、retained artifact 与 `close`。流式入口必须保持完整 optimizer minibatch 的 additive-statistic denominator，只允许改变 autograd 图生命周期；sources、Sobol cursor、resident loader、具体 batch/model、固定 sigma 和 ablation 均封在 concrete method/session 内。

### 配置

`ObjectiveTermCfg.func` 使用 `ClassVar`，因此不进入 OmegaConf。padding 由 `prepare()` 根据 resolved 资产的实际最大 JOINT/TIP 与 backbone 图距离推导；输入超出已推导范围时报告配置错误。

## Important Semantics

### Batch 三块

batch 的三块信息流如下，模型前向只消费前两块，truth 进入 objective：

- `model_input`：`q`、anchors、home、screws、graph、masks
- `readout_condition`：query、sigma、edge selectors
- `truth`：distance/density/`κ`/`g`、物理有效、active/zero、provenance

`StaticGeometryEvidence` 留在 model 的 `input_adapters/evidence.py`。`build_static_geometry_evidence()`、padding 和选
`A^(k)` 属于 `batch.py`；method 通过 façade 调用，不把这些实现复制到 Trainer。

### 双项损失与采样

每资产 8 套独立 anchor bank；同资产 q-block 共享并轮换；validation / independent q-bank / PPO 固定 `A^(0)`。`q` 来自完整 joint-limit Sobol，采样器保持原始构型测度。query 50/25/25；训练 sigma 4/16/64 mm ±10% jitter，validation 关闭 jitter。每个有效 JOINT、每个 q 固定 `2 active + 1 structural-zero`。ancestor mask 只用于监督归约。

run-local teacher baseline 只负责训练后归一化曲线，不进入 optimizer。density 使用 raw MSE，κ 使用固定 `0.1 m/rad` 无量纲残差；每个有效 JOINT、每个 q 固定 `2 active + 1 structural-zero`，active/zero 按 2:1 归约。joint-sign rewrite 每个 `(asset,q)` 以 0.20 概率翻一个有效 JOINT；density/distance 不变，对应 `κ/g` 翻号。训练按 64 个样本建立梯度图，以完整 512-pair denominator 分块反向；shared encoder 使用精确两任务 FairGrad，density/kappa readers 各自使用 private gradient，三个参数组分别裁剪。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/methods
pyright source/anymani/anymani/distill/methods
```
