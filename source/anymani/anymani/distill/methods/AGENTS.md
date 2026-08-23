# AGENTS.md

`methods/` 是 task-free 科学方法聚合根，对内组合 representation、model 与 objectives，对外向 SSL trainer 提供封闭接口。catalog、optimizer、resident window 和 MDP 分别由 Data、Trainer、runtime 与 task 层维护。

## Project Structure

```text
methods/
├── contracts.py
│   EmbodimentMethod、MethodSplitSession、MethodEvaluationReport
└── multi_anchor_gaussian_implicit_field/
    ├── __init__.py          绑定五项 ClassVar func
    ├── config.py            方法装配、ObjectiveTermCfg、五项权重
    ├── method.py            prepare / realize / forward / reduce / evaluate / export
    ├── evaluation.py        固定 bank digest 与配对 latent ablation
    ├── artifact.py          retained encoder 的严格下游 loader
    ├── provenance.py        physical realization 与 split-isolation 证据
    ├── batch.py             选 A^(k)、evidence、跨结构 padding、三块视图
    ├── context.py           derived-field 与 density q-JVP 各算一次
    ├── objectives.py        density / kappa / derived-field / Sobolev / chain
    ├── augmentation.py      20% 单 JOINT 符号改写
    └── state_measure.py     完整 joint-limit scrambled Sobol
```

当前 concrete method 是多锚点 Gaussian 隐式场。新方法以各自聚合根实现，只共享 `contracts.py` 中的窄协议。

## Development Style And Conventions

### 封装

Trainer 只调用 `prepare`、`open_session`、`forward_objectives`/可选流式 `backward_update`、`reduce_update`、`evaluate_session`、完整 state 保存/恢复、retained artifact 与 `close`。流式入口必须保持完整 accumulation group 的 additive-statistic denominator，只允许改变 autograd 图生命周期；sources、Sobol cursor、resident loader、具体 batch/model、固定 sigma 和 ablation 均封在 concrete method/session 内。

### 配置

`ObjectiveTermCfg.func` 使用 `ClassVar`，因此不进入 OmegaConf。padding 由 `prepare()` 根据 resolved 资产的实际最大 JOINT/TIP 与 backbone 图距离推导；输入超出已推导范围时报告配置错误。

## Important Semantics

### Batch 三块

batch 的三块信息流如下，模型前向只消费前两块，truth 进入 objective：

- `model_input`：`q`、anchors、home、screws、graph、masks
- `readout_condition`：query、sigma、edge selectors
- `truth`：distance/density/`κ`/`g`、物理有效、active/zero、provenance

`StaticGeometryEvidence` 留在 model。`build_static_geometry_evidence()`、padding 和选 `A^(k)` 属于 `batch.py`。

### 五项损失与采样

每资产 8 套独立 anchor bank；同资产 q-block 共享并轮换；validation / independent q-bank / PPO 固定 `A^(0)`。`q` 来自完整 joint-limit Sobol，采样器保持原始构型测度。query 50/25/25；训练 sigma 4/16/64 mm ±10% jitter；validation 关闭 jitter。每有效 JOINT：train `1+1`，validation `4+4`。ancestor mask 只用于监督归约。

主损失五项；paired 不是主损失。joint-sign rewrite：每个 `(asset,q)` 以 0.20 概率恰好翻一个有效 JOINT；density/distance 不变，对应 `κ/g` 翻号。归约按 `(asset,q)` 等权；active/zero 先分别平均再 `1:1`。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/methods
pyright source/anymani/anymani/distill/methods
```
