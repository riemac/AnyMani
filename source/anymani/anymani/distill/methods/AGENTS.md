# AGENTS.md

`methods/` 是 task-free 科学方法聚合根。对内耦合 representation、model 与 objectives；对外只给 SSL trainer 封闭接口。不拥有 catalog、optimizer、resident window 或 MDP。

## Project Structure

```text
methods/
├── contracts.py
│   EmbodimentMethod、FeatureSpec、MethodStep、MethodUpdate
└── multi_anchor_gaussian_implicit_field/
    ├── __init__.py          绑定五项 ClassVar func
    ├── config.py            方法装配、ObjectiveTermCfg、五项权重
    ├── method.py            prepare / realize / forward / reduce / evaluate / export
    ├── batch.py             选 A^(k)、evidence、跨结构 padding、三块视图
    ├── context.py           derived-field 与 density q-JVP 各算一次
    ├── objectives.py        density / kappa / derived-field / Sobolev / chain
    ├── augmentation.py      20% 单 JOINT 符号改写
    └── state_measure.py     完整 joint-limit scrambled Sobol
```

当前唯一 concrete method 是多锚点 Gaussian 隐式场。不要建立带 `representation/model/objectives` 字段的万能基类。

## Development Style And Conventions

### 封装

Trainer / lifecycle / evaluation 只能调用 `prepare`、`initialize_samplers`、`make_independent_samplers`、`realize_minibatch`、`forward_objectives`、`reduce_update`、`evaluate`、`feature_spec`、`retained_state_dict`、`close`。禁止读 `method.representation`、直接改 sigma、直接拿 padding layout，或在 trainer 里 `new SobolJointSampler`。

### 配置

`ObjectiveTermCfg.func` 必须是 `ClassVar`，不进入 OmegaConf。padding 由 `prepare()` 从 resolved 资产的实际最大 JOINT/TIP 与 backbone 图距离推导；超出则失败。

## Important Semantics

### Batch 三块

模型不得读 truth：

- `model_input`：`q`、anchors、home、screws、graph、masks
- `readout_condition`：query、sigma、edge selectors
- `truth`：distance/density/`κ`/`g`、物理有效、active/zero、provenance

`StaticGeometryEvidence` 留在 model。`build_static_geometry_evidence()`、padding 和选 `A^(k)` 属于 `batch.py`。

### 五项损失与采样

每资产 8 套独立 anchor bank；同资产 q-block 共享并轮换；validation / independent q-bank / PPO 固定 `A^(0)`。`q` 是完整 joint-limit Sobol，无在线自碰撞筛选。query 50/25/25；训练 sigma 4/16/64 mm ±10% jitter；validation 关闭 jitter。每有效 JOINT：train `1+1`，validation `4+4`。不把 ancestor mask 喂给模型。

主损失五项；paired 不是主损失。joint-sign rewrite：每个 `(asset,q)` 以 0.20 概率恰好翻一个有效 JOINT；density/distance 不变，对应 `κ/g` 翻号。归约按 `(asset,q)` 等权；active/zero 先分别平均再 `1:1`。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/methods
pyright source/anymani/anymani/distill/methods
```
