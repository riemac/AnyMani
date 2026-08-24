# AGENTS.md

`objectives/` 拥有可复用的预测/真值比较合同。具体三项 Gaussian 场公式由 `distill.methods.multi_anchor_gaussian_implicit_field` 拥有。本目录不采样、不持有 optimizer、不做 advantage estimation。

## Project Structure

```text
objectives/
├── contracts.py                     AdditiveStatistic、ObjectiveTermResult
├── representations/
│   ├── gauge_consistency.py         joint-sign 坐标改写与 parity 审计
│   ├── fk_reconstruction.py         候选 FK 重建；非当前主损失
│   └── gaussian_field_reconstruction.py  候选；不得冒充三项主损失
└── rl/                              未来 PPO scalar loss；机制仍归 rl/algorithms
```

paired latent MSE 只用于独立评估，不进入三项主损失。

## Development Style And Conventions

Method objective 通过 typed context 读预测与真值，不得回调 representation 重新采样。derived-field 解析组合由 method context 只算一次。Trainer 只合并已经按 $(asset,q)$ 归约好的标量。

## Important Semantics

gauge rewrite 供 method augmentation 与 validation parity audit 使用。当前训练图只对模型参数求普通一阶梯度；物理 $q$ 是输入条件，不建立输入梯度图。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/objectives -q
```
