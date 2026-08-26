# AGENTS.md

`objectives/` 拥有可复用的预测/真值比较合同。具体 rho/kappa Gaussian 场公式与 baseline 归约由 `distill.methods.multi_anchor_gaussian_implicit_field` 拥有。本目录不采样、不持有 optimizer、不做 advantage estimation。

## Project Structure

```text
objectives/
├── contracts.py                     AdditiveStatistic、ObjectiveTermResult
├── representations/
│   ├── gauge_consistency.py         joint-sign 物理坐标改写；不定义 latent parity loss
│   ├── fk_reconstruction.py         候选 FK 重建；非当前主损失
│   └── gaussian_field_reconstruction.py  候选；不得冒充双项主损失
└── rl/                              未来 PPO scalar loss；机制仍归 rl/algorithms
```

统一 $Z$ 不设 paired latent MSE。joint-sign 只在 diagnostics 检查 observable density/κ。

## Development Style And Conventions

Method objective 通过 typed context 读预测与真值，不得回调 representation 重新采样。Trainer 只合并已经按 $(asset,q)$ 归约好的标量；derived-field 解析组合留在 diagnostics evaluation。

## Important Semantics

gauge rewrite 供 method augmentation 与 validation parity audit 使用。当前训练图只对模型参数求普通一阶梯度；物理 $q$ 是输入条件，不建立输入梯度图。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/objectives -q
```
