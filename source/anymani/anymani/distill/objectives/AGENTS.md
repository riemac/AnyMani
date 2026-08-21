# AGENTS.md

`objectives/` 拥有可复用的预测/真值比较合同。具体五项 Gaussian 场公式由 `distill.methods.multi_anchor_gaussian_implicit_field` 拥有。本目录不采样、不持有 optimizer、不做 advantage estimation。

## Project Structure

```text
objectives/
├── contracts.py                     AdditiveStatistic、ObjectiveTermResult
├── representations/
│   ├── field_reconstruction.py      selected_density_coordinate_derivative 原语
│   ├── gauge_consistency.py         joint-sign 坐标改写与 parity 审计
│   ├── fk_reconstruction.py         候选 FK 重建；非当前主损失
│   └── gaussian_field_reconstruction.py  候选；不得冒充五项主损失
└── rl/                              未来 PPO scalar loss；机制仍归 rl/algorithms
```

不要恢复联合六项 `GeometryFieldObjective` 运行时，也不要把 paired 写回主损失。

## Development Style And Conventions

Method objective 通过 typed context 读预测与真值，不得回调 representation 重新采样。derived-field 与 density $q$-JVP 由 method context 各算一次。Trainer 只合并已经按 $(asset,q)$ 归约好的标量。

## Important Semantics

`field_reconstruction.py` 只保留固定 query/sigma 上 $\partial\hat\rho/\partial q_i$ 的抽样边导数，单位 rad$^{-1}$。gauge rewrite 可供 method augmentation 与 validation parity audit 使用，但 paired latent MSE 不是 schema 4 主损失。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/objectives -q
```
