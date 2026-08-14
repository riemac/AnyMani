# AGENTS.md

`ssl` 拥有 geometry representation pretraining 的实验编排、运行时调度、checkpoint 与 evidence；它消费 `assets -> robots` 交付的静态语义，不解析 URDF/hand.yaml，也不拥有 task MDP、PPO action 或 Isaac Sim 生命周期。

## 目录所有权

```text
ssl/config/       底层结构化 dataclass 与 Hydra/OmegaConf resolve 合同
ssl/experiments/  每个完整 pilot 的声明式科学配置组合
ssl/runtime/      assets/objective/validation/checkpointing/trainer 与 window/q scheduler
ssl/dataset.py    单资产静态 materialization、在线 query/target 与 padding oracle
ssl/pretrain.py  Hydra CLI façade，只重建 resolved config 并调用 runtime trainer
ssl/checkpoint.py 完整 resume 与 retained encoder transfer
ssl/calibration.py train-only 固定 loss/encoder-gradient calibration
```

## 配置与运行边界

实验模块必须在一个自包含配置函数中声明资产 family、physical split、query/target、model、objective、runtime budget、calibration、validation 和 output identity。允许浅组合现有 dataclass；不要用深继承或 runtime 内部常量隐藏实验选择。

`runtime` 是执行器，不拥有研究假设。`assets.py` 解析 physical split/manifest，`scheduler.py` 管理 resident window 与 Sobol/q/epoch cursor，`objective.py` 执行 microbatch 与 accumulation，`validation.py` 执行固定协议，`checkpointing.py` 维护 resume/selection lineage，`trainer.py` 只编排生命周期；`runtime/__init__.py` 仅导出稳定接口。GPU window 驱逐必须调用 robots 的 Warp lease release。

当前 canonical pilot 是 `experiments/canonical_residual_family.py`：配置显式冻结正式 mother+20 variants 的 21 条 bundle 清单，固定 mother 训练，`split_seed=20260813`，仅解释为 right LEAP、同 topology、16 DOF family 内的构型/形态表征试水。

## 训练语义

保留 UDF、50:25:25 query、多带宽 density、sampled `kappa`、derived field、Sobolev、chain 和 paired parity 六项目标。paired sign rewrite 必须同步改写 `(q, q_home, space_screws)`；不要输入裸 sign bit，也不要恢复 `screw_even/screw_odd` 双支。

official 资产不参与 train、calibration、checkpoint selection；validation 按 physical geometry hash 整组隔离。calibration 只消费固定 train batches，一次测量共享 encoder gradient median，写 `loss_calibration.yaml` 后冻结权重。

## 验证命令

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
pyright source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp；默认 contract suite 不启动 Isaac Sim。不要把 pilot 结果扩展为 cross-topology、cross-DOF、official zero-shot 或 PPO transfer 结论。
