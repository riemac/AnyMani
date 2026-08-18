# AGENTS.md

`ssl` 拥有 geometry representation pretraining 的实验编排、运行时调度、checkpoint 与 evidence；它消费 `assets` 交付的 typed semantics 和 `representations.sources` 的 simulator-independent oracle，不解析 URDF/hand.yaml，也不拥有 task MDP、PPO action 或 Isaac Sim 生命周期。

## 目录所有权

```text
ssl/config/       底层结构化 dataclass 与 Hydra/OmegaConf resolve 合同
ssl/experiments/  每个完整 pilot 的声明式科学配置组合
ssl/runtime/      GeometrySSLExperiment、assets/objective/validation/checkpointing/trainer 与 window/q scheduler
ssl/pretrain.py  Hydra CLI façade，只重建 resolved config 并调用 GeometrySSLExperiment.run()
ssl/checkpoint.py 完整 resume 与 retained encoder transfer
ssl/calibration.py train-only 固定 loss/encoder-gradient calibration
```

## 配置与运行边界

实验模块必须用一个具体 `GeometrySSLExperimentCfg` 子类声明 `asset_dataset_manifest`、representation、model、objective、protocol 与 run identity；资产选择唯一来自 assets 层 dataset YAML，trainer 由 Hydra config group 注入。禁止恢复 leaf 路径常量、builder function、`experiment_configs()` 或自定义 SSL task registry。

`runtime` 是执行器，不拥有研究假设。`GeometrySSLExperiment` 是唯一有副作用入口；`assets.py` 消费 `HandAssetDataset`、物化 train/validation sources 并生成 expanded physical manifest，`scheduler.py` 管理 resident window 与 Sobol/q/epoch cursor，`objective.py` 执行 microbatch 与 accumulation，`validation.py` 执行固定协议，`checkpointing.py` 维护 resume/selection lineage，`trainer.py` 提供私有生命周期内核。GPU window 驱逐必须调用 `representations.sources` 的 Warp lease release。

当前 canonical pilot 是 `experiments/canonical_residual_family.py`：实验根只引用 assets 层 `canonical_cross_mother_v1.yaml`。dataset manifest 以完整 variant set 为配置原子，当前包含 45 train、16 validation、16 unseen-variant-set 与 17 unseen-mother assets；leaf paths、lineage provenance 与 physical hashes 只在 resolve artifact 中展开。

## 训练语义

保留 UDF、50:25:25 query、多带宽 density、sampled `kappa`、derived field、Sobolev、chain 和 paired parity 六项目标。paired sign rewrite 必须同步改写 `(q, q_home, space_screws)`；不要输入裸 sign bit，也不要恢复 `screw_even/screw_odd` 双支。

official 资产不参与 train、calibration、checkpoint selection；variant set 是 dataset 配置原子，但所有 partitions/suites 在 geometry materialization 后仍须按 physical geometry hash 隔离。calibration 只消费固定 train batches，一次测量共享 encoder gradient median；结果写 `loss_calibration.yaml` 与 checkpoint metadata 后冻结，但不得覆盖 declared objective config。

## 验证命令

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
pyright source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp；默认 contract suite 不启动 Isaac Sim。不要把 pilot 结果扩展为 cross-topology、cross-DOF、official zero-shot 或 PPO transfer 结论。
