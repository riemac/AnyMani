# AGENTS.md

`ssl/` 负责 task-free embodiment pretraining 的声明组合、Trainer 生命周期、通用 checkpoint 容器与 evidence。它消费 `assets` 的 typed semantics 和 `distill.methods` 的封闭接口；URDF/`hand.yaml`、物理 teacher、固定评估测度、task MDP 与 Isaac Sim 生命周期继续由各自模块维护。

## Project Structure

```text
ssl/
├── experiment.py                    schema 7 pure-pretrain 根配置与 façade
├── post_training.py                 独立 validation/evaluation 根配置与 façade
├── config_store.py                  注册完整 Python 实验，不解析字段
├── contracts.py                     runtime_type 装配
├── pretrain.py                      calibration / pure-pretrain CLI
├── validate.py                      显式 checkpoint validation CLI
├── evaluate.py                      显式 checkpoint evaluation CLI
├── checkpoint.py                    通用 method/optimizer/trainer state 容器
├── experiments/
│   └── multi_anchor_gaussion_implicit_field.py   完整 Python 装配；文件名保留历史拼写
├── data/
│   └── hand_assets.py               只解析固定 dataset partitions
└── runtime/
    ├── sampling.py                  显式训练批数与独立固定评估 q-bank 日程
    ├── scheduler.py                 ResidentGeometryAssetWindow
    ├── lifecycle.py                 calibrate / pure pretrain / resume
    ├── post_training.py             fixed-bank validation / evaluation
    └── checkpointing.py             resume identity 与 checkpoint alias
```

Hydra 只从 ConfigStore 加载 `experiments/multi_anchor_gaussion_implicit_field.py` 的完整 pretrain/validation/evaluation 根配置。资产 YAML 只存在于 `assets/datasets/`。`EmbodimentPretrainCfg` 组合 `data / method / trainer / run`，事后阶段分别组合 `data / method / stage / run`。

## Development Style And Conventions

### Role 边界

Data 交付 assets、partitions、provenance。Method/session 拥有 realization、合法 microbatch 切分、三项 objective、固定评估测度、ablation、完整 state 和 retained export。Trainer 只拥有 epoch、新 minibatch 数、mini-epoch 遍历、optimizer 与训练 checkpoint。独立 validation/evaluation runtime 拥有 fixed-bank 编排，Run 拥有路径、seed 和 lineage。

### Checkpoint

新训练在首次 update 前保存 `epoch_000000.pt`，之后只在完整 epoch 结束后的 optimizer boundary 保存。payload 覆盖 schedule cursor、每资产 Sobol cursor、optimizer、预算计数、CPU/CUDA RNG 与轻量 dataset identity；`last.pt` 硬链接最终 immutable epoch checkpoint。full checkpoint 服务 SSL resume/事后评估；Method 保留 schema-4 retained payload builder，但当前尚无独立导出 CLI。

## Important Semantics

### Resident window

训练 catalog 按 seed 确定性打乱并组成完整资产批；走完 catalog 后重新打乱，直到生成 `max_epochs × num_minibatches` 个新批。epoch 不表示 catalog 遍历。resident window 只控制设备资产状态与 Warp lease，不改变 minibatch、microbatch、资产顺序或统计预算。

### 训练合同

Trainer 面向 Method/session 窄接口编排数据和更新，owner/query/edge 轴留在 Method 内。一个 epoch 先生成 `num_minibatches` 份新 teacher 数据，再由 `mini_epochs` 次确定性重排遍历；每个 `assets_per_minibatch × q_per_asset_per_minibatch` minibatch 独立执行 optimizer update，`microbatch_size` 只在该批内部累计梯度。calibration 强制 `mini_epochs=1`。pure pretrain 不调用 validation、evaluation、q-bank、physical audit、best selection 或 retained export。

official assets 当前仅保留为独立评估角色。train/预实验只消费 train partition；独立 checkpoint selection 消费 validation suites。显式事后 physical audit 按 `content_hash` 与 `physical_geometry_hash` 核验所有 split 隔离。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain
python -m anymani.distill.ssl.pretrain --phase calibrate_objectives --max_epochs 32 --num_minibatches 4 --assets_per_minibatch 64 --q_per_asset_per_minibatch 8 --mini_epochs 1 --microbatch_size 64 --seed 20260813
python -m anymani.distill.ssl.validate --baseline_checkpoint <epoch_000000.pt> --checkpoint <epoch_000032.pt>
python -m anymani.distill.ssl.evaluate --checkpoint <checkpoint.pt>
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp。跨手型泛化结论以正式 pilot、unseen-suite evaluation 和 PPO transfer 证据为准。
