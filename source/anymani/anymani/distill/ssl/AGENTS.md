# AGENTS.md

`ssl/` 负责 task-free embodiment pretraining 的声明组合、Trainer 生命周期、通用 checkpoint 容器与 evidence。它消费 `assets` 的 typed semantics 和 `distill.methods` 的封闭接口；URDF/`hand.yaml`、物理 teacher、固定评估测度、task MDP 与 Isaac Sim 生命周期继续由各自模块维护。

## Project Structure

```text
ssl/
├── experiment.py                    schema 6 EmbodimentPretrainCfg 与唯一 façade
├── config_store.py                  注册完整 Python 实验，不解析字段
├── contracts.py                     runtime_type 装配
├── pretrain.py                      python -m CLI
├── checkpoint.py                    通用 method/optimizer/trainer state 容器
├── experiments/
│   └── multi_anchor_gaussion_implicit_field.py   完整 Python 装配；文件名保留历史拼写
├── data/
│   └── hand_assets.py               只解析固定 dataset partitions
└── runtime/
    ├── sampling.py                  显式训练批数与独立固定评估 q-bank 日程
    ├── scheduler.py                 ResidentGeometryAssetWindow
    ├── lifecycle.py                 calibrate / pretrain / resume / export
    └── checkpointing.py             resume 与 validation-selection 状态
```

Hydra 只从 ConfigStore 加载 `experiments/multi_anchor_gaussion_implicit_field.py` 的完整 `EXPERIMENT`。资产 YAML 只存在于 `assets/datasets/`。`EmbodimentPretrainCfg` 组合 `data / method / trainer / run`。

## Development Style And Conventions

### Role 边界

Data 交付 assets、partitions、provenance。Method/session 拥有 realization、合法 microbatch 切分、三项 objective、固定评估测度、ablation、完整 state 和 retained export。Trainer 拥有 epoch、新 minibatch 数、mini-epoch 遍历、optimizer、validation promotion 与 final-evaluation 编排。Run 拥有路径、`phase`、resume seed 和 lineage。

### Checkpoint

checkpoint 只在完整 epoch 结束后的 optimizer boundary 保存，payload 覆盖 schedule cursor、每资产 Sobol cursor、optimizer、selection、预算计数与 CPU/CUDA RNG。full checkpoint 服务 SSL resume；IL/PPO 消费 standalone retained artifact，其中包含 encoder cfg/state、输入合同、`FeatureSpec` 与 lineage。

## Important Semantics

### Resident window

训练 catalog 按 seed 确定性打乱并组成完整资产批；走完 catalog 后重新打乱，直到生成 `max_epochs × num_minibatches` 个新批。epoch 不表示 catalog 遍历。resident window 只控制设备资产状态与 Warp lease，不改变 minibatch、microbatch、资产顺序或统计预算。

### 训练合同

Trainer 面向 Method/session 窄接口编排数据和更新，owner/query/edge 轴留在 Method 内。一个 epoch 先生成 `num_minibatches` 份新 teacher 数据，再由 `mini_epochs` 次确定性重排遍历；每个 `assets_per_minibatch × q_per_asset_per_minibatch` minibatch 独立执行 optimizer update，`microbatch_size` 只在该批内部累计梯度。calibration 强制 `mini_epochs=1`。validation 与 checkpoint 只在完整 epoch 边界执行。

official assets 当前仅保留为独立评估角色。train、预实验和 checkpoint selection 使用 dataset 声明的 train/validation partitions，split 按 `content_hash` 与 `physical_geometry_hash` 隔离。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain
python -m anymani.distill.ssl.pretrain --phase calibrate_objectives --max_epochs 32 --num_minibatches 4 --assets_per_minibatch 64 --q_per_asset_per_minibatch 8 --mini_epochs 1 --microbatch_size 64 --seed 20260813
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp。跨手型泛化结论以正式 pilot、unseen-suite evaluation 和 PPO transfer 证据为准。
