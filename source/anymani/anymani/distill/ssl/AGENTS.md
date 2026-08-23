# AGENTS.md

`ssl/` 负责 task-free embodiment pretraining 的声明组合、Trainer 生命周期、通用 checkpoint 容器与 evidence。它消费 `assets` 的 typed semantics 和 `distill.methods` 的封闭接口；URDF/`hand.yaml`、物理 teacher、固定评估测度、task MDP 与 Isaac Sim 生命周期继续由各自模块维护。

## Project Structure

```text
ssl/
├── experiment.py                    schema 5 EmbodimentPretrainCfg 与唯一 façade
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

Data 交付 assets、partitions、provenance。Method/session 拥有 realization、五项 objective、固定评估测度、ablation、完整 state 和 retained export。Trainer 拥有显式新 minibatch 数、mini-epoch 数据复用、accumulation、optimizer、validation promotion 与 final-evaluation 编排。Run 拥有路径、`phase`、resume seed 和 lineage。

### Checkpoint

checkpoint 在完整 mini-epoch 组结束后的 optimizer boundary 保存，payload 覆盖 schedule cursor、每资产 Sobol cursor、optimizer、selection 与 CPU/CUDA RNG。full checkpoint 服务 SSL resume；IL/PPO 消费 standalone retained artifact，其中包含 encoder cfg/state、输入合同、`FeatureSpec` 与 lineage。

## Important Semantics

### Resident window

训练 catalog 按 seed 确定性打乱并组成完整资产批；走完 catalog 后重新打乱，直到生成 `num_minibatches` 个新批。resident window 打包整数个完整训练批，因此设备容量不改变资产顺序或统计预算。驱逐同时释放 Warp lease 和对应 device state 引用。

### 训练合同

Trainer 面向 Method/session 窄接口编排数据和更新，owner/query/edge 轴留在 Method 内。预实验与正式实验复用唯一 `num_minibatches / mini_epochs / sampling` 接口，每次运行可覆盖不同 preset。每组新数据 realization 一次并循环使用 `mini_epochs` 次；预实验执行 forward/JVP 并累计五项，正式实验每遍 backward/update。当前首个 8192-asset preset 为 `128 minibatches × 64 assets × 8 q × 5 mini-epochs`。validation 用三项重建指标选 best；final evaluation 读取冻结 best。

official assets 当前仅保留为独立评估角色。train、预实验和 checkpoint selection 使用 dataset 声明的 train/validation partitions，split 按 `content_hash` 与 `physical_geometry_hash` 隔离。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain
python -m anymani.distill.ssl.pretrain --phase calibrate_objectives --num_minibatches 128 --assets_per_minibatch 64 --q_per_asset_per_minibatch 8 --mini_epochs 5 --seed 20260813
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp。跨手型泛化结论以正式 pilot、unseen-suite evaluation 和 PPO transfer 证据为准。
