# AGENTS.md

`ssl/` 拥有 task-free embodiment pretraining 的声明组合、在线生命周期、evaluation、checkpoint 与 evidence。消费 `assets` 的 typed semantics 与 `distill.methods` 的封闭接口。不解析 URDF/`hand.yaml`，不拥有物理 teacher、task MDP 或 Isaac Sim 生命周期。

## Project Structure

```text
ssl/
├── experiment.py                    schema 4 EmbodimentPretrainCfg 与唯一 façade
├── config_store.py                  注册完整 Python 实验，不解析字段
├── contracts.py                     runtime_type 装配
├── pretrain.py                      python -m CLI
├── checkpoint.py                    full resume 与 standalone retained artifact
├── experiments/
│   └── multi_anchor_gaussion_implicit_field.py   完整 Python 装配；文件名保留历史拼写
├── data/
│   └── hand_assets.py               只解析固定 dataset partitions
└── runtime/
    ├── sampling.py                  window-major OnlineMinibatchSchedule
    ├── scheduler.py                 ResidentGeometryAssetWindow
    ├── lifecycle.py                 calibrate / pretrain / resume / export
    ├── validation.py                固定 bank、独立 q-bank、ablation
    ├── checkpointing.py             resume 科学配置核对
    └── evaluation.py                selection 协议
```

同级 `distill/presets/ssl/canonical_multi_anchor_gaussian.yaml` 只保留旧 CLI 名称。`EmbodimentPretrainCfg` 只组合 `data / method / trainer / evaluation / run`。不要恢复分片 YAML、集中式 parser、递归 `_target_` 或万能 registry。

## Development Style And Conventions

### Role 边界

Data 交付 assets、partitions、provenance。Method 拥有 realization、五项 objective、retained export。Trainer 拥有 window-major schedule、Sobol coverage、尾组、accumulation、optimizer。Evaluation 拥有 fixed bank、selection、独立 q-bank、ablation。Run 拥有路径、`phase`、resume seed、lineage。

### Checkpoint

只在 optimizer boundary 保存。payload 覆盖 schedule cursor、每资产 Sobol cursor、optimizer、selection、CPU/CUDA RNG。full checkpoint 只服务 SSL resume。IL/PPO 只能读 standalone retained artifact：encoder cfg/state、输入合同、`FeatureSpec`、lineage。

## Important Semantics

### Resident window

每个 epoch 打乱一次资产；同一窗完成全部 q coverage 后再切窗。驱逐必须释放 Warp lease，且不得保留已驱逐 device state 的外部强引用。

### 训练合同

Trainer 不解释 owner/query/edge 轴，也不读取 `method.representation`。`run.phase` 为 `calibrate_objectives` 或 `pretrain`。calibration 用同一 façade 和正式 `ssl.yaml`，算全部五项，不 `optimizer.step`，不自动改权重。pretrain 必须 fail-closed 核对 artifact 的 manifest hash、公式身份、method/model/representation、采样语义和 code revision。当前 `20 × 256` 不是已批准的 8192-asset 正式预算。

official assets 不参与 train、calibration 或 checkpoint selection。split 按 `content_hash` 与 `physical_geometry_hash` 隔离。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain
python -m anymani.distill.ssl.pretrain run.phase=calibrate_objectives
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp。没有正式 pilot、unseen-suite evaluation 或 PPO transfer 时，不得声明跨手型泛化成立。
