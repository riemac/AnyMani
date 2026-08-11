# Distill Diagnostics

`diagnostics` 是 SSL、RL 与 IL 共用的实验记录、固定评估和事后分析边界。源码位于本目录；真实运行产物写入仓库根 `logs/` 或用户显式输出目录，不得反向写回 `source/`。

## 当前实现

| 子目录 | 当前实现 | 不拥有 |
| --- | --- | --- |
| `recording/` | `GeometrySSLRunLogger`：TensorBoard、append-only JSONL、dense NPZ | loss 数学、optimizer、checkpoint 选择 |
| `evaluation/` | query-only、latent-shuffle；可选 Warp/Kaolin distance reference | trainer、run 目录生命周期、官方模型选择 |
| `analysis/` | 只读扩展边界，尚无通用分析实现 | 模型 forward、物理 target、训练状态 |

依赖方向固定为 stage runtime 调用 `recording`/`evaluation`；`analysis` 只读落盘产物。诊断不得反向参与模型输入、target 或 optimizer。

## Geometry SSL Run 目录

```text
logs/geometry_ssl/<experiment>/<UTC timestamp>/
├── resolved_config.yaml             # Hydra/OmegaConf 完整解析配置
├── asset_manifest.yaml              # train/validation/official 内容哈希 split
├── tensorboard/                     # 在线 loss 与梯度范数
├── metrics.jsonl                    # step/split/asset IDs/六项标量
├── train_dense_step_*.npz           # 最后 train batch latent/mask/error
├── validation_dense_step_*.npz      # fixed held-out bank latent/mask/error
└── checkpoints/step_*.pt            # 完整 resume + retained encoder state
```

当前 JSONL 的标量字段是 `total`、`density`、`kappa`、`derived_field`、`sobolev`、`chain`，训练记录另含 clip 前 `gradient_norm`。它不伪装成已经实现的 owner/stratum/shell 多维评估表。

当前 NPZ 保存：asset IDs、`zero_order [B,26,D0]`、`first_order [B,20,D1]`、entity/joint/field/edge validity masks、无量纲 density error 与 m/rad kappa error。被 mask 排除的数据仍保留为数组与 mask，不因不进入 loss 而从证据中消失。

## 显式评估

- `geometry_ssl_ablation_forward(..., ablation="query_only")` 将 $Z^{(0)}$ 与 $Z^{(1)}$ 清零，但保持同一 query encoder/decoder；
- `geometry_ssl_ablation_forward(..., ablation="latent_shuffle")` 只沿 batch 错配 latent，保持 query 与 selectors 不变；
- `compare_warp_and_kaolin_distances(...)` 在安装 Kaolin 0.18.0 后比较同一 GPU query 的 unsigned distance 和 kernel median latency。Kaolin 缺失时明确报错，不改变 Warp teacher 主路径。

跨 run 统计、checkpoint promotion、owner/stratum/shell/bandwidth 分层表和图表仍属于未来 `analysis/`；Research 引用运行结果时应同时锚定代码 revision、resolved config、asset manifest 与必要 checkpoint。
