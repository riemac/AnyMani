# Material-point Jacobian Autoresearch

本目录保存可复现的研究 probe，不是正式 SSL 训练入口。当前主题是 fixed-material anchor-relational Jacobian：owner-local home-surface material point 随 POE/FK 运动后，相对固定 PALM anchor constellation 的四通道关系导数。正式物理公式位于 `source/anymani/anymani/distill/representations/targets/material_point_jacobian.py`。

## Probe 顺序

### `armj_teacher_probe.py`

执行无网络物理审计：解析/中心差分、joint-coordinate sign、共同 SE(3)、reflection/chirality、structural zero、anchor measurement condition 与 target-only throughput。大规模 AR-MPJ-001 使用 64 assets × 2 q、每 edge 4 个固定 material points。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/armj_teacher_probe.py \
  --assets 64 --q-per-asset 2 --assets-per-minibatch 8 \
  --points-per-edge 4 --gauge-assets 8 \
  --output logs/autoresearch/material_point_jacobian/AR-MPJ-001/report_64x2.json
```

### `armj_tiny_overfit.py`

在一个 8-asset fixed bank 上比较 width-64/layers-2 full Z-conditioned 模型与 query-only 对照；可选 same-asset q shuffle 和 cross-asset latent shuffle。该脚本只验证可优化性与 latent 依赖，不产生跨 morphology 结论。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/armj_tiny_overfit.py \
  --assets 8 --q-per-asset 2 --points-per-edge 1 \
  --full-updates 1000 --query-only-updates 300 \
  --output-dir logs/autoresearch/material_point_jacobian/AR-MPJ-003-ablations
```

### `armj_gradient_coherence.py`

加载 AR-MPJ-003 模型，在独立 morphology batches 上计算全部 retained encoder 参数的总 objective 与逐 relation-channel 梯度。该结果用于和 N020 κ 的 cross-batch gradient coherence 做同层级比较。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/armj_gradient_coherence.py \
  --assets 128 --q-per-asset 2 --assets-per-batch 8 --skip-batches 1 \
  --checkpoint logs/autoresearch/material_point_jacobian/AR-MPJ-003-ablations/models.pt \
  --output logs/autoresearch/material_point_jacobian/AR-MPJ-004/report.json
```

### `armj_small_generalization.py`

构造 64 variant train、32 disjoint variant validation 与 32 unseen-mother test fixed banks，比较 full 与 query-only 的小规模跨 morphology 泛化。固定 bank 结果用于决定是否进入 online medium gate，不替代正式 online 训练。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/armj_small_generalization.py \
  --train-assets 64 --validation-assets 32 --mother-assets 32 \
  --q-per-asset 4 --full-updates 2000 --query-only-updates 1000 \
  --output-dir logs/autoresearch/material_point_jacobian/AR-MPJ-005
```

### `armj_medium_online.py`

执行 width-128/layers-4 中型 online fresh-q 实验。Train 使用前 512 variant assets；validation 使用 variant indices 512–575；external test 使用 64 unseen-mother assets。每个 epoch 完整遍历 train catalog并为每 asset 产生 2 个新 q。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/armj_medium_online.py \
  --train-assets 512 --validation-assets 64 --mother-assets 64 --epochs 16 \
  --assets-per-batch 8 --q-per-asset-per-batch 2 --validation-q-per-asset 4 \
  --checkpoint-every-epochs 4 --seed 20260830 \
  --output-dir logs/autoresearch/material_point_jacobian/AR-MPJ-006
```

## 当前边界

- 所有训练 probe 都是 Material-point Jacobian 单目标，不联合 Gaussian density，不使用 FairGrad。
- Probe 暂时借用 v0.7.5 evaluation session 获取 source、q 和 static evidence，因此 session realization 仍会生成旧 density/κ Warp teacher。报告将这部分时间和新 relation target/model 时间分开；它不是未来正式单目标 method 的真实吞吐。
- Material identity 固定为 owner-local home-surface point。不得换成每个 q 重新查询的 closest point，否则 anchor-distance derivative 会包含未建模的 tangent/barycentric drift。
- Anchor 轴是可变长度无序集合。Reader 必须对每个实际 anchor 共享参数，不能把存储下标当作稳定通道身份。
- Logs 位于 `logs/autoresearch/material_point_jacobian/`，不属于 Git 源码。研究判断同步记录在 ignored working docs `docs/joint-coordinate-gauge-policy/`。

## Formal Method Admission

### `density_gamma_method_smoke.py`

使用 `--config` 指定正式 v0.8.0 或 N040 v0.8.1 method，在真实 held-out sources 上闭合 density-only teacher、Gamma target、联合 forward、FairGrad backward 与参数分组；`--compile` 额外验证 fullgraph BF16 路径。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/density_gamma_method_smoke.py \
  --config geometry_ssl_density_material_jacobian_se3_v0_8_1 --compile
```

### `density_gamma_streaming_parity.py`

比较同一 16-assets × 2-q realization 的完整 32-row batch 与两个 16-row streaming units。FP32 用于验证数学 parity，BF16 差异单独解释为 GEMM reduction precision envelope。

### `n031_frame_gauge_audit.py`

在 64 held-out morphologies × 2 q 上共同重写 points、palm normal 与 spatial screws，分别测量 SO(2)、origin translation 和 arbitrary proper-SE(3) 下的 home features、screw features 与 Z parity。`--encoder legacy` 复现 N031 origin 泄漏；`--encoder se3` 把同一 N031 encoder state 迁入 N040 line frontend，隔离架构变化本身。

```bash
/home/hac/isaac/env_isaaclab/bin/python scripts/research/n031_frame_gauge_audit.py --encoder se3
```
