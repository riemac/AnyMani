# AGENTS.md

`representations/` 拥有物理真值：source、field、query、target、mask 与 provenance。这是科研分类，不是强制插件接口。不 import `torch.nn` 或 `distill.models`，不持有 checkpoint，不写 loss 权重。

## Project Structure

```text
representations/
├── geometry.py                      GeometryRepresentation：未 padding 的物理 teacher
├── sources/
│   ├── kinematics.py                POE / FK / Jacobian / owner graph
│   ├── collision_geometry.py        owner union、home/anchor、Warp lease
│   └── geometry_source.py           GeometrySourceCore / anchor finalize / CPU-GPU 生命周期
├── fields/
│   ├── density.py                   ρ = exp(-d² / 2σ²)
│   └── distance.py                  unsigned distance
├── queries/
│   └── spatial_sampling.py          50/25/25 workspace / shell / adjacent
└── targets/
    ├── geometry_field.py            Warp teacher、joint-first edges、sigma
    ├── field_samples.py             FieldTargetBatch / SensitivityTargetBatch
    └── warp_surface.py              closest-surface backend
```

`GeometryRepresentation` 只物化 CPU core、为当前 device subwindow finalize anchor/source、上传 surface cache，并按当前 `q` 生成物理 teacher。不构造 encoder evidence，不做跨结构 padding。

## Development Style And Conventions

### 所有权

`sources/` 只 lower `HandContainer.geometry_semantics`，不重读 `hand.yaml`/URDF。`GeometrySourceCfg.anchors` 使用 nested `AnchorBankCfg(bank_size=8, ...)`。`anchors` 保存 canonical `A^(0)`，`anchor_bank` 保存全部 realization。

`targets/` 只写物理标签与有效性。不要在这里写 loss 权重或 `(asset,q)` 归约。选 `A^(k)`、`build_static_geometry_evidence()` 和 padding 属于 `methods/.../batch.py`。`StaticGeometryEvidence` 留在 model。

### 候选目录

`fields/`、`queries/bps.py`、`targets/fk_points.py` 中的非主线文件是目录分类或后续消融，不表示 trainer 可自动选用。

## Important Semantics

### 物理对象

PALM/JOINT/TIP owner 的条件 Gaussian 邻近场。`q` 单位 rad，长度 m，frame `{h}`。`ρ` 无量纲，`κ` 为 m/rad，`g` 为 rad⁻¹。home-surface 每 owner 64 个 boundary 点；interior 只允许进入 anchor 支持。

### 采样与边

query 50/25/25；workspace 在 q-block 内共享，shell/adjacent 随 `q` 重采；一阶边只从 shell 抽。训练 sigma 中心 4/16/64 mm；validation 关闭 jitter 后仍用同一组。joint-first：train `1+1`，validation `4+4`。active 受最近点光滑 mask；zero 不因最近面不光滑而丢弃。不要把 ancestor mask 当作模型输入。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/representations -q
ruff check source/anymani/anymani/distill/representations/geometry.py \
           source/anymani/anymani/distill/representations/sources/geometry_source.py
```

完整 teacher 依赖 CUDA Warp。没有 held-out evaluation 时，不得把 source/query 合同写成泛化结论。
