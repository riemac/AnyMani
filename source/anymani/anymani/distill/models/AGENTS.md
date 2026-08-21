# AGENTS.md

`models/` 拥有可学习 adapter、backbone、decoder 与 policy heads。不生成物理 teacher，不写 loss，不定义 MDP。retained encoder 的输入合同是 `StaticGeometryEvidence` + 当前物理 `q`。

## Project Structure

```text
models/
├── geometry_ssl.py                  retained encoder + SSL-only readers 组装
├── input_adapters/
│   ├── geometry.py                  StaticGeometryEvidence、padding、build/pad evidence
│   └── grouped_tokens.py            策略侧 grouped token adapter
├── backbones/
│   ├── geometry_transformer.py      graph-biased encoder-only Transformer
│   └── candidates/                  非主线对照，不得 silently 替换 canonical
├── decoders/representations/
│   └── implicit_field.py            FiLM density reader 与 unbiased κ reader
├── heads/                           PPO action / value / auxiliary
├── policy.py                        EmbodimentPolicy 组装
├── tokens.py / relations.py         token 与关系特征
└── temporal_encoder.py              tactile history；不属于 SSL geometry encoder
```

`build_static_geometry_evidence()` 与跨结构 padding 的调用权在 `methods/.../batch.py`。本目录只定义 encoder 输入类型和实现。

## Development Style And Conventions

### Retained / disposable

SSL 后只保留 `encoder.` namespace。density/sensitivity readers、query backend 与 teacher 必须从 PPO/IL artifact 删除。loader 严格报告 missing/unexpected keys。

### 禁止泄漏

不得读取 current distance、最近点、surface Jacobian、query stratum、contact、object state、action 或 teacher 场标签。joint limits 不进入 encoder。

## Important Semantics

### 表示

$Z^{(0)}\in\mathbb R^{B\times G\times D_0}$ 与 owner 同索引；$z_i^{(1)}\in\mathbb R^{D_1}$ 与活动 JOINT 同索引，是整手场 Jacobian 第 $i$ 列的固定宽度表示，不是对自身 $z_i^{(0)}$ 求导。canonical $D_0=128$、$D_1=64$。physical anchors 是完整、无序、等地位集合；finger seed 只属于采样 provenance。

### Gauge

`{h}` 面内 $SO(2)$ 是 gauge；reflection/chirality 不是。joint-sign 成对改写下 $Z^{(0)}$ 偶，$z_i^{(1)}$、$\kappa$ 与同坐标动作为奇。

### Padding 与性能

跨结构容器可由 method 推导到实际最大 JOINT/TIP；默认实现支持最多 20 JOINT、5 TIP。entity/joint mask 必须屏蔽 attention、bias 与输出。RTX 5070 Ti、`B=4096`、p95 ≤ 40 ms；排除 decoder、policy、Isaac Sim。PPO full fine-tune 不得缓存会 stale 的 learned activation。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/models -q
pytest source/anymani/anymani/distill/tests/performance -m performance -q -s
```

人类阅读入口见 `README.md`。
