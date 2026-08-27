# AGENTS.md

`models/` 拥有可学习 adapter、backbone、decoder 与 policy heads。不生成物理 teacher，不写 loss，不定义 MDP。retained encoder 的输入合同是 `StaticGeometryEvidence` + 当前物理 `q`。

## Project Structure

```text
models/
├── geometry_ssl.py                  retained encoder + SSL-only readers 组装
├── input_adapters/
│   ├── geometry.py                  legacy compatibility exports
│   ├── evidence.py                  StaticGeometryEvidence、routing、build/stack/pad evidence
│   ├── encoder.py                   SO(2) anchor frontend 与 retained geometry encoder
│   └── grouped_tokens.py            策略侧 grouped token adapter
├── backbones/
│   ├── geometry_transformer.py      graph-biased encoder-only Transformer
│   └── candidates/                  非主线对照，不得 silently 替换 canonical
├── decoders/representations/
│   └── implicit_field.py            density FiLM 与 owner/JOINT-conditioned κ FiLM
├── heads/                           PPO action / value / auxiliary
├── policy.py                        EmbodimentPolicy 组装
├── tokens.py / relations.py         token 与关系特征
└── temporal_encoder.py              tactile history；不属于 SSL geometry encoder
```

`build_static_geometry_evidence()` 与跨结构 padding 的调用权在 `methods/.../batch.py`。`evidence.py` 定义输入类型和静态
routing/padding，`encoder.py` 定义可学习 frontend/backbone；`geometry.py` 只维持历史 import path，不新增实现。

## Development Style And Conventions

### Retained / disposable

SSL 后只保留 `encoder.` namespace。density/sensitivity readers、query backend 与 teacher 必须从 PPO/IL artifact 删除。loader 严格报告 missing/unexpected keys。

### 禁止泄漏

不得读取 current distance、最近点、surface Jacobian、query stratum、contact、object state、action 或 teacher 场标签。joint limits 不进入 encoder。

## Important Semantics

### 表示

$Z\in\mathbb R^{B\times G\times128}$ 直接取 graph-biased Transformer final-norm tokens，并与 PALM/JOINT/TIP owner 同索引。JOINT view 只通过 `joint_entity_index` 从同一 $Z$ gather，不产生第二 latent/head。physical anchors 是完整、无序、等地位集合；finger seed 只属于采样 provenance。

### Gauge

`{h}` 面内 $SO(2)$ 是 gauge；reflection/chirality 不是。joint-sign 只约束 observable density 不变与对应 $\kappa$/动作变号；不得恢复 latent parity loss 或手工 latent sign flip。

### Padding 与性能

跨结构容器可由 method 推导到实际最大 JOINT/TIP；默认实现支持最多 20 JOINT、5 TIP。entity/joint mask 必须屏蔽 attention、bias 与输出。RTX 5070 Ti、`B=4096`、p95 ≤ 40 ms；排除 decoder、policy、Isaac Sim。PPO full fine-tune 不得缓存会 stale 的 learned activation。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/models -q
pytest source/anymani/anymani/distill/tests/performance -m performance -q -s
```

人类阅读入口见 `README.md`。
