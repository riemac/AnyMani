# AGENTS.md

`distill` 拥有学习表征、共享模型、objective 与 SSL/IL/RL 编排。它消费 `assets` 的 typed geometry semantics 与 `tasks` 的环境接口；不解析 `hand.yaml`/URDF，不改写 scene/MDP，不 import `Research/`。

## Project Structure

```text
distill/
├── representations/
│   ├── geometry.py                 物理 teacher 组合根；不 import model
│   ├── sources/                    POE/FK/Jacobian、owner union、home/anchor、Warp lease
│   ├── fields/                     d、ρ、κ、g 的场定义
│   ├── queries/                    50/25/25 workspace/shell/adjacent
│   └── targets/                    物理标签、有效性、active/zero、provenance
├── methods/
│   ├── contracts.py                EmbodimentMethod 窄 Protocol
│   └── multi_anchor_gaussian_implicit_field/
│       ├── method.py               prepare/realize/forward/reduce/evaluate/export
│       ├── batch.py                选 A^(k)、evidence、padding、三块视图
│       └── objectives.py           五项比较公式
├── models/
│   ├── input_adapters/geometry.py  StaticGeometryEvidence 与 retained encoder
│   ├── backbones/                  graph-biased Transformer
│   ├── decoders/representations/   SSL-only density/κ readers
│   └── geometry_ssl.py             retained/disposable 组装
├── objectives/
│   ├── contracts.py                AdditiveStatistic / ObjectiveTermResult
│   └── representations/            JVP 原语、gauge rewrite；五项公式不在这里
├── ssl/
│   ├── experiments/                完整 Python 实验装配
│   ├── runtime/                    sampling、resident window、lifecycle、checkpoint
│   └── pretrain.py                 python -m CLI
├── rl/                             rl_games 入口、YAML、masked PPO
├── il/                             边界占位，尚无 trainer
├── diagnostics/                    recording / evaluation / analysis
├── presets/ssl/                    旧 CLI 名称的最小 YAML
└── tests/                          contracts / integration / performance / training_sanity
```

| 目录 | 拥有 | 不拥有 |
| --- | --- | --- |
| `representations/` | 物理 source/field/query/target | `torch.nn`、padding、loss 权重 |
| `methods/` | 科学聚合根；对外封闭给 trainer | catalog、optimizer、MDP |
| `models/` | adapter、backbone、decoder、policy heads | teacher、loss |
| `objectives/` | 可复用比较合同 | sampling、五项公式 |
| `ssl/` `rl/` `il/` | 各阶段数据流、入口、checkpoint | 共享 trunk 的重复实现 |
| `diagnostics/` | 记录、固定 evaluation、只读分析 | 训练选择或物理真值 |

移动内容时同步 TODO、docstring 和 tests。不要用万能 Manager 或深配置继承隐藏耦合。

## Development Style And Conventions

### 环境与入口

使用 `source ~/isaac/env_isaaclab/bin/activate`。SSL：`python -m anymani.distill.ssl.pretrain`。GM RL：`python -m anymani.distill.rl.train` / `play`。`tasks/inhand` 仍走仓库根 `scripts/rl_games/`。IL 尚未建立 trainer。

### 出清与注释

稳定后删除旧实现、旧字段、旧测试。科研核心文件遵守 `annotation` skill。完整目录 Ruff 的既有债不在触及路径外清理。

### 测试分层

默认 pytest 只跑 `distill/tests/contracts` 与 `integration`。`performance` / `training_sanity` 必须显式路径。Isaac Sim 证据在 `smokes/distill/`，不进本树。spawn/articulation 合同属于 `robots/tests/`。

## Important Semantics

### 信息边界

retained encoder 只读当前物理 `q` 与静态证据。distance、最近点、Jacobian、query stratum、contact、action、history、object state 不得进入。joint limits 只定义采样域。`z_i^(1)` 是整手场 Jacobian 第 `i` 列，不是对自身 `z_i^(0)` 求导。

### 几何 SSL 合同

主线是多锚点条件 Gaussian 场。五项损失：density、κ、derived-field、Sobolev、chain；paired 不是主损失。schema 4：`data / method / trainer / evaluation / run`。Trainer 只调 method 封闭接口，不得读 `method.representation` 或直接改 sigma。full checkpoint 只服务 SSL resume；RL/IL 只消费 standalone retained artifact。

official LEAP/Allegro 不参与 train、calibration 或 checkpoint selection。split 按 `physical_geometry_hash` 隔离；路径、asset ID 或 `content_hash` 不足以识别 limit-only 重复。

### 性能门槛

RTX 5070 Ti、`B=4096`、单结构组、20 预热 + 50 CUDA Event，p95 ≤ 40 ms。覆盖 adapter、聚合、backbone、零/一阶 heads；排除 materialization、decoder、policy、Isaac Sim。PPO full fine-tune 不得缓存会 stale 的 learned activation。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/methods source/anymani/anymani/distill/ssl
```

嵌套合同见 `methods/`、`representations/`、`models/`、`objectives/`、`ssl/`、`rl/`、`tests/` 的 `AGENTS.md`。人类阅读入口见各目录 README。没有正式 pilot 或 PPO transfer 时，不得声明跨手型泛化成立。
