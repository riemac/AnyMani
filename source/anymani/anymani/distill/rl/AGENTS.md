# AGENTS.md

本文件约束 `distill/rl/`。模型、frame、Research 与测试边界继承 `distill/AGENTS.md`。这里只记录 rl_games runtime、Gym alias、YAML、checkpoint 和日志。

## Project Structure

```text
rl/
├── train.py / play.py               GM AppLauncher、Hydra、runner；不定义 MDP
├── train_palm_rotation_mvp.py       MVP80 typed cfg、identity、structured rl_games launcher
├── __init__.py                      distill-owned training aliases
├── rl_games_backend.py              任何 `rl_games.*` import 前固定本地 backend
├── rl_games_networks.py             compatibility adapter；实现来自 `distill.models`
├── observers.py                     episode / TensorBoard 观察
├── canonical_evidence.py            canonical static geometry evidence builder
├── geo_obs.py                       legacy/deferred，不是 representation 真源
├── runtime/
│   ├── source_config.py             N040 exact static source realization
│   ├── structured_geometry.py       task binding到冻结q-dependent N040
│   ├── retained_geometry.py         冻结q-dependent Z与static frontend cache
│   ├── palm_rotation_geometry.py    encoder-only BF16、FP32 Z边界
│   └── palm_rotation_vecenv.py      named experience transport与rollout诊断
├── palm_rotation_ppo.py             分层采样、masked Normal、双optimizer custom agent
├── structured_runtime.py            named actor/critic与N040 package
├── structured_masked_distribution.py active-joint Gaussian probability
├── structured_ppo.py                direct GAE/clipped PPO
├── agents/                          single-asset与MVP80 rl_games YAML
└── algorithms/                      未来 advantage/PPO update；scalar loss 仍归 objectives.rl
```

Generated heterogeneous task位于`tasks/hetero`，network仍属于`distill.models`。掌托旋转MVP使用项目内custom
rl_games agent；`structured_ppo.py`只保留既有bounded probe用途。不要修改外部`/home/hac/isaac/rl_games`。

## Development Style And Conventions

### 入口顺序

GM入口固定 `python -m anymani.distill.rl.train` / `play`；MVP80入口为
`python -m anymani.distill.rl.train_palm_rotation_mvp`。`tasks/inhand`继续用根目录`scripts/rl_games/`。
Isaac Sim与rl_games import必须在`AppLauncher`之后；MVP80在此前只解析80-row manifest并设置静态scene routing，
直接实例化typed cfg，不做会改写frozen contact/pregrasp dataclass的Hydra round-trip。

### Alias 与 YAML

`AnyMani-GM-SingleAsset-MLP-v0`、LEAP与single-asset tactile aliases继续走rl_games。MVP80训练alias为
`AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0`；入口必须绑定80-row manifest、rank-0 catalog、
N040/precision、structured ABI、arm与replica routing identity。

## Important Semantics

### 几何边界

PALM/JOINT/TIP同索引。MVP80使用固定`[B,21]` owner / `[B,16]` joint ABI；ghost永远invalid。四层N040
保留FP32 master weights，仅encoder forward进入BF16 autocast，输出FP32$Z^e$。Vec-env每个rollout state只计算
一次$Z^e$并写入单份Dict experience；actor与privileged critic共享缓存，mini-epochs不得重算。Actor、critic、
loss和两套optimizers保持FP32且完全分参。RTX 5070 Ti、$B=2560$的provider→actor门为20 warmups、50 events、
p95<50 ms；完整训练还需满足CUDA allocated峰值<总显存85%。

### PPO 与诊断

正式batch为`2560×30=76800`，16个逐资产等量activation slices每4个累积为一个19,200-sample逻辑minibatch，
每mini-epoch仍执行4次optimizer step，共5 mini-epochs；唯一容量fallback为1280 env并按transitions补updates。
Actor base/global residual/critic初始LR分别为`3e-4/1e-4/5e-4`，adaptive schedule保持比例且只能向下调整或恢复到
该锚点，不能使用rl_games默认`1e-2`上限。
训练标量使用Polars 1.32.3写Zstd Parquet分片，checkpoint前flush并保存shard identity；selected trajectories用
gzip HDF5。TensorBoard只保留global/8-cell在线曲线。Actor base使用dynamic-first geometry FiLM：低维控制状态
先经TCN/MLP编码，$Z_j^e$只产生零初始化、有界的scale/shift；整手graph仍只输出有界action residual。

### Logs

输出根为 `logs/distill/rl_games/<config-name>/<run-name>/`。play 优先显式 `--checkpoint`。对比必须记录 commit、asset version、task ID、YAML、seed、backend 与 obs/action schema。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/rl -q
python -m anymani.distill.rl.train_palm_rotation_mvp --headless --smoke --arm residual
python -m anymani.distill.rl.evaluate_palm_rotation_mvp --headless --checkpoint <checkpoint.pth>
python -m anymani.distill.diagnostics.analysis.rl.palm_rotation <run-dir>
python scripts/research/palm_rotation_precision_performance.py --headless
```

人类运行说明见本目录 `README.md`。未经用户要求，不以完整长训练作为普通代码验证。
