# AGENTS.md

本文件约束 `distill/rl/`。模型、frame、Research 与测试边界继承 `distill/AGENTS.md`。这里只记录 rl_games runtime、Gym alias、YAML、checkpoint 和日志。

## Project Structure

```text
rl/
├── train.py / play.py               AppLauncher、Hydra、runner；不定义 MDP
├── __init__.py                      distill-owned training aliases
├── rl_games_backend.py              任何 `rl_games.*` import 前固定本地 backend
├── rl_games_networks.py             compatibility adapter；实现来自 `distill.models`
├── observers.py                     episode / TensorBoard 观察
├── canonical_evidence.py            canonical static geometry evidence builder
├── geo_obs.py                       legacy/deferred，不是 representation 真源
├── runtime/
│   ├── source_config.py             N040 exact static source realization
│   ├── structured_geometry.py       task binding到冻结q-dependent N040
│   └── retained_geometry.py         冻结q-dependent Z与static frontend cache
├── structured_runtime.py            named actor/critic与N040 package
├── structured_masked_distribution.py active-joint Gaussian probability
├── structured_ppo.py                direct GAE/clipped PPO
├── agents/                          single-asset/LEAP rl_games YAML
└── algorithms/                      未来 advantage/PPO update；scalar loss 仍归 objectives.rl
```

Generated heterogeneous task位于`tasks/hetero`，network仍属于`distill.models`；其baseline使用run-owned direct
structured PPO，不通过GM flat observation或rl_games alias。不要修改外部`/home/hac/isaac/rl_games`。

## Development Style And Conventions

### 入口顺序

入口固定 `python -m anymani.distill.rl.train` / `play`。`tasks/inhand` 继续用根目录 `scripts/rl_games/`。Isaac Sim 与 rl_games import 必须在 `AppLauncher` 之后；仅 argparse 与 launcher setup 可提前。`train` 与 `play` 解析同一 task alias 和 agent YAML。

### Alias 与 YAML

`AnyMani-GM-SingleAsset-MLP-v0`、LEAP与single-asset tactile aliases继续走rl_games。Generated heterogeneous
不注册兼容alias；入口显式给出pregrasp tier、env/update budget和structured checkpoint identity。

## Important Semantics

### 几何边界

PALM/JOINT/TIP同索引。Structured N040 route使用固定`[B,21]` owner / `[B,16]` joint ABI；ghost
永远invalid。Encoder从schema-5 artifact严格恢复并保持冻结；只缓存与$q$无关的learned static frontend和
graph bias，每步重算$q$ motion与geometry backbone。Contact、target、上一动作与History30只进入policy adapter。
完整actor性能合同为RTX 5070 Ti、$B=4096$、20 warmups + 50 events、p95严格小于48 ms。

### Logs

输出根为 `logs/distill/rl_games/<config-name>/<run-name>/`。play 优先显式 `--checkpoint`。对比必须记录 commit、asset version、task ID、YAML、seed、backend 与 obs/action schema。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/rl -q
# IsaacSim smoke 走 isaaclab.sh + timeout，路径在 smokes/isaacsim/
```

人类运行说明见本目录 `README.md`。未经用户要求，不以完整长训练作为普通代码验证。
