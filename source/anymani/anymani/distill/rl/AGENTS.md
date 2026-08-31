# AGENTS.md

本文件约束 `distill/rl/`。模型、frame、Research 与测试边界继承 `distill/AGENTS.md`。这里只记录 rl_games runtime、Gym alias、YAML、checkpoint 和日志。

## Project Structure

```text
rl/
├── train.py / play.py               AppLauncher、Hydra、runner；不定义 MDP
├── __init__.py                      distill-owned training aliases
├── rl_games_backend.py              任何 `rl_games.*` import 前固定本地 backend
├── rl_games_networks.py             compatibility adapter；实现来自 `distill.models`
├── masked_ppo.py                    canonical active-action contract 与 Runner 注册
├── observers.py                     episode / TensorBoard 观察
├── canonical_evidence.py            five-mother 静态 evidence
├── geo_obs.py                       legacy/deferred，不是 representation 真源
├── runtime/
│   ├── evidence.py                  N040 artifact/source/provider 装配
│   └── retained_geometry.py         冻结 q-dependent Z 与 static frontend cache
├── agents/                          single-asset/canonical legacy YAML
└── algorithms/                      未来 advantage/PPO update；scalar loss 仍归 objectives.rl
```

Heterogeneous task-local YAML 位于 `tasks/gm/config/heterogeneous_asset/agents/`；Python网络仍属于
`distill.models`。Tasks-owned env cfg仍在`anymani.tasks.gm`。不要修改外部`/home/hac/isaac/rl_games`。

## Development Style And Conventions

### 入口顺序

入口固定 `python -m anymani.distill.rl.train` / `play`。`tasks/inhand` 继续用根目录 `scripts/rl_games/`。Isaac Sim 与 rl_games import 必须在 `AppLauncher` 之后；仅 argparse 与 launcher setup 可提前。`train` 与 `play` 解析同一 task alias 和 agent YAML。

### Alias 与 YAML

`AnyMani-GM-SingleAsset-MLP-v0`是single-asset MDP probe。`AnyMani-GM-HeterogeneousAsset-N040-History30-PPO-v0`
绑定task-local N040 YAML与独立network name；旧`HeterogeneousAsset-TactileRotation-PPO`保留为69D hash-Z
infrastructure baseline。History30属于observation，rl_games`seq_length`固定为1。

## Important Semantics

### 几何边界

PALM/JOINT/TIP同索引。Heterogeneous N040 route使用固定`[B,21]` owner / `[B,16]` joint ABI；ghost
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
