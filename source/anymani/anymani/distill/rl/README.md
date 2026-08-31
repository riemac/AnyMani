# Reinforcement Learning Stage

`distill.rl` 是当前可运行的 RL stage。它拥有 distill 侧的 Gym training aliases、rl_games agent YAML、
train/play orchestration、checkpoint/log 路径与 rl_games network adapter；环境本身的 scene、observation、
action、reward、reset 和 termination 仍由 `tasks/gm` 拥有。

## Registered aliases

| Gym ID | 环境/网络用途 | 配置 |
| --- | --- | --- |
| `AnyMani-GM-SingleAsset-MLP-v0` | 当前 single-asset GM MDP probe 主线 | `agents/gm_single_asset_mlp_ppo.yaml` |
| `AnyMani-GM-Leap-MLP-v0` | LEAP GM 环境的 MLP alias | 复用 single-asset MLP YAML；不是当前主线结论 |
| `AnyMani-GM-SingleAsset-TactileRotation-GRU-v0` | current-frame observation + GRU history baseline | `agents/gm_tactile_rotation_gru_ppo.yaml` |
| `AnyMani-GM-SingleAsset-TactileRotation-TCN-v0` | explicit 30-frame history + causal TCN baseline | `agents/gm_tactile_rotation_tcn_ppo.yaml` |
| `AnyMani-GM-Canonical-Unified-PPO-v0` | five-mother canonical 16-DOF/25-body single-batch masked PPO | `agents/gm_canonical_unified_ppo.yaml` |
| `AnyMani-GM-HeterogeneousAsset-TactileRotation-PPO-v0` | 2048-asset N000 current-frame + hash-Z infra baseline | `agents/gm_heterogeneous_n000_ppo.yaml` |
| `AnyMani-GM-HeterogeneousAsset-N040-History30-PPO-v0` | frozen N040 + per-JOINT History30 + graph policy adapter | task-local `heterogeneous_asset/agents/` |

GRU/TCN 名称只属于 training alias，不进入 `tasks` 的 environment-semantic ID。两条 tactile baseline 共享
seed、PPO optimizer、4096 env、central critic schema、`horizon_length=30`、`minibatch_size=30720` 与
reward/ADR contract；差异由独立 YAML 显式记录。

Canonical unified PPO 使用 tasks-owned `AnyMani-GM-Canonical-InHand-v0`。五个 source mother 先由 assets/robots lower 为同一 16-DOF、25-body articulation，再以 `[env,joint]` active mask 处理 action target、observation、reward regularization、owner graph、Normal log-prob、entropy、KL/bounds 与 action history。policy observation 为 `[q, qd, previous_delta, limits, command, object_pos, object_rot6d, fingertip_contact, asset_row, active_mask]`，维度为 116；rl_games global input RMS 关闭，因为 `asset_row` 是离散 evidence routing。

本 alias 的模型在 `distill/models/policy.py`，rl_games boundary 在 `distill/rl/masked_ppo.py`。第一版只学习一个 global scalar `log_std`，shared JOINT mean head 与 PALM symmetric critic；TOPPO、task weighting、逐-token variance 和长时间策略学习均后置。

Heterogeneous N040 route消费1969D actor observation：逐JOINT`[q/pi,target/pi,last_action,owner-tip-contact]`
History30、soft limits、asset row与active mask；critic继续消费103D privileged state。Schema-5 N040按当前$q$
在线重算，冻结static frontend/graph bias可跨policy steps缓存；ordered history经共享小MLP编码，一层compiled
graph policy adapter输出逐JOINT Gaussian mean。RTX 5070 Ti、$B=4096$完整actor门为p95严格小于48 ms。

canonical route 会从原始 typed geometry semantics 物化 anchors、home surfaces、screws、`q_home` 与真实 owner graph，并按 `asset_row` gather。PPO fine-tune 不能跨 optimizer step 缓存 learned geometry activation；同一 minibatch 内只对唯一 asset rows 做一次可微静态前向。YAML 使用 `minibatch_size=640`，正好整除 `160×32=5120` rollout，并在 RTX 5070 Ti 上与 Isaac Sim 同驻留。

## Train

从仓库根目录运行：

```bash
source /home/hac/isaac/env_isaaclab/bin/activate

/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.rl.train \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 4096 \
  --seed 42 \
  --headless
```

小规模排查可以降低 `--num_envs`；入口会把 PPO minibatch 修正为可整除当前 rollout batch 的值。正式实验
不要把 debug 时的自动修正误记成标准训练配置。

查看完整参数：

```bash
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.rl.train --help
```

## Play

显式 checkpoint 是最可复现的回放方式：

```bash
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.rl.play \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --real-time
```

同一个 training task 用于 play，避免额外 play alias 让 observation/action schema 漂移。`--video` 会打开
camera pipeline；普通 headless training 或 GUI 人工回放不应无故承担 render 开销。

## Logs and checkpoints

训练输出锚定到仓库根，而不是 shell 当前目录：

```text
logs/distill/rl_games/<config-name>/<run-name>/
```

`config-name` 来自 YAML 的 `params.config.name`，`run-name` 默认是时间戳，也可由
`--experiment_name` 指定。回放优先使用 `--checkpoint`；省略时才通过 `--run_name` 与 latest/best 规则查找。

TensorBoard event 是诊断训练动态的主要结构化证据。比较 runs 时应同时核对 task ID、agent YAML、seed、
asset version、rl_games backend commit 与 checkpoint，而不只比较目录名或最终 reward。

## Runtime ownership

- `train.py`、`play.py`：AppLauncher、Hydra cfg、rl_games runner 与日志/checkpoint orchestration；
- `agents/`：PPO、network 与 player 配置；
- `rl_games_backend.py`：在 import `rl_games.*` 前固定本地 backend；
- `rl_games_networks.py`：AnyMani custom builder 与 grouped-token/temporal adapter；
- `geo_obs.py`：legacy/deferred geometry-observation material，不是新的 representation source of truth；
- `algorithms/`：未来 advantage、sampling 与 weighting 算法边界，当前没有实现。

共享 input adapter、backbone 与 heads 必须来自 [`../models/`](../models/README.md)，不能在 rl_games adapter
里复制 SSL/IL 中科研语义相同的网络。

## Validation

纯 contract tests：

```bash
pytest -q source/anymani/anymani/distill/tests
```

这些测试验证 registry、YAML 公平性、tensor shape、temporal encoder 与 rl_games adapter，不启动 Isaac Sim。
真正依赖 env reset/step、USD 或 PhysX 的命题应放入显式 runtime smoke，并使用 timeout。训练能否学习则需要
短训练 sanity 或完整实验，不能由 import test 宣称通过。

`tasks/inhand` 的既有 rl_games 路线仍使用根目录 `scripts/rl_games/train.py` 与
`scripts/rl_games/play.py`；`distill.rl` 不提供顶层兼容 wrapper。
