# Reinforcement Learning Stage

`distill.rl`同时承载GM single-asset rl_games入口与generated heterogeneous structured PPO。环境MDP由
`tasks/gm`或`tasks/hetero`拥有；学习侧拥有network、probability/GAE/PPO、checkpoint与日志。

## Registered aliases

| Gym ID | 环境/网络用途 | 配置 |
| --- | --- | --- |
| `AnyMani-GM-SingleAsset-MLP-v0` | 当前 single-asset GM MDP probe 主线 | `agents/gm_single_asset_mlp_ppo.yaml` |
| `AnyMani-GM-Leap-MLP-v0` | LEAP GM 环境的 MLP alias | 复用 single-asset MLP YAML；不是当前主线结论 |
| `AnyMani-GM-SingleAsset-TactileRotation-GRU-v0` | current-frame observation + GRU history baseline | `agents/gm_tactile_rotation_gru_ppo.yaml` |
| `AnyMani-GM-SingleAsset-TactileRotation-TCN-v0` | explicit 30-frame history + causal TCN baseline | `agents/gm_tactile_rotation_tcn_ppo.yaml` |

GRU/TCN 名称只属于 training alias，不进入 `tasks` 的 environment-semantic ID。两条 tactile baseline 共享
seed、PPO optimizer、4096 env、central critic schema、`horizon_length=30`、`minibatch_size=30720` 与
reward/ADR contract；差异由独立 YAML 显式记录。

Generated heterogeneous路线不注册rl_games Gym alias。`tasks/hetero`输出named structured tensors；
`structured_runtime.py`装配冻结四层N040、shared per-joint actor与独立scalar critic，`structured_ppo.py`实现
active-joint Normal、GAE与clipped PPO。研究入口为`scripts/research/train_hetero_structured_ppo.py`。
N040按current $q$在线重算，rollout只在encoder冻结时保存Z；actor/critic完全分参，asset row只作opaque routing。

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

Generated heterogeneous bounded PPO直接运行：

```bash
/home/hac/isaac/IsaacLab/isaaclab.sh -p scripts/research/train_hetero_structured_ppo.py \
  --tier support_basin --num-envs 128 --updates 100 --horizon 16 --eval-steps 400
```

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
- `runtime/structured_geometry.py`：N040 artifact/source/provider strict装配；
- `structured_runtime.py`、`structured_transport.py`：named task tensors到actor/critic/PPO边界；
- `structured_masked_distribution.py`、`structured_ppo.py`：active-action probability、GAE与PPO；
- `geo_obs.py`：deferred geometry-observation material，不是representation source of truth；
- `algorithms/`：未来 advantage、sampling 与 weighting 算法边界，当前没有实现。

共享 input adapter、backbone 与 heads 必须来自 [`../models/`](../models/README.md)，不能在 rl_games adapter
里复制 SSL/IL 中科研语义相同的网络。

## Validation

纯 contract tests：

```bash
pytest -q source/anymani/anymani/distill/tests
```

这些测试验证registry、single-asset YAML、structured tensor、masked probability与PPO，不启动Isaac Sim。
真正依赖 env reset/step、USD 或 PhysX 的命题应放入显式 runtime smoke，并使用 timeout。训练能否学习则需要
短训练 sanity 或完整实验，不能由 import test 宣称通过。

`tasks/inhand` 的既有 rl_games 路线仍使用根目录 `scripts/rl_games/train.py` 与
`scripts/rl_games/play.py`；`distill.rl` 不提供顶层兼容 wrapper。
