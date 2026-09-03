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
| `AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0` | 80手掌托旋转、structured actor/critic与cached N040 | `agents/heterogeneous_palm_rotation_mvp_ppo.yaml` |

GRU/TCN 名称只属于 training alias，不进入 `tasks` 的 environment-semantic ID。两条 tactile baseline 共享
seed、PPO optimizer、4096 env、central critic schema、`horizon_length=30`、`minibatch_size=30720` 与
reward/ADR contract；差异由独立 YAML 显式记录。

Generated heterogeneous掌托旋转使用独立rl_games alias。`tasks/hetero`输出named actor/critic tensors；
`runtime/palm_rotation_vecenv.py`在每个rollout state运行一次四层N040，并把FP32 $Z^e$连同raw tensors写入单份
Dict experience。`palm_rotation_ppo.py`在网络内按信息边界分流actor与privileged critic，所有mini-epochs复用
缓存。Asset row只作opaque routing和分层采样certificate，不进入连续policy feature。

Actor由逐JOINT History30 TCN、dynamic-first geometry FiLM、finger/hand base路径和一层graph-biased bounded residual组成。低维控制状态先编码为$h_{t,j}^{dyn}$，$Z_{t,j}^e$只产生零初始化、有界的FiLM scale/shift：

$$
h_{t,j}^{loc}=\left(1+0.25\tanh\gamma(Z_{t,j}^e)\right)\odot h_{t,j}^{dyn}+0.25\tanh\beta(Z_{t,j}^e).
$$

整手context随后只产生动作残差：

$$
\mu_j=\mu_j^{base}+0.2\tanh(r_j).
$$

Residual head零初始化，因此初始策略严格等于base。Actor与两层structured critic完全分参，分别由actor
optimizer（base与residual两个LR组）和critic optimizer更新；checkpoint同时保存两套Adam、value normalizer、
80-asset/8-cell curriculum、Parquet shard游标及dataset/catalog/N040 identity。

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

Generated heterogeneous 80手短matched pulse直接运行：

```bash
python -m anymani.distill.rl.train_palm_rotation_mvp \
  --headless --arm residual --num_envs 2560
```

默认是seed42的391-update短pulse。`--num_envs 1280`是唯一正式显存fallback，入口自动把默认updates翻倍以
保持transition预算。正式76,800-sample batch使用16个4,800-sample activation slices，每4片累积后才执行
一次optimizer step，因此维持原4个19,200-sample逻辑minibatches，同时把TCN反向峰值控制在16 GB显存内。
数据流smoke使用`--smoke`，固定80 env、4-step rollout和1 update；它不构成学习证据。

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

MVP80固定能力评估使用专用入口，不复用GM play：

```bash
python -m anymani.distill.rl.evaluate_palm_rotation_mvp \
  --headless --rl_games_strict --checkpoint /absolute/path/to/checkpoint.pth
```

默认协议为80资产×16 replicas、deterministic actor mean、120秒first trajectory，并输出cohort JSON、gzip
HDF5及left/right pair诊断。Run级事后分析直接读取Parquet：

```bash
python -m anymani.distill.diagnostics.analysis.rl.palm_rotation /absolute/path/to/run
```

## Logs and checkpoints

训练输出锚定到仓库根，而不是 shell 当前目录：

```text
logs/distill/rl_games/<config-name>/<run-name>/
```

`config-name` 来自 YAML 的 `params.config.name`，`run-name` 默认是时间戳，也可由
`--experiment_name` 指定。回放优先使用 `--checkpoint`；省略时才通过 `--run_name` 与 latest/best 规则查找。

MVP80每50 updates原子写Zstd Parquet shard，checkpoint前强制flush，正常结束合并为`metrics.parquet`；每个update
包含1条global、8条cell与80条asset记录。Selected-checkpoint dense trajectories写gzip HDF5，TensorBoard只保存
global/8-cell在线曲线。比较runs时应同时核对task ID、agent YAML、seed、manifest/catalog、N040 precision、
rl_games backend commit与checkpoint，而不只比较目录名或最终reward。

## Runtime ownership

- `train.py`、`play.py`：AppLauncher、Hydra cfg、rl_games runner 与日志/checkpoint orchestration；
- `train_palm_rotation_mvp.py`：MVP80 typed cfg、runtime identity、capacity/budget与custom runner；
- `agents/`：PPO、network 与 player 配置；
- `rl_games_backend.py`：在 import `rl_games.*` 前固定本地 backend；
- `rl_games_networks.py`：AnyMani custom builder 与 grouped-token/temporal adapter；
- `runtime/structured_geometry.py`：N040 artifact/source/provider strict装配；
- `runtime/palm_rotation_geometry.py`、`palm_rotation_vecenv.py`：encoder-only BF16与单份structured buffer；
- `palm_rotation_ppo.py`：masked Normal、逐资产等量minibatch、双optimizer与诊断checkpoint；
- `structured_runtime.py`、`structured_transport.py`：named task tensors到actor/critic/PPO边界；
- `structured_masked_distribution.py`、`structured_ppo.py`：active-action probability、GAE与PPO；
- `geo_obs.py`：deferred geometry-observation material，不是representation source of truth；
- `algorithms/`：未来 advantage、sampling 与 weighting 算法边界，当前没有实现。

共享 input adapter、backbone 与 heads 必须来自 [`../models/`](../models/README.md)，不能在 rl_games adapter
里复制 SSL/IL 中科研语义相同的网络。

## Validation

纯 contract tests：

```bash
pytest -q source/anymani/anymani/distill/tests/contracts/rl
pytest -q source/anymani/anymani/distill/tests/contracts/models/test_palm_rotation_policy.py
```

这些测试验证registry、YAML、structured tensor、finger permutation、masked probability、分层sampling、
checkpoint curriculum、Parquet/HDF5与能力门，不启动Isaac Sim。
真正依赖 env reset/step、USD 或 PhysX 的命题应放入显式 runtime smoke，并使用 timeout。训练能否学习则需要
短训练 sanity 或完整实验，不能由 import test 宣称通过。

```bash
python scripts/research/hetero_palm_rotation_mvp_smoke.py --all80 --steps 20
python scripts/research/palm_rotation_precision_performance.py --headless \
  --output outputs/hetero/performance/palm-rotation-mvp80-bf16.json
```

`tasks/inhand` 的既有 rl_games 路线仍使用根目录 `scripts/rl_games/train.py` 与
`scripts/rl_games/play.py`；`distill.rl` 不提供顶层兼容 wrapper。
