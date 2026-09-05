# Reinforcement Learning Stage

`distill.rl`同时承载GM single-asset rl_games入口与generated heterogeneous structured PPO。环境MDP由
`tasks/gm`或`tasks/hetero`拥有；学习侧拥有network、probability/GAE/PPO、checkpoint与日志。

阅读当前掌旋训练时，可以沿一条完整的数据链进入：任务生成具名观察，`runtime/palm_rotation_vecenv.py`附加每个状态的冻结几何表示，`runtime/palm_rotation_network.py`分流actor/critic并定义动作概率，`algorithms/ppo_batch.py`处理逐资产统计和采样，`palm_rotation_ppo.py`执行双optimizer更新与恢复。运行诊断和额外梯度求导分别由`runtime/palm_rotation_diagnostics.py`、`runtime/palm_rotation_probes.py`承接，最终交给`distill/diagnostics`保存证据。

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

Actor的History30路径可选择逐JOINT TCN，或把每个JOINT的$30\times5=150$个oldest-to-latest标量直接交给local MLP；后者保留全部固定lag并避免learned temporal bottleneck。两条路径随后共享dynamic-first geometry FiLM、finger/hand base和一层graph-biased bounded residual。低维控制状态先编码为$h_{t,j}^{dyn}$，$Z_{t,j}^e$只产生零初始化、有界的FiLM scale/shift：

$$
h_{t,j}^{loc}=\left(1+0.25\tanh\gamma(Z_{t,j}^e)\right)\odot h_{t,j}^{dyn}+0.25\tanh\beta(Z_{t,j}^e).
$$

这些局部信息先投影进入整手tokens，再经graph-biased Transformer形成contextual JOINT表示$H_j$。动作读出具有三个显式研究分支：

$$
\mu_j^{residual}=0.8\tanh(b_j)+0.2\tanh(r_j),\qquad \mu_j^{direct}=\tanh f([H_j,h_j^{loc}]),\qquad \mu_j^{direct\_token}=\tanh f(H_j).
$$

其中$h_j^{loc}$是关节局部控制latent，$H_j$已包含投影后的局部信息与整手上下文；token-only删除的是额外动作读出旁路，不是局部控制信息。Residual head零初始化，`base`对照则只执行局部base。Actor与两层structured critic完全分参，分别由actor optimizer（局部与contextual两个LR组）和critic optimizer更新；checkpoint同时保存两套Adam、value normalizer、当前支持集/cell curriculum、Parquet shard游标及dataset/catalog/N040 identity。掌旋PPO使用直接Embedding关系查表，不物化逐sample one-hot graph bias；该局部实现不改变共享SSL backbone的独立compile合同。

动作均值位于$[-1,1]$；随机策略在latent空间以$\operatorname{atanh}(\mu_j)$为Normal中心，经$tanh$推到实际动作空间。Likelihood包含同一变换的Jacobian，ghost关节从概率、熵、KL、动作和统计分母中排除。几何表示随当前$q$变化，每个rollout state计算一次，随后由该state对应的五轮PPO重复使用。

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
  --headless --arm residual --num_envs 1280
```

完整MVP80的YAML保留历史2560环境预算，1280是其内存回退点；它们不是任意cohort的统一推荐值。A16可以采用128或256环境，分别对应每资产8或16个副本。一次rollout含$B=N_{env}H$个新transitions；每update的optimizer步数为$E M/K$，其中$E$是mini-epochs、$M$是activation minibatches、$K$是梯度累积步数。例如$E=5,M=4,K=1$与$E=5,M=16,K=4$均为20个逻辑optimizer步骤。并行度、数据量、更新时间与墙钟应分别比较；PyTorch allocated不包含全部PhysX/context分配，实际driver余量仍需保留。

Member-level cohort由assets层发布的canonical lock给出。以下命令展示A16已用过的预算与方法配置，参数不代表所有规模的最优点：

```bash
python -m anymani.distill.rl.train_palm_rotation_mvp \
  --headless --cohort_lock source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/pure-leap-right-a16.canonical.lock.yaml \
  --num_envs 256 --max_updates 1000 --minibatches 4 --gradient_accumulation_steps 1 \
  --arm direct_token --advantage_normalization_scope per_asset_rollout \
  --gradient_probe_frequency 0 --tf32 --torch_compile default \
  --reward_release_start_turns 0 --reward_release_end_turns 2
```

逐资产advantage在完整rollout中估计moments，不随mini-epoch重算。梯度probe需要同一图多次求导，因而只能走eager；窄compile训练应关闭probe，在独立诊断进程审计冻结checkpoint。

Raw History30、TF32与窄compile都是显式候选，不会静默改写旧run：

```bash
python -m anymani.distill.rl.train_palm_rotation_mvp \
  --headless --num_envs 1280 \
  --history_encoder raw_stack --tf32 --torch_compile default
```

`--torch_compile`只编译原始actor/critic bound forwards，rl_games外层model保持eager，因此optimizer与checkpoint keys不出现`_orig_mod`。TF32只改变FP32 Linear/attention/Conv的内部乘法模式，参数、GAE、loss和Adam仍为FP32；其科学采用需要数值门与matched学习证据。

Single/few-embodiment closure使用冻结MVP80 rows的显式子集，并继续消费同一strict catalog、N040、task与PPO：

```bash
python -m anymani.distill.rl.train_palm_rotation_mvp \
  --headless --support_rows 873 --num_envs 1280 --max_updates 128 \
  --history_encoder raw_stack --tf32 --torch_compile default
```

子集必须显式给出`--num_envs`和`--max_updates`；它只服务single/few-support closure，不改变最终80-row manifest。数据流smoke使用`--smoke`、4-step rollout和1 update，不构成学习证据。

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

# Single/few-support checkpoint必须重放完全相同的有序rows。
python -m anymani.distill.rl.evaluate_palm_rotation_mvp \
  --headless --support_rows 873 --rl_games_strict \
  --checkpoint /absolute/path/to/subset-checkpoint.pth
```

默认协议为16 replicas、deterministic actor mean与120秒first trajectory。完整80资产输出cohort与pair诊断；
显式子集输出逐资产closure结果但不伪造54/80门。两者都保存JSON与gzip HDF5。Run级事后分析直接读取Parquet：

```bash
python -m anymani.distill.diagnostics.analysis.rl.palm_rotation /absolute/path/to/run
```

当前已实现的任务仍使用120秒、全owner binary观察；新的TIP-only和短回合研究协议应按独立方法身份验证，不由目录重构隐式改变。Cohort评估显式传入训练使用的`--cohort_lock`；主要物理指标为净圈、方向性和drop/axis联合生存率，strict goal tracking单独报告。

### 旧checkpoint与重构

Schema-4 method identity绑定实际源码bytes、资产和训练语义；Git HEAD单独写入`params/agent.yaml`的`code_provenance`。因此相同代码提交后仍可完整续训，真正改动执行源码则继续拒绝同identity恢复。Schema-3旧checkpoint可作显式Actor-only初始化，其他训练状态重置。

跨重构的只读评估使用精确绑定两端源码的等价证书。证书比较四种arm、两种历史编码、可选真实权重的前向/梯度/Adam更新和核心训练AST；它不是通用的“忽略identity”开关，不用于跨实现完整续训：

```bash
python -m anymani.distill.rl.scripts.check_palm_rotation_refactor \
  --reference_revision v0.8.3 --checkpoint <checkpoint.pth> --output <new-certificate.json>
python -m anymani.distill.rl.evaluate_palm_rotation_mvp \
  --headless --checkpoint <checkpoint.pth> --cohort_lock <canonical.lock.yaml> \
  --implementation_certificate <new-certificate.json> --output <new-evaluation.json>
```

任务、模型、N040、cohort和预抓取语义仍须一致，证书生成后任何被覆盖的源码变化都会使证书失效。数值等价与真实仿真回放分别验证，工程检查不替代学习能力或视觉验收。

## Logs and checkpoints

训练输出锚定到仓库根，而不是 shell 当前目录：

```text
logs/distill/rl_games/<config-name>/<run-name>/
```

`config-name` 来自 YAML 的 `params.config.name`，`run-name` 默认是时间戳，也可由
`--experiment_name` 指定。回放优先使用 `--checkpoint`；省略时才通过 `--run_name` 与 latest/best 规则查找。

掌旋PPO每50 updates原子写Zstd Parquet shard，checkpoint前强制flush，正常结束合并为`metrics.parquet`。每个update包含1条global、每个实际active cell一条cell与每个支持资产一条asset；完整MVP80仍是89行。Selected-checkpoint dense trajectories写gzip HDF5，TensorBoard只保存global/cell在线曲线。比较runs时应同时核对task ID、agent YAML、seed、支持rows、manifest/catalog、N040、History30路径、TF32/compile、rl_games commit与checkpoint，而不只比较目录名或最终reward。

## Runtime ownership

- `train.py`、`play.py`：AppLauncher、Hydra cfg、rl_games runner 与日志/checkpoint orchestration；
- `train_palm_rotation_mvp.py`：MVP80及其显式closure子集的typed cfg、runtime identity、capacity/budget与custom runner；
- `agents/`：PPO、network 与 player 配置；
- `rl_games_backend.py`：在 import `rl_games.*` 前固定本地 backend；
- `rl_games_networks.py`：AnyMani custom builder 与 grouped-token/temporal adapter；
- `runtime/structured_geometry.py`：N040 artifact/source/provider strict装配；
- `runtime/palm_rotation_geometry.py`、`palm_rotation_vecenv.py`：encoder-only BF16与单份structured buffer；
- `palm_rotation_ppo.py`：双optimizer、训练hook、checkpoint与注册，保留已有公开导入名称；
- `algorithms/ppo_batch.py`：完整rollout逐资产advantage、严格分层排列和有界学习率；
- `runtime/palm_rotation_network.py`：具名网络适配、squashed masked Normal、窄compile与参数分组；
- `runtime/palm_rotation_diagnostics.py`、`palm_rotation_probes.py`：运行统计、资源保护与可关闭的梯度探针；
- `structured_runtime.py`、`structured_transport.py`：named task tensors到actor/critic/PPO边界；
- `structured_masked_distribution.py`、`structured_ppo.py`：active-action probability、GAE与PPO；
- `geo_obs.py`：deferred geometry-observation material，不是representation source of truth；
- `algorithms/gradient_audit.py`：完整Actor逐资产/replica-half梯度的纯计算，不执行optimizer；
- `scripts/`：RL专属回放、性能、梯度审计与历史structured对照；使用`python -m anymani.distill.rl.scripts.<name>`。

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
python -m anymani.distill.rl.scripts.hetero_palm_rotation_mvp_smoke --all80 --steps 20
python -m anymani.distill.rl.scripts.palm_rotation_precision_performance --headless \
  --output outputs/hetero/performance/palm-rotation-mvp80-bf16.json
python -m anymani.distill.rl.scripts.profile_palm_rotation_update \
  --history_encoder raw_stack --tf32 --batch_size 2400
python scripts/benchmarks/benchmark_heterogeneous_rl.py \
  --output_dir logs/benchmarks/heterogeneous_rl/<name> -- <training-command>
```

`tasks/inhand` 的既有 rl_games 路线仍使用根目录 `scripts/rl_games/train.py` 与
`scripts/rl_games/play.py`；`distill.rl` 不提供顶层兼容 wrapper。
