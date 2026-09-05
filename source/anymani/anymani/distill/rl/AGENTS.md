# AGENTS.md

本文件约束 `distill/rl/`。模型、frame、Research 与测试边界继承 `distill/AGENTS.md`。这里只记录 rl_games runtime、Gym alias、YAML、checkpoint 和日志。

## Project Structure

```text
rl/
├── train.py / play.py               GM AppLauncher、Hydra、runner；不定义 MDP
├── train_palm_rotation_mvp.py       MVP80及显式支持子集的typed cfg、identity、launcher
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
├── palm_rotation_ppo.py             双optimizer、训练/恢复hook与注册；公开导入保持稳定
├── scripts/                         RL专属探针、回放、性能与历史structured对照
├── structured_runtime.py            named actor/critic与N040 package
├── structured_masked_distribution.py active-joint Gaussian probability
├── structured_ppo.py                direct GAE/clipped PPO
├── agents/                          single-asset与MVP80 rl_games YAML
└── algorithms/                      ppo_batch逐资产估计/分层采样；gradient_audit只读梯度计算
```

Generated heterogeneous task位于`tasks/hetero`，network仍属于`distill.models`。掌托旋转MVP使用项目内custom
rl_games agent；`structured_ppo.py`只保留既有bounded probe用途。不要修改外部`/home/hac/isaac/rl_games`。

`runtime/palm_rotation_network.py`拥有rl_games网络/分布适配，不重定义神经架构；`palm_rotation_diagnostics.py`归约agent与task已形成的事实，`palm_rotation_probes.py`执行显式只读梯度探针。主agent保留更新与恢复顺序，不以大型mixin隐藏训练状态。RL专属脚本放在本目录`scripts/`，资产生产、预抓取和SSL工具不机械迁入。

## Development Style And Conventions

### 入口顺序

GM入口固定 `python -m anymani.distill.rl.train` / `play`；MVP80入口为
`python -m anymani.distill.rl.train_palm_rotation_mvp`。`tasks/inhand`继续用根目录`scripts/rl_games/`。
Isaac Sim与rl_games import必须在`AppLauncher`之后；掌旋入口在此前只解析冻结80-row manifest及其显式
`--support_rows`子集，并设置静态scene routing。环境直接实例化typed cfg，不做会改写frozen
contact/pregrasp dataclass的Hydra round-trip。

### Alias 与 YAML

`AnyMani-GM-SingleAsset-MLP-v0`、LEAP与single-asset tactile aliases继续走rl_games。MVP80训练alias为
`AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0`；入口绑定冻结manifest或member-level canonical cohort lock、rank-0 catalog、N040/precision/History30/compile、structured ABI、arm和replica routing。Legacy `--support_rows`只选冻结MVP80子集；`--cohort_lock`独立提供明确成员轴，两者互斥，不改写父manifest。

## Important Semantics

### 几何边界

PALM/JOINT/TIP同索引。MVP80使用固定`[B,21]` owner / `[B,16]` joint ABI；ghost永远invalid。四层N040
保留FP32 master weights，仅encoder forward进入BF16 autocast，输出FP32$Z^e$。Vec-env每个rollout state只计算
一次$Z^e$并写入单份Dict experience；actor与privileged critic共享缓存，mini-epochs不得重算。Actor、critic、
loss和两套optimizers保持FP32且完全分参；显式TF32只改变Linear/attention/Conv内部乘法。大batch计时回答训练吞吐，$B=1$端到端才回答真机20 Hz时限，二者不可混用。完整训练同时检查PyTorch peak allocated与
CUDA driver free；后者至少保留配置中的安全余量，因为PhysX与context不受PyTorch allocator统计。

### PPO 与诊断

环境数是计算与统计选择，不设所有cohort共用的128/1280硬默认。预算同时记录$N_{env}H$新样本、逐资产副本数和$E M/K$逻辑optimizer步数；例如5个mini-epochs、4个minibatches、accumulation1为20步。改变并行数、采样、累积或时长必须进入run identity，不用epoch count替代等样本或等墙钟比较。
Actor base/global residual/critic初始LR分别为`3e-4/1e-4/5e-4`，adaptive schedule保持比例且只能向下调整或恢复到
该锚点；显式`--learning_rate`同比缩放各组及上限，不能使用rl_games默认`1e-2`上限。
训练标量使用Polars 1.32.3写Zstd Parquet分片，checkpoint前flush并保存shard identity；selected trajectories用
gzip HDF5。每update写global、实际active cells与全部支持资产；只有完整MVP80固定89行。Actor base使用
dynamic-first geometry FiLM：History30可经逐JOINT TCN或direct 150D raw stack进入local MLP，$Z_j^e$只产生
零初始化、有界的scale/shift；Residual输出局部base与整手修正，`direct`额外拼接local skip，`direct_token`只从contextual JOINT token读出全authority动作。局部信息已进入token-only的输入，不等于被丢弃。Compile只包装actor/critic bound forwards，
不得让外部rl_games把整个model改成`_orig_mod` checkpoint namespace。

诊断反归一化只读value moments；不可变rollout均值与rl_games逐minibatch更新的KL参考分开存储。Gradient probe只在eager运行，是否介入优化依赖独立rollout/replica-half可靠性证据。

Schema4把Git HEAD放在独立code provenance，method identity只绑定实际源码及科学合同；完整resume仍严格比较全部method字段。旧schema3 Actor-only初始化可用，跨重构只读评估须提供精确源码映射的等价证书，不能用忽略identity开关代替。修改任务/观察/算法是新研究分支，不冒充语义保持重构。

训练入口默认`--actor_contact tip`和随机20–60秒回合；全触觉/固定120秒只作显式对照。TIP-only同时清除当前关节帧、History30和owner token中的非TIP触觉，保留五通道权重接口、实体mask、本体状态及TIP信号，reward/critic仍读取完整接触。时长独立于ADR，每回合计划长度决定有限时域timeout；能力主评估显式使用`--steps 600 --num_replicas 16`，重要候选以2400步复查耐久。

课程可将净圈按计划时长换算到参考时长，提前失败不借实际短存活时间放大进度；`--reward_release_floor`是显式塑形下限，不是能力门。`--actor_init_checkpoint --init_critic`额外继承兼容critic和value统计，optimizer/课程重置；保存的normalizer初始计数加后续更新量才是其合法计数。`scripts.prepare_palm_rotation_resume`只链接checkpoint声明并哈希验证的不可变分片到独立run，完整恢复不重读初始化parent。

`--rotation_progress_clip_rad`改变有符号进展奖励的对称截断半宽，必须进入task/training identity。20 Hz下0.025/0.04 rad每步对应0.5/0.8 rad/s奖励饱和点，不改动作authority；评价仍使用未截断转角，不能把同轨迹奖励重算当作学习增益。

### Logs

输出根为 `logs/distill/rl_games/<config-name>/<run-name>/`。play 优先显式 `--checkpoint`。对比必须记录 commit、asset version、task ID、YAML、seed、backend 与 obs/action schema。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/rl -q
python -m anymani.distill.rl.train_palm_rotation_mvp --headless --smoke --arm residual
python -m anymani.distill.rl.train_palm_rotation_mvp --headless --support_rows 873 --num_envs 1280 --max_updates 128 --history_encoder raw_stack --tf32 --torch_compile default
python -m anymani.distill.rl.evaluate_palm_rotation_mvp --headless --checkpoint <checkpoint.pth>
python -m anymani.distill.diagnostics.analysis.rl.palm_rotation <run-dir>
python -m anymani.distill.rl.scripts.palm_rotation_precision_performance --headless
python -m anymani.distill.rl.scripts.profile_palm_rotation_update --history_encoder raw_stack --tf32 --batch_size 2400
python -m anymani.distill.rl.scripts.check_palm_rotation_refactor --checkpoint <checkpoint.pth> --output <new-certificate.json>
```

人类运行说明见本目录 `README.md`。未经用户要求，不以完整长训练作为普通代码验证。
