# Learning Diagnostics

平均loss或return不能单独说明模型学到了目标能力。Geometry SSL的误差可能集中在少数owner或尺度上，decoder可能绕过latent；异构RL的平均return也可能掩盖反向抖动、少数easy assets、critic尺度支配、自动reset污染和PhysX资源损坏。`diagnostics`因此不是训练日志的附属工具，而是区分这些解释的共同证据层。

> **Active development.** executable 已采用单一 PALM/JOINT/TIP typed `Z`、teacher-only naive baseline、rho/kappa 双主任务和每 4 epoch 的训练期 Z-gradient proxy。derived-field、density JVP 与 selected-parameter full-gradient Gram 已作为手动 evaluation API 保留；正式 256-epoch 训练与 checkpoint 后机制评估尚未运行。

## 三层证据职责

`recording`忠实写出调用方已经计算的TensorBoard、JSONL、NPZ、Parquet、HDF5与YAML，不重新运行模型或决定checkpoint。`evaluation`在固定bank或固定trajectory协议上运行模型与method-specific probes，形成可配对、可分层的充分统计。`analysis`只读冻结artifact，完成morphology/q/trajectory聚合、bootstrap、曲线和跨variant比较，不import模型、环境、method或teacher。

## 掌旋RL证据

异构掌旋训练以Parquet作为逐update事实源。每个支持资产先独立聚合rollout与optimizer统计，再对实际出现的morphology cells和全局做资产等权汇总，因此增加replica不会改变某只手的统计权重。完整MVP80每update产生1条global、8条cell和80条asset；single/few-support closure只写实际active cells，不用空行伪装完整cohort。

掌旋标量schema 2.7额外区分`policy_base_sigma`（可学习的全局潜高斯标准差）与`policy_sigma`（实际有效标准差，排除ghost后的关节均值）。`recovery_floor_fraction`是可选限位恢复规则实际提高sigma的真实关节比例，关闭时为0。这三项在每个PPO微批更新前记录、跨等大微批平均；下限是否触发属于优化分布事实，不能替代冻结均值策略的圈数评价。

Schema 2.8增加`actor_rejected_action_cost`与`actor_rejected_action_fraction`：前者是未乘权重的Actor辅助成本，后者是无TIP状态下当前均值被目标限位截断超过1e-6的有效关节比例；二者均先按手内有效关节平均，再按微批样本平均。权重为0时该辅助路径不计算，记录0表示禁用，不能解释为实际没有限位截断。统计使用本次PPO forward的均值，不与冻结rollout均值或物理已执行动作混淆；环境reward、Critic回报和正式净圈评价保持独立。

指标schema 2.5增加互斥的`reward_term_orientation_tracking`及update末ADR级别/半宽摘要；旧KD停用时其贡献明确为0。逐回合schema 1.1可附加角度推进数、实际位置ADR档位/偏移、首30秒净圈和窗口完整标记。`goal_count`始终是合格奖金事件数，`orientation_goal_count`另记纯角度推进；两者不再被隐式等同。所有ADR回合事实由pre-reset snapshot交付，避免记录到下一回合刚采样的扰动。

Global row同时保存四段互不混淆的时间：纯environment step、rollout+policy、PPO update和epoch total。显存记录区分PyTorch current/peak allocated、current/peak reserved与CUDA driver free/total；后者覆盖PhysX、CUDA context及其他非PyTorch allocation，是长期运行的真实安全边界。父进程benchmark另外以低频率记录进程树RSS、swap和NVML进程显存，并监听不可恢复PhysX/CUDA日志。发现scene corruption后直接停止，继续使用最近的故障前checkpoint，而不是在损坏状态下发布新权重。

固定评估使用deterministic actor mean和每个replica的首轨迹；主能力门为30秒、每资产16副本，120秒用于耐久复查，入口默认仍为2400步。Automatic reset之前的command snapshot冻结strict goal hits、signed net turns、absolute path、duration及drop/axis/timeout；后续新episode不进入该replica。Strict goal hit同时要求完整SO(3) orientation-keypoint误差小于5 mm和中心相对reset anchor小于25 mm，因此它衡量moving-goal tracking，不是直接30°轴向净转计数。Schema 1.2引入的逐trajectory pulse重计、理论$12N^+$、paired goal/turn ratio、orientation/position gate比例与最后goal时刻继续保留。

接力诊断的外层JSON schema为1.4、evaluation identity为1.6，显式记录`evaluation_role=diagnostic`与`actor_relay`。完成第k个动作后才替换Actor，第k+1个动作开始使用接替参数；`actor_checkpoint_phase`沿完整时间轴记录来源。`.boundary.h5`保存真实交接输入、控制目标、物理状态与前后动作，两个checkpoint及边界文件分别绑定SHA。`analysis/rl/palm_rotation_trace.py`核对参数阶段、存活集合、边界后首个实际动作以及关闭的正式能力门。边界输入可用于精确策略交叉前向；它只包含已声明的可见状态，不是PhysX内部状态的完整恢复包。

启用内部周期时钟的策略在固定 trace 中另存动作前 `pre_phase_clock[T,A,R,2]`，并绑定 `phase_clock` 契约和 `pre_phase_clock_semantics`；它与表示接力来源的 `actor_checkpoint_phase` 不同。独立审计按首轨迹 active 集合及 `policy_step-1` 重建 sin/cos，检验周期、编码顺序、FP32 数值和 reset 边界。重复验收还要求 trace 契约与同一冻结 checkpoint 的 `policy.phase_clock` 一致；缺失时钟见证不能冒充已验证的新策略。原净圈、方向、安全门和支持集分母均不受该诊断扩展影响。

任意显式支持子集都能形成逐资产中位数；single RL closure要求每项至少1净圈、方向一致性至少0.7、drop/axis各少于半数replicas。只有完整冻结80-row cohort才应用54/80、8个cells各5/10和left/right pair诊断。旧N000-relative formal gate同时混合绝对速度和strict goal tracking；跨形态新门冻结前应分别报告持续旋转、tracking与同形态specialist-relative retention。训练proxy、R1筛查、return或absolute path不替代正式R16能力结论。

对应入口：

```bash
python -m anymani.distill.rl.evaluate_palm_rotation_mvp \
  --headless --checkpoint <checkpoint.pth>

python -m anymani.distill.rl.evaluate_palm_rotation_mvp \
  --headless --support_rows 873 --checkpoint <subset-checkpoint.pth>

python -m anymani.distill.diagnostics.analysis.rl.palm_rotation <run-dir>

python scripts/benchmarks/benchmark_heterogeneous_rl.py \
  --output_dir <evidence-dir> -- <training-command>
```

## Geometry SSL证据

## 一次观测应保留什么

最小分析单位不是“某个 step 的 total loss”，而是带有 asset、q、owner、query stratum、sigma、distance shell、ancestor relation 与 validity mask 的预测—真值配对。训练记录同时保存 rho/kappa 的 raw loss、teacher-baseline normalized loss、skill、active/zero 分支和 denominator，使 tail asset group、跨结构 padding 和非光滑 mask 不会因先做全局标量混合而静默改变统计权重。

论文数据效率曲线以训练优化器首次消费的新 asset–configuration pairs 累计数 `new_pairs_seen` 为主横轴。JSONL 同时保存 `pair_uses`、`optimizer_update`、`teacher_pairs_realized` 与墙钟，避免多 mini-epoch 方法在相同新数据横轴上隐藏额外计算；逐 update TensorBoard 服务优化诊断，epoch/validation TensorBoard 才使用新 pair 横轴。validation/evaluation pairs 单独计数，不进入训练预算。

dense evidence 保留统一 $Z$、density/kappa prediction 与 target、closest-source、全部 mask/selectors 和采样 provenance，而不是只保存最终 error。这样 success threshold、sign accuracy、tolerance curve 与新分层可以在不重跑模型的情况下改变。被 mask 排除的一阶样本仍携带原始数值和排除原因；runtime evidence 另外记录 resident assets、BVH/triangle 数、load/release 时间、显存变化与吞吐。

## Checkpoint 选择不是单一均值

Generated validation assets 先按 `physical_geometry_hash` 与 train 整组隔离。固定 bank 为每项 held-out morphology 保存相同的 Sobol q、query realization、sigma 与 teacher digest；每次评估只改变模型参数。Density 与 kappa 相对冻结的 teacher-only naive baseline 归一化，再在 morphology 和物理 strata 上聚合。Epoch-0 network 只表示初始化，learned query-only decoder 只检验 shortcut；二者都不替代 naive baseline。

Pure pretrain 只更新和保存 immutable checkpoints，不在训练进程内执行 fixed-bank selection。Checkpoint 后 validation 保存 baseline identity、完整 selection history 与 promotion evidence；恢复或比较若丢失这些 lineage，即使权重可加载，也不能声称延续同一个选择过程。

## 反事实干预

重建任务最危险的捷径是 decoder 绕过 morphology latent。固定 ablations 因此保持 query、sigma、selectors 和 decoder 不变，只干预表示：

- query-only 把统一 $Z$ 清零，测量纯坐标路径能解释多少误差；
- same-asset q shuffle 错配同一手型的构型 latent，分离静态形态记忆与当前 q；
- cross-asset shuffle 同时破坏形态与构型匹配；
- JOINT-token shuffle 只错配有效 JOINT token，检查 kappa reader 是否读取 joint-specific response；
- 完整 joint-coordinate rewrite 检查 density 不变与 kappa 变号，不能用手工 latent sign flip 替代。

这些比较以相同 asset/q 样本配对，最终采用 asset/q 两级 bootstrap，而不是把同一手的数千 query 当成独立样本。canonical protocol 使用 2000 次重采样；区间反映当前固定 bank 上的配对不确定性，不外推为 official hand 或新 topology 的总体置信区间。

训练期以稀疏 cadence 记录实际 minibatch 上的 representation-gradient proxy；selected checkpoints 在独立固定 bank 上重算 representation、last-block 与 full retained-encoder gradients。两条 matched pilots 使用相同 cadence，诊断耗时单独记录。只有 proxy 与 full gradient 的 cosine sign、norm ordering 和趋势稳定一致后，前者才可作为 future balancing 输入。

## 证据边界

上述诊断可以证伪输入泄漏、latent bypass、错误 routing、mask coverage 退化和 checkpoint lineage 漂移，但不能单独证明 official zero-shot、cross-topology 泛化或 PPO transfer。任何结论至少应共同引用 code revision、resolved config、asset manifest、teacher-baseline artifact、selection history 与 checkpoint；TensorBoard 曲线或最终平均值不能替代这些事实源。

SSL原始证据写入`logs/ssl/<experiment>/<UTC timestamp>/`：dataset/resolved YAML保存实验与资产选择事实，TensorBoard提供在线标量与稀疏histogram，JSONL保存append-only指标和runtime事件，NPZ保存可重新阈值化的dense arrays，YAML保存teacher baseline、q-bank、selection、ablation、gradient summary与lineage。

RL原始证据写入`logs/distill/rl_games/<config>/<run>/`：`params/`保存resolved环境、agent与runtime identity，`metrics_shards/`和`metrics.parquet`保存标量，`evaluation/*.h5`保存trajectory，`nn/`保存与shard inventory一致的checkpoint。外层资源benchmark通常位于`logs/benchmarks/heterogeneous_rl/<run>/`。TensorBoard服务观察；结构化artifact保证可重算；任何记录层都不反向决定模型输入、optimizer或checkpoint选择。
