# AnyMani Heterogeneous Generated-Hand Tasks

`anymani.tasks.hetero`承载generated heterogeneous hand embodiments的ManagerBasedRLEnv任务。当前已实现pregrasp runtime state、partial-reset event与preload-aware action term；scene、其余MDP、Gym ID和训练入口仍未装配。

## 研究对象

同一个DexCube palm-supported manipulation任务可以并行绑定不同handedness、topology、active DoF、TIP数、joint limits和几何参数的generated hands。完整手—物系统仍是hand-level POMDP；JOINT只是structured observation token与factorized action维度。

首版支持域使用canonical-v1：最多16 JOINT、4 TIP和21 physical owners。Tensor shape固定padding以服务GPU并行，真实有效实体数量由每个asset的typed masks与graph决定。正式训练配置默认2048 unique assets；较小cohort只用于有界验证。

## 计划结构

```text
tasks/hetero/
├── AGENTS.md
├── README.md
├── __init__.py                       # 无eager Isaac import的package边界
├── config/generated/
│   ├── asset_binding.py
│   ├── tactile_rotation_env_cfg.py
│   └── agents/
├── mdp/
│   ├── actions.py                    # 1/24 rad policy-step target与PD preload恢复
│   ├── observations.py
│   ├── commands.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── events.py                     # identity provider解析与$q_s/q_t/T_{ho}$ partial reset
│   ├── runtime_state.py              # 纯Torch sidecar、mask和stale-row合同
│   ├── diagnostics.py
│   └── adr.py
└── tests/
```

图中未创建的模块仍只是实现地图；已列出的runtime文件具有可执行合同但不构成完整环境，不使用placeholder body或兼容壳。

## Structured MDP接口

Task层交付raw actor与critic observations，保留palm/jnt/tip/obj/task角色轴。Trainable geometry/temporal/policy/critic网络属于`distill`，不进入ManagerBasedRLEnv配置。

```text
assets/robots ──> typed physical identity + canonical articulation
pregrasp      ──> identity-keyed certified reset
tasks/hetero  ──> O^a, O^c, action/reset/reward/termination
distill       ──> frozen Z^e, actor/critic tokens, masked PPO
```

Actor attention读取整手有效tokens，并由同一个shared head对每个contextual joint token输出action mean。Critic使用独立structured privileged backbone并输出每env一个scalar value。Canonical mask只属于padding/transport，不改变科学动作空间。

## 当前未决项

- Critic使用独立TASK readout token还是显式mask-aware pooling；
- 当前固定任务是否需要actor TASK token；
- 128/2048资产pregrasp coverage与scale interval扩展；
- Hetero ADR各scope的状态与升级单位；
- 新任务验证后，`tasks/gm`旧多资产/canonical实现的精确删除清单。

在这些问题完成Develop与Plan前，本目录不创建可执行源码。
