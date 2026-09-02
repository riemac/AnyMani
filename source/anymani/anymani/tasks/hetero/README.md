# AnyMani Heterogeneous Generated-Hand Tasks

`anymani.tasks.hetero`承载generated heterogeneous hand embodiments的ManagerBasedRLEnv任务。当前Gym ID为`AnyMani-Hetero-Generated-TactileRotation-v0`，已实现generated canonical scene、分层pregrasp reset、structured observations、History30、contact、command、reward、termination、diagnostics与preload-aware action。Actor/critic与direct PPO位于`distill`；当前matched小cohort已完成，但没有形成rotation能力。

## 研究对象

同一个DexCube palm-supported manipulation任务可以并行绑定不同handedness、topology、active DoF、TIP数、joint limits和几何参数的generated hands。完整手—物系统仍是hand-level POMDP；JOINT只是structured observation token与factorized action维度。

首版支持域使用canonical-v1：最多16 JOINT、4 TIP和21 physical owners。Tensor shape固定padding以服务GPU并行，真实有效实体数量由每个asset的typed masks与graph决定。正式训练配置默认2048 unique assets；较小cohort只用于有界验证。

## 当前结构

```text
tasks/hetero/
├── AGENTS.md
├── README.md
├── __init__.py                       # 无eager Isaac import的package边界
├── config/generated/
│   ├── asset_binding.py              # formal row、canonical artifact与pregrasp catalog唯一轴
│   ├── scene.py                      # formal/search共享且不查询cache的physical scene
│   ├── pregrasp_harness_env_cfg.py   # 不注册Gym ID的搜索/physics harness
│   ├── tactile_rotation_env_cfg.py   # 完整ManagerBased scene/MDP配置
├── mdp/
│   ├── actions.py                    # 1/24 rad policy-step target与PD preload恢复
│   ├── observations.py
│   ├── commands.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── events.py                     # identity provider解析与$q_s/q_t/T_{ho}$ partial reset
│   ├── runtime_state.py              # 纯Torch sidecar、mask和stale-row合同
│   ├── diagnostics.py
└── tests/
```

当前scene/MDP、structured network与PPO均有可执行合同，不使用placeholder body或旧GM Gym alias。ADR仍关闭。

## Structured MDP接口

Task层交付raw actor与critic observations，保留palm/jnt/tip/obj/task角色轴。Trainable geometry/temporal/policy/critic网络属于`distill`，不进入ManagerBasedRLEnv配置。

```text
assets/robots ──> typed physical identity + canonical articulation
pregrasp      ──> identity-keyed certified reset
tasks/hetero  ──> O^a, O^c, action/reset/reward/termination
distill       ──> frozen Z^e, actor/critic tokens, masked PPO
```

Actor attention读取整手有效tokens，并由同一个shared head对每个contextual joint token输出action mean。Critic使用独立structured privileged backbone并输出每env一个scalar value。Canonical mask只属于padding/transport，不改变科学动作空间。

## 当前证据边界

- 128/2048资产pregrasp coverage与scale interval扩展；当前scale 1.2已有balanced16 support coverage；
- Hetero ADR各scope的状态与升级单位；
- contact action-sequence basin与actor object-orientation可观测性；
- 当前row16、seed42、204,800 transitions/arm的matched结果为0 subgoal/0 full turn，不能外推最终收敛。

旧GM multi-asset/canonical配置与Gym IDs已出清；历史checkpoint通过原commit复现，不保留compatibility alias。
