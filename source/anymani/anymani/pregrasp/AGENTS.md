# AGENTS.md

`pregrasp/`拥有hand-object-scale耦合初态的schema、identity、文件catalog与纯搜索数学。它不拥有scene、reward、policy、PPO或资产生成；Isaac物理编排留在显式research script，任务只通过provider消费已认证记录。

## Project Structure

```text
pregrasp/
├── schema.py / cache.py / provider.py  schema-2 tiered历史接口
├── good_catalog.py                     schema-3 Top-8 exact-key catalog
├── strict_gate.py                      MVP80 strict reset准入唯一谓词
├── mvp80_strict_search.py              Sobol、包络、低秩CEM纯Torch数学
├── isaac_runtime.py                    hand/world frame与PhysX contact解包
└── tests/                              schema、identity、搜索和buffer contracts
```

`scripts/research/generate_heterogeneous_mvp80_pregrasp_strict.py`拥有Isaac scene生命周期与批量物理筛选；稳定数学应留在本目录并有纯测试。`tasks/hetero/config/generated/*pregrasp_identity.py`冻结具体run协议，`tasks/hetero/mdp/events.py`执行fail-closed reset。

Pair fallback的多run证据只由`scripts/research/assemble_heterogeneous_mvp80_strict_catalog.py`组合；source generation/physics/gate digest必须一致，最终80项全部在发布前重验。

## Development Conventions

- Exact key绑定source/physical/canonical/routing identity、DexCube bytes、absolute scale、physics identity与generation identity；dataset row只作provenance。
- Catalog payload和index采用canonical JSON、content digest与同文件系统原子replace。同key不同Top-8严格冲突。
- `GoodPregraspCandidate`使用canonical `[16]`加active mask表达实际$n_i$自由度；ghost为0，单位rad，MVP要求$q_0=u_0$、upright object和零速度。
- v4是已发布宽松对照，身份固定在`good_pregrasp_identity_v4.py`。正式runtime active identity不得覆盖或伪装v4 artifact。
- 修改门限、角速度定义、proposal分布、CEM反馈或物理采样时必须改变generation identity；不得原地重新解释旧catalog。

## Strict V5 Semantics

Strict v5只判断cold-reset质量，不读取rotation reward、micro-roll或PPO结果。每个最终entry的Top-8全部满足：joint margin至少10%、三指TIP-center距离不超过10 cm、sector至少30°、penetration不超过0.5 mm、1 s位移不超过5 mm、倾角不超过10°、前0.2 s线速度不超过0.25 m/s、总角速度不超过2 rad/s、后0.5 s PALM support至少50%。Contact role是metadata，不形成TIP数量tier。

搜索预算为每资产256 Sobol proposals、cheap geometry Top-32 full physics；不足Top-8时最多3轮低秩CEM，每轮128项全部走相同1 s physics。失败后按冻结left/right pair候选顺序换整对，不放宽门或逐资产手调。

## Common Operations

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest -q source/anymani/anymani/pregrasp/tests
python scripts/research/generate_heterogeneous_mvp80_pregrasp_strict.py
```

依赖Isaac/PhysX的验证必须显式启动AppLauncher；默认pytest不得导入Kit。视觉截图和1 s hold只说明reset质量，不构成旋转学习结论。
