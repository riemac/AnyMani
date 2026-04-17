# Isaac Lab 异构 Articulation 并行训练调研

## Executive Summary

结论先行：**支持，但支持的是“同一结构族中的异构实例”而不是任意不同手型的任意混搭。** 在 [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) 里，`InteractiveSceneCfg.replicate_physics=False` 的语义就是“允许每个 env 拥有独立 asset instance”，官方 how-to 和 demo 甚至直接给出“不同 env 随机放 ANYmal-C 或 ANYmal-D articulation”的例子。[^1][^2][^3]

但这个支持有很强的边界：官方文档明确要求，当多个 articulated assets 通过同一个 physics interface (`Articulation`) 统一访问时，它们必须保持**相同的 links/joints 数量、collision bodies 数量以及对应名称**；`Articulation` 内部也是用一个 `ArticulationView.shared_metatype` 暴露共享的 `dof_count/dof_names/link_count/link_names`。所以，对你说的“同关节拓扑、不同几何参数 mutate URDF”这类 post-mutate 变体，答案是 **Yes**；对“Leap 家族 + Allegro 家族”这种更大跨度 family，如果没有人为标准化成同一 schema，就**不应该**直接塞进同一个默认 `Articulation` batch。[^4][^5]

训练层面，**一个 Specialist Policy 在同一并行任务里覆盖这些同拓扑变体是可行的**，因为 Isaac Lab 的 observation/action manager 和 Gym spaces 都是围绕单一 batched tensor space 构建的；只要所有 env 的 observation/action 维度一致，就能共享一个 policy。反之，如果 joint/link 数不同，默认 manager 会直接在 action/obs shape 层面卡住。[^6][^7][^8]

## Direct Answers

| 问题 | 结论 | 关键条件 |
|---|---|---|
| `replicate_physics=False` 下，不同 env 加载“关节拓扑一致、几何参数不同”的 Articulation 是否受支持？ | **支持**。这是 Isaac Lab 官方支持的异构 env 模式。[^1][^2][^3] | 同一物理接口下的 asset 必须保持相同的 articulation schema：同 links/joints 数、同 collision bodies 数、同名称/层级。[^4][^5] |
| 能否直接用这些变体在一个并行 RL 任务里训练单个 Specialist Policy？ | **可以**，而且这是合理的效率优化。[^6][^7][^8] | 观测/动作维度必须一致；最好显式输入 embodiment/morphology 特征，否则策略看到的是“潜变量未观测”的 morphology randomization。[^6][^7][^8][^10][^12] |

## Architecture / System Overview

```text
Mutated URDF bank
    │
    ├─ Option A: UrdfFileCfg / spawn_from_urdf
    └─ Option B (recommended): offline convert_urdf / convert_instanceable
                                -> instanceable USD bank
    │
    ▼
MultiAssetSpawnerCfg(list[SpawnerCfg])   or   MultiUsdFileCfg
    │
    ▼
InteractiveScene(replicate_physics=False)
    │
    ├─ env_0 / Robot_A
    ├─ env_1 / Robot_B
    ├─ env_2 / Robot_C
    └─ ...
    │
    ▼
Articulation._initialize_impl()
create_articulation_view("/World/envs/env_*/Robot...")
    │
    ▼
shared_metatype + batched obs/actions
    │
    ▼
single specialist policy
```

这个图里的关键不是“能不能每个 env 放不同资产”，而是“**这些不同资产能不能被同一个 `ArticulationView` 合法地打包成一个 batch**”。Isaac Lab 的答案是：**能，但只限同结构族**。[^1][^3][^4][^5]

## What Isaac Lab Actually Supports

### 1. `replicate_physics=False` 就是 Isaac Lab 的异构 env 开关

`InteractiveScene` 文档把两种模式写得很直接：`replicate_physics=True` 用于“所有 env 都是同一 asset 的复制”，`False` 用于“env 里有独立 assets”。初始化代码里甚至写了注释：当 `replicate_physics=False` 时，“we assume heterogeneous environments”，并先 clone env xforms，再由 spawner 做逐对象 cloning。配置类也明确说明：`False` 允许每个 env 拥有独立 USD prim 实例，但代价是 scene setup / physics parsing 更慢；同时 `clone_in_fabric` 不能和它一起用。[^1][^2]

这和官方对 USD-level randomization 的处理也是一致的：教程明确写出，prestartup 的 scale/color 级随机化要求 `replicate_physics=False`；测试还专门断言，一旦把它改回 `True` 就应当抛错。换句话说，只要你要在 stage/asset 身上做 per-env 差异，Isaac Lab 的默认答案就是关掉 physics replication。[^9]

### 2. 官方已经给了“异构 articulation 同场并行”的示例

`multi_asset.py` 不是只演示 rigid objects；它还在同一个 `ArticulationCfg` 里用 `MultiUsdFileCfg` 随机选择 ANYmal-C 或 ANYmal-D，并且整个 scene 明确用 `replicate_physics=False`。官方 how-to 也用这段 demo 作为“different assets under the same prim path”的 articulation 示例，并说明每个 env 可以有不同 articulation。测试文件 `test_spawn_wrappers.py` 和 `test_simulation_stage_in_memory.py` 也确实对这种 multi-USD 生成做了验证。[^3]

因此，从“Isaac Lab 是否支持不同 env 同时加载不同 articulation assets”这个角度，**源码层面已经是 yes，不需要你发明 workaround**。[^3]

### 3. 但它要求的是“同结构族异构”，不是“任意异构”

官方 how-to 的限制条件非常硬：当多个 articulations 共用同一个 physics interface 时，它们必须有**相同的 links/joints 数量、相同的 collision bodies 数量、以及相同的名称**；否则 physics parsing 可能失败。`Articulation` 类内部也正是围绕一个共享 `shared_metatype` 暴露 `dof_count/dof_names/link_count/link_names`，测试同样在验证 `max_dofs == shared_metatype.dof_count`、`max_links == shared_metatype.link_count` 以及 link names 的一致性。[^4][^5]

这意味着，对你这里的 mutate embodiments，**“同关节拓扑”只是必要条件，还不是充分条件**。在 Isaac Lab 里，你最好把“family”定义成更严格的 **articulation schema class**：

1. joint/link 数不变；[^4][^5]
2. joint/link 名称不变；[^4][^5]
3. collision body 的数量和命名也不变；[^4]
4. prim hierarchy / articulation root 路径保持相容。[^3][^4]

这正好和你当前“primitive skin + 奥卡姆剃刀”的资产思想相契合：如果所有 link（尤其非指尖）都由固定槽位的 box/cylinder/sphere 近似组成，你完全可以把 **“几何参数变化”** 与 **“schema 不变”** 解耦。对 Isaac Lab 而言，最友好的做法不是让某个 mutate variant 新增/删除 collider，而是**保留固定 collider 槽位，只改尺寸/位姿，必要时把某些槽位退化成极小 primitive**。你的 idea 文档本来就把几何信息塞到 joint-centric 节点特征里，并把多样性来源放在 pre-made topology 与 post-mutate geometry 上；这与 Isaac Lab 的结构约束是对齐的。[^4][^12][^14]

### 4. RL 层面能不能共享一个 policy，取决于 batched tensor shape

Isaac Lab 的 observation manager 要求每个 observation term 返回 `(num_envs, ...)` 形状；action manager 维护的是 `(num_envs, total_action_dim)` 的统一 action buffer，并在 `process_action()` 里显式检查 `action.shape[1]` 是否等于固定的 `total_action_dim`。`ManagerBasedRLEnv` 和 `DirectRLEnv` 也都是先构造单个 env 的 observation/action space，再用 `gym.vector.utils.batch_space` 批量化到 `num_envs`。[^6][^7][^8]

这意味着：

- **同 topology / 同 DOF 数 / 同 obs dim**：天然适合一个 Specialist Policy 并行训练。[^6][^7][^8]
- **不同 topology / 不同 DOF 数**：Isaac Lab 默认 manager-based / direct-RL 流水线不会自动帮你处理变长 action/obs。要么拆成多个 specialist families，要么自己做 padding/masking/custom action wrapper。[^6][^7][^8]

这个点和 GET-Zero / T(R,O) 很值得区分：**论文里的网络层可以处理变长结构，不等于 Isaac Lab 的 simulator batch 层也天然支持变长 control space。** GET-Zero 的 encoder 支持可变 joint token 数；T(R,O) 在附录里甚至显式把 link nodes zero-pad 到统一大小，以实现“不同 link 数 embodiment 的并行计算”。但 Isaac Lab 默认 RL manager 仍然是固定 batched tensor space。[^10][^12]

## URDF Question: Can This Be Driven from Mutated URDFs?

可以，但我建议把“研究上生成 URDF”和“训练时高效加载资产”分成两层来看。[^13]

1. **直接 URDF 路径**：Isaac Lab 有 `UrdfFileCfg` 和 `spawn_from_urdf()`；其实现会用 `UrdfConverter` 把 URDF 懒转换成 USD 后再导入 scene。[^13]
2. **推荐路径**：官方同时提供 `convert_urdf.py` 和 `convert_instanceable.py`，而 `UrdfConverter` 也明确说生成的是 learning 常用的 instanceable USD。对大规模 mutate asset bank 来说，离线批量转 instanceable USD，再用 `MultiUsdFileCfg` / `MultiAssetSpawnerCfg` 训练，工程上更稳，也更接近 Isaac Lab 官方示例。[^13]
3. **如果你坚持直接多 URDF 混放**：源码表明 `MultiAssetSpawnerCfg.assets_cfg` 接受通用 `SpawnerCfg` 列表，而 `spawn_multi_asset()` 会逐个调用 `asset_cfg.func(...)`；因此把多个 `UrdfFileCfg` 塞进 `MultiAssetSpawnerCfg` 在机制上是说得通的。只是 Isaac Lab 文档只提供了专门的 `MultiUsdFileCfg`，没有专门的 `MultiUrdfFileCfg`，所以这条路更像“源码可行”而不是“官方 how-to 主路径”。[^11][^13]

## Implications for Your AnyMani Pipeline

### A. 你现在的 `post-mutate` specialist 训练思路是可行的

对“同一 pre-made topology 下，不同 link 长度、碰撞体尺寸、指尖形状”的 mutate variants，Isaac Lab 的 multi-asset + `replicate_physics=False` 正是为这种场景准备的。官方 how-to 甚至把“robots with different link lengths”当作该功能的典型用途。[^4]

所以，对每个 **拓扑家族**，你完全可以不再训练 N 个单体 `Expert Policy`，而是一次性在一个并行任务里塞入多个 mutate embodiments，训练一个 **family-level Specialist Policy**。这会把你的训练范式从“one hand ↔ one expert”推进到“one topology family ↔ one specialist”。[^1][^3][^4][^6][^7][^8]

### B. 但 policy 最好不要是“盲”的

GET-Zero 的关键不是只把不同 embodiment 混到一个 batch，而是**显式以 embodiment graph 为条件**，并把多个 embodiment-specific experts 蒸馏到一个 embodiment-aware policy；T(R,O) 也是把 link geometry、spatial transformation 和 object context 明确编码进 graph。[^10][^12]

因此，如果你在 Isaac Lab 里直接把多个 mutate hands 混起来做 RL，我建议把下面这些静态 morphology 特征显式放进 observation / policy conditioning，而不是只依赖动力学差异让策略“自己猜”：

- 关节静态属性：axis、limit、rest pose 相对 SE(3)；[^10][^14]
- link geometry 特征：你 idea 里的 SDF/BPS/primitive 参数；[^12][^14]
- 指尖类型 / 尺寸编码；[^4][^14]
- family / pre-made topology id。[^10][^14]

否则这个问题在 RL 视角下更像一个 **POMDP with hidden embodiment latent**，会把 family-level specialist 退化成“盲域随机化策略”，通常不如显式 embodiment-aware 表达稳定。[^10][^12]

### C. Leap family 与 Allegro family 的统一，建议放在 specialist 之上

如果你的目标是“至少覆盖 LEAP + Allegro 的不同 family 变体”，那 Isaac Lab 的默认 batched articulation path 更适合作为**family 内 specialist 训练器**，而不是直接一步到位做跨 family unified policy。原因不是图网络做不到，而是 simulator batch / action space / parser schema 三层限制叠加后，直接把不同 family 硬塞到一个 `ArticulationView` 里很脆。[^4][^5][^6][^7][^8]

更稳的路线是：

1. **pre-made 阶段**：按 articulation schema 划 family；[^4][^14]
2. **post-mutate 阶段**：每个 family 在 Isaac Lab 里并行训练一个 Specialist Policy；[^1][^3][^4][^6][^7][^8]
3. **统一阶段**：再用 GET-Zero 式蒸馏，或者 T(R,O) / joint-graph 式 padding+masking，把多个 specialists 提升为一个 unified policy。GET-Zero 的 variable-token joint graph 和 T(R,O) 的 zero-padding 都是这一步的好启发。[^10][^12]

这和你图里的“层次通才 → 通才专家”路线是吻合的，只是我会把“层次”更明确地定义成 **Isaac Lab 可 batch 的结构族**，而不是仅凭直觉划分。[^14]

## Recommended Engineering Recipe

1. **先定义 specialist family 的硬标准**：同 joint/link/collision-body 数量与名称，同 prim hierarchy。[^4][^5]
2. **mutate 只改 geometry，不改 schema**：改长度、半径、box size、tip surface；不要随意增删 collider 数量。[^4][^14]
3. **批量离线把 URDF bank 转成 instanceable USD bank**，把训练时的 asset loading 固定下来。[^13]
4. **每个 family 用 `InteractiveSceneCfg(..., replicate_physics=False)` + `MultiUsdFileCfg`** 生成异构并行 env；若还在原型期，也可尝试 `MultiAssetSpawnerCfg([UrdfFileCfg(...) ...])`。[^1][^2][^3][^11][^13]
5. **policy 输入显式带 morphology conditioning**，不要只喂 joint state。[^10][^12][^14]
6. **family 内直接训练 Specialist Policy**；family 间再做蒸馏/统一。[^10][^12][^14]

## Key Repositories / Files

| Resource | Why it matters |
|---|---|
| `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py` | 定义 `replicate_physics=False` 的异构 env 语义。[^1] |
| `/home/hac/isaac/IsaacLab/docs/source/how-to/multi_asset_spawning.rst` | 官方 how-to；明确 articulation multi-asset 的支持边界与结构约束。[^3][^4] |
| `/home/hac/isaac/IsaacLab/scripts/demos/multi_asset.py` | 官方“异构 articulation 并行”示例。[^3] |
| `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` | `ArticulationView.shared_metatype` 与 regex-based batched articulation 初始化逻辑。[^5] |
| `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py` + `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py` | 证明单任务内 action/obs 维度默认必须固定。[^6][^7] |
| `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/` | URDF → USD 的训练入口。[^13] |
| `/home/hac/isaac/AnyMani/source/anymani/ideas/graph/idea.md` | 你的 joint-centric morphology 表达与 family/post-mutate 划分思路。[^14] |

## Confidence Assessment

**High confidence**

- Isaac Lab supports heterogeneous assets across envs when `replicate_physics=False`.[^1][^2][^3]
- This includes official articulation examples, not just rigid objects.[^3]
- A single batched RL task can train one policy across those envs if observation/action dimensions stay fixed.[^6][^7][^8]
- The structural constraint is stricter than “same kinematic idea”: same counts and names for links/joints/collision bodies are explicitly required.[^4][^5]

**Medium confidence / informed inference**

- For your mutate URDF bank, the most robust production path is offline USD conversion plus `MultiUsdFileCfg`, even though generic `MultiAssetSpawnerCfg + UrdfFileCfg` should also work mechanically.[^11][^13]
- LEAP-family and Allegro-family assets are best treated as separate Specialist families unless you deliberately canonicalize their articulation/collision schemas; this follows directly from Isaac Lab’s batching constraints, but the exact feasibility depends on how aggressively you standardize names, collider slots, and observation/action wrappers.[^4][^5][^6][^7][^8]
- A “blind” shared policy over morphology-randomized hands will likely underperform an explicitly embodiment-conditioned specialist; this is a methodological inference supported by GET-Zero and T(R,O), not an Isaac Lab API rule.[^10][^12][^14]

## Footnotes

[^1]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:54-68,108-113,150-177,225-234`
[^2]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:87-124`
[^3]: `/home/hac/isaac/IsaacLab/scripts/demos/multi_asset.py:181-215,223-299`; `/home/hac/isaac/IsaacLab/docs/source/how-to/multi_asset_spawning.rst:61-88,104-129`; `/home/hac/isaac/IsaacLab/source/isaaclab/test/sim/test_spawn_wrappers.py:130-161`; `/home/hac/isaac/IsaacLab/source/isaaclab/test/sim/test_simulation_stage_in_memory.py:146-201`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers.py:131-190`
[^4]: `/home/hac/isaac/IsaacLab/docs/source/how-to/multi_asset_spawning.rst:93-113`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py:45-67`
[^5]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:117-148,1506-1606`; `/home/hac/isaac/IsaacLab/source/isaaclab/test/assets/test_articulation.py:238-245,296-303`
[^6]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:41-49,58-59,317-430`
[^7]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py:32-43,74-77,183-216,247-265,372-393`
[^8]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:321-347`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:589-596`; `/home/hac/isaac/IsaacLab/docs/source/overview/core-concepts/task_workflows.rst:9-11`
[^9]: `/home/hac/isaac/IsaacLab/source/isaaclab/test/performance/test_robot_load_performance.py:37-54`; `/home/hac/isaac/IsaacLab/scripts/tutorials/03_envs/create_cube_base_env.py:250-266,288-291`; `/home/hac/isaac/IsaacLab/source/isaaclab/test/envs/test_scale_randomization.py:265-266,339-350`
[^10]: `/home/hac/isaac/AnyMani/source/anymani/papers/inhand/Patel和Song - 2024 - GET-Zero Graph Embodiment Transformer for Zero-shot Embodiment Generalization.pdf`, pp. 1-3; extracted text at `/tmp/1776243927560-copilot-tool-output-0hpfkc.txt:35-43`
[^11]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py:15-35`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers.py:22-29,74-125`
[^12]: `/home/hac/isaac/AnyMani/source/anymani/papers/grasp/Fei 等 - 2025 - T(R,O) Grasp Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment.pdf`, pp. 2-4, 11-12; extracted text at `/tmp/1776243927560-copilot-tool-output-0hpfkc.txt:94-104,123-128`
[^13]: `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py:113-121`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:79-99`; `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py:26-36,80-109`; `/home/hac/isaac/IsaacLab/docs/source/how-to/import_new_asset.rst:92-104`; `/home/hac/isaac/IsaacLab/scripts/tools/convert_instanceable.py:7-27`
[^14]: `/home/hac/isaac/AnyMani/source/anymani/ideas/graph/idea.md:17-54,58-77,90-118`
