<today_task>
1. pre-made 修正
   - quick.py 摒弃封装
   - 左右手包含
   - 产出文件夹架构、行为、命名重修改
    > pre-made, post-mutate, mixed, missing 
   - Validator 增添规则，增添 palm preset 和 thumb preset 的绑定检查
2. post-mutate 打通
   - 各变异算法的实现
   - HandMutatorCfg 与 MutatorBase的配置类解耦
   - 蒙特卡洛联合分布采样策略
   - Validator 划分为 pre-made validator 和 post-mutate validator（但还是在同一个 HandValidatorCfg 里）

3. 某一具体拓扑的不同变体层次通才专家的IsaacLab训练验证（后续再做）
   > 见 `AnyMani/source/anymani/anymani/assets/doc/资产生成与训练管线概览.png`
   - 
  
</today_task>

<details>

# pre-made 修正

## quick.py 摒弃封装

我不需要 `QuickRunCfg` 这层包装了。我希望 `quick.py` 顶部直接就是一个 `HandGeneratorCfg` 实例，我想改什么就改它的字段，改完直接 `python quick.py` 就能跑。

具体来说，`quick.py` 顶部应该长这样——把 `HandGeneratorCfg` 里我最常调的字段先单独声明出来，然后一起传进去构造 cfg：类似如下

```python
# ── 用户可编辑区 ──
HAND_PRESETS = ["single_palm_allegro", "single_palm_leap"]
CONNECTIVITY_PRESETS = None   # None = 自动展开全部合法 slot-level connectivity
MIXED = True
MISSING = True
RECOLORED = "anatomy_v1"
SAMPLING_STRATEGY = "enumerate"
MAX_ENUMERATE = None
ARTIFACT_LEVEL = "bundle"
OUTPUT_LAYOUT = "recursive"

_SHOW_REGISTRY = True
_PRINT_RESULT_LIMIT = 40

# ── 构造 cfg ──
RUN_CFG = HandGeneratorCfg(
    hand_presets=HAND_PRESETS,
    connectivity_presets=CONNECTIVITY_PRESETS,
    mixed=MIXED,
    missing=MISSING,
    recolored=RECOLORED,
    sampling_strategy=SAMPLING_STRATEGY,
    max_enumerate=MAX_ENUMERATE,
    artifact_level=ARTIFACT_LEVEL,
    output_layout=OUTPUT_LAYOUT,
)
```

这样我打开 quick.py 只需要改最上面那几行大写变量就行，不用翻到下面去找。

`QuickRunCfg` 类和 `make_generator_cfg()` 函数整体删掉。`enumerate_premade_bundles / main / print_registry_summary / print_result_summary` 的参数都改成接受 `HandGeneratorCfg`。`make_full_only_connectivity_presets()` 保留不动。

`tests/test_generator_quick.py` 也要跟着改，去掉 `QuickRunCfg` 的 import，直接用 `HandGeneratorCfg` 构造。

## 产出文件夹架构、行为、命名重修改

### handedness

`HandGeneratorCfg` 新增字段 `handedness: Literal["left", "right", "all"] = "all"`。`all` 同时生成左右手，`left` / `right` 只生成单一 handedness。这个字段会影响目录命名——每个拓扑文件夹名都要带 `right_` 或 `left_` 前缀。

### 顶层架构：时间戳文件夹

目前每次产出都直接写进 `generated/pre_made/`，下一次会覆盖上一次。我希望改成：每次运行在 `generated/` 下新建一个时间戳文件夹（`2026-03-10_13-48-46` 这种格式），这样不同次的产出互相不干扰。

时间戳文件夹内的结构：

```
2026-03-10_13-48-46/
├── summary.yaml          # 记录本次产出的总体情况
├── mixed/                # mixed-family 产物
├── single_palm_allegro/  # allegro hand_preset
└── single_palm_leap/     # leap hand_preset
```

`summary.yaml` 记录这次跑的是 pre-made 还是 post-mutate 还是两者联合，以及用了什么配置、产出了多少手型等信息。

### missing 合并

`missing/` 不再作为顶层文件夹单独存在。missing 的本质是某个 `hand_preset` 缺了一根手指，它就是那个 hand 的拓扑变体，应该放在对应 `hand_preset` 下面。比如 `single_palm_allegro/right_t3_i2_m2`（没有 ring，一目了然）。

### single-family 命名

在 `single_palm_allegro/` 或 `single_palm_leap/` 下面，拓扑变体的文件夹名不再重复 family 名（父目录已经说了 allegro），改用 `<handedness>_t<n>_i<n>_m<n>_r<n>` 的格式，比如 `right_t3_i2_m2_r4`。

missing 的体现就是自然地缺少某个 finger 的标记：`right_t3_i2_m2`（没有 `_r<n>` 就是 ring 没了）。

如果同一个数量拓扑有多种不同的 joint-delete 方案（比如都是 t3 但删的关节不同），用下划线后缀消歧：`right_t3_i2_m2_r4_2`，表示这种数量组合下的第 2 种 delete recipe。

### mixed 命名

mixed 比较复杂，分两层。第一层按 family 组合分组，第二层按具体拓扑：

```
mixed/
├── allegro_single_palm_allegro_thumb_leap_index_ring_middle/
│   ├── right_allegro_t4_leap_i3_r4_m4/
│   │   └── 79abf680/
│   └── ...
└── ...
```

命名格式是通用的——`<handedness>_<thumb_family>_t<n>_<non_thumb_family>_i<n>_<non_thumb_family>_m<n>...`，它能描述任意可以想象的 mixed 组合。实际跑出来的产物范围由 Validator 过滤（比如 palm-thumb 绑定规则），而不是命名格式本身去限制。

### post-mutate 在同一时间戳内

post-mutate 不另建新的时间戳文件夹。它作用在某个 pre-made 拓扑变体上，在那个拓扑文件夹下面生成多个变体子文件夹（联合采样，不是笛卡尔积，所以不需要再递归了）：

```
right_t3_i1_m1_r2/
├── 79abf680/        # pre-made 原始产物
├── a1b2c3d4/        # post-mutate 变体 1
├── e5f6g7h8/        # post-mutate 变体 2
└── ...
```

行为上有两种使用方式：
- **联合运行**：先 pre-made 再 post-mutate，中间有运行时 HandCfg，post-mutate 直接在内存里修改，不需要从文件读取。
- **独立运行**：单独对已有的 pre-made 产物（某个子文件夹里的 urdf / HandCfg）进行 post-mutate。这种方式需要产物文件夹里保留足够的信息让 post-mutate 能重建 HandCfg（要么从 sidecar yaml 恢复，要么从 urdf 提取）。

独立运行的信息恢复机制（是否需要 ExtractorCfg 之类的）暂时不在本轮讨论，先把架构定下来。

## Validator 增添规则：palm-thumb 绑定

mixed 不是任意 mixed。对于 non-thumb fingers，可以 leap/allegro 随意混合，但 thumb 必须和 palm 绑定——allegro palm 配 allegro thumb，leap palm 配 leap thumb。原因是 leap 和 allegro 的 thumb 实际上是同一种类型（`RegularThumbBuilderCfg`），只是参数不同，而 palm 对 thumb 的挂载点影响很大，不同 family 的 palm + thumb 组合在物理上不太合理。

所以 Validator 里要加一条规则：如果 `hand_preset` 是 `single_palm_allegro`，那么 thumb 的 finger preset 也必须是 allegro family 的；反之 leap palm 只能配 leap thumb。non-thumb slots（index / middle / ring）不受这条限制，它们可以跨 family。

相关代码在 `/home/hac/isaac/AnyMani/source/anymani/anymani/assets/validator/` 里，需要在现有的 hand-level validation 规则里新增这条 palm-thumb family 一致性检查。

# post-mutate 打通

## HandMutatorCfg 与各 MutatorTermCfg 的解耦

目前 `HandMutatorCfg`（在 `mutate/pipeline.py` 里）把所有已实现的变异工具都硬编码成固定字段（`joint_delete: JointDeleteCfg | None`、`link_scale: LinkScaleCfg | None`……），而且 fallback 版在 `hand_generator.py` 里用 `object | None` 做占位。这不好——每新增一个变异工具就要去改 `HandMutatorCfg` 加字段。

我希望改成类似 IsaacLab `RewardsCfg` + `RewardTermCfg` 的风格：`HandMutatorCfg` 本身是一个容器基类，不预声明具体的工具字段。用户在使用时，像写 reward term 一样，把各个 `MutatorTerm(...)` 作为类属性声明上去：

```python
@configclass
class MyMutatorCfg(HandMutatorCfg):
    scale = MutatorTerm(
        cfg=LinkScaleCfg(mode="relative", sigma=0.1),
    )
    tip = MutatorTerm(
        cfg=TipReplaceCfg(strategy="geometry_swap"),
    )
    limit = MutatorTerm(
        cfg=LimitTweakCfg(sigma=0.05, symmetric=True),
    )
```

这样新增工具只需要写好自己的 `*Cfg` 和 `*Mutator`，不用碰 `HandMutatorCfg`。`order`、`on_reject`、`step_validate` 这些编排参数留在 `HandMutatorCfg` 基类里。

另外，`joint_delete` 属于 pre-made 阶段的拓扑操作，不是 post-mutate 的连续参数变异，应该从 `HandMutatorCfg` 里移走，归入 connectivity_presets 体系。`finger_replace` 如果确认没有实际被调用过，也一并删掉。

## 各变异算法的分布接口统一

当前每个 mutator 工具内部各自调 `random.gauss()` / `random.uniform()` 做采样，分布类型和参数都硬编码在实现里。我希望统一成：每个 `MutatorTermCfg` 都有显式的分布相关字段——服从什么分布（均匀/正态/……）、分布参数是什么（如 `low/high` 或 `mean/sigma`）。

这样做的目的是为了蒙特卡洛联合采样：`HandGenerator` 可以把各个工具的分布拼成一个独立联合分布，一次采样就出一组完整参数，而不是各工具内部各管各的随机。具体的分布字段设计每个工具可以不同（比如 `link_scale` 用连续分布，`joint_delete` 可能用离散分布），但接口要有统一的基类约定。

算法本身（删关节重连、缩放 origin、替换指尖等）目前的实现基本够用，但需要按这个分布接口重构一下——把随机采样从算法逻辑里剥离出来，算法本身变成纯确定性变换，随机性全部来自上游传入的采样值。

## 蒙特卡洛联合分布采样策略

在 `sampling_strategy="sample"` + `n_samples=N` 的模式下，`HandGenerator` 的行为是：对于每个 pre-made 拓扑变体，把所有启用的 mutator 工具的分布组成一个独立联合分布，蒙特卡洛采样 N 次，每次产出一个变异个体。这和 pre-made 的 grid search 枚举是对立的——pre-made 走笛卡尔积，post-mutate 走随机采样。

组合联合分布和执行采样的逻辑应该放在 `HandGenerator`（或它调用的一个采样器辅助模块里），而不是放在各个 mutator 内部。

## Validator 显式划分为 pre-made 和 post-mutate 阶段

目前 `HandValidatorCfg` 里的规则没有区分阶段，在 `_generate_once()` 里只在 mutate 之后做一次 validate。我希望把 validator 规则显式分成两组，类似 IsaacLab 的 `ObservationsCfg` 里有 `PolicyCfg` 和 `CriticCfg` 两个内嵌子类的风格：

```python
@configclass
class HandValidatorCfg:

    @configclass
    class PreMadeCfg:
        """pre-made 阶段的结构性校验规则"""
        palm_thumb_binding = ValidatorTerm(...)
        topology_completeness = ValidatorTerm(...)
        ...

    @configclass
    class PostMutateCfg:
        """post-mutate 阶段的几何/参数合理性校验"""
        link_length_positive = ValidatorTerm(...)
        limit_ordering = ValidatorTerm(...)
        ...

    pre_made: PreMadeCfg = PreMadeCfg()
    post_mutate: PostMutateCfg = PostMutateCfg()
```

`_generate_once()` 在 connectivity lower 之后调 `pre_made` 规则，在 mutate 之后调 `post_mutate` 规则。上面提到的 palm-thumb 绑定检查属于 `PreMadeCfg`。

</details>