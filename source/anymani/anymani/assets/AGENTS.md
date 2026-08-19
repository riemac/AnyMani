# AGENTS.md

本文件描述 `AnyMani/source/anymani/anymani/assets/` 子项目的当前开发约定。

## 子项目定位

本子项目服务于**大批量手部资产生成**。当前已经不只是“把 URDF 拼出来”，还承担：

- pre-made topology / connectivity 空间的系统性生成；
- post-mutate 几何与运动学局部派生；
- validator 对机械合理性的显式闸门；
- exporter / sidecar / summary 的可追溯落盘；
- `asset_physics.py` 对最终 collision 几何的动力学闭包；
- asset bank 对单项或直接 source root 做路径解析、bundle 校验和虚拟视图，asset dataset 在其上冻结跨 run 的实验 partitions 与 evaluation suites。

## 核心原则

### 1. 声明式配置驱动

主要实现以 `@dataclass` 配置类 + 关联运行时类为主。

谨慎写：

- 大量过程式脚本把逻辑揉在一起；
- 尚未稳定的研究想法直接硬编码进主流程；
- 把“几何/运动学语义”和“动力学闭包语义”混在一个模块里。

原则是先把**结构、约束、阶段边界**讲清楚，再把生成、验证、导出逐层接上。

### 2. 职责解耦

资产子项目当前的稳定分层是：

- **Builder**：构造 canonical `HandCfg`；
- **Mutator**：局部几何 / 拓扑 / 参数派生；
- **Validator**：显式拒绝不合法资产；
- **Exporter**：URDF / sidecar / tree / mesh materialization；
- **Physics Closure**：`asset_physics.py`，只负责由最终 collision 几何重建 `mass / inertial`；
- **Asset Bank / Dataset**：bank 负责低层路径解析、bundle 校验与虚拟视图；dataset manifest 负责 partition 展开，`generator/dataset_build` 负责从 inventory 分层选择 lineages、并行 post-mutate 与发布 provenance；
- **Generator**：最高 façade，只编排阶段，不吞并各阶段职责。

尤其注意：**动力学闭包不要再塞回 builder、mutator 或 exporter**。  
它现在有专门的落点：`assets/asset_physics.py`。

pre-made topology 把 family composition 与 missing slots 视为正交轴。thumb 与 base palm 共享 family，但不参与 mixed 判定；只有存活 non-thumb 同时包含 LEAP 与 Allegro 才是真正 mixed，全为 base family 或全为 opposite family 都必须退出 mixed 空间。程序化 mesh 虽然必须在 physics closure 与 validator 前物化，但此时仍属于候选期文件；拒绝或异常候选只能回滚本次新写文件，不能在 generated 资产树中留下 OBJ-only 半成品。

### 3. 几何语义交付

`asset_schema_geometry.py` 定义版本化 `{a}->{h}`、完整 fixed/revolute 链、显式 $q_{home}$、limits、PALM/JOINT/TIP owner、collision component 与 anchor seed。exporter 在 `HandCfg` 真源仍在内存时写入 `hand.yaml.geometry_semantics`；bank 只在 `require_geometry_semantics=True` 时解析，新 generated sidecar 直接读取，旧 generated sidecar 确定性迁移，official 缺人工核验字段时严格拒绝。

`HandBank` 是单 bundle、单 source root 与显式 container 选择的低层交付入口；`HandAssetDataset` 读取 schema 2.0 最终 manifest，`generator/dataset_build` 读取 typed template 并冻结 selection lock。模板控制 cohort 分布和 post-mutate 数量，但具体 mutator/validator/physics 仍由 `HandGeneratorCfg` 定义。三者均保持下游中立；不要让 tasks/distill 重做目录展开，也不要把动态 FK/Jacobian、field/query、spawn 或 optimizer 逻辑放进 bank。

### 4. 自包含性

资产生成 contract 的权威实现、README、VERSION 与 CHANGELOG 保持在本目录。下游 `robots/tasks/distill`
可以引用导出的 URDF、sidecar、bank schema 与版本锚点，但不得复制或重新定义生成/验证/physics closure 逻辑。

### 5. 测试优先

`assets` 子项目已经形成较多测试。后续凡是改动 builder / mutator / validator / exporter / physics closure / sidecar schema，应优先补最小单元测试或回归测试，再改实现。测试重点是几何数值、拓扑顺序、导出 contract、mass / inertia 闭包与 validator 拒绝条件，而不是启动 Isaac Sim。

涉及大批量生成流程时，优先测试局部 deterministic contract；完整生成 run 可作为较少量 smoke，不应替代底层规则测试。

## 约定

### 资产命名空间

| 名称 | 目录或组成 | 固定含义 |
| --- | --- | --- |
| **Generation run（生成批次）** | `generated/<premade_timestamp>/` | 一次 pre-made 生产运行，不代表手型 family 或训练 split。 |
| **Production group（生产组）** | `<generation_run>/<group>/` | generator 的目录组织，如 `single_palm_leap`；不得用作实验资产集合名称。 |
| **Mother asset（母体资产）** | `<group>/<topology>/hand.urdf` | 可直接消费的 pre-made 基准手，也是对应 post-mutate 的唯一来源资产。 |
| **Variant set（变体集）** | `<mother>/<mutate_timestamp>/` | 同一 mother、同一次 post-mutate 配置与随机过程产生的完整批次，由 `summary.yaml` 记录 provenance。 |
| **Variant asset（变体资产）** | `<variant_set>/<asset_id>/hand.urdf` | variant set 中一只具体且可独立消费的手资产。 |
| **Mother lineage（母体系）** | 一只 mother 及其全部 variant sets | 生成谱系概念，不等同于某次实验实际使用的数据。 |
| **Asset cohort（实验资产集合）** | 从多个 mother lineages 中显式选取的 mothers / variant sets | 一次训练或评估主动选择的数据集合；不用 `group` 或 `cluster` 指代。 |
| **Dataset partition（数据划分）** | cohort 的 `train` / `validation` / `evaluation` 子集 | 训练角色，不改变资产自身的生成身份与 family 身份。 |

`family` 表示 base palm 的来源/机制族；完整 topology 必须结合 `family_composition`、`missing_slots`、`surviving_slots` 与 `slot_family_map` 判断，不能只读顶层 `family`。`topology_kind` 只保留为历史 sidecar/summary 的派生兼容标签。dataset YAML 以完整 variant set 为配置原子，并对每条 lineage 显式声明 `include_mother`；跨 partition 先按路径、asset ID 与 `content_hash` 拒绝重复，geometry consumer 再按 `physical_geometry_hash` 拒绝相同物理映射。

Dataset partition 以 canonical left/right mirror pair 为最小 morphology 分配单位；同一 pair 不得拆到 train 与 unseen-mother holdout。selection lock、generator config hash 与 build report 是正式 manifest 的 provenance，schema-2 `.build_state.yaml` 仅用于本地中断恢复，不进入版本库；每个 owned variant-set run 还必须有 `DATASET_BUILD_ATTEMPT.yaml` marker，rollback/adopt 只能依据 marker 与 lock/state 三方一致证据执行。

Post-mutate 的 identity mode 是合法 proposal 语义，不由 mutator 自行删除。`HandGeneratorCfg.post_mutate_require_unique_geometry=False` 允许研究者保留重复 identity 加权样本；正式资产数据集 recipe 设为 `True`，以 mother 和当前 variant set 已接受样本的静态 geometry fingerprint 为禁集逐槽补抽。该局部闸门不取代 `generator/dataset_build` 的跨 mother/partition 全局唯一性检查。

正式大批量 post-mutate 使用 `post_mutate_sdf_execution="central_gpu_batch"`：只有一个 spawn GPU service 可以初始化 PyTorch/Warp，CPU workers 不得初始化 CUDA，且每个 worker 处理一只 mother 后退出。central service 的通信失败、超时、GPU backend failure 或 batch/scalar parity failure 必须穿透普通 rejection sampling，立即停止整个 build；不得自动回退到 CPU 或 local CUDA validator。

### 文档与版本

从 **0.1** 开始，本子项目显式维护：

- `VERSION`：当前资产子项目版本；
- `CHANGELOG`：对外可读的行为变化记录；
- `README`：面向新人/协作者的快速入口；
- `AGENTS.md`：面向 AI agent 的约定与红线。

凡是影响生成 contract、目录 contract、physics closure、validator 规则、导出语义的变更，都应同步更新这些文件。

## 实用建议

- 遇到需求不明确时，优先提问而不是自行假设并实现。
- 用户装了 URDF Visualizer 的 VSCode 扩展，可以直接 3D 预览 URDF 资产。
- 若改动涉及 `generator/`，继续向下读取更近一层的 `assets/generator/AGENTS.md`。

## 未来展望

待稳定成熟后，计划将 assets 子项目单独沉淀为可复用 python 库。
