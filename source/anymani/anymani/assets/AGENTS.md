# AGENTS.md

本文件描述 `AnyMani/source/anymani/anymani/assets/` 子项目的当前开发约定。

## 子项目定位

本子项目服务于**大批量手部资产生成**。当前已经不只是“把 URDF 拼出来”，还承担：

- pre-made topology / connectivity 空间的系统性生成；
- post-mutate 几何与运动学局部派生；
- validator 对机械合理性的显式闸门；
- exporter / sidecar / summary 的可追溯落盘；
- `asset_physics.py` 对最终 collision 几何的动力学闭包。

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
- **Generator**：最高 façade，只编排阶段，不吞并各阶段职责。

尤其注意：**动力学闭包不要再塞回 builder、mutator 或 exporter**。  
它现在有专门的落点：`assets/asset_physics.py`。

### 3. 及时出清，避免臃肿

本子项目允许阶段性过渡实现，但一旦更高层 contract 已稳定，就应**立即删除旧实现、旧字段、旧测试和旧注释**。不用的东西不应作为“历史说明”继续留在代码里；它会污染科研语义，让读者误以为旧路线仍有建模价值。

具体要求：

- 职责已经上收后，旧层只允许保留当前 contract 需要的最小入口；旧算法、旧配置字段、旧 metadata 和旧 helper 必须删除；
- 不要用注释反复解释“以前这里做过什么但现在不做”。除非用户明确要求迁移期兼容，否则废弃路线在代码中视为本不应存在；
- 出清不是加 deprecated 壳，也不是补一段历史说明，而是把读者会误读成有效建模选择的痕迹直接移除；
- 若某段旧逻辑确实仍承担过渡职责，必须先和用户确认保留期限与删除条件，再写入注释。

### 4. 注释要求

本目录遵循 `annotation` skill。

对科研核心代码的额外要求：

- 公式、坐标系、量纲、数值锚点写在代码附近；
- `TODO / DONE / NOTE / Question / ref` 不得丢失科研语义；
- 若实现改变了物理或几何 contract，注释与文档要同步更新。

### 5. 自包含性

资产生成相关实现、文档、版本记录保持在 `AnyMani/source/anymani/anymani/assets/` 内部，不把子项目知识散落到别处。

## 文档与版本

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
