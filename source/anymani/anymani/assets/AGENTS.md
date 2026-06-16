# AGENTS.md

本文件描述 `AnyMani/source/anymani/anymani/assets/` 子项目的当前开发约定。

## 子项目定位

本子项目服务于**大批量手部资产生成**。当前已经不只是“把 URDF 拼出来”，还承担：

- pre-made topology / connectivity 空间的系统性生成；
- post-mutate 几何与运动学局部派生；
- validator 对机械合理性的显式闸门；
- exporter / sidecar / summary 的可追溯落盘；
- `asset_physics.py` 对最终 collision 几何的动力学闭包；
- asset bank 对已落盘资产集合做路径解析、bundle 校验、虚拟视图和可复现选择，供下游任务模块消费。

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
- **Asset Bank**：整理已落盘资产集合，负责路径解析、bundle 校验、虚拟视图和可复现选择；
- **Generator**：最高 façade，只编排阶段，不吞并各阶段职责。

尤其注意：**动力学闭包不要再塞回 builder、mutator 或 exporter**。  
它现在有专门的落点：`assets/asset_physics.py`。

### 3. 自包含性

资产生成相关实现、文档、版本记录保持在 `AnyMani/source/anymani/anymani/assets/` 内部，不把子项目知识散落到别处。

### 4. 测试优先

`assets` 子项目已经形成较多测试。后续凡是改动 builder / mutator / validator / exporter / physics closure / sidecar schema，应优先补最小单元测试或回归测试，再改实现。测试重点是几何数值、拓扑顺序、导出 contract、mass / inertia 闭包与 validator 拒绝条件，而不是启动 Isaac Sim。

涉及大批量生成流程时，优先测试局部 deterministic contract；完整生成 run 可作为较少量 smoke，不应替代底层规则测试。

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
