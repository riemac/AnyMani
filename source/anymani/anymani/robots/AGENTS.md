# AGENTS.md

`robots` 是 AnyMani 的 embodiment adapter 层：它消费 `assets.bank` 交付的资产 bundle 与静态语义，负责运动学、几何载体和 Isaac Lab articulation 的动态解释。

## 边界

`robots` 拥有两类实现：

- simulator-independent：`geometry_kinematics.py` 的批量 $SE(3)$/POE、当前轴线、owner 位姿与点 Jacobian；`owner_geometry.py` 的 owner-local collision union、表面和锚点缓存；
- Isaac Lab runtime：`hand_spawn.py` 与具体 hand cfg，把 bank selection lower 成 articulation/spawner 配置。

资产生成、sidecar schema、PALM/JOINT/TIP 人工语义、collision component ID 与 asset split 属于 `assets`。query/field target、神经网络、loss、trainer 与 checkpoint 属于 `distill`。scene、observation、action、reward、reset 和 termination 属于 `tasks`。

依赖方向固定为 `assets -> robots`。`robots` 不 import `tasks`、`distill` 或 `Research/`，也不复制资产生成器的 builder/validator/exporter 逻辑。

## 静态到动态

下游几何路径以 `HandContainer.geometry_semantics` 为唯一静态入口。需要运动学/几何时，调用方必须通过 `HandBankCfg.require_geometry_semantics=True` 请求它：generated 旧 sidecar 可由 bank 确定性迁移，official 缺人工核验字段时严格拒绝。

`HandGeometrySemanticsCfg` 保存 `{a}->{h}`、完整 fixed/revolute 导出链、显式 $q_{home}$、limits、owner/component 与 anchor seed。`robots` 将其 lower 为：

- 基准 `{h}` 空间旋量、owner home transforms 与祖先掩码；
- owner 图距离和 component-to-owner 变换；
- 同 owner 严格 Manifold Boolean union；
- 可复现的 boundary-only home points 与 palm surface/interior anchors。

不要让 `distill` 重新解析 `hand.yaml`、URDF、link 名或猜测 TIP。不要让 `robots` 根据训练 batch、field 带宽或网络结构改写资产语义。

## 物理约定

- `{a}` 是 raw asset/root frame，`{h}` 是 hand semantic frame；$p_h=R_{ha}p_a+t_{ha}$。
- 长度统一为 m，关节角与 RPY 统一为 rad；RPY 只用于明确的 URDF 固定轴 $R_zR_yR_x$ 边界。
- $q_{home}$ 是 POE/URDF 运动学参考，不要求落在控制 limits 内；limits 只服务合法采样与控制。
- fixed descendants 吸收到 link/owner home 变换，不作为零旋量混入活动 JOINT 轴。
- 非祖先 owner–JOINT Jacobian 必须精确为零；不同手指不能因全局 joint 顺序互相污染当前轴线。
- 同 owner 多 solid 必须使用真实 Boolean union；输入或输出非 volume 时严格失败，不得回退到 convex hull、包围盒或 buried-face heuristic。
- 不同 owner 永不做几何 union，因为最近点来源和一阶 Jacobian 必须可追踪。

## 测试

纯 Python/PyTorch 合同放在 `robots/tests/contracts/`，默认 pytest 收集。重点验证 asset-to-robot lowering、非零 home、branched POE、有限差分 Jacobian、owner coverage、Boolean 边界、surface-only 与 anchor provenance。

依赖 Isaac Sim、USD、PhysX handle 或 importer pose 的命题放在 `source/anymani/anymani/smokes/robots/`，只通过显式 Isaac Lab 命令运行。不要在普通测试模块导入 `hand_spawn.py` 或启动 `AppLauncher`。

## 设计风格

保持 simulator-independent 几何模块可由普通 Python 导入。`robots/__init__.py` 继续使用 lazy export，避免包初始化时加载 Isaac Lab。公共数学接口写清 frame、变换方向、shape、单位和结构零；性能优化必须先证明不改变这些合同。
