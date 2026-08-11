# AGENTS.md

`distill` 是 AnyMani 的训练、共享模型与学习目标层。它消费 `assets -> robots -> tasks` 已定义的资产、embodiment 与环境接口，不反向接管这些层的物理职责。

## 边界

`distill` 拥有物理场/query/target、可学习模型、objective、SSL/IL/RL 编排、checkpoint 与实验诊断。

不要在这里解析 `hand.yaml`、URDF、link 名或推断 PALM/JOINT/TIP；静态语义由 `assets.bank.HandContainer.geometry_semantics` 交付，批量 FK、owner 位姿、当前轴线、点 Jacobian 与 owner-local collision cache 由 `robots` 交付。不要用训练配置重新定义资产 owner 或运动学。

不要在这里改写 scene、observation、action、reward、reset 或 termination；这些属于 `tasks`。source code 不 import、解析或要求 `Research/` vault 存在。

## 子目录

| 目录 | 拥有 | 不拥有 |
| --- | --- | --- |
| `representations/` | field、query、target、mask 与 provenance | 资产/运动学真源、`torch.nn`、trainer |
| `models/` | input adapter、retained backbone、decoder、action/value heads | target generation、loss、MDP |
| `objectives/` | prediction/target/mask/weight 到 scalar loss | sampling、optimizer、advantage |
| `ssl/`、`il/`、`rl/` | 各训练阶段的数据流、配置、运行入口与 checkpoint | 共享模型的重复实现 |
| `diagnostics/` | run recording、固定 evaluation 与只读 analysis | 物理真值、trainer 与实验选择 |

移动内容时同步迁移 TODO、docstring 和 tests，保证 target、模型、loss、stage 与 retained/disposable 生命周期各有一个事实源。不要用通用 Manager façade 或深配置继承隐藏真实耦合。

## 几何 SSL

主线是 task-free、cross-embodiment 的多锚点条件隐式 Gaussian 场。部署保留输入只能包含当前物理 q 与静态手型证据；joint limits 只用于采样，不进入编码器。

训练期监督逐 owner 的多带宽邻近场 ρ、距离灵敏度 κ=∇q d、由链式法则派生的 g、同一密度预测器的 Sobolev/JVP 自导数与 chain consistency。current distance、最近点、surface Jacobian、query stratum、contact、action、history 和 object state 都不得进入 retained encoder。

PALM/JOINT/TIP entity、owner 与 decoder 轴同索引。physical anchors 在网络中是完整、无序、等地位的 K 集合；finger seed 只属于采样 provenance。home geometry 只含真实 owner union boundary，不混入 interior；interior 只允许进入 anchor 支持。

零阶表征对 joint-sign 成对改写为偶，一阶表征、κ/g 与对应动作坐标为奇。`{h}` 面内 SO(2) 重写不是 reflection；镜像手性不能被错误消除。所有 feature group 必须声明 frame、单位、reference 和变换方向。

input adapter、backbone 与零/一阶 heads 构成 retained geometry encoder。representation decoder、最近点教师和 target backend 只在 SSL 存在，导出到 PPO 时必须删除。RL、IL 与 SSL 不复制科研语义相同的 trunk。

official LEAP/Allegro 不参与 generated 训练、辅助权重校准或 checkpoint 选择；冻结后的 zero-shot 与独立 geometry-only adaptation 必须使用分离配置和证据。

## 入口与测试

当前 GM RL 入口仍为 `python -m anymani.distill.train` 与 `python -m anymani.distill.play`；`tasks/inhand` 的既有路线使用仓库根 `scripts/rl_games/train.py` / `play.py`。若未来迁移到 `rl/`，入口、文档与调用方必须在同一提交闭合。SSL/IL 入口只有在 resolved config、日志、checkpoint 和 sanity tests 闭合后才可声明可运行。

纯公式、query/target、模型、objective、checkpoint 与训练配置合同放在 `distill/tests/`。资产 lowering、FK/Jacobian 与 owner union 测试属于 `robots/tests/`。依赖 Isaac Sim 的 distill runtime 证据以后放在 `source/anymani/anymani/smokes/distill/`，不要混入默认 pytest。

隐式主线 retained encoder 的硬门槛为 RTX 5070 Ti、B=4096、单结构组、20 次预热和 50 次 CUDA Event，p95 不超过 40 ms。计时覆盖 adapter、集合聚合、backbone 与零/一阶 heads；排除磁盘/CPU cache materialization、decoder、target、policy、Isaac Sim 和 host-to-device copy。PPO full fine-tune 时不得缓存会随优化器更新而陈旧的 learned activation。

具体测试分层见 `tests/AGENTS.md`；人类阅读入口见 `README.md`、`representations/README.md`、`models/README.md` 与 `ssl/README.md`。
