# AGENTS.md

`robots` 是 AnyMani 的 Isaac Lab embodiment adapter。它消费 `assets.bank` 交付的 bundle，把手资产 lower 为 articulation/spawner 配置；不拥有 learning geometry source、任务 MDP 或训练算法。

## 所有权

| 路径 | 职责 |
| --- | --- |
| `hand_spawn.py` | HandBank selection 到 Isaac Lab articulation/spawner cfg 的公共 adapter |
| `leap.py`、`leap_urdf.py`、`leap_round_tip.py` | 具体机器人配置与兼容入口 |
| `tests/` | 不启动 Isaac Sim 的 robot cfg/spawn schema contracts |

资产生成、sidecar schema、PALM/JOINT/TIP 人工语义与 collision component ID 属于 `assets`。simulator-independent POE/FK/Jacobian、owner collision union、home/anchor realization、Warp cache 与 physical identity 属于 `distill.representations.sources`。scene、observation、action、reward、reset 和 termination 属于 `tasks`。

依赖方向固定为 `assets -> robots`。`robots` 不 import `tasks`、`distill` 或 `Research/`，也不复制 asset builder/validator/exporter 或 learning geometry source。

## 实现约束

- spawn 只消费 `HandContainer` 已解析的路径与 typed metadata，不重新解析 `hand.yaml` 来推断学习语义；
- 不根据训练 batch、field bandwidth、网络结构或任务 reward 改写 articulation；
- `{a}`/`{h}`、长度 m、关节角 rad 与 RPY 组合方向必须沿 assets schema 传递，不在 robot cfg 中创造第二套约定；
- `robots/__init__.py` 保持 lazy export，普通 Python import 不应提前加载 Isaac Lab runtime；
- 依赖 USD stage、PhysX handle、importer pose 或 reset/step 的命题放在显式 Isaac Sim smoke，不进入默认 pytest。

## 验证

纯配置测试只验证 bundle-to-spawn 路径、关节/执行器映射、frame/单位和 cfg schema。Isaac runtime smoke 放在 `source/anymani/anymani/smokes/robots/` 并通过 `isaaclab.sh` 显式运行。learning geometry 的 lowering/FK/Jacobian/owner-union contracts 统一位于 `distill/tests/contracts/representations/`。
