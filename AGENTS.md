# AGENTS.md

本文件为 AnyMani 项目的架构说明，面向 AI Agent 提供项目背景、结构和约定。

---

## 项目简介

**AnyMani** 是基于 Isaac Lab 的灵巧操作研究框架，专注于手内物体操作（in-hand manipulation）任务。

### 核心特性

- **多手型支持**：LeapHand、LeapHand-Round、未来扩展 Allegro、Shadow 等
- **模型复现**：稳定版本快照机制，解决训练模型无法复现问题
- **模块化设计**：MDP 组件解耦，便于组合新环境

---

## 项目架构（重构后）

### 目录结构

```
AnyMani/source/anymani/
├── anymani/                    # 主包
│   ├── tasks/                  # 任务定义
│   │   ├── inhand/             # 手内操作任务
│   │   │   ├── __init__.py
│   │   │   ├── inhand_env_cfg.py          # ⭐ MDP 组件库
│   │   │   ├── mdp/                       # MDP 函数实现
│   │   │   │   ├── observations.py        # 观测函数
│   │   │   │   ├── actions/               # 动作类实现
│   │   │   │   ├── rewards_task.py        # 任务奖励
│   │   │   │   ├── rewards_action.py      # 动作惩罚
│   │   │   │   ├── rewards_tactile.py     # 触觉奖励
│   │   │   │   ├── commands/              # 命令生成器
│   │   │   │   ├── recorders/             # 数据记录器
│   │   │   │   ├── events.py              # 随机化事件
│   │   │   │   └── terminations.py        # 终止条件
│   │   │   └── config/                    # 按手型分组
│   │   │       ├── leaphand/              # LeapHand 配置
│   │   │       │   ├── __init__.py        # Gym 注册
│   │   │       │   ├── leaphand_env_cfg.py         # ⭐ 环境变体
│   │   │       │   ├── leaphand_stable_env_cfg.py  # 稳定版本快照
│   │   │       │   └── agents/            # RL 算法配置
│   │   │       └── leaphand_round/        # 半球指尖变体
│   │   ├── direct/             # Direct workflow 任务，用于科研idea的快速原型设计
│   │   └── functional/         # 机制检验
│   └── robots/                 # 机器人资产定义
│       └── leap/               # LeapHand URDF/USD
├── assets/                     # USD 资产文件
├── config/                     # 全局配置
└── docs/                       # 文档
```

---

## 关键约定

### 1. 分支管理

- **main**：稳定版本，只接受经过验证的合并
- 其他分支, agent应用git自行判断

### 2. 代码风格

- 主要基于 `ManagerBasedRLEnv` 环境架构开发
- 目前基于 `DirectRLEnv` 环境架构开发，用于科研idea的快速原型设计
- 部分功能测试验证基于 `standalone app launcher` 开发
- 继承 Isaac Lab 的声明式配置风格：`@configclass` + `__post_init__`

### 3. 模块依赖

```
inhand_env_cfg.py (组件库)
    ↑
    │ 导入
    │
leaphand_env_cfg.py (环境配置)
    ↑
    │ 导入
    │
__init__.py (Gym 注册)
```

**原则**：
- 组件库不依赖手型配置
- 环境配置只组合组件，不定义新组件
- 禁止循环依赖

### 4. 命名规范

- **任务**：`inhand`、`grasp`（未来）
- **手型**：`leaphand`、`leaphand_round`、`allegro`（未来）
- **Gym ID**：`AnyMani-<Hand>-<Feature>-v0`

---

## 常用操作

### 列出所有环境

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
python scripts/list_envs.py
```

### 运行环境

```bash
# 随机 agent 测试,验证环境可用性
python scripts/random_agent.py --task AnyMani-LeapHand-Joint-v0 --num_envs 1 --headless

# RL 训练
python scripts/rl_games/train.py --task AnyMani-LeapHand-Joint-v0 --num_envs 4096 --headless
```

---

## 参考项目

| 项目 | 说明 | 路径 |
|------|------|------|
| **Isaac Lab** | 上游框架 | `/home/hac/isaac/IsaacLab` |
| **rl_games** | RL 算法库 | `/home/hac/isaac/rl_games` |