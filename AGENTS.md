# AGENTS.md

本文件为 AnyMani 项目的架构说明，面向 AI Agent 提供项目背景、结构和约定。

---

## 项目简介

**AnyMani** 是基于 Isaac Lab 的灵巧操作研究框架，专注于手内物体操作（in-hand manipulation）任务。

### 核心特性

- **多动作空间**：关节空间、SE(3) 旋量、仿射编队等
- **多手型支持**：LeapHand、LeapHand-Round、未来扩展 Allegro、Shadow 等
- **模型复现**：稳定版本快照机制，解决训练模型无法复现问题
- **模块化设计**：MDP 组件解耦，便于组合新环境

### 科研背景

用户为科研人员，研究方向：强化学习训练灵巧操作策略，策略蒸馏 → 模仿学习 → 跨手型/跨物体泛化

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
│   │   ├── direct/             # Direct workflow 任务
│   │   └── functional/         # Functional workflow 任务
│   ├── robots/                 # 机器人资产定义
│   │   └── leap/               # LeapHand URDF/USD
│   └── distillation/           # 在线蒸馏框架
├── assets/                     # USD 资产文件
├── config/                     # 全局配置
└── docs/                       # 文档
```

### 架构层级

#### 1. MDP 组件库（`inhand_env_cfg.py`）

**职责**：定义可复用的 MDP 配置组件，不包含手型特定信息。

**组件分类**：
- **场景**：`InHandObjectSceneCfg`、`TactileSceneCfg`
- **观测**：`JointSpaceObservationsCfg`、`Se3ObservationsCfg`、`TactileObservationsCfg`、`Se3TactileObservationsCfg`
- **动作**：`JointSpaceActionsCfg`、`Se3ActionsCfg`、`Se3EmaActionsCfg`、`AffineActionsCfg`
- **奖励**：`CommonRewardsCfg`、`Se3RewardsCfg`、`TactileRewardsCfg`、`Se3TactileRewardsCfg`
- **事件**：`CommonEventCfg`（域随机化）
- **终止**：`CommonTerminationsCfg`
- **命令**：`ContinuousRotationCommandsCfg`
- **课程**：暂无

#### 2. 手型配置（`config/leaphand/leaphand_env_cfg.py`）

**职责**：通过组合 MDP 组件定义 LeapHand 的环境变体。

**环境变体**：

| 环境 ID | 类名 | 动作空间 | 观测 | 说明 |
|---------|------|----------|------|------|
| `AnyMani-LeapHand-Joint-v0` | `LeapHandJointEnvCfg` | 关节空间 (16维) | 本体感受 | 基线 |
| `AnyMani-LeapHand-SE3-v0` | `LeapHandSe3EnvCfg` | SE(3) 旋量 (24维) | 本体感受 | 指尖 6D 控制 |
| `AnyMani-LeapHand-Tactile-v0` | `LeapHandTactileEnvCfg` | 关节空间 | 本体感受 + 触觉 | 触觉反馈 |
| `AnyMani-LeapHand-SE3-Tactile-v0` | `LeapHandSe3TactileEnvCfg` | SE(3) | 本体感受 + 触觉 | SE(3) + 触觉 |
| `AnyMani-LeapHand-Affine-v0` | `LeapHandAffineEnvCfg` | 仿射编队 (9维) | 本体感受 | 编队控制 |

所有环境都有对应的 `*-Play-v0` 变体用于评估（`num_envs=50`，禁用噪声）。

#### 3. 稳定版本机制（`leaphand_stable_env_cfg.py`）

**用途**：保存训练成功的环境配置快照，解决模型复现问题。

**工作流**：
1. 使用 `leaphand_env_cfg.py` 中的配置训练模型
2. 效果满意后，手动复制配置到 `leaphand_stable_env_cfg.py`
3. 添加版本后缀（如 `V1`、`V2`）
4. 在 docstring 中记录：
   - 创建日期、git commit
   - 模型保存路径
   - obs_dim、action_dim
   - 关键超参数
5. 标记 `⚠️ DO NOT MODIFY`
6. 复现时导入稳定版本配置

---

## 关键约定

### 1. 分支管理

- **main**：稳定版本，只接受经过验证的合并
- **refactor**：当前重构分支（2026-02-02 至今）
- **ik**、**temp**：实验分支

### 2. 代码风格

- 主要基于 `ManagerBasedRLEnv` 环境架构开发
- 部分功能测试验证基于 `standalone app launcher` 开发
- 继承 Isaac Lab 的配置类风格：`@configclass` + `__post_init__`

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
| **Hora** | IsaacGym 手内旋转 | `/home/hac/isaac/hora` |
| **rl_games** | RL 算法库 | `/home/hac/isaac/rl_games` |
