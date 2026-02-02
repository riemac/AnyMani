# AnyRotate → AnyMani 重构计划

> **创建时间**: 2026-02-02
> **状态**: 待执行

---

## 背景与目标

### 当前问题

1. **模型复现困难**：训练好的 `.pth` 因 `inhand_base_env_cfg.py` 改动无法复现
2. **任务文件繁杂**：7 个 `*_env_cfg.py`，基类牵一发动全身
3. **命名需更通用**：AnyRotate → AnyMani，支持未来更多机械手类型

### 解决方案

1. **架构重构**：采用 IsaacLab inhand 风格的 MDP 组合模式
2. **稳定版本管理**：`*_stable_env_cfg.py` + git tag
3. **目录重命名**：leaphand → anymani，按任务和手型分层

---

## 最终目录结构

```
AnyMani/source/anymani/tasks/
├── inhand/                              # 手内操作任务
│   ├── __init__.py
│   ├── inhand_env_cfg.py                # MDP 组件定义
│   │   ├── JointSpaceObsCfg / Se3ObsCfg
│   │   ├── JointSpaceActionsCfg / Se3ActionsCfg / AffineActionsCfg
│   │   ├── CommonRewardsCfg
│   │   └── 其他共享配置
│   │
│   ├── mdp/                             # MDP 函数实现（从现有代码迁移）
│   │   ├── __init__.py
│   │   ├── observations.py              # 自定义 obs 函数
│   │   ├── actions/                     # 自定义动作类（文件夹）
│   │   ├── rewards.py                   # 自定义 reward 函数
│   │   ├── recorders/                   # 自定义记录器（文件夹）
│   │   ├── events.py                    # 随机化事件
│   │   └── commands/                    # 命令生成器（文件夹）
│   │
│   └── config/                          # 按手型分
│       ├── leaphand/
│       │   ├── __init__.py              # Gym 环境注册
│       │   ├── leaphand_env_cfg.py      # 开发中的变体
│       │   │   ├── LeapHandJointEnvCfg
│       │   │   ├── LeapHandSe3EnvCfg
│       │   │   ├── LeapHandTactileEnvCfg
│       │   │   └── LeapHandRmaEnvCfg
│       │   ├── leaphand_stable_env_cfg.py  # 稳定版本（手动维护）
│       │   │   └── LeapHandJointStableV1Cfg
│       │   └── agents/                  # RL 算法配置
│       │       └── rl_games_ppo_cfg.yaml
│       │
│       └── leaphand_round/              # 半球指尖变体
│           ├── leaphand_round_env_cfg.py
│           └── agents/
│
├── inhand_float/                        # 臂手解耦任务（未来）
│   └── ...
│
└── grasp/                               # 抓取任务（未来）
```

---

## 任务分解

### Phase 1: 准备工作 (Day 0)

| ID | 任务 | 状态 |
|----|------|------|
| 1.1 | 确保 `refactor` 分支干净，无未提交更改 | ⬜ |
| 1.2 | 创建当前代码的备份 tag：`git tag pre-refactor-backup` | ⬜ |
| 1.3 | 确认当前可用的模型列表及对应的 env_cfg | ⬜ |

### Phase 2: 目录重命名 (Day 1)

| ID | 任务 | 状态 |
|----|------|------|
| 2.1 | `AnyRotate/` → `AnyMani/` | ⬜ |
| 2.2 | `source/leaphand/` → `source/anymani/` | ⬜ |
| 2.3 | 更新所有 `import leaphand` → `import anymani` | ⬜ |
| 2.4 | 更新 `pyproject.toml` / `setup.py` 中的包名 | ⬜ |
| 2.5 | 更新 Gym 环境注册名（如 `Isaac-Leaphand-*` → `Isaac-AnyMani-*`） | ⬜ |
| 2.6 | 验证：`python scripts/list_envs.py` 能列出所有环境 | ⬜ |

### Phase 3: 任务目录重组 (Day 1-2)

| ID | 任务 | 状态 |
|----|------|------|
| 3.1 | 创建 `tasks/inhand/` 目录结构 | ⬜ |
| 3.2 | 迁移现有 `mdp/` 到 `tasks/inhand/mdp/`（保持原有实现） | ⬜ |
| 3.3 | 创建 `tasks/inhand/config/leaphand/` | ⬜ |
| 3.4 | 创建 `tasks/inhand/config/leaphand_round/` | ⬜ |
| 3.5 | 迁移 `agents/` 到对应 config 目录下 | ⬜ |

### Phase 4: MDP 组件整理 (Day 2-3)

> **注意**：本阶段主要是**整理和复制现有代码**，而非重新设计实现

| ID | 任务 | 状态 |
|----|------|------|
| 4.1 | 设计 `inhand_env_cfg.py`：整理多套 ObsCfg/ActionsCfg 定义 | ⬜ |
| 4.2 | 从 `inhand_base_env_cfg.py` 提取 `JointSpaceObsCfg` | ⬜ |
| 4.3 | 从 `inhand_se3_env_cfg.py` 提取 `Se3ObsCfg` | ⬜ |
| 4.4 | 从 `inhand_base_env_cfg.py` 提取 `JointSpaceActionsCfg` | ⬜ |
| 4.5 | 从 `inhand_se3_env_cfg.py` 提取 `Se3ActionsCfg` | ⬜ |
| 4.6 | 从 `inhand_affine_env_cfg.py` 提取 `AffineActionsCfg` | ⬜ |
| 4.7 | 整理 `CommonRewardsCfg`（合并共享奖励配置） | ⬜ |
| 4.8 | 从 `inhand_tactile_env_cfg.py` 提取触觉相关配置 | ⬜ |

### Phase 5: 配置组装 (Day 3)

| ID | 任务 | 状态 |
|----|------|------|
| 5.1 | 创建 `leaphand_env_cfg.py`：组装各变体 | ⬜ |
| 5.2 | 实现 `LeapHandJointEnvCfg` | ⬜ |
| 5.3 | 实现 `LeapHandSe3EnvCfg` | ⬜ |
| 5.4 | 实现 `LeapHandTactileEnvCfg` | ⬜ |
| 5.5 | 实现 `LeapHandRmaEnvCfg` | ⬜ |
| 5.6 | 创建 `leaphand_stable_env_cfg.py`：迁移已验证的稳定配置 | ⬜ |
| 5.7 | 处理 `leaphand_round/` 手型变体 | ⬜ |

### Phase 6: Gym 注册与验证 (Day 4)

| ID | 任务 | 状态 |
|----|------|------|
| 6.1 | 更新 `config/leaphand/__init__.py`：注册所有 Gym 环境 | ⬜ |
| 6.2 | 验证：`python scripts/list_envs.py` | ⬜ |
| 6.3 | 验证：`python scripts/random_agent.py --env Isaac-AnyMani-LeapHand-Joint-v0` | ⬜ |
| 6.4 | 验证：使用稳定版配置加载老模型进行 play | ⬜ |

### Phase 7: 清理与文档 (Day 4)

| ID | 任务 | 状态 |
|----|------|------|
| 7.1 | 删除旧的 `*_env_cfg.py` 文件 | ⬜ |
| 7.2 | 更新 README.md | ⬜ |
| 7.3 | 创建 `stable_env_cfg.py` 使用指南 | ⬜ |
| 7.4 | 创建重构完成 tag：`git tag v2.0-anymani` | ⬜ |
| 7.5 | 合并 `refactor` 分支到 `main` | ⬜ |

---

## 现有任务迁移映射

| 原文件 | 迁移目标 |
|--------|---------|
| `inhand_base_env_cfg.py` | `inhand_env_cfg.py` + `leaphand_env_cfg.py::LeapHandJointEnvCfg` |
| `inhand_affine_env_cfg.py` | 整合到 `LeapHandAffineEnvCfg` |
| `inhand_rma_env_cfg.py` | 整合到 `LeapHandRmaEnvCfg` |
| `inhand_tactile_env_cfg.py` | 整合到 `LeapHandTactileEnvCfg` |
| `inhand_se3_env_cfg.py` | `inhand_env_cfg.py` + `LeapHandSe3EnvCfg` |
| `inhand_se3_tactile_env_cfg.py` | 整合到 `LeapHandSe3TactileEnvCfg` |
| `inhand_round_base_env_cfg.py` | `leaphand_round/leaphand_round_env_cfg.py` |
| `inhand_float_env_cfg.py` | 暂缓，未来 `inhand_float/` 任务 |

---

## 稳定版本管理流程

```
1. 训练模型 → 使用 leaphand_env_cfg.py 中的开发配置
2. 效果满意 → 手动复制配置到 leaphand_stable_env_cfg.py
   - 添加版本后缀（如 StableV1）
   - 在类 docstring 中记录：
     - 创建日期
     - 对应 git commit
     - 模型保存路径
     - obs_dim / action_dim
3. 打 git tag（可选但推荐）
4. 复现时 → import 稳定版配置
```

---

## 风险与注意事项

1. **import 路径变更**：所有使用 `leaphand` 的脚本需要更新
2. **Gym 环境名变更**：训练脚本、rl_games 配置中的 env 名需要同步更新
3. **现有模型**：确保迁移前记录所有需要保留的模型对应的 env_cfg
4. **渐进式迁移**：可以先保留旧文件，新旧并存一段时间

---

## 工作量估计

| Phase | 估计时间 |
|-------|---------|
| Phase 1-2 | 2-3 小时 |
| Phase 3 | 1-2 小时 |
| Phase 4 | 4-6 小时 |
| Phase 5 | 2-3 小时 |
| Phase 6-7 | 2-3 小时 |
| **总计** | **1.5-2 天** |

---

## 下一步

- [ ] 确认本计划无遗漏
- [ ] 开始 Phase 1 准备工作
