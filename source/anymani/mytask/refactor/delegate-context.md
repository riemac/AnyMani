# 重构委派上下文

## 背景

用户（科研人员）正在对 AnyRotate 项目进行重构。目前处于 `refactor` 分支。

### 原始问题

1. **模型复现困难**：过去训练好的 `.pth` 模型，因 `inhand_base_env_cfg.py` 的改动无法复现
2. **任务文件繁杂**：基类 `inhand_base_env_cfg.py` 牵一发动全身
3. **命名需更通用**：AnyRotate → AnyMani

### 用户长期目标

RL 策略蒸馏 → 模仿学习训练更通用策略

## 重构进度（2026-02-02 更新）

### ✅ 已完成

- **Phase 1**: 准备工作 - 创建 `pre-refactor-backup` tag
- **Phase 2**: 目录重命名 - AnyRotate → AnyMani，leaphand → anymani
- **Phase 3**: 任务目录重组 - 创建 `tasks/inhand/` 新结构
- **Phase 4**: MDP 组件整理 - 创建 `inhand_env_cfg.py` 组件库
- **Phase 5**: 配置组装和环境注册
- **Phase 6**: Gym 注册与验证 - 11 个环境验证通过
- **Phase 7**: 清理与文档更新

### 当前目录结构

```
AnyMani/source/anymani/anymani/tasks/
├── inhand/
│   ├── __init__.py
│   ├── inhand_env_cfg.py      # ⭐ MDP 组件库（新增）
│   ├── mdp/                    # MDP 函数实现
│   │   ├── actions/
│   │   ├── commands/
│   │   ├── recorders/
│   │   ├── observations.py
│   │   ├── rewards_*.py
│   │   └── ...
│   └── config/
│       ├── leaphand/           # LeapHand 配置
│       │   ├── agents/
│       │   ├── inhand_*_env_cfg.py
│       │   └── leaphand_stable_env_cfg.py
│       └── leaphand_round/     # 半球指尖变体
├── direct/
└── functional/
```

### MDP 组件库结构 (inhand_env_cfg.py)

```python
# 场景
InHandObjectSceneCfg, TactileSceneCfg

# 观测
JointSpaceObsGroupCfg, Se3ObsGroupCfg, TactileObsGroupCfg
ProprioceptionObsGroupCfg, RmaPrivInfoObsGroupCfg

# 动作
JointSpaceActionsCfg, Se3ActionsCfg, AffineActionsCfg

# 奖励
CommonRewardsCfg, Se3RewardsCfg, TactileRewardsCfg

# 事件/终止/命令
CommonEventCfg, CommonTerminationsCfg, ReorientationCommandsCfg
```

### 已验证的 Gym 环境

1. Template-Leaphand-Direct-v0
2. Template-Leaphand-ContinuousRot-Direct-v0
3. AnyMani-LeapHand-Joint-v0
4. AnyMani-LeapHand-SE3-v0
5. AnyMani-LeapHand-Affine-v0
6. AnyMani-LeapHand-Float-v0
7. AnyMani-LeapHand-RMA-v0
8. AnyMani-LeapHand-Tactile-v0
9. AnyMani-LeapHand-SE3-Tactile-v0
10. Template-Leaphand-Rot-Manager-v0 (兼容旧名)
11. AnyMani-LeapHand-RoundTip-v0

## 历史记录

| 时间 | 委派任务 | 结果 |
|------|----------|------|
| 2026-02-02 | Coding subagent 执行重构 Phase 1-7 | ✅ 完成 6 个 commits |
| 2026-02-02 | Coding subagent 执行 Phase 4+6 | ✅ 完成 2 个 commits |
