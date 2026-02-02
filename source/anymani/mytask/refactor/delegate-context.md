# 重构委派上下文

## 背景

用户（科研人员）正在对 AnyRotate 项目进行重构规划。目前处于 `refactor` 分支。

### 当前问题

1. **模型复现困难**：过去训练好的 `.pth` 模型，因 `inhand_base_env_cfg.py` 的改动（特别是观察项变动），无法复现训练结果

2. **任务文件繁杂**：`AnyRotate/source/leaphand/leaphand/tasks/manager_based/leaphand` 下的环境配置文件越来越多，基类 `inhand_base_env_cfg.py` 牵一发动全身

3. **命名需更通用**：AnyRotate → AnyMani，leaphand 命名也需调整，以支持未来更多机械手类型

### 用户长期目标

RL 策略蒸馏 → 模仿学习训练更通用策略。这是用户需要复现过去模型的根本原因。

### 参考架构

IsaacLab 官方的 `inhand` 任务组织结构：
```
manipulation/inhand/
├── __init__.py
├── inhand_env_cfg.py      # 通用基类配置
├── mdp/                   # 模块化 MDP 组件
└── config/
    └── allegro_hand/      # 具体手型的配置
        ├── allegro_env_cfg.py
        └── agents/
```

### 当前项目结构

```
leaphand/tasks/manager_based/leaphand/
├── inhand_base_env_cfg.py    # 基类，被多个任务继承
├── inhand_affine_env_cfg.py
├── inhand_float_env_cfg.py
├── inhand_rma_env_cfg.py
├── inhand_round_base_env_cfg.py
├── inhand_se3_env_cfg.py
├── inhand_se3_tactile_env_cfg.py
├── inhand_tactile_env_cfg.py
├── agents/
└── mdp/
```

## 待澄清问题

1. 模型复现：git 版本管理 vs 代码架构解耦，哪种方案更适合？
2. 测试/CI-CD：是否过度工程化？
3. 重构优先级与范围

## 历史记录

| 时间 | 委派任务 | 结果 |
|------|----------|------|
| 2026-02-02 | 待确认 | - |
