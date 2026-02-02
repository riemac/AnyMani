# 重构委派上下文

## 背景

用户（科研人员）正在对 AnyRotate 项目进行重构。目前处于 `refactor` 分支。

### 原始问题

1. **模型复现困难**：过去训练好的 `.pth` 模型，因 `inhand_base_env_cfg.py` 的改动无法复现
2. **任务文件繁杂**：基类 `inhand_base_env_cfg.py` 牵一发动全身
3. **命名需更通用**：AnyRotate → AnyMani

### 用户长期目标

RL 策略蒸馏 → 模仿学习训练更通用策略

## 重构进度（2026-02-02）

### ✅ 已完成

- **Phase 1**: 准备工作 - 创建 `pre-refactor-backup` tag
- **Phase 2**: 目录重命名 - AnyRotate → AnyMani，leaphand → anymani
- **Phase 3**: 任务目录重组 - 创建 `tasks/inhand/` 新结构
- **Phase 5**: 配置组装和环境注册
- **Phase 7**: 清理与文档更新

### 当前目录结构

```
AnyMani/source/anymani/anymani/tasks/
├── inhand/
│   ├── __init__.py
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
│       │   └── leaphand_stable_env_cfg.py  # ⭐ 稳定版本管理
│       └── leaphand_round/     # 半球指尖变体
├── direct/
└── functional/
```

### 稳定版本管理机制

已创建 `leaphand_stable_env_cfg.py`，用于保存训练好的、已验证的配置快照。

## 历史记录

| 时间 | 委派任务 | 结果 |
|------|----------|------|
| 2026-02-02 | Coding subagent 执行重构 | ✅ 完成 Phase 1-7，共 6 个 commits |
