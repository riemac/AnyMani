# 委派上下文：so(3) 相对增量指令重构

## 背景

用户希望重构手内操作任务的命令项，从绝对目标位姿改为 so(3) 相对增量指令。

## 当前任务

**修改需求**：`RelativeSO3Command.command` 返回值从 3D 改为 6D = `(pos_command_e, phi_ref_e)`

### 修改原因

用户反馈：不想让物体到处乱晃，需要位置约束。

### 具体改动

1. **`rotation_command.py` 中 `RelativeSO3Command.command` 属性**：
   - 当前：返回 `self.phi_ref_e` (3D)
   - 改为：返回 `torch.cat((self.pos_command_e, self.phi_ref_e), dim=-1)` (6D)

2. **`observations.py` 中 `so3_command` 观测函数**：
   - 更新 fallback 逻辑，支持 6D command 格式（取后 3 维）
   - 可选：新增 `pos_command` 观测函数，读取目标位置（取前 3 维）

3. **更新 `__str__` 方法**：反映新的命令维度

## 参考资料

- 核心idea：`AnyMani/source/anymani/mytask/GeneralCommond/relative_so3_command/idea.ipynb`
- 方案文档：`./relative_so3_refactor_plan.md`
- 命令项实现：`/home/hac/isaac/AnyMani/source/anymani/anymani/tasks/inhand/mdp/commands/rotation_command.py`
- 观测函数：`/home/hac/isaac/AnyMani/source/anymani/anymani/tasks/inhand/mdp/observations.py`

## 历史记录

| 时间 | 委派 | 结果 |
|------|------|------|
| 2026-02-04 | General subagent 讨论方案 | ✅ 完成 |
| 2026-02-04 | Coding subagent 实施 | ✅ 完成，commit 4e06513 |
| 2026-02-04 | Coding subagent：command 改为 6D | ✅ 完成，commit b2b621b |
| 2026-02-04 | Coding subagent：完善位置观察项应用 | ✅ 完成 |

## 当前待解决问题

用户反馈：返回的 3D 位置观察项 (pos_command) 根本没用到其他地方，需要：
1. 参考 IsaacLab 官方如何设计位置约束
2. 将位置观察项应用到奖励/观测配置中
3. 与用户讨论方案

## 落地结果摘要（对齐 IsaacLab 官方思路）

- **观测**：将 `pos_command` 作为 *Critic/特权观测* 写入 `inhand_env_cfg.py` 的观测组配置（policy 侧不暴露位置相关项，保持可部署假设）。
- **奖励**：补齐位置误差奖励函数 `track_pos_l2`（并提供 `goal_position_distance` 别名兼容旧配置），并在 `CommonRewardsCfg` 中加入该项（默认 weight=0.0，对齐官方默认关闭；需要时可改为负权重启用）。
