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
| 2026-02-04 | Coding subagent：完善位置约束应用 | ✅ 完成，commit 49fb4d5 |
| 2026-02-04 | Debug subagent：测试 LeapHandJointEnvCfg 训练 | ✅ 完成，commit 2db3908 |
| 2026-02-04 | Coding subagent：添加目标物体可视化 | ❌ 不符合用户期望，commit 22d8d6a |
| 2026-02-04 | Coding subagent：修正目标物体可视化 | 进行中 |

## 当前待解决问题

⚠️ **用户严重不满**：上一次可视化实现完全不符合预期！

**问题**：
1. 颜色不对：不是"幽灵绿色"
2. 位置不对

**正确参考**：
- `LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py`
- 必须仔细研究这个文件的实现方式

**要求**：
1. 仔细研究 LEAP_Hand_Isaac_Lab 的实现
2. 修改为完全一致的效果
3. 测试确认
4. 与用户讨论

落地摘要：
- `RelativeSO3CommandCfg.goal_pose_visualizer_cfg`：对齐 LEAP_Hand_Isaac_Lab，使用 DexCube USD 的**原生外观**（不改色/不半透明），并使用同款 scale=(1.2,1.2,1.2)。
- `RelativeSO3CommandCfg.goal_marker_pos_e`：对齐 LEAP，将 marker 放在每个环境原点附近的固定位置 (-0.2, -0.45, 0.68)（环境系），用于展示目标旋转。
- `*EnvCfg_PLAY`：默认开启 `self.commands.goal_pose.debug_vis = True`，确保在 IsaacSim 视口里能看到目标姿态 marker。

## 当前待解决问题

用户反馈：返回的 3D 位置观察项 (pos_command) 根本没用到其他地方，需要：
1. 参考 IsaacLab 官方如何设计位置约束
2. 将位置观察项应用到奖励/观测配置中
3. 与用户讨论方案

## 落地结果摘要（对齐 IsaacLab 官方思路）

- **观测**：将 `pos_command` 作为 *Critic/特权观测* 写入 `inhand_env_cfg.py` 的观测组配置（policy 侧不暴露位置相关项，保持可部署假设）。
- **奖励**：补齐位置误差奖励函数 `track_pos_l2`（并提供 `goal_position_distance` 别名兼容旧配置），并在 `CommonRewardsCfg` 中加入该项（默认 weight=0.0，对齐官方默认关闭；需要时可改为负权重启用）。
