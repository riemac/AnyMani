# so(3) 相对增量指令重构方案（对齐版）

> 日期：2026-02-04
>
> 目标：将手内操作任务的“绝对目标姿态指令”重构为 **so(3) 相对增量/参考指令**，提升可部署性与泛化能力。

---

## 1. 设计目标（已对齐）

- **policy 观测最终只包含**：
  - 本体感受（如 joint/body_twists/last_action 等）
  - 触觉
  - 3 维 so(3) 指令向量（rotvec）
- 不再依赖物体的绝对位姿/姿态观测（也不要求 policy 读取 object pose）。
- 坐标系：本次先按 **环境坐标系 {e}** 实现（工程上等价于 {w} 的姿态表示，不引入手掌系 {s} 的统一问题）。
- 采样角度范围：声明式配置，默认 $\theta_{max}=\pi/2$。

---

## 2. 核心数学对象

- 当前物体姿态：$R_c$（来自仿真状态）
- 参考指令（rotvec）：$\phi_{ref}\in\mathbb{R}^3$
- 相对旋转：$\Delta R = \exp(\phi_{ref})$

命令项内部维护目标姿态 $R_g$（用于 reward/metrics/可视化复用），但 policy 不直接看 $R_g$。

---

## 3. 命令项两种模式（本次都做）

### Mode A：fixed-goal（训练常用，重定向）

- resample 时冻结目标：
  $$R_g \leftarrow \exp(\phi_{ref})\,R_c(t_0)$$
- 之后 $R_g$ 固定，误差随控制衰减：
  $$\phi_{err}(t) = \log\big(R_g R_c(t)^{-1}\big)$$
- 成功判定：$\|\phi_{err}\| < \varepsilon$ 触发 resample。

### Mode B：rolling-goal（部署/推理用固定指令，持续旋转）

- 每个 timestep 更新目标：
  $$R_g(t) \leftarrow \exp(\phi_{ref})\,R_c(t)$$
- 则误差恒等于参考指令：
  $$\phi_{err}(t)=\log(R_g(t)R_c(t)^{-1})=\phi_{ref}$$
- 这使得部署时给定固定 $\phi_{ref}=\pi/8$（方向任意）在数学上自洽：policy 始终接收固定指令。

> 备注：rolling-goal 下不能使用“误差越小越好”的 tracking 奖励（会变成常数），需要改成“执行旋转”的奖励（见下文）。

---

## 4. 采样策略（通才关键）

- 轴向：球面均匀采样
  - $(x,y,z)\sim \mathcal{N}(0,1)$，$u=(x,y,z)/\|(x,y,z)\|$
- 角度：$\theta\sim U(\theta_{min},\theta_{max})$（默认 $\theta_{max}=\pi/2$）
- 指令：$\phi_{ref}=u\theta$

---

## 5. 观测项重构（policy 侧）

**policy obs 最终建议：**
- 保留：本体感受 + 触觉 + `so3_command`（3 维，等于 $\phi_{ref}$）
- 移除：
  - `goal_pose`（7 维绝对目标 pose）
  - `goal_quat_diff`（4 维）
  - `object_pos/object_quat` 等特权信息

> critic 侧是否保留特权信息可后续再议，本次重构不强制。

---

## 6. 奖励项适配

### fixed-goal
- 可直接复用现有 `track_orientation_inv_l2`（基于 `quat_error_magnitude` 计算角误差）。

### rolling-goal（需新增 reward）
- 建议新增“旋转执行”奖励（最低侵入实现优先）：
  1) **基于角速度**（若 `RigidObject.data.root_ang_vel_w` 可用）：
     - 对齐项：$\text{align}=\langle \hat{u}_{ref}, \widehat{\omega} \rangle$
     - 幅值项：$\|\omega\|$
  2) 或基于每步姿态增量 $\Delta R$ 的 rotvec（需要缓存上一帧姿态）。

---

## 7. 指标（metrics）建议

- 通用：`command_counter`、`consecutive_success`
- fixed-goal：
  - `orientation_error = ||\phi_{err}||`（角度误差）
  - `cumulative_rotation_cmd = Σ||\phi_{ref}||`（每次 resample / success 累加）
- rolling-goal：
  - `cumulative_rotation_actual`（积分或累加每步旋转量）
  - `rotation_alignment`（与期望轴的一致性）

这些指标为未来“蒸馏轨迹过滤”（卡住/跌落/旋转不足）预留接口。

---

## 8. 实施步骤清单（文件级）

1) **新增命令项类**（推荐新类）：
   - 文件：`AnyMani/source/anymani/anymani/tasks/inhand/mdp/commands/rotation_command.py`
   - 新类：`RelativeSO3Command`（或同名）
   - 内部 buffer：`phi_ref_e`、`quat_command_w` 等
   - 支持 mode：`fixed_goal` / `rolling_goal`

2) **新增命令项配置**：
   - 文件：`AnyMani/source/anymani/anymani/tasks/inhand/mdp/commands/commands_cfg.py`
   - 新 config：`RelativeSO3CommandCfg`
   - 字段：`theta_min/theta_max`、`mode`、`orientation_success_threshold`、`make_quat_unique`、`asset_name`、`init_pos_offset` 等

3) **新增 Observation term**：
   - 新函数：`so3_command(env, command_name) -> (N,3)`
   - 从 command term 取出 `phi_ref_e`

4) **修改 inhand 观测配置**（policy 侧）：
   - 文件：`AnyMani/source/anymani/anymani/tasks/inhand/inhand_env_cfg.py`
   - 以 `Se3TactileObservationsCfg.PolicyCfg` 为主要落点：
     - 删除 `goal_pose` / `goal_quat_diff` / `object_pos` / `object_quat`
     - 增加 `so3_command`

5) **rolling-goal 奖励新增**：
   - 文件：`AnyMani/source/anymani/anymani/tasks/inhand/mdp/rewards_task.py`
   - 新 reward：基于角速度对齐与幅值（或基于姿态增量）

6) **指标完善**：
   - 在 command term `_update_metrics` 中补齐需要的统计量

---

## 9. 未决问题（本次不做，但需记录）

1) 手掌坐标系 {s} 工程统一（不同机器人 URDF/USD 不一致）：
   - 未来做法：指定 palm link，从 `q_ws` 做共轭变换把 $\phi$ 表达到 {s}。
2) rolling-goal 的 reward/termination 细节：
   - 更偏“转得快”还是“转得稳”？是否要加防跌落/卡死惩罚？
3) 若未来引入物体状态估计，policy 输入是否扩展（本次刻意不依赖）。
