# GM Teacher 实现前总览

本文记录 `tasks/gm` 与 `distill/models` 在进入 Specialist / Teacher 实现阶段前的总体状态、已对齐语义、仍存在的运行阻塞与建议实现顺序。

## 总体判断

- Specialist / Teacher 阶段的研究语义已经基本闭合，但当前仍是 scaffold / contract 状态，不是 runnable training stack。
- 最近已完成的主线包括：raw rad action/state、grasp cache reset 语义、event MDP 边界、reward / curriculum / termination scaffold、`PALM / JOINT / TIP` token 路线、`hybrid_se3` attention bias 契约。
- 当前进入实现阶段是合理的，但应优先做单资产 vertical slice，而不是直接实现完整跨资产训练系统。

## 当前一致的语义

### Action / State Obs

- 动作空间采用 raw relative joint delta：
  $$
  \Delta_t = a_t^{raw}\,s, \qquad q_{t+1}^{cmd}=\mathrm{clamp}(q_t+\Delta_t,q^{min},q^{max}).
  $$
- state obs 使用 raw rad `q`、raw `dq`、上一帧实际下发的 `processed_actions`。
- `q_min / q_max` 语义归 morphology / geometry，但 Teacher RL 中由 runtime `soft_joint_pos_limits` 直接作为静态 obs term 暴露。

### Reset / Event / Grasp Cache

- 主线 reset 是 cache-driven：从 validated grasp cache 采样 `(q_hand, T^h_o)`。
- 插上 cache reset 后，不再叠加普通 object pose DR；无 cache 消融才使用 random object pose + random joint reset。
- object scale 是 startup / usd-time 离散 bucket，episode reset 只按 bucket 查 cache，不改变 mesh scale。
- hand dynamics DR 与 cache 正交，第一版可 startup 采样；hand joint state 在 cache reset 下由 cache entry 控制。

### Reward / Command / Termination

- command 语义为 `[axis_h, error_so3_h]`，reward / curriculum 读取 command term 内部 buffer，如 `goal_quat_w`、`axis_e`、`goal_success_count`。
- reward 主项是 AnyRotate 风格 orientation-only six-keypoint distance，success 默认仍用 SO(3) geodesic threshold。
- adaptive reward curriculum 用 `goal_success_count` 释放 contact / stable / action regularizers。
- termination 第一版只有 timeout 与 `object_out_of_hand`，后者基于 object root 相对 reset anchor 的 3D L2 距离。

### Token / Model / Bias

- 输入按 `PALM / JOINT / TIP` 分组投影，进入 Encoder-only global self-attention。
- actor 只读取 `JOINT` token；`TIP` 可选 auxiliary head；`PALM` 作为 hand-level context / value pooling 候选。
- edge 默认路线为 all-pairs dynamic SE(3)：
  $$
  E_{ij}^{t}=\log(T_i(q_t)^{-1}T_j(q_t))\in\mathbb{R}^{6}.
  $$
- attention bias 默认契约为 `hybrid_se3`：结构 bias + continuous SE(3) MLP bias；消融矩阵为 `none / structural / se3 / hybrid_se3`。

## 主要运行阻塞

1. `scene.robot` 仍为 `MISSING`，`build_hand_articulation_cfg(...)` 仍未实现；环境无法绑定 generated hand asset。
2. `ReorientCommandCfg.class_type=None`，`ReorientCommand` 是显式 `NotImplementedError` skeleton；command manager 不能运行。
3. `GmEventsCfg.reset_grasp_cache=None`，grasp cache store / sampler 仍是契约；cache-driven reset 未实现。
4. `GmRewardsCfg.track_orientation` 仍指向 zero placeholder；正式训练不能用当前 active reward。
5. `distill/rl` 还没有 rl_games adapter，模型侧 token schema 也没有可执行 tokenizer / backbone / heads。

## 风险与漏洞

- `object_out_of_hand` 依赖 reset anchor；cache reset 实现时必须写入 `_gm_object_reset_anchor_w` 或 `_gm_object_reset_anchor_e`，否则掉落判据会退回 default pose。
- command、reward、curriculum 三者强耦合 `goal_success_count` 与 `goal_quat_w`；实现时不能让 reward 从 obs 反推 command 状态。
- contact obs 契约已确认 IsaacLab `contact_pos_w` 是平均接触位置，不是多接触点最大力选择；若未来要主接触点，需要更底层 PhysX buffer。
- `hybrid_se3` bias 的连续 MLP 必须零初始化或加 near-zero gate，避免 PPO 初期 attention logits 被随机 bias 打爆。
- dynamic SE(3) edge 对 generated asset 友好，但真实 URDF frame 语义对齐仍是 student / sim2sim 风险，不应在 Teacher 第一版强行解决。
- static limits 当前进 policy obs；若未来启用 history length `H>1`，limits 必须拆到 static / geometry group，不能随时间历史重复堆叠。

## 建议实现顺序

### Phase 1: 单资产环境跑通

1. 实现 `build_hand_articulation_cfg(...)` 或先写一个 debug asset binding，让 `scene.robot` 可加载一个 generated hand。
2. 实现 `ReorientCommand` 最小可运行版本：reset、goal sampling、`goal_quat_w`、`axis_h/e`、`error_so3_h/e`、`goal_success_count`。
3. 先实现 minimal reset：若 cache 未就绪，可临时用 no-cache ablation reset，但必须清楚标注不是主线。
4. 替换 zero reward placeholder 为 keypoint reorientation reward + success bonus + action regularizers。
5. 跑 random-agent / zero-agent smoke test，验证 env step、reset、termination、reward tensor shape。

### Phase 2: Grasp Cache 主线接入

1. 实现 grasp cache metadata / tensor loading 与 sampler。
2. 实现 `reset_from_grasp_cache` event，写 hand q、object pose、零速度、action targets、reset anchor。
3. 用 nominal asset + nominal cube scale 生成或手工准备小 cache，先验证 reset 稳定性。
4. 扩展到 per-asset / per-scale bucket cache，并记录 manifest。

### Phase 3: Distill / Model Vertical Slice

1. 实现 token-ready batch 数据结构：`PALM / JOINT / TIP` feature groups + masks + slices。
2. 实现 minimal tokenizer、backbone、heads，不急于上 aux。
3. 先跑 `NoBias` 或 gated `hybrid_se3`，确保 PPO 不被 bias 不稳定性干扰。
4. 接 `distill/rl` 的 rl_games network adapter：负责 flatten/unflatten action 与 obs/token batch。
5. 再加入 relation builder 与 `HybridGraphSE3Bias`，做 `none / structural / se3 / hybrid_se3` 消融。

## 实现阶段原则

- 先保证 single-asset single-object runnable，再扩展到 asset bank。
- 任何会改变物理初始状态分布的实现，都优先写成显式 mode / cfg，而不是隐藏在 reset 函数内部。
- 每个阶段都应有最小 smoke test：能构建 env、能 reset、能 step、reward/done/obs/action shape 合法。
- 研究语义优先于抽象漂亮：若某处需要临时 adapter 支撑 rl_games，应放在 `distill/rl`，不要污染 `distill/models` 的纯模型语义。
