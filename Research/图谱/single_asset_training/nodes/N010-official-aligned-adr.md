---
id: N010
graph_id: single_asset_training
title: official-aligned ADR
stage: single_asset_training
status: accepted
parents:
  - N000
tags:
  - baseline
  - official-aligned ADR
  - leap-hand
  - adr
  - official-parity
anchors:
  commits:
    - cb710fe
  log_dirs:
    - logs/rl_games/leap_hand_reorient/official_aligned_leap_adr_seed42_4096
    - logs/rl_games/leap_hand_reorient/official_aligned_leap_adr_seed42_4096_fix_physx_material
---

# N010: official-aligned ADR

## 语义定位

这是从 N000 派生出的 official-aligned LEAP ADR 节点。其目标不是继续扩展 tactile 语义，而是在 AnyMani 中尽量复刻官方 LEAP_Hand_Isaac_Lab 的训练 contract：

- 96D proprio-target history actor observation。
- target-buffer relative action，scale = 1/24。
- 官方式 continuous rotation command。
- 官方式 reward / termination / ADR curriculum。

## 固定语义

- Task: `AnyMani-LeapHand-Tactile-ADR-v0`。
- Actor observation: 16D joint state + 16D target buffer history, 叠 3 帧。
- Action: target-buffer increment, `q_target <- clip(q_target + a/24)`。
- Reward: 官方 LEAP step reward 语义，ManagerBased 下抵消 `dt`。
- Curriculum: reset-hook based ADR scheduler, `z_rotation_steps=16`。
- Material ADR: reset 时仅按范围签名重采样 bucket，避免 PhysX material 泄漏。

## 证据锚点

- Code anchor: `cb710fe feat(inhand): restore tactile a51 baseline`。
- Preferred log dirs:
  - `logs/rl_games/leap_hand_reorient/official_aligned_leap_adr_seed42_4096`
  - `logs/rl_games/leap_hand_reorient/official_aligned_leap_adr_seed42_4096_fix_physx_material`

## 复盘结论

这条路线在前 300 epoch 的收敛速度仍落后官方 LEAP，但已经达到“足够好”的可用水平：能稳定学到连续 hand-in-object rotation，且 ADR 能正常推进。

剩余差距更像是 ManagerBased 复刻路径与官方 DirectRLEnv 之间的实现差异，而不是核心 MDP 语义错误。

## 后继约束

- 继续优化时，应优先保持 observation / action / reward / termination 的单变量不变。
- 若要追官方曲线，优先检查 curriculum 触发时序、reward 逐项一致性、以及 reset 时序对 success 统计的影响。
