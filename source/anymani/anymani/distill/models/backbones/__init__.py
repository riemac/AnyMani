r"""SSL、RL 与 IL 共用的 retained backbone 边界。

backbone 消费 ``models.input_adapters`` 输出的 owner-aware entity/joint features、valid mask
与必要结构 metadata，输出供 retained unified $Z$、representation decoder、policy/value head
或 distillation objective 使用的 latent。它不解析资产、不生成 target，也不
把某一种 field decoder 的输出 contract 写进 trunk。

当前 Geometry SSL canonical 使用 graph-biased encoder-only Transformer：注意力在全部有效 entity
之间保持全连接，无向最短路径、parent 与 child distance 只作为每头可学习加性 bias，不是 hard mask。
不同结构可由显式 entity/joint masks padding 到同一次前向；逐结构独立前向仍是输出与梯度 oracle。

PALM/JOINT/TIP entity 与 surface owner 同索引，动作路由由 JOINT 角色决定。input adapter 与 backbone
final-norm unified $Z$ 一起由 SSL checkpoint 迁入 PPO，并默认 full fine-tune。RTX 5070 Ti、$B=4096$、
单结构组的完整 retained geometry encoder，50 次 CUDA Event 计时 p95 不超过 40 ms。该门槛不裁定
PPO 的 temporal fusion、policy head 或 simulator step 设计；非 canonical 候选必须位于显式 candidate module/preset。
"""
