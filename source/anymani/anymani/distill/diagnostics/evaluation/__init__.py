r"""固定评估协议、分层 metric、latent probe 与性能协议的所有权边界。

评估层可消费 ``representations`` 的真值合同和 ``models`` 的结构化输出，但不运行 optimizer，
也不决定训练组合。当前几何表示路线要求保留 owner、distance shell、field scale、gauge pair、
hand family 与 held-out morphology 等分层轴，避免只报告一个全局平均误差。

当前提供 geometry SSL 的 query-only、latent-shuffle 与可选 Warp/Kaolin surface reference；通用分层 metric 与跨 run evaluator 仍待实现。
"""
