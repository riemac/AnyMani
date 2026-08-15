r"""Physical geometry/self-model pretraining objectives。

所有 objective 都消费 candidate-neutral target batch 与 model prediction。field family、query
layout、decoder type、gauge pairing 与 FK baseline 可以独立组合；任何 loss 降低都必须再由
held-out geometry probes 和 downstream heterogeneous PPO 证明价值。
"""
