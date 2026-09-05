r"""PPO估计、分层采样与梯度审计，不拥有MDP或模型参数。

ppo_batch提供完整rollout逐资产归一化与均衡排列，gradient_audit提供完整参数坐标的只读梯度比较。
Morphology-conditioned baseline、CVaR/group-DRO与gradient balancing仍是需要收益证据的候选，不默认启用。
"""

from .gradient_audit import compute_actor_gradient_scope_audit, per_asset_replica_half_gradients

__all__ = ["compute_actor_gradient_scope_audit", "per_asset_replica_half_gradients"]
