r"""RL update、advantage、sampling 与 variant weighting 的算法边界。

未来 heterogeneous PPO fairness 可在有 baseline evidence 后研究 per-variant advantage
centering/scaling、morphology-conditioned baseline、variant-balanced minibatch、CVaR/group-DRO
weighting 与 gradient conflict。它们不是 representation objective，也不属于 task MDP。

当前只保留边界，不声明算法文件、registry id、配置字段或默认开启项。首先必须记录
per-variant return/success、advantage moments、value error、clip fraction、sample count、
gradient norm/cosine 与 ADR level，区分无学习信号、baseline bias、尺度失衡和梯度冲突。
"""
"""PPO estimator与梯度审计的窄算法模块。"""

from .gradient_audit import compute_actor_gradient_scope_audit, per_asset_replica_half_gradients

__all__ = ["compute_actor_gradient_scope_audit", "per_asset_replica_half_gradients"]
