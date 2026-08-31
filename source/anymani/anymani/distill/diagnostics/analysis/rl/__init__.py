r"""只读RL runtime artifact汇总入口；heterogeneous PPO CLI从具体模块显式运行。"""

from .runtime import summarize_rl_runtime_artifacts

__all__ = ["summarize_rl_runtime_artifacts"]
