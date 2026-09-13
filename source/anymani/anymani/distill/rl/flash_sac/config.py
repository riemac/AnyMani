r"""FlashSAC异构掌旋的显式实验配置，预算以真实环境转移计数。

首轮以结构化Actor适配版为默认，保留上游分布Q/cross-batch/奖励缩放/温度机制。
本配置不解析或更改正在运行的PPO；完整运行时参数必须写入独立实验记录。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FlashSACConfig:
    r"""同一256资产、同819.2万新交互的首轮FlashSAC筛选配置。

    环境步为一个环境的0.05秒策略步；num_envs次这样的转移组成一次vector step。
    updates_per_transition表示Critic更新次数/新transition，绝不与PPO epoch直接等同。
    """

    seed: int = 42
    actor_variant: str = "structured"  # 首轮保留几何/历史Actor；flash_mlp仅作可选对照。
    asset_count: int = 256
    num_envs: int = 512  # 每资产两副本；实际运行前按资源验证。
    total_transitions: int = 8_192_000  # 含随机/温度预热采样，不排除前期数据。
    learning_starts: int = 100_352  # 约10万，能被512/1024/2048整除。
    replay_capacity: int = 2_097_152  # 紧凑CPU转移，非每条完整History30或N040缓存。
    replay_device: str = "cpu"
    batch_size: int = 2048  # 每资产8条样本，严格等额取样。
    history_steps: int = 30  # 1.5秒、oldest-to-latest，包含当前状态。
    n_step: int = 3  # 沿用上游IsaacLab脚本；遇有限终点提前截断。
    gamma: float = 0.99
    updates_per_transition: float = 2.0 / 1024.0  # N512时每vector step约一次Critic更新。
    actor_update_period: int = 2  # Critic每次更新，Actor/温度隔次更新。
    target_tau: float = 0.01  # 目标Critic参数EMA。
    learning_rate: float = 3.0e-4
    final_learning_rate: float = 1.5e-4
    actor_grad_clip: float = 1.0  # 结构化Actor保留有界梯度更新，作为明确的本地适配。
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    critic_bins: int = 101
    value_support: float = 5.0  # categorical atoms在[-5,5]，配合奖励缩放。
    actor_mlp_hidden_dim: int = 128
    actor_mlp_num_blocks: int = 2
    initial_temperature: float = 0.01
    target_sigma: float = 0.15  # 用Gaussian等效尺度定义目标熵，不是强制动作sigma。
    entropy_reduction: str = "mean_active"  # 概率仍沿有效关节求和，熵控制按有效DoF平均。
    noise_zeta_exponent: float = 2.0
    noise_max_repeat: int = 16  # 行噪声独立，重复间隔采用上游共享vector-step时钟。
    checkpoint_interval: int = 1_048_576  # 以累计新转移标记检查点，独立于优化次数。
    console_interval: int = 16_384  # 与007/008一轮新样本数相同，仅用于显示进度。
    use_amp: bool = False  # CPU验证关闭；GPU运行验证后可显式启用并记录。
    compile_mode: str | None = None  # 不由Python版本静默改变编译方式。
    gpu_headroom_bytes: int = 524_288_000  # 用户允许500MiB余量，真正运行时另外监控峰值。

    def __post_init__(self) -> None:
        r"""在创建模型/环境之前核对科学预算、序列可用性和基本数值范围。"""
        if self.actor_variant not in {"structured", "flash_mlp"}:
            raise ValueError("actor_variant must be structured or flash_mlp")
        if self.entropy_reduction != "mean_active":
            raise ValueError("this adaptation defines entropy regularization per active joint")
        if self.compile_mode is not None:
            raise ValueError("this FlashSAC entry currently supports eager execution only")  # 配置与实际执行一致。
        if self.asset_count < 1 or self.num_envs < self.asset_count or self.num_envs % self.asset_count:
            raise ValueError("num_envs must contain equal replicas of all declared assets")
        if self.batch_size < self.asset_count or self.batch_size % self.asset_count:
            raise ValueError("replay batch must give every asset an equal sample quota")
        if self.total_transitions < 1 or self.total_transitions % self.num_envs:
            raise ValueError("training budget must end on an exact vector-step boundary")
        if not 0 <= self.learning_starts < self.total_transitions:
            raise ValueError("learning_starts must lie inside the total sampling budget")
        if self.history_steps != 30 or self.n_step < 1:
            raise ValueError("the current task uses History30 and positive n-step returns")
        if self.replay_capacity // self.num_envs <= self.history_steps + self.n_step:
            raise ValueError("replay is too short to reconstruct valid history and future targets")
        if not 0 < self.gamma < 1 or not 0 < self.target_tau <= 1:
            raise ValueError("gamma and target_tau are outside their valid ranges")
        positive = (self.updates_per_transition, self.learning_rate, self.final_learning_rate,
                    self.initial_temperature, self.target_sigma, self.value_support, self.noise_zeta_exponent)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("optimizer, temperature, support and sampling values must be finite and positive")
        if min(self.actor_update_period, self.noise_max_repeat, self.console_interval, self.checkpoint_interval) < 1:
            raise ValueError("update/log/checkpoint intervals must be positive")
        if min(self.critic_hidden_dim, self.actor_mlp_hidden_dim) < 1 or self.critic_bins < 2:
            raise ValueError("network dimensions and categorical support are invalid")

    def critic_update_budget(self, collected_transitions: int) -> int:
        r"""返回到当前新交互总数应完成的Critic更新数，预热全部计数但不优化。"""
        available = max(0, min(int(collected_transitions), self.total_transitions) - self.learning_starts)
        return math.floor(available * self.updates_per_transition)

    def to_dict(self) -> dict[str, Any]:
        r"""记录实际解析配置，不将它当作训练成功或完整运行的证明。"""
        return asdict(self)
