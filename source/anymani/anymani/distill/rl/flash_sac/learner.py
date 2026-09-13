r"""FlashSAC核心更新的AnyMani适配：Actor→温度→分布双Q→目标参数EMA。

更新顺序、current/next cross-batch与categorical target参照Holiday Robotics FlashSAC，
MIT归属见LICENSE.upstream。局部适配明确为有效关节熵平均、结构化Actor及有限时域n-step mask。
本模块不创建仿真器，不操作PPO，也不把回放数据量当成已完成的环境交互。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from .config import FlashSACConfig
from .math import DiscountedRewardNormalizer, masked_target_entropy, project_categorical, select_min_q_log_probs
from .networks import ActorSample, FlashPalmActor, FlashPalmDoubleCritic
from .observations import concatenate_observations


class RepeatedGaussianNoise:
    r"""上游式截断Zeta重复：间隔时钟共享，每个环境的高斯向量独立。

    这是行为采样噪声，不进入SAC目标策略的独立重参数采样，也不使用PPO概率比率。
    最长16个20Hz步对应0.8秒；环境重置不会隐式改变全批重复时钟。
    """

    def __init__(self, num_envs: int, *, exponent: float, max_repeat: int, device: torch.device, seed: int):
        self.num_envs, self.exponent, self.max_repeat = num_envs, exponent, max_repeat
        self.device = device
        mass = torch.arange(1, max_repeat + 1, dtype=torch.float64).pow(-exponent)
        self.cdf = (mass / mass.sum()).cumsum(0)  # 仅在CPU采一个共享持续长度。
        self.interval_generator = torch.Generator(device="cpu").manual_seed(seed)
        self.noise_generator = torch.Generator(device=device).manual_seed(seed + 1)
        self.noise = torch.zeros(num_envs, 16, device=device)
        self.remaining = 0

    def sample(self) -> torch.Tensor:
        r"""每个vector step返回本轮行为噪声；返回值在下次区间刷新前保持。"""
        if self.remaining == 0:
            draw = torch.rand((), generator=self.interval_generator, dtype=torch.float64)
            self.remaining = int(torch.searchsorted(self.cdf, draw)) + 1
            self.noise = torch.randn(self.noise.shape, generator=self.noise_generator, device=self.device)
        self.remaining -= 1
        return self.noise

    def state_dict(self) -> dict[str, Any]:
        r"""保存两个随机流、当前噪声与重复时钟，恢复后不额外重抽。"""
        return {"num_envs": self.num_envs, "exponent": self.exponent, "max_repeat": self.max_repeat,
                "remaining": self.remaining, "noise": self.noise.clone(),
                "interval_rng": self.interval_generator.get_state(), "noise_rng": self.noise_generator.get_state()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""恢复必须保持相同环境数和探索分布；索引重排由调用方另行声明。"""
        if (state["num_envs"], state["exponent"], state["max_repeat"]) != (self.num_envs, self.exponent, self.max_repeat):
            raise ValueError("noise repetition configuration mismatch")
        self.remaining = int(state["remaining"])
        self.noise.copy_(state["noise"].to(self.device))
        self.interval_generator.set_state(state["interval_rng"].cpu())
        self.noise_generator.set_state(state["noise_rng"].cpu())


class FlashSACLearner:
    r"""独立SAC学习器，接收已重建几何/历史的样本批次，返回具名训练统计。

    Actor、Q和温度分别优化；目标Q仅做参数EMA，BN running moments由自己的cross-batch前向更新。
    原任务奖励由外部保存，reward normalizer只为Q训练缩放，不改物理评分。
    """

    def __init__(self, config: FlashSACConfig, device: torch.device | str = "cpu"):
        self.config, self.device = config, torch.device(device)
        torch.manual_seed(config.seed)
        self.actor = FlashPalmActor(config).to(self.device)
        self.critic = FlashPalmDoubleCritic(config).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(config.initial_temperature), device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=config.learning_rate)
        self.reward_normalizer = DiscountedRewardNormalizer(config.num_envs, config.gamma, config.value_support, self.device)
        self.exploration = RepeatedGaussianNoise(config.num_envs, exponent=config.noise_zeta_exponent,
                                               max_repeat=config.noise_max_repeat, device=self.device, seed=config.seed + 101)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.critic_updates = self.actor_updates = self.collected_transitions = 0

    @property
    def temperature(self) -> torch.Tensor:
        r"""学习温度alpha，无量纲熵正则的尺度系数，不等同于采样噪声倍率。"""
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, observation: Mapping[str, torch.Tensor], *, explore: bool = True) -> ActorSample:
        r"""环境交互使用重复噪声；固定评价使用确定性中心且不消耗行为随机流。"""
        noise = self.exploration.sample() if explore else None
        return self.actor(observation, training=False, noise=noise, deterministic=not explore)

    def observe_transition(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        r"""每次真实vector step调用一次，预热也计入预算和奖励统计。"""
        self.reward_normalizer.observe(rewards.to(self.device), terminated.to(self.device), truncated.to(self.device))
        self.collected_transitions += int(rewards.numel())

    def _set_learning_rates(self) -> float:
        r"""按实际采样预算作同一余弦下降，Actor延迟更新不改变学习率时钟。"""
        fraction = min(1.0, self.collected_transitions / self.config.total_transitions)
        rate = self.config.final_learning_rate + .5 * (self.config.learning_rate - self.config.final_learning_rate) * (1 + math.cos(math.pi * fraction))
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.temperature_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate
        return rate

    def _backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                       parameters: Any, *, clip: float | None = None) -> None:
        r"""一次完整梯度更新；AMP仅在CUDA显式配置开启，CPU数学路径始终FP32。"""
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("FlashSAC objective is non-finite")
        optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if clip is not None:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            optimizer.step()

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        r"""执行上游同次序更新；batch['discounts']已包含gamma**n及有限终止mask。"""
        observation, next_observation = batch["obs"], batch["next_obs"]
        actions = batch["actions"].to(self.device)
        rewards = self.reward_normalizer.scale(batch["rewards"].to(self.device))
        discounts = batch["discounts"].to(self.device)
        count = actions.shape[0]
        combined = concatenate_observations(observation, next_observation)
        info: dict[str, torch.Tensor | float] = {"learning_rate": self._set_learning_rates()}

        # 上游先Actor和温度，再以更新后的策略构造Critic目标；每隔period次才更新Actor。
        if self.critic_updates % self.config.actor_update_period == 0:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                sample_all = self.actor(combined, training=True)
                current_actions = sample_all.actions[:count]
                current_logp = sample_all.log_prob_per_active[:count]
                self.critic.requires_grad_(False)
                try:
                    q_values, _ = self.critic(observation, current_actions, training=False)
                finally:
                    self.critic.requires_grad_(True)
                minimum_q = q_values.min(dim=0).values
                actor_loss = (self.temperature.detach() * current_logp - minimum_q).mean()
            clip = self.config.actor_grad_clip if self.config.actor_variant == "structured" else None
            self._backward_step(actor_loss, self.actor_optimizer, self.actor.parameters(), clip=clip)
            self.actor.project_parameters()
            entropy = -current_logp.detach()
            target = masked_target_entropy(observation["jnt_valid"], self.config.target_sigma, reduction="mean")
            temperature_loss = self.temperature * (entropy - target).mean()
            self._backward_step(temperature_loss, self.temperature_optimizer, [self.log_temperature])
            self.actor_updates += 1
            info.update(actor_loss=actor_loss.detach(), entropy_per_active=entropy.mean(), temperature_loss=temperature_loss.detach())

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                next_sample = self.actor(next_observation, training=False)
                joined_actions = torch.cat((actions, next_sample.actions), dim=0)
                target_values, target_logp = self.target_critic(combined, joined_actions, training=True)
                selected_logp = select_min_q_log_probs(target_values[:, count:], target_logp[:, count:])
                target_probs = project_categorical(selected_logp, rewards, discounts,
                                                   self.temperature * next_sample.log_prob_per_active, self.critic.support)
            values, logp = self.critic(combined, joined_actions, training=True)
            critic_loss = -(target_probs.unsqueeze(0) * logp[:, :count].float()).sum(-1).mean()
        self._backward_step(critic_loss, self.critic_optimizer, self.critic.parameters())
        self.critic.project_parameters()
        with torch.no_grad():
            # 上游EMA仅更新参数。目标BN统计已由本次combined batch更新，不复制在线running buffers。
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target_parameter.lerp_(parameter, self.config.target_tau)
        self.critic_updates += 1
        info.update(critic_loss=critic_loss.detach(), q_mean=values[:, :count].detach().mean(),
                    target_mean=(target_probs * self.critic.support).sum(-1).mean(),
                    target_edge_mass=target_probs[:, [0, -1]].sum(-1).mean(), temperature=self.temperature.detach(),
                    critic_updates=float(self.critic_updates), actor_updates=float(self.actor_updates))
        return {name: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value) for name, value in info.items()}

    def state_dict(self) -> dict[str, Any]:
        r"""保存学习器全部状态；回放、静态bank及环境重启段由训练入口另行保存。"""
        return {"schema_version": 1, "config": self.config.to_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                "log_temperature": self.log_temperature.detach().clone(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                "temperature_optimizer": self.temperature_optimizer.state_dict(), "reward_normalizer": self.reward_normalizer.state_dict(),
                "exploration": self.exploration.state_dict(), "scaler": self.scaler.state_dict(),
                "critic_updates": self.critic_updates, "actor_updates": self.actor_updates,
                "collected_transitions": self.collected_transitions, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""严格恢复同一方法配置，拒绝把不同网络/预算的状态静默混接。"""
        if state.get("schema_version") != 1 or state.get("config") != self.config.to_dict():
            raise ValueError("FlashSAC checkpoint configuration mismatch")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        for name in ("actor_optimizer", "critic_optimizer", "temperature_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
        self.exploration.load_state_dict(state["exploration"])
        self.scaler.load_state_dict(state["scaler"])
        self.critic_updates, self.actor_updates = int(state["critic_updates"]), int(state["actor_updates"])
        self.collected_transitions = int(state["collected_transitions"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
