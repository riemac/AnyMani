r"""结构化Actor优先的FlashSAC网络适配，另保留官方式MLP Actor对照。

Critic采用FlashSAC的归一化ensemble残差骨架与categorical回报分布；输入加入合法历史和
只供Critic读取的状态。两条网络参数完全独立，所有神经动作输入显式清除ghost。
归一化层来自Holiday Robotics的MIT实现，归属见LICENSE.upstream。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from anymani.distill.models.palm_rotation_policy import PalmRotationDirectActor, expanded_policy_log_std

from .config import FlashSACConfig
from .math import masked_tanh_normal
from .normalized import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    UnitLinear,
    UnitRMSNorm,
    project_unit_parameters,
)
from .observations import ACTOR_FLAT_DIM, CRITIC_FLAT_DIM, actor_flat_features, actor_inputs, critic_flat_features


@dataclass
class ActorSample:
    r"""规范化动作与两种概率归约；log_prob是真正的有效关节联合密度。"""

    actions: torch.Tensor  # [B,16]，ghost精确零，送环境后仍由原1/24rad接口执行。
    mean_action: torch.Tensor  # [B,16]，固定评价使用的确定性动作中心。
    log_prob: torch.Tensor  # [B]，仅有效关节求和。
    log_prob_per_active: torch.Tensor  # [B]，只用于熵正则/温度，不能冒充联合密度。
    log_std: torch.Tensor  # [B,16]，变换前潜Gaussian尺度。
    active_count: torch.Tensor  # [B]，有效关节数。


class FlashPalmActor(nn.Module):
    r"""相同合法观测下的两种真实Actor实现；默认保留当前结构化骨架。

    structured复用现有几何/TCN网络，输出有界中心mu，SAC使用atanh(mu)作为潜均值，
    采用标准tanh重参数采样，不使用PPO的mean-preserving location修正。
    flash_mlp采用上游式BN/UnitLinear残差网络。两者的确定性评价都使用有界动作中心。
    """

    def __init__(self, config: FlashSACConfig):
        super().__init__()
        self.variant = config.actor_variant
        if self.variant == "structured":
            self.body = PalmRotationDirectActor(local_skip=False, history_encoder="tcn", sigma_mode="conditional")
        else:
            width = config.actor_mlp_hidden_dim
            self.embedder = FlashSACEmbedder(ACTOR_FLAT_DIM, width)
            self.blocks = nn.ModuleList([FlashSACBlock(width) for _ in range(config.actor_mlp_num_blocks)])
            self.post_norm = UnitRMSNorm(width)
            self.mean_w = UnitLinear(width, 16)
            self.mean_bias = nn.Parameter(torch.zeros(16))
            self.std_w = UnitLinear(width, 16)
            self.std_bias = nn.Parameter(torch.zeros(16))
        self.project_parameters()  # 上游归一化网络在初始化后也执行范数投影。

    def forward(self, observation: Mapping[str, torch.Tensor], *, training: bool = False,
                noise: torch.Tensor | None = None, deterministic: bool = False) -> ActorSample:
        r"""用合法输入生成动作；特权字段与资产索引不会被Actor特征函数读取。"""
        valid = observation["jnt_valid"].bool()
        if self.variant == "structured":
            actor_observation, geometry = actor_inputs(observation)
            self.body.train(training)
            output = self.body(actor_observation, geometry)
            center = output.mean.masked_fill(~valid, 0)
            latent_mean = torch.atanh(center.clamp(-1 + 1e-6, 1 - 1e-6))  # SAC潜均值，保持实际控制单位。
            log_std = expanded_policy_log_std(output.log_std, center, valid)
        else:
            features = actor_flat_features(observation)
            hidden = self.embedder(features, training)
            for block in self.blocks:
                hidden = block(hidden, training)
            hidden = self.post_norm(hidden)
            latent_mean = self.mean_w(hidden) + self.mean_bias
            raw_std = self.std_w(hidden) + self.std_bias
            log_std = -10.0 + 6.0 * (1.0 + torch.tanh(raw_std))  # 上游log-std范围[-10,2]。
            center = torch.tanh(latent_mean).masked_fill(~valid, 0)
        actions, joint_logp, mean_logp = masked_tanh_normal(
            latent_mean, log_std, valid, noise=noise, deterministic=deterministic,
        )
        if deterministic:
            actions = center  # 评价中心不因atanh/clamp的数值保护而改写。
        return ActorSample(actions, center, joint_logp, mean_logp, log_std.masked_fill(~valid, 0), valid.sum(-1))

    @torch.no_grad()
    def project_parameters(self) -> None:
        r"""默认Actor保留现有尺度约束；原式Actor使用FlashSAC范数投影。"""
        if self.variant == "structured":
            self.body.project_exploration_parameters()
        else:
            project_unit_parameters(self)


class FlashPalmDoubleCritic(nn.Module):
    r"""动作条件双Q，返回期望值[2,B]及回报原子log-prob[2,B,K]。

    这里不共享Actor可训练特征，Critic从原始合法历史、冻结几何和特权状态构造自己的输入。
    Cross-batch current/next拼接由learner负责；training控制BN统计，不等同于是否计算梯度。
    """

    def __init__(self, config: FlashSACConfig):
        super().__init__()
        width = config.critic_hidden_dim
        self.embedder = EnsembleFlashSACEmbedder(2, CRITIC_FLAT_DIM + 16, width)
        self.blocks = nn.ModuleList([EnsembleFlashSACBlock(2, width) for _ in range(config.critic_num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(2, width)
        self.predictor = EnsembleCategoricalValue(2, width, config.critic_bins, -config.value_support, config.value_support)
        project_unit_parameters(self)

    @property
    def support(self) -> torch.Tensor:
        r"""统一的回报支撑[K]；所有目标投影与期望Q共用该坐标。"""
        return self.predictor.bin_values.reshape(-1)

    def forward(self, observation: Mapping[str, torch.Tensor], actions: torch.Tensor, *, training: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""ghost动作在进入Q之前清零，防止Actor利用与物理无关的填充维度。"""
        valid = observation["jnt_valid"].bool()
        if actions.shape != valid.shape:
            raise ValueError("critic action shape disagrees with active-joint mask")
        action_input = actions.masked_fill(~valid, 0)
        features = torch.cat((critic_flat_features(observation), action_input), dim=-1)
        hidden = self.embedder(features.unsqueeze(0).expand(2, -1, -1), training)
        for block in self.blocks:
            hidden = block(hidden, training)
        hidden = self.post_norm(hidden)
        values, info = self.predictor(hidden, training)
        return values, info["log_prob"]

    @torch.no_grad()
    def project_parameters(self) -> None:
        r"""每次Critic优化后归一化参数，与上游受约束Q骨架保持一致。"""
        project_unit_parameters(self)
