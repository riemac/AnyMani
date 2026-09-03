r"""MVP80 structured actor/privileged critic的rl_games custom PPO边界。

本模块保持三项训练语义：

1. actor与critic读取同一份rollout-cached FP32 $Z^e$，但参数与optimizers完全分离；
2. canonical ghost joints不进入Normal log-prob、entropy、KL、bounds或动作执行；
3. 每个minibatch对80个assets严格等量，正式$B=76800,M=16$时每个minibatch含每资产60项。

N040不属于本network module，因此5个mini-epochs不会触发encoder forward，也不会把冻结encoder写入
actor/critic optimizer。Actor optimizer内部保留base与global-residual两个参数组，其学习率始终维持
$3\!:\!1$；critic使用独立optimizer和$5\times10^{-4}$初始学习率。
"""

from __future__ import annotations

import math
import os
import random
import resource
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import common_losses
from torch import nn
from torch.nn.utils import clip_grad_norm_

from anymani.distill.diagnostics.recording.rl.palm_rotation import PalmRotationMetricsRecorder
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
)
from anymani.tasks.hetero.mdp.curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
)

from .masked_ppo import (
    AnyManiMaskedContinuousModel,
    AnyManiMaskedPpoAgent,
    AnyManiMaskedPpoPlayer,
    AnyManiMaskedRunner,
    register_anymani_masked_ppo,
)
from .runtime.palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)

PALM_ROTATION_PPO_ALGO = "anymani_palm_rotation_ppo"
PALM_ROTATION_NETWORK = "anymani_palm_rotation"
CRITIC_OPTIMIZER_KEY = "anymani_critic_optimizer"
DIAGNOSTICS_RECORDER_KEY = "anymani_metrics_recorder"
TRAINING_CONTINUATION_KEY = "anymani_training_continuation"


def bounded_adaptive_learning_rate(requested_lr: float, reference_lr: float) -> float:
    r"""把rl_games adaptive scheduler限制为只从方法锚点向下调节。

    rl_games会在KL低于阈值一半时每次乘1.5，且默认上限为$10^{-2}$。MVP每update执行16×5个optimizer
    steps，早期zero-init residual产生很小KL，若无此门会在两个updates内把$3\times10^{-4}$推到$10^{-2}$。
    这里保留高KL时降低LR、低KL时恢复LR的机制，但恢复不能越过预先声明的方法锚点。
    """

    if not (requested_lr > 0.0 and reference_lr > 0.0):
        raise ValueError("adaptive and reference learning rates must be positive")
    return min(float(requested_lr), float(reference_lr))


def _linux_memory_snapshot() -> dict[str, int]:
    r"""读取当前训练进程RSS/swap、峰值RSS和系统可用内存，单位byte。

    Isaac Sim、PhysX、N040与PPO位于同一Python进程，因此``/proc/self``覆盖本run主要host分配。GPU峰值由
    CUDA allocator独立记录；两者不能互相替代。
    """

    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmSwap:")):
            name, value, _unit = line.split()
            status[name.rstrip(":")] = int(value) * 1024  # Linux status以KiB报告
    available = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024  # 系统可回收后available bytes
            break
    return {
        "process_rss_bytes": status.get("VmRSS", 0),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "process_swap_bytes": status.get("VmSwap", 0),
        "system_available_memory_bytes": available,
    }


def stratified_asset_permutation(
    prototype_index: torch.Tensor,
    *,
    asset_count: int,
    minibatch_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""构造每个minibatch逐资产等量的sample permutation。

    对资产$i$的样本集合$\mathcal I_i$独立随机排列并均分成$M$份，再令第$m$个minibatch为：

    $$
    \mathcal M_m=\bigcup_{i=0}^{A-1}\mathcal I_i^{(m)}.
    $$

    Args:
        prototype_index (torch.Tensor): selection-local asset index，形状`[B]`或`[B,1]`。
        asset_count (int): 训练支持域资产数$A$；正式MVP固定80。
        minibatch_count (int): 每update minibatch数$M$；正式MVP固定16。
        generator (torch.Generator | None): 可选确定性随机生成器，仅控制组内顺序。

    Returns:
        torch.Tensor: long `[B]`，连续切成$M$段后每段资产计数完全相同。

    Raises:
        ValueError: asset标签缺失、计数不平衡，或单资产计数不能被$M$整除。
    """

    labels = prototype_index.reshape(-1).long()  # `[B]`，asset row只作sampling certificate
    if asset_count < 1 or minibatch_count < 1:
        raise ValueError("asset_count and minibatch_count must be positive")
    if labels.numel() < asset_count or bool(((labels < 0) | (labels >= asset_count)).any().item()):
        raise ValueError("prototype_index contains an invalid or incomplete asset axis")

    # 每个asset必须具有相同rollout cardinality；否则global advantage仍可算，但不满足matched预算。
    counts = torch.bincount(labels, minlength=asset_count)  # `[A]`
    if bool((counts != counts[0]).any().item()) or int(counts[0].item()) % minibatch_count != 0:
        raise ValueError("stratified PPO requires equal per-asset counts divisible by minibatch_count")
    per_asset_per_minibatch = int(counts[0].item()) // minibatch_count  # 2560-env正式$960/16=60$

    # `parts[m][i]`保存第m个minibatch的第i个asset samples；最后按minibatch主序拼接。
    parts: list[list[torch.Tensor]] = [[] for _ in range(minibatch_count)]
    for asset_index in range(asset_count):
        members = torch.nonzero(labels == asset_index, as_tuple=False).squeeze(-1)  # $\mathcal I_i$
        order = torch.randperm(members.numel(), device=members.device, generator=generator)  # 组内随机
        members = members[order]
        for minibatch_index in range(minibatch_count):
            start = minibatch_index * per_asset_per_minibatch  # 当前asset在第m份的起点
            stop = start + per_asset_per_minibatch
            parts[minibatch_index].append(members[start:stop])

    # 每个minibatch内部再随机化asset拼接顺序，避免网络连续看到同一手型样本。
    minibatches: list[torch.Tensor] = []
    for asset_parts in parts:
        indices = torch.cat(asset_parts, dim=0)  # `[B/M]`，每asset严格相同数量
        random_order = torch.randperm(indices.numel(), device=indices.device, generator=generator)
        minibatches.append(indices[random_order])
    return torch.cat(minibatches, dim=0)  # `[B]`，rl_games contiguous slicing直接消费


class PalmRotationRlGamesBuilder:
    r"""rl_games builder：按Dict observation ABI构造正式actor/critic package。"""

    def __init__(self, **kwargs: Any) -> None:
        r"""初始化空配置；实际参数由``load``在Runner build阶段注入。"""

        _ = kwargs
        self.params: dict[str, Any] = {}  # YAML network mapping

    def load(self, params: dict[str, Any]) -> None:
        r"""保存residual开关与checkpoint identity。"""

        self.params = params

    def build(self, name: str, **kwargs: Any) -> PalmRotationRlGamesNetwork:
        r"""构造满足continuous-logstd contract的structured network。"""

        _ = name
        return PalmRotationRlGamesNetwork(self.params, **kwargs)


class PalmRotationRlGamesNetwork(nn.Module):
    r"""将单份experience Dict严格分流到actor、critic和共享geometry。

    Actor构造函数只接收``actor_*``、masks和geometry；``critic_*``从未出现在其dataclass中。
    Critic读取privileged tensors，但不读取``prototype_index``。该index只由agent的分层sampler使用。
    """

    def __init__(self, params: Mapping[str, Any], **kwargs: Any) -> None:
        r"""验证16-action与完整named-shape ABI后实例化独立actor/critic。"""

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))  # canonical action slots，必须为16
        input_shape = kwargs.pop("input_shape")  # Dict[str, sample shape]
        self.value_size = int(kwargs.pop("value_size", 1))  # hand-level scalar value
        self.num_seqs = int(kwargs.pop("num_seqs", 1))  # 非RNN，仅保留rl_games接口字段
        if actions_num != 16 or self.value_size != 1:
            raise ValueError("palm-rotation PPO requires 16 canonical actions and scalar value")
        if not isinstance(input_shape, Mapping):
            raise TypeError("palm-rotation PPO requires a Dict observation space")
        expected_shapes = {
            **PALM_ROTATION_FLOAT_SHAPES,
            **PALM_ROTATION_BOOL_SHAPES,
            **PALM_ROTATION_INT16_SHAPES,
        }
        normalized_shapes = {key: tuple(int(dim) for dim in shape) for key, shape in input_shape.items()}
        if normalized_shapes != expected_shapes:
            missing = sorted(set(expected_shapes) - set(normalized_shapes))
            extra = sorted(set(normalized_shapes) - set(expected_shapes))
            wrong = sorted(
                key
                for key in set(expected_shapes) & set(normalized_shapes)
                if expected_shapes[key] != normalized_shapes[key]
            )
            raise ValueError(f"palm-rotation observation ABI mismatch: missing={missing}, extra={extra}, wrong={wrong}")

        network_cfg = params.get("palm_rotation", {})  # method-specific YAML block
        residual_enabled = bool(network_cfg.get("residual_enabled", True))  # base-only matched arm关闭该branch
        initial_log_std = float(network_cfg.get("initial_log_std", -0.5))  # shared scalar$\log\sigma$
        max_log_std = float(network_cfg.get("max_log_std", -0.43))  # N000 early-budget exploration ceiling
        base_action_limit = float(network_cfg.get("base_action_limit", 0.8))  # 与0.2 residual构成exact action bound
        self.package = PalmRotationActorCritic(
            residual_enabled=residual_enabled,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=base_action_limit,
        )  # actor/critic完全分参；N040不属于此module
        identity = params.get("anymani_identity")
        if not isinstance(identity, dict):
            raise ValueError("palm-rotation network requires a JSON-safe AnyMani runtime identity")
        self.anymani_identity = identity  # checkpoint pre-load identity gate
        self.last_active_joint_mask: torch.Tensor | None = None  # masked Normal读取的当前batch$[B,16]$
        self.last_residual_mean: torch.Tensor | None = None  # scalar diagnostics的detached action residual
        self.last_film_modulation_rms: torch.Tensor | None = None  # `[B,16]` geometry FiLM贡献

    def is_rnn(self) -> bool:
        r"""History30由environment observation显式交付，模型不是rl_games recurrent network。"""

        return False

    def get_default_rnn_state(self) -> None:
        r"""非RNN模型无隐状态。"""

    def get_aux_loss(self) -> None:
        r"""MVP不添加distillation/auxiliary loss。"""

    def get_value_layer(self) -> nn.Module:
        r"""返回critic scalar head，供rl_games introspection。"""

        return self.package.critic.value_head

    def actor_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        r"""将actor参数拆成base-$3e{-4}$与global-residual-$1e{-4}$两组。

        ``geometry_adapter``及owner/dynamic projections只服务global contextual branch，因此与一层graph
        backbone及residual head共同使用较小LR。TCN、local/finger/hand/base和shared log-std属于base组。
        """

        actor = self.package.actor  # 正式actor module
        residual_modules: tuple[nn.Module, ...] = (
            actor.geometry_adapter,
            actor.owner_contact_projection,
            actor.palm_dynamic_projection,
            actor.joint_dynamic_projection,
            actor.tip_dynamic_projection,
            actor.global_backbone,
            actor.residual_head,
        )
        residual_ids = {id(parameter) for module in residual_modules for parameter in module.parameters()}
        base = [parameter for parameter in actor.parameters() if id(parameter) not in residual_ids]  # 主动作路径
        residual = [parameter for parameter in actor.parameters() if id(parameter) in residual_ids]  # bounded correction
        if {id(parameter) for parameter in base} & {id(parameter) for parameter in residual}:
            raise RuntimeError("actor base/residual optimizer groups overlap")
        if len(base) + len(residual) != len(list(actor.parameters())):
            raise RuntimeError("actor optimizer groups do not cover all parameters")
        return base, residual

    def forward(
        self, input_dict: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        r"""执行actor mean/logstd与privileged scalar value前向。

        Args:
            input_dict (Mapping[str, Any]): rl_games mapping，``obs``为named tensor Dict。

        Returns:
            tuple: ``(mu, logstd, value, None)``，shape为`[B,16]`、`[B,16]`、`[B,1]`。
        """

        observation = input_dict.get("obs")
        if not isinstance(observation, Mapping):
            raise TypeError("palm-rotation network expects a named observation mapping")

        # Shared geometry只由rollout-cached tensors重建；本forward绝不持有或调用N040 encoder。
        geometry = PalmRotationGeometry(
            tokens=observation["geometry_tokens"].float(),  # FP32 `[B,21,128]`
            owner_valid=observation["owner_valid"].bool(),
            shortest_path=observation["shortest_path"].long(),  # int16 storage -> exact embedding indices
            parent_direction=observation["parent_direction"].long(),
            child_direction=observation["child_direction"].long(),
        )
        actor_observation = PalmRotationActorObservation(
            jnt_current=observation["actor_jnt_current"].float(),
            jnt_history=observation["actor_jnt_history"].float(),
            jnt_limits=observation["actor_jnt_limits"].float(),
            owner_contact=observation["actor_owner_contact"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )  # actor无法引用任何`critic_*` key
        critic_observation = PalmRotationCriticObservation(
            jnt_state=observation["critic_jnt_state"].float(),
            owner_contact=observation["critic_owner_contact"].float(),
            obj=observation["critic_obj"].float(),
            task=observation["critic_task"].float(),
            reward_release=observation["critic_reward_release"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )  # privileged critic不读取prototype/cell one-hot

        actor_output = self.package.actor(actor_observation, geometry)  # base + bounded global residual
        value = self.package.critic(critic_observation, geometry).unsqueeze(-1)  # `[B,1]`
        self.last_active_joint_mask = actor_observation.jnt_valid  # probability/entropy/KL ghost mask
        self.last_residual_mean = actor_output.residual_mean.detach()  # diagnostics only，不保留autograd graph
        self.last_film_modulation_rms = actor_output.film_modulation_rms.detach()  # local hidden调制幅度
        logstd = actor_output.log_std.expand_as(actor_output.mean)  # 一个共享scalar$\log\sigma$
        return actor_output.mean, logstd, value, None


class PalmRotationMaskedContinuousModel(AnyManiMaskedContinuousModel):
    r"""动作级mean-preserving tanh-squashed masked Normal与机制side-channels。

    Actor直接输出有界动作均值$\bar a=0.8\tanh b+0.2\tanh r\in[-1,1]$。分布先将其映射为
    latent location$m=\operatorname{atanh}(\bar a)$，再采样$z\sim\mathcal N(m,\sigma^2)$并执行
    $a=\tanh z$。因此deterministic action仍严格等于base+residual分解，而随机动作、likelihood与物理
    action space使用同一个变量。
    """

    class Network(AnyManiMaskedContinuousModel.Network):
        r"""计算squashed likelihood/Jacobian并交付residual/FiLM side-channels。"""

        _ACTION_EPS = 1.0e-6  # float32 atanh/log-Jacobian边界，不改变常规open-interval samples

        @classmethod
        def _action_to_latent(cls, action: torch.Tensor) -> torch.Tensor:
            r"""把物理动作$a\in[-1,1]$稳定映射为$z=\operatorname{atanh}(a)$。"""

            bounded = action.clamp(min=-1.0 + cls._ACTION_EPS, max=1.0 - cls._ACTION_EPS)
            return torch.atanh(bounded)

        @classmethod
        def _squashed_per_joint_neglogp(
            cls,
            actions: torch.Tensor,
            action_mean: torch.Tensor,
            sigma: torch.Tensor,
            logstd: torch.Tensor,
        ) -> torch.Tensor:
            r"""返回tanh push-forward在每个joint上的exact negative log-density。

            $$
            -\log\pi_A(a)
            =-\log\mathcal N\!\left(\operatorname{atanh}a;\operatorname{atanh}\bar a,\sigma^2\right)
             +\log(1-a^2).
            $$
            """

            latent_action = cls._action_to_latent(actions)  # $z=\operatorname{atanh}a$
            latent_mean = cls._action_to_latent(action_mean)  # $m=\operatorname{atanh}\bar a$
            normal_neglogp = (
                0.5 * ((latent_action - latent_mean) / sigma).square()
                + logstd
                + 0.5 * math.log(2.0 * math.pi)
            )
            log_jacobian = torch.log((1.0 - actions.square()).clamp_min(cls._ACTION_EPS))
            return normal_neglogp + log_jacobian

        def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor | None]:
            r"""返回rl_games兼容的bounded action、likelihood、KL parameters与机制诊断。"""

            is_train = bool(input_dict.get("is_train", True))
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            action_mean, logstd, value, states = self.a2c_network(input_dict)
            active_mask = self.a2c_network.last_active_joint_mask
            if not isinstance(active_mask, torch.Tensor):
                raise RuntimeError("palm-rotation squashed policy did not expose active-joint mask")
            if bool((action_mean.abs() > 1.0 + 1.0e-6).any().item()):
                raise RuntimeError("palm-rotation deterministic action mean escaped [-1,1]")
            active_float = active_mask.to(dtype=action_mean.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            sigma = torch.exp(logstd)  # latent Normal standard deviation，ghost随后由mask排除
            latent_mean = self._action_to_latent(action_mean)
            distribution = torch.distributions.Normal(latent_mean, sigma, validate_args=False)

            if is_train:
                previous_actions = input_dict.get("prev_actions")
                if not isinstance(previous_actions, torch.Tensor):
                    raise RuntimeError("squashed PPO update requires bounded previous actions")
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    previous_actions,
                    action_mean,
                    sigma,
                    logstd,
                )
                prev_neglogp = (per_joint_neglogp * active_float).sum(dim=-1)
                entropy_latent = distribution.rsample()  # current-policy Monte Carlo differential entropy sample
                entropy_action = torch.tanh(entropy_latent) * active_float
                entropy_per_joint = self._squashed_per_joint_neglogp(
                    entropy_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                entropy = (entropy_per_joint * active_float).sum(dim=-1) / active_count
                result: dict[str, torch.Tensor | None] = {
                    "prev_neglogp": prev_neglogp,
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": action_mean,  # PPO buffer/player保存deterministic物理动作均值
                    "sigmas": sigma,  # KL所需latent标准差
                }
            else:
                latent_action = distribution.sample()
                selected_action = torch.tanh(latent_action) * active_float  # 物理动作严格位于open interval
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    selected_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                result = {
                    "neglogpacs": (per_joint_neglogp * active_float).sum(dim=-1),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": action_mean,
                    "sigmas": sigma,
                }
            residual = getattr(self.a2c_network, "last_residual_mean", None)
            if not isinstance(residual, torch.Tensor):
                raise RuntimeError("palm-rotation actor did not expose bounded action residual")
            result["residuals"] = residual  # 已在network中detach，不延长rollout autograd graph
            film = getattr(self.a2c_network, "last_film_modulation_rms", None)
            if not isinstance(film, torch.Tensor):
                raise RuntimeError("palm-rotation actor did not expose geometry FiLM diagnostics")
            result["film_modulations"] = film  # `[B,16]`，与joint active mask同轴
            return result


class PalmRotationPpoAgent(AnyManiMaskedPpoAgent):
    r"""双optimizer、严格分层minibatch与cached-N040 identity的PPO agent。"""

    def __init__(self, base_name: str, params: dict[str, Any]) -> None:
        r"""让upstream完成Runner状态构造，再替换为actor/critic独立Adam optimizers。"""

        super().__init__(base_name, params)
        if self.has_central_value:
            raise ValueError("palm-rotation custom package already owns the privileged critic; duplicate CV is forbidden")
        if self.mixed_precision:
            raise ValueError("actor, critic, PPO losses and optimizers must remain FP32")
        if self.multi_gpu:
            raise ValueError("MVP80 dual-optimizer agent currently supports one GPU only")
        network = self.model.a2c_network  # AnyManiMaskedContinuousModel facade下的正式network
        if not isinstance(network, PalmRotationRlGamesNetwork):
            raise TypeError("palm-rotation PPO agent received an incompatible network")

        base_parameters, residual_parameters = network.actor_parameter_groups()  # disjoint actor groups
        critic_parameters = list(network.package.critic.parameters())  # completely separate$\theta^c$
        actor_ids = {id(parameter) for parameter in (*base_parameters, *residual_parameters)}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids:
            raise RuntimeError("actor and critic optimizer parameters overlap")

        # 三个LR锚点来自MVP计划；adaptive scheduler只更新base LR，update_lr保持固定比例。
        self._base_lr_reference = float(self.config["learning_rate"])  # 默认$3e-4$
        self._base_lr_ceiling = float(self.config.get("adaptive_lr_max", self._base_lr_reference))
        if self._base_lr_ceiling != self._base_lr_reference:
            raise ValueError("MVP adaptive_lr_max must equal the declared actor base learning-rate anchor")
        self._residual_lr_ratio = float(self.config.get("residual_learning_rate", 1.0e-4)) / self._base_lr_reference
        self._critic_lr_ratio = float(self.config.get("critic_learning_rate", 5.0e-4)) / self._base_lr_reference
        self._gradient_accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1))
        if self._gradient_accumulation_steps < 1 or self.num_minibatches % self._gradient_accumulation_steps != 0:
            raise ValueError("gradient accumulation must divide the number of stratified activation minibatches")
        fused = torch.device(self.ppo_device).type == "cuda"  # CUDA正式训练使用fused Adam
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "lr": self.last_lr, "name": "actor_base"},
                {
                    "params": residual_parameters,
                    "lr": self.last_lr * self._residual_lr_ratio,
                    "name": "actor_global_residual",
                },
            ],
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )  # actor checkpoint optimizer；覆盖upstream临时single optimizer
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=self.last_lr * self._critic_lr_ratio,
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )  # independent critic optimizer
        self.asset_count = int(self.config.get("asset_count", 80))  # 正式支持域$A=80$
        self.last_stratified_permutation: torch.Tensor | None = None  # diagnostics/test evidence
        identity = self._runtime_identity()
        if not isinstance(identity, dict) or not isinstance(identity.get("identity_digest"), str):
            raise RuntimeError("palm-rotation diagnostics require the exact runtime identity")
        self.metrics_recorder = PalmRotationMetricsRecorder(
            self.experiment_dir,
            identity_digest=identity["identity_digest"],
            flush_every_updates=int(self.config.get("diagnostics_flush_updates", 50)),
        )  # run-owned Parquet shard lifecycle
        self._optimization_count = torch.zeros(self.asset_count, device=self.ppo_device)  # mini-epoch samples$[A]$
        self._optimization_sums = {
            name: torch.zeros(self.asset_count, device=self.ppo_device)
            for name in (
                "advantage",
                "advantage_square",
                "value_error",
                "kl",
                "clip_fraction",
                "action_rms",
                "policy_mean_rms",
                "policy_mean_near_bound_fraction",
                "base_mean_rms",
                "residual_rms",
                "residual_fraction",
                "film_modulation_rms",
            )
        }  # 当前update跨全部minibatches×mini-epochs之和
        self._optimizer_step_count = 0  # 当前update真实optimizer step次数
        self._optimizer_microbatch_count = 0  # 当前update真实forward/backward microbatch次数
        self._gradient_microbatch_index = 0  # 必须在每个update边界回到0
        self._optimizer_scalar_sums = {
            name: 0.0
            for name in (
                "actor_loss",
                "critic_loss",
                "entropy",
                "policy_sigma",
                "actor_grad_norm",
                "critic_grad_norm",
            )
        }  # 无per-asset归属的optimizer量只进入global row

    def init_tensors(self) -> None:
        r"""在upstream experience buffer增加detached action-residual side-channel。"""

        super().init_tensors()
        batch = self.num_agents * self.num_actors  # rollout并行样本数$N$
        self.experience_buffer.tensor_dict["residuals"] = torch.zeros(
            self.horizon_length,
            batch,
            16,
            dtype=torch.float32,
            device=self.ppo_device,
        )  # `[H,N,16]`，与actions/mus同axis
        self.update_list.append("residuals")  # play_steps从custom model输出写入buffer
        self.tensor_list.append("residuals")  # rollout结束后swap env/time并flatten
        self.experience_buffer.tensor_dict["film_modulations"] = torch.zeros(
            self.horizon_length,
            batch,
            16,
            dtype=torch.float32,
            device=self.ppo_device,
        )  # `[H,N,16]`，逐joint local-hidden FiLM RMS
        self.update_list.append("film_modulations")
        self.tensor_list.append("film_modulations")

    def update_lr(self, lr: float) -> None:
        r"""保持base/residual/critic学习率比例随adaptive schedule同步缩放。"""

        current = bounded_adaptive_learning_rate(float(lr), self._base_lr_ceiling)  # 禁止低KL指数越过锚点
        self.last_lr = current  # upstream先写入未限幅值，必须同步恢复scheduler state
        for group in self.optimizer.param_groups:
            group["lr"] = current if group.get("name") == "actor_base" else current * self._residual_lr_ratio
        for group in self.critic_optimizer.param_groups:
            group["lr"] = current * self._critic_lr_ratio

    def _assert_actor_learning_rate_ratio(self) -> None:
        r"""在真实step前验证base/residual LR仍保持声明的$3:1$比例。

        rl_games ``A2CAgent.train_actor_critic``会在每个microbatch后把所有actor groups无条件写成
        ``last_lr``。梯度累积使该副作用恰好发生在下一次逻辑step之前，导致residual实际使用base LR。
        本agent覆盖该wrapper，并在此以step-time值fail closed，update尾日志不再替代真实执行证据。
        """

        groups = {str(group.get("name")): float(group["lr"]) for group in self.optimizer.param_groups}
        expected_base = float(self.last_lr)
        expected_residual = expected_base * self._residual_lr_ratio
        if abs(groups.get("actor_base", -1.0) - expected_base) > 1.0e-12:
            raise RuntimeError(f"actor base LR drifted before optimizer step: {groups}")
        if abs(groups.get("actor_global_residual", -1.0) - expected_residual) > 1.0e-12:
            raise RuntimeError(f"actor residual LR ratio drifted before optimizer step: {groups}")

    def train_actor_critic(self, input_dict: dict[str, Any]):
        r"""执行custom gradient step且禁止upstream把两个actor LR groups合并。

        Returns:
            tuple[Any, ...]: rl_games训练循环消费的标准loss/KL/LR/mu/sigma结果。
        """

        self.set_train()
        self.calc_gradients(input_dict)
        return self.train_result

    @staticmethod
    def masked_policy_kl(
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""利用tanh双射，在latent Normal中计算物理squashed policy的精确KL。"""

        current_latent = PalmRotationMaskedContinuousModel.Network._action_to_latent(current_mu)
        old_latent = PalmRotationMaskedContinuousModel.Network._action_to_latent(old_mu)
        c1 = torch.log(old_sigma / current_sigma + 1.0e-5)
        c2 = (current_sigma.square() + (old_latent - current_latent).square()) / (
            2.0 * (old_sigma.square() + 1.0e-5)
        )
        weights = active_mask.to(dtype=current_mu.dtype)
        return ((c1 + c2 - 0.5) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    @staticmethod
    def _index_dataset_value(value: Any, indices: torch.Tensor) -> Any:
        r"""对dataset tensor或一层Dict统一应用batch permutation。"""

        if isinstance(value, dict):
            return {key: tensor[indices] for key, tensor in value.items()}  # named experience tensors
        return value[indices] if isinstance(value, torch.Tensor) else value

    def prepare_dataset(self, batch_dict: dict[str, Any]) -> None:
        r"""先做global advantage/value normalization，再排成严格等量asset minibatches。"""

        residuals = batch_dict.get("residuals")  # detached rollout residual`[B,16]`
        if not isinstance(residuals, torch.Tensor):
            raise RuntimeError("palm-rotation rollout is missing residual side-channel")
        super().prepare_dataset(batch_dict)  # 保持upstream GAE/value normalization与PPO fields
        self.dataset.values_dict["residuals"] = residuals  # optimizer diagnostics，不进入loss
        film_modulations = batch_dict.get("film_modulations")
        if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != residuals.shape:
            raise RuntimeError("palm-rotation rollout is missing geometry FiLM side-channel")
        self.dataset.values_dict["film_modulations"] = film_modulations  # detached mechanism diagnostic
        observation = self.dataset.values_dict.get("obs")
        if not isinstance(observation, dict) or "prototype_index" not in observation:
            raise RuntimeError("stratified PPO requires prototype_index in cached observations")
        permutation = stratified_asset_permutation(
            observation["prototype_index"],
            asset_count=self.asset_count,
            minibatch_count=self.num_minibatches,
        )  # `[B]`，每连续`minibatch_size`严格平衡
        if permutation.numel() != self.batch_size or self.minibatch_size * self.num_minibatches != self.batch_size:
            raise RuntimeError("stratified permutation disagrees with rl_games batch geometry")
        self.dataset.values_dict = {
            key: self._index_dataset_value(value, permutation)
            for key, value in self.dataset.values_dict.items()
        }  # 所有old policy/value/action/obs字段保持同一sample correspondence
        self.last_stratified_permutation = permutation.detach()  # scalar/table diagnostics可审计本update顺序

    def calc_gradients(self, input_dict: dict[str, Any]) -> None:
        r"""同一前向图分别对$\theta^a$与$\theta^c$执行FP32 PPO/value更新。"""

        value_predictions = input_dict["old_values"]  # rollout normalized values`[M,1]`
        old_neglogp = input_dict["old_logp_actions"]  # masked action negative log probability`[M]`
        advantage = input_dict["advantages"]  # globally normalized GAE`[M]`
        old_mu = input_dict["mu"]  # rollout actor means`[M,16]`
        old_sigma = input_dict["sigma"]  # rollout actor stds`[M,16]`
        returns = input_dict["returns"]  # normalized return targets`[M,1]`
        actions = input_dict["actions"]  # sampled canonical actions`[M,16]`
        observation = self._preproc_obs(input_dict["obs"])  # named Dict；normalize_input=False
        result = self.model({"is_train": True, "prev_actions": actions, "obs": observation})
        new_neglogp = result["prev_neglogp"]  # masked active-joint likelihood
        values = result["values"]  # privileged critic prediction`[M,1]`
        entropy = result["entropy"]  # mean entropy per active DoF`[M]`
        mu = result["mus"]  # current actor means`[M,16]`
        sigma = result["sigmas"]  # shared scalar expanded to`[M,16]`

        # Actor objective只含clipped surrogate、active-DoF entropy与masked bounds；无critic gradient path。
        actor_loss_vector = self.actor_loss_func(old_neglogp, new_neglogp, advantage, self.ppo, self.e_clip)
        bounds_loss_vector = self.bound_loss(mu)  # active-DoF mean bounds penalty
        actor_terms, _ = torch_ext.apply_masks(
            [actor_loss_vector.unsqueeze(1), entropy.unsqueeze(1), bounds_loss_vector.unsqueeze(1)],
            None,
        )
        actor_loss, entropy_loss, bounds_loss = actor_terms
        actor_objective = actor_loss - entropy_loss * self.entropy_coef + bounds_loss * self.bounds_loss_coef

        # Critic objective使用独立structured critic和optimizer；0.5保持upstream PPO value-loss convention。
        critic_vector = common_losses.critic_loss(
            self.model,
            value_predictions,
            values,
            self.e_clip,
            returns,
            self.clip_value,
        )
        critic_terms, _ = torch_ext.apply_masks([critic_vector], None)
        critic_loss = critic_terms[0]
        critic_objective = 0.5 * self.critic_coef * critic_loss

        # 非有限objective会污染Adam moments与后续checkpoint，任何backward前立即fail closed。
        finite_forward = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "bounds_loss": bounds_loss,
            "actor_objective": actor_objective,
            "critic_objective": critic_objective,
            "mu": mu,
            "sigma": sigma,
            "value": values,
        }
        for name, value in finite_forward.items():
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"palm-rotation PPO produced non-finite {name}")

        # 四个activation microbatches组成一个原76,800/4逻辑minibatch；参数只在组末更新一次。
        accumulation_offset = self._gradient_microbatch_index % self._gradient_accumulation_steps
        if accumulation_offset == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_objective / self._gradient_accumulation_steps).backward()
        (critic_objective / self._gradient_accumulation_steps).backward()
        network = self.model.a2c_network  # validated PalmRotationRlGamesNetwork
        if not self.truncate_grads:
            raise RuntimeError("palm-rotation PPO requires independent actor/critic gradient clipping")
        accumulation_boundary = accumulation_offset + 1 == self._gradient_accumulation_steps
        if accumulation_boundary:
            self._assert_actor_learning_rate_ratio()  # 必须读取step-time optimizer groups
            actor_grad_norm = clip_grad_norm_(network.package.actor.parameters(), self.grad_norm)  # 逻辑batch$\|g_a\|_2$
            critic_grad_norm = clip_grad_norm_(network.package.critic.parameters(), self.grad_norm)  # 逻辑batch$\|g_c\|_2$
            if not bool(torch.isfinite(actor_grad_norm).item()) or not bool(torch.isfinite(critic_grad_norm).item()):
                raise FloatingPointError(
                    f"palm-rotation PPO produced non-finite gradients: actor={actor_grad_norm}, critic={critic_grad_norm}"
                )
            self.optimizer.step()
            self.critic_optimizer.step()
            network.package.actor.project_exploration_parameters()  # optimizer后立即恢复$\log\sigma\le-0.43$
            self._optimizer_step_count += 1
            self._optimizer_scalar_sums["actor_grad_norm"] += float(actor_grad_norm.item())
            self._optimizer_scalar_sums["critic_grad_norm"] += float(critic_grad_norm.item())
        self._gradient_microbatch_index += 1

        # Loss/entropy/sigma按每个等大microbatch累计；梯度范数只在逻辑optimizer boundary累计。
        self._optimizer_microbatch_count += 1
        microbatch_scalars = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy_loss.detach(),
            "policy_sigma": sigma.detach().mean(),
        }
        for name, value in microbatch_scalars.items():
            self._optimizer_scalar_sums[name] += float(value.item())

        # Adaptive scheduler消费active-DoF-normalized KL，ghost sigma/mean不影响统计。
        active_mask = network.last_active_joint_mask
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("palm-rotation network did not expose active-joint mask")
        with torch.no_grad():
            kl_per_sample = self.masked_policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma, active_mask)
            kl = kl_per_sample.mean()
            ratio = torch.exp(old_neglogp - new_neglogp.detach())  # PPO importance ratio`[M]`
            clip_fraction = (torch.abs(ratio - 1.0) > self.e_clip).float()  # clipped sample indicator
            active_float = active_mask.float()
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            action_rms = torch.sqrt((actions.square() * active_float).sum(dim=-1) / active_count)
            residuals = input_dict.get("residuals")
            if not isinstance(residuals, torch.Tensor) or residuals.shape != actions.shape:
                raise RuntimeError("PPO minibatch residual diagnostics disagree with action shape")
            residual_rms = torch.sqrt((residuals.square() * active_float).sum(dim=-1) / active_count)
            policy_mean_rms = torch.sqrt((old_mu.square() * active_float).sum(dim=-1) / active_count)
            policy_mean_near_bound_fraction = ((old_mu.abs() >= 0.95).float() * active_float).sum(dim=-1) / active_count
            base_mean = old_mu - residuals  # rollout identity$\mu^{base}=\mu-\Delta\mu^{global}$
            base_mean_rms = torch.sqrt((base_mean.square() * active_float).sum(dim=-1) / active_count)
            residual_fraction = residual_rms / (base_mean_rms + residual_rms).clamp_min(1.0e-8)
            film_modulations = input_dict.get("film_modulations")
            if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != actions.shape:
                raise RuntimeError("PPO minibatch geometry FiLM diagnostics disagree with action shape")
            film_modulation_rms = (film_modulations * active_float).sum(dim=-1) / active_count
            labels = observation["prototype_index"].reshape(-1).long()  # sampler certificate，不进模型
            optimization_values = {
                "advantage": advantage.detach(),
                "advantage_square": advantage.detach().square(),
                "value_error": torch.abs(values.detach().squeeze(-1) - returns.squeeze(-1)),
                "kl": kl_per_sample,
                "clip_fraction": clip_fraction,
                "action_rms": action_rms,
                "policy_mean_rms": policy_mean_rms,
                "policy_mean_near_bound_fraction": policy_mean_near_bound_fraction,
                "base_mean_rms": base_mean_rms,
                "residual_rms": residual_rms,
                "residual_fraction": residual_fraction,
                "film_modulation_rms": film_modulation_rms,
            }
            self._optimization_count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            for name, per_sample in optimization_values.items():
                self._optimization_sums[name].scatter_add_(0, labels, per_sample.float())
        self.diagnostics.mini_batch(
            self,
            {
                "values": value_predictions,
                "returns": returns,
                "new_neglogp": new_neglogp,
                "old_neglogp": old_neglogp,
                "masks": None,
            },
            self.e_clip,
            0,
        )
        self.train_result = (
            actor_loss.detach(),
            critic_loss.detach(),
            entropy_loss.detach(),
            kl.detach(),
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            bounds_loss.detach(),
        )  # 与rl_games ContinuousA2CBase.train_epoch tuple contract一致

    def _reset_optimization_metrics(self) -> None:
        r"""在每次rollout/update前清零mini-epoch optimization统计。"""

        self._optimization_count.zero_()
        for total in self._optimization_sums.values():
            total.zero_()
        self._optimizer_step_count = 0
        self._optimizer_microbatch_count = 0
        self._gradient_microbatch_index = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        for name in self._optimizer_scalar_sums:
            self._optimizer_scalar_sums[name] = 0.0

    def _drain_optimizer_scalars(self) -> dict[str, float]:
        r"""返回当前update跨全部minibatches/mini-epochs的global优化统计。"""

        if self._optimizer_step_count < 1 or self._optimizer_microbatch_count < 1:
            raise RuntimeError("optimizer scalar diagnostics observed no update steps")
        if self._gradient_microbatch_index % self._gradient_accumulation_steps != 0:
            raise RuntimeError("PPO update ended inside a logical accumulated minibatch")
        microbatch_denominator = float(self._optimizer_microbatch_count)  # 正式$16\times5=80$
        step_denominator = float(self._optimizer_step_count)  # 正式$(16/4)\times5=20$
        return {
            **{
                name: self._optimizer_scalar_sums[name] / microbatch_denominator
                for name in ("actor_loss", "critic_loss", "entropy", "policy_sigma")
            },
            **{
                name: self._optimizer_scalar_sums[name] / step_denominator
                for name in ("actor_grad_norm", "critic_grad_norm")
            },
            "optimizer_microbatches": microbatch_denominator,
            "optimizer_steps": step_denominator,
        }

    def _drain_optimization_metrics(self) -> dict[str, torch.Tensor]:
        r"""返回跨全部mini-epochs的per-asset优化均值与advantage标准差。"""

        if bool((self._optimization_count <= 0).any().item()):
            raise RuntimeError("optimization diagnostics did not cover every asset")
        count = self._optimization_count
        advantage_mean = self._optimization_sums["advantage"] / count
        advantage_variance = self._optimization_sums["advantage_square"] / count - advantage_mean.square()
        return {
            "advantage_mean": advantage_mean.detach().cpu(),
            "advantage_std": advantage_variance.clamp_min(0.0).sqrt().detach().cpu(),
            "value_error_mean": (self._optimization_sums["value_error"] / count).detach().cpu(),
            "kl_per_active_dof": (self._optimization_sums["kl"] / count).detach().cpu(),
            "clip_fraction": (self._optimization_sums["clip_fraction"] / count).detach().cpu(),
            "action_rms": (self._optimization_sums["action_rms"] / count).detach().cpu(),
            "policy_mean_rms": (self._optimization_sums["policy_mean_rms"] / count).detach().cpu(),
            "policy_mean_near_bound_fraction": (
                self._optimization_sums["policy_mean_near_bound_fraction"] / count
            ).detach().cpu(),
            "base_mean_rms": (self._optimization_sums["base_mean_rms"] / count).detach().cpu(),
            "residual_rms": (self._optimization_sums["residual_rms"] / count).detach().cpu(),
            "residual_fraction": (self._optimization_sums["residual_fraction"] / count).detach().cpu(),
            "film_modulation_rms": (self._optimization_sums["film_modulation_rms"] / count).detach().cpu(),
        }

    @staticmethod
    def _mean_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
        r"""对同一cell或global的per-asset标量作资产等权均值。"""

        return {
            field: float(sum(float(row[field]) for row in rows) / len(rows))
            for field in fields
        }

    def _record_update_metrics(self, epoch_result: tuple[Any, ...]) -> None:
        r"""合并task/optimization/curriculum/resource事实并写89行宽表。"""

        vec_env: Any = self.vec_env  # Runner创建的PalmRotationRlGamesGpuEnv
        if vec_env is None or not hasattr(vec_env, "drain_rollout_metrics"):
            raise RuntimeError("palm-rotation vec env does not expose rollout diagnostics")
        rollout = vec_env.drain_rollout_metrics()  # per-asset post-physics facts`[A]`
        optimization = self._drain_optimization_metrics()  # per-asset optimizer facts`[A]`
        optimizer_scalars = self._drain_optimizer_scalars()  # global loss/sigma/gradient facts
        wrapper = getattr(vec_env, "env", None)  # PalmRotationRlGamesGpuEnv -> structured wrapper
        runtime = getattr(wrapper, "unwrapped", None)
        curriculum = getattr(runtime, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if not isinstance(curriculum, HeterogeneousRewardReleaseState):
            raise RuntimeError("reward-release curriculum state is unavailable for diagnostics")
        fields = tuple(rollout) + tuple(optimization)
        fields = tuple(field for field in fields if field != "sample_count")  # schema只记录均值，不复制count
        transitions = int(self.frame + self.curr_frames)  # 当前update完成后的nominal transition坐标

        # Asset rows保留formal dataset row与8-cell label；counterfactual ADR在ADR-0阶段恒为0。
        asset_rows: list[dict[str, Any]] = []
        cell_ids = curriculum.cell_ids_by_asset.detach().cpu()
        for asset_index in range(self.asset_count):
            row = {
                "update": int(self.epoch_num),
                "transitions": transitions,
                "scope": "asset",
                "scope_index": asset_index,
                "dataset_row": int(curriculum.dataset_rows_by_asset[asset_index]),
                "cell_id": int(cell_ids[asset_index].item()),
                "candidate_lambda": float(curriculum.asset_candidate_lambda[asset_index].item()),
                "actual_lambda": float(curriculum.cell_lambda[cell_ids[asset_index]].item()),
                "counterfactual_adr_level": 0.0,
            }
            row.update({field: float((rollout | optimization)[field][asset_index].item()) for field in fields})
            asset_rows.append(row)

        # Cell/global都由per-asset rows等权聚合，replica数量不能改变某只手的统计权重。
        aggregate_fields = fields + ("candidate_lambda", "actual_lambda", "counterfactual_adr_level")
        cell_rows: list[dict[str, Any]] = []
        for cell_id in range(8):
            members = [row for row in asset_rows if row["cell_id"] == cell_id]
            if len(members) != 10:
                raise RuntimeError(f"MVP80 diagnostics expected 10 assets in cell {cell_id}, got {len(members)}")
            cell_rows.append(
                {
                    "update": int(self.epoch_num),
                    "transitions": transitions,
                    "scope": "cell",
                    "scope_index": cell_id,
                    "cell_id": cell_id,
                    **self._mean_fields(members, aggregate_fields),
                }
            )
        global_row = {
            "update": int(self.epoch_num),
            "transitions": transitions,
            "scope": "global",
            "scope_index": 0,
            **self._mean_fields(asset_rows, aggregate_fields),
            **optimizer_scalars,
        }
        learning_rates = {str(group.get("name")): float(group["lr"]) for group in self.optimizer.param_groups}
        global_row["actor_base_lr"] = learning_rates["actor_base"]
        global_row["actor_residual_lr"] = learning_rates["actor_global_residual"]
        global_row["critic_lr"] = float(self.critic_optimizer.param_groups[0]["lr"])
        global_row.update(_linux_memory_snapshot())  # 同一update的host memory/swap evidence
        # rl_games epoch tuple的total time覆盖rollout+5 mini-epochs；GPU peak包含scene、buffer与optimizer。
        total_time = float(epoch_result[3])
        global_row["steps_per_second"] = float(self.curr_frames / max(total_time, 1.0e-9))
        if torch.cuda.is_available() and torch.device(self.ppo_device).type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(torch.device(self.ppo_device)))
            total_memory = int(torch.cuda.get_device_properties(torch.device(self.ppo_device)).total_memory)
            global_row["gpu_memory_bytes"] = peak_memory
            if peak_memory / total_memory >= float(self.config.get("gpu_memory_fraction_limit", 0.85)):
                raise RuntimeError(
                    "palm-rotation training exceeded GPU memory safety fraction: "
                    f"peak={peak_memory}, total={total_memory}"
                )
        self.metrics_recorder.record([global_row, *cell_rows, *asset_rows])  # 89 rows/update

        # TensorBoard只保存global与8-cell在线曲线；80-asset详情仅进入Parquet。
        if self.writer is not None:
            for field in (
                "reward_mean",
                "goal_count_mean",
                "net_turns_mean",
                "drop_rate",
                "kl_per_active_dof",
                "action_clamp_fraction",
                "residual_rms",
                "film_modulation_rms",
            ):
                self.writer.add_scalar(f"mvp80/global/{field}", global_row[field], transitions)
                for row in cell_rows:
                    self.writer.add_scalar(f"mvp80/cell_{row['cell_id']}/{field}", row[field], transitions)

    def train_epoch(self):
        r"""运行标准rollout/update，并在dataset释放前记录紧凑per-asset证据。"""

        if hasattr(self, "_resume_last_mean_rewards"):
            self.last_mean_rewards = float(self._resume_last_mean_rewards)  # 恢复被upstream train()重置的best门
            del self._resume_last_mean_rewards
        self._reset_optimization_metrics()
        result = super().train_epoch()
        self._record_update_metrics(result)
        return result

    def train(self):
        r"""执行rl_games训练循环，并在正常预算结束后发布单个metrics.parquet。"""

        result = super().train()
        self.metrics_recorder.finalize()
        return result

    def write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        actor_losses,
        critic_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ) -> None:
        r"""沿用rl_games统计，并在每320 updates保存固定评估锚点checkpoint。

        ``ContinuousA2CBase.train``在调用本方法前已经把``self.frame``增加当前batch，因此这里保存的frame、
        两套optimizers、课程和Parquet cursor都对应完整update，而不是collection前状态。
        """

        super().write_stats(
            total_time,
            epoch_num,
            step_time,
            play_time,
            update_time,
            actor_losses,
            critic_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            frame,
            scaled_time,
            scaled_play_time,
            curr_frames,
        )
        cadence = int(self.config.get("evaluation_frequency", 320))
        if cadence > 0 and int(epoch_num) % cadence == 0:
            path = f"{self.nn_dir}/evaluation_{self.config['name']}_ep_{int(epoch_num):05d}"
            self.save(path)  # full identity/model/dual-optimizer/curriculum/diagnostic state

    def get_full_state_weights(self) -> dict[str, Any]:
        r"""保存模型、两套optimizer、课程、诊断与可精确续接的随机/调度状态。"""

        self.metrics_recorder.flush(reason="checkpoint")  # checkpoint不得领先于durable metric rows
        state = super().get_full_state_weights()  # model、actor optimizer、normalizer、env state、identity
        state[CRITIC_OPTIMIZER_KEY] = self.critic_optimizer.state_dict()  # 独立critic Adam moments
        state[DIAGNOSTICS_RECORDER_KEY] = self.metrics_recorder.state_dict()  # shard inventory/append cursor
        state[TRAINING_CONTINUATION_KEY] = {
            "schema_version": "1.0.0",
            "last_lr": float(self.last_lr),  # adaptive scheduler下一update的base LR
            "entropy_coef": float(self.entropy_coef),  # scheduler可能共同修改的exploration权重
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }  # pickle-safe完整随机状态；rank-0 reset无噪声，但Normal action sampling仍依赖Torch RNG
        return state

    def save(self, filename: str) -> None:
        r"""在同一文件系统以temporary→replace原子发布完整checkpoint。

        Args:
            filename (str): rl_games传入的不含``.pth``目标路径。
        """

        destination = Path(filename if filename.endswith(".pth") else f"{filename}.pth")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")  # `<name>.pth.tmp`
        if temporary.exists():
            temporary.unlink()  # 只清理当前目标上次未发布的run-owned temporary
        state = self.get_full_state_weights()  # 先flush Parquet，再冻结同一update checkpoint state
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())  # rename前确保checkpoint bytes已提交到底层文件系统
        temporary.replace(destination)

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        r"""在identity gate通过后恢复model、actor optimizer、课程及critic optimizer。"""

        if CRITIC_OPTIMIZER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing independent critic optimizer state")
        if DIAGNOSTICS_RECORDER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing metrics recorder state")
        continuation = weights.get(TRAINING_CONTINUATION_KEY)
        if not isinstance(continuation, Mapping) or continuation.get("schema_version") != "1.0.0":
            raise RuntimeError("palm-rotation checkpoint is missing exact training continuation state")
        super().set_full_state_weights(weights, set_epoch=set_epoch)  # 先执行AnyMani identity验证
        self.critic_optimizer.load_state_dict(weights[CRITIC_OPTIMIZER_KEY])  # 精确恢复critic Adam moments
        self.metrics_recorder.load_state_dict(weights[DIAGNOSTICS_RECORDER_KEY])  # 核对durable Parquet shards
        self.last_lr = float(continuation["last_lr"])  # scheduler scalar不能只依赖optimizer param-group LR
        self.entropy_coef = float(continuation["entropy_coef"])
        self.update_lr(self.last_lr)  # 三参数组恢复与adaptive ratio一致的当前LR
        random.setstate(continuation["python_random_state"])
        np.random.set_state(continuation["numpy_random_state"])
        torch.set_rng_state(continuation["torch_cpu_rng_state"])
        cuda_states = continuation.get("torch_cuda_rng_states", [])
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        self._resume_last_mean_rewards = float(weights.get("last_mean_rewards", -1.0e9))


class PalmRotationPpoRunner(AnyManiMaskedRunner):
    r"""在进程局部Runner factories中注册MVP80 custom PPO。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""保留现有masked model/player并增加双optimizer algorithm factory。"""

        super().__init__(*args, **kwargs)
        self.algo_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: PalmRotationPpoAgent(**factory_kwargs),
        )
        self.player_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: AnyManiMaskedPpoPlayer(**factory_kwargs),
        )


def register_palm_rotation_ppo() -> None:
    r"""注册custom masked model与MVP structured network builder。"""

    register_anymani_masked_ppo()  # `anymani_masked_continuous`及shared player contract
    model_builder.register_model("anymani_palm_rotation_masked_continuous", PalmRotationMaskedContinuousModel)
    model_builder.register_network(PALM_ROTATION_NETWORK, PalmRotationRlGamesBuilder)


__all__ = [
    "CRITIC_OPTIMIZER_KEY",
    "DIAGNOSTICS_RECORDER_KEY",
    "PALM_ROTATION_NETWORK",
    "PALM_ROTATION_PPO_ALGO",
    "TRAINING_CONTINUATION_KEY",
    "bounded_adaptive_learning_rate",
    "PalmRotationPpoAgent",
    "PalmRotationPpoRunner",
    "PalmRotationMaskedContinuousModel",
    "PalmRotationRlGamesBuilder",
    "PalmRotationRlGamesNetwork",
    "register_palm_rotation_ppo",
    "stratified_asset_permutation",
]
