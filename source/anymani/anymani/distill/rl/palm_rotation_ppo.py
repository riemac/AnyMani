r"""异构掌旋PPO的训练生命周期与rl_games注册入口。

本模块保持三项训练语义：

1. actor与critic读取同一份rollout-cached FP32 $Z^e$，但参数与optimizers完全分离；
2. canonical ghost joints不进入Normal log-prob、entropy、KL、bounds或动作执行；
3. 每个minibatch对当前支持集的assets严格等量，归一化与排列来自algorithms.ppo_batch。

冻结N040在rollout中计算一次，不进入本模块的optimizers。网络/概率适配位于runtime.palm_rotation_network，
运行诊断与只读梯度探针分别位于runtime.palm_rotation_diagnostics和palm_rotation_probes。主agent保留参数
更新、状态恢复和调用顺序，使训练流程可以连续阅读；公开名称继续服务已有入口和checkpoint审计工具。
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import common_losses
from torch import nn
from torch.nn.utils import clip_grad_norm_

from anymani.distill.diagnostics.recording.rl.palm_rotation import PalmRotationMetricsRecorder

from .algorithms.ppo_batch import (
    bounded_adaptive_learning_rate,
    normalize_advantages_per_asset,
    stratified_asset_permutation,
)
from .masked_ppo import (
    AnyManiMaskedPpoAgent,
    AnyManiMaskedPpoPlayer,
    AnyManiMaskedRunner,
    register_anymani_masked_ppo,
)
from .runtime import palm_rotation_diagnostics as diagnostics
from .runtime import palm_rotation_probes as probes
from .runtime.palm_rotation_diagnostics import (
    PalmRotationPpoDiagnostics,
    rollout_policy_mechanism_metrics,
)
from .runtime.palm_rotation_network import (
    PalmRotationMaskedContinuousModel,
    PalmRotationRlGamesBuilder,
    PalmRotationRlGamesNetwork,
    denormalize_value_readonly,
)
from .runtime.palm_rotation_warm_start import (
    load_actor_init_checkpoint,
    load_critic_init_checkpoint,
    should_load_actor_init_checkpoint,
)

PALM_ROTATION_PPO_ALGO = "anymani_palm_rotation_ppo"
PALM_ROTATION_NETWORK = "anymani_palm_rotation"
CRITIC_OPTIMIZER_KEY = "anymani_critic_optimizer"
DIAGNOSTICS_RECORDER_KEY = "anymani_metrics_recorder"
TRAINING_CONTINUATION_KEY = "anymani_training_continuation"


def validate_gradient_probe_compile_compatibility(
    compile_mode: str | None,
    probe_frequency: int,
    full_gradient_shadow_frequency: int = 0,
) -> None:
    r"""拒绝Inductor donated-buffer与多次head-gradient反向的不兼容组合。

    当前gradient probe对同一forward graph按资产执行多次``autograd.grad(retain_graph=True)``；PyTorch
    AOTAutograd在compiled backward启用non-empty donated buffers时要求单次``retain_graph=False``。Eager路径已通过
    真实16资产probe；compile仍可用于probe关闭的纯性能/训练run。该门在scene创建前由launcher调用。
    """

    if probe_frequency < 0 or full_gradient_shadow_frequency < 0:
        raise ValueError("gradient probe frequencies must be non-negative")
    if compile_mode is not None and (probe_frequency > 0 or full_gradient_shadow_frequency > 0):
        raise ValueError(
            "head-gradient probe requires eager actor/critic forward; disable --torch_compile or the probe"
        )


class PalmRotationPpoAgent(AnyManiMaskedPpoAgent):
    r"""双optimizer、严格分层minibatch与cached-N040 identity的PPO agent。"""

    def __init__(self, base_name: str, params: dict[str, Any]) -> None:
        r"""让upstream完成Runner状态构造，再替换为actor/critic独立Adam optimizers。"""

        super().__init__(base_name, params)
        if self.has_central_value:
            raise ValueError(
                "palm-rotation custom package already owns the privileged critic; duplicate CV is forbidden"
            )
        if self.mixed_precision:
            raise ValueError("actor, critic, PPO losses and optimizers must remain FP32")
        if self.multi_gpu:
            raise ValueError("MVP80 dual-optimizer agent currently supports one GPU only")
        self.diagnostics = PalmRotationPpoDiagnostics()  # 公式不变，只移除upstream逐microbatch`.cpu()`同步
        network = self.model.a2c_network  # AnyManiMaskedContinuousModel facade下的正式network
        if not isinstance(network, PalmRotationRlGamesNetwork):
            raise TypeError("palm-rotation PPO agent received an incompatible network")

        # Actor-only迁移发生在fresh optimizers构造前；checkpoint其余state从未交给rl_games restore。
        actor_init_path = str(self.config.get("actor_init_checkpoint", "")).strip()
        runtime_identity = network.anymani_identity
        training_identity = runtime_identity.get("training") if isinstance(runtime_identity, Mapping) else None
        warm_start = training_identity.get("actor_warm_start") if isinstance(training_identity, Mapping) else None
        load_actor_init = should_load_actor_init_checkpoint(
            actor_init_path=actor_init_path,
            warm_start=warm_start,
            full_checkpoint_resume=bool(self.config.get("full_checkpoint_resume", False)),
        )
        if load_actor_init:
            loaded_keys = load_actor_init_checkpoint(
                network.package.actor,
                actor_init_path,
                expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"]),  # type: ignore[index]
            )
            if len(loaded_keys) != int(warm_start["loaded_tensor_count"]):  # type: ignore[index]
                raise RuntimeError("actor-init loaded tensor count disagrees with inspected identity")
            if isinstance(warm_start, Mapping) and bool(warm_start.get("initialize_critic", False)):
                load_critic_init_checkpoint(
                    self.model, actor_init_path, expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"])
                )

        base_parameters, contextual_parameters = network.actor_parameter_groups()  # disjoint actor groups
        critic_parameters = list(network.package.critic.parameters())  # completely separate$\theta^c$
        actor_ids = {id(parameter) for parameter in (*base_parameters, *contextual_parameters)}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids:
            raise RuntimeError("actor and critic optimizer parameters overlap")

        # 三个LR锚点来自MVP计划；adaptive scheduler只更新base LR，update_lr保持固定比例。
        self._base_lr_reference = float(self.config["learning_rate"])  # 默认$3e-4$
        self._base_lr_ceiling = float(self.config.get("adaptive_lr_max", self._base_lr_reference))
        if self._base_lr_ceiling != self._base_lr_reference:
            raise ValueError("MVP adaptive_lr_max must equal the declared actor base learning-rate anchor")
        self.actor_arm = network.arm  # 四种arm决定side-channel、机制指标与第二参数组名称
        secondary_lr_key = (
            "contextual_learning_rate" if self.actor_arm in {"direct", "direct_token"} else "residual_learning_rate"
        )
        self._secondary_lr_ratio = float(self.config.get(secondary_lr_key, 1.0e-4)) / self._base_lr_reference
        self._secondary_group_name = (
            "actor_contextual_direct" if self.actor_arm in {"direct", "direct_token"} else "actor_global_residual"
        )
        self._critic_lr_ratio = float(self.config.get("critic_learning_rate", 5.0e-4)) / self._base_lr_reference
        self._gradient_accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1))
        if self._gradient_accumulation_steps < 1 or self.num_minibatches % self._gradient_accumulation_steps != 0:
            raise ValueError("gradient accumulation must divide the number of stratified activation minibatches")
        fused = torch.device(self.ppo_device).type == "cuda"  # CUDA正式训练使用fused Adam
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "lr": self.last_lr, "name": "actor_base"},
                {
                    "params": contextual_parameters,
                    "lr": self.last_lr * self._secondary_lr_ratio,
                    "name": self._secondary_group_name,
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
        self.advantage_normalization_scope = str(self.config.get("advantage_normalization_scope", "global"))
        if self.advantage_normalization_scope not in {"global", "per_asset_rollout"}:
            raise ValueError("advantage_normalization_scope must be global or per_asset_rollout")
        if not self.normalize_advantage:
            raise ValueError("palm-rotation PPO requires normalize_advantage for a declared normalization scope")
        self.last_stratified_permutation: torch.Tensor | None = None  # diagnostics/test evidence
        self.last_advantage_asset_means: torch.Tensor | None = None  # full-rollout raw GAE mean `[A]`
        self.last_advantage_asset_stds: torch.Tensor | None = None  # full-rollout raw GAE sample std `[A]`
        identity = self._runtime_identity()
        if not isinstance(identity, dict) or not isinstance(identity.get("identity_digest"), str):
            raise RuntimeError("palm-rotation diagnostics require the exact runtime identity")
        self.metrics_recorder = PalmRotationMetricsRecorder(
            self.experiment_dir,
            identity_digest=identity["identity_digest"],
            flush_every_updates=int(self.config.get("diagnostics_flush_updates", 50)),
        )  # run-owned Parquet shard lifecycle
        self._optimization_count = torch.zeros(self.asset_count, device=self.ppo_device)  # mini-epoch samples$[A]$
        common_optimization_fields = (
            "advantage",
            "advantage_square",
            "value_error",
            "return_target",
            "return_target_square",
            "value_prediction",
            "value_prediction_square",
            "value_residual_square",
            "value_error_physical",
            "value_clip_fraction",
            "kl",
            "clip_fraction",
            "action_rms",
            "policy_mean_rms",
            "policy_mean_near_bound_fraction",
            "film_modulation_rms",
        )
        self._mechanism_metric_fields = (
            ("direct_mean_rms", "direct_mean_near_bound_fraction", "direct_pre_tanh_derivative_mean")
            if self.actor_arm in {"direct", "direct_token"}
            else ("base_mean_rms", "residual_rms", "residual_fraction")
        )
        self._optimization_sums = {
            name: torch.zeros(self.asset_count, device=self.ppo_device)
            for name in (*common_optimization_fields, *self._mechanism_metric_fields)
        }  # 当前update跨全部minibatches×mini-epochs之和
        self._gradient_probe_per_asset: dict[str, torch.Tensor] | None = None
        self._gradient_probe_global: dict[str, float] | None = None
        self._optimizer_step_count = 0  # 当前update真实optimizer step次数
        self._optimizer_microbatch_count = 0  # 当前update真实forward/backward microbatch次数
        self._gradient_microbatch_index = 0  # 必须在每个update边界回到0
        self._optimizer_scalar_sums = {
            name: torch.zeros((), dtype=torch.float32, device=self.ppo_device)
            for name in (
                "actor_loss",
                "critic_loss",
                "entropy",
                "policy_sigma",
                "actor_grad_norm",
                "critic_grad_norm",
            )
        }  # 无per-asset归属的标量留在GPU累计，update边界才执行六次host transfer

    def init_tensors(self) -> None:
        r"""在upstream experience buffer增加detached action-residual side-channel。"""

        super().init_tensors()
        batch = self.num_agents * self.num_actors  # rollout并行样本数$N$
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        self.experience_buffer.tensor_dict[mechanism_key] = torch.zeros(
            self.horizon_length,
            batch,
            16,
            dtype=torch.float32,
            device=self.ppo_device,
        )  # `[H,N,16]`，与actions/mus同axis
        self.update_list.append(mechanism_key)  # play_steps从custom model输出写入buffer
        self.tensor_list.append(mechanism_key)  # rollout结束后swap env/time并flatten
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
            group["lr"] = current if group.get("name") == "actor_base" else current * self._secondary_lr_ratio
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
        expected_secondary = expected_base * self._secondary_lr_ratio
        if abs(groups.get("actor_base", -1.0) - expected_base) > 1.0e-12:
            raise RuntimeError(f"actor base LR drifted before optimizer step: {groups}")
        if abs(groups.get(self._secondary_group_name, -1.0) - expected_secondary) > 1.0e-12:
            raise RuntimeError(f"actor contextual LR ratio drifted before optimizer step: {groups}")

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
        c2 = (current_sigma.square() + (old_latent - current_latent).square()) / (2.0 * (old_sigma.square() + 1.0e-5))
        weights = active_mask.to(dtype=current_mu.dtype)
        return ((c1 + c2 - 0.5) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def _gradient_probe_parameters(self) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return probes.gradient_probe_parameters(self)

    @staticmethod
    def _per_asset_gradient_matrix(
        objective: torch.Tensor,
        labels: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
        *,
        asset_count: int,
    ) -> torch.Tensor:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return probes.per_asset_gradient_matrix(objective, labels, parameters, asset_count=asset_count)

    def _run_gradient_probe(
        self,
        *,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return probes.run_gradient_probe(
            self, actor_objective=actor_objective, critic_objective=critic_objective, labels=labels
        )

    def _run_full_actor_gradient_shadow(
        self,
        *,
        global_objective: torch.Tensor,
        per_asset_objective: torch.Tensor,
        labels: torch.Tensor,
        replica_halves: torch.Tensor,
    ) -> None:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return probes.run_full_actor_gradient_shadow(
            self,
            global_objective=global_objective,
            per_asset_objective=per_asset_objective,
            labels=labels,
            replica_halves=replica_halves,
        )

    @staticmethod
    def _index_dataset_value(value: Any, indices: torch.Tensor) -> Any:
        r"""对dataset tensor或一层Dict统一应用batch permutation。"""

        if isinstance(value, dict):
            return {key: tensor[indices] for key, tensor in value.items()}  # named experience tensors
        return value[indices] if isinstance(value, torch.Tensor) else value

    def prepare_dataset(self, batch_dict: dict[str, Any]) -> None:
        r"""形成global/per-asset两份GAE，按显式scope选择Actor输入，再分层排列minibatches。"""

        raw_returns = batch_dict.get("returns")
        raw_values = batch_dict.get("values")
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("palm-rotation rollout lacks raw return/value tensors")
        rollout_observation = batch_dict.get("obses")
        if not isinstance(rollout_observation, Mapping) or "prototype_index" not in rollout_observation:
            raise RuntimeError("palm-rotation rollout lacks prototype labels for advantage normalization")
        rollout_labels = rollout_observation["prototype_index"].reshape(-1).long()  # permutation前的`[B]` asset axis
        raw_advantages = (raw_returns - raw_values).sum(dim=1)  # upstream PPO定义的物理GAE `[B]`
        if raw_advantages.numel() % self.horizon_length != 0:
            raise RuntimeError("flattened rollout does not divide into complete environment trajectories")
        flat_index = torch.arange(raw_advantages.numel(), device=raw_advantages.device)
        environment_index = torch.div(flat_index, self.horizon_length, rounding_mode="floor")  # env-major flatten axis
        expected_labels = environment_index.remainder(self.asset_count)  # runtime route定义$k_e=e\bmod A$
        if not bool(torch.equal(rollout_labels, expected_labels)):
            raise RuntimeError("rollout prototype labels disagree with env-major round-robin routing")
        replica_index = torch.div(environment_index, self.asset_count, rounding_mode="floor")
        replica_halves = replica_index.remainder(2)  # even/odd replica IDs形成确定性、近等量的独立轨迹halves
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        mechanism = batch_dict.get(mechanism_key)  # detached arm-specific rollout diagnostic`[B,16]`
        if not isinstance(mechanism, torch.Tensor):
            raise RuntimeError(f"palm-rotation rollout is missing {mechanism_key} side-channel")
        rollout_mean = batch_dict.get("mus")  # rollout策略生成动作时的有界均值`[B,16]`
        if not isinstance(rollout_mean, torch.Tensor) or rollout_mean.shape != mechanism.shape:
            raise RuntimeError("palm-rotation rollout mean and mechanism side-channel shapes disagree")
        frozen_rollout_mean = rollout_mean.detach().clone()  # 与dataset可变KL reference断开storage alias
        super().prepare_dataset(batch_dict)  # 保持upstream GAE/value normalization与PPO fields
        global_advantages = self.dataset.values_dict.get("advantages")
        if not isinstance(global_advantages, torch.Tensor) or global_advantages.shape != raw_advantages.shape:
            raise RuntimeError("upstream global advantage tensor disagrees with rollout GAE shape")
        per_asset_advantages, asset_means, asset_stds = normalize_advantages_per_asset(
            raw_advantages,
            rollout_labels,
            asset_count=self.asset_count,
        )  # `[B]`与`[A]`；在任何minibatch permutation前固定完整rollout moments
        self.dataset.values_dict["global_advantages"] = global_advantages.detach()
        self.dataset.values_dict["per_asset_advantages"] = per_asset_advantages.detach()
        self.dataset.values_dict["replica_halves"] = replica_halves.detach()
        self.dataset.values_dict["advantages"] = (
            per_asset_advantages if self.advantage_normalization_scope == "per_asset_rollout" else global_advantages
        )  # 只有该具名字段进入主Actor surrogate
        self.last_advantage_asset_means = asset_means.detach()
        self.last_advantage_asset_stds = asset_stds.detach()
        self.dataset.values_dict["raw_returns"] = raw_returns.detach()  # normalization前物理return target`[B,1]`
        self.dataset.values_dict["raw_values"] = raw_values.detach()  # rollout时denormalized value`[B,1]`
        self.dataset.values_dict["rollout_mu"] = frozen_rollout_mean  # 五轮PPO均只读的$\mu^{rollout}$
        self.dataset.values_dict[mechanism_key] = mechanism  # optimizer diagnostics，不进入loss
        film_modulations = batch_dict.get("film_modulations")
        if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != mechanism.shape:
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
            key: self._index_dataset_value(value, permutation) for key, value in self.dataset.values_dict.items()
        }  # 所有old policy/value/action/obs字段保持同一sample correspondence
        self.last_stratified_permutation = permutation.detach()  # scalar/table diagnostics可审计本update顺序

    def calc_gradients(self, input_dict: dict[str, Any]) -> None:
        r"""同一前向图分别对$\theta^a$与$\theta^c$执行FP32 PPO/value更新。"""

        value_predictions = input_dict["old_values"]  # rollout normalized values`[M,1]`
        old_neglogp = input_dict["old_logp_actions"]  # masked action negative log probability`[M]`
        advantage = input_dict["advantages"]  # identity-selected global/per-asset GAE`[M]`
        global_advantages = input_dict.get("global_advantages")  # 同一rollout的upstream global标准化GAE
        per_asset_advantages = input_dict.get("per_asset_advantages")  # full-rollout逐资产标准化GAE
        replica_halves = input_dict.get("replica_halves")  # even/odd runtime replica split `[M]`
        if not all(
            isinstance(value, torch.Tensor) for value in (global_advantages, per_asset_advantages, replica_halves)
        ):
            raise RuntimeError("PPO minibatch lacks advantage-scope or replica-half shadow tensors")
        kl_reference_mu = input_dict["mu"]  # rl_games逐minibatch更新的KL参考means`[M,16]`
        kl_reference_sigma = input_dict["sigma"]  # 与means同生命周期的KL参考stds`[M,16]`
        rollout_mean = input_dict.get("rollout_mu")  # 当前rollout采样时冻结的策略means`[M,16]`
        if not isinstance(rollout_mean, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks immutable rollout policy means")
        returns = input_dict["returns"]  # normalized return targets`[M,1]`
        raw_returns = input_dict.get("raw_returns")  # value normalization前物理return targets`[M,1]`
        raw_values = input_dict.get("raw_values")  # rollout时denormalized value predictions`[M,1]`
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks raw return/value diagnostics")
        actions = input_dict["actions"]  # sampled canonical actions`[M,16]`
        observation = self._preproc_obs(input_dict["obs"])  # named Dict；normalize_input=False
        labels = observation["prototype_index"].reshape(-1).long()  # sampler certificate，不进模型
        result = self.model({"is_train": True, "prev_actions": actions, "obs": observation})
        new_neglogp = result["prev_neglogp"]  # masked active-joint likelihood
        values = result["values"]  # privileged critic prediction`[M,1]`
        entropy = result["entropy"]  # mean entropy per active DoF`[M]`
        mu = result["mus"]  # current actor means`[M,16]`
        sigma = result["sigmas"]  # shared scalar expanded to`[M,16]`

        # Actor objective只含clipped surrogate、active-DoF entropy与masked bounds；无critic gradient path。
        actor_loss_vector = self.actor_loss_func(old_neglogp, new_neglogp, advantage, self.ppo, self.e_clip)
        bounds_loss_vector = self.bound_loss(mu)  # active-DoF mean bounds penalty
        actor_objective_vector = (
            actor_loss_vector - entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
        )  # `[M]`，仅供per-asset gradient probe；主loss归约仍由apply_masks定义
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
        critic_objective_vector = 0.5 * self.critic_coef * critic_vector.reshape(-1)  # `[M]`

        # 非有限objective会污染Adam moments与后续checkpoint；设备异步断言保留逐项故障名但不阻塞host。
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
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(value).all(),
                f"palm-rotation PPO produced non-finite {name}",
            )

        # 可选完整Actor shadow在同一forward/minibatch上只改变advantage scope，不写主gradient或optimizer。
        full_gradient_shadow_frequency = int(self.config.get("full_gradient_shadow_frequency", 0))
        if (
            full_gradient_shadow_frequency > 0
            and int(self.epoch_num) % full_gradient_shadow_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            common_actor_regularizer = -entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
            global_actor_objective = (
                self.actor_loss_func(
                    old_neglogp,
                    new_neglogp,
                    cast(torch.Tensor, global_advantages),
                    self.ppo,
                    self.e_clip,
                )
                + common_actor_regularizer
            )
            per_asset_actor_objective = (
                self.actor_loss_func(
                    old_neglogp,
                    new_neglogp,
                    cast(torch.Tensor, per_asset_advantages),
                    self.ppo,
                    self.e_clip,
                )
                + common_actor_regularizer
            )
            self._run_full_actor_gradient_shadow(
                global_objective=global_actor_objective,
                per_asset_objective=per_asset_actor_objective,
                labels=labels,
                replica_halves=cast(torch.Tensor, replica_halves),
            )

        # 固定cadence只在本update首个stratified minibatch形成一次head-gradient proxy。
        gradient_probe_frequency = int(self.config.get("gradient_probe_frequency", 0))
        if (
            gradient_probe_frequency > 0
            and int(self.epoch_num) % gradient_probe_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            self._run_gradient_probe(
                actor_objective=actor_objective_vector,
                critic_objective=critic_objective_vector,
                labels=labels,
            )

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
            actor_grad_norm = clip_grad_norm_(
                network.package.actor.parameters(), self.grad_norm
            )  # 逻辑batch$\|g_a\|_2$
            critic_grad_norm = clip_grad_norm_(
                network.package.critic.parameters(), self.grad_norm
            )  # 逻辑batch$\|g_c\|_2$
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(actor_grad_norm) & torch.isfinite(critic_grad_norm),
                "palm-rotation PPO produced non-finite actor or critic gradient norm",
            )
            self.optimizer.step()
            self.critic_optimizer.step()
            network.package.actor.project_exploration_parameters()  # optimizer后立即恢复$\log\sigma\le-0.43$
            self._optimizer_step_count += 1
            self._optimizer_scalar_sums["actor_grad_norm"].add_(actor_grad_norm.detach().float())
            self._optimizer_scalar_sums["critic_grad_norm"].add_(critic_grad_norm.detach().float())
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
            self._optimizer_scalar_sums[name].add_(value.float())  # scalar detach已阻断autograd graph

        # Adaptive scheduler消费active-DoF-normalized KL，ghost sigma/mean不影响统计。
        active_mask = network.last_active_joint_mask
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("palm-rotation network did not expose active-joint mask")
        with torch.no_grad():
            kl_per_sample = self.masked_policy_kl(
                mu.detach(), sigma.detach(), kl_reference_mu, kl_reference_sigma, active_mask
            )
            kl = kl_per_sample.mean()
            ratio = torch.exp(old_neglogp - new_neglogp.detach())  # PPO importance ratio`[M]`
            clip_fraction = (torch.abs(ratio - 1.0) > self.e_clip).float()  # clipped sample indicator
            active_float = active_mask.float()
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            action_rms = torch.sqrt((actions.square() * active_float).sum(dim=-1) / active_count)
            value_prediction_physical = denormalize_value_readonly(self.model, values.detach()).reshape(-1)
            return_target_physical = raw_returns.reshape(-1)
            value_residual_physical = value_prediction_physical - return_target_physical
            value_clip_fraction = (
                (values.detach().reshape(-1) - value_predictions.reshape(-1)).abs() > self.e_clip
            ).float()  # normalized value space中的PPO clip激活
            mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
            mechanism = input_dict.get(mechanism_key)  # 与$\mu^{rollout}$同一次forward保存的arm-specific量
            if not isinstance(mechanism, torch.Tensor):
                raise RuntimeError(f"PPO minibatch lacks {mechanism_key} diagnostics")
            mechanism_metrics = rollout_policy_mechanism_metrics(
                rollout_mean,
                mechanism,
                active_mask,
                actor_arm=cast(Literal["base", "residual", "direct", "direct_token"], self.actor_arm),
            )
            film_modulations = input_dict.get("film_modulations")
            if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != actions.shape:
                raise RuntimeError("PPO minibatch geometry FiLM diagnostics disagree with action shape")
            film_modulation_rms = (film_modulations * active_float).sum(dim=-1) / active_count
            optimization_values = {
                "advantage": advantage.detach(),
                "advantage_square": advantage.detach().square(),
                "value_error": torch.abs(values.detach().squeeze(-1) - returns.squeeze(-1)),
                "return_target": return_target_physical,
                "return_target_square": return_target_physical.square(),
                "value_prediction": value_prediction_physical,
                "value_prediction_square": value_prediction_physical.square(),
                "value_residual_square": value_residual_physical.square(),
                "value_error_physical": value_residual_physical.abs(),
                "value_clip_fraction": value_clip_fraction,
                "kl": kl_per_sample,
                "clip_fraction": clip_fraction,
                "action_rms": action_rms,
                "policy_mean_rms": mechanism_metrics["policy_mean_rms"],
                "policy_mean_near_bound_fraction": mechanism_metrics["policy_mean_near_bound_fraction"],
                "film_modulation_rms": film_modulation_rms,
            }
            optimization_values.update(
                {name: mechanism_metrics[name] for name in self._mechanism_metric_fields}
            )  # arm-specific字段均来自同一冻结rollout分解，不混入后续KL reference
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
            self._optimizer_scalar_sums[name].zero_()  # 保留device scalar storage，避免每update重新分配
        self._gradient_probe_per_asset = None
        self._gradient_probe_global = None

    def _drain_optimizer_scalars(self) -> dict[str, float]:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return diagnostics.drain_optimizer_scalars(self)

    def _drain_optimization_metrics(self) -> dict[str, torch.Tensor]:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return diagnostics.drain_optimization_metrics(self)

    @staticmethod
    def _mean_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return diagnostics.mean_fields(rows, fields)

    def _record_update_metrics(self, epoch_result: tuple[Any, ...]) -> None:
        r"""委托专用模块处理同一agent状态，保持训练hook和参数语义。"""

        return diagnostics.record_update_metrics(self, epoch_result)

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
    "denormalize_value_readonly",
    "normalize_advantages_per_asset",
    "validate_gradient_probe_compile_compatibility",
    "PalmRotationPpoAgent",
    "PalmRotationPpoRunner",
    "PalmRotationMaskedContinuousModel",
    "PalmRotationRlGamesBuilder",
    "PalmRotationRlGamesNetwork",
    "register_palm_rotation_ppo",
    "stratified_asset_permutation",
]
