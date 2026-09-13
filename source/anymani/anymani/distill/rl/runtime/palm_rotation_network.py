r"""掌旋策略与rl_games之间的网络、概率分布适配边界。

神经网络结构仍由distill.models拥有；本模块把缓存的具名张量分流给actor和privileged critic，并把
有界动作均值转换为tanh推前Normal分布。适配不调用冻结N040、不混合actor/critic参数，也不执行optimizer。
模型始终注册在package下；窄compile只包装forward，因此state_dict键与参数分组不随执行方式改变。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import torch
from torch import nn

from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationActorOutput,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
    expanded_policy_log_std,
)
from anymani.distill.rl.algorithms.popart import PopArtValueNormalizer

from ..masked_ppo import AnyManiMaskedContinuousModel
from .palm_rotation_phase import normalize_phase_period_steps
from .palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
    palm_rotation_float_shapes,
)


def denormalize_value_readonly(model: Any, value: torch.Tensor) -> torch.Tensor:
    r"""在不更新value running moments的条件下恢复物理尺度预测。

    ``rl_games.BaseModelNetwork.denorm_value``内部调用同一个``RunningMeanStd.forward``；该module若处于
    train mode，会在执行``denorm=True``前先把输入写入running mean/variance/count。PPO诊断输入是normalized
    Critic output，不属于value-target统计总体，因此这里仅临时切换value normalizer本身，调用后恢复原生命周期。

    Args:
        model (Any): rl_games model facade，需暴露``normalize_value``、``value_mean_std``与``denorm_value``。
        value (torch.Tensor): normalized Critic prediction，形状``[M,1]``。

    Returns:
        torch.Tensor: 采用调用前冻结moments恢复的物理value，形状与输入一致。
    """

    if not bool(getattr(model, "normalize_value", False)):
        return value  # 未启用value normalization时物理空间与model输出空间相同
    normalizer = getattr(model, "value_mean_std", None)
    if not isinstance(normalizer, nn.Module):
        raise TypeError("normalized value model must expose an nn.Module value_mean_std")
    was_training = bool(normalizer.training)  # 只保存目标normalizer模式，不改变Actor/Critic train状态
    normalizer.eval()  # ``RunningMeanStd.forward``在eval mode只读已冻结moments
    try:
        return model.denorm_value(value)  # upstream保留clamp与epsilon的唯一反归一化实现
    finally:
        normalizer.train(was_training)  # 异常路径也恢复下一次合法return/value统计更新所需模式


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
        network_cfg = params.get("palm_rotation", {})
        self.phase_period_steps = normalize_phase_period_steps(network_cfg.get("phase_period_steps"))
        expected_shapes = {
            **palm_rotation_float_shapes(self.phase_period_steps),
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

        arm_raw = str(network_cfg.get("arm", "residual"))
        if arm_raw not in {"base", "residual", "direct", "direct_token"}:
            raise ValueError(f"unsupported palm-rotation actor arm: {arm_raw!r}")
        self.arm = cast(Literal["base", "residual", "direct", "direct_token"], arm_raw)
        initial_log_std = float(network_cfg.get("initial_log_std", -0.5))  # shared scalar$\log\sigma$
        max_log_std = float(network_cfg.get("max_log_std", -0.43))  # N000 early-budget exploration ceiling
        base_action_limit = float(network_cfg.get("base_action_limit", 0.8))  # 与0.2 residual构成exact action bound
        history_encoder_raw = str(network_cfg.get("history_encoder", "tcn"))  # History30归纳偏置run identity
        if history_encoder_raw not in {"tcn", "raw_stack"}:
            raise ValueError(f"unsupported palm-rotation history encoder: {history_encoder_raw!r}")
        history_encoder = cast(Literal["tcn", "raw_stack"], history_encoder_raw)
        self.package = PalmRotationActorCritic(
            arm=self.arm,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=base_action_limit,
            history_encoder=history_encoder,
            sigma_mode=network_cfg.get("sigma_mode", "global"),
            recovery_sigma_floor=network_cfg.get("recovery_sigma_floor"),
            phase_clock_enabled=self.phase_period_steps is not None,
        )  # actor/critic完全分参；N040不属于此module
        compile_mode_raw = network_cfg.get("compile_mode")  # 只允许编译纯actor/critic forward，不包装rl_games model
        if compile_mode_raw not in {None, "default", "reduce-overhead"}:
            raise ValueError(f"unsupported palm-rotation compile mode: {compile_mode_raw!r}")
        self.compile_mode = None if compile_mode_raw is None else str(compile_mode_raw)
        self._actor_forward: Callable[..., PalmRotationActorOutput] = self.package.actor.forward
        self._critic_forward: Callable[..., torch.Tensor] = self.package.critic.forward
        if self.compile_mode is not None:
            self._actor_forward = torch.compile(self._actor_forward, mode=self.compile_mode)
            self._critic_forward = torch.compile(self._critic_forward, mode=self.compile_mode)
        # Compiled bound functions不是nn.Module children；checkpoint keys与optimizers继续锚定`package`原始参数。
        identity = params.get("anymani_identity")
        if not isinstance(identity, dict):
            raise ValueError("palm-rotation network requires a JSON-safe AnyMani runtime identity")
        self.anymani_identity = identity  # checkpoint pre-load identity gate
        self.last_active_joint_mask: torch.Tensor | None = None  # masked Normal读取的当前batch$[B,16]$
        self.last_residual_mean: torch.Tensor | None = None  # scalar diagnostics的detached action residual
        self.last_direct_mean: torch.Tensor | None = None  # direct arm的detached完整authority mean
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
        contextual_modules: tuple[nn.Module, ...] = (
            actor.geometry_adapter,
            actor.owner_contact_projection,
            actor.palm_dynamic_projection,
            actor.joint_dynamic_projection,
            actor.tip_dynamic_projection,
            actor.global_backbone,
            actor.direct_head if self.arm in {"direct", "direct_token"} else actor.residual_head,  # type: ignore[union-attr]
        )
        contextual_ids = {id(parameter) for module in contextual_modules for parameter in module.parameters()}
        if actor.conditional_sigma_head is not None:
            contextual_ids.update(id(parameter) for parameter in actor.conditional_sigma_head.parameters())
        phase_adapter = getattr(actor, "phase_contextual_adapter", None)
        if phase_adapter is not None:
            contextual_ids.update(id(parameter) for parameter in phase_adapter.parameters())
        base = [
            parameter for parameter in actor.parameters() if id(parameter) not in contextual_ids
        ]  # temporal/local trunk
        contextual = [
            parameter for parameter in actor.parameters() if id(parameter) in contextual_ids
        ]  # graph action path
        if {id(parameter) for parameter in base} & {id(parameter) for parameter in contextual}:
            raise RuntimeError("actor local/contextual optimizer groups overlap")
        if len(base) + len(contextual) != len(list(actor.parameters())):
            raise RuntimeError("actor optimizer groups do not cover all parameters")
        return base, contextual

    def forward(self, input_dict: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
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

        if self.phase_period_steps is None:
            actor_output = self._actor_forward(actor_observation, geometry)
            value = self._critic_forward(critic_observation, geometry).unsqueeze(-1)
        else:
            phase = observation["phase_clock"].float()  # 使用同一 rollout sample 的 phase，不在重放时重新计时。
            actor_output = self._actor_forward(actor_observation, geometry, phase_clock=phase)
            value = self._critic_forward(critic_observation, geometry, phase_clock=phase).unsqueeze(-1)
        self.last_active_joint_mask = actor_observation.jnt_valid  # probability/entropy/KL ghost mask
        self.last_residual_mean = (
            actor_output.residual_mean.detach() if actor_output.residual_mean is not None else None
        )  # residual/base diagnostics；direct保持None
        self.last_direct_mean = (
            actor_output.direct_mean.detach() if actor_output.direct_mean is not None else None
        )  # direct diagnostics；residual/base保持None
        self.last_film_modulation_rms = actor_output.film_modulation_rms.detach()  # local hidden调制幅度
        logstd = expanded_policy_log_std(actor_output.log_std, actor_output.mean, actor_observation.jnt_valid)
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

        def __init__(self, a2c_network, **kwargs: Any) -> None:
            r"""模型创建时确定value坐标策略，使训练、加载与诊断使用同一种normalizer。"""
            super().__init__(a2c_network, **kwargs)
            identity = getattr(a2c_network, "anymani_identity", {})
            training = identity.get("training", {}) if isinstance(identity, Mapping) else {}
            mode = training.get("value_normalization", "rms")
            if mode not in {"rms", "popart"}:
                raise ValueError("unknown value normalization strategy")
            if mode == "popart":
                if not self.normalize_value:
                    raise ValueError("PopArt requires value normalization")
                self.value_mean_std = PopArtValueNormalizer()

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
                0.5 * ((latent_action - latent_mean) / sigma).square() + logstd + 0.5 * math.log(2.0 * math.pi)
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
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(action_mean.abs() <= 1.0 + 1.0e-6),
                "palm-rotation deterministic action mean escaped [-1,1]",
            )  # analytic bound仍逐forward fail closed，但不把CUDA stream同步回host
            active_float = active_mask.to(dtype=action_mean.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            logstd = expanded_policy_log_std(logstd, action_mean, active_mask)  # 全局/条件输出统一后再进入概率计算。
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
            if self.a2c_network.arm in {"direct", "direct_token"}:
                direct = getattr(self.a2c_network, "last_direct_mean", None)
                if not isinstance(direct, torch.Tensor):
                    raise RuntimeError("palm-rotation direct actor did not expose its bounded mean")
                result["direct_means"] = direct  # 已detach，与rollout action mean逐值一致
            else:
                residual = getattr(self.a2c_network, "last_residual_mean", None)
                if not isinstance(residual, torch.Tensor):
                    raise RuntimeError("palm-rotation residual/base actor did not expose bounded residual")
                result["residuals"] = residual  # 已detach，不延长rollout autograd graph
            film = getattr(self.a2c_network, "last_film_modulation_rms", None)
            if not isinstance(film, torch.Tensor):
                raise RuntimeError("palm-rotation actor did not expose geometry FiLM diagnostics")
            result["film_modulations"] = film  # `[B,16]`，与joint active mask同轴
            return result
