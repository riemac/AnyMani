r"""canonical masked PPO 的 rl_games compatibility 层。

rl_games 的标准 continuous model 会把 `[B,16]` 每一维都纳入 Normal log-prob、entropy
与 bounds loss。本模块在本地 rl_games boundary 实现 canonical active-action contract：

$$
\log p(a|s)=\sum_{j:m_j=1}\log\mathcal N(a_j;\mu_j,\sigma_j),
\qquad
H=\frac{1}{\sum_jm_j}\sum_{j:m_j=1}H_j.
$$

inactive action 在 model sampling 后清零；inactive mean 为 0、inactive log-std 为 0
（即 sigma=1），使 rl_games 内部 KL 统计不会引入 ghost 项。bounds / regularisation
由 ``AnyManiMaskedPpoAgent`` 按 active joint 数取均值。该文件不修改本地
``/home/hac/isaac/rl_games`` 源码，Runner factory 只在 AnyMani 训练进程中注册。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from rl_games.algos_torch import a2c_continuous, model_builder, models, network_builder, players, torch_ext
from rl_games.torch_runner import Runner

from anymani.distill.models.input_adapters.geometry import GeometryEncoderCfg, ImplicitGeometryEncoder
from anymani.distill.models.policy import (
    CANONICAL_JOINT_COUNT,
    CANONICAL_OWNER_COUNT,
    CanonicalEvidenceBank,
    CanonicalPolicyCfg,
    EmbodimentPolicy,
    EmbodimentPolicyInput,
)

CANONICAL_FLAT_JOINT_FEATURES = 5
"""每个 joint 在 flat GM actor obs 中的 `[q, qd, previous_delta, q_min, q_max]` 维度。"""

CANONICAL_COMMAND_FEATURES = 6
"""PALM/task command 的 `[axis_h, error_so3_h]` 维度。"""

CANONICAL_OBJECT_POSITION_FEATURES = 3
"""PALM/object 相对位置特征维度，单位 m。"""

CANONICAL_OBJECT_ORIENTATION_FEATURES = 6
"""PALM/object rot6d 特征维度，无量纲。"""

CANONICAL_TIP_CONTACT_FEATURES = 4
"""四个 TIP owner 的 contact binary 特征维度。"""

ANYMANI_CHECKPOINT_IDENTITY_KEY = "anymani_identity"
"""rl_games checkpoint 顶层的 AnyMani dataset/provider 身份字段。"""

CANONICAL_MASKED_OBS_DIM = (
    CANONICAL_JOINT_COUNT * CANONICAL_FLAT_JOINT_FEATURES
    + CANONICAL_COMMAND_FEATURES
    + CANONICAL_OBJECT_POSITION_FEATURES
    + CANONICAL_OBJECT_ORIENTATION_FEATURES
    + CANONICAL_TIP_CONTACT_FEATURES
    + 1
    + CANONICAL_JOINT_COUNT
)
"""canonical flat actor obs 维度：joint state + command + asset_row + active mask。"""


def validate_anymani_checkpoint_identity(
    *,
    runtime_identity: dict[str, Any],
    checkpoint_identity: object,
) -> None:
    r"""在 state-dict restore 前验证当前 runtime 与 checkpoint 的资产/$Z$ 身份。

    actor checkpoint 内的 frozen buffers 自身不足以保证正确恢复：若当前环境已经按另一份资产顺序
    构造，直接 ``load_state_dict`` 会用旧 buffer 覆盖新 provider，却不会改变环境 routing。因此必须
    先核对 dataset digest、canonical manifest digest、有序 asset IDs、physical hashes 和 $Z$ table
    digest；任一字段不同都拒绝恢复。

    Args:
        runtime_identity: 当前环境/provider 的 JSON-safe identity mapping。
        checkpoint_identity: checkpoint 顶层读取的候选 identity。

    Raises:
        RuntimeError: checkpoint 缺失 identity 或任一身份字段不一致。
    """

    if not isinstance(checkpoint_identity, dict):
        raise RuntimeError("heterogeneous AnyMani checkpoint is missing required anymani_identity metadata")
    compared_fields = tuple(sorted(set(runtime_identity) | set(checkpoint_identity)))  # schema 增删也视为不兼容
    mismatched = [field for field in compared_fields if runtime_identity.get(field) != checkpoint_identity.get(field)]
    if mismatched:
        runtime_digest = runtime_identity.get("identity_digest", "missing")  # 当前 2048-row runtime 身份摘要
        checkpoint_digest = checkpoint_identity.get("identity_digest", "missing")  # 待恢复 checkpoint 摘要
        raise RuntimeError(
            "AnyMani checkpoint identity mismatch before model restore: "
            f"fields={mismatched}, runtime_digest={runtime_digest}, checkpoint_digest={checkpoint_digest}"
        )


class CanonicalMaskedPpoBuilder(network_builder.NetworkBuilder):
    r"""把 rl_games flat obs lower 成 shared ``EmbodimentPolicy`` 输入。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        r"""保存 YAML network mapping，实际 module 在 build 阶段按 env shape 实例化。"""

        self.params = params

    def build(self, name: str, **kwargs: Any) -> CanonicalMaskedPpoNetwork:
        r"""构造 canonical 16-DOF masked policy network。"""

        _ = name
        return CanonicalMaskedPpoNetwork(self.params, **kwargs)


class CanonicalMaskedPpoNetwork(network_builder.NetworkBuilder.BaseNetwork):
    r"""rl_games network facade，内部只持有正式 ``EmbodimentPolicy``。"""

    def __init__(self, params: dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = tuple(int(value) for value in kwargs.pop("input_shape"))
        self.value_size = int(kwargs.pop("value_size", 1))
        self.num_seqs = int(kwargs.pop("num_seqs", 1))
        if actions_num != CANONICAL_JOINT_COUNT:
            raise ValueError(f"canonical masked PPO requires actions_num=16, got {actions_num}")
        if input_shape != (CANONICAL_MASKED_OBS_DIM,):
            raise ValueError(
                f"canonical masked PPO expects flat obs shape {(CANONICAL_MASKED_OBS_DIM,)}, got {input_shape}"
            )
        policy_cfg = CanonicalPolicyCfg(**params.get("canonical_policy", {}))
        self.policy = EmbodimentPolicy(policy_cfg)
        evidence_bank = params.get("canonical_evidence_bank")
        if evidence_bank is not None and not isinstance(evidence_bank, CanonicalEvidenceBank):
            raise TypeError("canonical_evidence_bank must be a CanonicalEvidenceBank")
        self.evidence_bank: CanonicalEvidenceBank | None = evidence_bank
        self.geometry_encoder = ImplicitGeometryEncoder(GeometryEncoderCfg()) if evidence_bank is not None else None
        if self.geometry_encoder is not None:
            if policy_cfg.geometry_entity_width != self.geometry_encoder.config.backbone.hidden_width:
                raise ValueError("policy geometry_entity_width must match retained unified entity width")
        self.actions_num = actions_num
        self.last_active_joint_mask: torch.Tensor | None = None

    def _gather_evidence(self, asset_row: torch.Tensor, device: torch.device) -> Any:
        r"""按 runtime row gather device-resident canonical raw evidence。"""

        if self.evidence_bank is None:
            return None
        if self.evidence_bank.evidence.anchors.device != device:
            self.evidence_bank = self.evidence_bank.to(device)
        return self.evidence_bank.gather(asset_row)

    def is_rnn(self) -> bool:
        r"""canonical masked PPO 第一版不使用 recurrent state。"""

        return False

    def get_default_rnn_state(self):
        r"""非 RNN network 没有 hidden state。"""

        return

    def get_aux_loss(self):
        r"""正式 canonical PPO 没有额外 SSL/decoder loss。"""

        return

    def get_value_layer(self):
        r"""返回 PALM critic head，供 rl_games introspection。"""

        return self.policy.value_head

    def _build_policy_input(self, obs: torch.Tensor) -> EmbodimentPolicyInput:
        r"""把 flat actor obs 解析为 `[B,21]` owners 与 `[B,16]` joints。"""

        if obs.ndim != 2 or obs.shape[-1] != CANONICAL_MASKED_OBS_DIM:
            raise ValueError(f"canonical masked PPO obs must have shape [B,{CANONICAL_MASKED_OBS_DIM}]")
        batch_size = obs.shape[0]
        q_end = CANONICAL_JOINT_COUNT
        qd_end = q_end + CANONICAL_JOINT_COUNT
        previous_delta_end = qd_end + CANONICAL_JOINT_COUNT
        joint_end = previous_delta_end + 2 * CANONICAL_JOINT_COUNT
        limits = obs[:, previous_delta_end:joint_end].reshape(batch_size, CANONICAL_JOINT_COUNT, 2)
        dynamic_joint_features = torch.cat(
            (
                obs[:, :q_end].unsqueeze(-1),
                obs[:, q_end:qd_end].unsqueeze(-1),
                obs[:, qd_end:previous_delta_end].unsqueeze(-1),
                limits,
            ),
            dim=-1,
        )  # flat block order -> per-joint `[q,qd,previous_delta,q_min,q_max]`
        joint_features = torch.zeros(
            batch_size,
            CANONICAL_JOINT_COUNT,
            self.policy.config.joint_feature_dim,
            dtype=obs.dtype,
            device=obs.device,
        )
        joint_features[:, :, :CANONICAL_FLAT_JOINT_FEATURES] = dynamic_joint_features
        command_start = joint_end
        command_end = command_start + CANONICAL_COMMAND_FEATURES
        command = obs[:, command_start:command_end]
        object_pos_end = command_end + CANONICAL_OBJECT_POSITION_FEATURES
        object_pos = obs[:, command_end:object_pos_end]
        object_orientation_end = object_pos_end + CANONICAL_OBJECT_ORIENTATION_FEATURES
        object_orientation = obs[:, object_pos_end:object_orientation_end]
        contact_end = object_orientation_end + CANONICAL_TIP_CONTACT_FEATURES
        fingertip_contact = obs[:, object_orientation_end:contact_end]
        asset_row = obs[:, contact_end].round().long()
        active_mask = obs[:, -CANONICAL_JOINT_COUNT:] > 0.5

        # PALM receives task command；JOINT owners receive joint state；TIP owners receive only validity.
        owner_features = torch.zeros(
            batch_size,
            CANONICAL_OWNER_COUNT,
            self.policy.config.owner_feature_dim,
            dtype=obs.dtype,
            device=obs.device,
        )
        owner_features[:, 0, :CANONICAL_COMMAND_FEATURES] = command
        owner_features[:, 0, CANONICAL_COMMAND_FEATURES : CANONICAL_COMMAND_FEATURES + 3] = object_pos
        owner_features[:, 0, CANONICAL_COMMAND_FEATURES + 3 : CANONICAL_COMMAND_FEATURES + 3 + 6] = object_orientation
        owner_features[:, 17 : 17 + 4, 0] = fingertip_contact[:, [1, 2, 3, 0]]
        # ContactSensor layout 是 thumb/index/middle/ring；canonical TIP owner 跟随
        # manifest/PhysX finger order index/middle/ring/thumb，故边界处显式换轴。
        evidence = self._gather_evidence(asset_row, obs.device)
        if evidence is None:
            owner_mask = torch.zeros(batch_size, CANONICAL_OWNER_COUNT, dtype=torch.bool, device=obs.device)
            owner_mask[:, 0] = True  # PALM 永远有效
            owner_mask[:, 1 : 1 + CANONICAL_JOINT_COUNT] = active_mask
            tip_mask = active_mask.reshape(batch_size, 4, 4).any(dim=1)  # index/middle/ring/thumb
            owner_mask[:, 1 + CANONICAL_JOINT_COUNT :] = tip_mask
            graph = torch.zeros(
                batch_size,
                CANONICAL_OWNER_COUNT,
                CANONICAL_OWNER_COUNT,
                dtype=torch.long,
                device=obs.device,
            )
            shortest_path = parent_direction = child_direction = graph
        else:
            if evidence.joint_valid_mask is None or evidence.entity_valid_mask is None:
                raise RuntimeError("canonical evidence must expose owner/joint masks")
            if not torch.equal(active_mask, evidence.joint_valid_mask):
                raise RuntimeError("runtime active mask disagrees with canonical evidence bank")
            owner_mask = evidence.entity_valid_mask
            shortest_path = evidence.shortest_path
            parent_direction = evidence.parent_direction
            child_direction = evidence.child_direction
        return EmbodimentPolicyInput(
            owner_features=owner_features,
            joint_features=joint_features,
            owner_valid_mask=owner_mask,
            joint_valid_mask=active_mask,
            shortest_path=shortest_path,
            parent_direction=parent_direction,
            child_direction=child_direction,
            asset_row=asset_row,
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        r"""执行 rl_games `(mu, logstd, value, states)` network contract。"""

        obs = obs_dict["obs"]
        policy_input = self._build_policy_input(obs)
        if self.geometry_encoder is not None:
            unique_rows, evidence_row_index = torch.unique(
                policy_input.asset_row,
                sorted=True,
                return_inverse=True,
            )
            evidence = self._gather_evidence(unique_rows, obs.device)
            if evidence is None:
                raise RuntimeError("geometry encoder requires canonical evidence")
            geometry = self.geometry_encoder(
                policy_input.joint_features[:, :, 0],
                evidence,
                evidence_row_index=evidence_row_index,
            )
            policy_input = replace(
                policy_input,
                geometry_entities=geometry.entities,
            )
        output = self.policy(policy_input)
        self.last_active_joint_mask = output.joint_valid_mask.detach()
        logstd = torch.where(
            output.joint_valid_mask,
            output.action_log_std,
            torch.zeros_like(output.action_log_std),
        )  # inactive sigma=1，使标准 rl_games KL 不产生 ghost 项
        return output.action_mean, logstd, output.value, None


class AnyManiMaskedContinuousModel(models.BaseModel):
    r"""按 active joint mask 计算 continuous Normal 的 rl_games model。"""

    def __init__(self, network) -> None:
        super().__init__("a2c")
        self.network_builder = network

    class Network(models.BaseModelNetwork):
        r"""masked Normal wrapper，保持 rl_games model 输出字典兼容。"""

        def __init__(self, a2c_network, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def forward(self, input_dict: dict[str, torch.Tensor]):
            is_train = input_dict.get("is_train", True)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states = self.a2c_network(input_dict)
            active_mask = self.a2c_network.last_active_joint_mask
            if active_mask is None:
                raise RuntimeError("canonical masked network did not expose active joint mask")
            sigma = torch.exp(logstd)
            distribution = torch.distributions.Normal(mu, sigma, validate_args=False)
            active_float = active_mask.to(dtype=mu.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            if is_train:
                previous_actions = input_dict["prev_actions"]
                prev_neglogp = self._masked_neglogp(previous_actions, mu, sigma, logstd, active_float)
                entropy = (distribution.entropy() * active_float).sum(dim=-1) / active_count
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            selected_action = distribution.sample() * active_float  # sampling 后、送入 env 前清零 ghost action
            neglogp = self._masked_neglogp(selected_action, mu, sigma, logstd, active_float)
            return {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }

        @staticmethod
        def _masked_neglogp(
            actions: torch.Tensor,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            logstd: torch.Tensor,
            active_float: torch.Tensor,
        ) -> torch.Tensor:
            r"""对 active dimensions 求 Normal negative log-prob 的和。"""

            per_joint = (
                0.5 * ((actions - mu) / sigma).square()
                + 0.5 * torch.log(torch.as_tensor(2.0 * torch.pi, device=actions.device, dtype=actions.dtype))
                + logstd
            )
            return (per_joint * active_float).sum(dim=-1)  # PPO ratio 使用有效 joint log-prob 总和


class AnyManiMaskedPpoAgent(a2c_continuous.A2CAgent):
    r"""vanilla PPO agent 的 active-joint bounds loss 修正版。"""

    def _runtime_identity(self) -> dict[str, Any] | None:
        r"""读取 actor network 暴露的 AnyMani dataset/provider 身份。"""

        network = getattr(self.model, "a2c_network", None)  # rl_games model facade 内部的 AnyMani network
        identity = getattr(network, "anymani_identity", None)  # heterogeneous builder 返回 JSON/YAML-safe mapping
        return identity if isinstance(identity, dict) else None

    def get_full_state_weights(self) -> dict[str, Any]:
        r"""把 heterogeneous 资产/$Z$ 身份追加到 rl_games 完整 checkpoint。"""

        state = super().get_full_state_weights()  # model、optimizer、central critic、epoch/frame 与 env state
        identity = self._runtime_identity()
        if identity is not None:
            state[ANYMANI_CHECKPOINT_IDENTITY_KEY] = identity  # actor buffer 之外的 pre-load provenance gate
        return state

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        r"""在任何 actor buffer 被覆盖前核对 checkpoint 与当前 runtime 身份。"""

        runtime_identity = self._runtime_identity()
        if runtime_identity is not None:
            validate_anymani_checkpoint_identity(
                runtime_identity=runtime_identity,
                checkpoint_identity=weights.get(ANYMANI_CHECKPOINT_IDENTITY_KEY),
            )  # dataset/manifest/row/$Z$ 任一变化均 fail closed
        super().set_full_state_weights(weights, set_epoch=set_epoch)

    @staticmethod
    def masked_policy_kl(
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""复现 rl_games diagonal-Normal KL，并按每个样本 active joint 数归一化。"""

        c1 = torch.log(old_sigma / current_sigma + 1.0e-5)
        c2 = (current_sigma.square() + (old_mu - current_mu).square()) / (2.0 * (old_sigma.square() + 1.0e-5))
        per_joint = c1 + c2 - 0.5
        weights = active_mask.to(dtype=per_joint.dtype)
        return (per_joint * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def calc_gradients(self, input_dict) -> None:
        r"""沿用 upstream PPO update，只替换 optimizer/scheduler 消费的 KL reduction。"""

        super().calc_gradients(input_dict)
        a_loss, c_loss, entropy, _kl, last_lr, lr_mul, mu, sigma, b_loss = self.train_result
        active_mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("canonical PPO update did not expose the active joint mask")
        with torch.no_grad():
            kl = self.masked_policy_kl(
                mu,
                sigma,
                input_dict["mu"],
                input_dict["sigma"],
                active_mask,
            ).mean()
        self.train_result = (a_loss, c_loss, entropy, kl, last_lr, lr_mul, mu, sigma, b_loss)

    def bound_loss(self, mu: torch.Tensor) -> torch.Tensor:
        r"""按 active joint 数均值计算 bounds loss，ghost mean=0 不计入。"""

        if self.bounds_loss_coef is None:
            return torch.zeros(mu.shape[0], device=mu.device)
        mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.shape != mu.shape:
            inherited = super().bound_loss(mu)
            return inherited if isinstance(inherited, torch.Tensor) else torch.zeros(mu.shape[0], device=mu.device)
        weights = mask.to(dtype=mu.dtype)
        soft_bound = 1.1
        high = torch.clamp_min(mu - soft_bound, 0.0).square()
        low = torch.clamp_max(mu + soft_bound, 0.0).square()
        return ((high + low) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def reg_loss(self, mu: torch.Tensor) -> torch.Tensor:
        r"""按 active joint 数均值计算 regularisation bounds loss。"""

        if self.bounds_loss_coef is None:
            return torch.zeros(mu.shape[0], device=mu.device)
        mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.shape != mu.shape:
            inherited = super().reg_loss(mu)
            return inherited if isinstance(inherited, torch.Tensor) else torch.zeros(mu.shape[0], device=mu.device)
        weights = mask.to(dtype=mu.dtype)
        return (mu.square() * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class AnyManiMaskedPpoPlayer(players.PpoPlayerContinuous):
    r"""canonical masked PPO 的 player facade；sampling mask 由 custom model 执行。"""

    def restore(self, fn: str) -> None:
        r"""在回放 model buffer 被覆盖前验证 heterogeneous dataset/provider identity。"""

        checkpoint = torch_ext.load_checkpoint(fn)
        network = getattr(self.model, "a2c_network", None)
        runtime_identity = getattr(network, "anymani_identity", None)
        if isinstance(runtime_identity, dict):
            validate_anymani_checkpoint_identity(
                runtime_identity=runtime_identity,
                checkpoint_identity=checkpoint.get(ANYMANI_CHECKPOINT_IDENTITY_KEY),
            )
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        env_state = checkpoint.get("env_state")
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)


class AnyManiMaskedRunner(Runner):
    r"""在本地 Runner factory 中注册 custom algorithm/model/player。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.algo_factory.register_builder(
            "anymani_masked_ppo",
            lambda **factory_kwargs: AnyManiMaskedPpoAgent(**factory_kwargs),
        )
        self.player_factory.register_builder(
            "anymani_masked_ppo",
            lambda **factory_kwargs: AnyManiMaskedPpoPlayer(**factory_kwargs),
        )


def register_anymani_masked_ppo() -> None:
    r"""注册 model/network 名称；重复调用保持幂等覆盖到同一 class。"""

    from anymani.distill.rl.heterogeneous_masked_ppo import register_heterogeneous_masked_network

    model_builder.register_network("anymani_canonical_masked", CanonicalMaskedPpoBuilder)
    register_heterogeneous_masked_network()  # 独立 69D N000-frame + frozen-Z builder
    model_builder.register_model("anymani_masked_continuous", AnyManiMaskedContinuousModel)


__all__ = [
    "ANYMANI_CHECKPOINT_IDENTITY_KEY",
    "ANYMANI_MASKED_PPO_ALGO_KEY",
    "AnyManiMaskedPpoAgent",
    "AnyManiMaskedPpoPlayer",
    "AnyManiMaskedRunner",
    "CanonicalMaskedPpoBuilder",
    "CanonicalMaskedPpoNetwork",
    "register_anymani_masked_ppo",
    "validate_anymani_checkpoint_identity",
]


ANYMANI_MASKED_PPO_ALGO_KEY = "anymani_masked_ppo"
