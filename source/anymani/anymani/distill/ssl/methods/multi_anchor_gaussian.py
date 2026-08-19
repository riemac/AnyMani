r"""多锚点 Gaussian 零阶/一阶特权预训练方法的显式计算图。

Representation 拥有 physical source、query/sigma/edge realization 和 privileged targets；model 拥有
retained encoder 与 disposable readers；objective terms 各自拥有一个数学约束。本模块只串联这三层，
并缓存一次 minibatch 内共享的派生场、density q-JVP 和 paired second forward。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from omegaconf import MISSING

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.objectives.representations.field_reconstruction import selected_density_coordinate_derivative
from anymani.distill.objectives.representations.gauge_consistency import (
    deterministic_partial_joint_sign,
    joint_sign_paired_loss_additive_components,
    rewrite_joint_sign_coordinates,
)
from anymani.distill.objectives.representations.geometry_terms import MultiAnchorObjectiveContext
from anymani.distill.representations.geometry import (
    GeometryRepresentation,
    GeometryRepresentationCfg,
    PaddedOnlineGeometryBatch,
)
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.ssl.contracts import FeatureSpec, ObjectiveTermResult, build_runtime


@dataclass(frozen=True)
class ObjectiveCalibrationCfg:
    r"""固定 train minibatches 上的一次性 shared-encoder 梯度量级校准。"""

    minibatches: int = 8  # 不推进正式 train sampler cursor
    min_weight: float = 1.0e-2  # 校准后无量纲权重下界
    max_weight: float = 1.0e3  # 校准后无量纲权重上界
    reference_term: str = "density"  # 当前以零阶密度梯度 median 为参考

    def __post_init__(self) -> None:
        r"""验证校准预算、裁剪区间和 reference 名称。"""

        if self.minibatches < 1 or self.min_weight <= 0.0 or self.max_weight < self.min_weight:
            raise ValueError("objective calibration minibatches or weight bounds are invalid")
        if not self.reference_term:
            raise ValueError("objective calibration requires a reference term")


@dataclass(frozen=True)
class MultiAnchorMethodStep:
    r"""一次 method forward 的 retained/prediction 输出与独立 objective 结果。"""

    prediction: GeometrySSLForward  # latents、query features、density 与 kappa
    objectives: dict[str, ObjectiveTermResult]  # 与 cfg mapping 同顺序的具名 terms


class _MultiAnchorContext(MultiAnchorObjectiveContext):
    r"""一次 minibatch 内按需物化并缓存共享训练图节点。"""

    def __init__(
        self,
        *,
        model: GeometrySSLModel,
        q: torch.Tensor,
        prediction: GeometrySSLForward,
        batch: PaddedOnlineGeometryBatch,
        pair_step: int,
    ) -> None:
        r"""保存基础 forward；复杂派生节点只在 objective 首次访问时计算。"""

        self.model = model
        self.q = q
        self.prediction = prediction
        self.batch = batch
        self.pair_step = int(pair_step)
        self._derived_field: torch.Tensor | None = None
        self._auto_field: torch.Tensor | None = None
        self._paired_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def density_prediction(self) -> torch.Tensor:
        r"""返回 density reader 预测。"""

        return self.prediction.density

    @property
    def density_target(self) -> torch.Tensor:
        r"""返回在线 privileged density target。"""

        return self.batch.field_targets.density

    @property
    def density_valid_mask(self) -> torch.Tensor:
        r"""返回 owner/query 有效 mask。"""

        return self.batch.field_targets.valid_mask

    @property
    def kappa_prediction(self) -> torch.Tensor:
        r"""返回 sampled edges 上的 distance sensitivity 预测。"""

        return self.prediction.kappa

    @property
    def kappa_target(self) -> torch.Tensor:
        r"""返回解析 distance sensitivity target。"""

        return self.batch.sensitivity_targets.kappa

    @property
    def edge_valid_mask(self) -> torch.Tensor:
        r"""返回最近点平滑性与结构边共同形成的一阶 mask。"""

        return self.batch.sensitivity_targets.valid_mask

    @property
    def field_sensitivity_target(self) -> torch.Tensor:
        r"""返回 sampled edges、全部实际 sigma 上的解析场灵敏度。"""

        return self.batch.sensitivity_targets.field_sensitivity

    @staticmethod
    def _select_owner_queries(
        values: torch.Tensor,
        owner_index: torch.Tensor,
        query_index: torch.Tensor,
    ) -> torch.Tensor:
        r"""从 `[B,G,N_Q,...]` 读取共享或逐样本 sampled edge selectors。"""

        if owner_index.ndim == 1:
            return values[:, owner_index, query_index]
        batch_index = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        return values[batch_index, owner_index, query_index]

    @property
    def derived_field_sensitivity(self) -> torch.Tensor:
        r"""只计算一次 $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$。"""

        if self._derived_field is None:
            targets = self.batch.sensitivity_targets
            field = self.batch.field_targets
            selected_density = self._select_owner_queries(
                self.prediction.density,
                targets.owner_index,
                targets.query_index,
            )
            selected_distance = self._select_owner_queries(
                field.distance,
                targets.owner_index,
                targets.query_index,
            )
            inverse_sigma_squared = field.bandwidths.square().reciprocal()
            inverse_sigma_squared = (
                inverse_sigma_squared.view(1, 1, -1)
                if inverse_sigma_squared.ndim == 1
                else inverse_sigma_squared.unsqueeze(1)
            )
            self._derived_field = (
                -selected_distance.unsqueeze(-1)
                * inverse_sigma_squared
                * selected_density
                * self.prediction.kappa.unsqueeze(-1)
            )
        return self._derived_field

    @property
    def auto_field_sensitivity(self) -> torch.Tensor:
        r"""只计算一次固定 query/sigma 的 density q-JVP，并保留二阶训练图。"""

        if self._auto_field is None:
            targets = self.batch.sensitivity_targets
            self._auto_field = selected_density_coordinate_derivative(
                self.prediction.density,
                self.q,
                targets.owner_index,
                targets.query_index,
                targets.joint_index,
                create_graph=True,
            )
        return self._auto_field

    @property
    def paired_additive_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""只执行一次 joint-sign second encoder forward，并分别保留零/一阶 MSE 分母。"""

        if self._paired_components is None:
            joint_valid = self.batch.evidence.joint_valid_mask
            if joint_valid is None:
                joint_valid = torch.ones_like(self.q, dtype=torch.bool)
            if joint_valid.ndim == 1:
                joint_valid = joint_valid.unsqueeze(0).expand(self.q.shape[0], -1)
            joint_sign = deterministic_partial_joint_sign(
                joint_valid,
                step=self.pair_step,
                dtype=self.q.dtype,
            )
            rewritten_q, rewritten_evidence, joint_sign = rewrite_joint_sign_coordinates(
                self.q,
                self.batch.evidence,
                joint_sign=joint_sign,
            )
            rewritten_latents = self.model.encoder(rewritten_q, rewritten_evidence)
            self._paired_components = joint_sign_paired_loss_additive_components(
                self.prediction.latents,
                rewritten_latents,
                joint_sign=joint_sign,
                entity_valid_mask=self.batch.evidence.entity_valid_mask,
                joint_valid_mask=joint_valid,
            )
        return self._paired_components


class MultiAnchorGaussianMethod:
    r"""显式装配 GeometryRepresentation、GeometrySSLModel 与独立 objective terms。"""

    def __init__(self, config: MultiAnchorGaussianMethodCfg) -> None:
        r"""保存配置并构造无 IO 的 representation/objective runtimes。"""

        self.config = config
        self._validate_composed()
        self.representation = GeometryRepresentation(config.representation)
        self.objectives = {name: build_runtime(term_cfg) for name, term_cfg in config.objectives.items()}
        self.objective_weights = {
            name: float(term_cfg.weight) for name, term_cfg in config.objectives.items()
        }  # 一次性 calibration 只更新 runtime evidence，不改写 frozen cfg
        self.model: GeometrySSLModel | None = None

    def set_objective_weights(self, weights: dict[str, float]) -> None:
        r"""冻结一次性 calibration 结果，并拒绝 term 集或权重域漂移。"""

        if set(weights) != set(self.objectives):
            raise ValueError("calibrated objective names do not match composed method terms")
        if any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in weights.values()):
            raise ValueError("calibrated objective weights must be finite and non-negative")
        self.objective_weights = {name: float(weights[name]) for name in self.objectives}

    def _validate_composed(self) -> None:
        r"""在物化资产前验证 representation/model/objective 类型与当前方法能力。"""

        if not isinstance(self.config.representation, GeometryRepresentationCfg):
            raise TypeError("multi-anchor method requires GeometryRepresentationCfg")
        if not isinstance(self.config.model, GeometrySSLModelCfg):
            raise TypeError("multi-anchor method requires GeometrySSLModelCfg")
        if not self.config.objectives:
            raise ValueError("multi-anchor method requires at least one objective term")
        if self.config.calibration.reference_term not in self.config.objectives:
            raise ValueError("objective calibration reference term is not enabled in the method")
        required = {"density", "kappa", "derived_field", "sobolev", "chain", "paired"}
        unknown = set(self.config.objectives) - required
        if unknown:
            raise ValueError(f"multi-anchor method received unknown objective names: {sorted(unknown)}")
        mismatched = tuple(
            name
            for name, objective in self.config.objectives.items()
            if getattr(objective, "name", name) != name and getattr(type(objective), "runtime_type", None) is None
        )
        if mismatched:
            raise TypeError(f"objective configs lack valid runtime bindings: {mismatched}")

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> GeometrySSLModel:
        r"""在 Trainer 已冻结 device/dtype 后一次性构造 learned model。"""

        if self.model is not None:
            raise RuntimeError("multi-anchor method model is already initialized")
        self.model = GeometrySSLModel(self.config.model).to(device=device, dtype=dtype)
        return self.model

    def require_model(self) -> GeometrySSLModel:
        r"""返回已初始化模型；setup 顺序错误时明确失败。"""

        if self.model is None:
            raise RuntimeError("multi-anchor method model has not been initialized")
        return self.model

    def materialize_sources(self, assets: tuple[HandContainer, ...]) -> tuple[GeometrySource, ...]:
        r"""按 representation source 配置物化 CPU physical sources。"""

        return tuple(self.representation.materialize_source(asset) for asset in assets)

    def forward_objectives(self, batch: PaddedOnlineGeometryBatch, *, pair_step: int) -> MultiAnchorMethodStep:
        r"""完成一次基础模型前向，并让六项 terms 共享惰性计算上下文。"""

        model = self.require_model()
        q = batch.q.detach().requires_grad_(True)  # privileged target 停止梯度；保留模型对物理 q 的 JVP
        prediction = model(
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            owner_index=batch.sensitivity_targets.owner_index,
            query_index=batch.sensitivity_targets.query_index,
            joint_index=batch.sensitivity_targets.joint_index,
        )
        context = _MultiAnchorContext(model=model, q=q, prediction=prediction, batch=batch, pair_step=pair_step)
        results = {name: objective.evaluate(context) for name, objective in self.objectives.items()}
        return MultiAnchorMethodStep(prediction=prediction, objectives=results)

    def feature_spec(self) -> FeatureSpec:
        r"""返回下游消费的零阶实体序列与逐 JOINT 一阶序列合同。"""

        heads = self.config.model.encoder.heads
        return FeatureSpec(
            zero_order_width=heads.zero_order_width,
            first_order_width=heads.first_order_width,
        )

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回只含 retained encoder namespace 的 standalone transfer state。"""

        return self.require_model().retained_state_dict()


@dataclass(frozen=True)
class MultiAnchorGaussianMethodCfg:
    r"""多锚点 Gaussian method 的 representation/model/objective 组合配置。"""

    runtime_type: ClassVar[type[MultiAnchorGaussianMethod]] = MultiAnchorGaussianMethod
    representation: Any = MISSING  # concrete GeometryRepresentationCfg 由 Hydra group 注入
    model: Any = MISSING  # concrete GeometrySSLModelCfg 由 Hydra group 注入
    objectives: dict[str, Any] = field(default_factory=dict)  # 具名 concrete term cfgs
    calibration: ObjectiveCalibrationCfg = field(default_factory=ObjectiveCalibrationCfg)


__all__ = [
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorMethodStep",
    "ObjectiveCalibrationCfg",
]
