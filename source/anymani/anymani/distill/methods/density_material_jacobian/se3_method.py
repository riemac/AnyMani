"""N040 training method using the invariant encoder and density plus material Jacobian supervision."""


from __future__ import annotations

import torch
from torch._functorch import config as functorch_config  # pyright: ignore[reportPrivateImportUsage]

from anymani.distill.methods.contracts import FeatureSpec
from anymani.distill.models.se3_density_material_jacobian_ssl import SE3DensityMaterialJacobianSSLModel

from .batch import PaddedDensityGammaBatch
from .method import DensityMaterialJacobianMethod
from .se3_augmentation import maybe_rewrite_density_gamma_batch_se3
from .se3_config import SE3DensityMaterialJacobianMethodCfg


class SE3DensityMaterialJacobianMethod(DensityMaterialJacobianMethod):


    def __init__(self, config: SE3DensityMaterialJacobianMethodCfg) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config
        self.model: SE3DensityMaterialJacobianSSLModel | None = None

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> SE3DensityMaterialJacobianSSLModel:


        if self.model is not None:
            raise RuntimeError("N040 model is already initialized")
        self.model = SE3DensityMaterialJacobianSSLModel(self.config.model).to(device=device, dtype=dtype)
        if self.execution_policy is not None and bool(self.execution_policy.compile_enabled):
            functorch_config.donated_buffer = False
            self._compiled_forward = torch.compile(
                self.model,
                mode=str(self.execution_policy.compile_mode),
                fullgraph=True,
            )
        return self.model

    def require_model(self) -> SE3DensityMaterialJacobianSSLModel:


        if self.model is None:
            raise RuntimeError("N040 model has not been initialized")
        return self.model

    def feature_spec(self) -> FeatureSpec:


        return FeatureSpec(
            entity_width=self.config.model.encoder.backbone.hidden_width,
            frame_contract="proper-SE(3)-invariant hand-coordinate representation; reflection-sensitive chirality",
            coordinate_rewrite_contract="density invariant; material_jacobian selected-column sign-equivariant",
        )

    def _forward_with_prediction(self, batch: PaddedDensityGammaBatch, *, mode: str):


        if mode == "train":
            seed = int(batch.q_index[0]) + int(batch.anchor_index[0]) * 1_000_003
            batch = maybe_rewrite_density_gamma_batch_se3(
                batch,
                config=self.config.se3_coordinate_rewrite,
                seed=seed,
            )
        return super()._forward_with_prediction(batch, mode=mode)


SE3DensityMaterialJacobianMethodCfg.runtime_type = SE3DensityMaterialJacobianMethod


__all__ = ["SE3DensityMaterialJacobianMethod"]
