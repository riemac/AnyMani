r"""N040 proper-SE(3)-invariant density + Gamma concrete method。"""

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
    r"""继承稳定 source/batch/objective 生命周期，只替换 encoder/model 与 SE3 coordinate augmentation。"""

    def __init__(self, config: SE3DensityMaterialJacobianMethodCfg) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config
        self.model: SE3DensityMaterialJacobianSSLModel | None = None

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> SE3DensityMaterialJacobianSSLModel:
        r"""构造 N040 invariant model，并保持 compiled lifecycle。"""

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
        r"""返回已初始化 N040 model。"""

        if self.model is None:
            raise RuntimeError("N040 model has not been initialized")
        return self.model

    def feature_spec(self) -> FeatureSpec:
        r"""声明 proper-SE3 Z invariance 与 joint-sign Gamma equivariance。"""

        return FeatureSpec(
            entity_width=self.config.model.encoder.backbone.hidden_width,
            frame_contract="proper-SE(3)-invariant hand-coordinate representation; reflection-sensitive chirality",
            coordinate_rewrite_contract="density invariant; material_jacobian selected-column sign-equivariant",
        )

    def _forward_with_prediction(self, batch: PaddedDensityGammaBatch, *, mode: str):
        r"""训练前执行 q-block proper-SE3 rewrite，再复用 density/Gamma FairGrad forward。"""

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
