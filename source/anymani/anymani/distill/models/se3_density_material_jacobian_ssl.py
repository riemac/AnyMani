r"""Proper-SE(3)-invariant unified encoder + density/Gamma readers 模型组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .decoders.representations.implicit_field import ScalarSigmaFiLMDensityDecoderCfg
from .decoders.representations.material_point_jacobian import AnchorRelationalJacobianDecoderCfg
from .density_material_jacobian_ssl import (
    DensityMaterialJacobianModelCfg,
    DensityMaterialJacobianSSLModel,
)
from .input_adapters.encoder import GeometryEncoderCfg, SO2AnchorFrontendCfg
from .input_adapters.se3_invariant_encoder import (
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)


@dataclass(frozen=True)
class SE3DensityMaterialJacobianModelCfg:
    r"""N040 invariant encoder 与两个 disposable readers 的强类型配置。"""

    encoder: SE3InvariantGeometryEncoderCfg = field(default_factory=SE3InvariantGeometryEncoderCfg)
    density: ScalarSigmaFiLMDensityDecoderCfg = field(default_factory=ScalarSigmaFiLMDensityDecoderCfg)
    material_jacobian: AnchorRelationalJacobianDecoderCfg = field(
        default_factory=AnchorRelationalJacobianDecoderCfg
    )

    def __post_init__(self) -> None:
        r"""验证 reader width 与 invariant encoder 输出一致。"""

        if self.material_jacobian.latent_width != self.encoder.backbone.hidden_width:
            raise ValueError("Gamma reader latent_width must match SE3 encoder hidden_width")
        if self.material_jacobian.relation_width != self.encoder.frontend.relation_width:
            raise ValueError("Gamma reader relation_width must match SE3 frontend relation_width")


class SE3DensityMaterialJacobianSSLModel(DensityMaterialJacobianSSLModel):
    r"""以独立 SE3InvariantGeometryEncoder 替换 legacy frontend 的联合模型。"""

    def __init__(
        self,
        config: SE3DensityMaterialJacobianModelCfg = SE3DensityMaterialJacobianModelCfg(),
    ) -> None:
        r"""保持 reader/state namespaces 与 N031 兼容，只替换 encoder class 和 config identity。"""

        legacy_frontend = SO2AnchorFrontendCfg(
            relation_width=config.encoder.frontend.relation_width,
            home_width=config.encoder.frontend.home_width,
            screw_width=config.encoder.frontend.screw_width,
            role_width=config.encoder.frontend.role_width,
            length_scale_m=config.encoder.frontend.length_scale_m,
        )
        legacy_config = DensityMaterialJacobianModelCfg(
            encoder=GeometryEncoderCfg(frontend=legacy_frontend, backbone=config.encoder.backbone),
            density=config.density,
            material_jacobian=config.material_jacobian,
        )
        super().__init__(legacy_config)
        self.se3_config = config
        self.encoder = SE3InvariantGeometryEncoder(config.encoder)


__all__ = ["SE3DensityMaterialJacobianModelCfg", "SE3DensityMaterialJacobianSSLModel"]
