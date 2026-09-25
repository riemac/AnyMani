"""Stable geometry adapter exports used by the shared method base and retained encoder loaders."""


from .encoder import (
    GeometryEncoderCfg,
    GeometryLatents,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
    SO2AnchorRelationEncoder,
)
from .evidence import (
    GeometryPaddingCfg,
    StaticGeometryEvidence,
    build_static_geometry_evidence,
    canonicalize_static_geometry_evidence,
    pad_static_geometry_evidence,
    stack_static_geometry_evidence,
)

__all__ = [
    "GeometryEncoderCfg",
    "GeometryLatents",
    "GeometryPaddingCfg",
    "ImplicitGeometryEncoder",
    "SO2AnchorFrontendCfg",
    "SO2AnchorRelationEncoder",
    "StaticGeometryEvidence",
    "build_static_geometry_evidence",
    "canonicalize_static_geometry_evidence",
    "pad_static_geometry_evidence",
    "stack_static_geometry_evidence",
]
