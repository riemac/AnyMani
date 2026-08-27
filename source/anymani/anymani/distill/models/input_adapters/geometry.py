r"""Geometry input-adapter compatibility surface。

静态 evidence/mask/routing/padding 由 ``evidence.py`` 拥有；可学习 SO(2)-aware frontend 与 retained
encoder 由 ``encoder.py`` 拥有。本模块只保留历史 import path，避免修改 PPO、method 与测试的公开
符号，同时使依赖方向清晰：evidence 不依赖 encoder，encoder 只消费 evidence contract。
"""

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
