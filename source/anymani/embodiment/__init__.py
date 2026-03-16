"""Embodiment asset generation toolkit.

This package implements a minimal yet extensible pipeline for:
1) Parsing URDF into a canonical Hand Intermediate Representation (HIR)
2) Mutating HIR to generate embodiment variants
3) Serializing HIR back to URDF with metadata and validation
"""

from .schema.hir_v01 import HandHIR

__all__ = ["HandHIR"]
