'Runtime contracts for source config.'

from __future__ import annotations

from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg

N040_PPO_SOURCE_CFG = GeometrySourceCfg(
    home_points_per_owner=64,  # canonical PALM/JOINT/TIP axis
    home_surface_oversample_factor=8,
    static_sampling_seed=0,
    anchors=AnchorBankCfg(
        bank_size=8,
        anchors_per_finger=10,
        radius_m=0.05,  # units m
        radial_decay_scale_m=0.025,  # units m
        surface_fraction=0.5,
    ),
)
'Definition for N040 PPO source cfg.'

__all__ = ["N040_PPO_SOURCE_CFG"]
