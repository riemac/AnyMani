r"""RL消费N040 retained encoder时固定的static geometry evidence realization。"""

from __future__ import annotations

from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg

N040_PPO_SOURCE_CFG = GeometrySourceCfg(
    home_points_per_owner=64,  # 每个PALM/JOINT/TIP owner的静态surface evidence点数
    home_surface_oversample_factor=8,  # mesh采样后确定性下采样的oversampling倍率
    static_sampling_seed=0,  # checkpoint identity固定的静态evidence seed
    anchors=AnchorBankCfg(
        bank_size=8,  # SSL训练使用的anchor realizations总数
        anchors_per_finger=10,  # 每指条件anchor数量
        radius_m=0.05,  # anchor局部邻域半径，单位m
        radial_decay_scale_m=0.025,  # radial weighting尺度，单位m
        surface_fraction=0.5,  # surface/workspace anchors各占一半
    ),
)
r"""N040 snapshot的exact source realization；PPO固定消费anchor bank $A^{(0)}$。"""

__all__ = ["N040_PPO_SOURCE_CFG"]
