r"""Density prediction 到物理 distance/tolerance/PGS 指标的合同。"""

from __future__ import annotations

import torch
from anymani.distill.methods.density_material_jacobian.evaluation_metrics import DensityPhysicalMetricAccumulator
from anymani.distill.representations.fields.density import gaussian_density_from_distance


def test_exact_multiband_density_yields_zero_distance_error_and_full_success() -> None:
    r"""精确 Gaussian field 反解后应获得零 MAE、完整 tolerance/PGS 与 4 mm contact F1。"""

    distance = torch.tensor(
        [[[[0.001, 0.003, 0.006, 0.012]]]],
        dtype=torch.float64,
    ).reshape(1, 1, 4)  # `[B,G,N_Q]`，m
    bandwidths = torch.tensor(((0.004, 0.016, 0.064),), dtype=torch.float64)
    density = gaussian_density_from_distance(distance, bandwidths)
    valid = torch.ones_like(distance, dtype=torch.bool)
    stratum = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)  # 全部属于 PGS shell/adjacent
    role = torch.tensor(((1,),), dtype=torch.long)
    accumulator = DensityPhysicalMetricAccumulator()
    accumulator.update(density, distance, bandwidths, valid, stratum, role)
    report = accumulator.finalize()

    assert report["all"]["mae_m"] < 1.0e-12
    assert report["all"]["query_success"]["1mm"] == 1.0
    assert report["posed_geometry_success"]["PGS@4mm,80%"] == 1.0
    assert report["posed_geometry_success"]["PGS@4mm,90%"] == 1.0
    assert report["contact_4mm"]["f1"] == 1.0
    assert report["headline"]["metric"] == "PGS@4mm,80%"
