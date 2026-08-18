r"""训练资产固定 calibration microbatches 的一次性梯度量级校准。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml

from anymani.distill.models.geometry_ssl import GeometrySSLModel
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjective,
    GeometryFieldObjectiveCfg,
)
from anymani.distill.representations.geometry import PaddedOnlineGeometryBatch


def calibrate_geometry_ssl_weights(
    model: GeometrySSLModel,
    objective_factory: Callable[[GeometryFieldObjectiveCfg], GeometryFieldObjective],
    batches: tuple[PaddedOnlineGeometryBatch, ...],
    forward_terms: Callable[[GeometrySSLModel, GeometryFieldObjective, PaddedOnlineGeometryBatch], Any],
    *,
    output_path: Path,
    min_weight: float = 1.0e-2,
    max_weight: float = 1.0e3,
) -> GeometryFieldObjectiveCfg:
    r"""在固定 generated train batches 上测量六项共享 encoder gradient，并冻结权重。

    每项独立建立 objective、清零梯度并反传；只读取模型 encoder 参数的梯度范数。参考项为
    density，权重为 ``median(g_density)/median(g_term)`` 后裁剪到声明区间。validation 与全部
    evaluation suites 绝不能传入 `batches`；本函数不更新 model 参数，也不使用动态重标定。
    """

    if not batches:
        raise ValueError("geometry SSL calibration requires at least one fixed train batch")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError("calibration weight bounds are invalid")
    names = ("density", "kappa", "derived_field", "sobolev", "chain", "paired")
    measurements: dict[str, list[float]] = {name: [] for name in names}
    encoder_parameters = tuple(parameter for parameter in model.encoder.parameters() if parameter.requires_grad)
    for batch in batches:
        for name in names:
            model.zero_grad(set_to_none=True)
            objective = objective_factory(
                GeometryFieldObjectiveCfg(**{term: 1.0 if term == name else 0.0 for term in names})
            )
            terms = forward_terms(model, objective, batch)
            value = getattr(terms, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(f"calibration term {name!r} must be one scalar tensor")
            value.backward()
            squared_norm = torch.zeros((), device=next(model.parameters()).device)
            for parameter in encoder_parameters:
                if parameter.grad is not None:
                    squared_norm = squared_norm + parameter.grad.detach().square().sum()
            measurements[name].append(float(torch.sqrt(squared_norm)))
    medians = {name: float(median(values)) for name, values in measurements.items()}
    reference = medians["density"]
    if reference <= 0.0:
        raise FloatingPointError("density calibration gradient median must be positive")
    weights = {
        name: min(max(reference / value, min_weight), max_weight) if value > 0.0 else max_weight
        for name, value in medians.items()
    }
    resolved_weights = GeometryFieldObjectiveCfg(**weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            {
                "source": "generated_train_fixed_calibration_batches",
                "batch_count": len(batches),
                "gradient_norms": measurements,
                "median_gradient_norms": medians,
                "reference": "density",
                "weights": asdict(resolved_weights),
                "clip": {"min": min_weight, "max": max_weight},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return resolved_weights


__all__ = ["calibrate_geometry_ssl_weights"]
