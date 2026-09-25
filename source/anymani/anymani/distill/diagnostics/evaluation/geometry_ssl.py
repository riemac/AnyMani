'Run fixed query-only and latent-shuffle diagnostics on recorded geometry evidence.'

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import PaddedOnlineGeometryBatch
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.models.input_adapters.geometry import (
    GeometryLatents,
    StaticGeometryEvidence,
)

GeometrySSLAblation = Literal[
    "query_only",
    "latent_shuffle",
    "joint_token_shuffle",
]

StratifiedComponents = dict[
    str,
    dict[str, dict[str, tuple[tuple[float, ...], tuple[float, ...]]]],
]  # metric -> axis -> bin -> `(per-sample numerator, per-sample denominator)`


def geometry_ssl_ablation_forward(
    model: GeometrySSLModel,
    q: torch.Tensor,  # shapes [B,N_J]; units rad
    evidence: StaticGeometryEvidence,
    query_points_h: torch.Tensor,  # shapes [B,G,N_Q,3]; units m; semantic hand frame
    bandwidths: torch.Tensor,
    *,
    owner_index: torch.Tensor,  # shapes [E], [B,E]
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
    ablation: GeometrySSLAblation,
    batch_permutation: torch.Tensor | None = None,  # shapes [B]
    evidence_row_index: torch.Tensor | None = None,  # `[B]` q row -> unique static-evidence row
    joint_coordinate_sign: torch.Tensor | None = None,  # shapes [B,N_J]
) -> GeometrySSLForward:
    'Handle geometry SSL ablation forward; shapes [B,N_J], [B,G,N_Q,3], [B]; units rad, m.'

    latents = model.encoder(q, evidence, evidence_row_index, joint_coordinate_sign)  # shapes [B,G,D]
    query_features = model.encoder.encode_points(
        query_points_h.detach(), evidence, evidence_row_index
    )  # shapes [B,G,N_Q,D_q]
    entity_valid = evidence.entity_valid_mask  # shapes [B,G]
    if entity_valid is not None:
        if evidence_row_index is not None and entity_valid.ndim == 2:
            entity_valid = entity_valid[evidence_row_index]
        if entity_valid.ndim == 1:
            entity_valid = entity_valid.unsqueeze(0).expand(q.shape[0], -1)  # `[B,G]`
        query_features = query_features * entity_valid.unsqueeze(-1).unsqueeze(-1)

    if ablation == "query_only":
        ablated = GeometryLatents(torch.zeros_like(latents.entities))
    elif ablation == "latent_shuffle":
        if batch_permutation is None or batch_permutation.shape != (q.shape[0],):  # shapes [B]
            raise ValueError("latent_shuffle requires batch_permutation with shape [B]")
        expected = torch.arange(q.shape[0], device=batch_permutation.device)
        if not torch.equal(torch.sort(batch_permutation).values, expected):
            raise ValueError("batch_permutation must be a bijection of [0,B)")
        ablated = GeometryLatents(latents.entities.index_select(0, batch_permutation))
    elif ablation == "joint_token_shuffle":
        joint_valid = evidence.joint_valid_mask
        if joint_valid is None:
            joint_valid = torch.ones(q.shape, device=q.device, dtype=torch.bool)
        if evidence_row_index is not None and joint_valid.ndim == 2:
            joint_valid = joint_valid[evidence_row_index]
        if joint_valid.ndim == 1:
            joint_valid = joint_valid.unsqueeze(0).expand(q.shape[0], -1)
        joint_entities = evidence.joint_entity_index
        if evidence_row_index is not None and joint_entities.ndim == 2:
            joint_entities = joint_entities[evidence_row_index]
        if joint_entities.ndim == 1:
            joint_entities = joint_entities.unsqueeze(0).expand(q.shape[0], -1)
        shuffled = latents.entities.clone()
        for batch_index in range(q.shape[0]):
            valid_joint_slots = torch.where(joint_valid[batch_index])[0]
            if len(valid_joint_slots) < 2:
                raise ValueError("joint_token_shuffle requires at least two valid JOINTs per sample")
            entity_slots = joint_entities[batch_index, valid_joint_slots]
            shuffled[batch_index, entity_slots] = latents.entities[batch_index, entity_slots].roll(1, dims=0)
        ablated = GeometryLatents(shuffled)
    else:
        raise ValueError(f"unknown geometry SSL ablation={ablation!r}")
    return model.decode_latents(
        ablated,
        query_features,
        bandwidths=bandwidths,
        entity_valid_mask=entity_valid,
        joint_entity_index=(
            evidence.joint_entity_index[evidence_row_index]
            if evidence_row_index is not None and evidence.joint_entity_index.ndim == 2
            else evidence.joint_entity_index
        ),
        owner_index=owner_index,
        query_index=query_index,
        joint_index=joint_index,
    )


def same_asset_q_permutation(asset_ids: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
    'Handle same asset q permutation.'

    permutation = torch.arange(len(asset_ids), device=device)
    for asset_id in dict.fromkeys(asset_ids):
        indices = [index for index, candidate in enumerate(asset_ids) if candidate == asset_id]
        if len(indices) < 2:
            raise ValueError("same-asset q shuffle requires at least two q samples per asset")
        source = torch.tensor(indices, device=device)
        permutation[source] = source.roll(1)
    return permutation


def cross_asset_permutation(asset_ids: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
    'Handle cross asset permutation.'

    batch_size = len(asset_ids)
    for shift in range(1, batch_size):
        candidate = tuple((index + shift) % batch_size for index in range(batch_size))
        if all(asset_ids[source] != asset_ids[target] for target, source in enumerate(candidate)):
            return torch.tensor(candidate, device=device)
    raise ValueError("cross-asset shuffle requires a batch permutation with different source assets")


def geometry_ssl_reconstruction_metrics(
    prediction: GeometrySSLForward,
    batch: PaddedOnlineGeometryBatch,
) -> dict[str, float]:
    'Handle geometry SSL reconstruction metrics.'

    field_mask = batch.field_targets.valid_mask.unsqueeze(-1).expand_as(prediction.density)
    edge_mask = batch.sensitivity_targets.valid_mask
    edge_band_mask = edge_mask.unsqueeze(-1).expand_as(batch.sensitivity_targets.field_sensitivity)
    density_error = prediction.density - batch.field_targets.density
    kappa_error = prediction.kappa - batch.sensitivity_targets.kappa
    owner_index = batch.sensitivity_targets.owner_index
    query_index = batch.sensitivity_targets.query_index
    batch_index = torch.arange(prediction.density.shape[0], device=prediction.density.device).unsqueeze(1)
    selected_density = prediction.density[batch_index, owner_index, query_index]
    selected_distance = batch.field_targets.distance[batch_index, owner_index, query_index]
    inverse_sigma_squared = _edge_inverse_sigma_squared(batch.field_targets.bandwidths)
    derived = (
        -selected_distance.unsqueeze(-1)
        * inverse_sigma_squared
        * selected_density
        * prediction.kappa.unsqueeze(-1)
    )
    derived_error = derived - batch.sensitivity_targets.field_sensitivity
    return {
        "density": float(density_error.square()[field_mask].mean().detach()),
        "kappa": float(kappa_error.square()[edge_mask].mean().detach()),
        "derived_field": float(derived_error.square()[edge_band_mask].mean().detach()),
    }


def geometry_ssl_reconstruction_metrics_per_sample(
    prediction: GeometrySSLForward,
    batch: PaddedOnlineGeometryBatch,
) -> dict[str, tuple[float | None, ...]]:
    'Handle geometry SSL reconstruction metrics per sample.'

    field_mask = batch.field_targets.valid_mask.unsqueeze(-1).expand_as(prediction.density)
    edge_mask = batch.sensitivity_targets.valid_mask
    edge_band_mask = edge_mask.unsqueeze(-1).expand_as(batch.sensitivity_targets.field_sensitivity)
    density_error = prediction.density - batch.field_targets.density
    kappa_error = prediction.kappa - batch.sensitivity_targets.kappa
    owner_index = batch.sensitivity_targets.owner_index
    query_index = batch.sensitivity_targets.query_index
    batch_index = torch.arange(prediction.density.shape[0], device=prediction.density.device).unsqueeze(1)
    selected_density = prediction.density[batch_index, owner_index, query_index]
    selected_distance = batch.field_targets.distance[batch_index, owner_index, query_index]
    inverse_sigma_squared = _edge_inverse_sigma_squared(batch.field_targets.bandwidths)
    derived = (
        -selected_distance.unsqueeze(-1)
        * inverse_sigma_squared
        * selected_density
        * prediction.kappa.unsqueeze(-1)
    )
    derived_error = derived - batch.sensitivity_targets.field_sensitivity
    return {
        "density": _per_sample_masked_mse(density_error, field_mask),
        "kappa": _per_sample_masked_mse(kappa_error, edge_mask),
        "derived_field": _per_sample_masked_mse(derived_error, edge_band_mask),
    }


def geometry_ssl_stratified_components_per_sample(
    prediction: GeometrySSLForward,
    batch: PaddedOnlineGeometryBatch,
) -> StratifiedComponents:
    'Handle geometry SSL stratified components per sample.'

    density_error = prediction.density - batch.field_targets.density
    kappa_error = prediction.kappa - batch.sensitivity_targets.kappa  # Shapes [B,E]; units m/rad.
    derived_error = _derived_field_error(prediction, batch)  # units rad
    batch_size, owner_count, query_count, bandwidth_count = density_error.shape
    field_valid = batch.field_targets.valid_mask.unsqueeze(-1).expand_as(density_error)
    edge_valid = batch.sensitivity_targets.valid_mask  # shapes [B,E]
    edge_band_valid = edge_valid.unsqueeze(-1).expand_as(derived_error)

    # shapes [B,G]
    owner_role = batch.field_targets.owner_role  # shapes [G], [B,G]
    if owner_role.ndim == 1:
        owner_role = owner_role.unsqueeze(0).expand(batch_size, -1)
    owner_index = _batched_selector(batch.sensitivity_targets.owner_index, batch_size)  # `[B,E]`
    query_index = _batched_selector(batch.sensitivity_targets.query_index, batch_size)  # `[B,E]`
    batch_index = torch.arange(batch_size, device=density_error.device).unsqueeze(1)  # `[B,1]`
    edge_owner_role = owner_role.gather(1, owner_index)  # shapes [B,E]; canonical PALM/JOINT/TIP axis
    edge_query_stratum = batch.field_targets.query_stratum[batch_index, owner_index, query_index]  # `[B,E]`
    edge_distance = batch.field_targets.distance[batch_index, owner_index, query_index]  # shapes [B,E]; units m
    ancestor = _batched_selector(batch.sensitivity_targets.ancestor_mask, batch_size)  # `[B,E]` bool

    result: StratifiedComponents = {
        "density": {"owner_role": {}, "query_stratum": {}, "bandwidth": {}, "distance_shell": {}},
        "kappa": {"owner_role": {}, "query_stratum": {}, "distance_shell": {}, "ancestor": {}},
        "derived_field": {
            "owner_role": {},
            "query_stratum": {},
            "bandwidth": {},
            "distance_shell": {},
            "ancestor": {},
        },
    }


    for role_value, role_name in ((0, "palm"), (1, "joint"), (2, "tip")):
        density_mask = field_valid & (owner_role[:, :, None, None] == role_value)
        edge_mask = edge_valid & (edge_owner_role == role_value)  # `[B,E]`
        result["density"]["owner_role"][role_name] = _per_sample_masked_components(
            density_error, density_mask
        )
        result["kappa"]["owner_role"][role_name] = _per_sample_masked_components(kappa_error, edge_mask)
        result["derived_field"]["owner_role"][role_name] = _per_sample_masked_components(
            derived_error, edge_mask.unsqueeze(-1).expand_as(derived_error)
        )
    for stratum_value, stratum_name in ((0, "workspace"), (1, "owner_shell"), (2, "adjacent")):
        density_mask = field_valid & (batch.field_targets.query_stratum.unsqueeze(-1) == stratum_value)
        edge_mask = edge_valid & (edge_query_stratum == stratum_value)
        result["density"]["query_stratum"][stratum_name] = _per_sample_masked_components(
            density_error, density_mask
        )
        result["kappa"]["query_stratum"][stratum_name] = _per_sample_masked_components(kappa_error, edge_mask)
        result["derived_field"]["query_stratum"][stratum_name] = _per_sample_masked_components(
            derived_error, edge_mask.unsqueeze(-1).expand_as(derived_error)
        )


    for bandwidth_index in range(bandwidth_count):
        bin_name = f"sigma_{bandwidth_index}"
        density_band_mask = field_valid[..., bandwidth_index]  # `[B,G,N_Q]`
        derived_band_mask = edge_band_valid[..., bandwidth_index]  # `[B,E]`
        result["density"]["bandwidth"][bin_name] = _per_sample_masked_components(
            density_error[..., bandwidth_index], density_band_mask
        )
        result["derived_field"]["bandwidth"][bin_name] = _per_sample_masked_components(
            derived_error[..., bandwidth_index], derived_band_mask
        )


    for shell_name, field_shell, edge_shell in _distance_shell_masks(
        batch.field_targets.distance,
        edge_distance,
        batch.field_targets.bandwidths,
    ):
        density_mask = field_valid & field_shell.unsqueeze(-1)
        edge_mask = edge_valid & edge_shell
        result["density"]["distance_shell"][shell_name] = _per_sample_masked_components(
            density_error, density_mask
        )
        result["kappa"]["distance_shell"][shell_name] = _per_sample_masked_components(kappa_error, edge_mask)
        result["derived_field"]["distance_shell"][shell_name] = _per_sample_masked_components(
            derived_error, edge_mask.unsqueeze(-1).expand_as(derived_error)
        )


    for ancestor_value, ancestor_name in ((True, "ancestor"), (False, "non_ancestor")):
        edge_mask = edge_valid & (ancestor == ancestor_value)
        result["kappa"]["ancestor"][ancestor_name] = _per_sample_masked_components(kappa_error, edge_mask)
        result["derived_field"]["ancestor"][ancestor_name] = _per_sample_masked_components(
            derived_error, edge_mask.unsqueeze(-1).expand_as(derived_error)
        )
    return result


def aggregate_geometry_ssl_stratified_components(
    blocks: tuple[tuple[tuple[str, ...], StratifiedComponents], ...],
) -> dict[str, object]:
    'Aggregate geometry SSL stratified components; shapes [str,object].'

    if not blocks:
        raise ValueError("stratified validation aggregation requires non-empty blocks")
    # shapes [numerator_sum,denominator_sum]
    accumulated: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for asset_ids, components in blocks:
        for metric, axes in components.items():
            metric_store = accumulated.setdefault(metric, {})
            for axis, bins in axes.items():
                axis_store = metric_store.setdefault(axis, {})
                for bin_name, (numerators, denominators) in bins.items():
                    if len(numerators) != len(asset_ids) or len(denominators) != len(asset_ids):
                        raise ValueError("stratified component batch axis does not match asset IDs")
                    bin_store = axis_store.setdefault(bin_name, {})
                    for asset_id, numerator, denominator in zip(asset_ids, numerators, denominators):
                        totals = bin_store.setdefault(asset_id, [0.0, 0.0])
                        totals[0] += float(numerator)
                        totals[1] += float(denominator)

    bin_scores: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    axis_scores: dict[str, dict[str, float]] = {}
    metric_scores: dict[str, float] = {}
    morphology_values: dict[str, dict[str, list[float]]] = {}
    for metric, axes in accumulated.items():
        bin_scores[metric] = {}
        axis_scores[metric] = {}
        morphology_values[metric] = {}
        for axis, bins in axes.items():
            bin_scores[metric][axis] = {}
            nonempty_bin_scores: list[float] = []
            for bin_name, by_asset in bins.items():
                morphology_scores = [
                    numerator / denominator
                    for numerator, denominator in by_asset.values()
                    if denominator > 0.0
                ]
                if not morphology_scores:
                    continue
                score = sum(morphology_scores) / len(morphology_scores)
                bin_scores[metric][axis][bin_name] = {
                    "mse": score,
                    "morphology_count": len(morphology_scores),
                }
                nonempty_bin_scores.append(score)
                for asset_id, (numerator, denominator) in by_asset.items():
                    if denominator > 0.0:
                        morphology_values[metric].setdefault(asset_id, []).append(numerator / denominator)
            if nonempty_bin_scores:
                axis_scores[metric][axis] = sum(nonempty_bin_scores) / len(nonempty_bin_scores)
        if not axis_scores[metric]:
            raise ValueError(f"stratified validation metric={metric!r} has no non-empty axes")
        metric_scores[metric] = sum(axis_scores[metric].values()) / len(axis_scores[metric])
    expected_metrics = {"density", "kappa", "derived_field"}
    if set(metric_scores) != expected_metrics:
        raise ValueError("stratified validation must produce density, kappa and derived_field scores")
    morphology_scores = {
        metric: {
            asset_id: sum(values) / len(values)
            for asset_id, values in sorted(by_asset.items())
            if values
        }
        for metric, by_asset in morphology_values.items()
    }
    return {
        "metric_scores": metric_scores,
        "axis_scores": axis_scores,
        "bin_scores": bin_scores,
        "morphology_scores": morphology_scores,
    }


def _per_sample_masked_mse(error: torch.Tensor, mask: torch.Tensor) -> tuple[float | None, ...]:
    'Handle per sample masked mse.'

    weight = mask.to(error.dtype)
    flattened_error = error.reshape(error.shape[0], -1)
    flattened_weight = weight.reshape(weight.shape[0], -1)
    numerators = (flattened_error.square() * flattened_weight).sum(dim=1)
    denominators = flattened_weight.sum(dim=1)
    return tuple(
        float(numerator.detach() / denominator.detach()) if float(denominator.detach()) > 0.0 else None
        for numerator, denominator in zip(numerators, denominators)
    )


def _per_sample_masked_components(
    error: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    'Handle per sample masked components.'

    weight = mask.to(error.dtype)
    flattened_error = error.reshape(error.shape[0], -1)  # shapes [B,K]
    flattened_weight = weight.reshape(weight.shape[0], -1)  # `[B,K]`
    numerators = (flattened_error.square() * flattened_weight).sum(dim=1)  # shapes [B]
    denominators = flattened_weight.sum(dim=1)  # shapes [B]
    return (
        tuple(float(value.detach()) for value in numerators),
        tuple(float(value.detach()) for value in denominators),
    )


def _batched_selector(selector: torch.Tensor, batch_size: int) -> torch.Tensor:
    'Handle batched selector; shapes [E], [B,E].'

    return selector.unsqueeze(0).expand(batch_size, -1) if selector.ndim == 1 else selector


def _derived_field_error(
    prediction: GeometrySSLForward,
    batch: PaddedOnlineGeometryBatch,
) -> torch.Tensor:
    'Handle derived field error.'

    batch_size = prediction.density.shape[0]
    owner_index = _batched_selector(batch.sensitivity_targets.owner_index, batch_size)  # `[B,E]`
    query_index = _batched_selector(batch.sensitivity_targets.query_index, batch_size)  # `[B,E]`
    batch_index = torch.arange(batch_size, device=prediction.density.device).unsqueeze(1)  # `[B,1]`
    selected_density = prediction.density[batch_index, owner_index, query_index]  # `[B,E,L]`
    selected_distance = batch.field_targets.distance[batch_index, owner_index, query_index]  # `[B,E]`
    inverse_sigma_squared = _edge_inverse_sigma_squared(batch.field_targets.bandwidths)  # `[1|B,1,L]`
    derived = (
        -selected_distance.unsqueeze(-1)
        * inverse_sigma_squared
        * selected_density
        * prediction.kappa.unsqueeze(-1)
    )  # shapes [B,E,L]; units rad
    return derived - batch.sensitivity_targets.field_sensitivity  # `[B,E,L]`


def _distance_shell_masks(
    field_distance: torch.Tensor,
    edge_distance: torch.Tensor,
    bandwidths: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    'Handle distance shell masks.'

    if bandwidths.ndim not in {1, 2}:
        raise ValueError("distance-shell bandwidths must have shape [L] or [B,L]")
    if torch.any(bandwidths[..., 1:] <= bandwidths[..., :-1]):
        raise ValueError("distance-shell stratification requires strictly increasing bandwidths")
    shells: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    lower = None
    for index in range(bandwidths.shape[-1]):
        upper = bandwidths[index] if bandwidths.ndim == 1 else bandwidths[:, index]
        field_upper = upper if bandwidths.ndim == 1 else upper[:, None, None]
        edge_upper = upper if bandwidths.ndim == 1 else upper[:, None]
        if lower is None:
            field_mask = field_distance <= field_upper
            edge_mask = edge_distance <= edge_upper
            name = f"le_sigma_{index}"
        else:
            field_lower = lower if bandwidths.ndim == 1 else lower[:, None, None]
            edge_lower = lower if bandwidths.ndim == 1 else lower[:, None]
            field_mask = (field_distance > field_lower) & (field_distance <= field_upper)
            edge_mask = (edge_distance > edge_lower) & (edge_distance <= edge_upper)
            name = f"sigma_{index - 1}_to_{index}"
        shells.append((name, field_mask, edge_mask))
        lower = upper
    if lower is None:
        raise ValueError("distance-shell stratification requires at least one bandwidth")
    field_lower = lower if bandwidths.ndim == 1 else lower[:, None, None]
    edge_lower = lower if bandwidths.ndim == 1 else lower[:, None]
    shells.append(("gt_sigma_last", field_distance > field_lower, edge_distance > edge_lower))
    return tuple(shells)


def _edge_inverse_sigma_squared(bandwidths: torch.Tensor) -> torch.Tensor:
    'Handle edge inverse sigma squared; shapes [L], [B,L].'

    inverse = bandwidths.square().reciprocal()  # Units m^-2.
    return inverse.view(1, 1, -1) if inverse.ndim == 1 else inverse.unsqueeze(1)


def joint_sign_observable_metrics(
    reference: GeometrySSLForward,
    rewritten: GeometrySSLForward,
    *,
    joint_sign: torch.Tensor,
    joint_index: torch.Tensor,
    density_valid_mask: torch.Tensor,
    edge_valid_mask: torch.Tensor,
) -> dict[str, float]:
    'Measure dimensionless density invariance and sensitivity sign response after a JOINT coordinate rewrite.'

    if reference.density.shape != rewritten.density.shape or reference.kappa.shape != rewritten.kappa.shape:
        raise ValueError("joint-sign observable predictions must share density/kappa shapes")
    batch_size = reference.kappa.shape[0]
    signs = joint_sign.to(reference.kappa)
    if signs.ndim == 1:
        signs = signs.unsqueeze(0).expand(batch_size, -1)
    selectors = _batched_selector(joint_index, batch_size)
    edge_sign = torch.gather(signs, 1, selectors)  # shapes [B,E]
    density_weight = density_valid_mask.to(reference.density.dtype).unsqueeze(-1).expand_as(reference.density)
    edge_weight = edge_valid_mask.to(reference.kappa.dtype)
    density_error = rewritten.density - reference.density
    kappa_error = rewritten.kappa - edge_sign * reference.kappa
    return {
        "density_invariance_mse": float(
            (density_error.square() * density_weight).sum() / density_weight.sum().clamp_min(1.0)
        ),
        "kappa_sign_equivariance_mse": float(
            (kappa_error.square() * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)
        ),
    }


def density_configuration_jvp(
    model: GeometrySSLModel,
    q: torch.Tensor,
    evidence: StaticGeometryEvidence,
    query_points_h: torch.Tensor,
    bandwidths: torch.Tensor,
    *,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Handle density configuration jvp; units rad.'

    if direction.shape != q.shape:
        raise ValueError("density JVP direction must have the same shape as q")

    def density_from_configuration(configuration: torch.Tensor) -> torch.Tensor:
        return model(
            configuration,
            evidence,
            query_points_h,
            bandwidths,
            owner_index,
            query_index,
            joint_index,
        ).density

    primal, tangent = torch.autograd.functional.jvp(
        density_from_configuration,
        q.detach(),
        direction.detach(),
        create_graph=False,
        strict=True,
    )
    return cast(torch.Tensor, primal), cast(torch.Tensor, tangent)


def task_gradient_gram(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    baselines: Mapping[str, float],
) -> dict[str, float]:
    'Handle task gradient gram.'

    if set(losses) != {"density", "kappa"} or set(baselines) != {"density", "kappa"}:
        raise ValueError("task gradient Gram requires density/kappa losses and baselines")
    parameter_tuple = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameter_tuple:
        raise ValueError("task gradient Gram requires at least one trainable parameter")
    gradients: dict[str, torch.Tensor] = {}
    for index, name in enumerate(("density", "kappa")):
        parts = torch.autograd.grad(
            losses[name] / float(baselines[name]),
            parameter_tuple,
            retain_graph=index == 0,
            allow_unused=True,
        )
        gradients[name] = torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                for parameter, gradient in zip(parameter_tuple, parts)
            ]
        )
    rho = gradients["density"]
    kappa = gradients["kappa"]
    rho_sq = float(rho.square().sum())
    kappa_sq = float(kappa.square().sum())
    dot = float((rho * kappa).sum())
    determinant = max(rho_sq * kappa_sq - dot * dot, 0.0)
    trace = rho_sq + kappa_sq
    discriminant = max(trace * trace - 4.0 * determinant, 0.0)
    largest = 0.5 * (trace + math.sqrt(discriminant))
    smallest = 0.5 * (trace - math.sqrt(discriminant))
    return {
        "rho_norm": math.sqrt(max(rho_sq, 0.0)),
        "kappa_norm": math.sqrt(max(kappa_sq, 0.0)),
        "dot": dot,
        "cosine": dot / math.sqrt(max(rho_sq * kappa_sq, 1.0e-30)),
        "gram_determinant": determinant,
        "gram_condition": largest / max(smallest, 1.0e-30),
    }


__all__ = [
    "GeometrySSLAblation",
    "cross_asset_permutation",
    "aggregate_geometry_ssl_stratified_components",
    "geometry_ssl_ablation_forward",
    "geometry_ssl_reconstruction_metrics",
    "geometry_ssl_reconstruction_metrics_per_sample",
    "geometry_ssl_stratified_components_per_sample",
    "density_configuration_jvp",
    "joint_sign_observable_metrics",
    "same_asset_q_permutation",
    "task_gradient_gram",
]
