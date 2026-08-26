r"""几何 SSL 的 query-only 与 latent-shuffle 必要性诊断。

完整模型预测写作 $f(x,q,s)$：$x$ 是固定 `{h}` query，$q$ 是当前物理关节角，$s$ 是静态手型证据。
诊断共享同一 query encoder 和 disposable decoder，仅改变 unified morphology latent：

$$
f_{query}(x)=f(x,\mathbf 0),
\qquad
f_{shuffle}^{(b)}(x)=f(x,Z_{\pi(b)}).
$$

若完整模型不优于 query-only，表示 decoder 可能绕过 hand conditioning；若 latent shuffle 不显著恶化，
表示 latent 与本样本几何未建立可辨识对应。两者都是必要性诊断，不替代 held-out geometry 指标。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal  # 只允许预注册 ablation，避免自由字符串改变实验含义

import torch  # latent、query 与 batch permutation 全部保持 PyTorch 计算图

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import PaddedOnlineGeometryBatch
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel  # 完整 SSL 组装边界
from anymani.distill.models.input_adapters.geometry import (  # retained latent/evidence 类型合同
    GeometryLatents,
    StaticGeometryEvidence,
)

GeometrySSLAblation = Literal[
    "query_only",
    "latent_shuffle",
    "joint_token_shuffle",
]  # 受控诊断枚举

StratifiedComponents = dict[
    str,
    dict[str, dict[str, tuple[tuple[float, ...], tuple[float, ...]]]],
]  # metric -> axis -> bin -> `(per-sample numerator, per-sample denominator)`


def geometry_ssl_ablation_forward(
    model: GeometrySSLModel,  # 同一组 frozen/evaluated 参数，不能为 baseline 另训 decoder
    q: torch.Tensor,  # `[B,N_J]`，rad
    evidence: StaticGeometryEvidence,  # 同结构或 padding 后跨结构静态证据
    query_points_h: torch.Tensor,  # `[B,G,N_Q,3]`，`{h}`，m
    bandwidths: torch.Tensor,  # `[N_σ]` 或 `[B,N_σ]`，显式 decoder 条件
    *,
    owner_index: torch.Tensor,  # `[E]` 或 `[B,E]`
    query_index: torch.Tensor,  # 与 owner selector 同形状
    joint_index: torch.Tensor,  # 与 owner selector 同形状
    ablation: GeometrySSLAblation,  # `query_only` 或 `latent_shuffle`
    batch_permutation: torch.Tensor | None = None,  # shuffle 时 `[B]` 双射
) -> GeometrySSLForward:
    r"""保持同一 decoder/query path，仅移除或错配 morphology latent。

    Args:
        model (GeometrySSLModel): 待诊断的完整 SSL 模型；不更新参数。
        q (torch.Tensor): 当前物理构型，形状 ``[B,N_J]``，单位 rad。
        evidence (StaticGeometryEvidence): anchors/home/screws/graph 与 padding masks。
        query_points_h (torch.Tensor): 固定 hand-frame query，形状 ``[B,G,N_Q,3]``，单位 m。
        owner_index (torch.Tensor): sampled sensitivity owner selectors。
        query_index (torch.Tensor): sampled query selectors。
        joint_index (torch.Tensor): sampled JOINT selectors。
        ablation (GeometrySSLAblation): 删除 latent 或沿 batch 错配 latent。
        batch_permutation (torch.Tensor | None): ``latent_shuffle`` 的 ``[B]`` 双射。

    Returns:
        GeometrySSLForward: ablated latent、未改变 query features 与对应 density/κ 预测。

    Raises:
        ValueError: permutation 不是 ``[0,B)`` 双射或 ablation 名未知时抛出。
    """

    latents = model.encoder(q, evidence)  # 原始 unified $Z:[B,G,D]$
    query_features = model.encoder.encode_points(  # 与完整模型完全相同的 point-anchor 前端
        query_points_h.detach(), evidence
    )  # `[B,G,N_Q,D_q]`；固定 query 不接收 sampler 梯度
    entity_valid = evidence.entity_valid_mask  # `[B,G]` 或 None；padding 不是可学习实体
    if entity_valid is not None:  # 只在跨结构稠密容器中需要显式 owner mask
        if entity_valid.ndim == 1:  # 单结构共享 mask 扩成 batch 视图，不复制物理证据
            entity_valid = entity_valid.unsqueeze(0).expand(q.shape[0], -1)  # `[B,G]`
        query_features = query_features * entity_valid.unsqueeze(-1).unsqueeze(-1)  # invalid owner 精确零

    if ablation == "query_only":  # 保留 query path，但删除全部 hand/q conditioning
        ablated = GeometryLatents(torch.zeros_like(latents.entities))  # $Z\leftarrow0$，decoder 容量不变
    elif ablation == "latent_shuffle":  # 保留 latent 边缘分布，只破坏样本对应关系
        if batch_permutation is None or batch_permutation.shape != (q.shape[0],):  # 必须显式 `[B]`
            raise ValueError("latent_shuffle requires batch_permutation with shape [B]")  # 不猜 permutation
        expected = torch.arange(q.shape[0], device=batch_permutation.device)  # 合法索引集合 `[0,B)`
        if not torch.equal(torch.sort(batch_permutation).values, expected):  # 拒绝重复/遗漏样本
            raise ValueError("batch_permutation must be a bijection of [0,B)")  # 保持 batch 分布不变
        ablated = GeometryLatents(latents.entities.index_select(0, batch_permutation))  # $Z_b\leftarrow Z_{\pi(b)}$
    elif ablation == "joint_token_shuffle":  # 只破坏每个 hand 内的 JOINT token binding
        joint_valid = evidence.joint_valid_mask
        if joint_valid is None:
            joint_valid = torch.ones(q.shape, device=q.device, dtype=torch.bool)
        if joint_valid.ndim == 1:
            joint_valid = joint_valid.unsqueeze(0).expand(q.shape[0], -1)
        joint_entities = evidence.joint_entity_index
        if joint_entities.ndim == 1:
            joint_entities = joint_entities.unsqueeze(0).expand(q.shape[0], -1)
        shuffled = latents.entities.clone()  # PALM/TIP 和未选 entity 原样保留
        for batch_index in range(q.shape[0]):
            valid_joint_slots = torch.where(joint_valid[batch_index])[0]
            if len(valid_joint_slots) < 2:
                raise ValueError("joint_token_shuffle requires at least two valid JOINTs per sample")
            entity_slots = joint_entities[batch_index, valid_joint_slots]  # 有效 JOINT 在统一 entity 轴的位置
            shuffled[batch_index, entity_slots] = latents.entities[batch_index, entity_slots].roll(1, dims=0)
        ablated = GeometryLatents(shuffled)
    else:  # 未注册诊断不得静默退回完整 forward
        raise ValueError(f"unknown geometry SSL ablation={ablation!r}")
    return model.decode_latents(  # decoder 权重、query features 与 selectors 均与完整模型相同
        ablated,  # 唯一被干预的变量
        query_features,  # 未打乱的本样本 query path
        bandwidths=bandwidths,  # 与完整 forward 相同的实际 sigma realization
        entity_valid_mask=entity_valid,  # invalid owner prediction 继续严格清零
        joint_entity_index=evidence.joint_entity_index,  # 保持原 JOINT routing，故 shuffle 是故意错配
        owner_index=owner_index,  # 保留本样本 sampled owner
        query_index=query_index,  # 保留本样本 sampled query
        joint_index=joint_index,  # 保留本样本 sampled JOINT
    )


def same_asset_q_permutation(asset_ids: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
    r"""构造每个资产内部循环移位的 batch permutation，用于同手跨 q latent shuffle。"""

    permutation = torch.arange(len(asset_ids), device=device)
    for asset_id in dict.fromkeys(asset_ids):
        indices = [index for index, candidate in enumerate(asset_ids) if candidate == asset_id]
        if len(indices) < 2:
            raise ValueError("same-asset q shuffle requires at least two q samples per asset")
        source = torch.tensor(indices, device=device)
        permutation[source] = source.roll(1)
    return permutation


def cross_asset_permutation(asset_ids: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
    r"""构造每个样本都映射到不同资产的循环 permutation。"""

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
    r"""计算无需 q-autograd 的 density、κ 与 derived-g raw MSE，供固定 ablation 比较。"""

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
    r"""按每个 `(asset,q)` 样本返回 raw MSE，供配对 bootstrap 与形态等权聚合。

    某个样本若没有有效一阶 edge，则对应 κ/derived 指标为 `None`；不能把空分母写成 0
    后伪装为完美预测。零阶 field 每个真实样本必须有有效 query。
    """

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
    r"""返回 owner/stratum/bandwidth/distance/ancestor 分层的逐样本充分统计量。

    每个 bin 保存平方误差和 $N$ 与有效标量数 $D$，后续先在同一 morphology 内跨 q 求
    $\sum_qN/\sum_qD$，再对 morphology 等权。各轴是独立边际，不构造高维笛卡尔积，避免
    4 个 held-out morphology 被数百个稀疏交叉格切碎。

    Returns:
        StratifiedComponents: ``metric -> axis -> bin -> (numerators, denominators)``；两个 tuple
        的长度均为 batch size。适用轴为：

        - density：owner role、50:25:25 query stratum、bandwidth、distance shell；
        - κ：owner role、query stratum、distance shell、ancestor/non-ancestor；
        - derived-g：上述全部五个轴。
    """

    density_error = prediction.density - batch.field_targets.density  # `[B,G,N_Q,N_σ]`，无量纲
    kappa_error = prediction.kappa - batch.sensitivity_targets.kappa  # `[B,E]`，m/rad
    derived_error = _derived_field_error(prediction, batch)  # `[B,E,N_σ]`，1/rad
    batch_size, owner_count, query_count, bandwidth_count = density_error.shape  # 四条监督轴
    field_valid = batch.field_targets.valid_mask.unsqueeze(-1).expand_as(density_error)  # `[B,G,N_Q,N_σ]`
    edge_valid = batch.sensitivity_targets.valid_mask  # `[B,E]`，唯一最近点与 padding 共同有效
    edge_band_valid = edge_valid.unsqueeze(-1).expand_as(derived_error)  # `[B,E,N_σ]`

    # 统一 owner role 为 `[B,G]`，再按 sampled edge owner selector收集一阶 role。
    owner_role = batch.field_targets.owner_role  # `[G]` 或 `[B,G]`，0/1/2
    if owner_role.ndim == 1:
        owner_role = owner_role.unsqueeze(0).expand(batch_size, -1)  # 同结构 batch 共享 role
    owner_index = _batched_selector(batch.sensitivity_targets.owner_index, batch_size)  # `[B,E]`
    query_index = _batched_selector(batch.sensitivity_targets.query_index, batch_size)  # `[B,E]`
    batch_index = torch.arange(batch_size, device=density_error.device).unsqueeze(1)  # `[B,1]`
    edge_owner_role = owner_role.gather(1, owner_index)  # `[B,E]`，sampled owner 的 PALM/JOINT/TIP
    edge_query_stratum = batch.field_targets.query_stratum[batch_index, owner_index, query_index]  # `[B,E]`
    edge_distance = batch.field_targets.distance[batch_index, owner_index, query_index]  # `[B,E]`，m
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

    # owner role 与 query stratum 是 density/κ/g 共享的物理边际轴。
    for role_value, role_name in ((0, "palm"), (1, "joint"), (2, "tip")):
        density_mask = field_valid & (owner_role[:, :, None, None] == role_value)  # `[B,G,N_Q,N_σ]`
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

    # sigma 轴只适用于 density 与 derived-g；κ 是共享的距离导数，不复制 $N_\sigma$ 轴。
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

    # distance shell 使用实际 bandwidth 边界：`<=σ0`、逐相邻 σ 区间与 `>σ_last`。
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

    # ancestor/non-ancestor 只适用于 sampled一阶 edges；结构零也必须作为独立 bin 被报告。
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
    r"""聚合 density/kappa 与 post-hoc derived-field 三项诊断；selection 只消费前两项。

    同一 ``asset_id`` 的所有 q 先对 numerator/denominator 求和得到该 morphology 的 bin MSE；
    每个 bin 再对具有有效 denominator 的 morphology 等权；每个 axis 对非空 bins 等权；每个
    metric 对适用 axes 等权。这样不同 q block 大小、尾资产组、bandwidth bin 数和 ancestor bin
    数都不会隐式改变 morphology 或 axis 权重。

    Returns:
        dict[str, object]: 包含 ``metric_scores``、``axis_scores``、``bin_scores`` 及每 bin 的
        morphology 数；可直接写入 checkpoint selection evidence。
    """

    if not blocks:
        raise ValueError("stratified validation aggregation requires non-empty blocks")
    # `metric/axis/bin/asset -> [numerator_sum, denominator_sum]` 是重排 block 后不变的充分统计量。
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
    for metric, axes in accumulated.items():
        bin_scores[metric] = {}
        axis_scores[metric] = {}
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
                    continue  # 空 bin 不以 0 进入 axis 均值
                score = sum(morphology_scores) / len(morphology_scores)  # morphology 等权
                bin_scores[metric][axis][bin_name] = {
                    "mse": score,
                    "morphology_count": len(morphology_scores),
                }
                nonempty_bin_scores.append(score)
            if nonempty_bin_scores:
                axis_scores[metric][axis] = sum(nonempty_bin_scores) / len(nonempty_bin_scores)  # bin 等权
        if not axis_scores[metric]:
            raise ValueError(f"stratified validation metric={metric!r} has no non-empty axes")
        metric_scores[metric] = sum(axis_scores[metric].values()) / len(axis_scores[metric])  # axis 等权
    expected_metrics = {"density", "kappa", "derived_field"}
    if set(metric_scores) != expected_metrics:
        raise ValueError("stratified validation must produce density, kappa and derived_field scores")
    return {"metric_scores": metric_scores, "axis_scores": axis_scores, "bin_scores": bin_scores}


def _per_sample_masked_mse(error: torch.Tensor, mask: torch.Tensor) -> tuple[float | None, ...]:
    r"""沿 batch 外全部轴归约 masked MSE，空 denominator 返回 `None`。"""

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
    r"""沿 batch 外全部轴返回平方误差和与有效标量数，不在此层做平均。"""

    weight = mask.to(error.dtype)  # bool mask -> 0/1，维持 error dtype 的乘法路径
    flattened_error = error.reshape(error.shape[0], -1)  # `[B,K]`，K 是当前 bin 的候选标量轴
    flattened_weight = weight.reshape(weight.shape[0], -1)  # `[B,K]`
    numerators = (flattened_error.square() * flattened_weight).sum(dim=1)  # `[B]`，平方误差和
    denominators = flattened_weight.sum(dim=1)  # `[B]`，有效标量数
    return (
        tuple(float(value.detach()) for value in numerators),
        tuple(float(value.detach()) for value in denominators),
    )


def _batched_selector(selector: torch.Tensor, batch_size: int) -> torch.Tensor:
    r"""把共享 `[E]` selector 扩为 `[B,E]`，不复制物理索引。"""

    return selector.unsqueeze(0).expand(batch_size, -1) if selector.ndim == 1 else selector


def _derived_field_error(
    prediction: GeometrySSLForward,
    batch: PaddedOnlineGeometryBatch,
) -> torch.Tensor:
    r"""返回链式预测 $\hat g=-d\sigma^{-2}\hat\rho\hat\kappa$ 与 teacher 的误差。"""

    batch_size = prediction.density.shape[0]  # $B$
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
    )  # `[B,E,L]`，1/rad
    return derived - batch.sensitivity_targets.field_sensitivity  # `[B,E,L]`


def _distance_shell_masks(
    field_distance: torch.Tensor,
    edge_distance: torch.Tensor,
    bandwidths: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    r"""按递增物理 bandwidth 构造互斥且完备的 distance-shell masks。"""

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
    r"""把 ``[L]`` 或 ``[B,L]`` sigma 变形成 sampled-edge 可广播的 ``[1|B,1,L]``。"""

    inverse = bandwidths.square().reciprocal()  # $\sigma^{-2}$，m⁻²
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
    r"""评估完整 coordinate rewrite 后 density 不变与 $\kappa$ sign-equivariance。

    该函数不对 latent 做比较。调用方必须先同步改写 $q,q_{home},\mathcal S$；策略侧还需在其边界
    同步 limits、velocity、PD target、previous action 与 action。这里只根据 sampled edge 的 JOINT
    selector 形成目标 $\kappa'_{o,i}=s_i\kappa_{o,i}$。
    """

    if reference.density.shape != rewritten.density.shape or reference.kappa.shape != rewritten.kappa.shape:
        raise ValueError("joint-sign observable predictions must share density/kappa shapes")
    batch_size = reference.kappa.shape[0]
    signs = joint_sign.to(reference.kappa)
    if signs.ndim == 1:
        signs = signs.unsqueeze(0).expand(batch_size, -1)
    selectors = _batched_selector(joint_index, batch_size)
    edge_sign = torch.gather(signs, 1, selectors)  # `[B,E]` 的对应坐标方向
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
    r"""手动计算真实网络 $\partial\hat\rho/\partial q\,v$，不进入 active training graph。

    ``direction`` 与 $q$ 同形状、单位为 rad；返回 primal density 与沿该方向的 tangent。query、sigma、
    selectors 和静态证据固定，因此该 JVP 只测 retained encoder 对 current configuration 的依赖。
    """

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

    return torch.autograd.functional.jvp(
        density_from_configuration,
        q.detach(),
        direction.detach(),
        create_graph=False,
        strict=True,
    )


def task_gradient_gram(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    baselines: Mapping[str, float],
) -> dict[str, float]:
    r"""对调用方指定参数层级计算 rho/kappa full-gradient Gram 统计。

    调用方可传 unified representation frontend、最后一个 Transformer block 或完整 retained encoder
    参数；本函数只返回 norm/dot/cosine/condition，不保存梯度向量，也不执行 optimizer update。
    """

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
]  # 稳定诊断公开面
