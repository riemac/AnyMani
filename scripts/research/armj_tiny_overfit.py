r"""AR-MPJ-003：anchor-relational Material-point Jacobian 单目标 tiny-overfit。

该 probe 不联合 Gaussian density、不读取 closest point/query/κ，也不使用 FairGrad。它只验证一个更早的
可行性命题：小型 retained encoder 是否能在固定 morphology/q bank 上，把静态 home surface、PALM
anchors、space screws 与当前 q 组合成可读的一阶关系真值。

每个有效 JOINT 固定选择两条 descendant owner edge 与一条 PALM structural-zero edge；每条 edge 选择
固定 owner-local home-surface material identity。相同 asset 的不同 q 始终使用相同 material point，不允许
用 q-dependent point identity 帮助记忆。Reader 对每个实际 anchor 共享参数，因此 $K$ 轴保持 permutation
equivariance。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.material_point_jacobian import (
    BilinearAnchorRelationalJacobianDecoder,
    BilinearAnchorRelationalJacobianDecoderCfg,
)
from anymani.distill.models.input_adapters.encoder import (
    GeometryEncoderCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
)
from anymani.distill.representations.targets.material_point_jacobian import (
    MaterialPointRelationJacobianTarget,
    generate_material_point_relation_jacobian_targets,
)
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow
from torch import nn

# 四通道尺度来自 AR-MPJ-001 的 64-asset teacher-only RMS，只用于本 probe 的无量纲优化条件。
CHANNEL_SCALE = (0.30, 0.30, 0.13, 0.13)


@dataclass(frozen=True)
class FixedRelationBatch:
    r"""固定 q/evidence 与 padded material-edge truth。

    `edge_valid_mask` 区分实际 sampled edge 与跨结构 padding；`anchor_valid_mask` 来自 evidence 的
    可变 $K$ padding。Radius channel 还需 `radius_valid_mask`，其余三个 relation channels 在
    $r_{\parallel}=0$ 时仍有定义。
    """

    q: torch.Tensor  # `[B,N_J^max]`，rad
    evidence: Any  # batched StaticGeometryEvidence；脚本避免复制项目 dataclass import surface
    evidence_row_index: torch.Tensor  # `[B]`，q row -> static asset row
    owner_index: torch.Tensor  # `[B,E]`
    joint_index: torch.Tensor  # `[B,E]`
    material_point_index: torch.Tensor  # `[B,E]`，owner home-surface 固定索引
    target: torch.Tensor  # `[B,E,K,4]`，rad$^{-1}$
    edge_valid_mask: torch.Tensor  # `[B,E]`
    active_mask: torch.Tensor  # `[B,E]`，True=descendant，False=structural zero/padding
    radius_valid_mask: torch.Tensor  # `[B,E,K]`
    anchor_valid_mask: torch.Tensor  # `[B,K]`
    asset_ids: tuple[str, ...]  # `[B]`，用于证据与 morphology 聚合


class RelationJacobianReader(nn.Module):
    r"""共享 owner/JOINT context 与 per-anchor static material query 的四通道 reader。"""

    def __init__(self, *, latent_width: int, relation_width: int, hidden_width: int = 64) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(2 * latent_width, relation_width),
            nn.GELU(),
            nn.Linear(relation_width, relation_width),
        )  # $(Z_g,Z_i)\mapsto c_{gi}\in\mathbb R^{D_r}$
        self.output = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 4),
        )  # 每个 material-point/anchor pair 输出固定四通道 $\hat\Gamma$

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        static_pair_feature: torch.Tensor,
        *,
        use_latent: bool,
    ) -> torch.Tensor:
        r"""返回 `[B,E,K,4]` relation sensitivities；query-only 时显式删除动态 latent。"""

        if use_latent:
            context = self.context(torch.cat((owner_latent, joint_latent), dim=-1))  # `[B,E,D_r]`
        else:
            context = torch.zeros_like(static_pair_feature[..., 0, :])  # 不向 reader 泄漏 q/owner/joint latent
        fused = static_pair_feature + context.unsqueeze(-2)  # `[B,E,K,D_r]`，每个 anchor 接收同 edge context
        return self.output(fused)  # `[B,E,K,4]`，rad$^{-1}$


class TinyRelationJacobianModel(nn.Module):
    r"""小型 retained geometry encoder 与 disposable relation reader。"""

    def __init__(
        self,
        *,
        hidden_width: int = 64,
        layers: int = 2,
        relation_width: int = 32,
        reader_kind: str = "additive",
    ) -> None:
        super().__init__()
        frontend = SO2AnchorFrontendCfg(
            relation_width=relation_width,
            home_width=relation_width,
            screw_width=relation_width,
            role_width=8,
            length_scale_m=0.1,
        )  # 默认 tiny 宽度 32；中型 probe 可恢复到 64
        backbone = GraphBiasedTransformerCfg(
            hidden_width=hidden_width,
            layers=layers,
            attention_heads=4,
            feedforward_width=2 * hidden_width,
            dropout=0.0,
            max_graph_distance=8,
        )  # 默认 width64/layers2；中型 probe 使用 width128/layers4
        self.latent_width = backbone.hidden_width  # unified owner/JOINT token width $D=64$
        self.encoder = ImplicitGeometryEncoder(GeometryEncoderCfg(frontend=frontend, backbone=backbone))
        self.reader_kind = reader_kind
        if reader_kind == "additive":
            self.reader = RelationJacobianReader(
                latent_width=backbone.hidden_width,
                relation_width=frontend.relation_width,
                hidden_width=hidden_width,
            )
        elif reader_kind == "bilinear":
            self.reader = BilinearAnchorRelationalJacobianDecoder(
                BilinearAnchorRelationalJacobianDecoderCfg(
                    latent_width=backbone.hidden_width,
                    relation_width=frontend.relation_width,
                    hidden_width=hidden_width,
                    readout_rank=64,
                )
            )
        else:
            raise ValueError(f"unknown relation reader_kind={reader_kind!r}")

    def forward(
        self,
        batch: FixedRelationBatch,
        *,
        use_latent: bool = True,
        latent_row_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""编码当前 q 后，按 owner/joint/material/anchor selectors 执行共享读出。"""

        batch_size, edge_count = batch.owner_index.shape  # 固定 bank 的 `[B,E]`
        batch_axis = torch.arange(batch_size, device=batch.q.device).unsqueeze(1)  # `[B,1]`
        evidence_rows = batch.evidence_row_index  # `[B]`，q row -> asset evidence row

        # Query-only 对照不执行 graph encoder，确保任何可学习信号只来自 q-independent static pair features。
        if use_latent:
            entities = self.encoder(
                batch.q,
                batch.evidence,
                evidence_row_index=evidence_rows,
            ).entities  # `[B,G,D]`，当前 q-conditioned unified Z
            latent_entities = entities if latent_row_index is None else entities[latent_row_index]  # matched/shuffled Z
            latent_evidence_rows = (
                evidence_rows if latent_row_index is None else evidence_rows[latent_row_index]
            )  # shuffled morphology 的 JOINT->entity routing
            owner_latent = latent_entities[batch_axis, batch.owner_index]  # `[B,E,D]`
            joint_entity_by_row = batch.evidence.joint_entity_index[latent_evidence_rows]  # `[B,N_J]`
            selected_joint_entity = joint_entity_by_row[batch_axis, batch.joint_index]  # `[B,E]`
            joint_latent = latent_entities[batch_axis, selected_joint_entity]  # `[B,E,D]`
        else:
            latent_shape = (batch_size, edge_count, self.latent_width)  # query-only 仍保持 reader tensor contract
            owner_latent = torch.zeros(latent_shape, device=batch.q.device, dtype=batch.q.dtype)
            joint_latent = torch.zeros_like(owner_latent)

        # Static query 使用同一个 material identity 的 home `{h}` point；它不含当前 q 或 target Jacobian。
        home_by_row = batch.evidence.home_surface_points[evidence_rows]  # `[B,G,M,3]`，m
        home_points = home_by_row[
            batch_axis,
            batch.owner_index,
            batch.material_point_index,
        ]  # `[B,E,3]`，selected fixed material identity at home
        anchors_by_row = batch.evidence.anchors[evidence_rows]  # `[B,K,3]`，m
        normal_by_row = batch.evidence.palm_normal[evidence_rows]  # `[B,3]`
        static_pair_feature = self.encoder.point_anchor_encoder.encode_per_anchor(
            home_points,
            anchors_by_row,
            normal_by_row,
            batch.anchor_valid_mask,
        )  # `[B,E,K,D_r]`，permutation-equivariant static pair condition
        if isinstance(self.reader, RelationJacobianReader):
            return self.reader(owner_latent, joint_latent, static_pair_feature, use_latent=use_latent)
        return self.reader(owner_latent, joint_latent, static_pair_feature)


def _sample_edges(spec: Any, *, points_per_edge: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    r"""每 joint 固定选择 2 active + 1 PALM zero，并分配 q-independent material identities。"""

    owners: list[int] = []
    joints: list[int] = []
    points: list[int] = []
    active_flags: list[bool] = []
    home_count = 64  # 当前 source contract 每 owner 固定 64 个 boundary material points
    for joint in range(spec.space_screws.shape[0]):
        active_owner = torch.where(spec.owner_ancestor_mask[:, joint])[0].tolist()  # 当前 joint descendants
        if not active_owner:
            raise ValueError(f"joint {joint} has no descendant owner for material-point supervision")
        selected = (active_owner[0], active_owner[-1], 0)  # proximal/distal/PALM structural zero
        for role, owner in enumerate(selected):
            for point_slot in range(points_per_edge):
                owners.append(int(owner))
                joints.append(joint)
                # Material identity 只依赖 asset-local owner/joint/role/slot，不依赖 q row。
                points.append((17 * joint + 11 * int(owner) + 7 * role + 23 * point_slot) % home_count)
                active_flags.append(role < 2)
    return (
        torch.tensor(owners, device=device, dtype=torch.long),
        torch.tensor(joints, device=device, dtype=torch.long),
        torch.tensor(points, device=device, dtype=torch.long),
        torch.tensor(active_flags, device=device, dtype=torch.bool),
    )


def _build_fixed_relation_batch(
    batch: Any,
    session: Any,
    *,
    points_per_edge: int,
) -> FixedRelationBatch:
    r"""从一次 fixed q realization 构造 variable-structure padded relation target。"""

    device = batch.q.device
    evidence_rows = batch.evidence_row_index
    if evidence_rows is None:
        evidence_rows = torch.arange(batch.q.shape[0], device=device, dtype=torch.long)
    anchor_valid_source = batch.evidence.anchor_valid_mask
    if anchor_valid_source is None:
        anchor_valid_source = torch.ones(
            batch.evidence.anchors.shape[:-1],
            device=device,
            dtype=torch.bool,
        )
    anchor_valid = anchor_valid_source[evidence_rows]  # `[B,K]`
    resident: dict[str, Any] = session.window._resident

    # 先按每项资产实际 N_J 形成 selector/target，再 pad 到当前 batch 的最大 E/K。
    row_targets: list[MaterialPointRelationJacobianTarget] = []
    row_point_indices: list[torch.Tensor] = []
    row_active: list[torch.Tensor] = []
    max_edge_count = 0
    for row, asset_id in enumerate(batch.asset_ids):
        state = resident[asset_id]
        spec = state.spec
        joint_count = spec.space_screws.shape[0]
        q = batch.q[row : row + 1, :joint_count]
        owner_index, joint_index, point_index, active = _sample_edges(
            spec,
            points_per_edge=points_per_edge,
            device=device,
        )
        local_home = torch.as_tensor(
            state.source.home_surface.points_owner_local_m,
            device=device,
            dtype=q.dtype,
        )  # `[G,64,3]` owner-local fixed material bank
        local_points = local_home[owner_index, point_index]  # `[E,3]`，m
        evidence_row = int(evidence_rows[row])
        valid_anchor = anchor_valid_source[evidence_row]
        anchors = batch.evidence.anchors[evidence_row, valid_anchor]
        normal = batch.evidence.palm_normal[evidence_row]
        target = generate_material_point_relation_jacobian_targets(
            spec,
            q,
            owner_index,
            joint_index,
            local_points,
            anchors,
            normal,
        )
        row_targets.append(target)
        row_point_indices.append(point_index)
        row_active.append(active)
        max_edge_count = max(max_edge_count, owner_index.numel())

    batch_size = batch.q.shape[0]
    max_anchor_count = batch.evidence.anchors.shape[1]
    owner = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.long)
    joint = torch.zeros_like(owner)
    point = torch.zeros_like(owner)
    target_tensor = torch.zeros(batch_size, max_edge_count, max_anchor_count, 4, device=device, dtype=batch.q.dtype)
    edge_valid = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.bool)
    active_mask = torch.zeros_like(edge_valid)
    radius_valid = torch.zeros(batch_size, max_edge_count, max_anchor_count, device=device, dtype=torch.bool)
    for row, target in enumerate(row_targets):
        edge_count = target.owner_index.shape[1]
        anchor_count = target.relation_sensitivity_per_rad.shape[2]
        owner[row, :edge_count] = target.owner_index[0]
        joint[row, :edge_count] = target.joint_index[0]
        point[row, :edge_count] = row_point_indices[row]
        target_tensor[row, :edge_count, :anchor_count] = target.relation_sensitivity_per_rad[0]
        edge_valid[row, :edge_count] = True
        active_mask[row, :edge_count] = row_active[row]
        radius_valid[row, :edge_count, :anchor_count] = target.radius_valid_mask[0]
    return FixedRelationBatch(
        q=batch.q,
        evidence=batch.evidence,
        evidence_row_index=evidence_rows,
        owner_index=owner,
        joint_index=joint,
        material_point_index=point,
        target=target_tensor,
        edge_valid_mask=edge_valid,
        active_mask=active_mask,
        radius_valid_mask=radius_valid,
        anchor_valid_mask=anchor_valid,
        asset_ids=batch.asset_ids,
    )


def _loss_and_metrics(
    prediction: torch.Tensor,
    batch: FixedRelationBatch,
) -> tuple[torch.Tensor, dict[str, Any]]:
    r"""使用固定 channel scales 报告 zero-baseline skill、active sign 与 structural-zero leakage。"""

    scale = torch.tensor(CHANNEL_SCALE, device=prediction.device, dtype=prediction.dtype)  # `[4]`
    channel_valid = torch.ones_like(batch.target, dtype=torch.bool)  # height/dot/chirality 全域有效
    channel_valid[..., 1] = batch.radius_valid_mask  # radius 只在 $r_{\parallel}\ne0$ 时有效
    valid = (
        batch.edge_valid_mask[:, :, None, None]
        & batch.anchor_valid_mask[:, None, :, None]
        & channel_valid
    )  # `[B,E,K,4]` 完整 target mask
    residual = (prediction - batch.target) / scale  # 统一数值尺度后的无量纲残差
    normalized_target = batch.target / scale  # zero-predictor 对应的无量纲 target
    loss = residual.square()[valid].mean()  # 当前 2 active + 1 zero edge 自然形成 2:1 权重
    zero_baseline = normalized_target.square()[valid].mean()

    active = valid & batch.active_mask[:, :, None, None]
    structural_zero = valid & batch.edge_valid_mask[:, :, None, None] & ~batch.active_mask[:, :, None, None]
    channel_metrics: dict[str, Any] = {}
    channel_names = ("height", "radius", "dot", "chirality")
    for channel, name in enumerate(channel_names):
        channel_mask = active[..., channel]
        channel_error = prediction[..., channel] - batch.target[..., channel]
        channel_zero = batch.target[..., channel].square()[channel_mask].mean()
        channel_mse = channel_error.square()[channel_mask].mean()
        nonzero = channel_mask & (batch.target[..., channel].abs() >= 1.0e-5)
        channel_metrics[name] = {
            "active_mse": float(channel_mse.detach()),
            "active_zero_baseline": float(channel_zero.detach()),
            "active_skill": float((1.0 - channel_mse / channel_zero.clamp_min(1.0e-12)).detach()),
            "active_sign_accuracy": float(
                (torch.sign(prediction[..., channel][nonzero]) == torch.sign(batch.target[..., channel][nonzero]))
                .float()
                .mean()
                .detach()
            ),
        }
    metrics = {
        "objective": float(loss.detach()),
        "zero_baseline": float(zero_baseline.detach()),
        "skill": float((1.0 - loss / zero_baseline.clamp_min(1.0e-12)).detach()),
        "active_prediction_rms": float(prediction[active].square().mean().sqrt().detach()),
        "active_target_rms": float(batch.target[active].square().mean().sqrt().detach()),
        "structural_zero_prediction_rms": float(prediction[structural_zero].square().mean().sqrt().detach()),
        "channels": channel_metrics,
    }
    return loss, metrics


def _train_model(
    fixed_batch: FixedRelationBatch,
    *,
    updates: int,
    learning_rate: float,
    use_latent: bool,
    seed: int,
    reader_kind: str = "additive",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    r"""在同一个 fixed bank 上训练 full 或 query-only 模型，并保存稀疏学习轨迹。"""

    torch.manual_seed(seed)
    model = TinyRelationJacobianModel(reader_kind=reader_kind).to(device=fixed_batch.q.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    trajectory: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    started = perf_counter()
    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fixed_batch, use_latent=use_latent)
        loss, metrics = _loss_and_metrics(prediction, fixed_batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == updates:
            trajectory.append({"update": update, "gradient_norm": float(gradient_norm), **metrics})
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction = model(fixed_batch, use_latent=use_latent)
        _, final_metrics = _loss_and_metrics(final_prediction, fixed_batch)
        ablations: dict[str, Any] = {}
        if use_latent:
            # Same-asset q shuffle 保持 morphology 不变，只错配当前构型；每项资产的 q rows 逆序。
            rows_by_asset: dict[str, list[int]] = {}
            for row, asset_id in enumerate(fixed_batch.asset_ids):
                rows_by_asset.setdefault(asset_id, []).append(row)
            same_asset_index = torch.arange(len(fixed_batch.asset_ids), device=fixed_batch.q.device)
            for rows in rows_by_asset.values():
                same_asset_index[torch.tensor(rows, device=fixed_batch.q.device)] = torch.tensor(
                    list(reversed(rows)),
                    device=fixed_batch.q.device,
                )

            # Cross-asset shuffle 保持每项资产内部 q slot，循环错配到下一 morphology。
            groups = list(rows_by_asset.values())
            cross_asset_index = torch.empty_like(same_asset_index)
            for group_index, rows in enumerate(groups):
                source_rows = groups[(group_index + 1) % len(groups)]
                if len(source_rows) != len(rows):
                    raise ValueError("cross-asset latent shuffle requires equal q_per_asset")
                cross_asset_index[torch.tensor(rows, device=fixed_batch.q.device)] = torch.tensor(
                    source_rows,
                    device=fixed_batch.q.device,
                )
            for name, permutation in (
                ("same_asset_q_shuffle", same_asset_index),
                ("cross_asset_shuffle", cross_asset_index),
            ):
                ablated_prediction = model(
                    fixed_batch,
                    use_latent=True,
                    latent_row_index=permutation,
                )
                _, ablations[name] = _loss_and_metrics(ablated_prediction, fixed_batch)
    report = {
        "use_latent": use_latent,
        "reader_kind": reader_kind,
        "updates": updates,
        "learning_rate": learning_rate,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "final": final_metrics,
        "ablations": ablations,
        "trajectory": trajectory,
    }
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return report, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-overfit anchor-relational Material-point Jacobian target.")
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--q-per-asset", type=int, default=2)
    parser.add_argument("--points-per-edge", type=int, default=1)
    parser.add_argument("--full-updates", type=int, default=500)
    parser.add_argument("--query-only-updates", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--reader", choices=("additive", "bilinear"), default="additive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-003"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.assets, args.q_per_asset, args.points_per_edge, args.full_updates, args.query_only_updates) < 1:
        raise ValueError("all count arguments must be positive")
    device = torch.device("cuda:0")
    cfg = compose_evaluation_cfg(config_ref="geometry_ssl_multitask_representation_v0_7_5")
    catalog = build_runtime(cfg.data).resolve_evaluation()
    method = build_runtime(cfg.method)
    method.configure_source_artifacts(
        root=cfg.evaluation.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="evaluation",
    )
    method.prepare(catalog, role="evaluation", device=device, dtype=torch.float32)
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset,
        device=device,
        dtype=torch.float32,
        max_resident_assets=args.assets,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        args.assets,
        q_per_asset=args.q_per_asset,
        assets_per_minibatch=args.assets,
        q_per_asset_per_minibatch=args.q_per_asset,
        max_resident_assets=args.assets,
    )
    try:
        realized = session.realize(schedule.next(), schedule=schedule, step=0)
        fixed_batch = _build_fixed_relation_batch(
            realized,
            session,
            points_per_edge=args.points_per_edge,
        )
    finally:
        session.close()
        method.close()

    # Full model 与 query-only 从独立但相同 seed 初始化；动态 Z 是唯一实验差异。
    full_report, full_state = _train_model(
        fixed_batch,
        updates=args.full_updates,
        learning_rate=args.learning_rate,
        use_latent=True,
        seed=20260830,
        reader_kind=args.reader,
    )
    query_report, query_state = _train_model(
        fixed_batch,
        updates=args.query_only_updates,
        learning_rate=args.learning_rate,
        use_latent=False,
        seed=20260830,
        reader_kind=args.reader,
    )
    report = {
        "case": "AR-MPJ-003",
        "population": {
            "assets": args.assets,
            "q_per_asset": args.q_per_asset,
            "rows": fixed_batch.q.shape[0],
            "max_edges": fixed_batch.owner_index.shape[1],
            "valid_edges": int(fixed_batch.edge_valid_mask.sum()),
            "active_edges": int((fixed_batch.edge_valid_mask & fixed_batch.active_mask).sum()),
            "structural_zero_edges": int((fixed_batch.edge_valid_mask & ~fixed_batch.active_mask).sum()),
            "max_anchors": fixed_batch.target.shape[2],
            "valid_anchor_pairs": int(
                (
                    fixed_batch.edge_valid_mask[:, :, None]
                    & fixed_batch.anchor_valid_mask[:, None, :]
                ).sum()
            ),
        },
        "model": {
            "encoder_hidden_width": 64,
            "encoder_layers": 2,
            "relation_width": 32,
            "channel_scale": list(CHANNEL_SCALE),
            "reader_kind": args.reader,
        },
        "full": full_report,
        "query_only": query_report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "schema": "armj-tiny-overfit-v1",
            "report": report,
            "full_state": full_state,
            "query_only_state": query_state,
        },
        args.output_dir / "models.pt",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
