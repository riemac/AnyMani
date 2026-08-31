r"""几何 SSL 的 TensorBoard、JSONL 与 dense NPZ 同步记录器。

三种产物承担不同证据角色：TensorBoard 服务在线趋势；JSONL 保存逐 update 标量与预算坐标；NPZ 保存
``[B,G,N_Q,N_sigma]`` density、``[B,E]`` κ 的 prediction/target、统一 $Z$ 和全部 masks/selectors。任何被 mask
排除的样本仍留在 NPZ，避免“损失没看见”被误解释为“数据中不存在”。
"""

from __future__ import annotations

import json  # JSONL 使用标准 JSON，保证每行可独立审计
from collections.abc import Mapping
from pathlib import Path  # 运行目录由 trainer 显式传入
from typing import Any  # 标量记录包含字符串、列表与 float

import numpy as np  # dense latent/mask/error 使用压缩 NPZ
from torch.utils.tensorboard import SummaryWriter  # 在线曲线不替代 JSONL 事实源

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    PaddedOnlineGeometryBatch,  # target、mask 与资产身份
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward  # latent/density/κ 预测包


class GeometrySSLRunLogger:
    r"""同时保存在线曲线、逐步可审计记录与定期稠密误差快照。

    logger 不做均值重算、不改 loss 权重，也不决定 checkpoint 优劣；它只记录 trainer 已计算的事实。
    train 与 validation 通过 ``split`` 命名空间隔离，official evaluation 由独立冻结后流程记录。
    """

    def __init__(self, output_dir: Path, *, purge_step: int | None = None) -> None:
        r"""创建输出目录、TensorBoard writer 和 append-only JSONL。

        Args:
            output_dir (Path): 当前 resolved experiment 的唯一运行目录。
        """

        output_dir.mkdir(parents=True, exist_ok=True)  # 只创建本次实验目录，不扫描其他 runs
        self.output_dir = output_dir  # NPZ 与结构化记录共同根
        self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), purge_step=purge_step)  # event 文件
        self.jsonl_path = output_dir / "metrics.jsonl"  # append-only 标量证据
        self.runtime_jsonl_path = output_dir / "runtime.jsonl"  # window/memory/throughput 生命周期证据

    def continuation_offsets(self) -> dict[str, int]:
        r"""刷新 TensorBoard，并返回两个 JSONL 的 epoch-transaction byte offsets。"""

        self.writer.flush()
        return {
            "metrics_jsonl_bytes": self.jsonl_path.stat().st_size if self.jsonl_path.is_file() else 0,
            "runtime_jsonl_bytes": self.runtime_jsonl_path.stat().st_size if self.runtime_jsonl_path.is_file() else 0,
        }

    def restore_continuation(self, offsets: Mapping[str, int], *, purge_step: int) -> None:
        r"""把 JSONL 截到 recovery epoch barrier，并以 TensorBoard purge step 重开 writer。"""

        for name, path in (("metrics_jsonl_bytes", self.jsonl_path), ("runtime_jsonl_bytes", self.runtime_jsonl_path)):
            raw_offset = offsets.get(name)
            if not isinstance(raw_offset, int) or raw_offset < 0:
                raise ValueError(f"recovery logger offset {name!r} must be a non-negative integer")
            current_size = path.stat().st_size if path.is_file() else 0
            if current_size < raw_offset:
                raise ValueError(f"recovery logger file {path} is shorter than checkpoint offset")
            if path.is_file():
                with path.open("r+b") as stream:
                    stream.truncate(raw_offset)
        self.writer.close()
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"), purge_step=purge_step)

    def log_runtime_event(self, event: dict[str, Any]) -> None:
        r"""追加一条 resident-window 或 optimizer-update 运行时事件。

        runtime 事实与物理 loss 分文件保存，避免内存、吞吐或 lease 生命周期被误作训练指标。
        ``device_memory_before/after`` 是设备全局 free-memory 口径，``torch_*`` 只覆盖 PyTorch
        caching allocator；前者包含 Warp BVH 与其他 CUDA 分配，不能解释成 BVH 独占字节数。

        Args:
            event (dict[str, Any]): 仅含 JSON 基础类型的已命名运行时事件。
        """

        with self.runtime_jsonl_path.open("a", encoding="utf-8") as stream:  # append-only 生命周期序列
            stream.write(json.dumps(event, sort_keys=True) + "\n")  # 每行可独立恢复和审计

        if event.get("event") == "optimizer_update":  # 在线趋势只记录 update-level 连续标量
            optimizer_update = int(event["optimizer_update"])
            for name in (
                "step_seconds",
                "q_samples_per_second",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
            ):
                value = event.get(name)  # CPU/不可用口径允许显式 None
                if value is not None:
                    self.writer.add_scalar(f"runtime/{name}", float(value), optimizer_update)

    def log_terms(
        self,
        *,
        optimizer_update: int,
        epoch: int,
        mini_epoch: int,
        minibatch_in_epoch: int,
        global_minibatch: int,  # 新 teacher 数据身份；跨 mini-epoch 复用时保持不变
        new_pairs_seen: int,
        pair_uses: int,
        teacher_pairs_realized: int,
        microbatches_consumed: int,
        wall_time_seconds: float,
        split: str,  # `train` 或 `validation`
        terms: dict[str, float],  # 当前 update 的 rho/kappa raw $(asset,q)$ 等权 MSE
        denominators: dict[str, float],
        asset_ids: tuple[str, ...],  # `[B]` 路由身份
        gradient_groups: dict[str, dict[str, Any]] | None = None,  # 三组独立 clip 前/后范数
        batch: PaddedOnlineGeometryBatch | None = None,  # q cursor provenance
        gradient_evidence: dict[str, float] | None = None,  # FairGrad 与每 4 epochs unified-Z proxy
        diagnostic_seconds: float = 0.0,  # 当前 update 的 proxy wall time；非 cadence 为 0
        diagnostics: dict[str, float] | None = None,  # active/zero、RMS 与 valid ratio
    ) -> None:
        r"""记录训练时可知的分任务 objective、资产路由、FairGrad 与分组裁剪事实。

        ``density`` 是原始无量纲 MSE；``kappa`` 是物理残差除以 0.1 m/rad 后的无量纲 MSE。完整
        teacher baseline 尚未闭合，因此本函数不写 normalized loss 或 skill；这些字段只进入独立的
        ``metrics_finalized.jsonl``。
        """

        scalars = {f"raw/{name}": value for name, value in terms.items()}
        for name, value in scalars.items():  # TensorBoard 命名与 JSONL 字段保持同构
            self.writer.add_scalar(f"{split}_update/{name}", value, optimizer_update)
        for group_name, group in (gradient_groups or {}).items():
            for field_name in ("pre_clip_norm", "post_clip_norm", "clip_ratio"):
                self.writer.add_scalar(
                    f"{split}_gradient/{group_name}/{field_name}",
                    float(group[field_name]),
                    optimizer_update,
                )
        for name, value in (gradient_evidence or {}).items():
            self.writer.add_scalar(f"{split}_z_gradient/{name}", value, optimizer_update)
        for name, value in (diagnostics or {}).items():
            self.writer.add_scalar(f"{split}_diagnostic/{name}", value, optimizer_update)
        if diagnostic_seconds > 0.0:
            self.writer.add_scalar(f"{split}_z_gradient/diagnostic_seconds", diagnostic_seconds, optimizer_update)
        record: dict[str, Any] = {  # 一行完整保存本次指标与资产路由
            "epoch": epoch,
            "mini_epoch": mini_epoch,
            "minibatch_in_epoch": minibatch_in_epoch,
            "global_minibatch": global_minibatch,
            "minibatch_reuse_identity": [global_minibatch, mini_epoch],
            "optimizer_update": optimizer_update,
            "new_pairs_seen": new_pairs_seen,
            "pair_uses": pair_uses,
            "teacher_pairs_realized": teacher_pairs_realized,
            "microbatches_consumed": microbatches_consumed,
            "wall_time_seconds": wall_time_seconds,
            "denominators": dict(denominators),
            "split": split,
            "asset_ids": list(asset_ids),
            "q_index": (
                batch.q_index.detach().cpu().tolist() if batch is not None and batch.q_index is not None else None
            ),
            **scalars,
        }
        if gradient_groups:
            record["gradient_groups"] = gradient_groups
        if gradient_evidence:
            record["z_gradient_evidence"] = dict(gradient_evidence)  # JSONL 保存与 TensorBoard 同源值
            record["z_gradient_diagnostic_seconds"] = diagnostic_seconds
        if diagnostics:
            record["diagnostics"] = dict(diagnostics)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:  # 不覆盖此前 update
            stream.write(json.dumps(record, sort_keys=True) + "\n")  # 每条记录单行、可流式恢复

    def finalize_training_metrics(
        self,
        *,
        teacher_baselines: Mapping[str, object],
        expected_optimizer_updates: int,
        lineage_metrics_path: Path | None = None,
    ) -> Path:
        r"""用完整 run teacher distribution 重算固定 normalized error 与 skill。

        原始 ``metrics.jsonl`` 保持字节不变。resume 时可把前序 run 的 JSONL 作为 lineage prefix；相同
        update 重复、缺失或超出 ``1..expected_optimizer_updates`` 都 fail closed。
        """

        baselines: dict[str, float] = {}
        for name, record in teacher_baselines.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"final teacher baseline lacks mapping for {name}")
            value = record.get("baseline_mse")
            if not isinstance(value, (float, int)) or float(value) <= 0.0:
                raise ValueError(f"final teacher baseline {name}.baseline_mse must be positive")
            baselines[name] = float(value)

        if not baselines:
            raise ValueError("final teacher baselines must contain at least one objective mapping")
        sources = []
        if lineage_metrics_path is not None and lineage_metrics_path.resolve() != self.jsonl_path.resolve():
            sources.append(lineage_metrics_path)
        sources.append(self.jsonl_path)
        by_update: dict[int, dict[str, Any]] = {}
        for source in sources:
            if not source.is_file():
                raise ValueError(f"training metric lineage file does not exist: {source}")
            for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                record = json.loads(line)
                update = record.get("optimizer_update")
                if not isinstance(update, int):
                    raise ValueError(f"metric record {source}:{line_number} lacks integer optimizer_update")
                if update in by_update:
                    raise ValueError(f"duplicate optimizer_update={update} in training metric lineage")
                by_update[update] = record
        expected = set(range(1, expected_optimizer_updates + 1))
        if set(by_update) != expected:
            missing = sorted(expected - set(by_update))
            unexpected = sorted(set(by_update) - expected)
            raise ValueError(
                f"training metric lineage is not a complete update prefix: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )

        output = self.output_dir / "metrics_finalized.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for update in range(1, expected_optimizer_updates + 1):
                raw = by_update[update]
                missing_raw = [name for name in baselines if f"raw/{name}" not in raw]
                if missing_raw:
                    raise ValueError(f"metric record optimizer_update={update} lacks raw terms={missing_raw}")
                normalized = {
                    name: float(raw[f"raw/{name}"]) / baselines[name]
                    for name in baselines
                }
                finalized = {
                    **raw,
                    "final_teacher_baselines": dict(baselines),
                    **{f"normalized/{name}": value for name, value in normalized.items()},
                    **{f"skill/{name}": 1.0 - value for name, value in normalized.items()},
                }
                stream.write(json.dumps(finalized, sort_keys=True) + "\n")
                for name, value in normalized.items():
                    self.writer.add_scalar(f"train_finalized/normalized/{name}", value, update)
                    self.writer.add_scalar(f"train_finalized/skill/{name}", 1.0 - value, update)
        temporary.replace(output)
        return output

    def log_epoch_terms(
        self,
        *,
        epoch: int,
        new_pairs_seen: int,
        pair_uses: int,
        optimizer_updates: int,
        terms: dict[str, float],
    ) -> None:
        r"""按新 asset-configuration pair 数记录 epoch 聚合曲线。"""

        for name, value in terms.items():
            self.writer.add_scalar(f"train_epoch/{name}", value, new_pairs_seen)
        self.writer.add_scalar("progress/epoch", epoch, new_pairs_seen)
        self.writer.add_scalar("progress/pair_uses", pair_uses, new_pairs_seen)
        self.writer.add_scalar("progress/optimizer_updates", optimizer_updates, new_pairs_seen)

    def log_validation_metrics(
        self,
        *,
        epoch: int,
        new_pairs_seen: int,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        r"""以训练新 pair 数为横轴记录具名 validation suite 指标。"""

        for suite, suite_metrics in metrics.items():
            for name, value in suite_metrics.items():
                self.writer.add_scalar(f"validation/{suite}/{name}", value, new_pairs_seen)
        self.writer.add_scalar("validation/epoch", epoch, new_pairs_seen)

    def save_dense_snapshot(
        self,
        *,
        optimizer_update: int,  # snapshot 对应参数更新序号
        split: str,  # train/validation 文件名前缀
        prediction: GeometrySSLForward,  # 同一 batch 的模型输出
        batch: PaddedOnlineGeometryBatch,  # target/mask/asset identity
    ) -> Path:
        r"""保存 latent/mask/error arrays，供 post-hoc owner/JOINT/带宽诊断。

        Returns:
            Path: 写入的 ``<split>_dense_update_<optimizer_update>.npz`` 路径。

        Raises:
            ValueError: batch 不是 padding batch、缺 entity/joint masks 时抛出。
        """

        entity_valid = batch.evidence.entity_valid_mask  # `[B,26]`，真实 owner 槽
        joint_valid = batch.evidence.joint_valid_mask  # `[B,20]`，真实 JOINT 槽
        if entity_valid is None or joint_valid is None:  # 本记录 schema 明确要求跨结构 mask
            raise ValueError("dense SSL snapshot requires padded entity/joint validity masks")
        evidence_row_index = batch.evidence_row_index
        if evidence_row_index is not None:
            entity_valid = entity_valid[evidence_row_index]
            joint_valid = joint_valid[evidence_row_index]
            joint_entity_index = batch.evidence.joint_entity_index[evidence_row_index]
        else:
            joint_entity_index = batch.evidence.joint_entity_index
        path = self.output_dir / f"{split}_dense_update_{optimizer_update:08d}.npz"
        np.savez_compressed(  # dense arrays 较大，使用无损压缩保留逐元素误差
            path,  # 输出文件
            asset_ids=np.asarray(batch.asset_ids),  # `[B]` Unicode asset IDs
            q_index=(
                batch.q_index.detach().cpu().numpy()
                if batch.q_index is not None
                else np.full(len(batch.asset_ids), -1, dtype=np.int64)
            ),  # `[B]` asset-local Sobol absolute cursor
            q=batch.q.detach().cpu().numpy(),  # `[B,20]` physical rad，配合 joint mask 解释
            query_points_h=batch.queries.query_points_h.detach().cpu().numpy(),  # `[B,26,N_Q,3]`，m
            query_stratum=batch.queries.query_stratum.detach().cpu().numpy(),  # 0/1/2 的 50:25:25 provenance
            adjacent_owner_index=batch.queries.adjacent_owner_index.detach().cpu().numpy(),  # adjacent routing
            workspace_anchor_index=batch.queries.workspace_anchor_index.detach().cpu().numpy(),  # anchor routing
            owner_role=batch.field_targets.owner_role.detach().cpu().numpy(),  # PALM/JOINT/TIP 分层轴
            bandwidths_m=batch.field_targets.bandwidths.detach().cpu().numpy(),  # `[B,N_σ]` actual sigma
            entities=prediction.latents.entities.detach().cpu().numpy(),  # `[B,26,D]` unified $Z$
            entity_valid_mask=entity_valid.detach().cpu().numpy(),  # `[B,26]` bool
            joint_valid_mask=joint_valid.detach().cpu().numpy(),  # `[B,20]` bool
            evidence_row_index=(
                evidence_row_index.detach().cpu().numpy()
                if evidence_row_index is not None
                else np.arange(len(batch.asset_ids), dtype=np.int64)
            ),
            anchor_index=(
                batch.anchor_index.detach().cpu().numpy()
                if batch.anchor_index is not None
                else np.zeros(len(batch.asset_ids), dtype=np.int64)
            ),
            field_valid_mask=batch.field_targets.valid_mask.detach().cpu().numpy(),  # `[B,26,N_Q]`
            edge_valid_mask=batch.sensitivity_targets.valid_mask.detach().cpu().numpy(),  # `[B,E]`
            ancestor_mask=batch.sensitivity_targets.ancestor_mask.detach().cpu().numpy(),  # 祖先/非祖先
            active_mask=batch.sensitivity_targets.active_mask.detach().cpu().numpy(),  # active/structural-zero
            edge_owner_index=batch.sensitivity_targets.owner_index.detach().cpu().numpy(),  # sampled owner
            edge_query_index=batch.sensitivity_targets.query_index.detach().cpu().numpy(),  # sampled query
            edge_joint_index=batch.sensitivity_targets.joint_index.detach().cpu().numpy(),  # sampled JOINT
            joint_entity_index=joint_entity_index.detach().cpu().numpy(),  # `[B,N_J]` JOINT view routing
            closest_point_h_m=batch.sensitivity_targets.closest_point.detach().cpu().numpy(),  # `[B,E,3]`
            closest_source=batch.sensitivity_targets.closest_source.detach().cpu().numpy(),  # owner/face provenance
            uniqueness_margin_m=batch.sensitivity_targets.uniqueness_margin.detach().cpu().numpy(),  # smooth mask 证据
            distance_m=batch.field_targets.distance.detach().cpu().numpy(),  # distance shell 分层真值
            density_prediction=prediction.density.detach().cpu().numpy(),  # `[B,G,N_Q,N_sigma]`
            density_target=batch.field_targets.density.detach().cpu().numpy(),  # teacher density
            kappa_prediction=prediction.kappa.detach().cpu().numpy(),  # `[B,E]`，m/rad
            kappa_target=batch.sensitivity_targets.kappa.detach().cpu().numpy(),  # teacher κ，m/rad
            density_error=(prediction.density - batch.field_targets.density).detach().cpu().numpy(),  # 无量纲
            kappa_error=(prediction.kappa - batch.sensitivity_targets.kappa).detach().cpu().numpy(),  # m/rad
            central_difference=(
                batch.sensitivity_targets.central_difference.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference is not None
                else np.zeros_like(batch.sensitivity_targets.kappa.detach().cpu().numpy())
            ),
            central_difference_valid_mask=(
                batch.sensitivity_targets.central_difference_valid_mask.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_valid_mask is not None
                else np.zeros_like(batch.sensitivity_targets.valid_mask.detach().cpu().numpy())
            ),
            central_difference_plus_face=(
                batch.sensitivity_targets.central_difference_plus_face.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_plus_face is not None
                else np.full_like(batch.sensitivity_targets.closest_source.detach().cpu().numpy(), -1)
            ),
            central_difference_minus_face=(
                batch.sensitivity_targets.central_difference_minus_face.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_minus_face is not None
                else np.full_like(batch.sensitivity_targets.closest_source.detach().cpu().numpy(), -1)
            ),
        )
        return path  # 调用者可记录或检查该 artifact

    def close(self) -> None:
        r"""flush 并关闭 TensorBoard event writer；JSONL/NPZ 每次写入已自行关闭。"""

        self.writer.flush()  # 把进程内 event buffer 推到磁盘
        self.writer.close()  # 释放 event file handle


__all__ = ["GeometrySSLRunLogger"]  # 记录层唯一公开入口
