r"""几何 SSL 的 TensorBoard、JSONL 与 dense NPZ 同步记录器。

三种产物承担不同证据角色：TensorBoard 服务在线趋势；JSONL 保存逐 step 标量与资产路由；NPZ 保存
``[B,G,N_Q,L]`` density error、``[B,E]`` κ error、latent 和全部 padding/target masks。任何被 mask
排除的样本仍留在 NPZ，避免“损失没看见”被误解释为“数据中不存在”。
"""

from __future__ import annotations

import json  # JSONL 使用标准 JSON，保证每行可独立审计
from pathlib import Path  # 运行目录由 trainer 显式传入
from typing import Any  # 标量记录包含字符串、列表与 float

import numpy as np  # dense latent/mask/error 使用压缩 NPZ
from torch.utils.tensorboard import SummaryWriter  # 在线曲线不替代 JSONL 事实源

from anymani.distill.models.geometry_ssl import GeometrySSLForward  # latent/density/κ 预测包
from anymani.distill.objectives.representations.field_reconstruction import GeometrySSLTerms  # 五项损失包
from anymani.distill.ssl.dataset import PaddedOnlineGeometryBatch  # target、mask 与资产身份


class GeometrySSLRunLogger:
    r"""同时保存在线曲线、逐步可审计记录与定期稠密误差快照。

    logger 不做均值重算、不改 loss 权重，也不决定 checkpoint 优劣；它只记录 trainer 已计算的事实。
    train 与 validation 通过 ``split`` 命名空间隔离，official evaluation 由独立冻结后流程记录。
    """

    def __init__(self, output_dir: Path) -> None:
        r"""创建输出目录、TensorBoard writer 和 append-only JSONL。

        Args:
            output_dir (Path): 当前 resolved experiment 的唯一运行目录。
        """

        output_dir.mkdir(parents=True, exist_ok=True)  # 只创建本次实验目录，不扫描其他 runs
        self.output_dir = output_dir  # NPZ 与结构化记录共同根
        self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))  # event 文件
        self.jsonl_path = output_dir / "metrics.jsonl"  # append-only 标量证据

    def log_terms(
        self,
        *,
        step: int,  # optimizer step，从 1 开始
        split: str,  # `train` 或 `validation`
        terms: GeometrySSLTerms,  # 当前 batch 的五项标量损失
        asset_ids: tuple[str, ...],  # `[B]` 路由身份
        gradient_norm: float | None = None,  # train-only clip 前总范数
    ) -> None:
        r"""记录五项损失、总损失、资产路由和可选共享参数梯度范数。

        ``density`` 无量纲；``kappa`` 的误差来自 m/rad；``derived_field``、``sobolev`` 与 ``chain``
        来自 1/rad 场灵敏度。loss 已在 objective 中平方并对有效标量归一化，因此这里统一记录标量，
        但不宣称不同物理项可直接比较绝对大小。
        """

        scalars = {  # 六个量保持独立键，禁止只存 total 隐藏失效分支
            "total": float(terms.total.detach()),  # 加权联合标量
            "density": float(terms.density.detach()),  # 多带宽零阶 MSE
            "kappa": float(terms.kappa.detach()),  # sampled distance sensitivity MSE
            "derived_field": float(terms.derived_field.detach()),  # chain-rule 显式路径 MSE
            "sobolev": float(terms.sobolev.detach()),  # density 对物理 q 自导数 MSE
            "chain": float(terms.chain.detach()),  # 两条预测灵敏度路径一致性 MSE
        }
        for name, value in scalars.items():  # TensorBoard 命名与 JSONL 字段保持同构
            self.writer.add_scalar(f"{split}/{name}", value, step)  # 横轴固定 optimizer step
        if gradient_norm is not None:  # validation 不反向，因此该字段为空
            self.writer.add_scalar(f"{split}/gradient_norm", gradient_norm, step)  # clip 前 L2 总范数
        record: dict[str, Any] = {  # 一行完整保存本次指标与资产路由
            "step": step,
            "split": split,
            "asset_ids": list(asset_ids),
            **scalars,
        }
        if gradient_norm is not None:  # train record 才写入梯度证据
            record["gradient_norm"] = gradient_norm  # 与 TensorBoard 数值相同
        with self.jsonl_path.open("a", encoding="utf-8") as stream:  # 不覆盖此前 step
            stream.write(json.dumps(record, sort_keys=True) + "\n")  # 每条记录单行、可流式恢复

    def save_dense_snapshot(
        self,
        *,
        step: int,  # snapshot 对应 optimizer step
        split: str,  # train/validation 文件名前缀
        prediction: GeometrySSLForward,  # 同一 batch 的模型输出
        batch: PaddedOnlineGeometryBatch,  # target/mask/asset identity
    ) -> Path:
        r"""保存 latent/mask/error arrays，供 post-hoc owner/JOINT/带宽诊断。

        Returns:
            Path: 写入的 ``<split>_dense_step_<step>.npz`` 路径。

        Raises:
            ValueError: batch 不是 padding batch、缺 entity/joint masks 时抛出。
        """

        entity_valid = batch.evidence.entity_valid_mask  # `[B,26]`，真实 owner 槽
        joint_valid = batch.evidence.joint_valid_mask  # `[B,20]`，真实 JOINT 槽
        if entity_valid is None or joint_valid is None:  # 本记录 schema 明确要求跨结构 mask
            raise ValueError("dense SSL snapshot requires padded entity/joint validity masks")
        path = self.output_dir / f"{split}_dense_step_{step:08d}.npz"  # step 稳定、可排序
        np.savez_compressed(  # dense arrays 较大，使用无损压缩保留逐元素误差
            path,  # 输出文件
            asset_ids=np.asarray(batch.asset_ids),  # `[B]` Unicode asset IDs
            zero_order=prediction.latents.zero_order.detach().cpu().numpy(),  # `[B,26,D_0]`
            first_order=prediction.latents.first_order.detach().cpu().numpy(),  # `[B,20,D_1]`
            entity_valid_mask=entity_valid.detach().cpu().numpy(),  # `[B,26]` bool
            joint_valid_mask=joint_valid.detach().cpu().numpy(),  # `[B,20]` bool
            field_valid_mask=batch.field_targets.valid_mask.detach().cpu().numpy(),  # `[B,26,N_Q]`
            edge_valid_mask=batch.sensitivity_targets.valid_mask.detach().cpu().numpy(),  # `[B,E]`
            density_error=(prediction.density - batch.field_targets.density).detach().cpu().numpy(),  # 无量纲
            kappa_error=(prediction.kappa - batch.sensitivity_targets.kappa).detach().cpu().numpy(),  # m/rad
        )
        return path  # 调用者可记录或检查该 artifact

    def close(self) -> None:
        r"""flush 并关闭 TensorBoard event writer；JSONL/NPZ 每次写入已自行关闭。"""

        self.writer.flush()  # 把进程内 event buffer 推到磁盘
        self.writer.close()  # 释放 event file handle


__all__ = ["GeometrySSLRunLogger"]  # 记录层唯一公开入口
