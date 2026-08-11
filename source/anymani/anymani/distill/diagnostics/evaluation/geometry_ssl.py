r"""几何 SSL 的 query-only 与 latent-shuffle 必要性诊断。

完整模型预测写作 $f(x,q,s)$：$x$ 是固定 `{h}` query，$q$ 是当前物理关节角，$s$ 是静态手型证据。
两项诊断共享同一 query encoder 和 disposable decoder，仅改变 morphology latent：

$$
f_{query}(x)=f(x,\mathbf 0,\mathbf 0),
\qquad
f_{shuffle}^{(b)}(x)=f(x,z_{\pi(b)}^{(0)},z_{\pi(b)}^{(1)}).
$$

若完整模型不优于 query-only，表示 decoder 可能绕过 hand conditioning；若 latent shuffle 不显著恶化，
表示 latent 与本样本几何未建立可辨识对应。两者都是必要性诊断，不替代 held-out geometry 指标。
"""

from __future__ import annotations

from typing import Literal  # 只允许两项预注册 ablation，避免自由字符串改变实验含义

import torch  # latent、query 与 batch permutation 全部保持 PyTorch 计算图

from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel  # 完整 SSL 组装边界
from anymani.distill.models.input_adapters.geometry import (  # retained latent/evidence 类型合同
    GeometryLatents,
    StaticGeometryEvidence,
)

GeometrySSLAblation = Literal["query_only", "latent_shuffle"]  # 受控诊断枚举


def geometry_ssl_ablation_forward(
    model: GeometrySSLModel,  # 同一组 frozen/evaluated 参数，不能为 baseline 另训 decoder
    q: torch.Tensor,  # `[B,N_J]`，rad
    evidence: StaticGeometryEvidence,  # 同结构或 padding 后跨结构静态证据
    query_points_h: torch.Tensor,  # `[B,G,N_Q,3]`，`{h}`，m
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

    latents = model.encoder(q, evidence)  # 原始 $z^{(0)}:[B,G,D_0]$ 与 $z^{(1)}:[B,N_J,D_1]$
    query_features = model.encoder.encode_points(  # 与完整模型完全相同的 point-anchor 前端
        query_points_h.detach(), evidence
    )  # `[B,G,N_Q,D_q]`；固定 query 不接收 sampler 梯度
    entity_valid = evidence.entity_valid_mask  # `[B,G]` 或 None；padding 不是可学习实体
    if entity_valid is not None:  # 只在跨结构稠密容器中需要显式 owner mask
        if entity_valid.ndim == 1:  # 单结构共享 mask 扩成 batch 视图，不复制物理证据
            entity_valid = entity_valid.unsqueeze(0).expand(q.shape[0], -1)  # `[B,G]`
        query_features = query_features * entity_valid.unsqueeze(-1).unsqueeze(-1)  # invalid owner 精确零

    if ablation == "query_only":  # 保留 query path，但删除全部 hand/q conditioning
        ablated = GeometryLatents(  # 形状不变，避免 baseline 获得不同 decoder 容量
            torch.zeros_like(latents.zero_order),  # $z^{(0)}\leftarrow0$
            torch.zeros_like(latents.first_order),  # $z^{(1)}\leftarrow0$
        )
    elif ablation == "latent_shuffle":  # 保留 latent 边缘分布，只破坏样本对应关系
        if batch_permutation is None or batch_permutation.shape != (q.shape[0],):  # 必须显式 `[B]`
            raise ValueError("latent_shuffle requires batch_permutation with shape [B]")  # 不猜 permutation
        expected = torch.arange(q.shape[0], device=batch_permutation.device)  # 合法索引集合 `[0,B)`
        if not torch.equal(torch.sort(batch_permutation).values, expected):  # 拒绝重复/遗漏样本
            raise ValueError("batch_permutation must be a bijection of [0,B)")  # 保持 batch 分布不变
        ablated = GeometryLatents(  # 同一 permutation 同时作用零阶/一阶 latent
            latents.zero_order.index_select(0, batch_permutation),  # $z_b^{(0)}\leftarrow z_{\pi(b)}^{(0)}$
            latents.first_order.index_select(0, batch_permutation),  # $z_b^{(1)}\leftarrow z_{\pi(b)}^{(1)}$
        )
    else:  # 未注册诊断不得静默退回完整 forward
        raise ValueError(f"unknown geometry SSL ablation={ablation!r}")
    return model.decode_latents(  # decoder 权重、query features 与 selectors 均与完整模型相同
        ablated,  # 唯一被干预的变量
        query_features,  # 未打乱的本样本 query path
        entity_valid_mask=entity_valid,  # invalid owner prediction 继续严格清零
        owner_index=owner_index,  # 保留本样本 sampled owner
        query_index=query_index,  # 保留本样本 sampled query
        joint_index=joint_index,  # 保留本样本 sampled JOINT
    )


__all__ = ["GeometrySSLAblation", "geometry_ssl_ablation_forward"]  # 稳定诊断公开面
