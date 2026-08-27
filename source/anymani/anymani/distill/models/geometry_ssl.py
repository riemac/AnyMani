r"""多锚点条件隐式几何 SSL 的 retained/disposable 模型组装。

该模块只组装已经各自拥有清晰职责的模型组件：

```text
retained after SSL
  StaticGeometryEvidence + physical q
    -> ImplicitGeometryEncoder
    -> unified Z [B,G,128]

training-only disposable
  shared point-anchor query features
    + owner Z + explicit sigma -> ConditionalDensityDecoder -> rho [B,G,N_Q,N_sigma]
    + owner/JOINT Z -> DistanceSensitivityDecoder -> kappa [B,E]
```

query encoder 与 home-surface encoder 复用 ``ImplicitGeometryEncoder.point_anchor_encoder``，避免 decoder
拥有第二套坐标系统。query stratum、最近点、distance、Jacobian 和 teacher labels 不在 forward 参数中；
它们只进入 objective。导出 retained-only checkpoint 时只保存 ``encoder``，两个 decoder 整体删除。
"""

from __future__ import annotations

from dataclasses import dataclass, field  # 模型配置与前向结果均冻结为类型化合同

import torch  # retained/disposable 张量与 state_dict
from torch import nn  # 模型组装基类

from .decoders.representations.implicit_field import (  # SSL-only readers
    ConditionalDensityDecoder,  # $\hat\rho_{g,\ell}(x;q)$
    DistanceSensitivityDecoder,  # $\hat\kappa_{g,i}(x;q)$
    GeometrySSLDecoderCfg,  # 两个训练期 readers 的公开容量配置
)
from .input_adapters.geometry import (  # 部署期 retained 路径
    GeometryEncoderCfg,  # frontend 与 unified graph-biased backbone
    GeometryLatents,  # 单一 PALM/JOINT/TIP typed $Z$
    ImplicitGeometryEncoder,  # task-free hand conditioning encoder
    StaticGeometryEvidence,  # anchors/home/screws/graph/masks
)


@dataclass(frozen=True)
class GeometrySSLModelCfg:
    r"""retained encoder 与 disposable decoder 的显式组装配置。

    decoder 输入宽度只从 encoder backbone 派生，禁止训练配置单独复制 $D/D_q$ 后发生漂移。
    sigma 数量属于 target 的动态采样轴；model 只保存对数比例的参考长度，不保存固定输出通道数。
    """

    encoder: GeometryEncoderCfg = field(default_factory=GeometryEncoderCfg)  # retained 参数与输出类型
    ssl_decoders: GeometrySSLDecoderCfg = field(default_factory=GeometrySSLDecoderCfg)  # disposable readers


@dataclass(frozen=True)
class GeometrySSLForward:
    r"""一次模型前向中保留表征与训练期预测的类型化结果。

    ``latents`` 是 SSL 后迁入 PPO 的唯一 learned state；``query_features``、density 与 κ 都属于训练/诊断
    生命周期。结果对象同时供 objective、NPZ logger 和受控 ablation 使用。
    """

    latents: GeometryLatents  # retained unified $Z$
    query_features: torch.Tensor  # `[B,G,N_Q,D_q]`，共享点—锚点前端
    density: torch.Tensor  # `[B,G,N_Q,N_sigma]`
    kappa: torch.Tensor  # `[B,E]`


class GeometrySSLModel(nn.Module):
    r"""统一 encoder/query/decoder 调用，但保持 checkpoint 生命周期可分离。

    模型不读取 distance、closest point、Jacobian、query stratum、joint limits、object 或 task state。
    物理 q 与 query 是模型条件；rho/kappa 两项目标对 encoder 和两个 decoder 的参数建立梯度图。
    """

    def __init__(self, config: GeometrySSLModelCfg = GeometrySSLModelCfg()) -> None:
        r"""构造 retained encoder 和两个 disposable decoder。

        Args:
            config (GeometrySSLModelCfg): retained encoder 与训练期 readers 的显式配置。
        """

        super().__init__()  # 注册 PyTorch parameter/module 生命周期
        self.config = config  # resolved config 应随完整 checkpoint 保存
        self.encoder = ImplicitGeometryEncoder(config.encoder)  # SSL 后迁入 PPO
        entity_width = config.encoder.backbone.hidden_width  # final-norm entity token 宽度 $D$
        query_width = config.encoder.frontend.relation_width
        self.density_decoder = ConditionalDensityDecoder(
            config.ssl_decoders.density,
            entity_width=entity_width,
            query_width=query_width,
        )  # SSL-only FiLM density reader
        self.sensitivity_decoder = DistanceSensitivityDecoder(
            config.ssl_decoders.sensitivity,
            entity_width=entity_width,
            query_width=query_width,
        )  # SSL-only query-main dual-token FiLM sensitivity reader

    def forward(
        self,
        q: torch.Tensor,  # `[B,N_J]` 或 padding `[B,20]`，rad
        evidence: StaticGeometryEvidence,  # 静态 hand evidence 与 masks
        query_points_h: torch.Tensor,  # `[B,G,N_Q,3]`，`{h}`，m
        bandwidths: torch.Tensor,  # `[N_σ]` 或 `[B,N_σ]`，m，显式 decoder 条件
        owner_index: torch.Tensor,  # `[E]`/`[B,E]`
        query_index: torch.Tensor,  # `[E]`/`[B,E]`
        joint_index: torch.Tensor,  # `[E]`/`[B,E]`
        evidence_row_index: torch.Tensor | None = None,  # `[B]` q 行到 unique evidence table
    ) -> GeometrySSLForward:
        r"""完成 retained 编码和两个 training-only 预测头。

        Args:
            q (torch.Tensor): ``[B,N_J]`` 当前物理关节角，rad，不要求输入梯度。
            evidence (StaticGeometryEvidence): 当前结构模式的静态可部署证据。
            query_points_h (torch.Tensor): ``[B,G,N_Q,3]`` 固定 `{h}` queries，m，已停止采样梯度。
            bandwidths (torch.Tensor): ``[N_sigma]`` 或 ``[B,N_sigma]`` 实际 sigma，m。
            owner_index (torch.Tensor): ``[E]`` 或跨结构 ``[B,E]`` sampled owner selectors。
            query_index (torch.Tensor): 与 owner selector 同形状的 query selectors。
            joint_index (torch.Tensor): 与 owner selector 同形状的 JOINT selectors。

        Returns:
            GeometrySSLForward: 类型化 latents、共享 query features、density 与 κ 预测。
        """

        owner_count = evidence.entity_role.shape[-1]  # $G$；batched/unbatched role 都读尾轴
        if query_points_h.ndim != 4 or query_points_h.shape[:2] != (q.shape[0], owner_count):  # `[B,G]`
            raise ValueError("query_points_h must have shape [B,G,N_Q,3] matching q/evidence")  # 不广播
        latents = self.encoder(q, evidence, evidence_row_index)  # retained path；静态 evidence 可按资产去重
        query_features = self.encoder.encode_points(
            query_points_h.detach(), evidence, evidence_row_index
        )  # query 点已有 q-row 轴，anchors/normal 通过同一 row index 路由
        entity_valid = evidence.entity_valid_mask  # `[B,G]`/`[G]` 或原生可变长时 None
        if entity_valid is not None:  # padding container 才需要显式零化
            if evidence_row_index is not None and entity_valid.ndim == 2:
                entity_valid = entity_valid[evidence_row_index]
            if entity_valid.ndim == 1:  # 同结构 batch 共享实体 mask
                entity_valid = entity_valid.unsqueeze(0).expand(q.shape[0], -1)  # `[B,G]` view
            query_features = query_features * entity_valid.unsqueeze(-1).unsqueeze(-1)  # invalid owner 精确零
        return self.decode_latents(  # 集中 disposable 路径供完整模型与 ablation 共用
            latents,  # retained unified $Z$
            query_features,  # `[B,G,N_Q,D_q]`
            bandwidths=bandwidths,  # 显式 sigma 数据轴
            entity_valid_mask=entity_valid,  # padding owner mask
            joint_entity_index=(
                evidence.joint_entity_index[evidence_row_index]
                if evidence_row_index is not None and evidence.joint_entity_index.ndim == 2
                else evidence.joint_entity_index
            ),  # JOINT 坐标到 unified entity 轴的 q-row 路由
            owner_index=owner_index,  # sampled owner
            query_index=query_index,  # sampled query
            joint_index=joint_index,  # sampled JOINT
        )

    def decode_latents(
        self,
        latents: GeometryLatents,  # 可为完整/zero/shuffled latent
        query_features: torch.Tensor,  # 固定 query path
        *,
        bandwidths: torch.Tensor,  # `[N_σ]` 或 `[B,N_σ]`
        entity_valid_mask: torch.Tensor | None,  # `[B,G]`
        joint_entity_index: torch.Tensor,  # `[N_J]`/`[B,N_J]`
        owner_index: torch.Tensor,  # `[E]`/`[B,E]`
        query_index: torch.Tensor,  # `[E]`/`[B,E]`
        joint_index: torch.Tensor,  # `[E]`/`[B,E]`
    ) -> GeometrySSLForward:
        r"""从显式 latent/query features 运行 disposable heads，供受控 ablation 复用。

        density 预测为 $\hat\rho\in\mathbb R^{B\times G\times N_Q\times N_\sigma}$；κ 只在 sampled edges
        输出 $\hat\kappa\in\mathbb R^{B\times E}$。该函数不重新编码 q/evidence，因此 ablation 可以只干预
        latent 而保持 decoder/query path 相同。
        """

        entities = latents.entities  # `[B,G,D]`，density 与 kappa 共享同一 retained 表征
        density = self.density_decoder(entities, query_features, bandwidths)  # `[B,G,N_Q,N_σ]`
        if entity_valid_mask is not None:  # padding owner 不属于物理监督测度
            density = density * entity_valid_mask.unsqueeze(-1).unsqueeze(-1)  # padding owner 不产生虚假场值
        if owner_index.ndim not in {1, 2} or query_index.shape != owner_index.shape or joint_index.shape != owner_index.shape:
            raise ValueError("owner/query/joint selectors must share [E] or [B,E] shape")
        if owner_index.ndim == 1:
            owner_latent = entities.index_select(1, owner_index)  # `[B,E,D]` 的 $z_o$
            if joint_entity_index.ndim == 1:
                joint_entities = joint_entity_index.index_select(0, joint_index)  # edge JOINT -> entity slot
                joint_latent = entities.index_select(1, joint_entities)  # `[B,E,D]` 的 $z_i$
            elif joint_entity_index.ndim == 2 and joint_entity_index.shape[0] == entities.shape[0]:
                batch_index = torch.arange(entities.shape[0], device=entities.device).unsqueeze(1)
                joint_entities = joint_entity_index[:, joint_index]
                joint_latent = entities[batch_index, joint_entities]
            else:
                raise ValueError("joint_entity_index must have shape [N_J] or [B,N_J]")
            selected_query = query_features[:, owner_index, query_index]  # `[B,E,D_q]`
        else:
            if owner_index.shape[0] != entities.shape[0] or joint_entity_index.ndim != 2:
                raise ValueError("batched selectors/routing must share B with entity latents")
            batch_index = torch.arange(entities.shape[0], device=entities.device).unsqueeze(1)  # `[B,1]`
            owner_latent = entities[batch_index, owner_index]  # 每个样本自己的 edge owner token
            joint_entities = torch.gather(joint_entity_index, 1, joint_index)  # `[B,E]` entity selectors
            joint_latent = entities[batch_index, joint_entities]  # 每个样本自己的 edge JOINT token
            selected_query = query_features[batch_index, owner_index, query_index]  # `[B,E,D_q]`
        kappa = self.sensitivity_decoder(owner_latent, joint_latent, selected_query)  # `[B,E]` signed scalar
        return GeometrySSLForward(latents, query_features, density, kappa)  # 类型化 objective 输入

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回只含部署 encoder 的 checkpoint 参数。

        key 保留 ``encoder.`` 前缀，使完整与 retained-only checkpoint 可以用同一加载审计工具比较；
        density/sensitivity decoder 参数绝不出现在返回值中。
        """

        return {  # 保留完整模型中的稳定 namespace
            f"encoder.{key}": value for key, value in self.encoder.state_dict().items()  # 不含 decoder
        }


__all__ = [  # 模型层稳定公开面
    "GeometrySSLForward",  # typed prediction
    "GeometrySSLModel",  # retained+disposable assembly
    "GeometrySSLModelCfg",  # assembly config
]
