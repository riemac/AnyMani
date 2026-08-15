r"""物理 source、空间场、query 与 target 的在线 Geometry Representation 组合。

该模块不把 $(asset,q,query,target)$ 全量离线固化。每项资产只物化一次静态证据：owner union、
    home boundary、anchors、owner triangle sampling table、kinematic spec 与 Warp BVH；训练 step
再从 joint limits 用 scrambled Sobol 采样合法 q，生成 workspace/shell/adjacent query 和 Warp teacher。

跨结构 batch 使用明确上限：最多 20 个 JOINT、5 个 TIP、26 个 owner。padding 只是 GPU 稠密容器：

- 实际 q/screw/owner/query/edge 复制到前缀有效槽；
- entity/joint/field/edge mask 显式记录有效范围；
- padding token、target 与 prediction 在模型/损失边界清零；
- 无 padding 的逐结构前向保留为输出/梯度 oracle。

单资产、pre-made 母体及其 post-mutate variants、同/跨 family generated 资产都走同一接口。
official 资产是否进入训练不由本模块猜测；实验配置必须在 HandBank selection 和 split manifest 中排除。
"""

from __future__ import annotations  # 前向类型引用不在 import 时求值

from dataclasses import dataclass  # 避免 ``field`` 配置槽遮蔽 helper
from dataclasses import field as dataclass_field

import torch  # Sobol、张量 padding、GPU evidence 与 target

from anymani.distill.models.input_adapters.geometry import (  # retained 静态输入与 padding
    GeometryPaddingCfg,  # 20 JOINT/5 TIP/26 owner 上限
    StaticGeometryEvidence,  # anchors/home/screws/graph/masks
    build_static_geometry_evidence,  # assets+robots -> model evidence
    pad_static_geometry_evidence,  # 跨结构静态轴 padding
)
from anymani.distill.representations.queries.spatial_sampling import (  # 50/25/25 query 测度
    OwnerSurfaceSamplingCache,  # GPU owner-local triangle/area/normal static cache
    SpatialQueryBatch,  # query 坐标/stratum/adjacent provenance
    SpatialQuerySamplerCfg,  # $N_W/N_S/N_A$
    materialize_owner_surface_sampling_cache,  # owner union -> GPU 在线 proposal 测度
    sample_spatial_queries,  # 当前 q owner-shell/adjacent
)
from anymani.distill.representations.sources.geometry_source import (
    DeviceGeometrySource,
    GeometrySource,
    GeometrySourceCfg,
)
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec
from anymani.distill.representations.targets.field_samples import (  # 类型化 $d/\\rho/\\kappa/g$ targets
    FieldTargetBatch,
    SensitivityTargetBatch,
)
from anymani.distill.representations.targets.geometry_field import (  # Warp teacher assembly
    GaussianProximityFieldCfg,  # train/validation sigma measure
    GeometryFieldTargetCfg,  # edges/mask thresholds
    generate_geometry_field_targets,  # online target 主路径
)


@dataclass(frozen=True)
class GeometryRepresentationCfg:
    r"""source、field、query、target 与跨结构 layout 的正交组合配置。"""

    source: GeometrySourceCfg = dataclass_field(default_factory=GeometrySourceCfg)  # q-independent physical oracle
    field: GaussianProximityFieldCfg = dataclass_field(default_factory=GaussianProximityFieldCfg)  # sigma measure
    query: SpatialQuerySamplerCfg = dataclass_field(default_factory=SpatialQuerySamplerCfg)  # query measure
    target: GeometryFieldTargetCfg = dataclass_field(default_factory=GeometryFieldTargetCfg)  # edge/mask teacher
    layout: GeometryPaddingCfg = dataclass_field(default_factory=GeometryPaddingCfg)  # dense cross-structure axes


@dataclass(frozen=True)
class GeometryRepresentationState:
    r"""一项资产在指定 device 上供 query、target 与 retained adapter 共用的状态。"""

    source: GeometrySource  # CPU physical truth 与 provenance
    device_source: DeviceGeometrySource  # GPU POE 与 Warp BVH lease
    surface_sampling: OwnerSurfaceSamplingCache  # owner triangle/normal/area proposal tables
    evidence: StaticGeometryEvidence  # GPU retained encoder 静态输入

    @property
    def spec(self) -> EmbodimentGeometrySpec:
        r"""返回与模型、query 和 target 同 device/dtype 的动态运动学规格。"""

        return self.device_source.spec

    @property
    def warp_cache(self):
        r"""返回 closest-surface target 使用的 GPU owner BVH cache。"""

        return self.device_source.warp_cache


class GeometryRepresentation:
    r"""把 physical source、query measure 与 field teacher 组合成类型化训练样本。

    本类拥有 representation 行为但不拥有 q coverage、optimizer 或 checkpoint。所有随机 realization
    通过显式 seed 传入，不在对象内维护隐藏 RNG state。
    """

    def __init__(self, config: GeometryRepresentationCfg) -> None:
        r"""保存纯声明配置；构造阶段不读取资产、不初始化 CUDA。"""

        self.config = config  # 完整 source/query/target/layout 科研合同

    def materialize_source(self, container) -> GeometrySource:
        r"""按 source config 物化一项 q-independent CPU physical oracle。"""

        return GeometrySource.materialize(container, config=self.config.source)

    def to_device(
        self,
        source: GeometrySource,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> GeometryRepresentationState:
        r"""构造一项资产的 device source、surface proposal 与 retained evidence。"""

        device_source = source.to_device(device=device, dtype=dtype)  # POE/Warp lease 一次 materialize
        try:
            target_device = torch.device(device)  # query/model 共用规范化 device
            surface_sampling = materialize_owner_surface_sampling_cache(
                source.geometry_cache,
                device=target_device,
                dtype=dtype,
            )
            semantics = source.container.geometry_semantics  # owner roles 与 palm normal 的唯一真源
            if semantics is None:
                raise ValueError("geometry source lost its typed semantics")
            evidence = build_static_geometry_evidence(
                semantics,
                device_source.spec,
                source.home_surface,
                source.anchors,
                device=target_device,
                dtype=dtype,
            )
            return GeometryRepresentationState(source, device_source, surface_sampling, evidence)
        except Exception:
            device_source.release()  # proposal/evidence 任一失败都归还已取得的 Warp lease
            raise

    def sample(
        self,
        state: GeometryRepresentationState,
        q: torch.Tensor,
        *,
        sampling_seed: int,
        q_index: torch.Tensor | None = None,
    ) -> OnlineGeometrySample:
        r"""按当前配置为同资产 q realization 生成未 padding query/teacher sample。"""

        return sample_online_geometry(
            state,
            q,
            field_config=self.config.field,
            query_config=self.config.query,
            target_config=self.config.target,
            sampling_seed=sampling_seed,
            q_index=q_index,
        )


@dataclass(frozen=True)  # 单 q teacher 样本构造后只读
class OnlineGeometrySample:  # variable-length sample
    r"""一项资产、一个当前 q 的未 padding 在线监督。

    每项固定 batch size 为 1，保留真实 $N_J/G/E$；只有多个样本合并时才进入 20/26 稠密容器。
    """

    asset_id: str  # bank 稳定路由 ID
    q: torch.Tensor  # `[1,N_J]`，rad
    evidence: StaticGeometryEvidence  # 单资产，无 batch 静态轴
    queries: SpatialQueryBatch  # `[1,G,N_Q,...]`
    field_targets: FieldTargetBatch  # `[1,G,N_Q,L]` $\\rho$ 与 distance
    sensitivity_targets: SensitivityTargetBatch  # `[1,E]` $\\kappa$、`[1,E,L]` g
    q_index: torch.Tensor | None = None  # `[Q]`，资产本地 Sobol 序列中的绝对 q cursor


@dataclass(frozen=True)  # 一次 forward 的 q/query/target 轴共同冻结
class PaddedOnlineGeometryBatch:  # heterogeneous dense batch
    r"""不同结构可共同进入一次模型前向的完整 batch。

    padding 只改变存储，不改变监督测度：entity/joint/field/edge masks 分别屏蔽所有无效槽；损失按有效
    标量数归一化。不同结构独立前向与本容器有效位置的输出/参数梯度有 contract oracle。
    """

    asset_ids: tuple[str, ...]  # `[B]` batch routing identity
    q: torch.Tensor  # `[B,20]`，padding JOINT 为 0
    evidence: StaticGeometryEvidence  # `[B,26,...]` + entity/joint masks
    queries: SpatialQueryBatch  # `[B,26,N_Q,...]`
    field_targets: FieldTargetBatch  # invalid owner/query 由 valid_mask 屏蔽
    sensitivity_targets: SensitivityTargetBatch  # `[B,E_max]` selectors + valid_mask
    q_index: torch.Tensor | None = None  # `[B]`，每个样本的资产本地 Sobol cursor


class SobolJointSampler:  # 每资产独立低差异 q 序列
    r"""在每项资产完整 joint-limit 超矩形中连续产生 scrambled Sobol q。

    对单位 Sobol 样本 $u\in[0,1]^{N_J}$ 使用
    $q_i=l_i+u_i(u_i^{max}-l_i)$。这里的 limits 只定义采样域，不进入 encoder；完整域包含
    self-collision，避免把碰撞先验偷渡进 task-free geometry representation。
    """

    def __init__(self, spec: EmbodimentGeometrySpec, *, seed: int) -> None:
        r"""保存 CPU rad limits 并初始化 $N_J$ 维独立 scrambled SobolEngine。"""

        if spec.joint_limits is None:  # robots spec 必须显式交付采样域
            raise ValueError("EmbodimentGeometrySpec must contain joint_limits for q sampling")  # 不猜 [-pi,pi]
        self.limits = spec.joint_limits.detach().cpu().to(torch.float64)  # `[N_J,2]`，rad
        self.seed = int(seed)  # engine 重建与 checkpoint resume 的 deterministic identity
        self.cursor = 0  # 已消费的 Sobol q 数；不是 optimizer step
        self.engine = torch.quasirandom.SobolEngine(  # 连续 draw 保留低差异序列状态
            dimension=self.limits.shape[0],  # $N_J$ 随资产变化
            scramble=True,  # Owen scrambling 提供 seed 可复现随机化
            seed=self.seed,  # 每资产独立派生 seed
        )

    def draw(
        self,
        count: int,  # 连续样本数
        *,
        device: torch.device | str,  # 目标模型/Warp device
        dtype: torch.dtype,  # 目标 model dtype
    ) -> torch.Tensor:  # `[count,N_J]`，rad
        r"""返回 `[count,N_J]` 合法 q，完整域包含 self-collision 构型。"""

        if count < 1:  # SobolEngine 空 draw 不属于 batcher 合同
            raise ValueError("Sobol draw count must be positive")  # fail-fast
        unit = self.engine.draw(count, dtype=torch.float64)  # $[0,1]^{N_J}$ 低差异序列
        q = self.limits[:, 0] + unit * (self.limits[:, 1] - self.limits[:, 0])  # $q=l+u(h-l)$，rad
        self.cursor += int(count)  # draw 成功后推进，异常不会伪造已消费 q
        return q.to(device=device, dtype=dtype)  # 只在最终边界上传/转换

    def state_dict(self) -> dict[str, int]:
        r"""返回可写入 checkpoint 的低差异序列状态。"""

        return {"seed": self.seed, "cursor": self.cursor, "dimension": int(self.limits.shape[0])}

    def load_state_dict(self, state: dict[str, int]) -> None:
        r"""从 seed+cursor 重建 Sobol engine，确保 resume 后下一个 q 完全一致。"""

        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("Sobol checkpoint seed does not match asset sampler")
        if int(state.get("dimension", -1)) != self.limits.shape[0]:
            raise ValueError("Sobol checkpoint dimension does not match asset joint count")
        cursor = int(state.get("cursor", -1))
        if cursor < 0:
            raise ValueError("Sobol checkpoint cursor must be non-negative")
        self.engine = torch.quasirandom.SobolEngine(
            dimension=self.limits.shape[0],
            scramble=True,
            seed=self.seed,
        )
        if cursor:
            self.engine.fast_forward(cursor)
        self.cursor = cursor


def sample_online_geometry(
    state: GeometryRepresentationState,  # 当前资产 representation device state
    q: torch.Tensor,  # `[Q,N_J]`，同一资产的构型 block，rad
    *,
    field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),  # sigma measure
    query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),  # 50/25/25
    target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),  # teacher
    sampling_seed: int = 0,  # shell/adjacent/edge sampling realization
    q_index: torch.Tensor | None = None,  # `[Q]` asset-local absolute Sobol cursor
    ) -> OnlineGeometrySample:  # unpadded one-asset teacher sample
    r"""为一项资产的 ``[Q,N_J]`` q block 生成 query 与 Warp teacher。

    query 与 teacher 都从 ``q.detach()`` 的物理构型生成；模型对 q 的 Sobolev 图在 trainer 中另建，
    因此 teacher 几何路径不接收模型梯度。
    """

    if q.ndim != 2 or q.shape[1] != state.spec.space_screws.shape[0] or q.shape[0] < 1:
        raise ValueError("sample_online_geometry expects [Q,N_J] with the asset's true N_J")
    if q_index is not None and q_index.shape != (q.shape[0],):
        raise ValueError("q_index must have shape [Q] matching the asset q block")
    queries = sample_spatial_queries(  # 当前 q 下 `[1,G,N_Q,3]`
        q,  # 物理 rad；Q 个构型共享 query/teacher batch 轴
        state.spec,  # owner FK/graph
        state.surface_sampling,  # 完整 owner triangle/area/normal proposal
        state.evidence.anchors,  # 固定 `{h}` anchors；workspace realization 每 q 子批次重采
        config=query_config,  # stratum 比例/壳厚
        sampling_seed=sampling_seed,  # 当前 realization
    )
    field_targets, sensitivity_targets = generate_geometry_field_targets(  # GPU Warp teacher
        q,  # 当前 owner transforms/Jacobian；同一资产一次处理 Q 个构型
        state.spec,  # POE/ancestor masks
        state.source.geometry_cache,  # CPU face/component provenance
        state.warp_cache,  # GPU BVHs
        queries,  # 当前 query batch
        field_config=field_config,  # $\\sigma_\\ell$ centers/jitter
        target_config=target_config,  # sampled edges/margins
        edge_sampling_seed=sampling_seed,  # sampled `(g,r,i)` realization
    )
    return OnlineGeometrySample(  # 保留真实 variable lengths，padding 延后
        asset_id=state.source.asset_id,  # batch route
        q=q,  # `[1,N_J]`
        evidence=state.evidence,  # unbatched static evidence
        queries=queries,  # `[1,G,N_Q,...]`
        field_targets=field_targets,  # zero-order teacher
        sensitivity_targets=sensitivity_targets,  # sampled-edge teacher
        q_index=q_index.detach().cpu() if q_index is not None else None,  # provenance 不进入 CUDA teacher
    )


def split_online_geometry_sample(sample: OnlineGeometrySample) -> tuple[OnlineGeometrySample, ...]:
    r"""把一次同资产 `[Q,N_J]` teacher block 展开为 padding 所需的 `[1,N_J]` 样本。

    split 只切 batch 轴，不重新采样 query、最近点、BVH 或 Jacobian；因此 `Q>1` 与逐 q
    循环共享完全相同的 teacher 数值，差别只在底层 FK/target 是否被一次批量调用。
    """

    q_count = sample.q.shape[0]
    return tuple(
        OnlineGeometrySample(
            asset_id=sample.asset_id,
            q=sample.q[index : index + 1],
            evidence=sample.evidence,
            queries=SpatialQueryBatch(
                sample.queries.query_points_h[index : index + 1],
                sample.queries.query_stratum[index : index + 1],
                sample.queries.adjacent_owner_index[index : index + 1],
                sample.queries.workspace_anchor_index[index : index + 1],
            ),
            field_targets=FieldTargetBatch(
                query_points=sample.field_targets.query_points[index : index + 1],
                query_stratum=sample.field_targets.query_stratum[index : index + 1],
                distance=sample.field_targets.distance[index : index + 1],
                density=sample.field_targets.density[index : index + 1],
                valid_mask=sample.field_targets.valid_mask[index : index + 1],
                owner_role=sample.field_targets.owner_role,
                bandwidths=(
                    sample.field_targets.bandwidths
                    if sample.field_targets.bandwidths.ndim == 1
                    else sample.field_targets.bandwidths[index : index + 1]
                ),
                provenance=sample.field_targets.provenance,
            ),
            sensitivity_targets=SensitivityTargetBatch(
                owner_index=sample.sensitivity_targets.owner_index,
                query_index=sample.sensitivity_targets.query_index,
                joint_index=sample.sensitivity_targets.joint_index,
                ancestor_mask=sample.sensitivity_targets.ancestor_mask,
                closest_point=sample.sensitivity_targets.closest_point[index : index + 1],
                closest_source=sample.sensitivity_targets.closest_source[index : index + 1],
                uniqueness_margin=sample.sensitivity_targets.uniqueness_margin[index : index + 1],
                kappa=sample.sensitivity_targets.kappa[index : index + 1],
                field_sensitivity=sample.sensitivity_targets.field_sensitivity[index : index + 1],
                valid_mask=sample.sensitivity_targets.valid_mask[index : index + 1],
                provenance=sample.sensitivity_targets.provenance,
            ),
            q_index=sample.q_index[index : index + 1] if sample.q_index is not None else None,
        )
        for index in range(q_count)
    )


def pad_online_geometry_samples(
    samples: list[OnlineGeometrySample],  # 不同资产/结构的 B 项样本
    *,
    padding: GeometryPaddingCfg = GeometryPaddingCfg(),  # 20 JOINT/5 TIP/26 owner
    ) -> PaddedOnlineGeometryBatch:  # `[B,20]/[B,26]` model batch
    r"""把不同 $N_J/G/E$ 的在线样本填充为统一训练 batch。

    每个样本的真实轴写入前缀：``q[:N_J]``、``owner[:G]``、``edge[:E]``；其余槽保持零值并由
    ``joint_valid/entity_valid/field_valid/edge_valid`` 四类 mask 屏蔽。selector padding 使用合法 0 索引，
    但 ``edge_valid=False``，从而避免 gather 越界且不进入 loss。

    Returns:
        PaddedOnlineGeometryBatch: 一次 heterogeneous forward 的完整 q/evidence/query/target。
    """

    if not samples:  # batch 必须至少包含一个物理样本
        raise ValueError("at least one OnlineGeometrySample is required")  # 防止从 samples[0] 猜轴
    device = samples[0].q.device  # 全 batch 唯一 GPU
    dtype = samples[0].q.dtype  # q/geometry/target 浮点 dtype
    query_count = samples[0].queries.query_points_h.shape[2]  # 固定 $N_Q$
    bandwidth_count = samples[0].field_targets.bandwidths.shape[-1]  # 动态 $N_\sigma$ 数据轴
    if any(sample.q.device != device or sample.q.dtype != dtype for sample in samples):  # device/dtype 一致性
        raise ValueError("all online samples must share device and dtype")  # 禁止隐式 copy/cast
    if any(sample.queries.query_points_h.shape[2] != query_count for sample in samples):  # query 轴
        raise ValueError("all samples must share N_Q")  # 当前 decoder 稠密 query 轴不 padding
    if any(sample.field_targets.bandwidths.shape[-1] != bandwidth_count for sample in samples):
        raise ValueError("all samples in one dense batch must share N_sigma")

    batch_size = len(samples)  # $B$
    max_owner_count = padding.max_owner_count  # $G_{max}=26=1+20+5$
    max_joint_count = padding.max_joint_count  # $N_{J,max}=20$
    max_edge_count = max(  # 只 padding 到当前 batch 最大 $E$，不固定浪费 26*edges
        sample.sensitivity_targets.kappa.shape[1] for sample in samples
    )
    evidence = pad_static_geometry_evidence(  # anchors/home/screws/graph + entity/joint masks
        [sample.evidence for sample in samples], config=padding
    )
    q = torch.zeros(batch_size, max_joint_count, device=device, dtype=dtype)  # `[B,20]`，rad
    query_points = torch.zeros(  # `[B,26,N_Q,3]`，`{h}`，m
        batch_size, max_owner_count, query_count, 3, device=device, dtype=dtype
    )
    query_stratum = torch.zeros(  # `[B,26,N_Q]`；padding 值 0=WORKSPACE 但 field_valid=False
        batch_size, max_owner_count, query_count, device=device, dtype=torch.long
    )
    adjacent_owner = torch.full_like(query_stratum, -1)  # 非 adjacent/padding sentinel
    workspace_anchor = torch.full_like(query_stratum, -1)  # 非 workspace/padding sentinel
    bandwidths = torch.zeros(  # `[B,N_σ]`，每个样本实际采样的 sigma，m
        batch_size, bandwidth_count, device=device, dtype=dtype
    )
    distance = torch.zeros(  # `[B,26,N_Q]` unsigned owner distance，m
        batch_size, max_owner_count, query_count, device=device, dtype=dtype
    )
    density = torch.zeros(  # `[B,26,N_Q,L]`，无量纲
        batch_size, max_owner_count, query_count, bandwidth_count, device=device, dtype=dtype
    )
    field_valid = torch.zeros(  # `[B,26,N_Q]`；唯一 zero-order loss 归一化 mask
        batch_size, max_owner_count, query_count, device=device, dtype=torch.bool
    )
    owner_role = torch.zeros(  # `[B,26]`；padding 角色值无意义，由 entity mask 屏蔽
        batch_size, max_owner_count, device=device, dtype=torch.long
    )
    owner_index = torch.zeros(  # `[B,E_max]` sampled edge owner；padding 合法指向 0
        batch_size, max_edge_count, device=device, dtype=torch.long
    )
    edge_query_index = torch.zeros_like(owner_index)  # `[B,E_max]` query selector
    joint_index = torch.zeros_like(owner_index)  # `[B,E_max]` JOINT selector
    ancestor_mask = torch.zeros(  # `[B,E_max]` 拓扑祖先结构零标记
        batch_size, max_edge_count, device=device, dtype=torch.bool
    )
    closest_point = torch.zeros(  # `[B,E_max,3]`，`{h}`，m
        batch_size, max_edge_count, 3, device=device, dtype=dtype
    )
    closest_source = torch.zeros(  # `[B,E_max]` face/component provenance ID
        batch_size, max_edge_count, device=device, dtype=torch.long
    )
    uniqueness_margin = torch.zeros(  # `[B,E_max]` local triangle feature margin，m
        batch_size, max_edge_count, device=device, dtype=dtype
    )
    kappa = torch.zeros(  # `[B,E_max]` distance sensitivity，m/rad
        batch_size, max_edge_count, device=device, dtype=dtype
    )
    field_sensitivity = torch.zeros(  # `[B,E_max,L]`，1/rad
        batch_size, max_edge_count, bandwidth_count, device=device, dtype=dtype
    )
    edge_valid = torch.zeros(  # `[B,E_max]`；padding/non-smooth edge loss mask
        batch_size, max_edge_count, device=device, dtype=torch.bool
    )
    q_index = torch.full((batch_size,), -1, device=device, dtype=torch.long)  # `[B]`，未知 cursor 为 -1

    for batch_index, sample in enumerate(samples):  # 每项独立真实 $N_J/G/E$
        joint_count = sample.q.shape[1]  # 当前 $N_J$
        owner_count = sample.queries.query_points_h.shape[1]  # 当前 $G$
        edge_count = sample.sensitivity_targets.kappa.shape[1]  # 当前 $E$
        q[batch_index, :joint_count] = sample.q[0]  # rad q 写入 `[0:N_J)`
        query_points[batch_index, :owner_count] = sample.queries.query_points_h[0]  # `{h}` query
        query_stratum[batch_index, :owner_count] = sample.queries.query_stratum[0]  # 0/1/2 provenance
        adjacent_owner[batch_index, :owner_count] = sample.queries.adjacent_owner_index[0]  # neighbor owner
        workspace_anchor[batch_index, :owner_count] = sample.queries.workspace_anchor_index[0]  # anchor provenance
        sample_bandwidths = sample.field_targets.bandwidths  # `[N_σ]` 或 split 后 `[1,N_σ]`
        bandwidths[batch_index] = sample_bandwidths if sample_bandwidths.ndim == 1 else sample_bandwidths[0]
        distance[batch_index, :owner_count] = sample.field_targets.distance[0]  # m
        density[batch_index, :owner_count] = sample.field_targets.density[0]  # `[G,N_Q,L]`
        field_valid[batch_index, :owner_count] = sample.field_targets.valid_mask[0]  # zero-order mask
        role = sample.field_targets.owner_role  # `[G]` 或 already batched `[1,G]`
        owner_role[batch_index, :owner_count] = (  # 保留 PALM/JOINT/TIP 角色
            role if role.ndim == 1 else role[0]
        )

        sensitivity = sample.sensitivity_targets  # 当前未 padding sampled edges
        owner_index[batch_index, :edge_count] = sensitivity.owner_index  # $g_e$
        edge_query_index[batch_index, :edge_count] = sensitivity.query_index  # $r_e$
        joint_index[batch_index, :edge_count] = sensitivity.joint_index  # $i_e$
        ancestor_mask[batch_index, :edge_count] = sensitivity.ancestor_mask  # 拓扑结构零
        closest_point[batch_index, :edge_count] = sensitivity.closest_point[0]  # `{h}`，m
        closest_source[batch_index, :edge_count] = sensitivity.closest_source[0]  # provenance
        uniqueness_margin[batch_index, :edge_count] = sensitivity.uniqueness_margin[0]  # m
        kappa[batch_index, :edge_count] = sensitivity.kappa[0]  # m/rad
        field_sensitivity[batch_index, :edge_count] = sensitivity.field_sensitivity[0]  # 1/rad
        edge_valid[batch_index, :edge_count] = sensitivity.valid_mask[0]  # edge loss mask
        if sample.q_index is not None:
            if sample.q_index.numel() != 1:
                raise ValueError("pad_online_geometry_samples expects split samples with one q_index")
            q_index[batch_index] = sample.q_index.reshape(-1)[0].to(device=device)

    queries = SpatialQueryBatch(  # decoder/sampler provenance 包
        query_points, query_stratum, adjacent_owner, workspace_anchor
    )
    field_targets = FieldTargetBatch(  # zero-order target + valid normalization mask
        query_points=query_points,  # `[B,26,N_Q,3]`，`{h}`，m
        query_stratum=query_stratum,  # 不进入 decoder
        distance=distance,  # `[B,26,N_Q]`，m
        density=density,  # `[B,26,N_Q,L]`，无量纲
        valid_mask=field_valid,  # invalid owner/query 全 False
        owner_role=owner_role,  # `[B,26]`
        bandwidths=bandwidths,  # `[B,N_σ]`，每个样本实际 sigma，m
        provenance={  # frame/unit/backend/padding 必须可审计
            "frame": "h",  # hand semantic frame
            "length_unit": "m",  # SI length
            "backend": "warp_mesh_query_point",  # online main backend
            "padding": f"joint={max_joint_count},owner={max_owner_count}",  # 稠密上限
        },
    )
    sensitivity_targets = SensitivityTargetBatch(  # sampled first-order target + edge mask
        owner_index=owner_index,  # `[B,E_max]`
        query_index=edge_query_index,  # `[B,E_max]`
        joint_index=joint_index,  # `[B,E_max]`
        ancestor_mask=ancestor_mask,  # 非祖先 target 精确零
        closest_point=closest_point,  # `[B,E_max,3]`，`{h}`，m
        closest_source=closest_source,  # face/component provenance
        uniqueness_margin=uniqueness_margin,  # m
        kappa=kappa,  # m/rad
        field_sensitivity=field_sensitivity,  # `[B,E_max,L]`，1/rad
        valid_mask=edge_valid,  # invalid/padding/non-smooth edges 不进 loss
        provenance={  # first-order frame/unit/padding
            "frame": "h",  # closest point frame
            "distance_unit": "m",  # d/closest/κ numerator
            "joint_unit": "rad",  # κ/g derivative denominator
            "padding": f"edge={max_edge_count}",  # 当前 batch E 上限
        },
    )
    return PaddedOnlineGeometryBatch(  # model/objective/logger 共用的一致 batch
        asset_ids=tuple(sample.asset_id for sample in samples),  # `[B]`
        q=q,  # `[B,20]`，rad
        evidence=evidence,  # `[B,26,...]` + masks
        queries=queries,  # `[B,26,N_Q,...]`
        field_targets=field_targets,  # zero-order
        sensitivity_targets=sensitivity_targets,  # first-order
        q_index=q_index,  # asset-local Sobol provenance
    )


class OnlineGeometryBatcher:  # deterministic multi-asset online sampler
    r"""在预物化资产间轮转、采 q/target 并输出跨结构 padding batch。

    资产选择使用确定性 round-robin：第 ``step`` 个 batch 的槽 $b$ 选择
    $(step\cdot B+b)\bmod N_{asset}$。每项资产拥有独立持续 SobolEngine，因此改变其他资产 DOF
    不会改变本资产 q 序列维度或 limits。
    """

    def __init__(
        self,
        states: list[GeometryRepresentationState],  # 预物化 generated representation device states
        *,
        seed: int,  # Sobol/query/edge 总种子
        field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),  # sigma measure
        query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),  # 50/25/25
        target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),  # Warp teacher
        padding: GeometryPaddingCfg = GeometryPaddingCfg(),  # 20/5/26 上限
    ) -> None:  # 初始化 per-asset Sobol state
        r"""保存 GPU asset states，并为每项资产建立独立 Sobol 序列。

        Raises:
            ValueError: asset state 列表为空时抛出。
        """

        if not states:  # optimizer batcher 必须有 generated asset
            raise ValueError("OnlineGeometryBatcher requires at least one asset state")  # 防止 modulo 0
        self.states = tuple(states)  # 冻结 routing 顺序，与 manifest 一致
        self.seed = int(seed)  # Python int 复现锚点
        self.field_config = field_config  # actual sigma realization 规则
        self.query_config = query_config  # stratum 与 shell 数值
        self.target_config = target_config  # sampled edge/margin
        self.padding = padding  # 稠密容器上限
        self.samplers = tuple(  # 每资产独立维度/limits/Sobol state
            SobolJointSampler(  # CPU engine，只把 draw 结果上传
                state.source.spec_cpu, seed=self.seed + asset_index  # 稳定派生 seed
            )
            for asset_index, state in enumerate(states)  # manifest/routing 顺序
        )

    def sample(self, *, batch_size: int, step: int) -> PaddedOnlineGeometryBatch:
        r"""按 round-robin 资产平衡生成一个 batch；不同 family 共享模型参数。

        Args:
            batch_size (int): 当前 microbatch 样本数 $B$。
            step (int): 唯一在线采样步，用于资产路由与 query/edge seed。

        Returns:
            PaddedOnlineGeometryBatch: heterogeneous model/objective 输入。
        """

        if batch_size < 1 or step < 0:  # 生命周期离散域
            raise ValueError("batch_size must be positive and step non-negative")  # 不修正 caller bug
        samples: list[OnlineGeometrySample] = []  # 先保留真实 lengths，再统一 padding
        for batch_offset in range(batch_size):  # $b=0,...,B-1$
            asset_index = (step * batch_size + batch_offset) % len(self.states)  # deterministic round-robin
            state = self.states[asset_index]  # 当前资产 GPU static state
            q = self.samplers[asset_index].draw(  # 当前资产连续 Sobol q
                1,  # 每资产槽一个构型
                device=state.spec.space_screws.device,  # 与 GPU spec/model 同 device
                dtype=state.spec.space_screws.dtype,  # 与 spec/model 同 dtype
            )
            q_cursor = self.samplers[asset_index].cursor - 1  # 当前 draw 的 asset-local absolute cursor
            samples.append(  # 完整在线 query/teacher 未 padding 样本
                sample_online_geometry(  # Warp GPU main path
                    state,  # 当前 asset
                    q,  # `[1,N_J]` rad
                    field_config=self.field_config,  # actual sigma realization
                    query_config=self.query_config,  # 50/25/25
                    target_config=self.target_config,  # $d/\\rho/\\kappa/g$
                    sampling_seed=self.seed + step * batch_size + batch_offset,  # 唯一 realization
                    q_index=torch.tensor([q_cursor]),  # 采样 provenance
                )
            )
        return pad_online_geometry_samples(samples, padding=self.padding)  # `[B,20]/[B,26]` batch

    def sample_asset_blocks(
        self,
        *,
        assets_per_microbatch: int,
        q_per_asset: int,
        step: int,
    ) -> PaddedOnlineGeometryBatch:
        r"""按 `A_mb` 资产、每资产 `Q_mb` 构型一次生成逻辑 `A_mb*Q_mb` batch。

        资产按 manifest 顺序确定性轮转；每个资产只调用一次 batched query/target backend，随后
        才切成现有 padding 容器。因此这个接口是 multi-q runtime 的实际入口，旧 `sample` 保持
        `Q=1` 兼容路径供历史 tiny-overfit 使用。
        """

        if assets_per_microbatch < 1 or q_per_asset < 1 or step < 0:
            raise ValueError("assets_per_microbatch, q_per_asset and step must be positive/non-negative")
        samples: list[OnlineGeometrySample] = []
        for asset_offset in range(assets_per_microbatch):
            asset_index = (step * assets_per_microbatch + asset_offset) % len(self.states)
            state = self.states[asset_index]
            q_block = self.samplers[asset_index].draw(
                q_per_asset,
                device=state.spec.space_screws.device,
                dtype=state.spec.space_screws.dtype,
            )
            q_start = self.samplers[asset_index].cursor - q_per_asset  # block 起始 absolute cursor
            block = sample_online_geometry(
                state,
                q_block,
                field_config=self.field_config,
                query_config=self.query_config,
                target_config=self.target_config,
                sampling_seed=self.seed + step * assets_per_microbatch + asset_offset,
                q_index=torch.arange(q_start, q_start + q_per_asset),  # Q block asset-local provenance
            )
            samples.extend(split_online_geometry_sample(block))
        return pad_online_geometry_samples(samples, padding=self.padding)

    def state_dict(self) -> dict[str, object]:
        r"""返回每资产 Sobol cursor 与采样 seed，供 checkpoint runtime state 使用。"""

        return {
            "seed": self.seed,
            "asset_ids": tuple(state.source.asset_id for state in self.states),
            "samplers": tuple(sampler.state_dict() for sampler in self.samplers),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        r"""严格恢复与当前 manifest 同序的每资产 q cursor。"""

        seed = state.get("seed", -1)
        if not isinstance(seed, int) or seed != self.seed:
            raise ValueError("batcher checkpoint seed does not match resolved training seed")
        raw_asset_ids = state.get("asset_ids", ())
        if not isinstance(raw_asset_ids, (tuple, list)):
            raise ValueError("batcher checkpoint asset IDs must be a sequence")
        asset_ids = tuple(str(asset_id) for asset_id in raw_asset_ids)
        expected_ids = tuple(asset.source.asset_id for asset in self.states)
        if asset_ids != expected_ids:
            raise ValueError("batcher checkpoint asset order does not match manifest")
        sampler_states = state.get("samplers", ())
        if not isinstance(sampler_states, (tuple, list)) or len(sampler_states) != len(self.samplers):
            raise ValueError("batcher checkpoint sampler count does not match manifest")
        for sampler, sampler_state in zip(self.samplers, sampler_states):
            if not isinstance(sampler_state, dict):
                raise ValueError("invalid Sobol sampler checkpoint state")
            sampler.load_state_dict(sampler_state)


__all__ = [  # SSL data stage 稳定公开面
    "GeometryRepresentation",  # source/query/target runtime façade
    "GeometryRepresentationCfg",  # 正交组合配置
    "GeometryRepresentationState",  # GPU source + retained evidence
    "OnlineGeometryBatcher",  # online routing/teacher
    "OnlineGeometrySample",  # variable-length single sample
    "PaddedOnlineGeometryBatch",  # heterogeneous batch
    "SobolJointSampler",  # limits-only q sampler
    "pad_online_geometry_samples",  # variable -> dense masks
    "sample_online_geometry",  # q -> query/teacher
    "split_online_geometry_sample",  # Q block -> current padding oracle
]
