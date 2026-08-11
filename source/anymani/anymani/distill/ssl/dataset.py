r"""多资产在线几何 SSL 数据物化、Sobol q 采样与跨结构 padding。

该模块不把 $(asset,q,query,target)$ 全量离线固化。每项资产只物化一次静态证据：owner union、
home boundary、anchors、workspace bank、robots kinematic spec 与 Warp BVH；训练 step 再从 joint limits
用 scrambled Sobol 采样合法 q，生成当前 shell/adjacent query 和 Warp teacher。

跨结构 batch 使用明确上限：最多 20 个 JOINT、5 个 TIP、26 个 owner。padding 只是 GPU 稠密容器：

- 实际 q/screw/owner/query/edge 复制到前缀有效槽；
- entity/joint/field/edge mask 显式记录有效范围；
- padding token、target 与 prediction 在模型/损失边界清零；
- 无 padding 的逐结构前向保留为输出/梯度 oracle。

单资产、pre-made 母体及其 post-mutate variants、同/跨 family generated 资产都走同一接口。
official 资产是否进入训练不由本模块猜测；实验配置必须在 HandBank selection 和 split manifest 中排除。
"""

from __future__ import annotations  # 前向类型引用不在 import 时求值

from dataclasses import dataclass  # 静态资产状态、在线样本与 padding batch 均冻结

import torch  # Sobol、张量 padding、GPU evidence 与 target

from anymani.assets.bank import HandContainer  # assets -> robots/distill 唯一 bundle 边界
from anymani.distill.models.input_adapters.geometry import (  # retained 静态输入与 padding
    GeometryPaddingCfg,  # 20 JOINT/5 TIP/26 owner 上限
    StaticGeometryEvidence,  # anchors/home/screws/graph/masks
    build_static_geometry_evidence,  # assets+robots -> model evidence
    pad_static_geometry_evidence,  # 跨结构静态轴 padding
)
from anymani.distill.representations.queries.spatial_sampling import (  # 50/25/25 query 测度
    SpatialQueryBatch,  # query 坐标/stratum/adjacent provenance
    SpatialQuerySamplerCfg,  # $N_W/N_S/N_A$
    build_workspace_query_bank,  # 跨 q 固定 `{h}` workspace
    sample_spatial_queries,  # 当前 q owner-shell/adjacent
)
from anymani.distill.representations.targets.field_samples import (  # 类型化 $d/\\rho/\\kappa/g$ targets
    FieldTargetBatch,
    SensitivityTargetBatch,
)
from anymani.distill.representations.targets.geometry_field import (  # Warp teacher assembly
    GeometryFieldTargetCfg,  # 带宽、edges/mask thresholds
    generate_geometry_field_targets,  # online target 主路径
)
from anymani.robots.geometry_kinematics import (  # 静态语义 -> 动态 POE spec
    EmbodimentGeometrySpec,
    lower_hand_geometry_semantics,
)
from anymani.robots.owner_geometry import (  # owner union 与 CPU/GPU geometry caches
    AnchorSamples,  # palm surface/interior supports
    HomeSurfaceSamples,  # owner boundary-only samples
    OwnerGeometryCache,  # CPU strict Manifold union
    WarpOwnerGeometryCache,  # GPU owner BVHs
    materialize_owner_geometry_cache,  # collision components -> owner union
    materialize_warp_owner_geometry_cache,  # owner union -> Warp mesh
    sample_owner_home_surfaces,  # area candidates + farthest point
    sample_palm_anchor_supports,  # seed-neighborhood palm supports
)


@dataclass(frozen=True)  # resolved 后禁止训练中修改静态采样测度
class GeometryAssetMaterializationCfg:  # CPU cache 数值配置
    r"""每项资产固定一次的 static geometry 采样配置。

    数值锚点：每 owner 64 个 boundary points、每 finger 10 个 palm anchors、anchor 支持半径 5 cm，
    surface/interior 各半；workspace AABB 外扩 3 cm。正式实验可 override，但 resolved config 必须记录。
    """

    home_points_per_owner: int = 64  # $M_g$
    anchors_per_finger: int = 10  # 首个可运行 anchor 数值锚点
    anchor_radius_m: float = 0.05  # palm seed 支持半径，m
    anchor_surface_fraction: float = 0.5  # surface/interior 各半
    static_sampling_seed: int = 0  # owner-aware 派生后逐资产固定
    workspace_padding_m: float = 0.03  # home geometry AABB 外扩，m

    def __post_init__(self) -> None:
        r"""验证静态点预算、米制半径与 surface/interior 混合比例。"""

        if self.home_points_per_owner < 1 or self.anchors_per_finger < 1:  # 点集不可为空
            raise ValueError("home/anchor point budgets must be positive")  # encoder 集合合同
        if self.anchor_radius_m <= 0.0 or self.workspace_padding_m < 0.0:  # 米制距离域
            raise ValueError("anchor radius must be positive and workspace padding non-negative")  # fail-fast
        if not 0.0 <= self.anchor_surface_fraction <= 1.0:  # convex mixture 权重
            raise ValueError("anchor_surface_fraction must lie in [0,1]")  # 不 clamp 改变测度


@dataclass(frozen=True)  # CPU 静态证据按资产内容冻结
class GeometryAssetRuntime:  # q-independent asset state
    r"""一项资产的 CPU 静态物化结果。

    该对象与 q/step 无关，可跨所有训练 step 复用；CPU meshes 保留真实 triangles/provenance，只有
    ``spec/evidence/workspace`` 张量在上传时转换 dtype/device。
    """

    container: HandContainer  # bank bundle + geometry semantics
    spec_cpu: EmbodimentGeometrySpec  # CPU float32 POE/graph/component transforms
    geometry_cache: OwnerGeometryCache  # owner-local strict unions
    home_surface: HomeSurfaceSamples  # `[G,M,3]` boundary-only owner local
    anchors: AnchorSamples  # `[K,3]` hand-frame palm supports
    workspace_query_bank_h: torch.Tensor  # `[N_W,3]`，CPU float32，跨 q 固定
    evidence_cpu: StaticGeometryEvidence  # CPU retained evidence reference

    @property
    def asset_id(self) -> str:
        r"""返回 bank 稳定资产 ID，供 batch routing/logging 使用。"""

        return self.container.asset_id  # 不以路径 basename 重新推断


@dataclass(frozen=True)  # GPU cache identity 不随 step 变化
class GeometryAssetDeviceState:  # device-resident asset state
    r"""一项资产驻留指定 GPU 的训练状态。

    ``runtime`` 继续持有 CPU meshes；``spec/evidence`` 是 PyTorch CUDA tensors；``warp_cache`` 持有
    Warp vertices/indices/BVH 强引用。该对象不含当前 q 或 learned activation。
    """

    runtime: GeometryAssetRuntime  # CPU provenance/cache owner
    spec: EmbodimentGeometrySpec  # GPU POE/graph tensors
    warp_cache: WarpOwnerGeometryCache  # GPU owner BVHs
    evidence: StaticGeometryEvidence  # GPU retained static inputs


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
        self.engine = torch.quasirandom.SobolEngine(  # 连续 draw 保留低差异序列状态
            dimension=self.limits.shape[0],  # $N_J$ 随资产变化
            scramble=True,  # Owen scrambling 提供 seed 可复现随机化
            seed=int(seed),  # 每资产独立派生 seed
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
        return q.to(device=device, dtype=dtype)  # 只在最终边界上传/转换


def materialize_geometry_asset_runtime(
    container: HandContainer,  # require_geometry_semantics=True 的 bank bundle
    *,
    query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),  # workspace $N_W$
    config: GeometryAssetMaterializationCfg = GeometryAssetMaterializationCfg(),  # 点预算
    ) -> GeometryAssetRuntime:  # CPU static asset materialization
    r"""从 bank container 构造一项资产全部 CPU 静态 cache。

    顺序是静态语义 lowering → owner Boolean union → boundary home points → palm anchors → fixed workspace
    bank → retained evidence。前一阶段失败不得被后一阶段近似替代。

    Returns:
        GeometryAssetRuntime: 与 q/step 无关、可复用的 CPU 静态状态。
    """

    semantics = container.geometry_semantics  # assets 层唯一静态语义入口
    if semantics is None:  # distill 不允许自己解析 hand.yaml/URDF
        raise ValueError("container must be resolved with require_geometry_semantics=True")  # bank 配置错误
    spec = lower_hand_geometry_semantics(semantics, dtype=torch.float32)  # CPU robots 动态规格
    geometry_cache = materialize_owner_geometry_cache(container, spec)  # 严格 owner union
    home_surface = sample_owner_home_surfaces(  # 真实 union boundary，不含 interior
        geometry_cache,  # `[G]` owner unions
        points_per_owner=config.home_points_per_owner,  # 每 owner 固定 $M$
        sampling_seed=config.static_sampling_seed,  # owner ID 派生可复现 seed
    )
    anchors = sample_palm_anchor_supports(  # 每 finger seed 邻域内 palm surface/interior supports
        geometry_cache,  # palm union solid
        semantics,  # anchor seeds 与 `{a}->{h}`
        spec,  # palm owner home transform
        anchors_per_finger=config.anchors_per_finger,  # 每 seed 固定数量
        sampling_seed=config.static_sampling_seed,  # 可复现实例
        radial_support_radius_m=config.anchor_radius_m,  # 球形支持半径，m
        surface_fraction=config.anchor_surface_fraction,  # surface/interior 比例
    )
    workspace = build_workspace_query_bank(  # 固定 `{h}` 采样，不能随 q 共动
        geometry_cache,  # owner 轴与资产 AABB provenance
        spec,  # owner home poses
        home_surface,  # boundary samples 构造 hand AABB
        query_count=query_config.stratum_counts[0],  # $N_W$
        padding_m=config.workspace_padding_m,  # AABB 外扩，m
        sampling_seed=config.static_sampling_seed,  # 固定 workspace realization
    )
    evidence = build_static_geometry_evidence(  # retained encoder 的全部静态输入
        semantics,  # owner roles/normal
        spec,  # screws/q_home/graph
        home_surface,  # `[G,M,3]`
        anchors,  # `[K,3]`
        device="cpu",  # CPU reference/caching
        dtype=torch.float32,  # 正式训练基础 dtype
    )
    return GeometryAssetRuntime(  # 把同一内容哈希下各静态证据绑定
        container,
        spec,
        geometry_cache,
        home_surface,
        anchors,
        workspace,
        evidence,
    )


def move_geometry_asset_to_device(
    runtime: GeometryAssetRuntime,  # CPU 静态状态
    *,
    device: torch.device | str = "cuda:0",  # GPU/Warp target device
    dtype: torch.dtype = torch.float32,  # model/spec/evidence dtype
    ) -> GeometryAssetDeviceState:  # GPU/Warp resident state
    r"""上传 kinematic/evidence 张量并构造一次 GPU Warp BVH。

    Warp cache 按 ``(asset_content_hash,device)`` 在 robots 层复用；本函数不复制 CPU triangles，也不在
    每个训练 batch 重建 BVH。
    """

    target_device = torch.device(device)  # 规范化 `cuda`/`cuda:0`
    spec = runtime.spec_cpu.to(device=target_device, dtype=dtype)  # POE/graph 张量上传
    warp_cache = materialize_warp_owner_geometry_cache(  # owner BVH 上传或 hash cache hit
        runtime.geometry_cache, device=str(target_device)
    )
    semantics = runtime.container.geometry_semantics  # evidence roles/normal 来源
    if semantics is None:  # frozen runtime 理论上不应丢失 container 语义
        raise ValueError("runtime container lost geometry semantics")  # 防御性一致性闸门
    evidence = build_static_geometry_evidence(  # 直接在目标 device 构造，避免每 batch H2D
        semantics,  # roles/normal
        spec,  # GPU screws/q_home/graph
        runtime.home_surface,  # CPU numpy -> GPU tensor
        runtime.anchors,  # CPU numpy -> GPU tensor
        device=target_device,  # resident device
        dtype=dtype,  # 与 model 一致
    )
    return GeometryAssetDeviceState(runtime, spec, warp_cache, evidence)  # 完整 GPU asset state


def sample_online_geometry(
    state: GeometryAssetDeviceState,  # 当前资产 GPU state
    q: torch.Tensor,  # `[1,N_J]`，rad
    *,
    query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),  # 50/25/25
    target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),  # teacher
    sampling_seed: int = 0,  # shell/adjacent/edge sampling realization
    ) -> OnlineGeometrySample:  # unpadded one-asset teacher sample
    r"""为一项资产的 ``[1,N_J]`` q 生成 query 与 Warp teacher。

    query 与 teacher 都从 ``q.detach()`` 的物理构型生成；模型对 q 的 Sobolev 图在 trainer 中另建，
    因此 teacher 几何路径不接收模型梯度。
    """

    if q.shape != (1, state.spec.space_screws.shape[0]):  # 当前接口一资产一 q
        raise ValueError("sample_online_geometry expects one q with the asset's true N_J")  # 不 padding 此层
    queries = sample_spatial_queries(  # 当前 q 下 `[1,G,N_Q,3]`
        q,  # 物理 rad
        state.spec,  # owner FK/graph
        state.runtime.geometry_cache,  # owner boundary
        state.runtime.home_surface,  # shell/adjacent 候选
        state.runtime.workspace_query_bank_h,  # 固定 workspace
        config=query_config,  # stratum 比例/壳厚
        sampling_seed=sampling_seed,  # 当前 realization
    )
    field_targets, sensitivity_targets = generate_geometry_field_targets(  # GPU Warp teacher
        q,  # 当前 owner transforms/Jacobian
        state.spec,  # POE/ancestor masks
        state.runtime.geometry_cache,  # CPU face/component provenance
        state.warp_cache,  # GPU BVHs
        queries,  # 当前 query batch
        config=target_config,  # $\\sigma_\\ell$/edges/margins
        edge_sampling_seed=sampling_seed,  # sampled `(g,r,i)` realization
    )
    return OnlineGeometrySample(  # 保留真实 variable lengths，padding 延后
        asset_id=state.runtime.asset_id,  # batch route
        q=q,  # `[1,N_J]`
        evidence=state.evidence,  # unbatched static evidence
        queries=queries,  # `[1,G,N_Q,...]`
        field_targets=field_targets,  # zero-order teacher
        sensitivity_targets=sensitivity_targets,  # sampled-edge teacher
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
    bandwidths = samples[0].field_targets.bandwidths  # 固定 `[L]`，m
    if any(sample.q.device != device or sample.q.dtype != dtype for sample in samples):  # device/dtype 一致性
        raise ValueError("all online samples must share device and dtype")  # 禁止隐式 copy/cast
    if any(sample.queries.query_points_h.shape[2] != query_count for sample in samples):  # query 轴
        raise ValueError("all samples must share N_Q")  # 当前 decoder 稠密 query 轴不 padding
    if any(not torch.equal(sample.field_targets.bandwidths, bandwidths) for sample in samples):  # 物理 $\\sigma$
        raise ValueError("all samples must share physical bandwidths")  # 禁止同通道不同单位/尺度

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
    distance = torch.zeros(  # `[B,26,N_Q]` unsigned owner distance，m
        batch_size, max_owner_count, query_count, device=device, dtype=dtype
    )
    density = torch.zeros(  # `[B,26,N_Q,L]`，无量纲
        batch_size, max_owner_count, query_count, bandwidths.numel(), device=device, dtype=dtype
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
        batch_size, max_edge_count, bandwidths.numel(), device=device, dtype=dtype
    )
    edge_valid = torch.zeros(  # `[B,E_max]`；padding/non-smooth edge loss mask
        batch_size, max_edge_count, device=device, dtype=torch.bool
    )

    for batch_index, sample in enumerate(samples):  # 每项独立真实 $N_J/G/E$
        joint_count = sample.q.shape[1]  # 当前 $N_J$
        owner_count = sample.queries.query_points_h.shape[1]  # 当前 $G$
        edge_count = sample.sensitivity_targets.kappa.shape[1]  # 当前 $E$
        q[batch_index, :joint_count] = sample.q[0]  # rad q 写入 `[0:N_J)`
        query_points[batch_index, :owner_count] = sample.queries.query_points_h[0]  # `{h}` query
        query_stratum[batch_index, :owner_count] = sample.queries.query_stratum[0]  # 0/1/2 provenance
        adjacent_owner[batch_index, :owner_count] = sample.queries.adjacent_owner_index[0]  # neighbor owner
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

    queries = SpatialQueryBatch(  # decoder/sampler provenance 包
        query_points, query_stratum, adjacent_owner
    )
    field_targets = FieldTargetBatch(  # zero-order target + valid normalization mask
        query_points=query_points,  # `[B,26,N_Q,3]`，`{h}`，m
        query_stratum=query_stratum,  # 不进入 decoder
        distance=distance,  # `[B,26,N_Q]`，m
        density=density,  # `[B,26,N_Q,L]`，无量纲
        valid_mask=field_valid,  # invalid owner/query 全 False
        owner_role=owner_role,  # `[B,26]`
        bandwidths=bandwidths,  # `[L]`，m
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
    )


class OnlineGeometryBatcher:  # deterministic multi-asset online sampler
    r"""在预物化资产间轮转、采 q/target 并输出跨结构 padding batch。

    资产选择使用确定性 round-robin：第 ``step`` 个 batch 的槽 $b$ 选择
    $(step\cdot B+b)\bmod N_{asset}$。每项资产拥有独立持续 SobolEngine，因此改变其他资产 DOF
    不会改变本资产 q 序列维度或 limits。
    """

    def __init__(
        self,
        states: list[GeometryAssetDeviceState],  # 预物化 generated GPU assets
        *,
        seed: int,  # Sobol/query/edge 总种子
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
        self.query_config = query_config  # stratum 与 shell 数值
        self.target_config = target_config  # bandwidth/edge/margin
        self.padding = padding  # 稠密容器上限
        self.samplers = tuple(  # 每资产独立维度/limits/Sobol state
            SobolJointSampler(  # CPU engine，只把 draw 结果上传
                state.runtime.spec_cpu, seed=self.seed + asset_index  # 稳定派生 seed
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
            samples.append(  # 完整在线 query/teacher 未 padding 样本
                sample_online_geometry(  # Warp GPU main path
                    state,  # 当前 asset
                    q,  # `[1,N_J]` rad
                    query_config=self.query_config,  # 50/25/25
                    target_config=self.target_config,  # $d/\\rho/\\kappa/g$
                    sampling_seed=self.seed + step * batch_size + batch_offset,  # 唯一 realization
                )
            )
        return pad_online_geometry_samples(samples, padding=self.padding)  # `[B,20]/[B,26]` batch


__all__ = [  # SSL data stage 稳定公开面
    "GeometryAssetDeviceState",  # GPU static asset
    "GeometryAssetMaterializationCfg",  # CPU sampling config
    "GeometryAssetRuntime",  # CPU static asset
    "OnlineGeometryBatcher",  # online routing/teacher
    "OnlineGeometrySample",  # variable-length single sample
    "PaddedOnlineGeometryBatch",  # heterogeneous batch
    "SobolJointSampler",  # limits-only q sampler
    "materialize_geometry_asset_runtime",  # bank -> CPU state
    "move_geometry_asset_to_device",  # CPU -> GPU/Warp state
    "pad_online_geometry_samples",  # variable -> dense masks
    "sample_online_geometry",  # q -> query/teacher
]
