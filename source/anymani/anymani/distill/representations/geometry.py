r"""物理 source、空间场、query 与 target 的在线 Geometry Representation 组合。

该模块不把 $(asset,q,query,target)$ 全量离线固化。每项资产只物化一次静态证据：owner union、
home boundary、anchors、owner triangle sampling table、kinematic spec 与 Warp BVH；训练 step
再采样合法 $q$，生成 workspace/shell/adjacent query 和 Warp teacher。

本模块不构造 encoder 输入，也不做跨结构 padding。retained evidence、选 $A^{(k)}$ 与稠密容器
属于 method 的 batch 适配层。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import torch

from anymani.distill.representations.queries.spatial_sampling import (
    OwnerSurfaceSamplingCache,
    SpatialQueryBatch,
    SpatialQuerySamplerCfg,
    materialize_owner_surface_sampling_cache,
    sample_spatial_queries,
)
from anymani.distill.representations.sources.anchor_sampling import AnchorClassificationStats
from anymani.distill.representations.sources.geometry_source import (
    DeviceGeometrySource,
    GeometrySource,
    GeometrySourceCfg,
    GeometrySourceCore,
)
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms_and_spatial_screws,
)
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
    r"""source、field、query 与 target 的正交物理组合；稠密 padding 由 method 从 dataset/model 推导。"""

    source: GeometrySourceCfg = dataclass_field(default_factory=GeometrySourceCfg)  # q-independent physical oracle
    field: GaussianProximityFieldCfg = dataclass_field(default_factory=GaussianProximityFieldCfg)  # sigma measure
    query: SpatialQuerySamplerCfg = dataclass_field(default_factory=SpatialQuerySamplerCfg)  # query measure
    target: GeometryFieldTargetCfg = dataclass_field(default_factory=GeometryFieldTargetCfg)  # edge/mask teacher


@dataclass(frozen=True)
class GeometryRepresentationState:
    r"""一项资产在指定 device 上供 query 与 teacher 共用的物理状态。

    不含 encoder evidence。method 按当前 $A^{(k)}$ 另造 retained 输入。
    """

    source: GeometrySource  # CPU physical truth 与 provenance
    device_source: DeviceGeometrySource  # GPU POE 与 Warp BVH lease
    surface_sampling: OwnerSurfaceSamplingCache  # owner triangle/normal/area proposal tables
    anchor_classification: AnchorClassificationStats | None = None  # CUDA finalization 证据；CPU source 为 None

    @property
    def anchor_realization(self):
        r"""返回 selected hot-path realization；legacy full source 为 ``None``。"""

        return self.source.anchor_realization

    @property
    def spec(self) -> EmbodimentGeometrySpec:
        r"""返回与 query/target 同 device/dtype 的动态运动学规格。"""

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

        self.config = config  # 完整 source/query/field/target 科研合同

    def materialize_source(self, container, *, anchor_device: str = "cpu") -> GeometrySource:
        r"""按 source config 物化 q-independent physical oracle，并显式选择 anchor classifier device。"""

        return GeometrySource.materialize(container, config=self.config.source, anchor_device=anchor_device)

    def materialize_core(self, container) -> GeometrySourceCore:
        r"""只物化 CPU geometry/home/identity，为主线程 CUDA anchor finalization 做准备。"""

        return GeometrySourceCore.materialize(container, config=self.config.source)

    def to_device(
        self,
        source: GeometrySource | GeometrySourceCore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        bank_index: int = 0,
    ) -> GeometryRepresentationState:
        r"""构造一项资产的 device source 与 surface proposal；core 输入同时完成 GPU anchors。"""

        if isinstance(source, GeometrySourceCore):
            finalized, device_source, anchor_stats = source.finalize_selected_on_device(
                config=self.config.source,
                bank_index=bank_index,
                device=device,
                dtype=dtype,
            )
        else:
            finalized = source
            device_source = source.to_device(device=device, dtype=dtype)
            anchor_stats = None
        return self.assemble_device_state(
            finalized,
            device_source,
            anchor_stats=anchor_stats,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def assemble_device_state(
        source: GeometrySource,
        device_source: DeviceGeometrySource,
        *,
        anchor_stats: AnchorClassificationStats | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> GeometryRepresentationState:
        r"""把已解析 anchor/source/device lease 组装为 query/teacher state。"""

        try:
            surface_sampling = materialize_owner_surface_sampling_cache(
                source.geometry_cache,
                device=torch.device(device),
                dtype=dtype,
                arrays=source.surface_sampling_arrays,
            )
            return GeometryRepresentationState(source, device_source, surface_sampling, anchor_stats)
        except Exception:
            device_source.release()
            raise

    def sample(
        self,
        state: GeometryRepresentationState,
        q: torch.Tensor,
        *,
        sampling_seed: int,
        q_index: torch.Tensor | None = None,
        anchor_index: int = 0,
        supervision_split: str = "train",
    ) -> PhysicalOnlineGeometrySample:
        r"""按当前配置为同资产 q realization 生成未 padding 物理 teacher。"""

        return sample_online_geometry(
            state,
            q,
            field_config=self.config.field,
            query_config=self.config.query,
            target_config=self.config.target,
            sampling_seed=sampling_seed,
            q_index=q_index,
            anchor_index=anchor_index,
            supervision_split=supervision_split,
        )


@dataclass(frozen=True)
class PhysicalOnlineGeometrySample:
    r"""一项资产、一个当前 $q$ 的未 padding 物理监督，不含 encoder 输入合同。"""

    asset_id: str  # bank 稳定路由 ID
    q: torch.Tensor  # `[Q,N_J]`，rad
    queries: SpatialQueryBatch  # `[Q,G,N_Q,...]`，`{h}`，m
    field_targets: FieldTargetBatch  # `[Q,G,N_Q,L]` $\rho$ 与 distance
    sensitivity_targets: SensitivityTargetBatch  # `[Q,E]` $\kappa$、$g$
    anchor_index: int = 0  # 本 q-block 选用的 $A^{(k)}$
    q_index: torch.Tensor | None = None  # `[Q]`，资产本地 Sobol cursor


def sample_online_geometry(
    state: GeometryRepresentationState,
    q: torch.Tensor,
    *,
    field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),
    query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),
    target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),
    sampling_seed: int = 0,
    q_index: torch.Tensor | None = None,
    anchor_index: int = 0,
    supervision_split: str = "train",
) -> PhysicalOnlineGeometrySample:
    r"""为一项资产的 ``[Q,N_J]`` q block 生成 query 与 Warp teacher。

    workspace 相对当前 $A^{(k)}$ 采样；teacher 几何不读取 encoder evidence。
    """

    if q.ndim != 2 or q.shape[1] != state.spec.space_screws.shape[0] or q.shape[0] < 1:
        raise ValueError("sample_online_geometry expects [Q,N_J] with the asset's true N_J")
    if q_index is not None and q_index.shape != (q.shape[0],):
        raise ValueError("q_index must have shape [Q] matching the asset q block")
    bank = state.source.anchor_bank
    if not bank:
        raise ValueError("geometry source is missing its physical anchor realization")
    realization = state.anchor_realization
    if realization is not None:
        if int(anchor_index) != realization.bank_index:
            raise IndexError(
                f"requested anchor_index={anchor_index} does not match resident bank={realization.bank_index}"
            )
        selected_anchors = realization.samples
    else:
        if not 0 <= int(anchor_index) < len(bank):
            raise IndexError(f"anchor_index={anchor_index} is outside bank size {len(bank)}")
        selected_anchors = bank[int(anchor_index)]
    anchors = torch.as_tensor(
        selected_anchors.anchors_hand_m,
        device=q.device,
        dtype=q.dtype,
    )  # `[K,3]`，`{h}`，m；$A^{(k)}$ 只改变 workspace 测度
    owner_transforms, current_spatial_screws = forward_owner_transforms_and_spatial_screws(
        state.spec,
        q.detach(),
    )  # 主路径一次 q-block FK，同时得到 Jacobian 所需当前 screw
    queries = sample_spatial_queries(
        q,
        state.spec,
        state.surface_sampling,
        anchors,
        config=query_config,
        sampling_seed=sampling_seed,
        owner_transforms=owner_transforms,
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
        supervision_split=supervision_split,  # train 1+1 / validation 4+4
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
        q_index=q_index,
    )
    return PhysicalOnlineGeometrySample(
        asset_id=state.source.asset_id,
        q=q,
        queries=queries,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        anchor_index=int(anchor_index),
        q_index=q_index.detach().cpu() if q_index is not None else None,
    )


__all__ = [
    "GeometryRepresentation",
    "GeometryRepresentationCfg",
    "GeometryRepresentationState",
    "PhysicalOnlineGeometrySample",
    "sample_online_geometry",
]
