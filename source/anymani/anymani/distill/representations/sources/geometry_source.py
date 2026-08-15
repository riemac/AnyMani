r"""资产静态语义到 candidate-neutral physical geometry source 的物化生命周期。

``GeometrySourceCfg`` 只声明与 q 无关的物理证据 realization：逐 owner 基准表面点、每根手指
mount-conditioned palm anchors 以及确定性采样 seed。``GeometrySource`` 消费一项已由 HandBank
解析的 ``HandContainer``，组合本目录的纯张量运动学与 owner collision truth；它不知道 Gaussian
field、query mixture、decoder、loss、epoch 或 optimizer。

CPU source 在一项资产的整个实验生命周期内保持不变。``to_device()`` 只上传 POE/graph 张量并取得
Warp BVH lease，返回 ``DeviceGeometrySource``；resident asset window 驱逐时必须调用 ``release()``。
这种边界保证 source cache 可以同时服务 sampled implicit field、固定 BPS 或未来其他读取布局，而不会
把某一训练 stage 的 batch 语义写进物理真源。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.assets.bank import HandContainer

from .collision_geometry import (
    AnchorSamples,
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    WarpOwnerGeometryCache,
    geometry_identity,
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    release_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
    sample_palm_anchor_supports,
)
from .kinematics import EmbodimentGeometrySpec, lower_hand_geometry_semantics


@dataclass(frozen=True)
class GeometrySourceCfg:
    r"""每项资产固定一次的静态 physical-source realization 配置。

    数值锚点为：每 owner 64 个 boundary points、home-surface 候选 oversample factor 8、每根
    finger 10 个 anchors、支持半径 $R_a=0.05\,\mathrm m$、径向衰减尺度
    $\tau_a=0.025\,\mathrm m$，surface/interior 各半。workspace query 半径属于 query config，
    不与 anchor 支持半径隐式绑定。
    """

    home_points_per_owner: int = 64  # $M_g$；每个 owner 的 retained boundary-point 预算
    home_surface_oversample_factor: int = 8  # 面积 proposal 数为 $8M_g$，随后做 farthest-point selection
    anchors_per_finger: int = 10  # 每根 finger mount seed 的 physical anchor 数
    anchor_radius_m: float = 0.05  # palm seed 球形支持半径 $R_a$，单位 m
    anchor_radial_decay_scale_m: float = 0.025  # 截断 Gaussian $\tau_a=R_a/2$，单位 m
    anchor_surface_fraction: float = 0.5  # anchor proposal 中 surface/interior 的凸混合权重
    static_sampling_seed: int = 0  # owner/finger ID 派生后的逐资产固定 seed

    def __post_init__(self) -> None:
        r"""拒绝空点集、非法米制半径和不可解释的采样混合。"""

        if self.home_points_per_owner < 1 or self.home_surface_oversample_factor < 1:  # surface evidence 非空
            raise ValueError("home-surface point and oversample budgets must be positive")
        if self.anchors_per_finger < 1:  # 每个声明的 finger seed 必须产生至少一个物理 landmark
            raise ValueError("anchors_per_finger must be positive")
        if not 0.0 < self.anchor_radial_decay_scale_m <= self.anchor_radius_m:  # $0<\tau_a\le R_a$
            raise ValueError("anchor radial decay scale must lie in (0, anchor_radius_m]")
        if not 0.0 <= self.anchor_surface_fraction <= 1.0:  # 不 clamp 改变声明的物理测度
            raise ValueError("anchor_surface_fraction must lie in [0,1]")


@dataclass(frozen=True)
class GeometrySource:
    r"""一项资产在 CPU 上可跨 representation/stage 复用的静态物理真源。

    ``spec_cpu`` 使用 float64 保存 POE、图关系与 component-to-owner transforms，作为 physical
    identity 的数值真值；home surface 与 anchors 保存实际 realization/provenance。该对象不包含
    当前 q、query、field labels、learned activation 或训练 cursor。
    """

    container: HandContainer  # bank bundle 与版本化 geometry semantics
    spec_cpu: EmbodimentGeometrySpec  # CPU float64 POE/graph/component transforms
    geometry_cache: OwnerGeometryCache  # owner-local strict surface/solid union
    home_surface: HomeSurfaceSamples  # `[G,M,3]` owner-local boundary-only realization
    anchors: AnchorSamples  # `[K,3]` hand-frame palm surface/interior supports
    identity: GeometryIdentity  # physical mapping 与 configuration-domain 双重身份

    @property
    def asset_id(self) -> str:
        r"""返回 HandBank 稳定资产 ID，供 split、routing 与日志共同使用。"""

        return self.container.asset_id  # 不从目录 basename 重新猜测身份

    @classmethod
    def materialize(
        cls,
        container: HandContainer,
        *,
        config: GeometrySourceCfg = GeometrySourceCfg(),
    ) -> GeometrySource:
        r"""从一项 bank container 构造全部 CPU physical-source evidence。

        顺序固定为 semantic lowering、owner Boolean union、physical identity、boundary home points
        与 palm anchors。任何一步失败都拒绝该资产，不使用 hull、补洞或匿名 fallback。

        Args:
            container (HandContainer): 以 ``require_geometry_semantics=True`` 解析的资产 bundle。
            config (GeometrySourceCfg): q-independent source realization 参数。

        Returns:
            GeometrySource: 与 q/step 无关、可跨 representation 复用的 CPU source。
        """

        semantics = container.geometry_semantics  # assets 层交付的唯一静态 owner/frame 真源
        if semantics is None:  # source 不自行重读 hand.yaml 或 URDF
            raise ValueError("container must be resolved with require_geometry_semantics=True")

        # 先以 CPU float64 lower 运动学与 collision carrier，确保 hash/audit 不受训练 dtype 影响。
        spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)  # `[N_J,6]` 等静态张量
        geometry_cache = materialize_owner_geometry_cache(container, spec)  # 真实逐 owner union surface
        identity = geometry_identity(semantics, spec, geometry_cache)  # physical/configuration identity 分离

        # retained home points 只来自真实 owner boundary；oversample factor 是 realization provenance。
        home_surface = sample_owner_home_surfaces(
            geometry_cache,
            points_per_owner=config.home_points_per_owner,
            sampling_seed=config.static_sampling_seed,
            oversample_factor=config.home_surface_oversample_factor,
        )

        # anchors 允许 palm interior，但支持域、径向测度和 surface 比例全部由 facade 显式声明。
        anchors = sample_palm_anchor_supports(
            geometry_cache,
            semantics,
            spec,
            anchors_per_finger=config.anchors_per_finger,
            sampling_seed=config.static_sampling_seed,
            radial_support_radius_m=config.anchor_radius_m,
            radial_decay_scale_m=config.anchor_radial_decay_scale_m,
            surface_fraction=config.anchor_surface_fraction,
        )
        return cls(container, spec, geometry_cache, home_surface, anchors, identity)  # 同一 asset hash 的真源包

    def to_device(
        self,
        *,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> DeviceGeometrySource:
        r"""上传运动学张量并取得该资产在指定 device 上的 Warp BVH lease。

        Args:
            device (torch.device | str): 在线 query/target 与模型共同使用的 CUDA device。
            dtype (torch.dtype): 训练张量 dtype；当前正式路径使用 float32。

        Returns:
            DeviceGeometrySource: 不含 query/field/model state 的 device-resident source。
        """

        target_device = torch.device(device)  # 规范化 `cuda`/`cuda:0` 设备身份
        spec = self.spec_cpu.to(device=target_device, dtype=dtype)  # POE/graph 张量一次上传
        warp_cache = materialize_warp_owner_geometry_cache(  # 按 surface hash/version/device 取得 lease
            self.geometry_cache,
            device=str(target_device),
        )
        return DeviceGeometrySource(source=self, spec=spec, warp_cache=warp_cache)  # 物理 source device view


@dataclass(frozen=True)
class DeviceGeometrySource:
    r"""一项资产的 device-resident kinematics 与 closest-surface backend lease。"""

    source: GeometrySource  # CPU meshes、realization 与 provenance owner
    spec: EmbodimentGeometrySpec  # GPU POE/graph tensors
    warp_cache: WarpOwnerGeometryCache  # GPU owner BVHs 与 face provenance

    def release(self) -> bool:
        r"""释放本 device source 持有的全局 Warp cache lease。"""

        return release_warp_owner_geometry_cache(self.warp_cache)  # lease 归零后移除全局强引用


__all__ = ["DeviceGeometrySource", "GeometrySource", "GeometrySourceCfg"]
