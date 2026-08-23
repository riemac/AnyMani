r"""资产静态语义到 candidate-neutral physical geometry source 的物化生命周期。

``GeometrySourceCfg`` 只声明与 q 无关的物理证据 realization：逐 owner 基准表面点、每根手指
mount-conditioned palm anchors 以及确定性采样 seed。``GeometrySource`` 消费一项已由 HandBank
解析的 ``HandContainer``，组合本目录的纯张量运动学与 owner collision truth；它不知道 Gaussian
field、query mixture、decoder、loss、epoch 或 optimizer。

CPU source 对固定资产/config 是不可变 realization，可由进程内有界 arena 复用或在驱逐后精确重建。
``to_device()`` 只上传 POE/graph 张量并取得 Warp BVH lease，返回 ``DeviceGeometrySource``；resident
asset window 驱逐时必须调用 ``release()``。这种边界保证 source arena 可以同时服务 sampled implicit
field、固定 BPS 或未来其他读取布局，而不会把某一训练 stage 的 batch 语义写进物理真源。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class AnchorBankCfg:
    r"""每资产独立的有限 physical anchor constellation bank。

    训练按 q-block 均衡轮换 $A^{(k)}$；validation、独立 q-bank 与 PPO 固定 $A^{(0)}$。
    数值锚点：每指 10 个 anchors、$R_a=0.05\,\mathrm m$、$\tau_a=0.025\,\mathrm m$，
    surface/interior 各半，bank size $K=8$。
    """

    bank_size: int = 8  # 有限 Monte-Carlo realization 数
    anchors_per_finger: int = 10  # 每根 finger mount seed 的 physical anchor 数
    radius_m: float = 0.05  # palm seed 球形支持半径 $R_a$，单位 m
    radial_decay_scale_m: float = 0.025  # 截断 Gaussian $\tau_a=R_a/2$，单位 m
    surface_fraction: float = 0.5  # surface/interior 凸混合权重

    def __post_init__(self) -> None:
        r"""拒绝空 bank、非法米制半径和不可解释的采样混合。"""

        if self.bank_size < 1 or self.anchors_per_finger < 1:
            raise ValueError("anchor bank size and anchors_per_finger must be positive")
        if not 0.0 < self.radial_decay_scale_m <= self.radius_m:
            raise ValueError("anchor radial decay scale must lie in (0, radius_m]")
        if not 0.0 <= self.surface_fraction <= 1.0:
            raise ValueError("anchor surface_fraction must lie in [0,1]")


@dataclass(frozen=True)
class GeometrySourceCfg:
    r"""每项资产固定一次的静态 physical-source realization 配置。

    数值锚点为：每 owner 64 个 boundary points、home-surface 候选 oversample factor 8。
    workspace query 半径属于 query config，不与 anchor 支持半径隐式绑定。
    """

    home_points_per_owner: int = 64  # $M_g$；每个 owner 的 retained boundary-point 预算
    home_surface_oversample_factor: int = 8  # 面积 proposal 数为 $8M_g$，随后做 farthest-point selection
    static_sampling_seed: int = 0  # owner/finger ID 派生后的逐资产固定 seed
    anchors: AnchorBankCfg = field(default_factory=AnchorBankCfg)

    def __post_init__(self) -> None:
        r"""拒绝空点集。"""

        if self.home_points_per_owner < 1 or self.home_surface_oversample_factor < 1:
            raise ValueError("home-surface point and oversample budgets must be positive")

    @property
    def anchors_per_finger(self) -> int:
        r"""兼容旧字段读取：每指 anchor 数来自 nested bank cfg。"""

        return self.anchors.anchors_per_finger

    @property
    def anchor_bank_size(self) -> int:
        r"""兼容旧字段读取：bank size 来自 nested bank cfg。"""

        return self.anchors.bank_size

    @property
    def anchor_radius_m(self) -> float:
        return self.anchors.radius_m

    @property
    def anchor_radial_decay_scale_m(self) -> float:
        return self.anchors.radial_decay_scale_m

    @property
    def anchor_surface_fraction(self) -> float:
        return self.anchors.surface_fraction


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
    anchors: AnchorSamples  # `[K,3]` canonical $A^{(0)}$，validation/PPO 固定使用
    anchor_bank: tuple[AnchorSamples, ...]  # 有限 Monte-Carlo realization $\{A^{(0)},\ldots,A^{(K-1)}\}$
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

        # 每套 anchors 独立采样；bank 是 retained-input realization，不改变 teacher 几何。
        anchor_bank = tuple(
            sample_palm_anchor_supports(
                geometry_cache,
                semantics,
                spec,
                anchors_per_finger=config.anchors.anchors_per_finger,
                sampling_seed=config.static_sampling_seed + bank_index * 1_000_003,
                radial_support_radius_m=config.anchors.radius_m,
                radial_decay_scale_m=config.anchors.radial_decay_scale_m,
                surface_fraction=config.anchors.surface_fraction,
            )
            for bank_index in range(config.anchors.bank_size)
        )
        return cls(container, spec, geometry_cache, home_surface, anchor_bank[0], anchor_bank, identity)

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


__all__ = ["AnchorBankCfg", "DeviceGeometrySource", "GeometrySource", "GeometrySourceCfg"]
