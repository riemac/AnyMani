r"""物理几何、语义分区、运动学与静态缓存的 source 层。

source 层把 generated manifest、official URDF/sidecar 与 collision evidence 整理为
candidate-neutral 的物理输入，但不生成网络 token。它明确分离四类信息：

1. collision surface / volume 的真实几何；
2. collision pieces 到 semantic geometry group 的人工可审计归属；
3. 当前 $q$、有序 screw chain、topology 与 home geometry；
4. 可以按 asset 缓存的静态加速结构与 provenance。

动态 command、contact、object state、history 与当前 posed-field label 都不属于纯几何
source 的最小输入。它们可能属于下游 policy observation，但不能因为 policy 需要而
泄漏进 SSL partial input。
"""

from .collision_geometry import (
    AnchorSamples,
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    OwnerSurfaceRecord,
    WarpOwnerGeometryCache,
    geometry_identity,
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    release_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
    sample_palm_anchor_supports,
)
from .geometry_source import DeviceGeometrySource, GeometrySource, GeometrySourceCfg
from .kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms,
    lower_hand_geometry_semantics,
    selected_point_jacobian,
    transform_owner_points,
)

__all__ = [
    "AnchorSamples",
    "EmbodimentGeometrySpec",
    "DeviceGeometrySource",
    "GeometryIdentity",
    "GeometrySource",
    "GeometrySourceCfg",
    "HomeSurfaceSamples",
    "OwnerGeometryCache",
    "OwnerSurfaceRecord",
    "WarpOwnerGeometryCache",
    "forward_owner_transforms",
    "geometry_identity",
    "lower_hand_geometry_semantics",
    "materialize_owner_geometry_cache",
    "materialize_warp_owner_geometry_cache",
    "release_warp_owner_geometry_cache",
    "sample_owner_home_surfaces",
    "sample_palm_anchor_supports",
    "selected_point_jacobian",
    "transform_owner_points",
]
