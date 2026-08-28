r"""PPO train partition 到 canonical heterogeneous scene routing 的装配边界。

本模块由 IsaacLab env cfg 在 AppLauncher 之后 import。``assets`` 负责只解析 ``ppo.yaml.train``，
``robots`` 负责 canonical URDF/USD cache 与 articulation cfg，``tasks`` 只保存 scene 必需的有序
asset rows、mask、q-home 和 contact layout。它不构造 frozen $Z$ 或 PPO network。
"""

from __future__ import annotations

import hashlib
import os

from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.prepared_train import resolve_prepared_train
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandGroupManifest
from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase
from anymani.robots.hand_spawn import CanonicalRuntimeCfg, HandSpawnAdapter, HandSpawnCfg, HandUrdfSpawnCfg

from ...contact_sensors import GmContactSensorLayout

PPO_DATASET_PATH = (
    resolve_anymani_root()
    / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo.yaml"
)
"""正式 2048-asset train selection manifest；路径不依赖 Hydra/shell cwd。"""

PPO_DATASET = HandAssetDataset.from_yaml(PPO_DATASET_PATH)
"""严格 schema-2 dataset 声明；此时只读取 YAML bytes。"""

FORMAL_PPO_ASSET_COUNT = 2048
"""`128 mothers + 128×15 variants` 的正式 train row 数。"""


def _asset_limit() -> int | None:
    r"""解析显式 smoke 前缀限制；formal route 默认返回 ``None``。"""

    raw = os.environ.get("ANYMANI_HETEROGENEOUS_ASSET_LIMIT")
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value < 1 or value > FORMAL_PPO_ASSET_COUNT:
        raise ValueError(
            "ANYMANI_HETEROGENEOUS_ASSET_LIMIT must be within "
            f"[1,{FORMAL_PPO_ASSET_COUNT}], got {value}"
        )
    return value


HETEROGENEOUS_ASSET_LIMIT = _asset_limit()
"""当前进程的显式 smoke asset count；``None`` 表示完整 2048。"""

record_optional_rl_phase(
    "asset_resolve_train",
    "start",
    requested_assets=HETEROGENEOUS_ASSET_LIMIT or FORMAL_PPO_ASSET_COUNT,
)
_TRAIN_PARTITION, HETEROGENEOUS_PREPARED_CACHE_HIT = resolve_prepared_train(
    PPO_DATASET,
    require_geometry_semantics=True,
    max_assets=HETEROGENEOUS_ASSET_LIMIT,
)
"""只展开 train bundles；smoke 前缀不会启动后续 lineage IO。"""
record_optional_rl_phase(
    "asset_resolve_train",
    "complete",
    resolved_assets=len(_TRAIN_PARTITION.assets),
    dataset_digest=PPO_DATASET.source_sha256,
    prepared_cache_hit=HETEROGENEOUS_PREPARED_CACHE_HIT,
)

HETEROGENEOUS_SOURCE_ASSETS = _TRAIN_PARTITION.assets
"""保持 ``ppo.yaml`` train 顺序的 source ``HandContainer`` 轴。"""

HETEROGENEOUS_HAND_SPAWN_CFG = HandSpawnCfg(
    urdf=HandUrdfSpawnCfg(
        activate_contact_sensors=True,
        use_stable_usd_cache=True,
        force_usd_conversion=False,
    ),
    canonical_runtime=CanonicalRuntimeCfg(
        enabled=True,
        output_root="outputs",
        schema_version=CANONICAL_HAND_SCHEMA_V1.version,
        validate_artifact=True,
    ),
    asset_routing="round_robin",
    restore_visual_materials=False,
    validate_same_schema=True,
)
"""统一 16-DOF/25-body PhysX batch 与 mesh-aware stable USD cache 配置。"""

HETEROGENEOUS_HAND_ADAPTER = HandSpawnAdapter(
    HETEROGENEOUS_HAND_SPAWN_CFG,
    resolved_assets=HETEROGENEOUS_SOURCE_ASSETS,
)
"""robots-owned runtime adapter；不会重新解析 dataset manifest。"""

record_optional_rl_phase("canonical_materialize", "start", asset_count=len(HETEROGENEOUS_SOURCE_ASSETS))
HETEROGENEOUS_CANONICAL_ARTIFACTS = HETEROGENEOUS_HAND_ADAPTER.canonical_artifacts
"""dataset row 顺序的 canonical URDF/routing manifests。"""
record_optional_rl_phase(
    "canonical_materialize",
    "complete",
    asset_count=len(HETEROGENEOUS_CANONICAL_ARTIFACTS),
)

HETEROGENEOUS_CANONICAL_ASSETS = HETEROGENEOUS_HAND_ADAPTER.selection.assets
"""与 artifacts 同序、供 MultiAssetSpawner 消费的 canonical virtual containers。"""

HETEROGENEOUS_ACTIVE_MASK_ROWS = tuple(
    artifact.routing.active_joint_mask for artifact in HETEROGENEOUS_CANONICAL_ARTIFACTS
)
"""`[A,16]` JSON-safe active action/observation masks。"""

HETEROGENEOUS_ASSET_ROWS = tuple(range(len(HETEROGENEOUS_CANONICAL_ARTIFACTS)))
"""frozen-$Z$/manifest 的离散 dataset row ``0..A-1``。"""

HETEROGENEOUS_Q_HOME_ROWS = tuple(artifact.routing.q_home for artifact in HETEROGENEOUS_CANONICAL_ARTIFACTS)
"""`[A,16]` canonical q-home；reset 时 active slots 取真实值，ghost 清零。"""

_GROUP_MANIFEST = CanonicalHandGroupManifest(
    schema_version=CANONICAL_HAND_SCHEMA_V1.version,
    schema_digest=CANONICAL_HAND_SCHEMA_V1.digest,
    artifacts=HETEROGENEOUS_CANONICAL_ARTIFACTS,
)
_GROUP_MANIFEST_PATH = (
    resolve_anymani_root()
    / "outputs/canonical_runtime"
    / CANONICAL_HAND_SCHEMA_V1.version
    / "groups"
    / PPO_DATASET.source_sha256
    / f"train-{len(HETEROGENEOUS_CANONICAL_ARTIFACTS)}.json"
)
_GROUP_MANIFEST.write(_GROUP_MANIFEST_PATH)  # 可审计项目派生物；缓存 hit 时内容保持 bitwise stable
HETEROGENEOUS_GROUP_MANIFEST_PATH = _GROUP_MANIFEST_PATH
"""当前完整/前缀 selection 的有序 canonical group manifest 路径。"""

HETEROGENEOUS_GROUP_MANIFEST_DIGEST = hashlib.sha256(_GROUP_MANIFEST_PATH.read_bytes()).hexdigest()
"""frozen $Z$ identity 与 checkpoint restore 使用的有序 manifest digest。"""


def _contact_layout() -> GmContactSensorLayout:
    r"""由 canonical v1 schema 构造固定 4 tip + 19 non-tip + palm layout。

    全 2048 数据的静态审计确认 ``thumb_root`` 恒为无 collision 的 frame adapter，因此不安装
    sensor；index/middle/ring roots 与四指各四个 joint links 组成 19 个可能碰撞槽。该定义不依赖
    smoke prefix 恰好含满形态资产；对低 DOF row，ghost/no-collision body 自然产生结构零。
    """

    finger_order = CANONICAL_HAND_SCHEMA_V1.physx_finger_order  # index/middle/ring/thumb
    finger_chains = tuple(
        (
            f"{finger}_root",
            *(f"{finger}_link_j{depth}" for depth in range(4)),
            f"{finger}_tip",
        )
        for finger in finger_order
    )
    tip_links = tuple(f"{finger}_tip" for finger in finger_order)  # `[4]`
    non_tip_links = (
        "index_root",
        "middle_root",
        "ring_root",
        *(f"{finger}_link_j{depth}" for finger in finger_order for depth in range(4)),
    )  # `[19]`；thumb_root 不带 collision
    return GmContactSensorLayout(
        source_asset_id="canonical-v1-collision-union",
        palm_link_name="palm",
        finger_link_chains=finger_chains,
        fingertip_link_names=tip_links,
        finger_non_tip_link_names=non_tip_links,
        fingertip_sensor_names=tuple(f"contact_{link}" for link in tip_links),
        finger_non_tip_sensor_names=tuple(f"contact_{link}" for link in non_tip_links),
    )


HETEROGENEOUS_CONTACT_LAYOUT = _contact_layout()
"""固定 24 个 object-filtered sensors：4 tips + 19 finger non-tips + neutral palm。"""


if HETEROGENEOUS_ASSET_LIMIT is None and len(HETEROGENEOUS_CANONICAL_ARTIFACTS) != 2048:
    raise ValueError(f"formal PPO dataset must resolve exactly 2048 assets, got {len(HETEROGENEOUS_CANONICAL_ARTIFACTS)}")


__all__ = [
    "HETEROGENEOUS_ACTIVE_MASK_ROWS",
    "HETEROGENEOUS_ASSET_ROWS",
    "HETEROGENEOUS_CANONICAL_ARTIFACTS",
    "HETEROGENEOUS_CONTACT_LAYOUT",
    "HETEROGENEOUS_GROUP_MANIFEST_DIGEST",
    "HETEROGENEOUS_GROUP_MANIFEST_PATH",
    "HETEROGENEOUS_HAND_ADAPTER",
    "HETEROGENEOUS_HAND_SPAWN_CFG",
    "HETEROGENEOUS_PREPARED_CACHE_HIT",
    "HETEROGENEOUS_Q_HOME_ROWS",
    "PPO_DATASET",
]
