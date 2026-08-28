r"""Generated hand spawn adapter.

本模块是 `robots` 侧的 **IsaacLab runtime spawn 适配层**。它的长期职责是把
`assets.bank` 选出的 generated hand assets 包装成 IsaacLab 可消费的
`ArticulationCfg`，供 `tasks/gm` 与后续 `distill` runtime smoke 共同复用。

当前文件实现第一版 **URDF runtime spawn adapter**：字段、公式、接口、职责边界、
bank resolve、schema check、URDF importer cfg、可选 USD material restore 与 root pose
anchor lower。orientation reset 仍不在本文件实现，它属于 `mdp/events.py` 的 episode
级 reset 语义。

设计目标：

```text
HandSpawnCfg
  ├─ bank: HandBankCfg                  # assets 层资产选择配置
  ├─ frame: HandFrameCfg                # {a}->{h} 语义对齐与默认 T_eh anchor
  ├─ urdf: HandUrdfSpawnCfg             # URDF importer 参数
  ├─ actuator: HandActuatorSpawnCfg     # implicit actuator 参数
  └─ ...

HandSpawnAdapter(cfg)
  ├─ selection                          # lazy: HandBank(cfg.bank).resolve()
  ├─ build_articulation_cfg(...)        # -> ArticulationCfg
  ├─ build_multi_hand_spawn_cfg(...)    # -> MultiAssetSpawnerCfg
  └─ semantic_R_ha                      # env cfg 显式同步给 command cfg
```

边界约定：

- `assets.bank` 负责路径解析、资产选择、虚拟 bundle、URDF mesh / color 解析；
- `robots.hand_spawn` 只负责把已选 embodiment lower 成 IsaacLab robot cfg；
- `tasks/gm` 只消费 robot cfg 来表达 MDP，不拥有资产生成或训练算法；
- `distill` 负责训练时选用哪个 task / agent YAML / checkpoint / manifest。

Frame 语义：

- `{a}`：raw asset/root frame，即 URDF/USD 被 IsaacLab 加载后的资产根坐标系；
- `{h}`：hand semantic frame，任务语义使用的手坐标系；
- `semantic_R_ha` 表示 $R_{ha}$，即 $v^h = R_{ha}v^a$；
- `semantic_p_ha` 表示 $p_{ha}$，即 `{a}` 原点在 `{h}` 中的位置；
- `anchor_R_eh` / `anchor_p_eh` 表示 reset / spawn 的默认 hand semantic pose
  $T_{eh}^{anchor}$；
- 第一版默认目标是让 hand semantic frame 初始满足 $R_{wh}=I$，即
  $R_{eh}^{anchor}=I$。

在 IsaacLab cloned env 默认只相对 world 平移、无旋转的假设下，默认 anchor 对应：

$$
T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha},\qquad
R_{ea}^{anchor}=R_{eh}^{anchor}R_{ha},\qquad
p_{ea}^{anchor}=p_{eh}^{anchor}+R_{eh}^{anchor}p_{ha}.
$$

当前默认 $R_{eh}^{anchor}=I$，所以退化为 $R_{ea}=R_{ha}$，
$p_{ea}=p_{eh}+p_{ha}$。episode 级任意 hand orientation 由 `events.py` 中的
orientation reset scaffold 表达：采样 $\Delta R_h$ 并右乘到 anchor 上，而不是
在 spawn 层局部随机化 root pose。

TOAGENT: 本文件只实现 spawn/bank adapter。episode 级 orientation reset、object reset、
Grasp Cache 和 command update 不要塞进这里。
"""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import field
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass

from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot
from anymani.assets.bank import HandBank, HandBankCfg, HandContainer, HandSelection
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.urdf_utils import parse_urdf_visual_rgba_by_name
from anymani.assets.canonical_runtime import (
    CANONICAL_HAND_SCHEMA_V1,
    CanonicalHandArtifact,
    CanonicalHandSchemaCfg,
    compute_canonical_startup_joint_positions,
    lower_hand_to_canonical,
    materialize_canonical_artifact,
    validate_canonical_artifact,
)
from anymani.robots._hand_schema import (
    validate_canonical_hand_schema as _validate_canonical_hand_schema,
)
from anymani.robots._hand_schema import validate_same_hand_schema as _validate_same_hand_schema
from anymani.robots._visual_materials import VisualMaterialRestorePlan as _VisualMaterialRestorePlan
from anymani.robots._visual_materials import (
    build_visual_material_restore_plan as _build_visual_material_restore_plan,
)
from anymani.robots._visual_materials import parse_urdf_visual_link_by_name as _parse_urdf_visual_link_by_name
from anymani.robots._visual_materials import (
    serialize_visual_material_restore_plan as _serialize_visual_material_restore_plan,
)
from anymani.robots._visual_materials import (
    spawn_urdf_with_restored_visual_materials as _spawn_urdf_with_restored_visual_materials_impl,
)
from anymani.robots.usd_cache import build_urdf_usd_cache_dir

DEFAULT_HAND_ANCHOR_POS_E = (0.0, 0.0, 0.5)
r"""默认 hand semantic origin anchor 在 env frame `{e}` 中的位置，单位 m。"""

IDENTITY_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
"""Row-major $3\times3$ identity rotation。"""

Vector3 = tuple[float, float, float]
"""三维平移/方向向量。"""

Matrix3 = tuple[Vector3, Vector3, Vector3]
"""固定维度 row-major $3\times3$ 矩阵；用于 frame composition 的静态 shape 证书。"""


def _spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    r"""保持既有 ``UrdfFileCfg.func`` 名称的 façade；实现位于 ``_visual_materials``。"""

    return _spawn_urdf_with_restored_visual_materials_impl(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


@configclass
class HandFrameCfg:
    r"""Hand raw asset frame `{a}` 与 hand semantic frame `{h}` 的对齐配置。

    配置层使用 $(R,p)$；未来实现层应组合为 $SE(3)$ 计算，最后仅在 IsaacLab 边界
    转成 quaternion。`semantic_*` 是资产校准 $T_{ha}$；`anchor_*` 是 hand semantic
    frame 在 env frame 中的默认参考 pose $T_{eh}^{anchor}$。reset-time orientation
    DR 应以该 anchor 为默认参考，而不是覆盖 `{h}` 的语义定义。
    """

    semantic_R_ha: tuple[float, ...] = IDENTITY_R
    r"""$R_{ha}$，row-major 9 个 float，语义为 $v^h=R_{ha}v^a$。"""

    semantic_p_ha: tuple[float, float, float] = (0.0, 0.0, 0.0)
    r"""$p_{ha}$，raw asset origin `{a}` 在 hand semantic frame `{h}` 中的位置，单位 m。"""

    anchor_R_eh: tuple[float, ...] = IDENTITY_R
    r"""$R_{eh}^{anchor}$，hand semantic frame `{h}` 在 env frame `{e}` 中的默认参考朝向。"""

    anchor_p_eh: tuple[float, float, float] = DEFAULT_HAND_ANCHOR_POS_E
    r"""$p_{eh}^{anchor}$，hand semantic origin `{h}` 在 env frame `{e}` 中的默认参考位置，单位 m。"""

    align_hand_frame_to_env: bool = True
    r"""是否按 `anchor_R_eh` / `anchor_p_eh` 自动推导 spawn root pose。

    第一版设计只支持 `True`：spawn 使用 $T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha}$。
    任意 hand orientation 的 episode 级采样应由 reset event 在该 anchor 上右乘扰动。
    """


@configclass
class HandJointInitCfg:
    r"""Hand articulation 初始关节状态配置。"""

    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    """默认关节位置，key 为 IsaacLab joint regex。"""

    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    """默认关节速度，key 为 IsaacLab joint regex。"""


@configclass
class HandUrdfSpawnCfg:
    r"""Generated hand URDF importer 参数 scaffold。

    数值锚点来自 `heterogeneous_test_env_cfg.py` 的 generated-hand MVP：它已经通过
    3 个 same-schema post-mutate variants 的 IsaacLab GUI / random-agent smoke。这里
    把这些数值迁移到可复用 adapter，避免每个 GM env 重复维护 URDF importer 细节。
    """

    fix_base: bool = True
    merge_fixed_joints: bool = False
    force_usd_conversion: bool = False
    use_stable_usd_cache: bool = False
    r"""是否把 importer 指向 mesh-aware、Isaac-versioned AnyMani USD cache；heterogeneous formal route 显式开启。"""
    make_instanceable: bool = True
    r"""是否让 URDF converter 生成 instanceable USD。

    `restore_visual_materials=True` 的 debug 可视化路径会在 child `UrdfFileCfg` 中强制
    设为 `False`：颜色恢复需要在 spawned prim 子树上 author material binding，而 GUI
    模式遍历 instance proxy 曾在第三个 heterogeneous prototype 之后触发 Kit hang。
    动力学 smoke / 训练路径不依赖 URDF debug 色，仍可保持默认 instanceable 优化。
    """

    collision_from_visuals: bool = False
    self_collision: bool = True
    activate_contact_sensors: bool = False
    drive_stiffness: float = 3.0
    drive_damping: float = 0.1


@configclass
class HandActuatorSpawnCfg:
    r"""Generated hand implicit actuator 参数 scaffold。"""

    joint_names_expr: tuple[str, ...] = (".*",)
    effort_limit_sim: float = 0.95
    velocity_limit_sim: float = 8.48
    stiffness: float = 3.0
    damping: float = 0.1
    friction: float = 0.01
    armature: float = 0.001


@configclass
class CanonicalRuntimeCfg:
    r"""canonical 16-DOF runtime lowering 配置。

    开启后，``HandSpawnAdapter`` 从每个 container 的 typed ``hand_cfg`` sidecar 恢复
    ``HandCfg``，调用 assets 层 materializer 生成统一 16-DOF / 25-body URDF，再交给
    IsaacLab 的同一个 ``MultiAssetSpawnerCfg``。``output_root`` 只存放已忽略的 derived
    outputs；generated source bundle 永远不被修改。
    """

    enabled: bool = False
    """是否启用 canonical materialization；默认保持既有 native spawn。"""

    output_root: str = "outputs"
    r"""派生 cache 根；最终路径为 ``<output_root>/canonical_runtime/v1/<cache-key>/``。"""

    schema_version: str = "v1"
    """当前 canonical schema namespace；v1 只覆盖四指、每指最多四个 revolute。"""

    validate_artifact: bool = True
    """是否在交给 IsaacLab 前读取 URDF 并验证 manifest/schema/hash 合同。"""

    asset_row_start: int = 0
    """selection 中第一个 asset 的 evidence-bank row；随后按 selection 顺序递增。"""


@configclass
class HandSpawnCfg:
    r"""GM hand spawn 声明式配置 scaffold。

    `bank` 保持嵌套，避免在 GM 层重复 asset-bank schema。便利写法应通过
    `HandBankCfg.containers=("id0", "id1")` 这类资产层接口解决。
    """

    bank: HandBankCfg = field(default_factory=HandBankCfg)
    """资产选择配置；由 `assets.bank.HandBank` 负责解析。"""

    frame: HandFrameCfg = field(default_factory=HandFrameCfg)
    """`{a}->{h}` frame 对齐配置；env cfg 应显式同步到 command cfg。"""

    joint_init: HandJointInitCfg = field(default_factory=HandJointInitCfg)
    """Articulation 初始关节状态。"""

    urdf: HandUrdfSpawnCfg = field(default_factory=HandUrdfSpawnCfg)
    """URDF importer 参数。"""

    actuator: HandActuatorSpawnCfg = field(default_factory=HandActuatorSpawnCfg)
    """Implicit actuator 参数。"""

    canonical_runtime: CanonicalRuntimeCfg = field(default_factory=CanonicalRuntimeCfg)
    """可选 canonical 16-DOF materialization 与 schema validation。"""

    spawn_backend: Literal["urdf", "usd"] = "urdf"
    """spawn 后端；`usd` 预留给未来离线 USD cache，第一版应显式报 `NotImplementedError`。"""

    asset_routing: Literal["round_robin", "random_choice"] = "round_robin"
    r"""多资产 env routing。

    `round_robin` 对应 IsaacLab `MultiAssetSpawnerCfg.random_choice=False`，确定且便于
    smoke；`random_choice` 透传 IsaacLab 全局 random，第一版不承诺 seed 可复现。
    """

    restore_visual_materials: bool = False
    """是否在 URDF spawn 后用 `HandContainer.visual_rgba_by_name` 恢复 generated debug color。"""

    validate_same_schema: bool = True
    """是否轻量检查 selection 内所有 assets 的 `topology_name` 与 `dof` 一致。"""


class HandSpawnAdapter:
    r"""`HandSpawnCfg` 的 runtime adapter。

    构造函数保持无 IO；首次访问 `selection` 或构造 articulation 时才调用
    `HandBank.resolve()`。这使 env cfg import 阶段仍保持轻量，而 IsaacLab 真正需要
    spawn cfg 时可以得到完整的 `MultiAssetSpawnerCfg`。
    """

    def __init__(self, cfg: HandSpawnCfg, *, resolved_assets: tuple[HandContainer, ...] | None = None):
        r"""保存配置与可选已解析资产；不在构造阶段扫描 dataset 或 asset bank。

        Args:
            cfg (HandSpawnCfg): IsaacLab spawn、frame、actuator 与 canonical runtime 配置。
            resolved_assets (tuple[HandContainer, ...] | None): ``HandAssetDataset.resolve_train().assets``
                等上游已解析 container 轴。给定后 adapter 不再调用 ``HandBank.resolve()``。
        """

        self.cfg = cfg  # 声明式 hand spawn 配置；不在此处触发文件 IO
        if resolved_assets is not None and not resolved_assets:
            raise ValueError("resolved_assets must be non-empty when explicitly provided")
        if resolved_assets is not None:
            asset_ids = tuple(container.asset_id for container in resolved_assets)  # dataset-preserved row order
            if len(set(asset_ids)) != len(asset_ids):
                raise ValueError("resolved_assets must have unique asset IDs")
        self._resolved_assets = tuple(resolved_assets) if resolved_assets is not None else None
        self._selection: HandSelection | None = None  # lazy resolve cache，保持 env import 轻量
        self._canonical_artifacts: tuple[CanonicalHandArtifact, ...] = ()  # routing/manifest 交付给 tasks/distill

    @property
    def selection(self) -> HandSelection:
        r"""Resolved hand selection。

        Returns:
            HandSelection: asset bank resolve 后的有序 hand container 列表。
        """

        if self._selection is None:
            raw_selection = (
                HandSelection(
                    assets=self._resolved_assets,
                    source_mode="mixed",
                    selection_mode="explicit",
                    sample_seed=None,
                    source_root=None,
                )
                if self._resolved_assets is not None
                else HandBank(self.cfg.bank).resolve()
            )  # dataset 注入保持原 row；普通 task 仍沿用 lazy HandBank selection
            self._selection = self._materialize_canonical_selection(raw_selection)  # 可选统一 schema lowering
        return self._selection

    @property
    def canonical_artifacts(self) -> tuple[CanonicalHandArtifact, ...]:
        r"""返回与 ``selection.assets`` 同序的 canonical manifest。"""

        _ = self.selection  # 确保 lazy materialization 已完成
        return self._canonical_artifacts

    @property
    def canonical_schema(self) -> CanonicalHandSchemaCfg | None:
        r"""canonical runtime 关闭时返回 ``None``，开启时返回唯一 v1 schema。"""

        return CANONICAL_HAND_SCHEMA_V1 if self.cfg.canonical_runtime.enabled else None

    def _materialize_canonical_selection(self, selection: HandSelection) -> HandSelection:
        r"""将 source selection lower 成一个同一 canonical articulation schema。"""

        runtime_cfg = self.cfg.canonical_runtime
        if not runtime_cfg.enabled:
            return selection
        if runtime_cfg.schema_version != CANONICAL_HAND_SCHEMA_V1.version:
            raise ValueError(f"unsupported canonical runtime schema version: {runtime_cfg.schema_version!r}")
        output_root = Path(runtime_cfg.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = resolve_anymani_root() / output_root  # Hydra/shell cwd 不得改变研究派生产物位置
        canonical_containers: list[HandContainer] = []  # 每个 row 一个派生 URDF，但共享 16-DOF schema
        artifacts: list[CanonicalHandArtifact] = []  # 同序 manifest，作为 tasks/distill routing 真源
        canonical_hands = []  # typed hands 用于求所有 prototypes 共同合法的 16D boot pose
        canonical_routings = []  # 与 typed hands 同序的 active limit selectors
        for asset_row, container in enumerate(selection.assets, start=runtime_cfg.asset_row_start):
            hand_cfg_raw = container.sidecar.get("hand_cfg")
            if not isinstance(hand_cfg_raw, dict):
                raise ValueError(f"canonical asset {container.asset_id!r} sidecar lacks typed hand_cfg")
            hand_cfg = restore_hand_cfg_snapshot(hand_cfg_raw)  # assets sidecar decoder 是唯一 typed restore 入口
            q_home = (
                tuple(container.geometry_semantics.q_home_rad) if container.geometry_semantics is not None else ()
            )  # typed geometry semantics 是 reset home 的唯一真源
            q_home_joint_names = (
                tuple(container.geometry_semantics.active_joint_names)
                if container.geometry_semantics is not None
                else ()
            )
            canonical_hand, canonical_routing = lower_hand_to_canonical(
                hand_cfg,
                asset_id=container.asset_id,
                schema=CANONICAL_HAND_SCHEMA_V1,
                asset_row=asset_row,
                topology=str(container.sidecar.get("topology_name", "unknown")),
                q_home=q_home,
                q_home_joint_names=q_home_joint_names,
            )  # canonical sidecar contact/layout 也必须引用派生 link names
            canonical_hands.append(canonical_hand)
            canonical_routings.append(canonical_routing)
            artifact = materialize_canonical_artifact(
                hand_cfg,
                asset_id=container.asset_id,
                output_root=output_root,
                source_urdf_path=container.urdf_path,
                schema=CANONICAL_HAND_SCHEMA_V1,
                asset_row=asset_row,
                topology=str(container.sidecar.get("topology_name", "unknown")),
                q_home=q_home,
                q_home_joint_names=q_home_joint_names,
            )
            if runtime_cfg.validate_artifact:
                validate_canonical_artifact(artifact, schema=CANONICAL_HAND_SCHEMA_V1)
            artifacts.append(artifact)
            canonical_urdf = Path(artifact.canonical_urdf_path).resolve(strict=True)
            virtual_to_real = {PurePosixPath("hand.urdf"): canonical_urdf}  # spawn adapter 只消费派生 URDF
            real_to_virtual = {canonical_urdf: PurePosixPath("hand.urdf")}
            canonical_sidecar = dict(container.sidecar)
            canonical_sidecar["hand_cfg"] = canonical_hand.to_dict()  # contact sensors 消费 canonical child links
            canonical_sidecar["canonical_runtime"] = artifact.to_manifest()  # JSON-safe provenance 交付
            canonical_containers.append(
                HandContainer(
                    asset_id=container.asset_id,
                    virtual_to_real=virtual_to_real,
                    real_to_virtual=real_to_virtual,
                    sidecar=canonical_sidecar,
                    source_kind=container.source_kind,
                    geometry_semantics=container.geometry_semantics,
                    visual_rgba_by_name=parse_urdf_visual_rgba_by_name(canonical_urdf),
                )
            )
        if not canonical_containers:
            raise ValueError("canonical runtime requires at least one selected hand asset")
        self.cfg.joint_init.joint_pos = compute_canonical_startup_joint_positions(
            canonical_hands,
            canonical_routings,
            schema=CANONICAL_HAND_SCHEMA_V1,
        )  # IsaacLab pre-event validation 用全局 boot pose；per-env reset 仍使用各自 q_home
        self._canonical_artifacts = tuple(artifacts)
        return HandSelection(
            assets=tuple(canonical_containers),
            source_mode=selection.source_mode,
            selection_mode=selection.selection_mode,
            sample_seed=selection.sample_seed,
            source_root=selection.source_root,
        )

    @property
    def semantic_R_ha(self) -> tuple[float, ...]:
        r"""供 env cfg 显式同步到 `ReorientCommandCfg.semantic_R_ha` 的矩阵。"""

        return tuple(float(value) for value in self.cfg.frame.semantic_R_ha)

    def build_articulation_cfg(self, *, prim_path: str) -> ArticulationCfg:
        r"""构造 IsaacLab `ArticulationCfg`。

        Args:
            prim_path (str): scene 中 robot articulation 的 prim path。

        Returns:
            ArticulationCfg: 可直接赋给 `scene.robot` 的 hand articulation 配置。
        """

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        if self.cfg.validate_same_schema:
            if self.cfg.canonical_runtime.enabled:
                _validate_canonical_hand_schema(self.selection.assets, self.canonical_artifacts)
            else:
                _validate_same_hand_schema(self.selection.assets)  # 多资产 articulation 必须同关节 schema

        root_pos_e, root_quat_ea = _compose_anchor_root_pose(self.cfg.frame)  # $T_{ea}=T_{eh}^{anchor}T_{ha}$
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=self.build_multi_hand_spawn_cfg(),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=root_pos_e,
                rot=root_quat_ea,
                joint_pos=dict(self.cfg.joint_init.joint_pos),
                joint_vel=dict(self.cfg.joint_init.joint_vel),
            ),
            actuators={"fingers": _build_implicit_actuator_cfg(self.cfg.actuator)},
            soft_joint_pos_limit_factor=1.0,
        )

    def build_multi_hand_spawn_cfg(self) -> sim_utils.MultiAssetSpawnerCfg:
        r"""构造同拓扑 generated hands 的 `MultiAssetSpawnerCfg`。

        Returns:
            sim_utils.MultiAssetSpawnerCfg: IsaacLab 多资产 spawner 配置。
        """

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        assets = self.selection.assets  # resolved post-mutate hand variants；同一个 spawner 内应为 same-schema
        visual_material_plan = (
            _build_visual_material_restore_plan(assets[0].urdf_path)
            if (self.cfg.restore_visual_materials and len(assets) > 0)
            else None
        )  # 同拓扑颜色/visual-link contract 只从 reference URDF 解析一次
        assets_cfg = [
            _build_hand_urdf_file_cfg(container, self.cfg, visual_material_plan=visual_material_plan)
            for container in assets
        ]  # 每个 child cfg 对应一个 post-mutate hand variant；材质计划共享
        return sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=cast(Any, assets_cfg),  # IsaacLab cfg stub 的 invariant-list 边界；不增加 runtime 子模块依赖
            random_choice=self.cfg.asset_routing == "random_choice",
            activate_contact_sensors=self.cfg.urdf.activate_contact_sensors,
        )


def _build_hand_urdf_file_cfg(
    container: HandContainer,
    cfg: HandSpawnCfg,
    *,
    visual_material_plan: _VisualMaterialRestorePlan | None = None,
) -> sim_utils.UrdfFileCfg:
    r"""为单个 generated hand container 构造 `UrdfFileCfg`。

    Args:
        container (HandContainer): asset bank 输出的单 hand container。
        cfg (HandSpawnCfg): GM hand spawn 配置。
        visual_material_plan (_VisualMaterialRestorePlan | None): 同拓扑 selection 共享的
            debug color 恢复计划；`None` 表示 wrapper 需要按当前 URDF fallback 解析。

    Returns:
        sim_utils.UrdfFileCfg: IsaacLab URDF importer cfg。
    """

    urdf_cfg = cfg.urdf  # URDF importer 超参锚点，来自 heterogeneous MVP
    # 材质恢复是 GUI/debug 语义：为了给每个 visual name 绑定 URDF RGB，需要 author USD material binding。
    # 若继续生成 instanceable USD，就必须遍历 instance proxy；该路径在 GUI smoke 中出现过 Kit hang。
    make_instanceable = False if cfg.restore_visual_materials else urdf_cfg.make_instanceable

    urdf_file_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(container.urdf_path.resolve()),
        fix_base=urdf_cfg.fix_base,
        merge_fixed_joints=urdf_cfg.merge_fixed_joints,
        force_usd_conversion=urdf_cfg.force_usd_conversion,
        make_instanceable=make_instanceable,
        self_collision=urdf_cfg.self_collision,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=urdf_cfg.drive_stiffness,
                damping=urdf_cfg.drive_damping,
            ),
        ),
        activate_contact_sensors=urdf_cfg.activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=urdf_cfg.self_collision,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    )
    # IsaacLab 2.3.2 实现了该字段但未给类型注解，configclass constructor stub 因而漏参。
    cast(Any, urdf_file_cfg).collision_from_visuals = urdf_cfg.collision_from_visuals
    if urdf_cfg.use_stable_usd_cache:
        _attach_stable_usd_cache(container, urdf_file_cfg)  # 训练与预热共享同一 lazy hit/miss 目录
    if cfg.restore_visual_materials:
        urdf_file_cfg.func = _spawn_urdf_with_restored_visual_materials  # 只恢复 GUI debug color，不改动力学
        # IsaacLab converter 会 JSON-hash `UrdfFileCfg.to_dict()`；因此挂载的计划必须只含 JSON-safe 数据。
        cast(Any, urdf_file_cfg)._anymani_visual_material_plan = (
            _serialize_visual_material_restore_plan(visual_material_plan) if visual_material_plan is not None else None
        )
    return urdf_file_cfg


def _sha256_runtime_file(path: Path) -> str:
    r"""计算 converter implementation 源文件 hash，补足 distribution version 无法覆盖的 dirty source。"""

    digest = hashlib.sha256()  # 当前 IsaacLab checkout 实现身份
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attach_stable_usd_cache(container: HandContainer, urdf_file_cfg: sim_utils.UrdfFileCfg) -> None:
    r"""给 child ``UrdfFileCfg`` 注入 mesh-aware、版本化的稳定 ``usd_dir``。

    ``AssetConverterBase`` 在 ``usd_dir`` 非空时负责创建目录、比较自身 ``.asset_hash`` 并执行
    lazy hit/miss。本函数只把 AnyMani 更完整的 physical input identity 映射到独立目录；不预先转换
    USD，也不把 selection-local ``asset_row`` 纳入 key。

    Args:
        container (HandContainer): 当前 source/canonical URDF 与 sidecar provenance。
        urdf_file_cfg (sim_utils.UrdfFileCfg): 尚未交给 IsaacLab converter 的 child cfg。
    """

    from isaaclab.utils.version import get_isaac_sim_version

    converter_config = dict(cast(Any, urdf_file_cfg).to_dict())  # IsaacLab 自身 converter hash 的同源 cfg mapping
    for path_field in ("asset_path", "usd_dir", "usd_file_name"):
        converter_config.pop(path_field, None)  # 内容 hash 已单独进入 key；cache 路径不能自引用
    canonical_document = container.sidecar.get("canonical_runtime", {})
    canonical_identity = (
        {
            "schema_version": canonical_document.get("schema_version"),
            "schema_digest": canonical_document.get("schema_digest"),
            "source_content_hash": canonical_document.get("source_content_hash"),
            "source_urdf_hash": canonical_document.get("source_urdf_hash"),
            "physical_geometry_hash": canonical_document.get("physical_geometry_hash"),
            "canonical_urdf_hash": canonical_document.get("canonical_urdf_hash"),
        }
        if isinstance(canonical_document, dict)
        else {}
    )  # routing.asset_row 明确不进入 physical USD identity
    converter_cfg_source = inspect.getsourcefile(UrdfConverterCfg)
    if converter_cfg_source is None:
        raise RuntimeError("cannot locate IsaacLab UrdfConverter implementation for cache identity")
    converter_source = Path(converter_cfg_source).with_name("urdf_converter.py")
    if not converter_source.is_file():
        raise RuntimeError(f"IsaacLab UrdfConverter implementation does not exist: {converter_source}")
    cache_dir = build_urdf_usd_cache_dir(
        urdf_path=container.urdf_path,
        converter_config=converter_config,
        isaaclab_version=distribution_version("isaaclab"),
        isaac_sim_version=str(get_isaac_sim_version()),
        converter_implementation_sha256=_sha256_runtime_file(converter_source),
        canonical_identity=canonical_identity,
    )  # `${ANYMANI_CACHE_DIR:-~/.cache/anymani}/isaaclab/usd/<sim>/<key>`
    urdf_file_cfg.usd_dir = str(cache_dir)
    urdf_file_cfg.usd_file_name = "hand.usd"  # key 已由独立目录承载，文件名固定便于人工审计


def _build_implicit_actuator_cfg(cfg: HandActuatorSpawnCfg) -> ImplicitActuatorCfg:
    r"""构造 generated hand 的 implicit actuator 配置。

    Args:
        cfg (HandActuatorSpawnCfg): hand actuator 数值锚点。

    Returns:
        ImplicitActuatorCfg: IsaacLab articulation actuator cfg。
    """

    return ImplicitActuatorCfg(
        joint_names_expr=list(cfg.joint_names_expr),
        effort_limit_sim=cfg.effort_limit_sim,
        velocity_limit_sim=cfg.velocity_limit_sim,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        friction=cfg.friction,
        armature=cfg.armature,
    )


def _compose_anchor_root_pose(
    frame: HandFrameCfg,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    r"""把 hand semantic anchor lower 成 IsaacLab raw root pose。

    核心公式：
    $$
    T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha},\qquad
    R_{ea}=R_{eh}^{anchor}R_{ha},\qquad
    p_{ea}=p_{eh}^{anchor}+R_{eh}^{anchor}p_{ha}.
    $$

    Args:
        frame (HandFrameCfg): `{a}->{h}` 静态校准与 `{h}` 在 `{e}` 中的 anchor。

    Returns:
        tuple[tuple[float, float, float], tuple[float, float, float, float]]: IsaacLab
            `InitialStateCfg` 需要的 `(pos, quat_wxyz)`。
    """

    if not frame.align_hand_frame_to_env:
        raise NotImplementedError("HandFrameCfg.align_hand_frame_to_env=False is reserved for future manual root pose")

    R_ha = _as_matrix3(frame.semantic_R_ha, label="semantic_R_ha")  # $R_{ha}$，raw asset axis -> hand semantic axis
    R_eh = _as_matrix3(frame.anchor_R_eh, label="anchor_R_eh")  # $R_{eh}^{anchor}$，hand semantic axis -> env axis
    p_ha: Vector3 = tuple(float(frame.semantic_p_ha[index]) for index in range(3))  # pyright: ignore[reportAssignmentType]
    p_eh: Vector3 = tuple(float(frame.anchor_p_eh[index]) for index in range(3))  # pyright: ignore[reportAssignmentType]

    R_ea = _matmul3(R_eh, R_ha)  # $R_{ea}=R_{eh}R_{ha}$，raw asset orientation in env frame
    p_ea = _vec_add3(p_eh, _matvec3(R_eh, p_ha))  # $p_{ea}=p_{eh}+R_{eh}p_{ha}$，raw root position in env frame
    quat_ea = _quat_wxyz_from_matrix3(R_ea)  # IsaacLab boundary 表示，内部语义仍是 $SO(3)$
    return p_ea, quat_ea


def _as_matrix3(values: tuple[float, ...], *, label: str) -> Matrix3:
    r"""把 row-major 9 元组解析为 $3\times3$ 旋转矩阵。"""

    if len(values) != 9:
        raise ValueError(f"{label} must contain 9 row-major values, got {len(values)}")
    scalar_values = tuple(float(value) for value in values)  # row-major $[r_{00},r_{01},...,r_{22}]$
    return (
        (scalar_values[0], scalar_values[1], scalar_values[2]),
        (scalar_values[3], scalar_values[4], scalar_values[5]),
        (scalar_values[6], scalar_values[7], scalar_values[8]),
    )


def _matmul3(
    lhs: Matrix3,
    rhs: Matrix3,
) -> Matrix3:
    r"""计算 $3\times3$ 矩阵乘法 $C=AB$。"""

    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # pyright: ignore[reportReturnType]  # 两个 range(3) 在 runtime 固定产生 3x3


def _matvec3(
    matrix: Matrix3,
    vector: Vector3,
) -> Vector3:
    r"""计算 $3\times3$ 矩阵与三维向量乘法 $y=Rv$。"""

    return (
        sum(matrix[0][col] * vector[col] for col in range(3)),
        sum(matrix[1][col] * vector[col] for col in range(3)),
        sum(matrix[2][col] * vector[col] for col in range(3)),
    )


def _vec_add3(lhs: Vector3, rhs: Vector3) -> Vector3:
    r"""计算三维平移向量相加 $p=p_1+p_2$。"""

    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _quat_wxyz_from_matrix3(matrix: Matrix3) -> tuple[float, float, float, float]:
    r"""把旋转矩阵转换为 IsaacLab `(w,x,y,z)` 四元数。

    该函数只用于 IsaacLab cfg 边界。内部 frame 语义仍以 $R\in SO(3)$ 表达，避免在
    研究代码里把四元数双覆盖问题扩散到上游配置。
    """

    m00, m01, m02 = matrix[0]  # 第一行，row-major $R_{0*}$
    m10, m11, m12 = matrix[1]  # 第二行，row-major $R_{1*}$
    m20, m21, m22 = matrix[2]  # 第三行，row-major $R_{2*}$
    trace = m00 + m11 + m22  # $	ext{tr}(R)$，选择稳定分支的数值锚点
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0  # $s=4q_w$，trace 正时 $q_w$ 分支稳定
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0  # $s=4q_x$，x 对角项主导
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0  # $s=4q_y$，y 对角项主导
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0  # $s=4q_z$，z 对角项主导
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)  # 数值归一化，抵消浮点舍入误差
    if norm == 0.0:
        raise ValueError("rotation matrix produced a zero quaternion")
    return (qw / norm, qx / norm, qy / norm, qz / norm)


__all__ = [
    "DEFAULT_HAND_ANCHOR_POS_E",
    "HandActuatorSpawnCfg",
    "HandFrameCfg",
    "HandJointInitCfg",
    "HandSpawnAdapter",
    "HandSpawnCfg",
    "HandUrdfSpawnCfg",
    "_compose_anchor_root_pose",
    "_parse_urdf_visual_link_by_name",
    "_spawn_urdf_with_restored_visual_materials",
]
