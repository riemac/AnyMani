(
    'Generated-hand spawn adapter for Isaac Lab. HandSpawnCfg combines bank '
    'selection, the {a}->{h} frame calibration/anchor, URDF importer, and '
    'actuator settings. HandSpawnAdapter lazily resolves assets, checks schemas, '
    'and builds single- or multi-hand articulation configs; it may also restore '
    'render-only materials and lower the root anchor. Asset generation/mesh '
    'parsing belongs to assets.bank, MDP use belongs to tasks/gm, and training '
    'selection belongs to distill. Frame contract: R_ha and p_ha map raw asset '
    'frame {a} into semantic hand frame {h}; T_eh_anchor is the default hand pose '
    'in env frame {e}. Spawn composes T_ea_anchor=T_eh_anchor*T_ha, with semantic '
    'hand orientation initially identity. Episode orientation randomization '
    'belongs in mdp/events.py and multiplies the anchor; it is not spawn-time '
    'root jitter. Keep object reset, grasp cache, and command updates out of this '
    'module.'
)

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import field
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import isaaclab.sim as sim_utils
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
from anymani.robots._hand_schema import (
    validate_same_hand_schema as _validate_same_hand_schema,
)
from anymani.robots._visual_materials import (
    VisualMaterialRestorePlan as _VisualMaterialRestorePlan,
)
from anymani.robots._visual_materials import (
    audit_visual_material_restore_plan as _audit_visual_material_restore_plan,
)
from anymani.robots._visual_materials import (
    build_visual_material_restore_plan as _build_visual_material_restore_plan,
)
from anymani.robots._visual_materials import (
    parse_urdf_visual_link_by_name as _parse_urdf_visual_link_by_name,
)
from anymani.robots._visual_materials import (
    serialize_visual_material_restore_plan as _serialize_visual_material_restore_plan,
)
from anymani.robots._visual_materials import (
    spawn_urdf_with_restored_visual_materials as _spawn_urdf_with_restored_visual_materials_impl,
)
from anymani.robots.usd_cache import build_urdf_usd_cache_dir
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass

DEFAULT_HAND_ANCHOR_POS_E = (0.0, 0.0, 0.5)
'Default hand-semantic origin anchor in env frame {e}, meters.'

IDENTITY_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
'Row-major 3x3 identity rotation.'

Vector3 = tuple[float, float, float]
'Three-dimensional translation/direction vector.'

Matrix3 = tuple[Vector3, Vector3, Vector3]
'Fixed row-major 3x3 matrix shape certificate for frame composition.'


def _spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    'Keep the existing UrdfFileCfg.func facade name; implementation is in _visual_materials.'

    return _spawn_urdf_with_restored_visual_materials_impl(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


@configclass
class HandFrameCfg:
    (
        'Calibration between raw asset frame {a} and semantic hand frame {h}. Compose '
        'rotation/translation as SE(3) internally and convert to a quaternion only at '
        'the Isaac Lab boundary. semantic_* defines asset calibration T_ha; anchor_* '
        'defines the default semantic-hand pose T_eh_anchor. Reset orientation '
        'randomization uses this anchor without redefining {h}.'
    )

    semantic_R_ha: tuple[float, ...] = IDENTITY_R
    'R_ha, row-major 9 values; v_h = R_ha * v_a.'

    semantic_p_ha: tuple[float, float, float] = (0.0, 0.0, 0.0)
    'p_ha: raw asset origin {a} expressed in hand frame {h}, meters.'

    anchor_R_eh: tuple[float, ...] = IDENTITY_R
    'Default hand-semantic orientation R_eh_anchor in env frame {e}.'

    anchor_p_eh: tuple[float, float, float] = DEFAULT_HAND_ANCHOR_POS_E
    'Default hand-semantic origin p_eh_anchor in env frame {e}, meters.'

    align_hand_frame_to_env: bool = True
    (
        'Automatic derivation of spawn root pose from anchor_R_eh/anchor_p_eh. V1 '
        'supports True only: T_ea_anchor=T_eh_anchor*T_ha. Sample episode orientation '
        'by multiplying the anchor in a reset event.'
    )


@configclass
class HandJointInitCfg:
    'Initial hand-articulation joint state config.'

    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    'Default joint positions keyed by Isaac Lab joint regex.'

    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    'Default joint velocities keyed by Isaac Lab joint regex.'


@configclass
class HandUrdfSpawnCfg:
    (
        'Generated-hand URDF importer parameter anchors from the generated-hand MVP, '
        'which passed Isaac Lab GUI/random-agent smoke with three same-schema '
        'post-mutation variants. Keep reusable importer values here instead of '
        'duplicating them in every GM env.'
    )

    fix_base: bool = True
    merge_fixed_joints: bool = False
    force_usd_conversion: bool = False
    use_stable_usd_cache: bool = False
    'Route through the mesh-aware, Isaac-versioned AnyMani USD cache; explicitly enabled by heterogeneous production configs.'
    make_instanceable: bool = True
    (
        'Whether the URDF converter emits instanceable USD. When render-only material '
        'restore is enabled, each child forces this off so editable prims can receive '
        'bindings. GUI traversal of instance proxies hung Kit after the third '
        'heterogeneous prototype. Physics smoke/training does not need debug colors '
        'and may keep the instanceable optimization.'
    )

    collision_from_visuals: bool = False
    self_collision: bool = True
    activate_contact_sensors: bool = False
    drive_stiffness: float = 3.0
    drive_damping: float = 0.1


@configclass
class HandActuatorSpawnCfg:
    'Generated-hand implicit actuator parameter scaffold.'

    joint_names_expr: tuple[str, ...] = (".*",)
    effort_limit_sim: float = 0.95
    velocity_limit_sim: float = 8.48
    stiffness: float = 3.0
    damping: float = 0.1
    friction: float = 0.01
    armature: float = 0.001


@configclass
class CanonicalRuntimeCfg:
    (
        'Canonical 16-DoF runtime lowering config. When enabled, restore typed '
        'HandCfg from each container sidecar, materialize a shared 16-DoF/25-body '
        'URDF, then pass it to one Isaac Lab MultiAssetSpawnerCfg. output_root stores '
        'ignored derived outputs only; never modify generated source bundles.'
    )

    enabled: bool = False
    'Whether to enable canonical materialization; native spawn remains the default.'

    output_root: str = "outputs"
    'Derived cache root; final path is <output_root>/canonical_runtime/v1/<cache-key>/.'

    schema_version: str = "v1"
    'Canonical schema namespace; v1 supports four fingers with at most four revolute joints each.'

    validate_artifact: bool = True
    'Validate URDF, manifest, schema, and hashes before passing to Isaac Lab.'

    asset_row_start: int = 0
    'Evidence-bank row for the first selected asset; increment by selection order.'


@configclass
class HandSpawnCfg:
    (
        'Declarative GM hand-spawn config. Keep bank settings nested; use '
        'assets.bank.HandBankCfg containers rather than duplicating asset-bank schema '
        'here.'
    )

    bank: HandBankCfg = field(default_factory=HandBankCfg)
    'Asset selection config, resolved by assets.bank.HandBank.'

    frame: HandFrameCfg = field(default_factory=HandFrameCfg)
    '{a}->{h} frame calibration; env config should also set command semantic_R_ha.'

    joint_init: HandJointInitCfg = field(default_factory=HandJointInitCfg)
    'Initial articulation joint state.'

    urdf: HandUrdfSpawnCfg = field(default_factory=HandUrdfSpawnCfg)
    'URDF importer parameters.'

    actuator: HandActuatorSpawnCfg = field(default_factory=HandActuatorSpawnCfg)
    'Implicit actuator parameters.'

    canonical_runtime: CanonicalRuntimeCfg = field(default_factory=CanonicalRuntimeCfg)
    'Optional canonical 16-DoF materialization and schema validation.'

    spawn_backend: Literal["urdf", "usd"] = "urdf"
    'Spawn backend; USD is reserved for future offline cache and must raise NotImplementedError in v1.'

    asset_routing: Literal["round_robin", "random_choice"] = "round_robin"
    (
        'Multi-asset environment routing. round_robin maps to '
        'MultiAssetSpawnerCfg.random_choice=False for deterministic smoke; '
        'random_choice delegates to Isaac Lab global RNG and is not promised '
        'seed-reproducible in v1.'
    )

    restore_visual_materials: bool = False
    "Restore generated debug color after URDF spawn using this child's source/canonical render plan."

    validate_same_schema: bool = True
    'Lightweight check that selected assets share topology_name and DoF.'


class HandSpawnAdapter:
    (
        'Runtime adapter for HandSpawnCfg. Construction performs no I/O; first '
        'selection/articulation access calls HandBank.resolve(). This keeps '
        'env-config imports light while still providing a complete '
        'MultiAssetSpawnerCfg when Isaac Lab requests it.'
    )

    def __init__(self, cfg: HandSpawnCfg, *, resolved_assets: tuple[HandContainer, ...] | None = None):
        (
            'Store config and optional resolved assets without scanning the dataset or '
            'asset bank. If resolved_assets is supplied, do not call HandBank.resolve() '
            'again.'
        )

        self.cfg = cfg  # Declarative hand-spawn config; do not perform file I/O here.
        if resolved_assets is not None and not resolved_assets:
            raise ValueError("resolved_assets must be non-empty when explicitly provided")
        if resolved_assets is not None:
            asset_ids = tuple(container.asset_id for container in resolved_assets)  # dataset-preserved row order
            if len(set(asset_ids)) != len(asset_ids):
                raise ValueError("resolved_assets must have unique asset IDs")
        self._resolved_assets = tuple(resolved_assets) if resolved_assets is not None else None
        self._selection: HandSelection | None = None  # Lazy resolve cache keeps environment imports lightweight.
        self._canonical_artifacts: tuple[CanonicalHandArtifact, ...] = ()  # Deliver routing/manifest to tasks and distill.
        self._visual_material_plans: tuple[_VisualMaterialRestorePlan, ...] = ()  # Independent render-only contract for each child.

    @property
    def selection(self) -> HandSelection:
        'Return the resolved hand selection in asset-bank order.'

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
            )  # Dataset injection preserves the original row; ordinary tasks still use lazy HandBank selection.
            self._selection = self._materialize_canonical_selection(raw_selection)  # Optional shared-schema lowering.
        return self._selection

    @property
    def canonical_artifacts(self) -> tuple[CanonicalHandArtifact, ...]:
        'Return canonical manifests aligned with selection.assets.'

        _ = self.selection  # Ensure lazy materialization has completed.
        return self._canonical_artifacts

    @property
    def canonical_schema(self) -> CanonicalHandSchemaCfg | None:
        'Return None when canonical lowering is disabled; otherwise return the single v1 schema.'

        return CANONICAL_HAND_SCHEMA_V1 if self.cfg.canonical_runtime.enabled else None

    @property
    def visual_material_audit(self) -> tuple[dict[str, object], ...]:
        (
            'Return read-only per-visual render evidence for each selected child. Parse '
            'source/canonical URDFs into JSON-safe mappings only; do not access USD stage '
            'or change canonical bytes, physical hashes, or simulation state. The main '
            'thread may save an audit artifact and use an explicit Isaac Sim smoke to '
            'inspect final bindings.'
        )

        _ensure_visual_material_plans(self)
        return tuple(_audit_visual_material_restore_plan(plan) for plan in self._visual_material_plans)

    def _materialize_canonical_selection(self, selection: HandSelection) -> HandSelection:
        'Lower the source selection to one shared canonical articulation schema.'

        runtime_cfg = self.cfg.canonical_runtime
        if not runtime_cfg.enabled:
            return selection
        if runtime_cfg.schema_version != CANONICAL_HAND_SCHEMA_V1.version:
            raise ValueError(f"unsupported canonical runtime schema version: {runtime_cfg.schema_version!r}")
        output_root = Path(runtime_cfg.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = resolve_anymani_root() / output_root  # Hydra/shell cwd must not change derived-artifact location.
        canonical_containers: list[HandContainer] = []  # One derived URDF per row; all share the 16-DoF schema.
        artifacts: list[CanonicalHandArtifact] = []  # Same-order manifest is the routing source of truth for tasks/distill.
        visual_material_plans: list[_VisualMaterialRestorePlan] = []  # Per-visual source-to-canonical render evidence.
        canonical_hands = []  # Typed hands define a boot pose valid for every prototype.
        canonical_routings = []  # Active-limit selectors aligned with typed hands.
        for asset_row, container in enumerate(selection.assets, start=runtime_cfg.asset_row_start):
            hand_cfg_raw = container.sidecar.get("hand_cfg")
            if not isinstance(hand_cfg_raw, dict):
                raise ValueError(f"canonical asset {container.asset_id!r} sidecar lacks typed hand_cfg")
            hand_cfg = restore_hand_cfg_snapshot(hand_cfg_raw)  # Assets sidecar decoder is the sole typed restore path.
            q_home = (
                tuple(container.geometry_semantics.q_home_rad) if container.geometry_semantics is not None else ()
            )  # Typed geometry semantics are the sole reset-home source.
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
            )  # Canonical sidecar contact/layout must also use derived link names.
            canonical_hands.append(canonical_hand)
            canonical_routings.append(canonical_routing)
            artifact = materialize_canonical_artifact(
                hand_cfg,
                asset_id=container.asset_id,
                output_root=output_root,
                source_urdf_path=container.urdf_path,
                portable_mesh_bindings=container.portable_mesh_bindings,
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
            if self.cfg.restore_visual_materials:
                visual_material_plans.append(
                    _build_visual_material_restore_plan(
                        container.urdf_path,
                        canonical_urdf_path=canonical_urdf,
                        source_sidecar=container.sidecar,
                        canonical_joint_name_by_source_name=dict(artifact.routing.source_to_canonical),
                    )
                )  # Canonical URDF is only the target parse; source colors come from source URDF/verified parent.
            virtual_to_real = {PurePosixPath("hand.urdf"): canonical_urdf}  # Spawn adapter consumes only the derived URDF.
            real_to_virtual = {canonical_urdf: PurePosixPath("hand.urdf")}
            canonical_sidecar = dict(container.sidecar)
            canonical_sidecar["hand_cfg"] = canonical_hand.to_dict()  # Contact sensors use canonical child links.
            canonical_sidecar["canonical_runtime"] = artifact.to_manifest()  # Deliver JSON-safe provenance.
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
        )  # Isaac Lab pre-event validation uses the global boot pose; per-env reset still uses its own q_home.
        self._canonical_artifacts = tuple(artifacts)
        self._visual_material_plans = tuple(visual_material_plans)
        return HandSelection(
            assets=tuple(canonical_containers),
            source_mode=selection.source_mode,
            selection_mode=selection.selection_mode,
            sample_seed=selection.sample_seed,
            source_root=selection.source_root,
        )

    @property
    def semantic_R_ha(self) -> tuple[float, ...]:
        'Matrix to copy explicitly into ReorientCommandCfg.semantic_R_ha.'

        return tuple(float(value) for value in self.cfg.frame.semantic_R_ha)

    def build_articulation_cfg(self, *, prim_path: str) -> ArticulationCfg:
        'Build an Isaac Lab ArticulationCfg for the given scene prim path.'

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        if self.cfg.validate_same_schema:
            if self.cfg.canonical_runtime.enabled:
                _validate_canonical_hand_schema(self.selection.assets, self.canonical_artifacts)
            else:
                _validate_same_hand_schema(self.selection.assets)  # Multi-asset articulations must share one joint schema.

        root_pos_e, root_quat_ea = _compose_anchor_root_pose(self.cfg.frame)  # Root transform composition: T_ea = T_eh_anchor * T_ha.
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
        'Build MultiAssetSpawnerCfg for generated hands with one topology.'

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        assets = self.selection.assets  # Resolved post-mutation hand variants in one spawner must share a schema.
        _ensure_visual_material_plans(self)  # Each child has its own source/canonical mapping and palette.
        assets_cfg = [
            _build_hand_urdf_file_cfg(
                container,
                self.cfg,
                visual_material_plan=(
                    self._visual_material_plans[index] if self.cfg.restore_visual_materials else None
                ),
            )
            for index, container in enumerate(assets)
        ]  # Each child cfg matches one post-mutation hand variant; material plans stay row-specific.
        return sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=cast(Any, assets_cfg),  # Isaac Lab cfg stub boundary; add no runtime submodule dependency.
            random_choice=self.cfg.asset_routing == "random_choice",
            activate_contact_sensors=self.cfg.urdf.activate_contact_sensors,
        )


def _build_hand_urdf_file_cfg(
    container: HandContainer,
    cfg: HandSpawnCfg,
    *,
    visual_material_plan: _VisualMaterialRestorePlan | None = None,
) -> sim_utils.UrdfFileCfg:
    (
        'Build a child UrdfFileCfg. visual_material_plan is specific to this child; '
        'None means the child has no reliable render provenance.'
    )

    urdf_cfg = cfg.urdf  # URDF importer parameter anchors from the heterogeneous MVP.
    if cfg.restore_visual_materials and visual_material_plan is None:
        visual_material_plan = _build_visual_material_restore_plan(
            container.urdf_path,
            source_sidecar=container.sidecar,
        )  # Direct adapter callers read the current source; formal selection plans each child in advance.
    bindable_visual_names = (
        set(visual_material_plan.visual_rgba_by_name) - set(visual_material_plan.unresolved_visual_names)
        if visual_material_plan is not None
        else set()
    )  # Enable render wrapper only with color evidence and a canonical target.
    restore_for_child = cfg.restore_visual_materials and bool(bindable_visual_names)
    # Material restore is GUI/debug only and needs editable prims; without color evidence, preserve importer appearance.
    # Avoid instanceable USD on this path: traversing instance proxies caused a Kit hang in GUI smoke.
    make_instanceable = False if restore_for_child else urdf_cfg.make_instanceable

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
    # Isaac Lab 2.3.2 implements this field without a type annotation, so the configclass constructor stub omits it.
    cast(Any, urdf_file_cfg).collision_from_visuals = urdf_cfg.collision_from_visuals
    if urdf_cfg.use_stable_usd_cache:
        _attach_stable_usd_cache(container, urdf_file_cfg)  # Training and warmup share the same lazy hit/miss directory.
    if restore_for_child:
        urdf_file_cfg.func = _spawn_urdf_with_restored_visual_materials  # Restore GUI debug color only; do not change dynamics.
        # Isaac Lab JSON-hashes UrdfFileCfg.to_dict(); attached plans must contain JSON-safe values only.
        cast(Any, urdf_file_cfg)._anymani_visual_material_plan = (
            _serialize_visual_material_restore_plan(visual_material_plan) if visual_material_plan is not None else None
        )
    return urdf_file_cfg


def _ensure_visual_material_plans(self: HandSpawnAdapter) -> None:
    (
        'Ensure selected assets and per-child render plans have identical order '
        'without starting USD/Isaac runtime. Canonical plans are built during '
        'materialization; native URDF plans are parsed lazily when render cfg/audit '
        'is requested. Without render intent, do not read color or mesh evidence.'
    )

    if not self.cfg.restore_visual_materials:
        self._visual_material_plans = ()
        return
    assets = self.selection.assets
    if len(self._visual_material_plans) == len(assets):
        return
    if self.cfg.canonical_runtime.enabled:
        raise RuntimeError("canonical visual material plans were not created during canonical materialization")
    self._visual_material_plans = tuple(
        _build_visual_material_restore_plan(
            container.urdf_path,
            source_sidecar=container.sidecar,
        )
        for container in assets
    )  # Native children also keep separate source palettes; do not share one coverage reference.


def _sha256_runtime_file(path: Path) -> str:
    'Hash converter implementation source to detect dirty code not represented by the distribution version.'

    digest = hashlib.sha256()  # Identity of the current Isaac Lab checkout.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attach_stable_usd_cache(container: HandContainer, urdf_file_cfg: sim_utils.UrdfFileCfg) -> None:
    (
        'Set a mesh-aware, versioned usd_dir on the child UrdfFileCfg. IsaacLab '
        'converter handles lazy directory creation and .asset_hash hit/miss. Map '
        'AnyMani physical input identity to an independent path; do not preconvert '
        'USD or include selection-local asset_row in the key.'
    )

    from isaaclab.utils.version import get_isaac_sim_version

    converter_config = dict(cast(Any, urdf_file_cfg).to_dict())  # Same-source cfg mapping used by Isaac Lab converter hashing.
    for path_field in ("asset_path", "usd_dir", "usd_file_name"):
        converter_config.pop(path_field, None)  # Content hash is already in the key; cache path must not reference itself.
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
    )  # routing.asset_row is excluded from physical USD identity.
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
    )  # Default cache: ANYMANI_CACHE_DIR or ~/.cache/anymani/isaaclab/usd/<sim>/<key>.
    urdf_file_cfg.usd_dir = str(cache_dir)
    urdf_file_cfg.usd_file_name = "hand.usd"  # The key selects a separate directory; keep the filename fixed for audit.


def _build_implicit_actuator_cfg(cfg: HandActuatorSpawnCfg) -> ImplicitActuatorCfg:
    'Build an implicit actuator config for the generated hand.'

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
    (
        'Lower the semantic hand anchor into Isaac Lab raw root pose: '
        'R_ea=R_eh_anchor*R_ha and p_ea=p_eh_anchor+R_eh_anchor*p_ha. Return '
        '(position, quaternion_wxyz) for InitialStateCfg.'
    )

    if not frame.align_hand_frame_to_env:
        raise NotImplementedError("HandFrameCfg.align_hand_frame_to_env=False is reserved for future manual root pose")

    R_ha = _as_matrix3(frame.semantic_R_ha, label="semantic_R_ha")  # R_ha maps raw asset axis to hand semantic axis.
    R_eh = _as_matrix3(frame.anchor_R_eh, label="anchor_R_eh")  # R_eh_anchor maps hand semantic axis to env axis.
    p_ha: Vector3 = tuple(
        float(frame.semantic_p_ha[index]) for index in range(3)
    )  # pyright: ignore[reportAssignmentType]
    p_eh: Vector3 = tuple(
        float(frame.anchor_p_eh[index]) for index in range(3)
    )  # pyright: ignore[reportAssignmentType]

    R_ea = _matmul3(R_eh, R_ha)  # R_ea = R_eh_anchor * R_ha: raw asset orientation in env frame.
    p_ea = _vec_add3(p_eh, _matvec3(R_eh, p_ha))  # p_ea = p_eh + R_eh_anchor * p_ha: raw root position in env frame.
    quat_ea = _quat_wxyz_from_matrix3(R_ea)  # Use Isaac Lab boundary representation; internal semantics remain SO(3).
    return p_ea, quat_ea


def _as_matrix3(values: tuple[float, ...], *, label: str) -> Matrix3:
    'Parse a row-major 9-value tuple as a 3x3 rotation matrix.'

    if len(values) != 9:
        raise ValueError(f"{label} must contain 9 row-major values, got {len(values)}")
    scalar_values = tuple(float(value) for value in values)  # Row-major [r00,r01,...,r22].
    return (
        (scalar_values[0], scalar_values[1], scalar_values[2]),
        (scalar_values[3], scalar_values[4], scalar_values[5]),
        (scalar_values[6], scalar_values[7], scalar_values[8]),
    )


def _matmul3(
    lhs: Matrix3,
    rhs: Matrix3,
) -> Matrix3:
    'Multiply two 3x3 matrices: C=A*B.'

    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3)) for row in range(3)
    )  # pyright: ignore[reportReturnType]  # range(3) always produces a 3x3 matrix at runtime.


def _matvec3(
    matrix: Matrix3,
    vector: Vector3,
) -> Vector3:
    'Multiply a 3x3 matrix by a 3-vector: y=R*v.'

    return (
        sum(matrix[0][col] * vector[col] for col in range(3)),
        sum(matrix[1][col] * vector[col] for col in range(3)),
        sum(matrix[2][col] * vector[col] for col in range(3)),
    )


def _vec_add3(lhs: Vector3, rhs: Vector3) -> Vector3:
    'Add two 3D translation vectors.'

    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _quat_wxyz_from_matrix3(matrix: Matrix3) -> tuple[float, float, float, float]:
    (
        'Convert a rotation matrix to an Isaac Lab (w,x,y,z) quaternion. Keep '
        'internal frame math in SO(3); confine quaternion double-cover handling to '
        'the Isaac Lab boundary.'
    )

    m00, m01, m02 = matrix[0]  # First row of row-major R.
    m10, m11, m12 = matrix[1]  # Second row of row-major R.
    m20, m21, m22 = matrix[2]  # Third row of row-major R.
    trace = m00 + m11 + m22  # trace(R), used to select the stable numeric branch.
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0  # s=4*q_w; stable q_w branch when trace is positive.
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s=4*q_x; dominant x diagonal branch.
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s=4*q_y; dominant y diagonal branch.
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s=4*q_z; dominant z diagonal branch.
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)  # Normalize numerically to reduce floating-point roundoff.
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
