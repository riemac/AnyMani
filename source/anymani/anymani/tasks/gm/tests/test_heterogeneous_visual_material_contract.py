r"""Pure config tests for GM heterogeneous hand spawn.

这些测试不启动 Isaac Sim，也不导入真实 USD / Omni binding。它们锁住本轮迁移后的
接口 contract：`heterogeneous_test_env_cfg.py` 不再维护私有 hand-set/helper，而是通过
`HandSpawnCfg + HandSpawnAdapter + HandBankCfg` 直接验证 asset bank 到 IsaacLab
`ArticulationCfg` 的 lower 路径。
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import MISSING, Field
from pathlib import Path

from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.urdf_utils import parse_urdf_visual_rgba_by_name


def _load_heterogeneous_cfg_module():
    r"""用最小 IsaacLab stub 加载 heterogeneous env cfg 文件。

    Returns:
        module: 包含 heterogeneous env cfg 与 `HandSpawnCfg` 默认实例的模块。
    """

    previous_modules = _install_isaaclab_cfg_stubs()
    target_modules = (
        "anymani.tasks.gm.hand_spawn",
        "anymani.tasks.gm.heterogeneous_test_env_cfg",
    )
    previous_target_modules = {name: sys.modules.get(name) for name in target_modules}
    for name in target_modules:
        sys.modules.pop(name, None)

    try:
        module = importlib.import_module("anymani.tasks.gm.heterogeneous_test_env_cfg")
    finally:
        for name, previous in previous_target_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        _restore_modules(previous_modules)
    return module


def _install_isaaclab_cfg_stubs() -> dict[str, types.ModuleType | None]:
    r"""安装加载 GM hand spawn cfg 所需的 IsaacLab stub module tree。

    Returns:
        dict[str, types.ModuleType | None]: 被本测试临时接管模块的旧状态。
    """

    class _Cfg:
        r"""接受任意关键字的配置占位类。"""

        def __init__(self, **kwargs):
            r"""保存字段，模拟 IsaacLab configclass 实例的开放属性语义。"""

            self.__dict__.update(kwargs)

        def __post_init__(self) -> None:
            r"""ManagerBasedRLEnvCfg stub 的 no-op post init。"""

    class _UrdfFileCfg(_Cfg):
        r"""`sim_utils.UrdfFileCfg` 的最小近似，只保留 `func` 与传入字段。"""

        func = object()

    class _MultiAssetSpawnerCfg(_Cfg):
        r"""`sim_utils.MultiAssetSpawnerCfg` 的最小近似。"""

    class _ArticulationCfg(_Cfg):
        r"""`ArticulationCfg` stub，含 nested `InitialStateCfg`。"""

        InitialStateCfg = _Cfg

    class _AssetBaseCfg(_Cfg):
        r"""`AssetBaseCfg` stub，含 nested `InitialStateCfg`。"""

        InitialStateCfg = _Cfg

    class _JointDriveCfg(_Cfg):
        r"""`UrdfConverterCfg.JointDriveCfg` stub。"""

        class PDGainsCfg(_Cfg):
            r"""PD gains stub，保留 stiffness / damping 字段。"""

    class _UrdfConverterCfg:
        r"""`UrdfConverterCfg` namespace stub。"""

        JointDriveCfg = _JointDriveCfg

    def _clone_config_default(value):
        r"""复刻 IsaacLab configclass 对 dataclasses.field default_factory 的处理。"""

        if isinstance(value, Field):
            if value.default_factory is not MISSING:  # type: ignore[attr-defined]
                return value.default_factory()  # type: ignore[misc]
            if value.default is not MISSING:
                return value.default
            return None
        return value

    def configclass(cls):
        r"""极小 `configclass`：按 annotated fields 生成 keyword-only init。"""

        annotations = getattr(cls, "__annotations__", {})

        def __init__(self, **kwargs):
            r"""用 class body default 与用户 kwargs 初始化配置对象。"""

            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                elif hasattr(cls, name):
                    value = _clone_config_default(getattr(cls, name))
                else:
                    value = None
                setattr(self, name, value)
            for name, value in kwargs.items():
                if name not in annotations:
                    setattr(self, name, value)

        cls.__init__ = __init__
        return cls

    isaaclab = types.ModuleType("isaaclab")
    actuators = types.ModuleType("isaaclab.actuators")
    assets = types.ModuleType("isaaclab.assets")
    envs = types.ModuleType("isaaclab.envs")
    envs_common = types.ModuleType("isaaclab.envs.common")
    envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    scene = types.ModuleType("isaaclab.scene")
    sim = types.ModuleType("isaaclab.sim")
    converters = types.ModuleType("isaaclab.sim.converters")
    physics_materials_cfg = types.ModuleType("isaaclab.sim.spawners.materials.physics_materials_cfg")
    utils = types.ModuleType("isaaclab.utils")
    utils_assets = types.ModuleType("isaaclab.utils.assets")

    for package in (isaaclab, envs, sim):
        package.__path__ = []

    actuators.ImplicitActuatorCfg = _Cfg
    assets.ArticulationCfg = _ArticulationCfg
    assets.AssetBaseCfg = _AssetBaseCfg
    envs.ManagerBasedRLEnvCfg = _Cfg
    envs_common.ViewerCfg = _Cfg
    envs_mdp.RelativeJointPositionActionCfg = _Cfg
    envs_mdp.joint_pos = object()
    envs_mdp.joint_vel = object()
    envs_mdp.is_alive = object()
    envs_mdp.time_out = object()
    managers.ObservationGroupCfg = _Cfg
    managers.ObservationTermCfg = _Cfg
    managers.RewardTermCfg = _Cfg
    managers.TerminationTermCfg = _Cfg
    scene.InteractiveSceneCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg
    sim.DomeLightCfg = _Cfg
    sim.GroundPlaneCfg = _Cfg
    sim.MultiAssetSpawnerCfg = _MultiAssetSpawnerCfg
    sim.PhysxCfg = _Cfg
    sim.PreviewSurfaceCfg = _Cfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.SimulationCfg = _Cfg
    sim.UrdfFileCfg = _UrdfFileCfg
    converters.UrdfConverterCfg = _UrdfConverterCfg
    physics_materials_cfg.RigidBodyMaterialCfg = _Cfg
    utils.configclass = configclass
    utils_assets.ISAAC_NUCLEUS_DIR = "/Isaac/Nucleus"

    replacements = {
        "isaaclab": isaaclab,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets,
        "isaaclab.envs": envs,
        "isaaclab.envs.common": envs_common,
        "isaaclab.envs.mdp": envs_mdp,
        "isaaclab.managers": managers,
        "isaaclab.scene": scene,
        "isaaclab.sim": sim,
        "isaaclab.sim.converters": converters,
        "isaaclab.sim.spawners.materials.physics_materials_cfg": physics_materials_cfg,
        "isaaclab.utils": utils,
        "isaaclab.utils.assets": utils_assets,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    return previous


def _restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    r"""恢复测试 stub 安装前的 `sys.modules` 状态。"""

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_generated_hand_urdf_visual_rgba_is_parsed_by_visual_name() -> None:
    r"""generated URDF 的 palm/finger/tip debug color 必须能按 visual name 解析。"""

    repo_root = resolve_anymani_root()
    urdf_path = repo_root / (
        "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
        "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0b6fbfce/hand.urdf"
    )

    visual_rgba_by_name = parse_urdf_visual_rgba_by_name(urdf_path)

    assert visual_rgba_by_name["palm_visual"] == (0.603921569, 0.149019608, 0.149019608, 1.0)
    assert visual_rgba_by_name["index_j0_vis"] == (0.866666667, 0.866666667, 0.0509803922, 1.0)
    assert visual_rgba_by_name["index_tip_mesh_vis"] == (0.92, 0.88, 0.78, 1.0)


def test_generated_hand_urdf_visual_names_map_to_parent_links() -> None:
    r"""颜色恢复应把 URDF visual name 稳定映射到 spawned USD 的 `<link>/visuals`。"""

    module = _load_heterogeneous_cfg_module()
    repo_root = resolve_anymani_root()
    urdf_path = repo_root / (
        "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
        "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0b6fbfce/hand.urdf"
    )

    visual_link_parser = module.HandSpawnAdapter.__init__.__globals__["_parse_urdf_visual_link_by_name"]
    visual_link_by_name = visual_link_parser(urdf_path)

    assert visual_link_by_name["palm_visual"] == "palm"
    assert visual_link_by_name["index_j0_vis"] == "index_mcp1"
    assert visual_link_by_name["thumb_j2_vis"] == "thumb_mcp"
    assert visual_link_by_name["index_tip_mesh_vis"] == "index_tip"


def test_heterogeneous_spawn_cfg_resolves_three_round_robin_assets() -> None:
    r"""heterogeneous smoke 必须通过 asset bank 选择 3 个 round-robin hand assets。"""

    module = _load_heterogeneous_cfg_module()
    spawn_cfg = module.DEFAULT_HETEROGENEOUS_HAND_SPAWN_CFG
    robot_cfg = module.HeterogeneousHandTestSceneCfg.robot

    assert tuple(str(container.path) for container in spawn_cfg.bank.containers) == module.HETEROGENEOUS_HAND_IDS
    assert robot_cfg.spawn.random_choice is False
    assert len(robot_cfg.spawn.assets_cfg) == 3
    assert [Path(child.asset_path).parent.name for child in robot_cfg.spawn.assets_cfg] == list(module.HETEROGENEOUS_HAND_IDS)


def test_restore_visual_materials_is_opt_in_on_urdf_child_cfg(tmp_path: Path) -> None:
    r"""只有 opt-in hand spawn cfg 会把 child `UrdfFileCfg.func` 替换成颜色恢复 wrapper。"""

    module = _load_heterogeneous_cfg_module()
    bundle_dir = tmp_path / "variant_a"
    bundle_dir.mkdir()
    (bundle_dir / "hand.urdf").write_text("<robot name='stub'/>", encoding="utf-8")
    (bundle_dir / "hand.yaml").write_text(
        "id: variant_a\n"
        "topology_name: stub_topology\n"
        "dof: 1\n"
        "surviving_slots: [index]\n"
        "fingers:\n"
        "- name: index\n"
        "  revolute_dof: 1\n",
        encoding="utf-8",
    )

    default_spawn_cfg = module.HandSpawnCfg(
        bank=module.HandBankCfg(
            selection_mode="explicit",
            containers=(str(bundle_dir),),
            validate_mesh_relpaths=False,
        ),
        restore_visual_materials=False,
    )
    restored_spawn_cfg = module.HandSpawnCfg(
        bank=module.HandBankCfg(
            selection_mode="explicit",
            containers=(str(bundle_dir),),
            validate_mesh_relpaths=False,
        ),
        restore_visual_materials=True,
    )

    default_child_cfg = module.HandSpawnAdapter(default_spawn_cfg).build_multi_hand_spawn_cfg().assets_cfg[0]
    restored_child_cfg = module.HandSpawnAdapter(restored_spawn_cfg).build_multi_hand_spawn_cfg().assets_cfg[0]

    assert default_child_cfg.func is not restored_child_cfg.func
    assert default_child_cfg.make_instanceable is True
    assert restored_child_cfg.make_instanceable is False
    assert restored_child_cfg.func.__name__ == "_spawn_urdf_with_restored_visual_materials"
    assert module.DEFAULT_HETEROGENEOUS_HAND_SPAWN_CFG.restore_visual_materials is True


def test_heterogeneous_scene_uses_clear_sky_dome_light() -> None:
    r"""异构 GUI smoke 应复用 DexSuite 的清天 HDRI，而不是纯灰 dome light。"""

    module = _load_heterogeneous_cfg_module()
    texture_file = module.HeterogeneousHandTestSceneCfg.light.spawn.texture_file

    assert module.HeterogeneousHandTestSceneCfg.light.prim_path == "/World/skyLight"
    assert module.HeterogeneousHandTestSceneCfg.light.spawn.intensity == 750.0
    assert texture_file.endswith("/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr")


def test_heterogeneous_hand_root_pose_aligns_semantic_frame_to_world() -> None:
    r"""异构 smoke 的 generated hand 初态应表达 $R_{wh}=I$ 和默认 hand anchor。"""

    module = _load_heterogeneous_cfg_module()
    robot_init_state = module.HeterogeneousHandTestSceneCfg.robot.init_state

    assert module.HETEROGENEOUS_HAND_INIT_ROT == (1.0, 0.0, 0.0, 0.0)
    assert robot_init_state.rot == module.HETEROGENEOUS_HAND_INIT_ROT
    assert robot_init_state.pos == module.DEFAULT_HAND_ANCHOR_POS_E
