r"""Pure declaration contract tests for the single-asset GM MDP probe env.

这些测试不导入 `single_asset_env_cfg.py`，因为真实 env cfg 会触发 Isaac Lab / USD /
`pxr` 运行时绑定。这里用 AST 与 Gym registry 锁住本阶段的科研合同：单资产环境必须
绑定 pre-made mother asset、复用当前 GM MDP scaffold、默认高并行训练规模，并且 contact
layout 必须来自 mother sidecar，而不是复用 multi-asset 默认 layout。第一轮 command
刻意采用 fixed `{h}` z 轴 + episode 目标，对齐 LEAP 官方 z-axis 成功基线。
"""

from __future__ import annotations

import ast
from pathlib import Path

import anymani.tasks.gm  # noqa: F401  # 注册 tasks-owned Gym aliases，不导入 env cfg 文件
import gymnasium as gym

SINGLE_ASSET_ENV_CFG_PATH = Path(__file__).resolve().parents[1] / "single_asset_env_cfg.py"
r"""被测试的 single-asset env cfg 源文件路径；只做 AST 读取，不执行模块。"""


def _source() -> str:
    r"""读取 single-asset env cfg 源码文本。"""

    return SINGLE_ASSET_ENV_CFG_PATH.read_text(encoding="utf-8")


def _module_ast() -> ast.Module:
    r"""解析 single-asset env cfg 的 AST，避免触发 Isaac runtime import。"""

    return ast.parse(_source())


def _constant_values() -> dict[str, object]:
    r"""读取本测试关心的模块级常量。"""

    values: dict[str, object] = {}
    target_names = {
        "GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH",
    }
    for node in _module_ast().body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in target_names:
            continue
        values[name] = _eval_literal_expr(node.value, values)
    return values


def _eval_literal_expr(node: ast.AST, values: dict[str, object]) -> object:
    r"""求值测试允许的安全字面量表达式。"""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return str(_eval_literal_expr(node.left, values)) + str(_eval_literal_expr(node.right, values))
    if isinstance(node, ast.Tuple):
        return tuple(_eval_literal_expr(element, values) for element in node.elts)
    raise ValueError(f"Unsupported declaration expression: {ast.dump(node)}")


def _module_assign_call(assign_name: str) -> ast.Call:
    r"""定位模块级 `assign_name = SomeCall(...)` 声明。"""

    for node in _module_ast().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == assign_name and isinstance(node.value, ast.Call):
                return node.value
    raise AssertionError(f"{assign_name} declaration not found")


def _class_assign_call(class_name: str, assign_name: str) -> ast.Call:
    r"""在指定 config class body 中定位 class-level cfg field call。"""

    for node in ast.walk(_module_ast()):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if isinstance(class_node, ast.Assign) and isinstance(class_node.value, ast.Call):
                if any(isinstance(target, ast.Name) and target.id == assign_name for target in class_node.targets):
                    return class_node.value
            if isinstance(class_node, ast.AnnAssign) and isinstance(class_node.target, ast.Name):
                if class_node.target.id == assign_name and isinstance(class_node.value, ast.Call):
                    return class_node.value
    raise AssertionError(f"{class_name}.{assign_name} declaration not found")


def _keyword_call(call: ast.Call, keyword_name: str) -> ast.Call:
    r"""从 cfg call 中读取某个 keyword 的嵌套 call。"""

    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Call):
            return keyword.value
    raise AssertionError(f"keyword call {keyword_name!r} not found")


def _keyword_literal(call: ast.Call, keyword_name: str, values: dict[str, object]) -> object:
    r"""从 cfg call 中读取某个 keyword 的安全字面量值。"""

    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return _eval_literal_expr(keyword.value, values)
    raise AssertionError(f"keyword {keyword_name!r} not found")


def _call_func_name(call: ast.Call) -> str:
    r"""返回简单函数调用名，服务声明式 cfg contract。"""

    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    raise AssertionError(f"Unsupported call func: {ast.dump(call.func)}")


def test_single_asset_task_alias_points_to_single_asset_cfg() -> None:
    r"""tasks-owned Gym aliases 应指向 single-asset env cfg。"""

    spec = gym.spec("AnyMani-GM-SingleAsset-v0")
    play_spec = gym.spec("AnyMani-GM-SingleAsset-Play-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith("single_asset_env_cfg:GmSingleAssetEnvCfg")
    assert play_spec.kwargs["env_cfg_entry_point"].endswith("single_asset_env_cfg:GmSingleAssetEnvCfg_PLAY")


def test_single_asset_binds_premade_mother_bundle_with_explicit_selection() -> None:
    r"""single-asset env 应绑定 pre-made mother bundle，而不是 post-mutate sample bank。"""

    values = _constant_values()
    source = _source()
    hand_spawn_call = _module_assign_call("GM_SINGLE_ASSET_HAND_SPAWN_CFG")
    bank_call = _keyword_call(hand_spawn_call, "bank")

    assert values["GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH"].endswith("single_palm_leap/right_t4_i4_m4_r4")
    assert _keyword_literal(bank_call, "source_mode", values) == "post_mutate"
    assert _keyword_literal(bank_call, "selection_mode", values) == "explicit"
    assert "containers=(_single_asset_bundle_path(),)" in source
    assert "resolve_bank_path(GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH)" in source
    assert _keyword_literal(bank_call, "validate_mesh_relpaths", values) is True
    assert _keyword_literal(bank_call, "parse_visual_rgba", values) is True


def test_single_asset_default_training_scale_and_object_init_are_declared() -> None:
    r"""单资产 probe 默认应使用 2048 env 和标定台导出的 GUI/contact-basin 初态。"""

    values = _constant_values()
    source = _source()
    scene_call = _class_assign_call("GmSingleAssetEnvCfg", "scene")
    hand_spawn_call = _module_assign_call("GM_SINGLE_ASSET_HAND_SPAWN_CFG")
    object_call = _class_assign_call("GmSingleAssetSceneCfg", "object")
    spawn_call = _keyword_call(object_call, "spawn")
    init_state_call = _keyword_call(object_call, "init_state")
    joint_init_call = _keyword_call(hand_spawn_call, "joint_init")
    calibrated_joint_pos = {
        "thumb_j0": 0.71999997,
        "index_j0": -0.0,
        "middle_j0": 0.0,
        "ring_j0": 0.11,
        "thumb_j1": 1.56999993,
        "index_j1": -0.52999997,
        "middle_j1": -0.12,
        "ring_j1": 0.44999999,
        "thumb_j2": 0.75999999,
        "index_j2": 1.23000002,
        "middle_j2": 1.13999999,
        "ring_j2": 1.29999995,
        "thumb_j3": 1.63,
        "index_j3": 0.94999999,
        "middle_j3": 0.91999996,
        "ring_j3": 0.66999996,
    }

    assert _keyword_literal(scene_call, "num_envs", values) == 2048
    assert _keyword_literal(scene_call, "replicate_physics", values) is False
    assert _keyword_literal(spawn_call, "scale", values) == (1.0, 1.0, 1.0)
    assert _call_func_name(joint_init_call) == "HandJointInitCfg"
    for joint_name, joint_pos in calibrated_joint_pos.items():
        assert f'"{joint_name}": {joint_pos}' in source
    assert _keyword_literal(init_state_call, "pos", values) == (0.02, 0.08, 0.56)
    assert "self.episode_length_s = 10.0" in source


def test_single_asset_uses_own_sidecar_contact_layout() -> None:
    r"""obs/reward contact sensor names 应来自 single-asset layout，而不是 GM default multi-asset layout。"""

    values = _constant_values()
    source = _source()
    layout_call = _module_assign_call("GM_SINGLE_ASSET_CONTACT_LAYOUT")

    assert _call_func_name(layout_call) == "build_contact_sensor_layout_from_hand_spawn"
    assert isinstance(layout_call.args[0], ast.Name) and layout_call.args[0].id == "GM_SINGLE_ASSET_HAND_SPAWN_CFG"
    assert _keyword_literal(layout_call, "validate_all_assets", values) is True
    assert "install_contact_sensors(self, GM_SINGLE_ASSET_CONTACT_LAYOUT)" in source
    assert "GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_sensor_names" in source
    assert "GM_SINGLE_ASSET_CONTACT_LAYOUT.non_tip_sensor_names" in source


def test_single_asset_policy_uses_hand_frame_object_pose_obs() -> None:
    r"""teacher policy 应读取 `{h}` 下 object pose / contact force，而不是 world-frame 表征。"""

    source = _source()
    object_pos_call = _class_assign_call("PolicyCfg", "object_pos_h")
    object_rot_call = _class_assign_call("PolicyCfg", "object_rot6d_h")
    force_call = _class_assign_call("PolicyCfg", "fingertip_force_h")

    assert _call_func_name(object_pos_call) == "ObsTerm"
    assert _call_func_name(object_rot_call) == "ObsTerm"
    assert _call_func_name(force_call) == "ObsTerm"
    assert "func=gm_mdp.object_pos_h" in source
    assert "func=gm_mdp.object_rot6d_h" in source
    assert "func=gm_mdp.fingertip_contact_force_h" in source
    assert '"semantic_R_ha": GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha' in source
    assert "func=isaac_mdp.root_pos_w" not in source
    assert "func=isaac_mdp.root_quat_w" not in source
    assert "fingertip_contact_force_w" not in source


def test_single_asset_uses_split_reset_events_and_fixed_z_axis_command() -> None:
    r"""第一轮单资产 probe 应拆分 reset 语义，并把 command 收窄到 fixed z-axis。"""

    values = _constant_values()
    source = _source()
    command_call = _class_assign_call("GmSingleAssetCommandsCfg", "goal_pose")
    reset_robot_call = _class_assign_call("GmSingleAssetEventsCfg", "reset_robot_joints")
    reset_object_call = _class_assign_call("GmSingleAssetEventsCfg", "reset_object")
    record_anchor_call = _class_assign_call("GmSingleAssetEventsCfg", "record_object_reset_anchor")
    structural_filter_call = _class_assign_call("GmSingleAssetEventsCfg", "apply_structural_collision_filter")
    action_call = _class_assign_call("GmSingleAssetActionsCfg", "hand_joint_pos")

    assert _keyword_literal(command_call, "axis_mode", values) == "fixed"
    assert _keyword_literal(command_call, "axis_resample_mode", values) == "episode"
    assert _keyword_literal(command_call, "fixed_axis_h", values) == (0.0, 0.0, 1.0)
    assert _keyword_literal(action_call, "scale", values) == 0.1
    assert _keyword_literal(action_call, "preserve_order", values) is True
    assert _call_func_name(reset_robot_call) == "EventTerm"
    assert _call_func_name(reset_object_call) == "EventTerm"
    assert _call_func_name(record_anchor_call) == "EventTerm"
    assert _call_func_name(structural_filter_call) == "EventTerm"
    assert "func=gm_mdp.apply_generated_structural_collision_filter" in source
    assert 'mode="prestartup"' in source
    assert '"palm_link_name": GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name' in source
    assert '"finger_link_chains": GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains' in source
    assert '"filter_palm_finger": True' in source
    assert '"filter_same_finger": True' in source
    assert "func=isaac_mdp.reset_joints_by_offset" in source
    assert '"position_range": (0.0, 0.0)' in source
    assert "func=isaac_mdp.reset_root_state_uniform" in source
    assert '"pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)}' in source
    assert "func=gm_mdp.record_object_reset_anchor" in source
    assert "func=gm_mdp.simple_no_cache_reset" not in source
