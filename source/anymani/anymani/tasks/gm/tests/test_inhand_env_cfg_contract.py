r"""Pure declaration contract tests for the executable GM in-hand env cfg.

这些测试不导入 `inhand_env_cfg.py`，因为真实文件会触发 Isaac Lab / USD / `pxr`
绑定，只适合在 Isaac Sim Python runtime 中导入。这里用 AST 读取声明式配置源码，
只锁住 first runnable slice 的科研合同：默认资产选择必须是固定 seed 的
post-mutate hand slice，默认并行规模由 hand count 与 env-per-hand 相乘得到，并且 hand
spawn 必须为 contact obs / reward 打开 URDF contact report。
"""

from __future__ import annotations

import ast
from pathlib import Path

import anymani.tasks.gm  # noqa: F401  # 注册 tasks-owned Gym task aliases；不导入 `inhand_env_cfg`
import gymnasium as gym

INHAND_ENV_CFG_PATH = Path(__file__).resolve().parents[1] / "inhand_env_cfg.py"
r"""被测试的 GM in-hand env cfg 源文件路径；只做 AST 读取，不执行模块。"""

REORIENT_COMMAND_PATH = Path(__file__).resolve().parents[1] / "mdp" / "commands" / "reorient_command.py"
r"""被测试的 command term 源文件路径；用于检查 command-owned debug marker contract。"""

MDP_INIT_PATH = Path(__file__).resolve().parents[1] / "mdp" / "__init__.py"
r"""被测试的 GM MDP re-export 文件路径；用于检查 `gm_mdp.xxx` 扁平 API。"""


def _module_ast() -> ast.Module:
    r"""解析 `inhand_env_cfg.py` 的 AST，避免纯 pytest 进程触发 Isaac Sim 绑定。

    Returns:
        ast.Module: Python 源码抽象语法树；后续测试只读取字面量与函数调用关键字。
    """

    return ast.parse(INHAND_ENV_CFG_PATH.read_text(encoding="utf-8"))  # 纯源码解析，不执行 import side effects


def _constant_values() -> dict[str, object]:
    r"""读取模块级常量，支持本文件用到的字符串拼接和乘法表达式。

    Returns:
        dict[str, object]: 常量名到 Python 值的映射，例如 `GM_DEFAULT_NUM_ENVS -> sample_count * envs_per_hand`。
    """

    values: dict[str, object] = {}  # 逐条保存已解析常量，供后续表达式引用
    target_names = {
        "GM_CLEAR_SKY_LIGHT_INTENSITY",
        "GM_CLEAR_SKY_TEXTURE_FILE",
        "GM_DEFAULT_ENVS_PER_HAND",
        "GM_DEFAULT_HAND_BANK_PATH",
        "GM_DEFAULT_HAND_SAMPLE_COUNT",
        "GM_DEFAULT_HAND_SAMPLE_SEED",
        "GM_DEFAULT_NUM_ENVS",
        "GM_DEFAULT_OBJECT_INIT_OFFSET_H",
        "GM_DEFAULT_OBJECT_INIT_POS_E",
    }  # 本测试只解析 first runnable slice 的实验规模锚点
    for node in _module_ast().body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id  # 模块级常量名
        if name not in target_names:
            continue  # 避免把 contact sensor name tuple 等无关声明纳入安全求值器
        values[name] = _eval_literal_expr(node.value, values)  # 只允许安全字面量表达式
    return values


def _eval_literal_expr(node: ast.AST, values: dict[str, object]) -> object:
    r"""求值本测试允许的声明式字面量表达式。

    Args:
        node (ast.AST): 待求值 AST 节点。
        values (dict[str, object]): 已解析常量表，服务 `GM_DEFAULT_NUM_ENVS=A*B` 这类表达式。

    Returns:
        object: 字符串、整数、布尔或简单乘法结果。

    Raises:
        ValueError: 当源码表达式超出测试允许范围时显式失败。
    """

    if isinstance(node, ast.Constant):
        return node.value  # 字符串 / 整数 / 布尔常量
    if isinstance(node, ast.Name):
        return values[node.id]  # 已解析模块常量引用
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value for part in node.values if isinstance(part, ast.Constant))  # f-string 字面量片段
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return str(_eval_literal_expr(node.left, values)) + str(_eval_literal_expr(node.right, values))  # 字符串拼接
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return int(_eval_literal_expr(node.left, values)) * int(_eval_literal_expr(node.right, values))  # $A\times B$
    if isinstance(node, ast.Tuple):
        return tuple(_eval_literal_expr(element, values) for element in node.elts)  # 如 fixed axis $k^h=(0,0,1)$
    raise ValueError(f"Unsupported declaration expression in contract test: {ast.dump(node)}")


def _default_hand_spawn_call() -> ast.Call:
    r"""定位 `DEFAULT_GM_HAND_SPAWN_CFG = HandSpawnCfg(...)` 调用。"""

    return _module_assign_call("DEFAULT_GM_HAND_SPAWN_CFG")


def _module_assign_call(assign_name: str) -> ast.Call:
    r"""定位模块级 `assign_name = SomeCall(...)` 声明。"""

    for node in _module_ast().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == assign_name and isinstance(node.value, ast.Call):
                return node.value  # 声明式 cfg / helper 调用节点
    raise AssertionError(f"{assign_name} declaration not found")


def _call_func_name(call: ast.Call) -> str:
    r"""返回简单函数调用的函数名，服务 AST contract test。"""

    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    raise AssertionError(f"Unsupported call func: {ast.dump(call.func)}")


def _keyword_call(call: ast.Call, keyword_name: str) -> ast.Call:
    r"""从一个配置调用中取出某个 keyword 的嵌套 call。"""

    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Call):
            return keyword.value  # 例如 `bank=HandBankCfg(...)` 或 `urdf=HandUrdfSpawnCfg(...)`
    raise AssertionError(f"keyword call {keyword_name!r} not found")


def _keyword_literal(call: ast.Call, keyword_name: str, values: dict[str, object]) -> object:
    r"""从配置调用中读取 keyword 的安全字面量值。"""

    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return _eval_literal_expr(keyword.value, values)  # 只允许常量 / 常量引用
    raise AssertionError(f"keyword {keyword_name!r} not found")


def _class_assign_call(class_name: str, assign_name: str) -> ast.Call:
    r"""在指定 class body 中定位形如 `assign_name = SomeCfg(...)` 的声明。"""

    for node in _module_ast().body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if isinstance(class_node, ast.Assign) and isinstance(class_node.value, ast.Call):
                if any(isinstance(target, ast.Name) and target.id == assign_name for target in class_node.targets):
                    return class_node.value  # class-level config field call
            if isinstance(class_node, ast.AnnAssign) and isinstance(class_node.target, ast.Name):
                if class_node.target.id == assign_name and isinstance(class_node.value, ast.Call):
                    return class_node.value  # class-level config field call
    raise AssertionError(f"{class_name}.{assign_name} declaration not found")


def _class_assigned_names(class_name: str) -> set[str]:
    r"""收集指定 config class body 中声明的 class-level 字段名。"""

    names: set[str] = set()  # scene cfg 中的 asset / sensor 字段名集合
    for node in _module_ast().body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if isinstance(class_node, ast.Assign):
                names.update(target.id for target in class_node.targets if isinstance(target, ast.Name))
            elif isinstance(class_node, ast.AnnAssign) and isinstance(class_node.target, ast.Name):
                names.add(class_node.target.id)
        return names
    raise AssertionError(f"class {class_name!r} not found")


def test_gm_inhand_default_hand_bank_is_fixed_reproducible_slice() -> None:
    r"""默认 hand bank 应表达可复现的 generated-hand 实验切片。"""

    values = _constant_values()  # 模块级数值锚点，如 sample_count / seed / env count
    bank_call = _keyword_call(_default_hand_spawn_call(), "bank")  # `HandBankCfg(...)` 声明

    assert _keyword_literal(bank_call, "source_mode", values) == "post_mutate"  # same-topology post-mutate 主线
    assert _keyword_literal(bank_call, "selection_mode", values) == "sample"  # 固定 seed 随机子集
    assert _keyword_literal(bank_call, "post_mutate_path", values) == values["GM_DEFAULT_HAND_BANK_PATH"]
    assert _keyword_literal(bank_call, "sample_count", values) == values["GM_DEFAULT_HAND_SAMPLE_COUNT"]
    assert values["GM_DEFAULT_HAND_SAMPLE_COUNT"] > 0  # 当前 preset 可调，但必须保留至少一个 hand
    assert _keyword_literal(bank_call, "sample_seed", values) == values["GM_DEFAULT_HAND_SAMPLE_SEED"] == 42


def test_gm_inhand_default_env_count_matches_hands_times_envs_per_hand() -> None:
    r"""默认并行规模应等于 hand sample count 乘以每手 env 数。"""

    values = _constant_values()  # 读取源码中的实验规模常量

    assert values["GM_DEFAULT_ENVS_PER_HAND"] == 32  # 当前 preset：每个 selected hand 分配 32 个 env
    assert values["GM_DEFAULT_NUM_ENVS"] == values["GM_DEFAULT_HAND_SAMPLE_COUNT"] * values["GM_DEFAULT_ENVS_PER_HAND"]
    assert values["GM_DEFAULT_NUM_ENVS"] > 0  # smoke / teacher 规模可调，但必须是正并行环境数


def test_gm_inhand_hand_spawn_enables_contact_reports() -> None:
    r"""hand URDF importer 必须打开 contact report，支撑 contact obs / reward。"""

    values = _constant_values()  # 布尔 / 字符串 keyword 求值上下文
    hand_spawn_call = _default_hand_spawn_call()  # `HandSpawnCfg(...)` 声明
    urdf_call = _keyword_call(hand_spawn_call, "urdf")  # `HandUrdfSpawnCfg(...)` 声明

    assert _keyword_literal(urdf_call, "activate_contact_sensors", values) is True  # ContactSensorCfg 依赖底层 report
    assert _keyword_literal(hand_spawn_call, "asset_routing", values) == "round_robin"  # 确定性 env-id routing
    assert _keyword_literal(hand_spawn_call, "restore_visual_materials", values) is True  # GUI 中保留 URDF anatomy debug 色
    assert _keyword_literal(hand_spawn_call, "validate_same_schema", values) is True  # 多资产 articulation fail-fast


def test_gm_inhand_object_init_uses_generated_hand_palm_offset() -> None:
    r"""object 初态应落在 generated hand `{h}` 掌心/指根区域，而不是沿用 LEAP 负 y 魔数。"""

    values = _constant_values()  # 解析 `{h}` 偏置和 `{e}` 初始位置常量
    object_call = _class_assign_call("GmInHandSceneCfg", "object")  # `object = RigidObjectCfg(...)`
    init_state_call = _keyword_call(object_call, "init_state")  # `RigidObjectCfg.InitialStateCfg(...)`

    assert values["GM_DEFAULT_OBJECT_INIT_OFFSET_H"] == (0.0, 0.055, 0.06)  # palm/finger 展开方向为 $+y^h$
    assert _keyword_literal(init_state_call, "pos", values) == values["GM_DEFAULT_OBJECT_INIT_POS_E"]
    assert values["GM_DEFAULT_OBJECT_INIT_POS_E"] == (0.0, 0.055, 0.56)  # $p^e_h=(0,0,0.5)$ 加 `{h}` 偏置


def test_gm_inhand_contact_layout_is_sidecar_derived_and_installed() -> None:
    r"""contact layout 应由 selected hand sidecar 推导，而不是在 env cfg 中硬编码四指字段。"""

    values = _constant_values()  # 解析 validate_all_assets=False 这类安全字面量
    source = INHAND_ENV_CFG_PATH.read_text(encoding="utf-8")  # 纯文本检查，不 import Isaac/pxr binding
    layout_call = _module_assign_call("GM_DEFAULT_CONTACT_LAYOUT")  # `build_contact_sensor_layout_from_hand_spawn(...)`
    scene_fields = _class_assigned_names("GmInHandSceneCfg")  # scene class body 中不应再有硬编码 contact 字段

    assert _call_func_name(layout_call) == "build_contact_sensor_layout_from_hand_spawn"
    assert isinstance(layout_call.args[0], ast.Name) and layout_call.args[0].id == "DEFAULT_GM_HAND_SPAWN_CFG"
    assert _keyword_literal(layout_call, "validate_all_assets", values) is False  # 默认只读首个 selected asset
    assert not any(name.startswith("contact_") for name in scene_fields)  # per-link sensors 改为 scene instance 动态安装
    assert "install_contact_sensors(self, GM_DEFAULT_CONTACT_LAYOUT)" in source  # scene __post_init__ 安装 sensors
    assert "GM_DEFAULT_CONTACT_LAYOUT.fingertip_sensor_names" in source  # obs/reward 从 layout 取 tip sensor names
    assert "GM_DEFAULT_CONTACT_LAYOUT.non_tip_sensor_names" in source  # bad contact 从 layout 取 non-tip sensor names


def test_gm_inhand_scene_uses_clear_sky_visual_preset() -> None:
    r"""GM in-hand GUI / smoke 应复用异构 smoke 的清天 HDRI，而不是纯灰 dome light。"""

    values = _constant_values()  # 解析 clear-sky HDRI 路径与 dome light 强度常量
    source = INHAND_ENV_CFG_PATH.read_text(encoding="utf-8")  # 纯文本检查 viewer 默认视角，不执行 cfg
    light_call = _class_assign_call("GmInHandSceneCfg", "light")  # `light = AssetBaseCfg(...)`
    spawn_call = _keyword_call(light_call, "spawn")  # `sim_utils.DomeLightCfg(...)`

    assert _keyword_literal(light_call, "prim_path", values) == "/World/skyLight"
    assert _keyword_literal(spawn_call, "intensity", values) == values["GM_CLEAR_SKY_LIGHT_INTENSITY"] == 750.0
    assert str(_keyword_literal(spawn_call, "texture_file", values)).endswith(
        "/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
    )
    assert "self.viewer.eye = (2.0, 2.0, 1.5)" in source  # 对齐 heterogeneous smoke 的可读观察高度
    assert "self.viewer.lookat = (0.0, 0.0, 0.5)" in source  # 对准 hand/object anchor，而不是默认看世界原点


def test_gm_mdp_preserves_flat_public_observation_and_reward_exports() -> None:
    r"""observations/rewards 转成 package 后，外部仍应通过 `gm_mdp.xxx` 扁平访问。"""

    source = MDP_INIT_PATH.read_text(encoding="utf-8")  # 纯文本检查 re-export，不 import Isaac runtime

    assert "from .observations import" in source  # package 化后仍从 `.observations` re-export obs terms
    assert "from .rewards import" in source  # package 化后仍从 `.rewards` re-export reward terms
    assert '"joint_pos_raw"' in source and '"fingertip_contact_binary"' in source
    assert '"fingertip_contact_force_h"' in source and "fingertip_contact_force_w" not in source
    assert '"object_pos_h"' in source and '"object_rot6d_h"' in source
    assert '"keypoint_reorientation_reward"' in source and '"good_fingertip_contact"' in source
    assert '"record_object_reset_anchor"' in source  # AnyMani 专属 reset anchor event 保持扁平导出
    assert "simple_no_cache_reset" not in source  # 不再导出聚合式 reset wrapper，避免 reset 语义重新杂糅


def test_gm_inhand_uses_split_reset_events() -> None:
    r"""GM in-hand 默认 reset 应拆成官方物理写入项和 AnyMani anchor 记录项。"""

    source = INHAND_ENV_CFG_PATH.read_text(encoding="utf-8")  # 纯文本检查，不触发 Isaac runtime
    reset_robot_call = _class_assign_call("GmEventsCfg", "reset_robot_joints")
    reset_object_call = _class_assign_call("GmEventsCfg", "reset_object")
    record_anchor_call = _class_assign_call("GmEventsCfg", "record_object_reset_anchor")

    assert _call_func_name(reset_robot_call) == "EventTerm"
    assert _call_func_name(reset_object_call) == "EventTerm"
    assert _call_func_name(record_anchor_call) == "EventTerm"
    assert "func=isaac_mdp.reset_joints_by_offset" in source
    assert '"position_range": (-0.05, 0.05)' in source
    assert "func=isaac_mdp.reset_root_state_uniform" in source
    assert '"yaw": (-0.2, 0.2)' in source  # 保留原 runnable slice 的小角度 object yaw 扰动
    assert "func=gm_mdp.record_object_reset_anchor" in source
    assert "func=gm_mdp.simple_no_cache_reset" not in source


def test_gm_inhand_task_aliases_point_to_tasks_env_cfg() -> None:
    r"""tasks-owned Gym aliases 应直接指向 GM in-hand env cfg，不携带训练 YAML。"""

    train_spec = gym.spec("AnyMani-GM-InHand-v0")  # 正式 GM task alias，不包含 distill 训练配置
    play_spec = gym.spec("AnyMani-GM-InHand-Play-v0")  # 小规模可视化 / smoke alias

    assert train_spec.kwargs["env_cfg_entry_point"].endswith("inhand_env_cfg:GmInHandEnvCfg")
    assert "rl_games_cfg_entry_point" not in train_spec.kwargs  # 训练算法入口属于 distill，不属于 tasks/gm
    assert play_spec.kwargs["env_cfg_entry_point"].endswith("inhand_env_cfg:GmInHandEnvCfg_PLAY")


def test_gm_inhand_play_uses_fixed_hand_z_axis_command() -> None:
    r"""PLAY 版应固定 `{h}` z 轴，服务人工可视检查与 fixed-axis smoke。"""

    values = _constant_values()  # fixed-axis tuple 解析上下文
    play_commands_call = _class_assign_call("GmInHandEnvCfg_PLAY", "commands")  # `commands = GmCommandsCfg(...)`
    play_goal_call = _keyword_call(play_commands_call, "goal_pose")  # `ReorientCommandCfg(...)`

    assert _keyword_literal(play_goal_call, "axis_mode", values) == "fixed"  # 固定轴，而非训练默认 random subgoal
    assert _keyword_literal(play_goal_call, "axis_resample_mode", values) == "episode"  # episode 内固定轴更适合视觉核对
    assert _keyword_literal(play_goal_call, "debug_vis", values) is True  # PLAY 入口必须显示 command-owned 虚拟目标物体
    assert _keyword_literal(play_goal_call, "fixed_axis_h", values) == (0.0, 0.0, 1.0)  # $k^h=e_z$，手心语义法向


def test_reorient_command_owns_goal_object_debug_marker() -> None:
    r"""目标姿态 marker 必须由 `ReorientCommand` 自己从 `goal_quat_w` 可视化。"""

    source = REORIENT_COMMAND_PATH.read_text(encoding="utf-8")  # 纯文本读取，避免 import Isaac/pxr binding

    assert "VisualizationMarkers" in source  # command term 创建 marker，而不是 env scene 放假物体
    assert "def _set_debug_vis_impl" in source  # Isaac Lab CommandTerm debug hook
    assert "def _debug_vis_callback" in source  # 每帧刷新目标姿态 marker
    assert "orientations=self.goal_quat_w" in source  # 姿态目标来自 command 内部单一真源
