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
        "GM_DEFAULT_ENVS_PER_HAND",
        "GM_DEFAULT_HAND_BANK_PATH",
        "GM_DEFAULT_HAND_SAMPLE_COUNT",
        "GM_DEFAULT_HAND_SAMPLE_SEED",
        "GM_FINGERTIP_CONTACT_SENSOR_NAMES",
        "GM_NON_TIP_CONTACT_SENSOR_NAMES",
        "GM_DEFAULT_NUM_ENVS",
        "GM_DEFAULT_OBJECT_INIT_OFFSET_H",
        "GM_DEFAULT_OBJECT_INIT_POS_E",
    }  # 本测试只解析 first runnable slice 的实验规模锚点，跳过 sensor tuple 等无关声明
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

    for node in _module_ast().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "DEFAULT_GM_HAND_SPAWN_CFG" and isinstance(node.value, ast.Call):
                return node.value  # HandSpawnCfg(...) 调用节点
    raise AssertionError("DEFAULT_GM_HAND_SPAWN_CFG declaration not found")


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


def test_gm_inhand_contact_layout_declares_tip_palm_and_non_tip_sensors() -> None:
    r"""contact layout 应覆盖 fingertip、palm 与 non-tip links，支撑 obs/reward 读取。"""

    values = _constant_values()  # 解析 sensor name tuple 常量
    scene_fields = _class_assigned_names("GmInHandSceneCfg")  # scene 中显式声明的 sensor cfg 字段

    expected_contact_fields = set(values["GM_FINGERTIP_CONTACT_SENSOR_NAMES"]) | set(
        values["GM_NON_TIP_CONTACT_SENSOR_NAMES"]
    )  # reward/obs 会按这些名字从 env.scene 读取 ContactSensor

    assert "contact_palm" in expected_contact_fields  # palm penalty 必须显式存在
    assert set(values["GM_FINGERTIP_CONTACT_SENSOR_NAMES"]) == {
        "contact_index_tip",
        "contact_middle_tip",
        "contact_ring_tip",
        "contact_thumb_tip",
    }  # 四指 fingertip binary contact 顺序合同
    assert expected_contact_fields.issubset(scene_fields)  # 每个被 obs/reward 引用的 sensor 都必须在 scene 声明


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
