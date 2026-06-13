r"""Pure config contract tests for `ClampedRelativeJointActionCfg`.

本文件专门防止一次真实 IsaacLab smoke 暴露的回归：`@configclass` 会在处理
class body 时捕获 dataclass field default；若先写 `class_type=None`、再在类定义后
后置赋值，则 `ClampedRelativeJointActionCfg()` 实例中的 `class_type` 仍可能是
`None`，最终导致 `ActionManager` 在构造 action term 时调用 `None`。

这里不启动 Isaac Sim，也不构造 `ManagerBasedRLEnv`。我们只核对配置层 contract：
运行时实例化出来的 cfg 必须携带真实 action class。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _load_action_module():
    r"""直接加载 action 文件，避免触发 AnyMani task registry 副作用。

    Returns:
        module: 包含 `ClampedRelativeJointActionCfg` 与
            `ClampedRelativeJointPositionAction` 的临时模块对象。
    """

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "actions" / "clamped_relative_action.py"
    spec = importlib.util.spec_from_file_location("gm_clamped_relative_action_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load clamped relative action module from {module_path}")

    # 只为本次 exec 安装极小 IsaacLab stub：目标是模拟 `@configclass` 捕获 field default，
    # 而不是启动真实 Isaac Sim / Omni binding。exec 结束后恢复 sys.modules，避免污染其他测试。
    stub_names = _install_isaaclab_action_stubs()  # 返回被本测试临时接管的模块名列表
    module = importlib.util.module_from_spec(spec)  # 临时模块，不污染 `anymani.tasks` 注册路径
    previous_target_module = sys.modules.get(spec.name)  # dataclass 解析字符串注解时需要模块可从 sys.modules 找回
    sys.modules[spec.name] = module  # 临时注册待 exec 模块，避免 dataclass 处理 `from __future__` 注解时报错
    try:
        spec.loader.exec_module(module)  # 执行文件，得到 configclass 处理后的 cfg 类
    finally:
        if previous_target_module is None:
            sys.modules.pop(spec.name, None)  # 本测试创建的临时模块，exec 后立即移除
        else:
            sys.modules[spec.name] = previous_target_module  # 恢复同名旧模块（理论上通常不存在）
        _restore_modules(stub_names)  # 恢复真实 IsaacLab 或其他测试注入的 stub
    return module


def _install_isaaclab_action_stubs() -> dict[str, types.ModuleType | None]:
    r"""安装 action cfg 测试所需的最小 IsaacLab stub。

    Returns:
        dict[str, types.ModuleType | None]: 每个被接管模块原先的 `sys.modules` 状态；
            `None` 表示原先不存在，恢复时应删除。
    """

    class ActionTerm:  # noqa: D401 - stub 类型只用于 `class_type` 注解
        r"""IsaacLab `ActionTerm` 的占位类型。"""

    class RelativeJointPositionAction:  # noqa: D401 - stub 父类只需允许继承
        r"""IsaacLab `RelativeJointPositionAction` 的占位父类。"""

        def __init__(self, cfg, env):
            r"""保存 cfg/env，满足子类 `super().__init__` 的最小调用语义。"""

            self.cfg = cfg  # action 配置对象，仅测试继承链是否可构造
            self.env = env  # env 占位对象，纯 config 测试不会读取

    @dataclass
    class RelativeJointPositionActionCfg:
        r"""IsaacLab 相对关节动作 cfg 的最小 dataclass 近似。

        这里保留 `class_type`、`asset_name` 与 `joint_names` 等字段，是为了让
        子类 `@configclass` 的 field default 捕获行为与真实失败模式一致：
        子类实例里的 `class_type` 必须来自 class body，而不是类定义后的后置赋值。
        """

        asset_name: str = ""  # articulation 名称，真实 cfg 中指向 `scene.robot`
        joint_names: list[str] | None = None  # 被控制关节名 regex 列表
        class_type: type | None = None  # 父类默认值；子类必须覆盖为真实 action 类
        scale: float = 1.0  # raw action 到 rad delta 的缩放系数
        preserve_order: bool = False  # 是否保留 articulation joint order
        clip: dict[str, tuple[float, float]] | None = None  # per-step delta clip 配置

    def configclass(cls):
        r"""用标准 dataclass 模拟 IsaacLab `configclass` 的 default 捕获。"""

        return dataclass(cls)  # dataclass 会在装饰时捕获 class body field default

    # 构造 dotted import 需要的 package/module 层级；package stub 设置 `__path__`。
    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    mdp = types.ModuleType("isaaclab.envs.mdp")
    actions = types.ModuleType("isaaclab.envs.mdp.actions")
    actions_cfg = types.ModuleType("isaaclab.envs.mdp.actions.actions_cfg")
    joint_actions = types.ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    managers = types.ModuleType("isaaclab.managers")
    action_manager = types.ModuleType("isaaclab.managers.action_manager")
    utils = types.ModuleType("isaaclab.utils")

    for package in (isaaclab, envs, mdp, actions, managers):
        package.__path__ = []  # 标记为 package，允许 Python 解析子模块 import

    actions_cfg.RelativeJointPositionActionCfg = RelativeJointPositionActionCfg  # cfg 父类 stub
    joint_actions.RelativeJointPositionAction = RelativeJointPositionAction  # action 父类 stub
    action_manager.ActionTerm = ActionTerm  # `class_type: type[ActionTerm]` 注解 stub
    utils.configclass = configclass  # 与真实 bug 相关的 dataclass/configclass 行为

    isaaclab.envs = envs  # 补齐 parent attribute，便于 import machinery 访问
    isaaclab.managers = managers  # 补齐 parent attribute
    isaaclab.utils = utils  # 补齐 parent attribute
    envs.mdp = mdp  # 补齐 `isaaclab.envs.mdp`
    mdp.actions = actions  # 补齐 `isaaclab.envs.mdp.actions`
    actions.actions_cfg = actions_cfg  # 补齐 leaf module attribute
    actions.joint_actions = joint_actions  # 补齐 leaf module attribute
    managers.action_manager = action_manager  # 补齐 `isaaclab.managers.action_manager`

    replacements = {
        "isaaclab": isaaclab,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp,
        "isaaclab.envs.mdp.actions": actions,
        "isaaclab.envs.mdp.actions.actions_cfg": actions_cfg,
        "isaaclab.envs.mdp.actions.joint_actions": joint_actions,
        "isaaclab.managers": managers,
        "isaaclab.managers.action_manager": action_manager,
        "isaaclab.utils": utils,
    }
    previous = {name: sys.modules.get(name) for name in replacements}  # 记录真实模块或其他测试 stub
    sys.modules.update(replacements)  # 临时接管 import，确保不触发真实 Isaac Sim
    return previous


def _restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    r"""恢复 `_install_isaaclab_action_stubs` 接管前的 `sys.modules` 状态。"""

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)  # 原先不存在：删除本测试 stub
        else:
            sys.modules[name] = module  # 原先存在：恢复真实模块或其他测试 stub


def test_clamped_relative_action_cfg_instance_keeps_action_class_type() -> None:
    r"""`cfg.class_type` 在实例上必须是可调用 action 类，而不是 `None`。

    该断言覆盖真实失败模式：类属性后置赋值看起来正确，但 dataclass/configclass
    实例字段 default 仍为 `None`。因此测试必须检查实例，而不仅是类属性。
    """

    module = _load_action_module()  # 直接加载目标文件，保持测试范围只覆盖 cfg contract
    cfg = module.ClampedRelativeJointActionCfg(asset_name="robot", joint_names=[".*"])  # 最小 action cfg

    assert cfg.class_type is module.ClampedRelativeJointPositionAction  # ActionManager 应实例化的真实 action 类
    assert callable(cfg.class_type)  # 防止 `NoneType is not callable` 在 smoke 阶段才暴露
