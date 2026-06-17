r"""Declaration contract tests for the GM teacher debug env cfg.

这些测试不导入 `gm_teacher_env_cfg.py`，因为该模块继承真实 `GmInHandEnvCfg`，会触发
Isaac Lab / USD / `pxr` 绑定。这里用源码级检查锁住 `distill` 对 `tasks/gm` 的
消费边界：teacher debug cfg 不再依赖已删除的 `asset_binding`，并沿用 `gm` 的
默认 hand selection / env-per-hand routing 合同。
"""

from __future__ import annotations

import ast
from pathlib import Path


GM_TEACHER_ENV_CFG_PATH = Path(__file__).resolve().parents[1] / "rl" / "gm_teacher_env_cfg.py"
r"""被测试的 distill teacher env cfg 源文件路径；只做 AST / 文本读取，不执行模块。"""


def _source_text() -> str:
    r"""读取 teacher env cfg 源码文本，避免 import-time Isaac Sim binding。"""

    return GM_TEACHER_ENV_CFG_PATH.read_text(encoding="utf-8")  # 纯文本读取，不触发任何注册或 USD binding


def _module_ast() -> ast.Module:
    r"""解析 teacher env cfg AST，供声明式 contract 检查使用。"""

    return ast.parse(_source_text())  # 只解析语法树，不执行 import


def _constant_assignment_names() -> set[str]:
    r"""收集模块级常量赋值名，确认 debug route 暴露了可读的数值锚点。"""

    names: set[str] = set()  # 记录如 `GM_TEACHER_DEBUG_NUM_ENVS` 的声明名
    for node in _module_ast().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names.add(node.targets[0].id)  # 模块级声明式常量名
    return names


def _imported_module_names() -> set[str]:
    r"""收集源码中的 import module 名称，只检查真实依赖，不检查说明性文档字符串。"""

    modules: set[str] = set()  # 记录 `import x` 与 `from x import y` 的 x
    for node in _module_ast().body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)  # 普通 import 依赖
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)  # from-import 依赖
    return modules


def test_gm_teacher_debug_cfg_does_not_import_deleted_asset_binding() -> None:
    r"""teacher debug cfg 不应再依赖旧 `asset_binding` 单资产接口。"""

    source = _source_text()  # 源码文本足以发现旧 import / 旧符号引用
    imported_modules = _imported_module_names()  # 真实 import 依赖，不含文档字符串

    assert "anymani.tasks.gm.asset_binding" not in imported_modules  # 已删除模块不能作为真实依赖
    assert "GmHandAssetRef" not in source  # 旧单资产引用类型不再属于 first runnable slice
    assert "build_hand_articulation_cfg" not in source  # 资产绑定应由 `GmInHandEnvCfg.scene.robot` 提供


def test_gm_teacher_debug_cfg_exposes_default_env_contract_constants() -> None:
    r"""teacher debug route 应显式暴露沿用 gm 默认规模的阅读锚点。"""

    names = _constant_assignment_names()  # 模块级数值锚点集合
    source = _source_text()  # 读取 import 行以确认常量来源

    assert "GM_DEFAULT_NUM_ENVS" in source  # 总 env 数来自 `tasks/gm` 默认合同
    assert "GM_DEFAULT_ENVS_PER_HAND" in source  # env-per-hand routing 锚点来自 `tasks/gm`
    assert "GM_TEACHER_DEBUG_NUM_ENVS" in names  # distill 侧给训练入口一个可读别名
    assert "GM_TEACHER_DEBUG_ENVS_PER_HAND" in names  # 保留 env-per-hand 解释锚点
    assert "GM_TEACHER_DEBUG_EPISODE_LENGTH_S" in names  # debug teacher 只调整 episode 长度


def test_gm_teacher_debug_cfg_does_not_rebind_scene_robot() -> None:
    r"""teacher debug cfg 只改训练时长，不再负责资产绑定。"""

    source = _source_text()  # 源码文本检查足以覆盖是否覆写 `scene.robot`

    assert "self.scene.robot" not in source  # 资产选择与 hand spawn 仍由 `tasks/gm` env cfg 持有
