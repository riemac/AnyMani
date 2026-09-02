r"""`tasks/hetero`自包含依赖方向的AST contract。"""

from __future__ import annotations

import ast
from pathlib import Path


def _production_python_files() -> tuple[Path, ...]:
    r"""返回hetero package中排除tests的全部生产源码。"""

    package_root = Path(__file__).resolve().parents[1]
    return tuple(
        path
        for path in sorted(package_root.rglob("*.py"))
        if "tests" not in path.relative_to(package_root).parts
    )


def test_hetero_production_code_does_not_import_legacy_task_families() -> None:
    r"""新任务不得以绝对import重新依赖GM或inhand运行时。"""

    forbidden = ("anymani.tasks.gm", "anymani.tasks.inhand")
    violations: list[str] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                continue
            for module in imported:
                if module.startswith(forbidden):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
    assert violations == []


def test_hetero_production_code_has_no_legacy_gym_id_literals() -> None:
    r"""新package不能通过字符串alias保留旧`AnyMani-GM-Heterogeneous*`入口。"""

    legacy_prefix = "AnyMani-GM-Heterogeneous"
    occurrences = [
        str(path)
        for path in _production_python_files()
        if legacy_prefix in path.read_text(encoding="utf-8")
    ]
    assert occurrences == []


def test_legacy_gm_multi_asset_ids_are_absent_from_gym_registry() -> None:
    r"""Import当前task packages后，旧multi-asset IDs必须消失且唯一新hetero ID存在。"""

    import gymnasium as gym

    import anymani.tasks.gm  # noqa: F401  # 注册仍受支持的single-asset/LEAP环境
    import anymani.tasks.hetero  # noqa: F401  # 注册当前generated heterogeneous环境

    retired_prefixes = (
        "AnyMani-GM-Heterogeneous",
        "AnyMani-GM-Canonical",
        "AnyMani-GM-InHand",
    )
    retired = sorted(env_id for env_id in gym.registry if env_id.startswith(retired_prefixes))
    assert retired == []
    assert "AnyMani-Hetero-Generated-TactileRotation-v0" in gym.registry
