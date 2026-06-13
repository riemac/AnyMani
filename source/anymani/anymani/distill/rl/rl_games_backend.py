r"""`distill` 自包含的 rl_games backend 固定工具。

本模块不修改全局 Python 环境，也不要求用户手动调整 `PYTHONPATH`。它只在
`distill/rl` 训练入口 import rl_games 之前，把本地研究用 `rl_games` 源码路径
放到 `sys.path` 最前面，保证 teacher training 的 RL backend 可复现。

当前约定：

- 使用 `/home/hac/isaac/rl_games` 本地仓库；
- 仓库应处于 upstream `v1.6.5` tag，commit
  `36edd38823197e6e20c6cc4531765e654d13b80f`；
- AnyMani agent yaml 显式写 `torch_compile: false`，避免第一阶段 Transformer
  contract/debug 被 `torch.compile` 混淆。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOCAL_RL_GAMES_ROOT = Path("/home/hac/isaac/rl_games")
"""本机研究环境中固定使用的 rl_games 源码仓库根目录。"""

EXPECTED_RL_GAMES_COMMIT = "36edd38823197e6e20c6cc4531765e654d13b80f"
"""upstream `v1.6.5` tag 对应 commit，用于训练日志中的可复现核对。"""


@dataclass(frozen=True)
class RlGamesBackendInfo:
    r"""rl_games backend 的可复现信息。

    Args:
        root (Path): 被插入 import path 的本地 rl_games 仓库。
        package_file (Path): 实际 import 到的 `rl_games.__file__`。
        git_commit (str | None): 当前本地仓库 HEAD commit。
        expected_commit (str): 期望的 `v1.6.5` commit。
        is_expected_commit (bool): 当前 commit 是否与期望一致。
    """

    root: Path
    package_file: Path
    git_commit: str | None
    expected_commit: str
    is_expected_commit: bool


def _git_commit(root: Path) -> str | None:
    r"""读取本地 rl_games 仓库 HEAD commit。

    Args:
        root (Path): rl_games git 仓库根目录。

    Returns:
        str | None: HEAD commit；若仓库不可读则返回 `None`。
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def prefer_local_rl_games(
    root: Path = LOCAL_RL_GAMES_ROOT,
    expected_commit: str = EXPECTED_RL_GAMES_COMMIT,
    strict: bool = False,
) -> RlGamesBackendInfo:
    r"""把本地 rl_games 源码路径固定为当前 Python 进程的优先 import 来源。

    Args:
        root (Path): 本地 rl_games 仓库根目录。
        expected_commit (str): 期望 commit，用于复现检查。
        strict (bool): 若为 True，commit 不匹配时抛错；默认只打印 warning，便于
            研究阶段临时检查不同 patch。

    Returns:
        RlGamesBackendInfo: 实际 import 信息。

    Raises:
        FileNotFoundError: 当本地 rl_games package 不存在时抛出。
        RuntimeError: 当 `strict=True` 且 commit 不匹配时抛出。
    """

    root = root.expanduser().resolve()  # 绝对路径，避免 cwd 影响 import 行为
    package_dir = root / "rl_games"  # Python package 目录
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Local rl_games package not found: {package_dir}")

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)  # 必须在 import rl_games 前执行，覆盖 site-packages 1.6.1

    if "rl_games" in sys.modules:
        imported_file = Path(sys.modules["rl_games"].__file__).resolve()
        if not imported_file.is_relative_to(package_dir):
            raise RuntimeError(
                "rl_games was imported before distill backend pinning and does not point to the local source. "
                f"Current: {imported_file}; expected under: {package_dir}."
            )

    rl_games = importlib.import_module("rl_games")
    package_file = Path(rl_games.__file__).resolve()  # 实际 import 到的 package 文件
    commit = _git_commit(root)  # 当前本地源码 commit
    is_expected = commit == expected_commit
    if strict and not is_expected:
        raise RuntimeError(f"Local rl_games commit mismatch: got {commit}, expected {expected_commit}.")
    if not is_expected:
        print(f"[WARN] Local rl_games commit is {commit}, expected v1.6.5 commit {expected_commit}.")
    print(f"[INFO] Using rl_games from: {package_file}")

    return RlGamesBackendInfo(
        root=root,
        package_file=package_file,
        git_commit=commit,
        expected_commit=expected_commit,
        is_expected_commit=is_expected,
    )


__all__ = [
    "EXPECTED_RL_GAMES_COMMIT",
    "LOCAL_RL_GAMES_ROOT",
    "RlGamesBackendInfo",
    "prefer_local_rl_games",
]
