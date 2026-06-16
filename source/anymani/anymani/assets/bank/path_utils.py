r"""Hand asset bank 的路径解析工具。

本模块只处理路径规范化，不读取 URDF / YAML / mesh 内容。设计上把“用户手写的相对
路径”统一锚定到 AnyMani 仓库根目录，而不是当前 shell 工作目录或 `assets/` 子目录，
这样训练配置在 VSCode、pytest、脚本入口之间移动时仍保持同一语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_anymani_root(*, start: Path | None = None) -> Path:
    r"""向上查找 AnyMani 仓库根目录。

    Args:
        start (Path | None): 可选起点；默认为本文件路径。

    Returns:
        Path: AnyMani 仓库根目录。

    Raises:
        RuntimeError: 当无法从当前源码位置识别仓库根时抛出。
    """

    probe = (start or Path(__file__)).resolve()  # 从源码文件位置出发，避免依赖 shell cwd
    for candidate in (probe, *probe.parents):
        package_root = candidate / "source" / "anymani" / "anymani"  # AnyMani repo 的稳定源码布局
        if (candidate / "AGENTS.md").is_file() and package_root.is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate AnyMani repository root from {probe}")


def resolve_bank_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    r"""把用户填写的 bank 路径规范化为绝对路径。

    Args:
        path (str | Path): 用户配置中的绝对或相对路径。
        base_dir (Path | None): 相对路径基准；为 `None` 时使用 AnyMani repo root。

    Returns:
        Path: 规范化后的绝对路径；不要求目标一定存在。
    """

    raw_path = Path(path).expanduser()  # 支持 `~/...`，但不改变普通相对路径语义
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    root = (base_dir or resolve_anymani_root()).resolve(strict=False)
    return (root / raw_path).resolve(strict=False)


def resolve_post_mutate_root(cfg: Any) -> Path:
    r"""从 `HandBankCfg` 风格对象解析 post-mutate run 根目录。

    支持两种用户写法：

    - 直接给 `post_mutate_path="source/.../<timestamp>"`；
    - 给 `pre_made_path="source/.../<topology>"` 与 `post_mutate_name="<timestamp>"`。

    若 `pre_made_path` 存在且 `post_mutate_path` 只是单段名称，也把它视为 run name，
    兼容手写配置中 `post_mutate_path="2026-..."` 的简洁形式。
    """

    pre_made_path = getattr(cfg, "pre_made_path", None)
    post_mutate_path = getattr(cfg, "post_mutate_path", None)
    post_mutate_name = getattr(cfg, "post_mutate_name", None)

    if post_mutate_name is not None:
        if pre_made_path is None:
            raise ValueError("post_mutate_name requires pre_made_path as its parent topology directory")
        if post_mutate_path is not None:
            raise ValueError("Use either post_mutate_path or post_mutate_name, not both")
        return (resolve_bank_path(pre_made_path) / str(post_mutate_name)).resolve(strict=False)

    if post_mutate_path is None:
        raise ValueError("post_mutate source requires post_mutate_path or post_mutate_name")

    raw_post_path = Path(post_mutate_path).expanduser()
    if pre_made_path is not None and not raw_post_path.is_absolute() and len(raw_post_path.parts) == 1:
        return (resolve_bank_path(pre_made_path) / raw_post_path).resolve(strict=False)
    return resolve_bank_path(raw_post_path)


def resolve_container_entry_path(path: str | Path, *, source_root: Path | None = None) -> Path:
    r"""解析单个 `HandContainerCfg.path`。

    Args:
        path (str | Path): 单个 container 的入口路径，可为 sample id、bundle 目录或 URDF 文件。
        source_root (Path | None): 相对 sample id 的 post-mutate run 根；为 `None` 时回退到 repo root。

    Returns:
        Path: 规范化后的绝对路径；不要求目标一定存在。
    """

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return resolve_bank_path(raw_path, base_dir=source_root)


__all__ = [
    "resolve_anymani_root",
    "resolve_bank_path",
    "resolve_container_entry_path",
    "resolve_post_mutate_root",
]
