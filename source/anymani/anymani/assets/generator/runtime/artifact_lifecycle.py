"""生成候选在 validator / exporter 边界上的临时文件生命周期。

程序化 `cs` fingertip 必须先物化成 OBJ，physics closure 才能读取真实 collision
几何；因此不能简单把 validator 移到写文件之前。本模块只负责记录并回滚“一次候选
尝试中新写出的文件”，不删除已有资产，也不参与 mesh 几何或动力学计算。
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path


def rollback_written_artifacts(written_paths: Iterable[Path], *, boundary_dir: Path) -> None:
    r"""删除一次失败候选新写出的文件，并清理其空父目录。

    回滚集合只来自 materializer 返回的 ``written_paths``。已存在、被多个 joint
    引用或来自历史 bundle 的 mesh 不在集合内，因此不会被误删。父目录清理严格停在
    当前 run root，不会越过生成运行的审计边界。

    Args:
        written_paths (Iterable[Path]): 本次候选真正新建的文件路径。
        boundary_dir (Path): 允许清理的最高目录，通常是当前 run root。

    Raises:
        ValueError: 任一待回滚路径不位于 ``boundary_dir`` 内时抛出，防止路径错误
            导致跨 run 删除。
    """

    boundary = Path(boundary_dir).resolve()  # run root 是回滚能触及的最高目录
    normalized_paths = tuple(dict.fromkeys(Path(path).resolve() for path in written_paths))

    # 先验证全部路径，再执行任何删除；这样一个越界路径不会造成部分回滚。
    for path in normalized_paths:
        try:
            path.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(f"refusing to roll back artifact outside run boundary: {path}") from exc

    # 深路径优先删除；同一 mesh 被多个 joint 引用时，去重后的文件只删除一次。
    for path in sorted(normalized_paths, key=lambda candidate: len(candidate.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        _prune_empty_parents(path.parent, boundary=boundary)


def rollback_created_directory(directory: Path, *, boundary_dir: Path) -> None:
    r"""删除一次候选新建的完整目录，用于回滚 exporter 的部分写入。

    该函数只能在调用方已经证明 ``directory`` 在候选开始前不存在时使用。与
    ``rollback_written_artifacts`` 不同，这里会删除目录内所有部分导出文件，例如已写出的
    `hand.urdf`、尚未完成的 sidecar 和候选 mesh。

    Args:
        directory (Path): 本次候选独占、需要整体回滚的 topology 根。
        boundary_dir (Path): 当前 run root，限制删除不能越过本轮生成边界。

    Raises:
        ValueError: 目标等于 run root 或位于 run root 之外时抛出。
    """

    boundary = Path(boundary_dir).resolve()
    target = Path(directory).resolve()
    if target == boundary:
        raise ValueError("refusing to roll back the run root itself")
    try:
        target.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"refusing to roll back directory outside run boundary: {target}") from exc

    if target.is_dir():
        shutil.rmtree(target)  # topology 根由当前候选独占，因此部分 URDF/sidecar/mesh 应一起回滚
    _prune_empty_parents(target.parent, boundary=boundary)


def _prune_empty_parents(start_dir: Path, *, boundary: Path) -> None:
    r"""从一个已删除文件的父目录向上移除空目录，但保留 run root。"""

    current = Path(start_dir).resolve()
    while current != boundary:
        try:
            current.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(f"refusing to prune directory outside run boundary: {current}") from exc

        try:
            current.rmdir()  # 只有目录完全为空时才成功；成功 bundle 或并行 sibling 会自然阻止上溯
        except OSError:
            break
        current = current.parent


__all__ = ["rollback_created_directory", "rollback_written_artifacts"]
