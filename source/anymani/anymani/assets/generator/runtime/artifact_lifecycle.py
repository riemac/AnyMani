"Tracks files created by one generation attempt for safe cleanup after a failed candidate."

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path


def rollback_written_artifacts(written_paths: Iterable[Path], *, boundary_dir: Path) -> None:
    'Rolls back written artifacts.'

    boundary = Path(boundary_dir).resolve()
    normalized_paths = tuple(dict.fromkeys(Path(path).resolve() for path in written_paths))


    for path in normalized_paths:
        try:
            path.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(f"refusing to roll back artifact outside run boundary: {path}") from exc


    for path in sorted(normalized_paths, key=lambda candidate: len(candidate.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        _prune_empty_parents(path.parent, boundary=boundary)


def rollback_created_directory(directory: Path, *, boundary_dir: Path) -> None:
    'Rolls back created directory.'

    boundary = Path(boundary_dir).resolve()
    target = Path(directory).resolve()
    if target == boundary:
        raise ValueError("refusing to roll back the run root itself")
    try:
        target.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"refusing to roll back directory outside run boundary: {target}") from exc

    if target.is_dir():
        shutil.rmtree(target)
    _prune_empty_parents(target.parent, boundary=boundary)


def _prune_empty_parents(start_dir: Path, *, boundary: Path) -> None:

    current = Path(start_dir).resolve()
    while current != boundary:
        try:
            current.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(f"refusing to prune directory outside run boundary: {current}") from exc

        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


__all__ = ["rollback_created_directory", "rollback_written_artifacts"]
