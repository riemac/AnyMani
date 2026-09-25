"Anchors relative bank paths at the source tree identified by VERSION and resolves explicit bundle paths."

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_anymani_root(*, start: Path | None = None) -> Path:
    "Resolves anymani root against the declared source paths and identity rules."

    probe = (start or Path(__file__)).resolve()
    for candidate in (probe, *probe.parents):
        package_root = candidate / "source" / "anymani" / "anymani"
        if (candidate / "VERSION").is_file() and package_root.is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate AnyMani repository root from {probe}")


def resolve_bank_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    "Resolves bank path against the declared source paths and identity rules."

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    root = (base_dir or resolve_anymani_root()).resolve(strict=False)
    return (root / raw_path).resolve(strict=False)


def resolve_post_mutate_root(cfg: Any) -> Path:
    "Resolves post mutate root against the declared source paths and identity rules."

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
    "Resolves container entry path against the declared source paths and identity rules."

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
