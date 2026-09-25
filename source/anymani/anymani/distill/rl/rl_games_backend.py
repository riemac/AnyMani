"""Select the installed rl-games release or an explicitly configured compatible source checkout."""

from __future__ import annotations

import importlib
from importlib import metadata, util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOCAL_RL_GAMES_ROOT: Path | None = None
EXPECTED_RL_GAMES_COMMIT = "36edd38823197e6e20c6cc4531765e654d13b80f"
EXPECTED_RL_GAMES_VERSION = "1.6.5"


@dataclass(frozen=True)
class RlGamesBackendInfo:
    """Resolved package provenance used by training and fixed evaluation."""

    root: Path
    package_file: Path
    git_commit: str | None
    expected_commit: str
    is_expected_commit: bool
    package_version: str | None
    expected_version: str
    is_expected_release: bool
    identity_source: str


def _git_commit(root: Path) -> str | None:
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        if Path(top_level.stdout.strip()).resolve() != root.resolve():
            return None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _installed_version() -> str | None:
    try:
        return metadata.version("rl-games")
    except metadata.PackageNotFoundError:
        return None


def _direct_url_commit() -> str | None:
    """Read a VCS revision recorded by pip for the installed distribution."""
    try:
        payload = metadata.distribution("rl-games").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not payload:
        return None
    try:
        document = json.loads(payload)
    except (TypeError, ValueError):
        return None
    vcs_info = document.get("vcs_info") if isinstance(document, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if not isinstance(commit, str) or len(commit) != 40:
        return None
    try:
        int(commit, 16)
    except ValueError:
        return None
    return commit.lower()


def _discover_root() -> Path:
    override = os.environ.get("ANYMANI_RL_GAMES_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=True)
    specification = util.find_spec("rl_games")
    if specification is None or specification.origin is None:
        raise FileNotFoundError("rl_games is not installed; install the pinned 1.6.5 release first")
    package_file = Path(specification.origin).expanduser().resolve(strict=True)
    if package_file.parent.name != "rl_games":
        raise RuntimeError(f"Unexpected rl_games package layout: {package_file}")
    return package_file.parent.parent


def prefer_local_rl_games(
    root: Path | None = LOCAL_RL_GAMES_ROOT,
    expected_commit: str = EXPECTED_RL_GAMES_COMMIT,
    expected_version: str = EXPECTED_RL_GAMES_VERSION,
    strict: bool = False,
) -> RlGamesBackendInfo:
    """Pin imports to an explicit source root or the installed Python package."""
    selected_root = (root or _discover_root()).expanduser().resolve(strict=True)
    package_dir = selected_root / "rl_games"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"rl_games package not found under backend root: {package_dir}")

    root_text = str(selected_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if "rl_games" in sys.modules:
        imported_file = Path(sys.modules["rl_games"].__file__).resolve()
        if not imported_file.is_relative_to(package_dir):
            raise RuntimeError(
                "rl_games was imported before backend pinning from a different package root. "
                f"Current: {imported_file}; expected under: {package_dir}."
            )

    module = importlib.import_module("rl_games")
    package_file = Path(module.__file__).resolve()
    if not package_file.is_relative_to(package_dir):
        raise RuntimeError(f"Imported rl_games package does not match selected root: {package_file}")
    checkout_commit = _git_commit(selected_root)
    distribution_commit = _direct_url_commit()
    commit = checkout_commit or distribution_commit
    identity_source = (
        "git_checkout"
        if checkout_commit is not None
        else "direct_url_vcs"
        if distribution_commit is not None
        else "distribution_version"
    )
    package_version = _installed_version()
    is_expected_commit = commit == expected_commit
    is_expected_release = is_expected_commit or (commit is None and package_version == expected_version)
    if strict and not is_expected_release:
        raise RuntimeError(
            f"rl_games backend mismatch: commit={commit!r}, version={package_version!r}; "
            f"expected commit {expected_commit} or release {expected_version}."
        )
    if not is_expected_release:
        print(
            f"[WARN] rl_games backend is commit={commit!r}, version={package_version!r}; "
            f"expected {expected_commit} or release {expected_version}."
        )
    print(f"[INFO] Using rl_games from: {package_file}")
    if commit is None:
        print(
            f"[INFO] Installed rl-games package version: {package_version!r}; "
            "identity is version-only because no VCS revision is available."
        )
    print(f"[INFO] rl_games identity source: {identity_source}")

    return RlGamesBackendInfo(
        root=selected_root,
        package_file=package_file,
        git_commit=commit,
        expected_commit=expected_commit,
        is_expected_commit=is_expected_commit,
        package_version=package_version,
        expected_version=expected_version,
        is_expected_release=is_expected_release,
        identity_source=identity_source,
    )


__all__ = [
    "EXPECTED_RL_GAMES_COMMIT",
    "EXPECTED_RL_GAMES_VERSION",
    "LOCAL_RL_GAMES_ROOT",
    "RlGamesBackendInfo",
    "prefer_local_rl_games",
]
