"""Download and verify the versioned paper models and assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("models", "assets")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_archive(spec: dict, cache: Path) -> Path:
    expected = spec["sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("The release manifest contains an invalid SHA-256 value.")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (expected + ".tar.gz")
    if target.is_file() and file_hash(target) == expected:
        return target
    partial = target.with_suffix(".part")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "AnyMani-Downloader/1.0"})
    print(f"Downloading {spec['url']}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = file_hash(partial)
        if actual != expected:
            raise ValueError(f"Download checksum mismatch: expected {expected}, got {actual}")
        if "bytes" in spec and partial.stat().st_size != int(spec["bytes"]):
            raise ValueError("Download size does not match the release manifest.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def install_component(name: str, spec: dict, destination: Path) -> None:
    archive = fetch_archive(spec, destination / ".downloads")
    target = destination / name
    receipt = target / ".bundle.json"
    if target.exists():
        if receipt.is_file() and json.loads(receipt.read_text()).get("sha256") == spec["sha256"]:
            print(f"{name}: already installed at {target}")
            return
        raise ValueError(f"{target} already exists with different contents. Choose another --data-dir.")
    stage = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination))
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            # Python's data filter confines paths and links to the extraction directory.
            bundle.extractall(stage, filter="data")
        content = stage / spec.get("archive_root", ".")
        if content.is_symlink() or not content.resolve().is_relative_to(stage.resolve()):
            raise ValueError("The archive root must remain inside the bundle.")
        required = "model_manifest.json" if name == "models" else "paper_bundle.json"
        if not (content / required).is_file():
            raise ValueError(f"The {name} archive is missing {required}.")
        (content / ".bundle.json").write_text(
            json.dumps({"sha256": spec["sha256"], "url": spec["url"]}, indent=2) + "\n"
        )
        content.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"{name}: installed and verified at {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=(*COMPONENTS, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "release.json")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    names = COMPONENTS if args.component == "all" else (args.component,)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        install_component(name, manifest["artifacts"][name], args.data_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
