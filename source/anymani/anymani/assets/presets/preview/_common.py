"Declares  common values used by reproducible hand construction."

from __future__ import annotations

from pathlib import Path
import sys
import tempfile



#   source/anymani/anymani/assets/presets/preview/_common.py



REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_ROOT = Path(__file__).resolve().parents[4]


def bootstrap_python_path() -> None:
    "Adds the source package root for direct preview-script execution."

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


def resolve_output_dir(output_dir: str | None, *, prefix: str) -> Path:
    'Resolves and creates the output directory.'

    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_"))


def print_export_result(*, label: str, output_dir: Path, written: list[Path] | None = None) -> None:
    "Prints generated bundle paths and validation status for a preview run."

    written = written or []
    if written:
        print(f"[INFO] {label} exported:")
        for path in written:
            print(f"  - {path}")
    else:
        print(f"[INFO] {label} finished. Output directory: {output_dir}")


def infer_family_from_palm_preset(preset_name: str) -> str | None:
    'Infers family from palm preset.'

    if preset_name.startswith("com_"):
        return preset_name.removeprefix("com_")
    if preset_name.startswith("single_box_"):
        return preset_name.removeprefix("single_box_")
    return None


__all__ = [
    "REPO_ROOT",
    "SOURCE_ROOT",
    "bootstrap_python_path",
    "resolve_output_dir",
    "print_export_result",
    "infer_family_from_palm_preset",
]
