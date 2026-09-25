"""Verified model, asset, reference, and cohort inputs for paper runtime commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from .paper_data import PaperAssetBundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "logs" / "benchmarks" / "heterogeneous_rotation"
DEFAULT_IMPLEMENTATION_CERTIFICATE = (
    REPOSITORY_ROOT / "source/anymani/anymani/publication/compatibility.json"
)

DEFAULT_REFERENCE_SOURCE_SHA256 = "65885b4916eab671658244f0cdde6eb753a9baa54a16d89dd0f7369a2d5c22c9"
DEFAULT_REFERENCE_CHECKPOINT_SHA256 = "afe71fa25e906c5f18eae96948585e2f8bcf8746adfb2a6fc97bbcb18bb7f110"
DEFAULT_REFERENCE_DOCUMENT: dict[str, Any] = {
    "artifact_type": "anymani.n000_fixed_mvp_reference",
    "schema_version": "2.2.0",
    "completed_steps": 600,
    "num_replicas": 16,
    "seed": 42,
    "protocol": {
        "horizon_s": 30.0,
        "policy_dt_s": 0.05,
        "first_trajectory_only": True,
        "deterministic_actor_mean": True,
        "object_scale": 1.1,
        "action_authority_rad_per_policy_step": 1.0 / 24.0,
    },
    "reference": {
        "goal_count_median": 52.0,
        "signed_net_turns_median": 3.638890266418457,
        "command_turn_ratio": 1.190839243855071,
        "interpretation": "(goals/12)/net_turns calibrates moving-goal tracking against physical net turns",
    },
    "reference_provenance": {
        "source_json_sha256": DEFAULT_REFERENCE_SOURCE_SHA256,
        "source_checkpoint_sha256": DEFAULT_REFERENCE_CHECKPOINT_SHA256,
        "completed_steps": 600,
        "replicas": 16,
        "evaluator_source_sha256": "2d301ec963afd3b4fa74b386441ef00f0d37f788e2f951e8a93767077aa1f239",
    },
}


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _contained_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Model path escapes its bundle: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class PaperModelBundle:
    """Verified paths and hashes from the installed paper model manifest."""

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path, *, expected_bundle_sha256: str | None = None) -> "PaperModelBundle":
        model_root = Path(root).expanduser().resolve(strict=True)
        receipt_path = model_root / ".bundle.json"
        manifest_path = model_root / "model_manifest.json"
        if not receipt_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing verified model bundle receipt or manifest under {model_root}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expected_bundle_sha256 is not None and receipt.get("sha256") != expected_bundle_sha256:
            raise ValueError("Installed model bundle SHA-256 disagrees with release.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_type") != "anymani.paper_model_bundle" or manifest.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported paper model bundle manifest")
        return cls(model_root, manifest)

    def path(self, name: str) -> Path:
        try:
            specification = self.manifest["models"][name]
        except KeyError as error:
            raise ValueError(f"Unknown paper model {name!r}") from error
        path = _contained_file(self.root, specification["path"])
        actual = sha256_file(path)
        if actual != specification["sha256"]:
            raise ValueError(f"Model checksum mismatch for {name}: expected {specification['sha256']}, got {actual}")
        if "bytes" in specification and path.stat().st_size != int(specification["bytes"]):
            raise ValueError(f"Model byte size mismatch for {name}")
        return path

    def sidecar_path(self, name: str) -> Path:
        try:
            specification = self.manifest[name]
        except KeyError as error:
            raise ValueError(f"Unknown paper model sidecar {name!r}") from error
        path = _contained_file(self.root, specification["path"])
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"Sidecar checksum mismatch for {name}")
        return path

    def teacher(self, family: str) -> Path:
        names = {"leap": "leap_teacher", "allegro": "allegro_teacher"}
        try:
            return self.path(names[family])
        except KeyError as error:
            raise ValueError(f"Teacher family must be 'leap' or 'allegro', got {family!r}") from error

    def student_actor(self) -> Path:
        return self.path("student")

    def student_torchscript(self) -> Path:
        return self.path("student_torchscript")

    def student_sidecar(self) -> Path:
        return self.sidecar_path("student_abi")

    def encoder(self) -> tuple[Path, str]:
        specification = self.manifest["models"]["encoder"]
        return self.path("encoder"), str(specification["sha256"])


@dataclass(frozen=True)
class StudentRuntimeArtifacts:
    """Resolved actor, TorchScript, and ABI sidecar files for student inference."""

    checkpoint: Path
    torchscript: Path
    sidecar: Path


def resolve_student_runtime_artifacts(
    models: PaperModelBundle,
    *,
    checkpoint: Path | None = None,
    torchscript: Path | None = None,
    sidecar: Path | None = None,
) -> StudentRuntimeArtifacts:
    """Resolve a matched student pair, using the packaged frozen actor by default."""
    if (checkpoint is None) != (torchscript is None):
        raise ValueError("--student-checkpoint and --student-torchscript must be supplied together")
    if checkpoint is None:
        actor_path = models.student_actor()
        torchscript_path = models.student_torchscript()
        sidecar_path = models.student_sidecar() if sidecar is None else sidecar
    else:
        actor_path = checkpoint
        torchscript_path = torchscript
        sidecar_path = Path(f"{torchscript_path}.json") if sidecar is None else sidecar
    paths = tuple(Path(path).expanduser().resolve(strict=True) for path in (actor_path, torchscript_path, sidecar_path))
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(f"Student runtime artifact is not a file: {missing}")
    return StudentRuntimeArtifacts(checkpoint=paths[0], torchscript=paths[1], sidecar=paths[2])


def evaluator_arguments_for_policy(
    policy: str,
    teacher_checkpoint: Path,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    """Keep the IL wrapper's teacher reference separate from the simulator checkpoint flag."""
    if policy == "student":
        if any(value == "--checkpoint" or value.startswith("--checkpoint=") for value in arguments):
            raise ValueError("student evaluator arguments must not include --checkpoint")
        return tuple(arguments)
    if policy == "teacher":
        return ("--checkpoint", str(teacher_checkpoint), *arguments)
    raise ValueError(f"unknown policy {policy!r}")


@dataclass(frozen=True)
class PaperRuntime:
    """Portable data inputs used before the task imports Isaac Lab."""

    repository_root: Path
    data_dir: Path
    models: PaperModelBundle
    assets: PaperAssetBundle

    @classmethod
    def open(cls, data_dir: str | Path = DEFAULT_DATA_DIR) -> "PaperRuntime":
        resolved_data = Path(data_dir).expanduser().resolve(strict=True)
        release_path = REPOSITORY_ROOT / "release.json"
        release = json.loads(release_path.read_text(encoding="utf-8"))
        artifacts = release.get("artifacts", {})
        model_sha = artifacts.get("models", {}).get("sha256")
        asset_sha = artifacts.get("assets", {}).get("sha256")
        models = PaperModelBundle.open(resolved_data / "models", expected_bundle_sha256=model_sha)
        asset_root = resolved_data / "assets"
        receipt_path = asset_root / ".bundle.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("sha256") != asset_sha:
            raise ValueError("Installed asset bundle SHA-256 disagrees with release.json")
        assets = PaperAssetBundle(asset_root)
        return cls(REPOSITORY_ROOT, resolved_data, models, assets)

    def cohort(self, name: str) -> "PaperCohortSelection":
        specification = self.assets.cohort(name)
        cohort = self.assets.resolve(name, ready_only=True)
        lock_path = self.assets.lock_path(name)
        if lock_path.resolve() != cohort.lock_path.resolve():
            raise ValueError("Paper cohort lock path disagrees with resolved cohort membership")
        return PaperCohortSelection(
            name=name,
            family=str(specification["family"]),
            cohort=cohort,
            lock_path=lock_path,
            catalog_root=self.assets.catalog_root(name),
            metadata={
                "cohort_name": name,
                "runtime_cohort_id": cohort.cohort_id,
                "runtime_lock_path": str(lock_path),
                "runtime_lock_sha256": cohort.lock_sha256,
                "parent_cohort_id": cohort.cohort_id,
                "parent_lock_sha256": cohort.lock_sha256,
                "parent_nominal_count": int(specification["nominal_count"]),
                "parent_ready_count": int(specification["ready_count"]),
                "selected_parent_member_indices": list(range(len(cohort.members))),
                "selected_source_member_keys": list(cohort.source_keys),
                "selected_asset_ids": [member.asset_id for member in cohort.members],
            },
        )

    def member_view(
        self,
        name: str,
        selector: int | str,
        *,
        lock_dir: str | Path,
    ) -> "PaperCohortSelection":
        view = self.assets.create_view(name, (selector,), lock_dir=lock_dir)
        specification = self.assets.cohort(name)
        return PaperCohortSelection(
            name=name,
            family=str(specification["family"]),
            cohort=view.cohort,
            lock_path=view.lock_path,
            catalog_root=self.assets.catalog_root(name),
            metadata=view.to_dict(),
        )

    def configure_task_environment(
        self,
        selection: "PaperCohortSelection",
        *,
        rl_games_root: Path | None = None,
    ) -> None:
        """Set package paths before any AppLauncher or task module is imported."""
        os.environ["ANYMANI_DATA_DIR"] = str(self.data_dir)
        os.environ["ANYMANI_HETERO_PAPER_COHORT"] = selection.name
        os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(selection.lock_path)
        os.environ["ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT"] = str(selection.catalog_root)
        if rl_games_root is not None:
            os.environ["ANYMANI_RL_GAMES_ROOT"] = str(rl_games_root.expanduser().resolve(strict=True))
        if selection.metadata.get("runtime_view_sha256") is not None:
            os.environ["ANYMANI_HETERO_PAPER_VIEW_LOCK"] = str(selection.lock_path)
        else:
            os.environ.pop("ANYMANI_HETERO_PAPER_VIEW_LOCK", None)

    def write_reference(self, output_dir: str | Path, reference_path: str | Path | None = None) -> Path:
        """Write the exact frozen N000 normalization values when no override is supplied."""
        if reference_path is not None:
            path = Path(reference_path).expanduser().resolve(strict=True)
            document = json.loads(path.read_text(encoding="utf-8"))
            _validate_reference(document)
            return path
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "n000-fixed-s1p1-adr0-reference.json"
        payload = (json.dumps(DEFAULT_REFERENCE_DOCUMENT, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"Reference output already exists with different contents: {path}")
        else:
            path.write_bytes(payload)
        return path


@dataclass(frozen=True)
class PaperCohortSelection:
    """Resolved ordered runtime cohort and its parent-lock/view provenance."""

    name: str
    family: str
    cohort: Any
    lock_path: Path
    catalog_root: Path
    metadata: dict[str, Any]


def _validate_reference(document: dict[str, Any]) -> None:
    if document.get("artifact_type") != "anymani.n000_fixed_mvp_reference":
        raise ValueError("Reference must be a frozen AnyMani N000 calibration artifact")
    reference = document.get("reference")
    protocol = document.get("protocol")
    if not isinstance(reference, dict) or not isinstance(protocol, dict):
        raise ValueError("Reference artifact must contain reference and protocol mappings")
    goal_median = reference.get("G0_goal_count_median", reference.get("goal_count_median"))
    if not isinstance(goal_median, (int, float)):
        raise ValueError("Reference artifact is missing the frozen goal-count median")
    turns_median = reference.get("N0_signed_net_turns_median", reference.get("signed_net_turns_median"))
    if not isinstance(turns_median, (int, float)):
        raise ValueError("Reference artifact is missing the frozen signed-turn median")
    if float(protocol.get("horizon_s", 0)) != 30.0:
        raise ValueError("Reference artifact must use the frozen 30-second horizon")


def create_run_directory(command: str, label: str = "") -> Path:
    """Create a unique benchmark case directory without overwriting prior evidence."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{label}" if label else ""
    path = DEFAULT_RUN_ROOT / f"publication-{command}-{timestamp}{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def checkpoint_manifest_sha256(path: Path) -> str | None:
    """Read the frozen cohort digest from a trusted, verified teacher checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = checkpoint.get("anymani_identity") if isinstance(checkpoint, dict) else None
    manifest = identity.get("manifest") if isinstance(identity, dict) else None
    digest = manifest.get("sha256") if isinstance(manifest, dict) else None
    return str(digest) if digest else None


def requires_cohort_transfer(selection: PaperCohortSelection, checkpoint_path: Path) -> bool:
    """Mark changed cohort bytes or a member view as explicit frozen-policy transfer."""
    if selection.metadata.get("runtime_view_sha256") is not None:
        return True
    return checkpoint_manifest_sha256(checkpoint_path) != selection.metadata.get("runtime_lock_sha256")


def run_module_main(module: str, arguments: Sequence[str]) -> int:
    """Run a fresh module process with this checkout first on its import path."""
    environment = os.environ.copy()
    checkout_source = str(REPOSITORY_ROOT / "source/anymani")
    inherited_paths = [
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    environment["PYTHONPATH"] = os.pathsep.join((checkout_source, *inherited_paths))
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        env=environment,
        check=False,
    )
    return int(result.returncode)


def run_subprocess(arguments: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    """Run one separate simulator process so each cohort gets a fresh Kit lifecycle."""
    result = subprocess.run([sys.executable, *arguments], env=env, check=False)
    return int(result.returncode)
