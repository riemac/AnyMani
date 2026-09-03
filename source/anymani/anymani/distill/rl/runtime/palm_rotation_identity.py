r"""掌托旋转policy checkpoint的method identity构造。

Identity绑定会改变策略数值语义的dataset/catalog/N040 precision/structured ABI、actor arm、task/reward与
PPO运行合同。训练resume要求完整run identity一致；固定evaluation使用checkpoint内method部分并另建独立
evaluation identity，因此可以使用不同replica数而不伪装成训练续接。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.diagnostics.recording.rl.palm_rotation import PALM_ROTATION_METRICS_SCHEMA_VERSION

from .palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)

TASK_ID = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"

_IMPLEMENTATION_PATHS = (
    "source/anymani/anymani/distill/models/palm_rotation_policy.py",
    "source/anymani/anymani/distill/rl/palm_rotation_ppo.py",
    "source/anymani/anymani/distill/rl/train_palm_rotation_mvp.py",
    "source/anymani/anymani/distill/rl/masked_ppo.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_identity.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_precision.py",
    "source/anymani/anymani/distill/rl/runtime/retained_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/structured_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_vecenv.py",
    "source/anymani/anymani/distill/rl/agents/heterogeneous_palm_rotation_mvp_ppo.yaml",
    "source/anymani/anymani/tasks/hetero/config/generated/palm_rotation_mvp_env_cfg.py",
    "source/anymani/anymani/tasks/hetero/mdp/actions.py",
    "source/anymani/anymani/tasks/hetero/mdp/commands.py",
    "source/anymani/anymani/tasks/hetero/mdp/contact_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/curriculum_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/events.py",
    "source/anymani/anymani/tasks/hetero/mdp/object_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/observation_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/observations.py",
    "source/anymani/anymani/tasks/hetero/mdp/rewards.py",
    "source/anymani/anymani/tasks/hetero/mdp/task_math.py",
    "source/anymani/anymani/tasks/hetero/contact_layout.py",
    "source/anymani/anymani/tasks/hetero/contact_sensors.py",
    "source/anymani/anymani/robots/hand_spawn.py",
)


class _Binding(Protocol):
    r"""Identity builder读取的最窄schema-3 binding surface。"""

    @property
    def key_json(self) -> str: ...


class PalmRotationPregraspIdentityCfg(Protocol):
    r"""避免共享identity模块导入Isaac EventTerm类型的结构合同。"""

    @property
    def catalog_root(self) -> str: ...

    @property
    def bindings(self) -> tuple[_Binding, ...]: ...

    @property
    def rank(self) -> int: ...

    @property
    def require_strict(self) -> bool: ...


def _sha256(path: Path) -> str:
    r"""流式计算manifest/catalog index identity。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    r"""对JSON-safe method identity计算canonical SHA-256。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    r"""仓库内路径统一写相对形式，外部路径保留absolute。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _git_head(root: Path) -> str:
    r"""读取run启动时真实Git HEAD；dirty语义另由逐文件SHA闭合。"""

    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError(f"unexpected AnyMani Git revision: {revision!r}")
    return revision


def build_palm_rotation_method_identity(
    *,
    provider_identity: dict[str, Any],
    manifest_path: Path,
    selected_rows: tuple[int, ...],
    pregrasp: PalmRotationPregraspIdentityCfg,
    arm: str,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    r"""构造训练resume与独立evaluation共同使用的exact method identity。"""

    if arm not in {"base", "residual"}:
        raise ValueError("palm-rotation arm must be base or residual")
    if len(selected_rows) != 80 or len(set(selected_rows)) != 80:
        raise ValueError("palm-rotation identity requires exactly 80 unique selected rows")
    if not run_contract:
        raise ValueError("palm-rotation identity requires a non-empty PPO run contract")
    if not pregrasp.require_strict or int(pregrasp.rank) != 0 or len(pregrasp.bindings) != 80:
        raise ValueError("palm-rotation method requires strict rank-0 bindings for all 80 assets")
    root = resolve_anymani_root()
    resolved_manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    catalog_root = Path(pregrasp.catalog_root)
    catalog_root = catalog_root if catalog_root.is_absolute() else root / catalog_root
    catalog_index = catalog_root / "index.json"
    if not resolved_manifest.is_file() or not catalog_index.is_file():
        raise FileNotFoundError("palm-rotation manifest or strict catalog index is missing")
    implementation_files = {path: _sha256(root / path) for path in _IMPLEMENTATION_PATHS}
    key_digests = [hashlib.sha256(binding.key_json.encode("utf-8")).hexdigest() for binding in pregrasp.bindings]
    payload = {
        "identity_schema_version": "3.0.0",
        "task_id": TASK_ID,
        "task_contract": {
            "object": "DexCube",
            "object_scale": 1.1,
            "rotation_axis_h": [0.0, 0.0, 1.0],
            "subgoal_degrees": 30.0,
            "episode_seconds": 120.0,
            "adr_enabled": False,
            "pregrasp_rank": 0,
            "pregrasp_strict": True,
            "stable_joint_reduction": "reference-dof-16",
            "linear_velocity_penalty": "world-l2-squared",
            "reward_release": "per-asset-ema-to-handedness-inclusive-cell-median",
        },
        "policy": {
            "arm": arm,
            "actor_contact": "all-owner-binary-no-force",
            "distribution": "mean-preserving-tanh-squashed-active-joint-diagonal-normal",
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
            "residual_decomposition": "bounded-0p8-dynamic-film-base-plus-bounded-0p2-global-action-residual",
        },
        "manifest": {
            "path": _relative_or_absolute(resolved_manifest, root),
            "sha256": _sha256(resolved_manifest),
            "selected_rows": list(selected_rows),
        },
        "pregrasp": {
            "catalog_root": _relative_or_absolute(catalog_root, root),
            "index_sha256": _sha256(catalog_index),
            "ordered_key_digests": key_digests,
        },
        "geometry_provider": provider_identity,
        "implementation": {
            "git_head": _git_head(root),
            "files": implementation_files,
        },
        "transport_abi": {
            "float_shapes": {key: list(shape) for key, shape in PALM_ROTATION_FLOAT_SHAPES.items()},
            "bool_shapes": {key: list(shape) for key, shape in PALM_ROTATION_BOOL_SHAPES.items()},
            "int16_shapes": {key: list(shape) for key, shape in PALM_ROTATION_INT16_SHAPES.items()},
        },
        "diagnostics": {
            "metrics_schema_version": PALM_ROTATION_METRICS_SCHEMA_VERSION,
            "parquet_writer": "polars-1.32.3-zstd",
            "trajectory_writer": "hdf5-gzip-v1",
        },
        "training": json.loads(json.dumps(dict(run_contract), sort_keys=True)),
    }
    return {**payload, "identity_digest": _stable_digest(payload)}


__all__ = ["TASK_ID", "PalmRotationPregraspIdentityCfg", "build_palm_rotation_method_identity"]
