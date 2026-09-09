r"""掌托旋转policy checkpoint的method identity构造。

Identity绑定会改变策略数值语义的dataset/catalog/N040 precision/structured ABI、actor arm、task/reward与
PPO运行合同。训练resume要求完整run identity一致；固定evaluation使用checkpoint内method部分并另建独立
evaluation identity，因此可以使用不同replica数而不伪装成训练续接。
"""

from __future__ import annotations

import hashlib
import json
import math
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
PALM_ROTATION_IDENTITY_SCHEMA_VERSION = "4.0.0"
"""源码bytes定义实现身份；Git提交号另存来源，不改变同代码的完整续训资格。"""

_IMPLEMENTATION_PATHS = (
    "source/anymani/anymani/distill/models/palm_rotation_policy.py",
    "source/anymani/anymani/distill/rl/palm_rotation_ppo.py",
    "source/anymani/anymani/distill/rl/algorithms/ppo_batch.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_network.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_diagnostics.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_probes.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_warm_start.py",
    "source/anymani/anymani/distill/models/temporal_encoder.py",
    "source/anymani/anymani/distill/models/backbones/geometry_transformer.py",
    "source/anymani/anymani/distill/models/input_adapters/encoder.py",
    "source/anymani/anymani/distill/models/input_adapters/se3_invariant_encoder.py",
    "source/anymani/anymani/distill/rl/algorithms/gradient_audit.py",
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
    "source/anymani/anymani/tasks/hetero/mdp/episode_horizon.py",
    "source/anymani/anymani/tasks/hetero/mdp/curriculums.py",
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


def palm_rotation_code_provenance() -> dict[str, str]:
    r"""返回独立保存的代码来源；提交号不是策略/MDP数值语义的一部分。"""

    return {"git_head": _git_head(resolve_anymani_root())}


def palm_rotation_implementation_files() -> dict[str, str]:
    r"""对实际执行的算法、模型与任务源码计算逐文件SHA-256。"""

    root = resolve_anymani_root()
    return {path: _sha256(root / path) for path in _IMPLEMENTATION_PATHS}


def validate_palm_rotation_evaluation_identity(
    *,
    runtime_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    implementation_certificate: Mapping[str, Any] | None = None,
) -> None:
    r"""验证只读评估身份；跨实现只能消费精确绑定两端源码的重构证书。

    完整续训仍使用masked_ppo中的全字段严格验证。此函数只用于不更新参数的评估：资产、预抓取、N040、
    actor信息、任务与训练合同逐值一致；Git来源和日志schema不决定动作语义。实现文件若不同，必须提供
    check_palm_rotation_refactor产生的同源码SHA映射证书，不能用布尔开关跳过代码检查。
    """

    for label, identity in (("runtime", runtime_identity), ("checkpoint", checkpoint_identity)):
        if identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
            raise RuntimeError(f"{label} evaluation identity has unsupported schema")
        payload = {key: value for key, value in identity.items() if key != "identity_digest"}
        if identity.get("identity_digest") != _stable_digest(payload):
            raise RuntimeError(f"{label} evaluation identity digest is inconsistent with its payload")
    fields = (
        "task_id",
        "task_contract",
        "policy",
        "manifest",
        "pregrasp",
        "geometry_provider",
        "transport_abi",
        "training",
    )
    mismatched = [
        name
        for name in fields
        if name not in runtime_identity or runtime_identity[name] != checkpoint_identity.get(name)
    ]
    if mismatched:
        raise RuntimeError(f"evaluation semantic identity mismatch: {mismatched}")
    runtime_files = runtime_identity.get("implementation", {}).get("files")
    checkpoint_files = checkpoint_identity.get("implementation", {}).get("files")
    if (
        not isinstance(runtime_files, dict)
        or not runtime_files
        or not isinstance(checkpoint_files, dict)
        or not checkpoint_files
    ):
        raise RuntimeError("evaluation requires non-empty implementation file identities")
    if runtime_files == checkpoint_files:
        return
    certificate = implementation_certificate
    if not isinstance(certificate, Mapping):
        raise RuntimeError("evaluation implementation changed; an exact refactor certificate is required")
    if (
        certificate.get("artifact_type") != "anymani.palm_rotation.refactor_equivalence"
        or certificate.get("schema_version") != "1.0.0"
        or certificate.get("passed") is not True
        or certificate.get("reference_implementation_files") != checkpoint_files
        or certificate.get("current_implementation_files") != runtime_files
    ):
        raise RuntimeError("evaluation refactor certificate does not cover these exact implementations")


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

    if arm not in {"base", "residual", "direct", "direct_token"}:
        raise ValueError("palm-rotation arm must be base, residual, direct or direct_token")
    if not selected_rows or len(set(selected_rows)) != len(selected_rows):
        raise ValueError("palm-rotation identity requires non-empty unique selected rows")
    if not run_contract:
        raise ValueError("palm-rotation identity requires a non-empty PPO run contract")
    progress_weight = float(run_contract.get("rotation_progress_reward_weight", 5.0))  # reward/rad，历史值为5
    if not math.isfinite(progress_weight) or progress_weight < 0.0:
        raise ValueError("rotation progress reward weight must be finite and non-negative")
    pose_weight = float(run_contract.get("pose_keypoint_reward_weight", 1.0))  # 所选位置/全位姿kernel系数，reward/s
    if not math.isfinite(pose_weight) or pose_weight < 0.0:
        raise ValueError("pose keypoint reward weight must be finite and non-negative")
    pose_mode = run_contract.get("pose_keypoint_mode", "full_pose")  # 历史缺字段表示原全位姿核
    if pose_mode not in ("full_pose", "position_only"):
        raise ValueError("pose keypoint mode must be full_pose or position_only")
    canonical_run_contract = dict(run_contract)  # 调用者的训练记录不被原地修改
    if pose_mode == "full_pose":
        canonical_run_contract.pop("pose_keypoint_mode", None)  # 显式/隐式默认使用同一规范字段布局
    joint_anchor_weight = float(run_contract.get("joint_pose_anchor_weight", -0.5))
    if not math.isfinite(joint_anchor_weight) or joint_anchor_weight > 0.0:
        raise ValueError("joint pose anchor weight must be finite and non-positive")
    if not pregrasp.require_strict or int(pregrasp.rank) != 0 or len(pregrasp.bindings) != len(selected_rows):
        raise ValueError("palm-rotation method requires one strict rank-0 binding per selected asset")
    root = resolve_anymani_root()
    resolved_manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    catalog_root = Path(pregrasp.catalog_root)
    catalog_root = catalog_root if catalog_root.is_absolute() else root / catalog_root
    catalog_index = catalog_root / "index.json"
    if not resolved_manifest.is_file() or not catalog_index.is_file():
        raise FileNotFoundError("palm-rotation manifest or strict catalog index is missing")
    implementation_files = palm_rotation_implementation_files()
    key_digests = [hashlib.sha256(binding.key_json.encode("utf-8")).hexdigest() for binding in pregrasp.bindings]
    payload = {
        "identity_schema_version": PALM_ROTATION_IDENTITY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_contract": {
            "object": "DexCube",
            "object_scale": 1.1,
            "rotation_axis_h": [0.0, 0.0, 1.0],
            "subgoal_degrees": 30.0,
            "training_mdp_anchor": "N000-gm-tactile-rotation-v0.5.0",
            "training_goal_bonus": "strict-full-pose-and-position-2p5cm",
            "evaluation_primary": "physical-frontier-net-turns-directionality-and-survival",
            "rotation_frontier_degrees": 30.0,
            "rotation_frontier_reward_weight": 0.0,
            "rotation_progress_clip_rad_per_step": float(
                run_contract.get("rotation_progress_clip_rad_per_step", 0.025)
            ),
            # 非默认系数属于MDP身份；旧checkpoint的隐式5保持原字段布局。
            **({"rotation_progress_reward_weight": progress_weight} if progress_weight != 5.0 else {}),
            **({"pose_keypoint_reward_weight": pose_weight} if pose_weight != 1.0 else {}),
            **({"pose_keypoint_mode": pose_mode} if pose_mode != "full_pose" else {}),  # 测量几何独立于系数
            "strict_tracking_reward_weight": float(run_contract.get("strict_goal_reward_weight", 10.0)),
            **({"joint_pose_anchor_weight": joint_anchor_weight} if joint_anchor_weight != -0.5 else {}),
            "critic_task_state": "axis-goal-error-max-positive-net-and-current-net",
            "episode_seconds": float(run_contract.get("episode_seconds_max", 120.0)),
            "episode_seconds_min": float(run_contract.get("episode_seconds_min", 120.0)),
            "episode_horizon_sampling": "uniform-policy-step-interval",
            "adr_enabled": False,
            "pregrasp_rank": 0,
            "pregrasp_strict": True,
            "stable_joint_reduction": "reference-dof-16",
            "linear_velocity_penalty": "world-l2-squared",
            "reward_release": {
                "aggregation": "per-asset-ema-to-handedness-inclusive-cell-median",
                "start_turns": float(run_contract.get("reward_release_start_turns", 1.0)),
                "end_turns": float(run_contract.get("reward_release_end_turns", 2.0)),
                "ema_alpha": float(run_contract.get("reward_release_ema_alpha", 0.05)),
                "floor": float(run_contract.get("reward_release_floor", 0.0)),
                "reference_seconds": float(run_contract.get("reward_release_reference_seconds", 120.0)),
            },
        },
        "policy": {
            "arm": arm,
            "actor_contact": "tip-only-binary"
            if run_contract.get("actor_contact", "all") == "tip"
            else "all-owner-binary-no-force",
            "distribution": "mean-preserving-tanh-squashed-active-joint-diagonal-normal",
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
            "residual_decomposition": (
                "bounded-0p8-dynamic-film-base-plus-bounded-0p2-global-action-residual"
                if arm in {"base", "residual"}
                else None
            ),
            "direct_decomposition": {
                "direct": "full-authority-contextual-plus-local-skip",
                "direct_token": "full-authority-contextual-token-only",
            }.get(arm),
        },
        "manifest": {
            "path": _relative_or_absolute(resolved_manifest, root),
            "sha256": _sha256(resolved_manifest),
            "support_asset_count": len(selected_rows),
            "selected_rows": list(selected_rows),
        },
        "pregrasp": {
            "catalog_root": _relative_or_absolute(catalog_root, root),
            "index_sha256": _sha256(catalog_index),
            "ordered_key_digests": key_digests,
        },
        "geometry_provider": provider_identity,
        "implementation": {
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
        "training": json.loads(json.dumps(canonical_run_contract, sort_keys=True)),  # 非默认模式保留以供评价恢复
    }
    return {**payload, "identity_digest": _stable_digest(payload)}


__all__ = [
    "TASK_ID",
    "PALM_ROTATION_IDENTITY_SCHEMA_VERSION",
    "PalmRotationPregraspIdentityCfg",
    "build_palm_rotation_method_identity",
    "palm_rotation_code_provenance",
    "palm_rotation_implementation_files",
    "validate_palm_rotation_evaluation_identity",
]
