r"""在accepted single-asset teacher固定轨迹上执行current actor mean-imitation tiny overfit。

该入口只训练`PalmRotationResidualActor`，冻结SSL retained geometry encoder，不创建environment、physics scene、
rl_games、critic或PPO。N040装配依赖Isaac USD类型，故入口先启动headless AppLauncher，再按运行时顺序导入provider。
Teacher标签是旧policy经环境wrapper裁剪后、按joint name重排到canonical轴的实际动作：

$$
y_t=P_{native\rightarrow canonical}\operatorname{clip}(\mu_t^{teacher},-1,1).
$$

Replica按固定modulo规则整体切分，相关时间sample不会跨train/validation。总体MSE之外独立报告饱和动作与非饱和
切换状态，避免teacher约85%的$\pm1$标签让常数或上一动作预测器伪装成成功student。

`--additional_dataset`按source文件聚合teacher-on-policy与DAgger student-occupancy correction states；每个source
先执行同一replica隔离，再在split内部拼接。Run identity逐source绑定SHA、behavior policy与sample count，最终
validation同时报告aggregate和per-source指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

from anymani.distill.il.n000_teacher_mean import (
    TeacherMeanMetrics,
    replica_modulo_split,
    teacher_mean_loss,
    teacher_mean_metrics,
    training_batch_indices,
)
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorObservation,
    PalmRotationGeometry,
    PalmRotationResidualActor,
)
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic CUDA matmul workspace contract

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATASET = ROOT / "outputs/hetero/distillation/n000-teacher-mean-r16-stride4-20260903.h5"
EXPECTED_TEACHER_CHECKPOINT_SHA256 = "afe71fa25e906c5f18eae96948585e2f8bcf8746adfb2a6fc97bbcb18bb7f110"
EXPECTED_SOURCE_ASSET_ID = "f5d8c069"
FORMAL_GEOMETRY_ROW = 848


@dataclass(frozen=True)
class SupervisedBank:
    r"""一个replica-isolated split的GPU-resident actor输入、标签与样本身份。"""

    observation: PalmRotationActorObservation
    target: torch.Tensor  # `[N,16]` canonical environment action
    previous_action: torch.Tensor  # `[N,16]`，teacher actor packet中的$a_{t-1}$
    replica_id: torch.Tensor  # long `[N]`
    policy_step: torch.Tensor  # long `[N]`

    @property
    def sample_count(self) -> int:
        r"""返回有效state-action samples数量。"""

        return int(self.target.shape[0])


def _sha256(path: Path) -> str:
    r"""流式计算dataset、实现文件与checkpoint identity。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    r"""对JSON-safe run identity计算canonical SHA-256。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _h5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    r"""读取一个必需numeric dataset，并拒绝同名group。"""

    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        raise RuntimeError(f"teacher HDF5 misses numeric dataset {name!r}")
    return cast(h5py.Dataset, value)


def _slice_observation(observation: PalmRotationActorObservation, index: torch.Tensor | slice) -> PalmRotationActorObservation:
    r"""沿sample轴同步切分current/history/contact/limits/masks。"""

    return PalmRotationActorObservation(
        jnt_current=observation.jnt_current[index],
        jnt_history=observation.jnt_history[index],
        jnt_limits=observation.jnt_limits[index],
        owner_contact=observation.owner_contact[index],
        jnt_valid=observation.jnt_valid[index],
        tip_valid=observation.tip_valid[index],
        owner_valid=observation.owner_valid[index],
    )


def _slice_geometry(geometry: PalmRotationGeometry, index: torch.Tensor | slice) -> PalmRotationGeometry:
    r"""沿sample轴同步切分N040 tokens与三类离散图。"""

    return PalmRotationGeometry(
        tokens=geometry.tokens[index],
        owner_valid=geometry.owner_valid[index],
        shortest_path=geometry.shortest_path[index],
        parent_direction=geometry.parent_direction[index],
        child_direction=geometry.child_direction[index],
    )


def _load_bank(
    handle: h5py.File,
    replica_ids: tuple[int, ...],
    *,
    device: torch.device,
) -> SupervisedBank:
    r"""读取完整replica trajectories，并仅展开`valid=True`的state-action pairs。"""

    rows = np.asarray(replica_ids, dtype=np.int64)
    valid = np.asarray(_h5_dataset(handle, "valid")[rows], dtype=np.bool_)
    if valid.ndim != 2 or not valid.any():
        raise RuntimeError("teacher dataset split contains no valid trajectory samples")
    replica_count, sampled_steps = valid.shape

    def dynamic(name: str) -> torch.Tensor:
        value = np.asarray(_h5_dataset(handle, name)[rows])
        if value.shape[:2] != valid.shape:
            raise RuntimeError(f"teacher dataset {name!r} does not share replica/sample axes")
        return torch.from_numpy(value[valid]).to(device=device)

    def static(name: str) -> torch.Tensor:
        value = np.asarray(_h5_dataset(handle, name)[rows])
        if value.shape[0] != replica_count:
            raise RuntimeError(f"teacher dataset {name!r} does not share the replica axis")
        expanded = np.broadcast_to(value[:, None], (replica_count, sampled_steps, *value.shape[1:]))
        return torch.from_numpy(expanded[valid].copy()).to(device=device)

    current = dynamic("student_jnt_current").float()
    target = dynamic("teacher_action_env_canonical").float()
    if current.shape != (int(valid.sum()), 16, 5) or target.shape != (int(valid.sum()), 16):
        raise RuntimeError("teacher dataset current/action shapes disagree with the 16-joint student ABI")
    sample_axis = np.asarray(_h5_dataset(handle, "sample_policy_step"), dtype=np.int64)
    if sample_axis.shape != (sampled_steps,) or np.any(np.diff(sample_axis) <= 0):
        raise RuntimeError("teacher dataset sample_policy_step must be a strictly increasing shared axis")
    replica_axis = np.broadcast_to(rows[:, None], valid.shape)[valid].copy()
    step_axis = np.broadcast_to(sample_axis[None, :], valid.shape)[valid].copy()
    observation = PalmRotationActorObservation(
        jnt_current=current,
        jnt_history=dynamic("student_jnt_history").float(),
        jnt_limits=static("student_jnt_limits_canonical").float(),
        owner_contact=dynamic("student_owner_contact").float(),
        jnt_valid=static("student_jnt_valid").bool(),
        tip_valid=static("student_tip_valid").bool(),
        owner_valid=static("student_owner_valid").bool(),
    )
    return SupervisedBank(
        observation=observation,
        target=target,
        previous_action=current[..., 2],
        replica_id=torch.from_numpy(replica_axis).to(device=device),
        policy_step=torch.from_numpy(step_axis).to(device=device),
    )


def _subset_bank(bank: SupervisedBank, index: torch.Tensor) -> SupervisedBank:
    r"""按固定sample index形成更小的显式tiny-overfit bank。"""

    return SupervisedBank(
        observation=_slice_observation(bank.observation, index),
        target=bank.target[index],
        previous_action=bank.previous_action[index],
        replica_id=bank.replica_id[index],
        policy_step=bank.policy_step[index],
    )


def _concatenate_banks(banks: tuple[SupervisedBank, ...]) -> SupervisedBank:
    r"""沿sample轴聚合teacher-on-policy与student-occupancy correction banks。

    每个source先各自按replica隔离train/validation，随后只在同一split内拼接。这样DAgger correction不会把
    validation replica的相邻状态泄漏到训练；source文件SHA与拼接顺序另由run identity冻结。

    Args:
        banks (tuple[SupervisedBank, ...]): 至少一个具有相同16-joint actor ABI的source bank。

    Returns:
        SupervisedBank: 样本轴为各source按CLI顺序串联的aggregate bank。
    """

    if not banks:
        raise ValueError("teacher mean aggregation requires at least one source bank")
    observations = tuple(bank.observation for bank in banks)
    observation = PalmRotationActorObservation(
        jnt_current=torch.cat([value.jnt_current for value in observations], dim=0),  # $[\sum_sN_s,16,5]$
        jnt_history=torch.cat([value.jnt_history for value in observations], dim=0),  # $[\sum_sN_s,30,16,5]$
        jnt_limits=torch.cat([value.jnt_limits for value in observations], dim=0),  # per-state canonical limits
        owner_contact=torch.cat([value.owner_contact for value in observations], dim=0),  # $[\sum_sN_s,21,1]$
        jnt_valid=torch.cat([value.jnt_valid for value in observations], dim=0),
        tip_valid=torch.cat([value.tip_valid for value in observations], dim=0),
        owner_valid=torch.cat([value.owner_valid for value in observations], dim=0),
    )
    return SupervisedBank(
        observation=observation,
        target=torch.cat([bank.target for bank in banks], dim=0),
        previous_action=torch.cat([bank.previous_action for bank in banks], dim=0),
        replica_id=torch.cat([bank.replica_id for bank in banks], dim=0),
        policy_step=torch.cat([bank.policy_step for bank in banks], dim=0),
    )


def _resolve_geometry(provider: Any, bank: SupervisedBank, *, chunk_size: int) -> PalmRotationGeometry:
    r"""固定数据集上预计算冻结N040；actor训练期间不会形成stale learned activation。"""

    chunks: dict[str, list[torch.Tensor]] = {
        "tokens": [],
        "owner_valid": [],
        "shortest_path": [],
        "parent_direction": [],
        "child_direction": [],
    }
    with torch.no_grad():
        for start in range(0, bank.sample_count, chunk_size):
            stop = min(start + chunk_size, bank.sample_count)
            observation = _slice_observation(bank.observation, slice(start, stop))
            prototype_index = torch.zeros(stop - start, dtype=torch.long, device=bank.target.device)
            geometry = provider.resolve(prototype_index, observation)
            chunks["tokens"].append(geometry.tokens)
            chunks["owner_valid"].append(geometry.owner_valid)
            chunks["shortest_path"].append(geometry.shortest_path)
            chunks["parent_direction"].append(geometry.parent_direction)
            chunks["child_direction"].append(geometry.child_direction)
    return PalmRotationGeometry(**{name: torch.cat(values, dim=0) for name, values in chunks.items()})


def _predict(
    actor: PalmRotationResidualActor,
    bank: SupervisedBank,
    geometry: PalmRotationGeometry,
    *,
    chunk_size: int,
) -> torch.Tensor:
    r"""以bounded chunks计算整个split的deterministic student mean。"""

    was_training = actor.training
    actor.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, bank.sample_count, chunk_size):
            stop = min(start + chunk_size, bank.sample_count)
            output = actor(
                _slice_observation(bank.observation, slice(start, stop)),
                _slice_geometry(geometry, slice(start, stop)),
            )
            predictions.append(output.mean)
    actor.train(was_training)
    return torch.cat(predictions, dim=0)


def _evaluate(
    actor: PalmRotationResidualActor,
    bank: SupervisedBank,
    geometry: PalmRotationGeometry,
    *,
    chunk_size: int,
) -> TeacherMeanMetrics:
    r"""计算总体、饱和、切换与previous-action-relative监督指标。"""

    prediction = _predict(actor, bank, geometry, chunk_size=chunk_size)
    return teacher_mean_metrics(prediction, bank.target, previous_action=bank.previous_action)


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    r"""原子发布resolved identity或最终报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    r"""逐evaluation cadence追加可审计训练指标。"""

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()


def _argument_parser() -> argparse.ArgumentParser:
    r"""声明N000 mean-imitation tiny-overfit的完整显式预算。"""

    parser = argparse.ArgumentParser(description="Fit the current palm-rotation actor mean to fixed teacher actions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--additional_dataset",
        type=Path,
        action="append",
        default=[],
        help="Additional replica-isolated teacher-label bank, e.g. a DAgger student-occupancy correction artifact.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--history_encoder", choices=("tcn", "raw_stack"), default="tcn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--full_batch", action="store_true", help="Use every fixed train sample exactly once per update.")
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--loss_reduction",
        choices=("element_mean", "balanced_action_regimes"),
        default="element_mean",
        help="Average all action elements or give saturated and transition regimes equal loss mass.",
    )
    parser.add_argument("--train_sample_limit", type=int, default=2048)
    parser.add_argument("--evaluation_interval", type=int, default=100)
    parser.add_argument("--geometry_chunk_size", type=int, default=512)
    parser.add_argument("--validation_modulus", type=int, default=4)
    parser.add_argument("--validation_remainder", type=int, default=3)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--train_mse_max", type=float, default=0.01)
    parser.add_argument("--train_sign_accuracy_min", type=float, default=0.99)
    parser.add_argument("--validation_sign_accuracy_min", type=float, default=0.90)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> dict[str, Any]:
    r"""启动最小Isaac runtime并确保任何USD/provider import发生在AppLauncher之后。"""

    args = _argument_parser().parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return _run(args)
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    r"""加载固定teacher bank、预计算N040、训练actor并发布独立监督证据。"""

    from anymani.distill.rl.runtime.palm_rotation_geometry import build_palm_rotation_bf16_geometry_provider
    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

    if args.output_dir is None:
        raise ValueError("teacher mean training requires an explicit --output_dir")
    if (
        args.max_updates < 1
        or args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.train_sample_limit < 0
        or args.evaluation_interval < 1
        or args.geometry_chunk_size < 1
    ):
        raise ValueError("teacher mean training budgets and learning rate must be positive")
    requested_datasets = (args.dataset, *tuple(args.additional_dataset))
    dataset_paths = tuple(
        (path if path.is_absolute() else ROOT / path).resolve(strict=True) for path in requested_datasets
    )
    if len(set(dataset_paths)) != len(dataset_paths):
        raise ValueError("teacher mean dataset list contains duplicate paths")
    output_dir = (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"teacher mean output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    precision = enforce_palm_rotation_precision(allow_tf32=bool(args.allow_tf32))
    device = torch.device(args.device or "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("teacher mean training requested CUDA but no CUDA device is available")

    expected_metadata = {
        "artifact_type": "anymani.n000_teacher_mean_bridge_trajectory",
        "source_asset_id": EXPECTED_SOURCE_ASSET_ID,
        "checkpoint_sha256": EXPECTED_TEACHER_CHECKPOINT_SHA256,
        "n040_attached": False,
    }
    source_metadata: list[dict[str, Any]] = []
    source_action_audits: list[dict[str, Any]] = []
    source_train_banks: list[SupervisedBank] = []
    source_validation_banks: list[SupervisedBank] = []
    train_replicas: tuple[int, ...] | None = None
    validation_replicas: tuple[int, ...] | None = None
    replica_count: int | None = None
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as handle:
            raw_metadata = handle.attrs.get("metadata_json")
            if not isinstance(raw_metadata, str):
                raise RuntimeError(f"teacher dataset lacks JSON metadata: {dataset_path}")
            metadata = json.loads(raw_metadata)
            if not isinstance(metadata, dict):
                raise RuntimeError(f"teacher dataset metadata must be a mapping: {dataset_path}")
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise RuntimeError(f"teacher dataset identity disagrees with the accepted mean-bridge contract: {dataset_path}")
            dataset_role = str(metadata.get("dataset_role", "legacy_teacher_on_policy_reference"))
            action_audit: dict[str, Any] = {"behavior_action_recorded": False}
            if dataset_role == "dagger_teacher_correction":
                if metadata.get("schema_version") != "1.2.0":
                    raise RuntimeError("DAgger correction datasets require schema 1.2 behavior-action evidence")
                valid_all = np.asarray(_h5_dataset(handle, "valid"), dtype=np.bool_)
                teacher_action = np.asarray(_h5_dataset(handle, "teacher_action_env_canonical"))[valid_all]
                behavior_action = np.asarray(_h5_dataset(handle, "behavior_action_env_canonical"))[valid_all]
                correction = teacher_action - behavior_action  # $a_T(s_{student})-a_S(s_{student})$
                action_audit = {
                    "behavior_action_recorded": True,
                    "valid_action_element_count": int(correction.size),
                    "teacher_behavior_mse": float(np.mean(np.square(correction))),
                    "teacher_behavior_mae": float(np.mean(np.abs(correction))),
                    "teacher_behavior_exact_equal_fraction": float(np.mean(correction == 0.0)),
                }
                if correction.size == 0 or bool(np.all(correction == 0.0)):
                    raise RuntimeError("DAgger correction artifact does not distinguish teacher labels from behavior actions")
            current_replica_count = int(metadata.get("replica_count", -1))
            if _h5_dataset(handle, "valid").shape[0] != current_replica_count:
                raise RuntimeError(f"teacher dataset metadata and valid replica axes disagree: {dataset_path}")
            if replica_count is None:
                replica_count = current_replica_count
                train_replicas, validation_replicas = replica_modulo_split(
                    replica_count,
                    validation_modulus=int(args.validation_modulus),
                    validation_remainder=int(args.validation_remainder),
                )
            elif current_replica_count != replica_count:
                raise RuntimeError("all aggregate teacher datasets must share the replica axis")
            if train_replicas is None or validation_replicas is None:
                raise AssertionError("replica-isolated split was not initialized")
            source_metadata.append(metadata)
            source_action_audits.append(action_audit)
            source_train_banks.append(_load_bank(handle, train_replicas, device=device))
            source_validation_banks.append(_load_bank(handle, validation_replicas, device=device))
    train_bank = _concatenate_banks(tuple(source_train_banks))
    validation_bank = _concatenate_banks(tuple(source_validation_banks))
    if train_replicas is None or validation_replicas is None:
        raise AssertionError("teacher dataset loop produced no replica split")

    # A fixed subsample is enough to test representation/interface overfit; its exact pair digest enters run identity.
    if args.train_sample_limit and args.train_sample_limit < train_bank.sample_count:
        cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        selected_cpu = torch.randperm(train_bank.sample_count, generator=cpu_generator)[: args.train_sample_limit]
        train_bank = _subset_bank(train_bank, selected_cpu.to(device=device))
    if args.full_batch and int(args.batch_size) != train_bank.sample_count:
        raise ValueError("--full_batch requires --batch_size equal to the resolved fixed train sample count")
    train_pair_digest = _stable_digest(
        {
            "replica_id": train_bank.replica_id.detach().cpu().tolist(),
            "policy_step": train_bank.policy_step.detach().cpu().tolist(),
        }
    )

    binding = build_generated_asset_binding((FORMAL_GEOMETRY_ROW,))
    if (
        len(binding.source_assets) != 1
        or binding.source_assets[0].asset_id != EXPECTED_SOURCE_ASSET_ID
        or len(binding.canonical_artifacts) != 1
        or binding.canonical_artifacts[0].routing.asset_row != 0
    ):
        raise RuntimeError("formal row848 did not materialize the expected selection-local row0 asset")
    provider = build_palm_rotation_bf16_geometry_provider(binding, device=device).eval()
    expected_physical_hashes = {
        str(metadata.get("audited_distill_physical_geometry_hash", "")) for metadata in source_metadata
    }
    if expected_physical_hashes != {str(provider.identity["physical_geometry_hashes"][0])}:
        raise RuntimeError("selection-local N040 physical identity disagrees with the audited teacher source")

    geometry_start = time.perf_counter()
    train_geometry = _resolve_geometry(provider, train_bank, chunk_size=args.geometry_chunk_size)
    validation_geometry = _resolve_geometry(provider, validation_bank, chunk_size=args.geometry_chunk_size)
    geometry_seconds = time.perf_counter() - geometry_start
    if not bool(torch.isfinite(train_geometry.tokens).all().item() and torch.isfinite(validation_geometry.tokens).all().item()):
        raise RuntimeError("selection-local N040 produced non-finite teacher-bank tokens")

    actor = PalmRotationResidualActor(history_encoder=args.history_encoder).to(device).train()
    actor.global_log_std.requires_grad_(False)  # mean imitation没有teacher-compatible variance target
    optimizer = torch.optim.Adam(
        (parameter for parameter in actor.parameters() if parameter.requires_grad),
        lr=float(args.learning_rate),
    )
    initial_train = _evaluate(actor, train_bank, train_geometry, chunk_size=args.geometry_chunk_size)
    initial_validation = _evaluate(actor, validation_bank, validation_geometry, chunk_size=args.geometry_chunk_size)

    implementation_files = (
        Path(__file__).resolve(),
        ROOT / "source/anymani/anymani/distill/il/n000_teacher_mean.py",
        ROOT / "source/anymani/anymani/distill/models/palm_rotation_policy.py",
        ROOT / "source/anymani/anymani/distill/rl/runtime/palm_rotation_geometry.py",
    )
    identity_payload: dict[str, Any] = {
        "schema_version": "1.2.0",
        "objective": "environment-action deterministic teacher mean MSE",
        "loss_reduction": str(args.loss_reduction),
        "datasets": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "metadata_digest": _stable_digest(metadata),
                "dataset_role": str(metadata.get("dataset_role", "legacy_teacher_on_policy_reference")),
                "collection_behavior_policy": str(metadata.get("collection_behavior_policy", "accepted_teacher")),
                "train_sample_count_before_aggregate_subsample": source_train_banks[index].sample_count,
                "validation_sample_count": source_validation_banks[index].sample_count,
                "action_audit": source_action_audits[index],
            }
            for index, (path, metadata) in enumerate(zip(dataset_paths, source_metadata, strict=True))
        ],
        "aggregate_dataset": {
            "train_replicas": list(train_replicas),
            "validation_replicas": list(validation_replicas),
            "train_sample_count": train_bank.sample_count,
            "validation_sample_count": validation_bank.sample_count,
            "train_pair_digest": train_pair_digest,
        },
        "geometry_provider": provider.identity,
        "formal_geometry_row": FORMAL_GEOMETRY_ROW,
        "history_encoder": args.history_encoder,
        "precision": precision,
        "seed": int(args.seed),
        "max_updates": int(args.max_updates),
        "batch_size": int(args.batch_size),
        "sampling_mode": "full_batch_without_replacement" if args.full_batch else "random_with_replacement",
        "learning_rate": float(args.learning_rate),
        "acceptance": {
            "train_mse_max": float(args.train_mse_max),
            "train_sign_accuracy_min": float(args.train_sign_accuracy_min),
            "validation_sign_accuracy_min": float(args.validation_sign_accuracy_min),
            "validation_skill_vs_previous_action_min_exclusive": 0.0,
            "validation_transition_skill_vs_previous_action_min_exclusive": 0.0,
            "validation_transition_skill_vs_zero_min_exclusive": 0.0,
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation_files
        },
    }
    identity = {**identity_payload, "identity_digest": _stable_digest(identity_payload)}
    _atomic_json(output_dir / "identity.json", identity)

    def record(update: int, elapsed_seconds: float) -> tuple[TeacherMeanMetrics, TeacherMeanMetrics]:
        r"""在固定train/validation banks上评估并追加一行。"""

        train_metrics = _evaluate(actor, train_bank, train_geometry, chunk_size=args.geometry_chunk_size)
        validation_metrics = _evaluate(
            actor,
            validation_bank,
            validation_geometry,
            chunk_size=args.geometry_chunk_size,
        )
        _append_jsonl(
            metrics_path,
            {
                "schema_version": "1.2.0",
                "identity_digest": identity["identity_digest"],
                "update": update,
                "sample_uses": update * int(args.batch_size),
                "elapsed_seconds": elapsed_seconds,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            },
        )
        return train_metrics, validation_metrics

    _append_jsonl(
        metrics_path,
        {
            "schema_version": "1.2.0",
            "identity_digest": identity["identity_digest"],
            "update": 0,
            "sample_uses": 0,
            "elapsed_seconds": 0.0,
            "train": asdict(initial_train),
            "validation": asdict(initial_validation),
        },
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_start = time.perf_counter()
    final_train, final_validation = initial_train, initial_validation
    for update in range(1, int(args.max_updates) + 1):
        index = training_batch_indices(
            sample_count=train_bank.sample_count,
            batch_size=int(args.batch_size),
            full_batch=bool(args.full_batch),
            generator=generator,
            device=device,
        )
        prediction = actor(
            _slice_observation(train_bank.observation, index),
            _slice_geometry(train_geometry, index),
        ).mean
        loss = teacher_mean_loss(
            prediction,
            train_bank.target[index],
            reduction=args.loss_reduction,
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"teacher mean loss became non-finite at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"teacher mean gradient became non-finite at update {update}")
        optimizer.step()
        if update % int(args.evaluation_interval) == 0 or update == int(args.max_updates):
            final_train, final_validation = record(update, time.perf_counter() - train_start)

    train_passed = bool(
        final_train.mse <= float(args.train_mse_max)
        and final_train.sign_accuracy >= float(args.train_sign_accuracy_min)
    )
    validation_passed = bool(
        final_validation.sign_accuracy >= float(args.validation_sign_accuracy_min)
        and (final_validation.skill_vs_previous_action or float("-inf")) > 0.0
        and (final_validation.transition_skill_vs_previous_action or float("-inf")) > 0.0
        and final_validation.transition_skill_vs_zero > 0.0
    )
    passed = train_passed and validation_passed

    # Aggregate指标之外保留每个occupancy source的held-out表现，防止较大的bank掩盖另一source回归。
    validation_by_source: dict[str, dict[str, Any]] = {}
    validation_offset = 0
    for source_index, source_bank in enumerate(source_validation_banks):
        source_stop = validation_offset + source_bank.sample_count
        source_metrics = _evaluate(
            actor,
            source_bank,
            _slice_geometry(validation_geometry, slice(validation_offset, source_stop)),
            chunk_size=args.geometry_chunk_size,
        )
        source_role = str(source_metadata[source_index].get("dataset_role", "legacy_teacher_on_policy_reference"))
        validation_by_source[f"source_{source_index:02d}_{source_role}"] = asdict(source_metrics)
        validation_offset = source_stop
    if validation_offset != validation_bank.sample_count:
        raise AssertionError("per-source validation slices did not cover the aggregate bank")

    checkpoint = {
        "artifact_type": "anymani.n000_teacher_mean_student",
        "schema_version": "1.2.0",
        "identity": identity,
        "update": int(args.max_updates),
        "actor": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_metrics": asdict(final_train),
        "validation_metrics": asdict(final_validation),
        "validation_metrics_by_source": validation_by_source,
        "train_passed": train_passed,
        "validation_passed": validation_passed,
        "passed": passed,
    }
    checkpoint_path = output_dir / "student-final.pt"
    checkpoint_temporary = output_dir / "student-final.pt.tmp"
    torch.save(checkpoint, checkpoint_temporary)
    checkpoint_temporary.replace(checkpoint_path)
    report = {
        "artifact_type": "anymani.n000_teacher_mean_tiny_overfit_report",
        "schema_version": "1.2.0",
        "identity_digest": identity["identity_digest"],
        "passed": passed,
        "train_passed": train_passed,
        "validation_passed": validation_passed,
        "initial_train": asdict(initial_train),
        "initial_validation": asdict(initial_validation),
        "final_train": asdict(final_train),
        "final_validation": asdict(final_validation),
        "final_validation_by_source": validation_by_source,
        "geometry_precompute_seconds": geometry_seconds,
        "training_seconds": time.perf_counter() - train_start,
        "actor_parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metrics": str(metrics_path),
        "known_boundary": "offline mean imitation only; no closed-loop student capability claim",
    }
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
