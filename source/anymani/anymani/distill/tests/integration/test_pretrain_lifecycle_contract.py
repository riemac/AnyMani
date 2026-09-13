r"""Geometry SSL schema-9 streaming train、epoch recovery、retained export 与 explicit evaluation 闭环。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from anymani.distill.methods.contracts import MethodEvaluationReport, MethodParameterGroup, MethodUpdate
from anymani.distill.ssl.post_training import EvaluationCfg, EvaluationRun, EvaluationRunCfg
from anymani.distill.ssl.runtime.lifecycle import fit_embodiment_pretrain
from anymani.distill.ssl.runtime.post_training import evaluate_checkpoint
from anymani.distill.ssl.runtime.pretrainer import AdamWCfg, EmbodimentPretrainTrainer, EmbodimentPretrainTrainerCfg
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

pytestmark = [pytest.mark.contract, pytest.mark.skipif(not torch.cuda.is_available(), reason="fit requires CUDA")]

_TERMS = ("density", "kappa")


class _Dataset:
    r"""Synthetic manifest identity；角色解析由 _Data 显式分离。"""

    source_sha256 = "synthetic-dataset-sha"
    source_path = Path("synthetic-ssl.yaml")

    @staticmethod
    def config_dict() -> dict[str, object]:
        return {"schema_version": "synthetic", "validation": None}


class _Catalog:
    r"""只暴露当前 role 的最小 catalog surface。"""

    def __init__(self, role: str) -> None:
        self.dataset = _Dataset()
        self.train = ("asset-a", "asset-b") if role == "train" else ()
        self.evaluation = {"unseen": ("asset-c",), "official_zero_shot": ()} if role == "evaluation" else {}

    @staticmethod
    def training_dataset_identity() -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "source_sha256": "synthetic-dataset-sha",
            "train_asset_count": 2,
            "train_asset_axis_sha256": "synthetic-axis",
        }


class _Data:
    r"""记录 train/evaluation resolver 是否发生越权调用。"""

    def __init__(self) -> None:
        self.resolved_roles: list[str] = []

    def resolve_train(self) -> _Catalog:
        self.resolved_roles.append("train")
        return _Catalog("train")

    def resolve_evaluation(self) -> _Catalog:
        self.resolved_roles.append("evaluation")
        return _Catalog("evaluation")


class _Unit:
    r"""Synthetic 2-pair stream unit；只提供 Trainer/logger 读取的 typed axes。"""

    def __init__(self, sample_count: int, device: torch.device) -> None:
        self.sample_count = sample_count
        self.q = torch.zeros(sample_count, 1, device=device)
        self.asset_ids = tuple(f"asset-{index}" for index in range(sample_count))
        self.q_index = torch.arange(sample_count, device=device)


class _Session:
    r"""Synthetic split session，train 通过 realize_units，evaluation 通过 realize。"""

    def __init__(self, role: str, suite: str, asset_count: int, device: torch.device, trace: list[object]) -> None:
        self.role = role
        self.suite = suite
        self.asset_count = asset_count
        self.device = device
        self.cursor = 0
        self.trace = trace

    def realize_units(self, item: Any, *, schedule: Any, step: int):
        del schedule, step
        self.trace.append(item)
        self.cursor += item.sample_count
        yield _Unit(item.sample_count, self.device)

    def realize(self, item: Any, *, schedule: Any, step: int) -> _Unit:
        del schedule, step
        self.cursor += item.sample_count
        return _Unit(item.sample_count, self.device)

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor, "role": self.role, "suite": self.suite}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.cursor = int(state["cursor"])

    def drain_runtime_events(self) -> tuple[dict[str, object], ...]:
        return ()

    def close(self) -> None:
        return None


class _Method:
    r"""只实现 schema-9 Trainer/explicit-evaluation 窄合同。"""

    def __init__(self) -> None:
        self.parameter: torch.nn.Parameter | None = None
        self.forward_steps: list[int] = []
        self.training_trace: list[object] = []
        self.opened_roles: list[str] = []
        self.prepared_roles: list[str] = []
        self.retained_export_count = 0
        self.z_gradient_requests: list[bool] = []
        self.fail_after_updates: int | None = None

    def prepare(self, catalog: _Catalog, *, role: str, device: torch.device, dtype: torch.dtype) -> None:
        del catalog, device, dtype
        self.prepared_roles.append(role)

    def configure_execution(self, policy: Any) -> None:
        self.execution = policy

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.density_private = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        self.kappa_private = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def _parameter(self) -> torch.nn.Parameter:
        assert self.parameter is not None
        return self.parameter

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def split_names(self, role: str) -> tuple[str, ...]:
        assert role == "evaluation"
        return ("unseen", "official_zero_shot")

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        assert role == "evaluation"
        return 0 if suite == "official_zero_shot" else 1

    def open_session(self, role: str, *, suite: str = "", device: torch.device, **_kwargs: Any) -> _Session:
        self.opened_roles.append(role)
        count = 2 if role == "train" else self.split_asset_count(role, suite=suite)
        return _Session(role, suite, count, device, self.training_trace)

    def asset_manifest(self, catalog: _Catalog) -> dict[str, object]:
        del catalog
        return {"schema_version": "synthetic", "train": [], "evaluation": {"unseen": ["asset-c"]}}

    def source_artifact_identity(self) -> dict[str, object]:
        return {}

    def declared_objective_weights(self) -> dict[str, float]:
        return {name: 1.0 for name in _TERMS}

    def formula_identity(self) -> dict[str, str]:
        return {name: f"synthetic.{name}" for name in _TERMS}

    def optimization_identity(self) -> dict[str, object]:
        return {"algorithm": "synthetic-fairgrad", "near_opposition_tolerance": 1.0e-6}

    def optimizer_parameter_groups(self) -> tuple[MethodParameterGroup, ...]:
        return (
            MethodParameterGroup("shared_encoder", (self._parameter(),)),
            MethodParameterGroup("density_reader", (self.density_private,)),
            MethodParameterGroup("kappa_reader", (self.kappa_private,)),
        )

    def teacher_baseline_statistics(self, batch: _Unit) -> dict[str, torch.Tensor]:
        count = torch.tensor(float(batch.sample_count), device=self._parameter().device)
        return {"density_sum": count, "kappa_sum": count, "count": count}

    def merge_teacher_baseline_statistics(self, total: Any, block: dict[str, torch.Tensor]):
        if total is None:
            return {name: value.clone() for name, value in block.items()}
        return {name: total[name] + block[name] for name in total}

    def finalize_teacher_baselines(self, statistics: Any) -> dict[str, object]:
        del statistics
        return {
            "density": {"predictor": "constant", "baseline_mse": 1.0},
            "kappa": {"predictor": "zero", "baseline_mse": 1.0},
        }

    def backward_update_units(
        self,
        units: Any,
        *,
        forward_step: int,
        logical_sample_count: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:
        del microbatch_size
        if self.fail_after_updates is not None and len(self.forward_steps) >= self.fail_after_updates:
            raise RuntimeError("synthetic interruption after completed epoch boundary")
        observed = sum(unit.sample_count for unit in units)
        assert observed == logical_sample_count
        self.forward_steps.append(forward_step)
        self.z_gradient_requests.append(collect_z_gradients)
        value = float(self._parameter().detach().square())
        self._parameter().grad = torch.full_like(self._parameter(), 2.0**0.5)
        self.density_private.grad = torch.zeros_like(self.density_private)
        self.kappa_private.grad = torch.zeros_like(self.kappa_private)
        evidence = {"fairgrad/cosine": 0.0} if collect_z_gradients else {}
        return MethodUpdate(
            terms={name: value for name in _TERMS},
            sample_count=observed,
            denominators={name: float(observed) for name in _TERMS},
            gradient_evidence=evidence,
        )

    def evaluate_session(self, session: _Session, schedule: Any, *, include_ablations: bool = False):
        while not schedule.complete:
            session.realize(schedule.next(), schedule=schedule, step=0)
        value = float(self._parameter().detach().square())
        metrics = {"density": value, "kappa": value, "derived_field": value}
        ablations = None
        if include_ablations:
            names = ("full", "query_only", "same_asset_q_shuffle", "cross_asset_shuffle", "joint_token_shuffle")
            ablations = {
                "pairing_key": ["asset_id", "q_index"],
                "ablations": names,
                "records": [{"asset_id": session.suite, "q_index": 0, "metrics": {name: metrics for name in names}}],
            }
        return MethodEvaluationReport(
            metrics=metrics,
            strata={"metric_scores": metrics, "bank_digest_sha256": f"fixed-{session.suite}"},
            teacher_baselines={"density": {"baseline_mse": 1.0}, "kappa": {"physical_baseline_mse": 1.0}},
            ablations=ablations,
        )

    def analyze_ablations(self, evidence: Any, *, bootstrap_replicates: int, seed: int) -> dict[str, object]:
        return {"record_count": len(evidence["records"]), "bootstrap_replicates": bootstrap_replicates, "seed": seed}

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "parameter": self._parameter().detach().clone(),
            "density_private": self.density_private.detach().clone(),
            "kappa_private": self.kappa_private.detach().clone(),
        }

    def load_training_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._parameter().data.copy_(state["parameter"])
        self.density_private.data.copy_(state["density_private"])
        self.kappa_private.data.copy_(state["kappa_private"])

    def retained_artifact_payload(self, *, metadata: dict[str, object], source_checkpoint: Path) -> dict[str, object]:
        self.retained_export_count += 1
        return {
            "schema_version": "synthetic",
            "artifact_type": "synthetic_retained",
            "retained_state": {"parameter": self._parameter().detach().cpu()},
            "lineage": {"source_checkpoint": str(source_checkpoint), "code_revision": metadata["code_revision"]},
        }

    def fit_z_compression_basis(self, session: _Session, schedule: Any):
        del session, schedule
        from anymani.distill.diagnostics.evaluation.z_compression import UnifiedPCABasis

        return UnifiedPCABasis(
            mean=torch.zeros(4, dtype=torch.float64),
            components=torch.eye(4, dtype=torch.float64),
            eigenvalues=torch.ones(4, dtype=torch.float64),
            sample_count=2,
        )

    def close(self) -> None:
        return None


def _trainer() -> EmbodimentPretrainTrainer:
    config = EmbodimentPretrainTrainerCfg(
        sampling=OnlineSamplingCfg(assets_per_minibatch=2, q_per_asset_per_minibatch=1, shuffle_assets=False, seed=17),
        max_epochs=4,
        num_minibatches=1,
        mini_epochs=1,
        microbatch_size=2,
        optimizer=AdamWCfg(learning_rate=0.1, weight_decay=0.0),
        checkpoint_every_epochs=2,
    )
    return EmbodimentPretrainTrainer(config)


def _resolved(trainer: EmbodimentPretrainTrainer) -> dict[str, object]:
    method = {
        "state_measure": {"kind": "synthetic"},
        "representation": {"kind": "synthetic"},
        "model": {"kind": "synthetic"},
        "objectives": {name: {} for name in _TERMS},
        "fairgrad": {"algorithm": "synthetic-fairgrad", "near_opposition_tolerance": 1.0e-6},
        "entity_permutation": {"enabled": True, "seed_offset": 31_337},
        "joint_sign_rewrite": {"probability": 0.2},
    }
    return {
        "schema_version": "9.0.0",
        "data": {"manifest": "synthetic-ssl.yaml"},
        "method": method,
        "trainer": asdict(trainer.config),
        "run": {"seed": 17, "source_cache_root": "", "source_cache_mode": "off", "new_run": False},
    }


def _evaluation_resolved(config: EvaluationCfg, run: EvaluationRunCfg) -> dict[str, object]:
    train = _resolved(_trainer())
    return {
        "schema_version": "1.0.0",
        "data": train["data"],
        "method": train["method"],
        "evaluation": asdict(config),
        "run": asdict(run),
    }


def test_train_exports_final_state_and_evaluation_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""训练只解析 train 并交付 last/retained；evaluation 只解析 held-out 且不产生 best。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    pretrain_dir = tmp_path / "pretrain"
    trainer = _trainer()
    data = _Data()
    method = _Method()
    fit_embodiment_pretrain(
        trainer=trainer,
        data=data,
        method=method,
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=pretrain_dir,
        resolved_config=_resolved(trainer),
    )

    assert data.resolved_roles == ["train"]
    assert method.prepared_roles == ["train"]
    assert method.opened_roles == ["train"]
    assert method.forward_steps == [0, 1, 2, 3]
    assert method.z_gradient_requests == [False, False, False, True]
    assert method.retained_export_count == 1
    assert (pretrain_dir / "checkpoints" / "epoch_000000.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000002.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000004.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "last.pt").is_file()
    assert (pretrain_dir / "retained_encoder.pt").is_file()
    assert not (pretrain_dir / "checkpoints" / "recovery.pt").exists()
    assert not (pretrain_dir / "INCOMPLETE").exists()
    assert (pretrain_dir / "COMPLETE").is_file()
    assert not (pretrain_dir / "checkpoints" / "best.pt").exists()
    final_payload = torch.load(pretrain_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    offsets = final_payload["trainer_state"]["log_continuation_offsets"]
    assert offsets["metrics_jsonl_bytes"] == (pretrain_dir / "metrics.jsonl").stat().st_size
    assert offsets["runtime_jsonl_bytes"] == (pretrain_dir / "runtime.jsonl").stat().st_size

    evaluation_cfg = EvaluationCfg(
        q_per_asset=1,
        assets_per_minibatch=1,
        q_per_asset_per_minibatch=1,
        bootstrap_replicates=2,
        max_resident_assets=1,
        source_cache_mode="off",
    )
    evaluation_run_cfg = EvaluationRunCfg(
        checkpoint=str(pretrain_dir / "checkpoints" / "last.pt"),
        analyses=("ablations",),
        seed=17,
        deterministic_algorithms=False,
    )
    evaluation_data = _Data()
    evaluation_method = _Method()
    evaluation_dir = tmp_path / "evaluation"
    evaluate_checkpoint(
        data=evaluation_data,
        method=evaluation_method,
        config=evaluation_cfg,
        run=EvaluationRun(evaluation_run_cfg),
        output_dir_override=evaluation_dir,
        resolved_config=_evaluation_resolved(evaluation_cfg, evaluation_run_cfg),
    )

    final = yaml.safe_load((evaluation_dir / "evaluation.yaml").read_text(encoding="utf-8"))["suites"]
    assert evaluation_data.resolved_roles == ["evaluation"]
    assert evaluation_method.prepared_roles == ["evaluation"]
    assert evaluation_method.opened_roles == ["evaluation"]
    assert final["unseen"]["metrics"]
    assert final["unseen"]["teacher_baselines"]["kappa"]["physical_baseline_mse"] == 1.0
    assert final["unseen"]["ablation_analysis"]["bootstrap_replicates"] == 2
    assert final["official_zero_shot"] == {"status": "empty", "asset_count": 0}
    assert not (evaluation_dir / "checkpoints" / "best.pt").exists()


def test_epoch_boundary_recovery_replays_only_the_incomplete_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""中断发生在下一 epoch 时，自动 recovery 只重做该 epoch 且继续原 forward cursor。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    base = _trainer().config
    config = replace(base, max_epochs=2, checkpoint_every_epochs=1)
    interrupted = _Method()
    interrupted.fail_after_updates = 1
    output_dir = tmp_path / "recovery"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        fit_embodiment_pretrain(
            trainer=EmbodimentPretrainTrainer(config),
            data=_Data(),
            method=interrupted,
            run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
            output_dir_override=output_dir,
            resolved_config=_resolved(EmbodimentPretrainTrainer(config)),
        )
    assert (output_dir / "checkpoints" / "recovery.pt").is_file()
    assert (output_dir / "INCOMPLETE").is_file()

    resumed = _Method()
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(config),
        data=_Data(),
        method=resumed,
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=output_dir,
        resolved_config=_resolved(EmbodimentPretrainTrainer(config)),
    )
    assert resumed.forward_steps == [1]
    assert (output_dir / "COMPLETE").is_file()
    assert not (output_dir / "INCOMPLETE").exists()
    assert not (output_dir / "checkpoints" / "recovery.pt").exists()


def test_completed_run_extension_matches_uninterrupted_training_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""`2 epochs + explicit extension to 4` 必须与从头声明 4 epochs 的完整轨迹逐值一致。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    base = _trainer().config
    prefix_config = replace(base, max_epochs=2, checkpoint_every_epochs=1)
    full_config = replace(base, max_epochs=4, checkpoint_every_epochs=1)

    full_dir = tmp_path / "full"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(full_config),
        data=_Data(),
        method=_Method(),
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=full_dir,
        resolved_config=_resolved(EmbodimentPretrainTrainer(full_config)),
    )

    prefix_dir = tmp_path / "prefix"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(prefix_config),
        data=_Data(),
        method=_Method(),
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=prefix_dir,
        resolved_config=_resolved(EmbodimentPretrainTrainer(prefix_config)),
    )
    prefix_checkpoint = prefix_dir / "checkpoints" / "last.pt"
    prefix_bytes = prefix_checkpoint.read_bytes()

    extension_dir = tmp_path / "extension"
    extension_method = _Method()
    extension_resolved = _resolved(EmbodimentPretrainTrainer(full_config))
    extension_resolved["run"] = dict(extension_resolved["run"], extend_completed_run=True)
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(full_config),
        data=_Data(),
        method=extension_method,
        run=PretrainRun(
            PretrainRunCfg(
                seed=17,
                deterministic_algorithms=False,
                source_cache_mode="off",
                resume_checkpoint=str(prefix_checkpoint),
                extend_completed_run=True,
            )
        ),
        output_dir_override=extension_dir,
        resolved_config=extension_resolved,
    )

    full = torch.load(full_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    extended = torch.load(extension_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    for name, value in full["method_state"].items():
        torch.testing.assert_close(extended["method_state"][name], value, atol=0.0, rtol=0.0)
    assert extended["optimizer_state"] == full["optimizer_state"]
    assert extended["trainer_state"]["sampling"] == full["trainer_state"]["sampling"]
    assert extension_method.forward_steps == [2, 3]
    assert prefix_checkpoint.read_bytes() == prefix_bytes
    assert not (extension_dir / "checkpoints" / "epoch_000000.pt").exists()
    assert (extension_dir / "checkpoints" / "epoch_000004.pt").is_file()
    def scientific_records(path: Path) -> list[dict[str, object]]:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            record.pop("wall_time_seconds", None)
            record.pop("z_gradient_diagnostic_seconds", None)
        return records

    assert scientific_records(extension_dir / "metrics_finalized.jsonl") == scientific_records(
        full_dir / "metrics_finalized.jsonl"
    )


def test_completed_extension_recovery_preserves_prefix_and_finishes_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Extension child 中断后恢复自身 segment，并继续引用 immutable source metrics prefix。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    base = _trainer().config
    prefix_config = replace(base, max_epochs=2, checkpoint_every_epochs=1)
    target_config = replace(base, max_epochs=4, checkpoint_every_epochs=1)
    prefix_dir = tmp_path / "prefix"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(prefix_config),
        data=_Data(),
        method=_Method(),
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=prefix_dir,
        resolved_config=_resolved(EmbodimentPretrainTrainer(prefix_config)),
    )
    prefix_checkpoint = prefix_dir / "checkpoints" / "last.pt"
    prefix_bytes = prefix_checkpoint.read_bytes()
    extension_resolved = _resolved(EmbodimentPretrainTrainer(target_config))
    extension_resolved["run"] = dict(extension_resolved["run"], extend_completed_run=True)
    extension_dir = tmp_path / "extension"
    interrupted = _Method()
    interrupted.fail_after_updates = 1
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        fit_embodiment_pretrain(
            trainer=EmbodimentPretrainTrainer(target_config),
            data=_Data(),
            method=interrupted,
            run=PretrainRun(
                PretrainRunCfg(
                    seed=17,
                    deterministic_algorithms=False,
                    source_cache_mode="off",
                    resume_checkpoint=str(prefix_checkpoint),
                    extend_completed_run=True,
                )
            ),
            output_dir_override=extension_dir,
            resolved_config=extension_resolved,
        )
    recovery = extension_dir / "checkpoints" / "recovery.pt"
    recovery_payload = torch.load(recovery, map_location="cpu", weights_only=False)
    assert recovery_payload["trainer_state"]["lineage_metrics_path"] == str(prefix_dir / "metrics.jsonl")

    resumed = _Method()
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(target_config),
        data=_Data(),
        method=resumed,
        run=PretrainRun(
            PretrainRunCfg(
                seed=17,
                deterministic_algorithms=False,
                source_cache_mode="off",
                resume_checkpoint=str(recovery),
                extend_completed_run=True,
            )
        ),
        output_dir_override=extension_dir,
        resolved_config=extension_resolved,
    )
    assert resumed.forward_steps == [3]
    assert prefix_checkpoint.read_bytes() == prefix_bytes
    assert (extension_dir / "COMPLETE").is_file()
    assert not recovery.exists()


def test_formal_train_can_publish_a_train_derived_compression_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""train 完成后可发布固定 q-bank basis，且 basis 不需要 evaluation role。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    trainer_config = replace(_trainer().config, emit_compression_basis=True, compression_q_per_asset=4)
    output_dir = tmp_path / "basis"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(trainer_config),
        data=_Data(),
        method=_Method(),
        run=PretrainRun(PretrainRunCfg(seed=17, deterministic_algorithms=False, source_cache_mode="off")),
        output_dir_override=output_dir,
        resolved_config=_resolved(EmbodimentPretrainTrainer(trainer_config)),
    )
    assert (output_dir / "z_compression_basis.npz").is_file()
    basis_metadata = yaml.safe_load((output_dir / "z_compression_basis.yaml").read_text(encoding="utf-8"))
    assert basis_metadata["source"] == "train_role_fixed_q_bank"
    assert basis_metadata["q_per_asset"] == 4
    assert len(basis_metadata["basis_sha256"]) == 64
