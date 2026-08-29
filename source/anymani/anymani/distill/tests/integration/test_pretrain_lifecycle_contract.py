r"""Geometry SSL schema-8 pure pretrain、run-local baseline、validation 与 evaluation 闭环。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from anymani.distill.methods.contracts import (
    MethodEvaluationReport,
    MethodParameterGroup,
    MethodStep,
    MethodUpdate,
)
from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult
from anymani.distill.ssl.post_training import (
    EvaluationCfg,
    EvaluationRun,
    EvaluationRunCfg,
    ValidationCfg,
    ValidationRun,
    ValidationRunCfg,
)
from anymani.distill.ssl.runtime.lifecycle import fit_embodiment_pretrain
from anymani.distill.ssl.runtime.post_training import evaluate_checkpoint, validate_checkpoints
from anymani.distill.ssl.runtime.pretrainer import (
    AdamWCfg,
    EmbodimentPretrainTrainer,
    EmbodimentPretrainTrainerCfg,
)
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

pytestmark = [pytest.mark.contract, pytest.mark.skipif(not torch.cuda.is_available(), reason="fit requires CUDA")]

_TERMS = ("density", "kappa")
_METRICS = ("density", "kappa", "derived_field")  # derived-field 只属于显式事后评估


class _Dataset:
    source_sha256 = "synthetic-dataset-sha"
    source_path = Path("synthetic-ssl.yaml")

    @staticmethod
    def config_dict() -> dict[str, object]:
        return {"schema_version": "synthetic"}


class _Data:
    def resolve(self):
        return SimpleNamespace(
            dataset=_Dataset(),
            training_dataset_identity=lambda: {
                "schema_version": "1.0.0",
                "source_sha256": "synthetic-dataset-sha",
                "train_asset_count": 2,
                "train_asset_axis_sha256": "synthetic-axis",
            },
        )


class _Session:
    def __init__(self, role: str, suite: str, asset_count: int, device: torch.device, trace: list[object]) -> None:
        self.role = role
        self.suite = suite
        self.asset_count = asset_count
        self.device = device
        self.cursor = 0
        self.trace = trace

    def realize(self, item, *, schedule, step: int):
        del schedule, step
        if self.role == "train":
            self.trace.append(item)
        self.cursor += item.sample_count
        return SimpleNamespace(sample_count=item.sample_count)

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor, "role": self.role, "suite": self.suite}

    def load_state_dict(self, state) -> None:
        self.cursor = int(state["cursor"])

    def close(self) -> None:
        return None


class _Method:
    r"""只实现 Trainer 窄合同；内部单参数让 validation 能确定 best checkpoint。"""

    def __init__(self) -> None:
        self.parameter: torch.nn.Parameter | None = None
        self.forward_steps: list[int] = []
        self.training_trace: list[object] = []
        self.opened_roles: list[str] = []
        self.retained_export_count = 0
        self.z_gradient_requests: list[bool] = []

    def prepare(self, catalog, *, device: torch.device, dtype: torch.dtype) -> None:
        del catalog, device, dtype

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.density_private = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        self.kappa_private = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def _parameter(self) -> torch.nn.Parameter:
        assert self.parameter is not None
        return self.parameter

    def parameters(self):
        return (self._parameter(), self.density_private, self.kappa_private)

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def split_names(self, role: str) -> tuple[str, ...]:
        return {
            "train": ("",),
            "training_evaluation": ("",),
            "validation": ("validation",),
            "evaluation": ("unseen", "official_zero_shot"),
        }[role]

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        if role == "evaluation" and suite == "official_zero_shot":
            return 0
        return 2 if role in {"train", "training_evaluation"} else 1

    def open_session(self, role: str, *, suite: str = "", device: torch.device, **_kwargs):
        self.opened_roles.append(role)
        return _Session(role, suite, self.split_asset_count(role, suite=suite), device, self.training_trace)

    def asset_manifest(self, catalog) -> dict[str, object]:
        del catalog
        return {"schema_version": "synthetic", "train": ["a", "b"], "validation": ["v"], "evaluation": ["e"]}

    def declared_objective_weights(self) -> dict[str, float]:
        return {name: 1.0 for name in _TERMS}

    def formula_identity(self) -> dict[str, str]:
        return {name: f"synthetic.{name}" for name in _TERMS}

    def optimization_identity(self) -> dict[str, object]:
        return {"algorithm": "synthetic-fairgrad", "near_opposition_tolerance": 1.0e-6}

    def optimizer_parameter_groups(self) -> tuple[MethodParameterGroup, ...]:
        # synthetic lifecycle 只有一个 shared 参数；两个 private 组用零乘辅助参数保持三组合同。
        return (
            MethodParameterGroup("shared_encoder", (self._parameter(),)),
            MethodParameterGroup("density_reader", (self.density_private,)),
            MethodParameterGroup("kappa_reader", (self.kappa_private,)),
        )

    def teacher_baseline_statistics(self, batch) -> dict[str, torch.Tensor]:
        count = torch.tensor(float(batch.sample_count), device=self._parameter().device)
        return {"density_sum": count, "kappa_sum": count, "count": count}

    def merge_teacher_baseline_statistics(self, total, block):
        if total is None:
            return {name: value.clone() for name, value in block.items()}
        return {name: total[name] + block[name] for name in total}

    def finalize_teacher_baselines(self, statistics) -> dict[str, object]:
        return {
            "density": {"predictor": "constant_teacher_mean_per_bandwidth_slot", "baseline_mse": 1.0},
            "kappa": {"predictor": "zero", "baseline_mse": 1.0},
        }

    def forward_objectives(
        self,
        batch,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:
        del mode, microbatch_size
        self.forward_steps.append(step)
        error = self._parameter().square()
        denominator = torch.tensor(float(batch.sample_count), device=error.device)
        objectives = {
            name: ObjectiveTermResult(
                name=name,
                components=(AdditiveStatistic(name, error * denominator, denominator),),
                metrics={"loss": error},
            )
            for name in _TERMS
        }
        return MethodStep(objectives=objectives, sample_count=batch.sample_count)

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        total_samples = sum(step.sample_count for step in steps)
        value = float(self._parameter().detach().square())
        return MethodUpdate(
            terms={name: value for name in _TERMS},
            sample_count=total_samples,
            denominators={name: float(total_samples) for name in _TERMS},
        )

    def backward_update(
        self,
        batch,
        *,
        forward_step: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:
        r"""模拟 concrete streaming backward，并记录 lifecycle 传入的 4-epoch cadence。"""

        self.z_gradient_requests.append(collect_z_gradients)
        step = self.forward_objectives(batch, step=forward_step, microbatch_size=microbatch_size)
        update = self.reduce_update((step,))
        self._parameter().grad = torch.full_like(self._parameter(), 2.0**0.5)
        self.density_private.grad = torch.zeros_like(self.density_private)
        self.kappa_private.grad = torch.zeros_like(self.kappa_private)
        if not collect_z_gradients:
            return update
        return MethodUpdate(
            terms=update.terms,
            sample_count=update.sample_count,
            denominators=update.denominators,
            gradient_evidence={"fairgrad/cosine": 0.0, "fairgrad/combined_norm": 2.0**0.5},
        )

    def evaluate_session(self, session, schedule, *, include_ablations: bool = False) -> MethodEvaluationReport:
        while not schedule.complete:
            session.realize(schedule.next(), schedule=schedule, step=0)
        value = float(self._parameter().detach().square())
        metrics = {name: value for name in _METRICS}
        ablations = None
        if include_ablations:
            ablation_names = ("full", "query_only", "same_asset_q_shuffle", "cross_asset_shuffle", "joint_token_shuffle")
            per_ablation = {name: {metric: value for metric in metrics} for name in ablation_names}
            ablations = {
                "pairing_key": ["asset_id", "q_index"],
                "ablations": ablation_names,
                "records": [{"asset_id": session.suite, "q_index": 0, "metrics": per_ablation}],
            }
        return MethodEvaluationReport(
            metrics=metrics,
            strata={"metric_scores": metrics, "bank_digest_sha256": f"fixed-{session.role}-{session.suite}"},
            teacher_baselines={
                "density": {"baseline_mse": 1.0},
                "kappa": {"baseline_mse": 100.0, "physical_baseline_mse": 1.0},
            },
            ablations=ablations,
        )

    def analyze_ablations(self, evidence, *, bootstrap_replicates: int, seed: int) -> dict[str, object]:
        return {"record_count": len(evidence["records"]), "bootstrap_replicates": bootstrap_replicates, "seed": seed}

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "parameter": self._parameter().detach().clone(),
            "density_private": self.density_private.detach().clone(),
            "kappa_private": self.kappa_private.detach().clone(),
        }

    def load_training_state_dict(self, state) -> None:
        self._parameter().data.copy_(state["parameter"])
        self.density_private.data.copy_(state["density_private"])
        self.kappa_private.data.copy_(state["kappa_private"])

    def retained_artifact_payload(self, *, metadata, source_checkpoint: Path) -> dict[str, object]:
        self.retained_export_count += 1
        return {
            "schema_version": "synthetic",
            "artifact_type": "synthetic_retained",
            "retained_state": {"parameter": self._parameter().detach().cpu()},
            "lineage": {"source_checkpoint": str(source_checkpoint), "code_revision": metadata["code_revision"]},
        }

    def close(self) -> None:
        return None


def _trainer(*, mini_epochs: int, max_epochs: int = 2, num_minibatches: int = 2) -> EmbodimentPretrainTrainer:
    config = EmbodimentPretrainTrainerCfg(
        sampling=OnlineSamplingCfg(
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=17,
        ),
        max_epochs=max_epochs,
        num_minibatches=num_minibatches,
        mini_epochs=mini_epochs,
        microbatch_size=2,
        optimizer=AdamWCfg(learning_rate=0.1, weight_decay=0.0),
        checkpoint_every_epochs=1,
        max_resident_assets=2,
    )
    return EmbodimentPretrainTrainer(config)


def _resolved(trainer: EmbodimentPretrainTrainer) -> dict[str, object]:
    return {
        "schema_version": "8.0.0",
        "data": {"manifest": "synthetic-ssl.yaml"},
        "method": {
            "state_measure": {"kind": "synthetic"},
            "representation": {"kind": "synthetic"},
            "model": {"kind": "synthetic"},
            "objectives": {name: {} for name in _TERMS},
            "fairgrad": {"algorithm": "synthetic-fairgrad", "near_opposition_tolerance": 1.0e-6},
            "entity_permutation": {"enabled": True, "seed_offset": 31_337},
            "joint_sign_rewrite": {"probability": 0.2},
        },
        "trainer": asdict(trainer.config),
        "run": {
            "seed": 17,
            "source_cache_root": "",
            "source_cache_mode": "off",
        },
    }


def _stage_resolved(*, role: str, stage_config: Any, run_config: Any) -> dict[str, object]:
    r"""构造 synthetic 事后阶段配置；data/method 必须与训练 checkpoint 完全一致。"""

    return {
        "schema_version": "1.0.0",
        "data": {"manifest": "synthetic-ssl.yaml"},
        "method": {
            "state_measure": {"kind": "synthetic"},
            "representation": {"kind": "synthetic"},
            "model": {"kind": "synthetic"},
            "objectives": {name: {} for name in _TERMS},
            "fairgrad": {"algorithm": "synthetic-fairgrad", "near_opposition_tolerance": 1.0e-6},
            "entity_permutation": {"enabled": True, "seed_offset": 31_337},
            "joint_sign_rewrite": {"probability": 0.2},
        },
        role: asdict(stage_config),
        "run": asdict(run_config),
    }


def test_pretrain_validation_and_evaluation_are_explicit_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """纯训练内闭合 baseline 且无评估副作用；事后阶段消费同一组 schema-8 checkpoints。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    pretrain_dir = tmp_path / "pretrain"
    pretrain_trainer = _trainer(mini_epochs=1, max_epochs=4, num_minibatches=1)
    pretrain_run = PretrainRun(
        PretrainRunCfg(
            seed=17,
            deterministic_algorithms=False,
            source_cache_mode="off",
        )
    )
    pretrain_method = _Method()
    fit_embodiment_pretrain(
        trainer=pretrain_trainer,
        data=_Data(),
        method=pretrain_method,
        run=pretrain_run,
        output_dir_override=pretrain_dir,
        resolved_config=_resolved(pretrain_trainer),
    )

    assert (pretrain_dir / "checkpoints" / "epoch_000000.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000001.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000002.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000004.pt").is_file()
    last = pretrain_dir / "checkpoints" / "last.pt"
    assert last.is_file()
    assert last.stat().st_ino == (pretrain_dir / "checkpoints" / "epoch_000004.pt").stat().st_ino
    assert not (pretrain_dir / "checkpoints" / "best.pt").exists()
    assert not (pretrain_dir / "retained_artifact.pt").exists()
    assert not (pretrain_dir / "final_evaluation.yaml").exists()
    assert not (pretrain_dir / "training_morphology_q_bank.yaml").exists()
    assert (pretrain_dir / "run_teacher_baselines.yaml").is_file()
    assert (pretrain_dir / "metrics_finalized.jsonl").is_file()
    assert (pretrain_dir / "training_summary.yaml").is_file()
    assert not (pretrain_dir / "asset_manifest.yaml").exists()
    assert pretrain_method.opened_roles == ["train"]
    assert pretrain_method.retained_export_count == 0
    assert len(pretrain_method.training_trace) == 4
    assert pretrain_method.forward_steps == list(range(4))
    assert pretrain_method.z_gradient_requests == [False, False, False, True]

    source_files = tuple(sorted(path.relative_to(pretrain_dir) for path in pretrain_dir.rglob("*") if path.is_file()))
    validation_config = ValidationCfg(
        q_per_asset=1,
        assets_per_minibatch=1,
        q_per_asset_per_minibatch=1,
        max_resident_assets=2,
    )
    validation_run_config = ValidationRunCfg(
        baseline_checkpoint=str(pretrain_dir / "checkpoints" / "epoch_000000.pt"),
        checkpoints=(
            str(pretrain_dir / "checkpoints" / "epoch_000001.pt"),
            str(pretrain_dir / "checkpoints" / "epoch_000002.pt"),
        ),
        seed=17,
        deterministic_algorithms=False,
    )
    validation_dir = tmp_path / "validation"
    validation_method = _Method()
    validate_checkpoints(
        data=_Data(),
        method=validation_method,
        config=validation_config,
        run=ValidationRun(validation_run_config),
        output_dir_override=validation_dir,
        resolved_config=_stage_resolved(
            role="validation",
            stage_config=validation_config,
            run_config=validation_run_config,
        ),
    )

    assert (validation_dir / "checkpoints" / "best.pt").is_file()
    selection = yaml.safe_load((validation_dir / "checkpoint_selection.yaml").read_text(encoding="utf-8"))
    assert selection["best_source_checkpoint"].endswith("epoch_000002.pt")
    assert validation_method.opened_roles == ["validation", "validation", "validation"]
    assert validation_method.retained_export_count == 0
    assert tuple(sorted(path.relative_to(pretrain_dir) for path in pretrain_dir.rglob("*") if path.is_file())) == source_files

    evaluation_config = EvaluationCfg(
        q_per_asset=1,
        assets_per_minibatch=1,
        q_per_asset_per_minibatch=1,
        bootstrap_replicates=2,
        max_resident_assets=2,
    )
    evaluation_run_config = EvaluationRunCfg(
        checkpoint=str(validation_dir / "checkpoints" / "best.pt"),
        baseline_checkpoint=str(pretrain_dir / "checkpoints" / "epoch_000000.pt"),
        seed=17,
        deterministic_algorithms=False,
    )
    evaluation_dir = tmp_path / "evaluation"
    evaluation_method = _Method()
    evaluate_checkpoint(
        data=_Data(),
        method=evaluation_method,
        config=evaluation_config,
        run=EvaluationRun(evaluation_run_config),
        output_dir_override=evaluation_dir,
        resolved_config=_stage_resolved(
            role="evaluation",
            stage_config=evaluation_config,
            run_config=evaluation_run_config,
        ),
    )

    final = yaml.safe_load((evaluation_dir / "evaluation.yaml").read_text(encoding="utf-8"))["suites"]
    assert final["unseen"]["metrics"]
    assert final["unseen"]["teacher_baselines"]["kappa"]["physical_baseline_mse"] == 1.0
    assert final["unseen"]["ablation_analysis"]["bootstrap_replicates"] == 2
    assert final["official_zero_shot"] == {"status": "empty", "asset_count": 0}
    assert (evaluation_dir / "training_morphology_q_bank.yaml").is_file()
    assert not (evaluation_dir / "retained_artifact.pt").exists()
    assert evaluation_method.retained_export_count == 0

    no_baseline_run_config = EvaluationRunCfg(
        checkpoint=str(validation_dir / "checkpoints" / "best.pt"),
        seed=17,
        deterministic_algorithms=False,
    )
    no_baseline_dir = tmp_path / "evaluation_without_baseline"
    no_baseline_method = _Method()
    evaluate_checkpoint(
        data=_Data(),
        method=no_baseline_method,
        config=evaluation_config,
        run=EvaluationRun(no_baseline_run_config),
        output_dir_override=no_baseline_dir,
        resolved_config=_stage_resolved(
            role="evaluation",
            stage_config=evaluation_config,
            run_config=no_baseline_run_config,
        ),
    )

    assert not (no_baseline_dir / "training_morphology_q_bank.yaml").exists()
    assert "training_evaluation" not in no_baseline_method.opened_roles
