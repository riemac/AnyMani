r"""正式 fit façade 的 calibration → pretrain → validation → final evaluation 闭环。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from anymani.distill.methods.contracts import MethodEvaluationReport, MethodStep, MethodUpdate
from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult
from anymani.distill.ssl.runtime.lifecycle import fit_embodiment_pretrain
from anymani.distill.ssl.runtime.pretrainer import (
    AdamWCfg,
    EmbodimentPretrainTrainer,
    EmbodimentPretrainTrainerCfg,
    FinalEvaluationCfg,
    ValidationCfg,
)
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

pytestmark = [pytest.mark.contract, pytest.mark.skipif(not torch.cuda.is_available(), reason="fit requires CUDA")]

_TERMS = ("density", "kappa", "derived_field")


class _Dataset:
    source_sha256 = "synthetic-dataset-sha"
    source_path = Path("synthetic-ssl.yaml")

    @staticmethod
    def config_dict() -> dict[str, object]:
        return {"schema_version": "synthetic"}


class _Data:
    def resolve(self):
        return SimpleNamespace(dataset=_Dataset())


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

    def prepare(self, catalog, *, device: torch.device, dtype: torch.dtype) -> None:
        del catalog, device, dtype

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))

    def _parameter(self) -> torch.nn.Parameter:
        assert self.parameter is not None
        return self.parameter

    def parameters(self):
        return (self._parameter(),)

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def split_names(self, role: str) -> tuple[str, ...]:
        return {"train": ("",), "validation": ("validation",), "evaluation": ("unseen", "official_zero_shot")}[role]

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        if role == "evaluation" and suite == "official_zero_shot":
            return 0
        return 2 if role == "train" else 1

    def open_session(self, role: str, *, suite: str = "", device: torch.device, **_kwargs):
        return _Session(role, suite, self.split_asset_count(role, suite=suite), device, self.training_trace)

    def asset_manifest(self, catalog) -> dict[str, object]:
        del catalog
        return {"schema_version": "synthetic", "train": ["a", "b"], "validation": ["v"], "evaluation": ["e"]}

    def declared_objective_weights(self) -> dict[str, float]:
        return {name: 1.0 for name in _TERMS}

    def formula_identity(self) -> dict[str, str]:
        return {name: f"synthetic.{name}" for name in _TERMS}

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
        loss = torch.zeros((), device=self._parameter().device)
        for step in steps:
            for name in _TERMS:
                loss = loss + step.objectives[name].components[0].numerator
        loss = loss / total_samples
        value = float(self._parameter().detach().square())
        return MethodUpdate(loss=loss, terms={name: value for name in _TERMS}, sample_count=total_samples)

    def evaluate_session(self, session, schedule, *, include_ablations: bool = False) -> MethodEvaluationReport:
        while not schedule.complete:
            session.realize(schedule.next(), schedule=schedule, step=0)
        value = float(self._parameter().detach().square())
        metrics = {name: value for name in ("density", "kappa", "derived_field")}
        ablations = None
        if include_ablations:
            ablation_names = ("full", "query_only", "same_asset_q_shuffle", "cross_asset_shuffle", "first_order_zero", "first_order_joint_shuffle", "first_order_sign_flip")
            per_ablation = {name: {metric: value for metric in metrics} for name in ablation_names}
            ablations = {
                "pairing_key": ["asset_id", "q_index"],
                "ablations": ablation_names,
                "records": [{"asset_id": session.suite, "q_index": 0, "metrics": per_ablation}],
            }
        return MethodEvaluationReport(
            metrics=metrics,
            strata={"metric_scores": metrics, "bank_digest_sha256": f"fixed-{session.role}-{session.suite}"},
            ablations=ablations,
        )

    def analyze_ablations(self, evidence, *, bootstrap_replicates: int, seed: int) -> dict[str, object]:
        return {"record_count": len(evidence["records"]), "bootstrap_replicates": bootstrap_replicates, "seed": seed}

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        return {"parameter": self._parameter().detach().clone()}

    def load_training_state_dict(self, state) -> None:
        self._parameter().data.copy_(state["parameter"])

    def retained_artifact_payload(self, *, metadata, source_checkpoint: Path) -> dict[str, object]:
        return {
            "schema_version": "synthetic",
            "artifact_type": "synthetic_retained",
            "retained_state": {"parameter": self._parameter().detach().cpu()},
            "lineage": {"source_checkpoint": str(source_checkpoint), "code_revision": metadata["code_revision"]},
        }

    def close(self) -> None:
        return None


def _trainer(*, mini_epochs: int) -> EmbodimentPretrainTrainer:
    config = EmbodimentPretrainTrainerCfg(
        sampling=OnlineSamplingCfg(
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=17,
        ),
        max_epochs=2,
        num_minibatches=2,
        mini_epochs=mini_epochs,
        microbatch_size=2,
        validation=ValidationCfg(
            q_per_asset=1,
            assets_per_minibatch=1,
            q_per_asset_per_minibatch=1,
            every_epochs=1,
        ),
        final_evaluation=FinalEvaluationCfg(
            q_per_asset=1,
            assets_per_minibatch=1,
            q_per_asset_per_minibatch=1,
            bootstrap_replicates=2,
        ),
        optimizer=AdamWCfg(learning_rate=0.1, weight_decay=0.0),
        checkpoint_every_epochs=1,
        max_resident_assets=2,
    )
    return EmbodimentPretrainTrainer(config)


def _resolved(trainer: EmbodimentPretrainTrainer, phase: str) -> dict[str, object]:
    return {
        "schema_version": "6.0.0",
        "data": {"manifest": "synthetic-ssl.yaml"},
        "method": {
            "state_measure": {"kind": "synthetic"},
            "representation": {"kind": "synthetic"},
            "model": {"kind": "synthetic"},
            "objectives": {name: {"weight": 1.0} for name in _TERMS},
            "joint_sign_rewrite": {"probability": 0.2},
        },
        "trainer": asdict(trainer.config),
        "run": {"phase": phase},
    }


def test_formal_fit_runs_calibration_then_best_checkpoint_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一采样语义先校准，正式训练再选 best、评估 unseen 并显式报告空 official suite。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    calibration_trainer = _trainer(mini_epochs=1)
    calibration_dir = tmp_path / "calibration"
    calibration_run = PretrainRun(
        PretrainRunCfg(seed=17, phase="calibrate_objectives", deterministic_algorithms=False)
    )
    calibration_method = _Method()
    fit_embodiment_pretrain(
        trainer=calibration_trainer,
        data=_Data(),
        method=calibration_method,
        run=calibration_run,
        output_dir_override=calibration_dir,
        resolved_config=_resolved(calibration_trainer, "calibrate_objectives"),
    )
    artifact = calibration_dir / "loss_calibration.yaml"
    assert artifact.is_file()
    assert calibration_method._parameter().grad is None
    assert not (calibration_dir / "checkpoints").exists()

    pretrain_dir = tmp_path / "pretrain"
    pretrain_trainer = _trainer(mini_epochs=2)
    pretrain_run = PretrainRun(
        PretrainRunCfg(
            seed=17,
            phase="pretrain",
            calibration_artifact=str(artifact),
            deterministic_algorithms=False,
        )
    )
    pretrain_method = _Method()
    fit_embodiment_pretrain(
        trainer=pretrain_trainer,
        data=_Data(),
        method=pretrain_method,
        run=pretrain_run,
        output_dir_override=pretrain_dir,
        resolved_config=_resolved(pretrain_trainer, "pretrain"),
    )

    assert (pretrain_dir / "checkpoints" / "best.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000001.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000002.pt").is_file()
    assert (pretrain_dir / "retained_artifact.pt").is_file()
    final = yaml.safe_load((pretrain_dir / "final_evaluation.yaml").read_text(encoding="utf-8"))
    assert final["unseen"]["metrics"]
    assert final["unseen"]["ablation_analysis"]["bootstrap_replicates"] == 2
    assert final["official_zero_shot"] == {"status": "empty", "asset_count": 0}
    assert calibration_method.training_trace == pretrain_method.training_trace
    assert calibration_method.forward_steps == [0, 1, 2, 3]
    assert pretrain_method.forward_steps == list(range(8))
    assert calibration_method._parameter().grad is None
