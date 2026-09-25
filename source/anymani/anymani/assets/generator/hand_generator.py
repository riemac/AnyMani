"Orchestrates pre-made construction, post-mutation, validation, mesh materialization, physics closure, and export. Pre-made work is CPU object generation, not GPU simulation; full mode and general URDF restoration remain unsupported."

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilder, HandBuilderCfg
from ..asset_physics import AssetPhysicsCfg, close_hand_physics
from ..exporter import HandExporter, HandExporterCfg
from ..geometry_identity import geometry_fingerprint_from_hand
from ..procedural_meshes import materialize_hand_procedural_meshes
from ..validator import HandValidator, HandValidatorCfg
from .mutate import HandMutator, HandMutatorCfg
from .premade.batch import (
    build_premade_tasks,
    run_premade_parallel,
    run_premade_serial,
)
from .premade.connectivity import (
    apply_connectivity_preset as _apply_premade_connectivity_preset,
)
from .premade.connectivity import (
    connectivity_names_for_hand_preset as _connectivity_names_for_premade_hand_preset,
)
from .premade.connectivity import (
    resolve_single_premade_selection as _resolve_single_premade_selection,
)
from .premade.identity import resolve_export_root as _resolve_premade_export_root
from .premade.identity import stable_premade_id
from .premade.normalize import normalize_connectivity_mapping, normalize_name_list
from .premade.topology import (
    build_base_hand as _build_premade_base_hand,
)
from .premade.topology import (
    candidate_hand_preset_names as _candidate_premade_hand_preset_names,
)
from .presentation.recolor import (
    RecolorSpec,
    describe_recolor_spec,
    normalize_recolor_spec,
    resolve_visual_recolor_materials,
)
from .result import HandGenerationResult
from .runtime.artifact_lifecycle import rollback_created_directory, rollback_written_artifacts
from .runtime.mutate_batch import PostMutateVariantSetResult, run_post_mutate_source_batch
from .runtime.mutate_sampling import run_mutate_batch_with_independent_proposals
from .runtime.restore import PostMutateSource, load_post_mutate_source
from .runtime.run_context import GenerationRunContext


def _has_enabled_mutation(cfg: HandMutatorCfg) -> bool:
    r"""Check whether any post-mutate tool is enabled in the cfg."""

    return cfg.has_terms()


def _sample_mutation_terms(mutator: HandMutator, target: HandCfg) -> dict[str, dict[str, Any]]:

    batch = mutator.sample_batch(target, batch_size=1)
    return batch[0] if batch else {}


# ============================================================================

# ============================================================================


@dataclass(frozen=True)
class PostMutateSourceCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    task_id: str
    source_topology_dir: Path | str
    n_samples: int
    seed: int
    build_id: str = ""
    "Stable semantic identifier preserved in the output metadata."

    selection_lock_sha256: str = ""
    "Exact-byte identity of the frozen selection lock."

    attempt_index: int = 0
    "Zero-based proposal attempt within one requested variant slot."

    generator_config_sha256: str = ""
    "SHA-256 of the generation configuration used for this run."

    child_config_sha256: str = ""
    "SHA-256 of the exact child configuration used for lowering."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        object.__setattr__(self, "source_topology_dir", Path(self.source_topology_dir))
        if not self.task_id.strip():
            raise ValueError("post-mutate source task_id cannot be empty")
        if self.n_samples < 0:
            raise ValueError("post-mutate source n_samples must be non-negative")
        if self.seed < 0:
            raise ValueError("post-mutate source seed must be non-negative")
        if self.attempt_index < 0:
            raise ValueError("post-mutate source attempt_index must be non-negative")


@dataclass
class HandGeneratorCfg(AssetCfgBase):
    """Declares generation stages, validators, physics closure, and output behavior.

    For pre-made generation, hand_presets and connectivity_presets are the two recipe lists. Family, handedness, missing-slot expansion, validation, and output are separate controls. The stage enumerates discrete combinations; max_enumerate caps that list. Post-mutation applies n_samples to an explicit mother. Pre-made workers use CPU object construction; SDF device selection belongs to post-mutation validation.
    """

    class_type: type[HandGenerator] | None = None
    "Associated runtime implementation for this configuration class."

    mode: Literal["made", "mutate", "full"] = "full"
    "Generation stage: made enumerates pre-made topologies; mutate derives variants from an explicit mother; full is unsupported."

    artifact_level: Literal["hand_cfg", "urdf", "bundle"] = "bundle"
    "Whether the pipeline returns a hand configuration or writes a complete asset bundle."

    n_samples: int = 1
    "Number of post-mutation variants requested per source topology."

    max_enumerate: int | None = None
    "Maximum number of discrete topology candidates; None requests the complete space."

    premade_parallel: bool = True
    "Whether pre-made tasks use bounded CPU workers."

    premade_parallel_workers: int | None = None
    "Optional worker limit; None selects a CPU-based default."

    premade_parallel_fallback: Literal["serial", "raise"] = "serial"
    "Behavior when a pre-made worker fails before producing a bundle."

    post_mutate_seed: int = 20260813
    "Root random seed for the joint mutation proposal stream."

    post_mutate_attempts_per_variant: int = 10
    "Maximum independent candidate attempts for each requested variant slot."

    post_mutate_require_unique_geometry: bool = False
    "Whether duplicate physical geometries are rejected and resampled."

    post_mutate_sources: list[PostMutateSourceCfg] = field(default_factory=list)
    "Ordered mother and variant-set sources for the post-mutation partition."

    post_mutate_parallel: bool = True
    "Whether post-mutation mothers use bounded CPU workers."

    post_mutate_parallel_workers: int | None = None
    "Maximum CPU worker count for post-mutation source tasks."

    post_mutate_sdf_execution: Literal["local", "central_gpu_batch"] = "local"
    "SDF execution mode; service failures are explicit and do not change device."

    Made: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    "Pre-made stage: enumerate and validate discrete topology candidates."

    Mutate: HandMutatorCfg = field(default_factory=HandMutatorCfg)
    "Post-mutate stage: apply local geometry or joint-parameter proposals to a source hand."

    Validate: HandValidatorCfg | None = None
    "Optional structural or geometry acceptance gates for generated candidates."

    Export: HandExporterCfg = field(default_factory=HandExporterCfg)
    "Optional URDF and sidecar export settings."

    Physics: AssetPhysicsCfg | None = field(default_factory=AssetPhysicsCfg)
    "Optional mass and inertia closure computed from final collision geometry."

    output_dir: Path | str = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "generated")
    "Root for files created by this generation run."

    source_topology_dir: Path | str | None = None
    "Explicit pre-made topology bundle used as the mutation source."

    handedness: Literal["left", "right", "all"] = "all"
    "Requested side; generated left hands follow the strict mirror contract."

    hand_presets: list[str] = field(default_factory=list)
    "Editable list of named base-hand recipes; remains a list to preserve the user-authored recipe order."

    connectivity_presets: dict[str, dict[str, list[str]]] | None = None
    "Optional map from each hand preset to its allowed, explicit finger-level connectivity recipes."

    mixed: bool = True
    "Whether non-thumb fingers may mix family mechanisms while the palm family stays fixed."

    missing: bool = True
    "Whether topology enumeration may remove eligible finger slots."

    recolored: RecolorSpec = None
    "Visual-only material preset; it must not affect collision geometry or physical identity."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandGenerator
        if self.Physics is not None and not isinstance(self.Physics, AssetPhysicsCfg):
            self.Physics = AssetPhysicsCfg(**dict(self.Physics))
        self.output_dir = Path(self.output_dir)
        if self.source_topology_dir is not None:
            self.source_topology_dir = Path(self.source_topology_dir)
        self.post_mutate_sources = [
            source if isinstance(source, PostMutateSourceCfg) else PostMutateSourceCfg(**source)
            for source in self.post_mutate_sources
        ]
        self.hand_presets = normalize_name_list(self.hand_presets, field_name="hand_presets")
        self.connectivity_presets = normalize_connectivity_mapping(self.connectivity_presets)
        self.recolored = normalize_recolor_spec(self.recolored)
        if self.premade_parallel_workers is not None and self.premade_parallel_workers < 1:
            raise ValueError("premade_parallel_workers must be >= 1 when provided")
        if self.premade_parallel_fallback not in {"serial", "raise"}:
            raise ValueError("premade_parallel_fallback must be either 'serial' or 'raise'")
        if int(self.post_mutate_attempts_per_variant) < 1:
            raise ValueError("post_mutate_attempts_per_variant must be >= 1")
        if not isinstance(self.post_mutate_require_unique_geometry, bool):
            raise TypeError("post_mutate_require_unique_geometry must be bool")
        if self.post_mutate_parallel_workers is not None and self.post_mutate_parallel_workers < 1:
            raise ValueError("post_mutate_parallel_workers must be >= 1 when provided")
        if self.post_mutate_sdf_execution not in {"local", "central_gpu_batch"}:
            raise ValueError("post_mutate_sdf_execution must be 'local' or 'central_gpu_batch'")
        task_ids = [source.task_id for source in self.post_mutate_sources]
        source_paths = [Path(source.source_topology_dir) for source in self.post_mutate_sources]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("post_mutate_sources task_id values must be unique")
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("post_mutate_sources source topology paths must be unique within one batch")



        if self.connectivity_presets is not None and not self.hand_presets:
            raise ValueError("connectivity_presets requires hand_presets to be provided together")




        if len(self.hand_presets) > 1 and self.Made.class_type is not HandBuilder:
            raise ValueError(
                "When hand_presets contains multiple base hand presets, Made must stay abstract; "
                "otherwise one concrete builder cfg would be incorrectly reused for all preset anchors."
            )
        if self.source_topology_dir is not None and self.post_mutate_sources:
            raise ValueError("source_topology_dir and post_mutate_sources are mutually exclusive")
        if self.mode == "mutate" and self.source_topology_dir is None and not self.post_mutate_sources:
            raise ValueError("mode='mutate' requires one source_topology_dir or non-empty post_mutate_sources")
        if self.mode != "mutate" and (self.source_topology_dir is not None or self.post_mutate_sources):
            raise ValueError("post-mutate source fields are only valid when mode='mutate'")
        if self.mode == "mutate" and not _has_enabled_mutation(self.Mutate):
            raise ValueError("mode='mutate' requires at least one enabled mutator term")


# ============================================================================

# ============================================================================


class HandGenerator:
    """Runs pre-made construction or independent post-mutation and records stage outputs.

    The generator builds hand objects and asset files; it does not start an Isaac simulation. Physics closure derives mass and inertia from the final collision geometry.
    """

    cfg: HandGeneratorCfg

    def __init__(self, cfg: HandGeneratorCfg):
        self.cfg = cfg
        self._run_context: GenerationRunContext | None = None
        self._mutate_source: PostMutateSource | None = None
        self._last_rejection_detail: dict[str, Any] | None = None
        self._post_mutate_geometry_registry: dict[str, str] | None = None

    def _ensure_run_context(self) -> GenerationRunContext:

        if self._run_context is not None:
            return self._run_context
        from .runtime.recipe_loader import RecipeLoader

        self._run_context = GenerationRunContext.create(
            self.cfg,
            config_dump=RecipeLoader.dump(self.cfg),
        )
        return self._run_context

    def _make_worker_run_context(self, run_root: Path) -> GenerationRunContext:

        self._run_context = GenerationRunContext(
            root_dir=Path(run_root),
            summary={"stats": {"by_topology": {}}},
        )
        return self._run_context

    def _load_mutate_source(self) -> PostMutateSource:

        if self._mutate_source is not None:
            return self._mutate_source
        if self.cfg.source_topology_dir is None:
            raise ValueError("Independent post-mutate requires 'source_topology_dir'")
        self._mutate_source = load_post_mutate_source(self.cfg.source_topology_dir)
        return self._mutate_source

    def _ensure_post_mutate_geometry_registry(self) -> dict[str, str]:

        if self._post_mutate_geometry_registry is None:
            source = self._load_mutate_source()
            mother_fingerprint = geometry_fingerprint_from_hand(source.hand_cfg)
            self._post_mutate_geometry_registry = {
                mother_fingerprint: f"mother:{source.origin_sample_id}",
            }
        return self._post_mutate_geometry_registry

    def _write_run_summary(self) -> None:

        if self._run_context is None:
            return
        self._run_context.write_summary()

    def _record_generation_rejection(
        self,
        *,
        stage: str,
        error_codes: tuple[str, ...] = (),
        write_summary: bool = True,
    ) -> None:

        self._ensure_run_context().record_rejection(
            stage=stage,
            error_codes=error_codes,
            write_summary=write_summary,
        )

    def _record_generation_success(self, result: HandGenerationResult, *, write_summary: bool = True) -> None:

        self._ensure_run_context().record_success(result, write_summary=write_summary)

    def _close_physics_if_enabled(
        self,
        hand_cfg: HandCfg,
        *,
        stage: str,
        path_metadata: dict[str, Any] | None = None,
    ) -> tuple[HandCfg, tuple[Path, ...]]:

        path_probe = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata=dict(path_metadata or {}),
        )
        # SDF validation and export consume real OBJ files, even when physics closure is disabled.
        # Validators and exporters need the materialized OBJ even when physics closure is disabled.
        materialized, written_paths = materialize_hand_procedural_meshes(
            hand_cfg,
            mesh_root_dir=self._resolve_mesh_root(result=path_probe),
        )
        try:
            closed = close_hand_physics(materialized, self.cfg.Physics, stage=stage)
        except Exception:

            rollback_written_artifacts(written_paths, boundary_dir=self._ensure_run_context().root_dir)
            raise
        return closed, tuple(written_paths)

    def _candidate_hand_preset_names(self) -> tuple[str, ...]:

        return _candidate_premade_hand_preset_names(self.cfg)

    def _connectivity_names_for_hand_preset(self, *, hand_preset_name: str) -> tuple[str, ...]:

        return _connectivity_names_for_premade_hand_preset(self.cfg, hand_preset_name=hand_preset_name)

    def _resolve_single_premade_selection(self) -> tuple[str | None, str | None] | None:

        return _resolve_single_premade_selection(self.cfg)

    def _build_base_hand(self, *, hand_preset_name: str | None) -> tuple[HandCfg, str]:

        return _build_premade_base_hand(self.cfg, hand_preset_name=hand_preset_name)

    def _apply_connectivity_preset(
        self,
        hand_cfg: HandCfg,
        *,
        connectivity_preset_name: str,
        hand_preset_name: str | None,
    ) -> tuple[HandCfg, dict[str, Any]]:

        return _apply_premade_connectivity_preset(
            self.cfg,
            hand_cfg,
            connectivity_preset_name=connectivity_preset_name,
            hand_preset_name=hand_preset_name,
        )

    def _resolve_export_root(self, *, result: HandGenerationResult) -> Path:

        run_root = self._ensure_run_context().root_dir
        if self.cfg.mode == "mutate":
            return run_root
        return _resolve_premade_export_root(self.cfg, result=result, run_root=run_root)

    def _resolve_mesh_root(self, *, result: HandGenerationResult) -> Path:

        export_root = self._resolve_export_root(result=result)
        if self.cfg.mode == "mutate":
            return export_root / self.cfg.Export.Urdf.canonical_mesh_dirname
        return export_root / self.cfg.Export.Urdf.canonical_mesh_dirname

    def _generate_once(
        self,
        *,
        hand_preset_name: str | None,
        connectivity_preset_name: str | None,
        enumerated: bool = False,
        sampled_mutation_terms: dict[str, dict[str, float]] | None = None,
        record_summary: bool = True,
    ) -> HandGenerationResult | None:

        self._ensure_run_context()

        validator = HandValidator(self.cfg.Validate) if self.cfg.Validate is not None else None
        validation_warnings: list[str] = []
        validation_metadata: dict[str, Any] = {}
        written_mesh_paths: tuple[Path, ...] = ()
        candidate_export_root: Path | None = None
        candidate_export_root_preexisted = False
        candidate_geometry_fingerprint: str | None = None

        if self.cfg.mode == "mutate":
            mutate_source = self._load_mutate_source()
            hand_cfg = mutate_source.hand_cfg.copy()
            builder_cfg_name = str(mutate_source.metadata.get("builder_cfg", "restored_hand_cfg"))
            premade_metadata = dict(mutate_source.metadata)
        else:
            hand_cfg, builder_cfg_name = self._build_base_hand(hand_preset_name=hand_preset_name)

            premade_metadata: dict[str, Any] = {}
            if connectivity_preset_name is not None:
                hand_cfg, premade_metadata = self._apply_connectivity_preset(
                    hand_cfg,
                    connectivity_preset_name=connectivity_preset_name,
                    hand_preset_name=hand_preset_name,
                )
            path_probe = HandGenerationResult(hand_cfg=hand_cfg, metadata=dict(premade_metadata))
            candidate_export_root = self._resolve_export_root(result=path_probe)
            candidate_export_root_preexisted = candidate_export_root.exists()
            hand_cfg, written_mesh_paths = self._close_physics_if_enabled(
                hand_cfg,
                stage="pre_made",
                path_metadata=premade_metadata,
            )




            if validator is not None:
                try:
                    pre_made_validation = validator.validate_pre_made(hand_cfg)
                except Exception:
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    raise
                if not pre_made_validation:
                    self._last_rejection_detail = {
                        "stage": "pre_made_validate",
                        "errors": list(pre_made_validation.errors),
                        "error_codes": list(pre_made_validation.error_codes),
                        "metadata": dict(pre_made_validation.metadata),
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="pre_made_validate",
                        error_codes=tuple(pre_made_validation.error_codes),
                        write_summary=record_summary,
                    )
                    return None
                validation_warnings.extend(pre_made_validation.warnings)
                validation_metadata["pre_made"] = dict(pre_made_validation.metadata)
        sampled_terms: dict[str, dict[str, float]] | None = None


        #



        if self.cfg.mode == "mutate" and _has_enabled_mutation(self.cfg.Mutate):
            mutator = HandMutator(self.cfg.Mutate)
            sampled_terms = sampled_mutation_terms or _sample_mutation_terms(mutator, hand_cfg)
            hand_cfg = mutator.mutate(hand_cfg, sampled_params=sampled_terms)
            if hand_cfg is None:
                self._last_rejection_detail = {
                    "stage": "mutate",
                    "errors": ["mutator returned None"],
                    "error_codes": ["mutate.returned_none"],
                    "metadata": {},
                }
                self._record_generation_rejection(
                    stage="mutate",
                    error_codes=("mutate.returned_none",),
                    write_summary=record_summary,
                )
                return None
            hand_cfg, written_mesh_paths = self._close_physics_if_enabled(
                hand_cfg,
                stage="post_mutate",
                path_metadata=premade_metadata,
            )
            if validator is not None:
                try:
                    post_mutate_validation = validator.validate_post_mutate(hand_cfg)
                except Exception:
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    raise
                if not post_mutate_validation:
                    self._last_rejection_detail = {
                        "stage": "post_mutate_validate",
                        "errors": list(post_mutate_validation.errors),
                        "error_codes": list(post_mutate_validation.error_codes),
                        "metadata": dict(post_mutate_validation.metadata),
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="post_mutate_validate",
                        error_codes=tuple(post_mutate_validation.error_codes),
                        write_summary=record_summary,
                    )
                    return None
                validation_warnings.extend(post_mutate_validation.warnings)
                validation_metadata["post_mutate"] = dict(post_mutate_validation.metadata)




            if self.cfg.post_mutate_require_unique_geometry:
                candidate_geometry_fingerprint = geometry_fingerprint_from_hand(hand_cfg)
                registry = self._ensure_post_mutate_geometry_registry()
                previous = registry.get(candidate_geometry_fingerprint)
                if previous is not None:
                    duplicate_kind = "mother" if previous.startswith("mother:") else "variant"
                    error_code = f"post_mutate.duplicate_{duplicate_kind}_geometry"
                    self._last_rejection_detail = {
                        "stage": "post_mutate_unique_geometry",
                        "errors": [f"static geometry duplicates {previous}"],
                        "error_codes": [error_code],
                        "metadata": {
                            "geometry_fingerprint": candidate_geometry_fingerprint,
                            "duplicate_of": previous,
                        },
                    }
                    rollback_written_artifacts(
                        written_mesh_paths,
                        boundary_dir=self._ensure_run_context().root_dir,
                    )
                    self._record_generation_rejection(
                        stage="post_mutate_unique_geometry",
                        error_codes=(error_code,),
                        write_summary=record_summary,
                    )
                    return None

        sample_id = uuid4().hex[:8]
        if self.cfg.mode != "mutate" and connectivity_preset_name is not None and enumerated:
            sample_id = stable_premade_id(
                hand_preset_name or hand_cfg.family,
                connectivity_preset_name,
            )

        metadata = {
            "id": sample_id,
            "builder_cfg": builder_cfg_name,
            "warnings": validation_warnings,
            "family": hand_cfg.family,
            "handedness": hand_cfg.handedness,
        }
        metadata.update({key: value for key, value in premade_metadata.items() if value is not None})
        if sampled_terms:
            metadata["post_mutate_samples"] = sampled_terms
        if validation_metadata:
            metadata["validation"] = validation_metadata
        if isinstance(hand_cfg.metadata.get("post_mutate_samples"), dict):
            merged_samples = dict(metadata.get("post_mutate_samples", {}))
            merged_samples.update(hand_cfg.metadata["post_mutate_samples"])
            metadata["post_mutate_samples"] = merged_samples
        recolor_metadata = describe_recolor_spec(self.cfg.recolored)
        if recolor_metadata is not None:
            metadata["recolored"] = recolor_metadata
        if candidate_geometry_fingerprint is not None:
            metadata["geometry_fingerprint"] = candidate_geometry_fingerprint

        result = HandGenerationResult(
            hand_cfg=hand_cfg,
            metadata=metadata,
        )



        if self.cfg.artifact_level != "hand_cfg":
            resolved_recolor_materials = resolve_visual_recolor_materials(hand_cfg, self.cfg.recolored)
            export_cfg = self.cfg.Export.replace(
                artifact_level=self.cfg.artifact_level,
                Urdf=self.cfg.Export.Urdf.replace(
                    recolored_materials=resolved_recolor_materials,
                ),
            )
            exporter = HandExporter(export_cfg)
            try:
                exporter.export(
                    result,
                    output_dir=self._resolve_export_root(result=result),
                    nest_sample_dir=self.cfg.mode == "mutate",
                    mesh_root_dir=self._resolve_mesh_root(result=result),
                )
            except Exception:
                run_root = self._ensure_run_context().root_dir
                if (
                    self.cfg.mode != "mutate"
                    and candidate_export_root is not None
                    and not candidate_export_root_preexisted
                ):

                    rollback_created_directory(candidate_export_root, boundary_dir=run_root)
                else:
                    rollback_written_artifacts(written_mesh_paths, boundary_dir=run_root)
                raise

        self._record_generation_success(result, write_summary=record_summary)
        if candidate_geometry_fingerprint is not None:
            self._ensure_post_mutate_geometry_registry()[candidate_geometry_fingerprint] = f"variant:{sample_id}"
        self._last_rejection_detail = None
        return result

    def generate(self) -> HandGenerationResult | None:
        "Runs the configured generation stages and records their results."




        #   cfg.mode: `made` / `mutate` / `full`
        #   cfg.artifact_level: `hand_cfg` / `urdf` / `bundle`




        #

        #






        #

        # TODO: Add a general URDF-to-HandCfg restoration route.
        # TODO: Record more detailed provenance and rejection statistics.
        #



        if self.cfg.mode == "full":
            raise NotImplementedError(
                "mode='full' is temporarily unsupported. "
                "This migration only covers mode='made' and independent mode='mutate'; "
                "the full pipeline has not been adapted to topology-root export semantics yet."
            )
        if self.cfg.mode == "mutate":
            if self.cfg.post_mutate_sources:
                raise ValueError("multi-source mutate cfg must use generate_variant_sets(), not generate()")
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        selection = self._resolve_single_premade_selection()
        if selection is None:
            return self._generate_once(hand_preset_name=None, connectivity_preset_name=None)
        return self._generate_once(hand_preset_name=selection[0], connectivity_preset_name=selection[1])

    def generate_batch(self) -> Iterator[HandGenerationResult]:
        "Runs a bounded batch of candidate hand-generation tasks."

        if self.cfg.mode == "full":
            raise NotImplementedError(
                "mode='full' is temporarily unsupported. "
                "This migration only covers mode='made' and independent mode='mutate'; "
                "the full pipeline has not been adapted to topology-root export semantics yet."
            )

        if self.cfg.mode == "mutate":
            if self.cfg.post_mutate_sources:
                raise ValueError("multi-source mutate cfg must use generate_variant_sets(), not generate_batch()")
            target_count = max(int(self.cfg.n_samples), 0)
            mutator = HandMutator(self.cfg.Mutate)
            source_hand = self._load_mutate_source().hand_cfg
            yield from run_mutate_batch_with_independent_proposals(
                generator=self,
                mutator=mutator,
                source_hand=source_hand,
                target_count=target_count,
                attempts_per_variant=int(self.cfg.post_mutate_attempts_per_variant),
                seed=int(self.cfg.post_mutate_seed),
            )
            return

        hand_preset_names = self._candidate_hand_preset_names()
        if hand_preset_names:
            tasks = build_premade_tasks(self)


            if self.cfg.mode == "made" and self.cfg.premade_parallel:
                try:
                    results = run_premade_parallel(self, tasks=tasks)
                except Exception:
                    if self.cfg.premade_parallel_fallback == "raise":
                        raise
                    results = run_premade_serial(self, tasks=tasks)
            else:
                results = run_premade_serial(self, tasks=tasks)

            for result in results:
                yield result
            return

        target_count = max(int(self.cfg.n_samples), 0)
        success_count = 0
        attempt_count = 0
        max_attempts = max(target_count * 10, 10)

        while success_count < target_count:
            attempt_count += 1
            if attempt_count > max_attempts:
                raise RuntimeError("too many rejected samples during generate_batch()")
            result = self.generate()
            if result is None:
                continue
            yield result
            success_count += 1

    def generate_variant_sets(
        self,
        *,
        on_report: Callable[[PostMutateVariantSetResult], None] | None = None,
    ) -> Iterator[PostMutateVariantSetResult]:
        "Generates independent post-mutate sets while preserving the source mother and attempt provenance."

        if self.cfg.mode != "mutate" or not self.cfg.post_mutate_sources:
            raise ValueError("generate_variant_sets() requires mode='mutate' and non-empty post_mutate_sources")
        yield from run_post_mutate_source_batch(
            self,
            tasks=tuple(self.cfg.post_mutate_sources),
            on_report=on_report,
        )

__all__ = [
    "HandGenerationResult",
    "HandGeneratorCfg",
    "HandGenerator",
    "PostMutateSourceCfg",
    "PostMutateVariantSetResult",
]
