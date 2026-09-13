# AnyMani

AnyMani is an Isaac Lab research framework for dexterous in-hand manipulation across hand morphologies. It separates
asset generation, robot spawning, task semantics, and policy training so that each experimental change has an explicit
contract.

## Architecture

```text
assets -> robots -> tasks -> distill
```

| Package | Responsibility |
| --- | --- |
| `source/anymani/anymani/assets/` | Generated-hand topology, mutation, validation, export, physics closure, and asset bank |
| `source/anymani/anymani/robots/` | Lower generated or real hand assets into Isaac Lab robot configurations |
| `source/anymani/anymani/tasks/` | Scene, observation, action, command, reward, reset, termination, and Gym registration |
| `source/anymani/anymani/distill/` | Physical representations, shared models, geometry SSL, family RL teachers, and offline IL students |

`Research/` is an independent downstream Obsidian vault for experiment evidence and scientific interpretation. Runtime
code does not require it.

## Installation

Install Isaac Lab first, then install AnyMani into the same Python environment:

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m pip install -e source/anymani
```

The project expects the local Isaac Lab checkout at `/home/hac/isaac/IsaacLab` for simulator-backed commands.

## Discover Tasks

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python scripts/list_envs.py
```

Current task families include:

- `AnyMani-LeapHand-*`: LEAP-style in-hand baselines and generated-hand controlled variants;
- `AnyMani-GM-*`: generalized-manipulation single-asset and LEAP baselines;
- `AnyMani-GM-*-MLP-v0`: GM task aliases with rl_games MLP training configuration.
- `AnyMani-Hetero-*`: generated heterogeneous-hand tasks with explicit canonical cohorts and shared policies.

Treat `scripts/list_envs.py` as the source of truth for exact IDs; research node numbers are intentionally not part of
AnyMani public names.

## Train and Play

### Heterogeneous in-hand rotation

The v1.0.0 research milestone freezes two family teachers and one shared LEAP-right / revised Allegro-right student. The teachers cover 96/128 and 89/128 training assets, respectively. The selected pure-distillation student covers 95/128 LEAP and 81/128 Allegro training assets and 76/128 strictly held-out hands. The paired three-seed comparison reports strict held-out coverage of 73.7 ± 3.2 for Ours, 52.7 ± 4.9 for No-Z, and 57.3 ± 5.0 for the FK auxiliary-head control. All coverage uses the fixed 30-second, R16 protocol and original denominators.

Teacher training is a staged lineage with explicit initialization, continuation and selection decisions. LEAP includes parameter averaging and optimizer-reset polishing; Allegro includes family-specific joint-limit revision and optimizer-preserving continuation. Final-stage configurations are frozen evaluation and continuation anchors; a common from-scratch recipe for both families has not been demonstrated. The shared student was trained with offline supervision; no RL fine-tuning is included.

The [v1.0.0 release](https://github.com/riemac/AnyMani/releases/tag/v1.0.0) binds the source milestone. Model weights, exact runtime snapshots, asset/cohort/pregrasp dependencies and recovery instructions are archived separately in the private Research release. These artifacts are required to recover the reported runs; a source checkout alone does not contain the experiment data. Historical run identities remain tied to their original source snapshots.
Training and fixed evaluation use `anymani.distill.rl.train_palm_rotation_mvp` and `anymani.distill.rl.evaluate_palm_rotation_mvp`. See the [RL module guide](source/anymani/anymani/distill/rl/README.md) for configuration and identity-aware entry points. The optional Research vault provides the detailed mathematical source material at `Research/总体/rl/RL 异构掌旋研究索引.md`; runtime code does not depend on the vault.

New benchmark outputs are grouped as `logs/benchmarks/<topic>/<case>/`. Existing outputs retain their historical paths; task-specific scripts belong to their owning modules, while cross-module orchestration uses topic subdirectories under `scripts/research/`.

### In-hand rl_games tasks

```bash
python scripts/rl_games/train.py \
  --task AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-v0 \
  --num_envs 4096 \
  --seed 42 \
  --headless
```

```bash
python scripts/rl_games/play.py \
  --task AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-Play-v0 \
  --num_envs 4 \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --real-time
```

### GM MLP tasks

```bash
python -m anymani.distill.rl.train \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 4096 \
  --headless
```

```bash
python -m anymani.distill.rl.play \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --checkpoint /absolute/path/to/checkpoint.pth
```

Use `scripts/random_agent.py` or `scripts/zero_agent.py` for lightweight task startup checks. These do not replace
task-specific contract tests or Isaac Sim runtime smoke tests.

### Geometry SSL

Geometry SSL is a task-free PyTorch/Warp process and does not launch Isaac Sim. The Python preset defines the method and dataset; ordinary flags define one run:

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_multitask_representation_v0_7_5 \
  --device cuda:0 \
  --seed 20260813
```

Schema 9 composes concrete `data / method / trainer / run` roles from the Python preset. Resolved configuration, dataset and expanded physical manifests, JSONL metrics, validation evidence, full resume checkpoints, and the standalone retained artifact are written under `logs/ssl/<experiment>/<UTC timestamp>/`. See `source/anymani/anymani/distill/ssl/README.md` for the physical target and lifecycle contract.

## Tests

Default pytest paths are contract-only and must not launch Isaac Sim:

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest -q
```

Simulator-backed tests live under `source/anymani/anymani/smokes/isaacsim/` and must be invoked explicitly with a
timeout. Example:

```bash
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

Code quality configuration is in `pyproject.toml`, `.pre-commit-config.yaml`, and `pytest.ini`.

## Documentation

- `AGENTS.md`: repository architecture, boundaries, and testing rules;
- `source/anymani/docs/GM_TEACHER_IMPLEMENTATION_OVERVIEW.md`: current GM implementation surface and remaining gaps;
- `source/anymani/docs/SINGLE_ASSET_COLLISION_FILTER_ABLATION.md`: generated single-asset collision-filter evidence;
- `source/anymani/docs/ISAACLAB_GUI_DRIVER_TROUBLESHOOTING.md`: GUI/driver troubleshooting;
- `source/anymani/anymani/assets/AGENTS.md`: generated asset subsystem boundaries and development rules;
- `source/anymani/anymani/distill/README.md`: distill stage status, scientific data flow, commands, and reading path;
- `source/anymani/docs/heterogeneous_ssl_prior_art/README.md`: auditable prior art for physical fields, gauge,
  morphology-conditioned policies, and temporal pretraining.
