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
| `source/anymani/anymani/distill/` | RL/IL entrypoints and shared policy/model components |

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
- `AnyMani-GM-*`: generalized-manipulation generated-hand, single-asset, LEAP, and heterogeneous spawn environments;
- `AnyMani-GM-*-MLP-v0`: GM task aliases with rl_games MLP training configuration.

Treat `scripts/list_envs.py` as the source of truth for exact IDs; research node numbers are intentionally not part of
AnyMani public names.

## Train and Play

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
python -m anymani.distill.train \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --num_envs 4096 \
  --headless
```

```bash
python -m anymani.distill.play \
  --task AnyMani-GM-SingleAsset-MLP-v0 \
  --checkpoint /absolute/path/to/checkpoint.pth
```

Use `scripts/random_agent.py` or `scripts/zero_agent.py` for lightweight task startup checks. These do not replace
task-specific contract tests or Isaac Sim runtime smoke tests.

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
- `source/anymani/anymani/assets/README`: generated asset subsystem contract and entrypoints.
