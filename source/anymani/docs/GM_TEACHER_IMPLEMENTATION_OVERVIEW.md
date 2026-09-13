# GM Task and Teacher Implementation Overview

This document summarizes the current source-implemented surface of `tasks/gm` and `distill`. It distinguishes code
that exists from behavior that has been proven by an Isaac Sim smoke or a training run.

## Current Status

GM contains generated single-asset and LEAP scene bindings, executable reorientation commands/rewards, contact sensors,
and single-asset rl_games aliases. Cross-topology generated-hand training is owned by `tasks/hetero` plus the structured
model/PPO implementation in `distill`.

The generated heterogeneous route now has an identity-keyed pregrasp cache, fail-closed reset, structured
PALM/JOINT/TIP observations, a frozen N040 geometry provider, separate actor/critic, and bounded direct PPO. Arbitrary
hand-orientation reset, full 2048-asset contact coverage, gravity-robust pregrasps, and end-to-end distillation remain out
of scope.

## Registered Task Surfaces

### Task-owned environments

- `AnyMani-GM-SingleAsset-v0` / `-Play-v0`: generated mother-asset probe;
- `AnyMani-GM-Leap-v0` / `-Play-v0`: official LEAP URDF comparison;
- `AnyMani-Hetero-Generated-TactileRotation-v0`: generated cross-topology structured tactile-rotation task.

Registrations live in the corresponding `tasks/gm/__init__.py` and `tasks/hetero/__init__.py` packages. GM no longer
registers same-topology/canonical multi-asset compatibility IDs.

### Training aliases

- `AnyMani-GM-SingleAsset-MLP-v0`;
- `AnyMani-GM-Leap-MLP-v0`.

These aliases bind task configs to `distill/rl/agents/gm_single_asset_mlp_ppo.yaml` and are consumed by
`python -m anymani.distill.rl.train` / `python -m anymani.distill.rl.play`. Heterogeneous bounded experiments use
`scripts/research/train_hetero_structured_ppo.py` directly, so task identity and reset tier remain explicit.

## Implemented Environment Semantics

### Asset and scene binding

`robots/hand_spawn.py` resolves `HandBankCfg`, validates same-schema selections, composes semantic hand anchors, and
lowers generated URDF assets to Isaac Lab `ArticulationCfg`. GM configs use this path for generated assets; the LEAP
variant uses `robots/leap_urdf.py`.

`tasks/hetero/config/generated/asset_binding.py` declares reproducible dataset selection, physical identity and
round-robin routing. Asset generation and long-lived train/validation split policy remain outside task MDP terms.

### Action and observation

Single-asset and LEAP probes use Isaac Lab relative joint-position or explicit ADR actions with variant-specific scales.
The heterogeneous task uses a preload-aware masked relative target that advances at most $1/24$ rad once per 20 Hz
policy step and holds that target across six 120 Hz physics substeps.

Observation contracts differ by variant. Implemented reusable terms include raw/normalized joint state, previous action,
joint limits, sidecar-derived contact signals, hand-relative object position, and object orientation encoded as rot6d,
quaternion, axis-angle, or matrix. Representation is a task-level contract, not a global priority.

### Command, reward, and curriculum

`ReorientCommand` is executable. It owns fixed/random hand-frame axes, world-frame quaternion goals, local orientation
errors, success counts, axis progress, subgoal resampling, and goal-marker visualization.

Active reward components include:

- six-keypoint orientation tracking;
- axis progress and SO(3)-based success bonus;
- fingertip-object good contact;
- non-tip-object contact penalty;
- curriculum-gated action and action-rate regularization.

The global curriculum reads command-owned success statistics. Some placeholder reward callables remain exported for
future work but are not the active single-asset/LEAP reward path.

### Reset, contact, and termination

Single-asset and LEAP variants use fixed grasp presets with object-yaw randomization. The heterogeneous task resolves an
exact schema-2.1 basin record by physical/cube/scale/physics/search identity and writes separate actual joint state,
controller preload target and hand-frame object pose; cache miss or tier mismatch fails closed before PhysX writes.

Generated contact topology is derived from `hand.yaml`: one filtered ContactSensor is installed per tip/non-tip link and
filtered to the manipulated object. The generated single-asset variant also authors structural collision filtering for
palm-finger and same-finger pairs while retaining cross-finger collision.

Terminations currently cover timeout and object displacement from the reset anchor. Hand-orientation randomization is a
declarative config scaffold only; no active reset event samples and writes a new hand root orientation.

## Rotation Representation Contract

- Use $R\in SO(3)$ and $T\in SE(3)$ to define frame composition and calibration.
- Use $\log(R)^{\vee}$ only as a local residual with an explicit principal branch and an observable/deterministic
  reference update.
- Isaac Lab `(w,x,y,z)` quaternion buffers are valid runtime state and may be used directly for composition.
- Policy features may use rot6d, matrix, quaternion, or local log when the config documents frame, reference, sign or
  branch handling, and Markov information.
- A moving reference hidden from the policy causes partial observability; relative representation itself is not
  inherently nonstationary.

## Model and Distillation Surface

The active GM aliases use rl_games MLP/GRU/TCN networks. The heterogeneous route consumes named role tensors, computes
q-dependent frozen N040 geometry, uses a shared per-joint actor head and a fully separate masked-pooling scalar critic,
and applies active-joint Gaussian/GAE/clipped PPO without flattening task semantics.

## Validation Evidence

Default tests under `tasks/gm/tests` and `distill/tests` validate tensor math, configs, frame semantics, contact layout,
reward ownership, collision pairs, and model shapes without launching Isaac Sim.

Single-asset runtime evidence remains
`source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py`. Heterogeneous evidence is produced
by `hetero_structured_env_smoke.py`, `hetero_structured_network_smoke.py` and the matched PPO artifacts. The current
row16 comparison completed no subgoal or full turn; runtime correctness therefore does not imply rotation capability.

## Remaining Closure Work

1. Add runtime smokes for remaining single-asset/LEAP command and contact boundaries.
2. Implement hand-orientation reset only after its frame/reference distribution is fixed and tested.
3. Expand heterogeneous pregrasp coverage only when a bounded cohort produces a positive learning signal.
4. Test contact action-sequence recoverability and actor object-orientation observability before increasing PPO scale.
5. Implement IL/student distillation separately; do not infer it from the working structured PPO infrastructure.
