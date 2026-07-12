# GM Task and Teacher Implementation Overview

This document summarizes the current source-implemented surface of `tasks/gm` and `distill`. It distinguishes code
that exists from behavior that has been proven by an Isaac Sim smoke or a training run.

## Current Status

GM is no longer a zero-reward or unbound-scene scaffold. The repository contains generated-hand and LEAP scene
bindings, executable reorientation commands and rewards, contact sensors, single-asset MLP training aliases, and a
minimal heterogeneous spawn environment.

The complete morphology-token teacher/distillation design is not implemented. In particular, there is no grasp-cache
reset pipeline, active hand-orientation reset event, full PALM/JOINT/TIP policy, or end-to-end heterogeneous teacher
training route.

## Registered Task Surfaces

### Task-owned environments

- `AnyMani-GM-InHand-v0` / `-Play-v0`: same-topology generated-hand GM assembly;
- `AnyMani-GM-SingleAsset-v0` / `-Play-v0`: generated mother-asset probe;
- `AnyMani-GM-Leap-v0` / `-Play-v0`: official LEAP URDF comparison;
- `AnyMani-GM-Heterogeneous-Test-v0`: three-hand spawn/reset/step test without the full manipulation MDP.

Registrations live in `source/anymani/anymani/tasks/gm/__init__.py`.

### Training aliases

- `AnyMani-GM-SingleAsset-MLP-v0`;
- `AnyMani-GM-Leap-MLP-v0`.

These aliases bind task configs to `distill/rl/agents/gm_single_asset_mlp_ppo.yaml` and are consumed by
`python -m anymani.distill.train` / `play`. No active alias trains the generic multi-asset GM environment with the
planned morphology-token model.

## Implemented Environment Semantics

### Asset and scene binding

`robots/hand_spawn.py` resolves `HandBankCfg`, validates same-schema selections, composes semantic hand anchors, and
lowers generated URDF assets to Isaac Lab `ArticulationCfg`. GM configs use this path for generated assets; the LEAP
variant uses `robots/leap_urdf.py`.

Environment configs may declare reproducible asset selection and routing because those values determine scene
construction. Asset generation and long-lived train/validation split policy remain outside GM MDP terms.

### Action and observation

The generic GM environment uses a current-joint-relative clamped action. Single-asset and LEAP probes use Isaac Lab
relative joint-position actions with variant-specific scales. ADR relative/EMA actions exist as reusable components but
are not selected by the current GM configs.

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

Generic GM uses joint/object reset events and records an object reset anchor. Single-asset and LEAP variants use fixed
grasp presets with object-yaw randomization. There is no grasp-cache loader/sampler in the current source.

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

The active GM training aliases use an rl_games MLP. A minimal flat-layout Transformer adapter exists, but no current YAML
selects it and its declared observation layout does not cover the complete generic GM policy group.

The planned PALM/JOINT/TIP tokenizer, relation builder, hybrid SE(3) attention bias, shared backbone, action/value heads,
IL loop, student, and distillation pipeline remain design targets rather than an executable stack.

## Validation Evidence

Default tests under `tasks/gm/tests` and `distill/tests` validate tensor math, configs, frame semantics, contact layout,
reward ownership, collision pairs, and model shapes without launching Isaac Sim.

The current GM runtime evidence is
`source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py`. It constructs two generated
single-asset environments, checks authored PhysX filtered pairs, and executes random steps with finite tensors. It does
not prove grasp quality, contact values, all reward terms, LEAP runtime, or a full training run.

## Remaining Closure Work

1. Add runtime smokes for command resampling, contact values, reward components, and LEAP GM.
2. Decide and implement grasp-cache ownership before claiming cache-driven reset.
3. Implement hand-orientation reset only after its frame/reference distribution is fixed and tested.
4. Reconcile the generic GM observation schema with a selected network adapter before heterogeneous teacher training.
5. Implement the morphology-token and distillation stack incrementally, without treating design dataclasses as working
   capability.
