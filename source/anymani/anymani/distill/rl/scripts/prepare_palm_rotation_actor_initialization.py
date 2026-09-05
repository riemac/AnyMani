r"""Prepare an initialization-only policy artifact with an explicit exploration reset.

Only the global latent Normal standard deviation changes. Actor mean weights,
critic and value statistics are copied exactly. No optimizer, environment or
training counters are exported, so this artifact is for --actor_init_checkpoint,
not for full PPO --checkpoint resume. The inherited mean policy remains intact;
whether broader exploration discovers useful skills is an experiment, not a
consequence of this conversion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from anymani.distill.rl.runtime.palm_rotation_warm_start import inspect_actor_init_checkpoint


def main() -> None:
    r"""Export validated initial weights while keeping source training provenance separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent_std", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.latent_std) or args.latent_std <= 0.0:
        raise ValueError("latent standard deviation must be finite and positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = copy.deepcopy(source["anymani_identity"])
    source_training = identity["training"]
    if args.latent_std > math.exp(float(source_training["max_log_std"])):
        raise ValueError("requested exploration exceeds the declared policy ceiling")
    key = "a2c_network.package.actor.global_log_std"
    model = {name: value.detach().clone() for name, value in source["model"].items()}
    if model[key].numel() != 1:
        raise ValueError("exploration reset requires the global-scalar policy contract")
    previous_std = float(model[key].exp())
    model[key].fill_(math.log(args.latent_std))
    assert all(torch.equal(value, source["model"][name]) for name, value in model.items() if name != key)
    assert all(bool(torch.isfinite(value).all()) for value in model.values())

    # This is a parameter-initialization stage, not a fabricated extra PPO update.
    identity["training"] = {
        "stage": "actor-initialization-only",
        "history_encoder": source_training["history_encoder"],
        "cohort_id": source_training.get("cohort_id"),
        "actor_contact": source_training["actor_contact"],
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_epoch": int(source["epoch"]),
        "source_frame": int(source["frame"]),
        "source_training_contract": source_training,
        "exploration_reset": {"previous_latent_std": previous_std, "initial_latent_std": args.latent_std},
    }
    identity.pop("identity_digest")
    identity["identity_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    artifact = {
        "artifact_type": "anymani.palm_rotation.actor_initialization",
        "schema_version": "1.0.0",
        "purpose": "actor-init-only-no-optimizer-state",
        "model": model,
        "anymani_identity": identity,
    }
    with args.output.open("xb") as stream:
        torch.save(artifact, stream)
    accepted = inspect_actor_init_checkpoint(
        args.output,
        target_arm=identity["policy"]["arm"],
        target_history_encoder=source_training["history_encoder"],
        target_provider_identity=identity["geometry_provider"],
        initialize_critic=True,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "source_epoch": int(source["epoch"]),
                "previous_latent_std": previous_std,
                "initial_latent_std": float(model[key].exp()),
                "unchanged_mean_critic_and_value_tensors": True,
                "initialization_loader_accepted": bool(accepted),
                "optimizer_state_exported": False,
            }
        )
    )


if __name__ == "__main__":
    main()
