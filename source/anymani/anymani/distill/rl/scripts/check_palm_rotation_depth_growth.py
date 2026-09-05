r"""CPU-only proof of function-preserving depth growth for the Direct actor.

Added Pre-LN blocks use zero attention-output and final-FFN projections, so
each new block initially computes x + 0 + 0. Its internal features remain
nonzero/random, allowing the output projections to receive a first gradient.
This checks numerical migration on synthetic full/partial joint masks, not
learning quality or physical control. No optimizer or simulator is created.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
)


def main() -> None:
    r"""Compare the inherited function/gradients and establish that new blocks can learn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    if args.layers <= 1 or args.output.exists():
        raise ValueError("choose a larger depth and a new output path")
    torch.set_num_threads(1)
    torch.manual_seed(7301)
    document = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert document["anymani_identity"]["policy"]["arm"] == "direct_token"
    source = PalmRotationDirectActor(local_skip=False, history_encoder="tcn").eval()
    prefix = "a2c_network.package.actor."
    source.load_state_dict(
        {key[len(prefix) :]: value for key, value in document["model"].items() if key.startswith(prefix)}
    )
    grown = copy.deepcopy(source)
    config = source.global_backbone.config
    for _ in range(args.layers - 1):
        block = type(source.global_backbone.layers[0])(
            config.hidden_width, config.attention_heads, config.feedforward_width, config.dropout
        )
        output_parameters = {
            "attention_output.weight",
            "attention_output.bias",
            "feedforward.3.weight",
            "feedforward.3.bias",
        }
        initialized = 0
        for name, parameter in block.named_parameters():
            if name in output_parameters:
                torch.nn.init.zeros_(parameter)
                initialized += 1
        assert initialized == 4
        grown.global_backbone.layers.append(block)
    grown.global_backbone.config = replace(config, layers=args.layers)
    grown.eval()

    # Depth-major slots: four full fingers; 8/12 DoF four-tip cases; one empty finger.
    lengths = torch.tensor(((4, 4, 4, 4), (1, 1, 3, 3), (1, 3, 4, 4), (0, 4, 3, 4)))
    valid = (torch.arange(4)[None, :, None] < lengths[:, None, :]).reshape(4, 16)
    tips = lengths > 0
    owner_valid = torch.cat((torch.ones(4, 1, dtype=torch.bool), valid, tips), dim=1)
    contacts = (torch.rand(4, 4) > 0.5).float() * tips
    current = torch.rand(4, 16, 5) - 0.5
    current[..., 1] = current[..., 0] + 0.02 * torch.randn(4, 16)
    current[..., 3] = 0.0
    current[..., 4] = contacts.repeat(1, 4)
    current *= valid[..., None]
    history = current[:, None].repeat(1, 30, 1, 1)
    history[:, :-1, :, :3] += 0.01 * torch.randn(4, 29, 16, 3) * valid[:, None, :, None]
    owner_contact = torch.zeros(4, 21, 1)
    owner_contact[:, 17:, 0] = contacts
    observation = PalmRotationActorObservation(
        jnt_current=current,
        jnt_history=history,
        jnt_limits=torch.tensor((-0.5, 0.5)).repeat(4, 16, 1) * valid[..., None],
        owner_contact=owner_contact,
        jnt_valid=valid,
        tip_valid=tips,
        owner_valid=owner_valid,
    )
    # Legal relation buckets suffice for an algebraic invariance test; they do not
    # purport to reproduce a physical hand. Each model receives exactly the same graph.
    relations = torch.full((4, 21, 21), 2, dtype=torch.long)
    relations.diagonal(dim1=1, dim2=2).zero_()
    geometry = PalmRotationGeometry(
        tokens=torch.randn(4, 21, 128) * owner_valid[..., None],
        owner_valid=owner_valid,
        shortest_path=relations,
        parent_direction=relations.clone(),
        child_direction=relations.clone(),
    )
    original_mean = source(observation, geometry).mean
    grown_mean = grown(observation, geometry).mean
    torch.testing.assert_close(original_mean, grown_mean, rtol=0, atol=0)
    assert torch.count_nonzero(grown_mean[~valid]) == 0
    weights = torch.randn_like(original_mean)
    (original_mean * weights).sum().backward()
    (grown_mean * weights).sum().backward()
    inherited = dict(source.named_parameters())
    expanded = dict(grown.named_parameters())
    for name, parameter in inherited.items():
        torch.testing.assert_close(parameter, expanded[name], rtol=0, atol=0)
        if parameter.grad is None:
            assert expanded[name].grad is None
        else:
            torch.testing.assert_close(parameter.grad, expanded[name].grad, rtol=0, atol=0)
    new_parameters = [parameter for name, parameter in expanded.items() if name not in inherited]
    nonzero_new_gradients = sum(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in new_parameters
    )
    assert nonzero_new_gradients > 0
    result = {
        "scope": "synthetic CPU function/gradient migration, not a learning or simulator result",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_parameters": sum(parameter.numel() for parameter in source.parameters()),
        "grown_parameters": sum(parameter.numel() for parameter in grown.parameters()),
        "source_layers": 1,
        "grown_layers": args.layers,
        "active_dof_cases": valid.sum(dim=1).tolist(),
        "tip_count_cases": tips.sum(dim=1).tolist(),
        "action_max_abs_difference": float((original_mean - grown_mean).detach().abs().max()),
        "inherited_parameter_and_gradient_equality": True,
        "new_nonzero_gradient_tensors": nonzero_new_gradients,
        "optimizer_steps": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
