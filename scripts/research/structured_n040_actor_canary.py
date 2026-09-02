r"""Balanced16 N040 + structured actor candidates + scalar critic correctness/performance canary。

计时从GPU-resident structured tensors开始，覆盖每step current-q provider与learned actor，不含Isaac/ContactSensor/H2D。
N040 static frontend可缓存，q-dependent final Z每次重算。B=4096使用20 warmup+50 CUDA events和严格p95<48 ms门。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import traceback
from pathlib import Path
from typing import cast

BALANCED_16_ROWS = (416, 417, 352, 353, 0, 1, 64, 65, 432, 433, 368, 369, 16, 17, 80, 81)
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in BALANCED_16_ROWS)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _parse_args() -> argparse.Namespace:
    r"""解析device、batch sizes与durable output。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", default="16,128,4096")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hetero/performance/structured-n040-actor-candidates.json"),
    )
    return parser.parse_args()


def _timing(torch, fn, *, warmup: int, samples: int) -> dict[str, float]:
    r"""用独立CUDA events测量一个GPU-resident callable。"""

    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            end.synchronize()
            elapsed.append(float(start.elapsed_time(end)))
    ordered = sorted(elapsed)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "median_ms": statistics.median(elapsed),
        "p95_ms": p95,
        "max_ms": max(elapsed),
        "warmup": warmup,
        "samples": samples,
    }


def main() -> int:
    r"""构造provider/candidates并运行correctness、timing与checkpoint probes。"""

    args = _parse_args()
    import torch
    from anymani.distill.models.heterogeneous_policy import (
        StructuredActorCfg,
        StructuredActorCriticPackage,
        StructuredCriticCfg,
        StructuredHeterogeneousActor,
    )
    from anymani.distill.models.structured_heterogeneous import (
        GeometryTokenBatch,
        StructuredActorObservation,
        StructuredCriticObservation,
    )
    from anymani.distill.rl.runtime.structured_geometry import build_structured_retained_geometry_provider
    from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("structured N040 performance canary requires CUDA")
    device = torch.device(args.device)
    batches = tuple(int(value) for value in args.batches.split(","))
    if not batches or any(value < 1 for value in batches):
        raise ValueError("batches must contain positive integers")
    binding = build_generated_asset_binding(BALANCED_16_ROWS)
    provider = build_structured_retained_geometry_provider(binding, device=device)
    if any(parameter.requires_grad for parameter in provider.parameters()) or provider.encoder.training:
        raise AssertionError("N040 provider must remain frozen eval")

    candidate_cfgs = {
        "local_d128": StructuredActorCfg(hidden_width=128, temporal_width=32, coordination="local"),
        "gated_pool_d128": StructuredActorCfg(hidden_width=128, temporal_width=32, coordination="gated_pool"),
        "cross_attention_d96": StructuredActorCfg(
            hidden_width=96, temporal_width=32, coordination="cross_attention", attention_heads=4
        ),
        "gated_pool_d64": StructuredActorCfg(hidden_width=64, temporal_width=32, coordination="gated_pool"),
    }
    candidates: dict[str, StructuredHeterogeneousActor] = {}
    for name, cfg in candidate_cfgs.items():
        actor = StructuredHeterogeneousActor(cfg)
        actor.to(device)
        actor.eval()
        candidates[name] = actor
    package = StructuredActorCriticPackage(
        actor_cfg=candidate_cfgs["gated_pool_d128"], critic_cfg=StructuredCriticCfg(hidden_width=128)
    )
    package.to(device)
    package.eval()
    runtime = StructuredHeterogeneousRuntime(provider, package)
    runtime.to(device)
    runtime.eval()
    actor_ids, critic_ids = package.trainable_parameter_sets()
    if not actor_ids.isdisjoint(critic_ids):
        raise AssertionError("actor and critic parameters overlap")
    q_home_bank = cast(torch.Tensor, getattr(provider, "evidence_q_home"))
    joint_valid_bank = cast(torch.Tensor, getattr(provider, "evidence_joint_valid_mask"))
    owner_valid_bank = cast(torch.Tensor, getattr(provider, "evidence_entity_valid_mask"))

    def inputs(batch: int):
        r"""从provider static q-home/masks构造GPU-resident structured tensors。"""

        route = torch.arange(batch, device=device, dtype=torch.long) % len(BALANCED_16_ROWS)
        q_home = q_home_bank[route]
        joint_valid = joint_valid_bank[route]
        owner_valid = owner_valid_bank[route]
        tip_valid = owner_valid[:, 17:21]
        q_rad = q_home + 0.03 * torch.sin(torch.arange(16, device=device).float()).unsqueeze(0) * joint_valid
        current = torch.stack((q_rad / torch.pi, q_home / torch.pi, torch.zeros_like(q_rad)), dim=-1)
        history_frame = torch.cat((current, torch.zeros(batch, 16, 1, device=device)), dim=-1)
        history = history_frame.unsqueeze(1).expand(-1, 30, -1, -1).contiguous()
        limits = torch.stack((torch.full_like(q_rad, -1.0), torch.full_like(q_rad, 1.0)), dim=-1)
        tip_contact = ((torch.arange(batch, device=device)[:, None] + torch.arange(4, device=device)) % 2).float()
        actor_observation = StructuredActorObservation(
            current,
            history,
            limits,
            tip_contact.unsqueeze(-1),
            joint_valid,
            tip_valid,
            owner_valid,
        )
        critic_observation = StructuredCriticObservation(
            torch.cat((current, torch.zeros(batch, 16, 1, device=device)), dim=-1),
            torch.zeros(batch, 21, 2, device=device),
            torch.zeros(batch, 1, 15, device=device),
            torch.zeros(batch, 1, 8, device=device),
            joint_valid,
            tip_valid,
            owner_valid,
        )
        return route, actor_observation, critic_observation

    results: dict[str, object] = {
        "artifact_type": "anymani.hetero.structured_n040_actor_performance",
        "schema_version": "1.0.0",
        "device": str(device),
        "dataset_rows": list(BALANCED_16_ROWS),
        "provider_identity": provider.identity,
        "frozen_geometry_parameters": sum(parameter.numel() for parameter in provider.parameters()),
        "actor_parameters": {name: sum(parameter.numel() for parameter in actor.parameters()) for name, actor in candidates.items()},
        "critic_parameters": sum(parameter.numel() for parameter in package.critic.parameters()),
        "batches": {},
    }
    torch.cuda.reset_peak_memory_stats(device)
    for batch in batches:
        route, actor_observation, critic_observation = inputs(batch)
        q_rad = actor_observation.jnt_current[..., 0] * torch.pi
        with torch.inference_mode():
            context = runtime.resolve_geometry(route, actor_observation)
            if context.tokens.tokens.requires_grad or context.tokens.tokens.shape != (batch, 21, 128):
                raise AssertionError("invalid frozen N040 structured output")
            for actor in candidates.values():
                output = actor(actor_observation, context.tokens)
                if output.mean.shape != (batch, 16) or output.log_std.numel() != 1:
                    raise AssertionError("structured actor output contract failed")
            if runtime.critic_forward(critic_observation, context).value.shape != (batch,):
                raise AssertionError("structured critic scalar contract failed")
        warmup, samples = (20, 50) if batch == 4096 else (10, 30)
        batch_result: dict[str, object] = {
            "provider_raw": _timing(
                torch,
                lambda: provider.resolve(route, q_rad),
                warmup=warmup,
                samples=samples,
            ),
            "provider_validated": _timing(
                torch,
                lambda: runtime.resolve_geometry(route, actor_observation),
                warmup=warmup,
                samples=samples,
            ),
            "actor_only": {},
        }
        for name, actor in candidates.items():
            batch_result["actor_only"][name] = _timing(  # type: ignore[index]
                torch,
                lambda actor=actor: actor(actor_observation, context.tokens),
                warmup=warmup,
                samples=samples,
            )
        batch_result["critic_only"] = _timing(
            torch,
            lambda: runtime.critic_forward(critic_observation, context),
            warmup=warmup,
            samples=samples,
        )
        batch_result["full_gated_actor"] = _timing(
            torch,
            lambda: runtime.actor_forward(actor_observation, runtime.resolve_geometry(route, actor_observation)),
            warmup=warmup,
            samples=samples,
        )
        if batch == 4096:
            try:
                class ProviderTokens(nn.Module):
                    r"""Compile boundary只返回q-dependent$Z$，不缓存最终activation。"""

                    def __init__(self, retained_provider) -> None:
                        super().__init__()
                        self.retained_provider = retained_provider

                    def forward(self, rows, q):
                        return self.retained_provider.resolve(rows, q).geometry_entities

                provider_tokens = ProviderTokens(provider).to(device).eval()
                compiled_provider = torch.compile(provider_tokens, mode="reduce-overhead", fullgraph=False)
                compiled_actor = torch.compile(
                    candidates["gated_pool_d128"], mode="reduce-overhead", fullgraph=False
                )

                def compiled_full_actor():
                    torch.compiler.cudagraph_mark_step_begin()
                    tokens = compiled_provider(route, q_rad)
                    geometry = GeometryTokenBatch(tokens, actor_observation.owner_valid)
                    return compiled_actor(actor_observation, geometry)

                with torch.inference_mode():
                    eager_tokens = provider.resolve(route, q_rad).geometry_entities
                    eager_mean = candidates["gated_pool_d128"](
                        actor_observation, GeometryTokenBatch(eager_tokens, actor_observation.owner_valid)
                    ).mean
                    torch.compiler.cudagraph_mark_step_begin()
                    compiled_tokens = compiled_provider(route, q_rad)
                    compiled_tokens_snapshot = compiled_tokens.clone()
                    compiled_mean = compiled_actor(
                        actor_observation, GeometryTokenBatch(compiled_tokens, actor_observation.owner_valid)
                    ).mean.clone()
                torch.cuda.synchronize(device)
                z_difference = float((compiled_tokens_snapshot - eager_tokens).abs().max().item())
                mean_difference = float((compiled_mean - eager_mean).abs().max().item())
                if z_difference > 1.0e-5 or mean_difference > 1.0e-5:
                    raise RuntimeError(
                        "compiled structured path changes outputs: "
                        f"z_max_abs={z_difference}, mean_max_abs={mean_difference}"
                    )
                batch_result["compiled_numerical_equivalence"] = {
                    "z_max_abs": z_difference,
                    "mean_max_abs": mean_difference,
                    "atol": 1.0e-5,
                }

                batch_result["compiled_provider_tokens"] = _timing(
                    torch,
                    lambda: compiled_provider(route, q_rad),
                    warmup=warmup,
                    samples=samples,
                )
                batch_result["compiled_gated_actor_only"] = _timing(
                    torch,
                    lambda: compiled_actor(actor_observation, context.tokens),
                    warmup=warmup,
                    samples=samples,
                )
                batch_result["full_compiled_gated_actor"] = _timing(
                    torch,
                    compiled_full_actor,
                    warmup=warmup,
                    samples=samples,
                )
            except Exception as exc:
                batch_result["compile_error"] = f"{type(exc).__name__}: {exc}"
        results["batches"][str(batch)] = batch_result  # type: ignore[index]

    # Current-q influence and checkpoint namespace are correctness facts outside timing loops.
    route, actor_observation, _ = inputs(min(batches))
    with torch.inference_mode():
        baseline = runtime.resolve_geometry(route, actor_observation).tokens.tokens
        changed_current = actor_observation.jnt_current.clone()
        changed_current[:, 0, 0] += 0.05 / torch.pi
        changed_observation = StructuredActorObservation(
            changed_current,
            actor_observation.jnt_history,
            actor_observation.jnt_limits,
            actor_observation.tip_contact,
            actor_observation.jnt_valid,
            actor_observation.tip_valid,
            actor_observation.owner_valid,
        )
        changed = runtime.resolve_geometry(route, changed_observation).tokens.tokens
    results["q_influence_max_abs"] = float((changed - baseline).abs().max().item())
    state_keys = tuple(runtime.state_dict())
    results["checkpoint_namespaces"] = {
        "geometry_provider": sum(key.startswith("geometry_provider.") for key in state_keys),
        "actor": sum(key.startswith("policy.actor.") for key in state_keys),
        "critic": sum(key.startswith("policy.critic.") for key in state_keys),
    }
    results["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    full_4096 = results["batches"].get("4096", {})  # type: ignore[union-attr]
    eager_p95 = full_4096.get("full_gated_actor", {}).get("p95_ms") if isinstance(full_4096, dict) else None
    compiled_p95 = (
        full_4096.get("full_compiled_gated_actor", {}).get("p95_ms") if isinstance(full_4096, dict) else None
    )
    available_p95 = [value for value in (eager_p95, compiled_p95) if isinstance(value, float)]
    p95 = min(available_p95) if available_p95 else None
    results["strict_full_actor_gate_passed"] = isinstance(p95, float) and p95 < 48.0
    results["selected_full_actor_p95_ms"] = p95
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "full_gated_actor_4096_p95_ms": p95,
                "eager_full_actor_4096_p95_ms": eager_p95,
                "compiled_full_actor_4096_p95_ms": compiled_p95,
                "strict_gate_passed": results["strict_full_actor_gate_passed"],
                "peak_allocated_MiB": results["peak_allocated_MiB"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
