r"""16个八组balanced source rows上的真实N040完整actor性能canary。"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    r"""解析每个固定morphology cell的asset数量。"""

    parser = argparse.ArgumentParser(description="Benchmark a balanced heterogeneous N040 full actor.")
    parser.add_argument("--rows_per_cell", type=int, choices=(2, 16), default=2)
    return parser.parse_args()


def _balanced_rows(rows_per_cell: int) -> tuple[int, ...]:
    r"""从formal 2048 group manifest按handedness×TIP×thumb-DoF取八组平衡rows。"""

    manifest = Path(
        "outputs/canonical_runtime/v1/groups/"
        "22320b8e00c8df73699d20cd0f56e68fbab720e8d63b36bba898ee964334a8dc/train-2048.json"
    )
    artifacts = json.loads(manifest.read_text())["artifacts"]
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for dataset_row, artifact in enumerate(artifacts):
        routing = artifact["routing"]
        key = (
            str(routing["handedness"]),
            sum(bool(value) for value in routing["active_tip_mask"]),
            sum(bool(routing["active_joint_mask"][index]) for index in (3, 7, 11, 15)),
        )
        groups[key].append(dataset_row)
    return tuple(row for key in sorted(groups) for row in groups[key][:rows_per_cell])


def main() -> int:
    r"""构造16-row真实evidence，以B=4096 round-robin q/history测完整actor。"""

    args = _parse_args()
    balanced_rows = _balanced_rows(args.rows_per_cell)
    os.environ["ANYMANI_HETEROGENEOUS_ASSET_ROWS"] = ",".join(str(row) for row in balanced_rows)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        import torch
        from anymani.distill.rl.heterogeneous_masked_ppo import (
            HETEROGENEOUS_N040_HISTORY_OBS_DIM,
            HeterogeneousN040HistoryPpoBuilder,
        )
        from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
        from anymani.distill.rl.runtime.evidence import attach_masked_runtime_evidence

        agent_cfg = {
            "params": {
                "algo": {"name": "anymani_masked_ppo"},
                "network": {
                    "name": "anymani_heterogeneous_n040_history30",
                    "retained_geometry": {
                        "artifact_path": (
                            "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched/"
                            "20260830T164445Z/retained_encoder.pt"
                        ),
                        "artifact_sha256": "cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea",
                    },
                    "parallel_geometry_temporal": True,
                    "compile_policy_adapter": True,
                    "temporal_encoder": "stack_mlp",
                    "heterogeneous_policy": {
                        "owner_feature_dim": 1,
                        "joint_feature_dim": 6,
                        "temporal_feature_dim": 32,
                        "geometry_entity_width": 128,
                        "hidden_width": 128,
                        "layers": 1,
                        "attention_heads": 4,
                        "feedforward_width": 256,
                        "dropout": 0.0,
                        "initial_log_std": -0.5,
                    },
                },
                "config": {},
            }
        }
        build_started = time.perf_counter()
        attach_masked_runtime_evidence(agent_cfg)
        provider = agent_cfg["params"]["network"]["retained_geometry_provider"]
        build_seconds = time.perf_counter() - build_started
        builder = HeterogeneousN040HistoryPpoBuilder()
        builder.load(agent_cfg["params"]["network"])
        model = AnyManiMaskedContinuousModel(builder).build(
            {
                "actions_num": 16,
                "input_shape": (HETEROGENEOUS_N040_HISTORY_OBS_DIM,),
                "value_size": 1,
                "normalize_input": False,
                "normalize_value": False,
            }
        ).to("cuda:0")
        batch_size = 4096
        obs = torch.zeros(batch_size, HETEROGENEOUS_N040_HISTORY_OBS_DIM, device="cuda:0")
        history = obs[:, : 30 * 16 * 4].reshape(batch_size, 30, 16, 4)
        history[:, :, :, :3].normal_(mean=0.0, std=0.2)
        history[:, :, :, 3].bernoulli_(0.4)
        limits = obs[:, 30 * 16 * 4 : 30 * 16 * 4 + 32].reshape(batch_size, 16, 2)
        limits[:, :, 0] = -1.0
        limits[:, :, 1] = 1.0
        obs[:, -17] = torch.arange(batch_size, device="cuda:0") % len(provider.asset_ids)
        masks = provider.evidence_joint_valid_mask[obs[:, -17].long()]
        obs[:, -16:] = masks
        network = model.a2c_network.eval()

        def profile_cuda(callable_) -> dict[str, float]:
            r"""以10 warmups + 30 CUDA Events报告component latency。"""

            with torch.inference_mode():
                for _ in range(10):
                    callable_()
                torch.cuda.synchronize()
                samples = []
                stream = torch.cuda.current_stream()
                for _ in range(30):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    callable_()
                    end.record(stream)
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
            ordered_component = sorted(samples)
            return {
                "median_ms": statistics.median(ordered_component),
                "p95_ms": ordered_component[math.ceil(0.95 * len(ordered_component)) - 1],
            }

        asset_rows = obs[:, -17].long()
        latest_q_rad = history[:, -1, :, 0] * torch.pi
        provider_profile = profile_cuda(lambda: provider.resolve(asset_rows, latest_q_rad))
        policy_input = network._build_history_policy_input(obs)
        policy_profile = profile_cuda(lambda: network._policy_forward(policy_input))
        with torch.inference_mode():
            for _ in range(20):
                network({"obs": obs})
            torch.cuda.synchronize()
            samples = []
            stream = torch.cuda.current_stream()
            for _ in range(50):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                network({"obs": obs})
                end.record(stream)
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
        ordered = sorted(samples)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        summary = {
            "asset_count": len(provider.asset_ids),
            "batch_size": batch_size,
            "build_seconds": build_seconds,
            "median_ms": statistics.median(ordered),
            "p95_ms": p95,
            "max_ms": ordered[-1],
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
            "identity_digest": provider.identity["identity_digest"],
            "provider_profile": provider_profile,
            "compiled_policy_profile": policy_profile,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if p95 >= 48.0:
            raise RuntimeError(
                f"balanced {len(provider.asset_ids)}-row full actor p95={p95:.3f} ms violates strict <48 ms gate"
            )
        return 0
    except BaseException:
        traceback.print_exc()
        return 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
