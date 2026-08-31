r"""在显式AppLauncher内验证source-backed heterogeneous N040 provider。

该canary不创建GM environment或PhysX scene，只使用AppLauncher提供的pxr/omni运行时解析ordered source
assets、canonical artifacts与N040 static geometry evidence，随后在GPU执行两个current-q forwards。它证明
artifact/source/routing/Z闭合，不证明actor时延或PPO学习能力。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback


def _parse_args() -> argparse.Namespace:
    r"""解析有界asset count与执行设备。"""

    parser = argparse.ArgumentParser(description="Canary the frozen N040 heterogeneous geometry provider.")
    parser.add_argument("--asset_count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.asset_count <= 16:
        parser.error("--asset_count must lie in [1,16] for this bounded canary")
    return args


def main() -> int:
    r"""启动Kit、构造provider并验证当前$q$会改变unified $Z$。"""

    args = _parse_args()
    os.environ["ANYMANI_HETEROGENEOUS_ASSET_LIMIT"] = str(args.asset_count)  # 必须早于task asset runtime import

    from isaaclab.app import AppLauncher

    print("[CANARY] before AppLauncher", flush=True)
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    print("[CANARY] after AppLauncher", flush=True)
    try:
        import torch
        print("[CANARY] after torch import", flush=True)

        from anymani.distill.rl.runtime.evidence import attach_masked_runtime_evidence
        print("[CANARY] after runtime evidence import", flush=True)

        agent_cfg = {
            "params": {
                "algo": {"name": "anymani_masked_ppo"},
                "network": {
                    "name": "anymani_heterogeneous_n000_masked",
                    "retained_geometry": {
                        "artifact_path": (
                            "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched/"
                            "20260830T164445Z/retained_encoder.pt"
                        ),
                        "artifact_sha256": "cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea",
                    },
                },
                "config": {},
            }
        }
        started = time.perf_counter()  # provider materialization wall-clock起点
        print("[CANARY] before provider attach", flush=True)
        try:
            attach_masked_runtime_evidence(agent_cfg)
        except BaseException:
            traceback.print_exc()
            return 2
        print("[CANARY] after provider attach", flush=True)
        provider = agent_cfg["params"]["network"]["retained_geometry_provider"].to(args.device)
        build_seconds = time.perf_counter() - started

        rows = torch.arange(args.asset_count, device=args.device, dtype=torch.long)
        q_zero = torch.zeros(args.asset_count, 16, device=args.device)
        q_changed = q_zero.clone()
        q_changed[:, 0] = 0.2  # current physical joint perturbation，rad
        with torch.inference_mode():
            baseline = provider.resolve(rows, q_zero).geometry_entities
            changed = provider.resolve(rows, q_changed).geometry_entities
        torch.cuda.synchronize(torch.device(args.device))
        delta = float((changed - baseline).abs().max().item())
        if baseline.shape != (args.asset_count, 21, 128) or not delta > 0.0:
            raise RuntimeError(f"invalid retained provider result shape={tuple(baseline.shape)} q_delta_max={delta}")
        print(
            json.dumps(
                {
                    "asset_count": args.asset_count,
                    "device": args.device,
                    "shape": list(baseline.shape),
                    "q_delta_max": delta,
                    "build_seconds": build_seconds,
                    "identity_digest": provider.identity["identity_digest"],
                    "evidence_tensor_digest": provider.identity["evidence_tensor_digest"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
