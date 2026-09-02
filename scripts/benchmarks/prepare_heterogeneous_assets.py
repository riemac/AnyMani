#!/usr/bin/env python3
r"""离线准备generated heterogeneous PPO partition的canonical artifacts。

该入口启动 AppLauncher 以满足 robots adapter 的 IsaacLab cfg imports，但不创建 environment、Stage
scene 或 PhysX view；只执行：``ppo.yaml train resolve``、canonical URDF/manifest materialization、
ordered manifest digest 与全局startup-pose计算。训练与预热共享同一``outputs/canonical_runtime/v1`` artifact cache，
因此可用父recorder分别测首次准备和cache-hit准备；本入口不查询pregrasp cache。
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    r"""解析可选 smoke prefix；默认完整 2048。"""

    parser = argparse.ArgumentParser(description="Prepare AnyMani heterogeneous canonical assets.")
    parser.add_argument("--max_assets", type=int, default=None, help="Optional ordered train prefix for diagnostics.")
    args = parser.parse_args()
    if args.max_assets is not None and args.max_assets < 1:
        parser.error("--max_assets must be positive")
    return args


def main() -> int:
    r"""物化并打印有序资产、group manifest 与 boot-pose 摘要。"""

    args = _parse_args()
    from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase

    record_optional_rl_phase("app_launcher", "start", headless=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _simulation_app = app_launcher.app  # 保持 Kit app 生命周期覆盖全部 IsaacLab cfg imports
    record_optional_rl_phase("app_launcher", "complete")
    from anymani.distill.rl.runtime.structured_geometry import canonical_group_manifest_digest
    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

    rows = None if args.max_assets is None else tuple(range(args.max_assets))
    binding = build_generated_asset_binding(rows)
    group_manifest_digest = canonical_group_manifest_digest(binding.canonical_artifacts)

    print(
        {
            "asset_count": len(binding.canonical_artifacts),
            "first_asset_id": binding.canonical_artifacts[0].asset_id,
            "last_asset_id": binding.canonical_artifacts[-1].asset_id,
            "group_manifest_digest": group_manifest_digest,
            "startup_joint_pos": dict(binding.hand_spawn_cfg.joint_init.joint_pos),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
