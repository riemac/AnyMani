#!/usr/bin/env python3
r"""离线准备 heterogeneous PPO train partition 的 canonical artifacts。

该入口启动 AppLauncher 以满足 robots adapter 的 IsaacLab cfg imports，但不创建 environment、Stage
scene 或 PhysX view；只执行：``ppo.yaml train resolve``、canonical URDF/manifest materialization、
group manifest 与全局 startup-pose 计算。训练与预热共享同一
``outputs/canonical_runtime/v1``，因此可用父 recorder 分别测首次准备和 cache-hit 准备。
"""

from __future__ import annotations

import argparse
import os


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
    if args.max_assets is None:
        os.environ.pop("ANYMANI_HETEROGENEOUS_ASSET_LIMIT", None)
    else:
        os.environ["ANYMANI_HETEROGENEOUS_ASSET_LIMIT"] = str(args.max_assets)
    from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase

    record_optional_rl_phase("app_launcher", "start", headless=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _simulation_app = app_launcher.app  # 保持 Kit app 生命周期覆盖全部 IsaacLab cfg imports
    record_optional_rl_phase("app_launcher", "complete")
    from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
        HETEROGENEOUS_CANONICAL_ARTIFACTS,
        HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
        HETEROGENEOUS_GROUP_MANIFEST_PATH,
        HETEROGENEOUS_HAND_SPAWN_CFG,
    )

    print(
        {
            "asset_count": len(HETEROGENEOUS_CANONICAL_ARTIFACTS),
            "first_asset_id": HETEROGENEOUS_CANONICAL_ARTIFACTS[0].asset_id,
            "last_asset_id": HETEROGENEOUS_CANONICAL_ARTIFACTS[-1].asset_id,
            "group_manifest": str(HETEROGENEOUS_GROUP_MANIFEST_PATH),
            "group_manifest_digest": HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
            "startup_joint_pos": dict(HETEROGENEOUS_HAND_SPAWN_CFG.joint_init.joint_pos),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
