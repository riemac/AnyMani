#!/usr/bin/env python3
r"""使用正式 heterogeneous child cfg 顺序预热 IsaacLab URDF->USD cache。

该入口启动 Kit/URDF importer，但不创建 USD Stage scene、PhysX view、object 或 ContactSensor。每个 child
``UrdfFileCfg`` 与正式训练由同一个 ``HandSpawnAdapter`` 构造，因此 miss 生成的 USD 会被后续训练直接命中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast


def _parse_args() -> argparse.Namespace:
    r"""解析有序 smoke prefix 与进度输出频率。"""

    parser = argparse.ArgumentParser(description="Prewarm AnyMani heterogeneous URDF-to-USD cache.")
    parser.add_argument("--max_assets", type=int, default=None, help="Optional ordered train prefix; default 2048.")
    parser.add_argument("--progress_every", type=int, default=32, help="Print one progress row every N assets.")
    args = parser.parse_args()
    if args.max_assets is not None and args.max_assets < 1:
        parser.error("--max_assets must be positive")
    if args.progress_every < 1:
        parser.error("--progress_every must be positive")
    return args


def main() -> int:
    r"""顺序触发 converter lazy hit/miss，并输出创建/复用计数。"""

    args = _parse_args()
    from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase

    record_optional_rl_phase("app_launcher", "start", headless=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _simulation_app = app_launcher.app  # 保持 URDF importer extension 生命周期覆盖全部转换
    record_optional_rl_phase("app_launcher", "complete")

    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding
    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

    rows = None if args.max_assets is None else tuple(range(args.max_assets))
    binding = build_generated_asset_binding(rows)
    children = cast(list[UrdfFileCfg], binding.hand_adapter.build_multi_hand_spawn_cfg().assets_cfg)
    if len(children) != len(binding.canonical_artifacts):
        raise RuntimeError("heterogeneous child cfg count does not match canonical artifact count")

    changed = 0
    reused = 0
    record_optional_rl_phase("usd_cache_prewarm", "start", asset_count=len(children))
    for row, (child, artifact) in enumerate(zip(children, binding.canonical_artifacts, strict=True)):
        usd_path = Path(child.usd_dir or "") / (child.usd_file_name or "")
        mtime_before = usd_path.stat().st_mtime_ns if usd_path.is_file() else None
        try:
            converter = UrdfConverter(child)
        except Exception:
            record_optional_rl_phase(
                "usd_cache_prewarm",
                "failed",
                asset_row=row,
                asset_id=artifact.asset_id,
                urdf_path=child.asset_path,
            )
            raise
        converted_path = Path(converter.usd_path).resolve(strict=True)
        mtime_after = converted_path.stat().st_mtime_ns
        was_changed = mtime_before is None or mtime_after != mtime_before
        changed += int(was_changed)
        reused += int(not was_changed)
        completed = row + 1
        if completed % args.progress_every == 0 or completed == len(children):
            print(
                {
                    "completed": completed,
                    "asset_count": len(children),
                    "last_asset_row": row,
                    "last_asset_id": artifact.asset_id,
                    "cache_changed": changed,
                    "cache_reused": reused,
                },
                flush=True,
            )
    record_optional_rl_phase(
        "usd_cache_prewarm",
        "complete",
        asset_count=len(children),
        cache_changed=changed,
        cache_reused=reused,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
