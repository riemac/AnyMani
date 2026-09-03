r"""在Isaac Sim GUI中分页检查MVP80 schema-3 Top-K good-pregrasp catalog。

脚本从最终``ppo_mvp80.yaml``读取formal rows，默认每页16手并消费rank-0。``frozen``只render exact reset；
``hold``以120 Hz持续下发缓存$q_0=u_0$并推进真实重力/接触物理，不运行policy或自动reset。由此用户可以分别
检查初始视觉包络和解除冻结后的弹射、滑移、翻倒风险。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import cast

import yaml

DEFAULT_MANIFEST = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml")
DEFAULT_CATALOG = Path("outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5")


def _parse_args() -> argparse.Namespace:
    r"""解析manifest page、Top-K rank与静态/动态查看模式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--offset", type=int, default=0, help="First asset offset in the final 80-row manifest.")
    parser.add_argument("--count", type=int, default=16, help="Assets shown in this GUI page.")
    parser.add_argument("--rank", type=int, choices=range(8), default=0)
    parser.add_argument("--spacing", type=float, default=0.65)
    parser.add_argument("--capture-dir", type=Path, default=None, help="Optional deterministic viewport PNG directory.")
    parser.add_argument(
        "--capture-steps",
        type=str,
        default="1,24,120",
        help="Comma-separated hold physics steps (or frozen render frames) to capture.",
    )
    parser.add_argument("--auto-exit", action="store_true", help="Exit after all requested captures are durable.")
    parser.add_argument(
        "--capture-closeups",
        action="store_true",
        help="After hold, freeze physics and capture one env-anchored close-up per asset.",
    )
    parser.add_argument(
        "--mode",
        choices=("frozen", "hold"),
        default="frozen",
        help="frozen只render exact reset；hold持续推进physics并保持缓存PD target。",
    )
    args = parser.parse_args()
    if args.offset < 0 or args.count < 1 or args.spacing <= 0.0:
        parser.error("offset must be non-negative; count/spacing must be positive")
    capture_steps = tuple(sorted({int(value.strip()) for value in args.capture_steps.split(",") if value.strip()}))
    if not capture_steps or capture_steps[0] < 1:
        parser.error("--capture-steps must contain positive integers")
    if args.auto_exit and args.capture_dir is None:
        parser.error("--auto-exit requires --capture-dir")
    if args.capture_closeups and not args.auto_exit:
        parser.error("--capture-closeups requires --auto-exit")
    args.capture_steps = capture_steps
    return args


ARGS = _parse_args()
MANIFEST_DOCUMENT = yaml.safe_load(ARGS.manifest.read_text(encoding="utf-8"))
ALL_ROWS = tuple(int(row) for row in MANIFEST_DOCUMENT["selected_rows"])
if len(ALL_ROWS) != 80 or len(set(ALL_ROWS)) != 80:
    raise ValueError("good-pregrasp viewer requires the final unique 80-row MVP manifest")
ROWS = ALL_ROWS[ARGS.offset : ARGS.offset + ARGS.count]
if not ROWS:
    raise ValueError("requested GUI page lies outside the 80-row manifest")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in ROWS)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(len(ROWS))
os.environ["ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS"] = "1"  # 恢复N000-style generated URDF palette
os.environ["ANYMANI_HETERO_N000_VISUAL_STYLE"] = "1"  # 使用N000 PolyHaven sky HDRI

from isaaclab.app import AppLauncher  # noqa: E402  # rows必须在scene/task import前冻结

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


def main() -> int:
    r"""创建scale-1.1 scene、执行一次schema-3 reset并保持GUI交互。"""

    import isaaclab.sim as sim_utils
    import omni.ui as ui
    from anymani.tasks.hetero.config.generated.pregrasp_harness_env_cfg import GeneratedPregraspHarnessEnvCfg
    from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING
    from anymani.tasks.hetero.mdp.events import reset_from_good_pregrasp_catalog
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    cfg = GeneratedPregraspHarnessEnvCfg()
    cfg.scene.env_spacing = float(ARGS.spacing)
    object_spawn = cast(sim_utils.UsdFileCfg, cfg.scene.object.spawn)
    object_spawn.scale = (1.1, 1.1, 1.1)  # exact MVP DexCube scale，prestartup-only
    cfg.viewer.eye = (4.2, 4.2, 3.4)
    cfg.viewer.lookat = (0.0, 0.0, 0.55)
    runtime_env = ManagerBasedRLEnv(cfg=cfg)
    try:
        runtime_env.sim._app_control_on_stop_handle = None
        runtime_env.reset()
        catalog_root = ARGS.catalog if ARGS.catalog.is_absolute() else Path.cwd() / ARGS.catalog
        reset_cfg = ASSET_BINDING.build_good_pregrasp_reset_cfg(
            num_envs=runtime_env.num_envs,
            rank=ARGS.rank,
            catalog_root=catalog_root,
        )
        reset_from_good_pregrasp_catalog(runtime_env, None, config=reset_cfg)
        sidecar = getattr(runtime_env, HETERO_PREGRASP_STATE_ATTR, None)
        if not isinstance(sidecar, HeterogeneousPregraspState) or not bool(sidecar.valid.all().item()):
            raise RuntimeError("GUI reset did not install every requested schema-3 entry")
        mapping = [
            {
                "env": env_id,
                "manifest_offset": ARGS.offset + env_id,
                "dataset_row": dataset_row,
                "asset_id": ASSET_BINDING.source_assets[env_id].asset_id,
                "entry_digest": sidecar.record_digests[env_id],
            }
            for env_id, dataset_row in enumerate(ROWS)
        ]
        print(
            {
                "mode": ARGS.mode,
                "rank": ARGS.rank,
                "page": [ARGS.offset, ARGS.offset + len(ROWS) - 1],
                "env_mapping": mapping,
                "instruction": "Inspect in Isaac Sim; close the window or press Ctrl+C to exit.",
            },
            flush=True,
        )
        capture_root = None
        if ARGS.capture_dir is not None:
            capture_root = ARGS.capture_dir if ARGS.capture_dir.is_absolute() else Path.cwd() / ARGS.capture_dir
            capture_root.mkdir(parents=True, exist_ok=True)
            mapping_path = capture_root / f"page-{ARGS.offset:02d}-{ARGS.offset + len(ROWS) - 1:02d}-mapping.json"
            mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # 独立overlay把viewport grid index映射到manifest row/asset ID，异常手可无歧义回报。
        label_window = ui.Window("MVP80 Pregrasp Mapping", width=520, height=620)
        with label_window.frame:
            with ui.VStack(spacing=3):
                ui.Label(
                    f"rank={ARGS.rank}  mode={ARGS.mode}  offsets={ARGS.offset}-{ARGS.offset + len(ROWS) - 1}",
                    style={"font_size": 16},
                )
                ui.Separator(height=4)
                for item in mapping:
                    ui.Label(
                        "env={env:02d}  offset={manifest_offset:02d}  row={dataset_row:04d}  asset={asset_id}".format(
                            **item
                        ),
                        style={"font_size": 14},
                    )
        robot = cast(Articulation, runtime_env.scene["robot"])
        frame_index = 0
        scheduled: dict[int, Path] = {}
        capture_handles = []  # future-like objects必须存活到文件写完
        post_capture_frames = 0
        while simulation_app.is_running():
            started = time.perf_counter()
            if ARGS.mode == "hold":
                robot.set_joint_position_target(sidecar.q_target_rad)  # $u=q_0$，不运行policy accumulator
                runtime_env.scene.write_data_to_sim()
                runtime_env.sim.step(render=True)
                runtime_env.scene.update(runtime_env.physics_dt)
            else:
                runtime_env.sim.render()
            frame_index += 1
            if capture_root is not None and frame_index in ARGS.capture_steps:
                capture_path = capture_root / (
                    f"page-{ARGS.offset:02d}-{ARGS.offset + len(ROWS) - 1:02d}-"
                    f"rank{ARGS.rank}-{ARGS.mode}-step{frame_index:04d}.png"
                )
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("Isaac GUI exposes no active viewport for automated capture")
                if capture_path.exists():
                    capture_path.unlink()  # 同一page重跑时必须等待本次frame真正写完
                capture_handles.append(capture_viewport_to_file(viewport, file_path=str(capture_path)))
                scheduled[frame_index] = capture_path
            if scheduled and len(scheduled) == len(ARGS.capture_steps):
                post_capture_frames += 1
                captures_ready = all(path.is_file() and path.stat().st_size > 0 for path in scheduled.values())
                if ARGS.auto_exit and captures_ready and post_capture_frames >= 8:
                    print(
                        {
                            "capture_dir": str(capture_root),
                            "captures": [str(path) for path in scheduled.values()],
                            "mapping": str(mapping_path),
                        },
                        flush=True,
                    )
                    break
            remaining = float(runtime_env.physics_dt) - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        if capture_root is not None and ARGS.capture_closeups:
            controller = runtime_env.viewport_camera_controller
            if controller is None:
                raise RuntimeError("Isaac GUI exposes no viewport camera controller for close-up capture")
            controller.update_view_location(
                eye=(0.45, 0.45, 0.85),
                lookat=(0.0, 0.0, 0.52),
            )  # controller构造时缓存eye/lookat；必须经公开方法同步近景参数
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac GUI exposes no active viewport for close-up capture")
            closeups: list[Path] = []
            for env_id, item in enumerate(mapping):
                controller.set_view_env_index(env_id)
                controller.update_view_to_env()
                for _ in range(6):
                    runtime_env.sim.render()  # 相机切换后稳定RTX accumulation，不再推进physics
                closeup = capture_root / (
                    f"offset-{item['manifest_offset']:02d}-row-{item['dataset_row']:04d}-"
                    f"asset-{item['asset_id']}-rank{ARGS.rank}-{ARGS.mode}-closeup.png"
                )
                if closeup.exists():
                    closeup.unlink()  # 本次capture拥有的确定性目标，避免旧文件伪装成异步完成
                capture_handles.append(capture_viewport_to_file(viewport, file_path=str(closeup)))
                for _ in range(30):
                    runtime_env.sim.render()
                    if closeup.is_file() and closeup.stat().st_size > 0:
                        break
                if not closeup.is_file() or closeup.stat().st_size == 0:
                    raise RuntimeError(f"close-up capture did not become durable: {closeup}")
                closeups.append(closeup)
            print({"closeup_captures": [str(path) for path in closeups]}, flush=True)
        return 0
    finally:
        runtime_env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
