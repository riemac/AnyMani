#!/usr/bin/env python3
r"""AnyMani single-asset contact-basin / pre-grasp calibrator.

This script is the thin Isaac Sim entry point for manual calibration:

1. choose generated or official LEAP hand asset;
2. restore the asset-scoped ``latest.yaml`` preset when available;
3. build a one-env IsaacLab scene;
4. open the GUI panel that edits joints/object pose and exports YAML.

The GUI callback details live in ``_single_asset_grasp_calibrator_ui.py`` so
this file stays focused on experiment setup and asset branch semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AnyMani single-asset pre-grasp/contact-basin GUI calibrator")
parser.add_argument(
    "--preset",
    type=str,
    default=None,
    help="Optional YAML preset to restore. Without it, the current asset-scoped latest.yaml is used if present.",
)
parser.add_argument(
    "--hand-bundle",
    type=str,
    default=None,
    help="Optional generated hand bundle path or asset-bank id. Ignored for official LEAP probes.",
)
parser.add_argument(
    "--official-leap-urdf",
    action="store_true",
    help=(
        "Use the official LEAP asset instead of an AnyMani generated bundle. "
        "Defaults to the project-local preconverted USD unless --official-leap-urdf-path is provided."
    ),
)
parser.add_argument(
    "--official-leap-urdf-path",
    type=str,
    default=None,
    help="Optional official LEAP raw URDF path. Only valid with --official-leap-urdf.",
)
parser.add_argument(
    "--official-leap-usd-path",
    type=str,
    default=None,
    help="Optional official LEAP preconverted USD path. Only valid with --official-leap-urdf.",
)
parser.add_argument(
    "--output-name",
    type=str,
    default=None,
    help="Optional exported preset filename. Without it, a timestamp file is written and latest.yaml is refreshed.",
)
parser.add_argument(
    "--object-source",
    choices=("local_cube", "dex_cube_usd"),
    default="local_cube",
    help="Object source: local_cube avoids remote USD dependency; dex_cube_usd keeps the original DexCube appearance.",
)
parser.add_argument(
    "--generated-collision-filter",
    choices=("none", "finger_palm", "finger_palm_same_finger"),
    default="none",
    help=(
        "Generated-hand collision-filter ablation. "
        "finger_palm_same_finger keeps cross-finger collisions but filters palm/finger and same-finger links."
    ),
)
parser.add_argument(
    "--smoke-seconds",
    type=float,
    default=None,
    help="Optional smoke duration after GUI/sim ready; the process hard-exits after this many wall-clock seconds.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim / Kit has been launched. Runtime modules that may depend on Kit
# extensions are imported only after this point.
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from anymani.robots.hand_spawn import DEFAULT_HAND_ANCHOR_POS_E, HandFrameCfg  # noqa: E402
from anymani.tools._grasp_preset_helpers import (  # noqa: E402
    PRESET_DIR,
    generated_asset_latest_preset_path,
    official_leap_latest_preset_path,
    select_start_preset,
)
from anymani.tools._single_asset_grasp_calibrator_helpers import (  # noqa: E402
    DEFAULT_HAND_BUNDLE_ID,
    _apply_generated_collision_filter,
    _build_generated_hand_articulation_cfg,
    _build_hand_spawn_cfg,
    _build_official_leap_articulation_cfg,
    _build_official_leap_usd_articulation_cfg,
    _hand_bundle_from_preset,
    _joint_pos_from_preset,
    _make_scene_cfg,
    _object_pose_from_preset,
    _resolve_hand_bundle_input,
    _resolve_official_leap_urdf_input,
    _resolve_official_leap_usd_input,
)
from anymani.tools._single_asset_grasp_calibrator_ui import (  # noqa: E402
    SingleAssetGraspCalibrationPanel,
    run_calibrator_simulator,
)


def _default_latest_preset_path_for_cli() -> Path:
    """Return the asset-scoped ``latest.yaml`` path selected by CLI flags."""

    if args_cli.official_leap_urdf:
        return official_leap_latest_preset_path()
    return generated_asset_latest_preset_path(args_cli.hand_bundle, default_hand_bundle_id=DEFAULT_HAND_BUNDLE_ID)


def main() -> None:
    """Create the single-env calibration scene and enter the GUI run loop."""

    default_latest_path = _default_latest_preset_path_for_cli()
    preset_path, preset_payload = select_start_preset(args_cli.preset, default_latest_path=default_latest_path)
    preset_hand_bundle = _hand_bundle_from_preset(preset_payload)

    if args_cli.official_leap_urdf:
        if args_cli.generated_collision_filter != "none":
            raise ValueError("--generated-collision-filter only applies to AnyMani generated hands, not official LEAP probes.")
        if args_cli.hand_bundle is not None:
            raise ValueError("--official-leap-urdf is an asset-level ablation; do not combine it with --hand-bundle.")
        if args_cli.official_leap_urdf_path is not None and args_cli.official_leap_usd_path is not None:
            raise ValueError("Use either --official-leap-urdf-path or --official-leap-usd-path, not both.")

        if args_cli.official_leap_urdf_path is not None:
            official_urdf_path = _resolve_official_leap_urdf_input(args_cli.official_leap_urdf_path)
            hand_articulation_cfg = _build_official_leap_articulation_cfg(official_urdf_path)
            hand_asset_ref = str(official_urdf_path)
            hand_source = "official_leap_urdf"
        else:
            official_usd_path = _resolve_official_leap_usd_input(args_cli.official_leap_usd_path)
            hand_articulation_cfg = _build_official_leap_usd_articulation_cfg(official_usd_path)
            hand_asset_ref = str(official_usd_path)
            hand_source = "official_leap_usd"
        hand_frame_cfg = HandFrameCfg(anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E)
    else:
        hand_bundle_path = _resolve_hand_bundle_input(args_cli.hand_bundle or preset_hand_bundle)
        hand_spawn_cfg = _build_hand_spawn_cfg(hand_bundle_path)
        hand_articulation_cfg = _build_generated_hand_articulation_cfg(hand_spawn_cfg)
        hand_frame_cfg = hand_spawn_cfg.frame
        hand_asset_ref = hand_bundle_path
        hand_source = "generated_bundle"

    initial_joint_pos = _joint_pos_from_preset(preset_payload)
    initial_object_pos_cfg, initial_object_rpy_xyz = _object_pose_from_preset(preset_payload)

    sim_cfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 1.0], [0.0, 0.06, 0.55])

    scene_cfg = _make_scene_cfg(
        hand_articulation_cfg,
        args_cli.object_source,
        initial_object_pos_cfg,
        initial_object_rpy_xyz,
        args_cli.device,
    )
    scene = InteractiveScene(scene_cfg)
    if hand_source == "generated_bundle":
        _apply_generated_collision_filter(scene, args_cli.generated_collision_filter)
    sim.reset()
    scene.update(sim.get_physics_dt())

    panel = SingleAssetGraspCalibrationPanel(
        scene=scene,
        robot=scene["robot"],
        obj=scene["object"],
        hand_frame_cfg=hand_frame_cfg,
        hand_asset_ref=hand_asset_ref,
        hand_source=hand_source,
        object_source=args_cli.object_source,
        initial_joint_pos=initial_joint_pos,
        initial_object_pos_cfg=initial_object_pos_cfg,
        initial_object_rpy_xyz=initial_object_rpy_xyz,
        output_name=args_cli.output_name,
        preset_dir=default_latest_path.parent,
        latest_preset_path=default_latest_path,
        loaded_preset_path=preset_path,
    )

    print("\n" + "=" * 88)
    print("AnyMani single-asset grasp calibrator is running.")
    print("Workflow: adjust sliders -> optionally unlock/gizmo/read object -> Export Preset.")
    print(f"Preset root: {PRESET_DIR}")
    print(f"Current asset preset directory: {default_latest_path.parent}")
    print(f"Current asset latest preset: {default_latest_path}")
    print("=" * 88 + "\n")

    run_calibrator_simulator(
        simulation_app=simulation_app,
        sim=sim,
        scene=scene,
        panel=panel,
        smoke_seconds=args_cli.smoke_seconds,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
