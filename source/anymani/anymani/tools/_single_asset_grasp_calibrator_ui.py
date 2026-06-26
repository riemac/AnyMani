"""GUI panel and run loop for ``single_asset_grasp_calibrator.py``.

The main calibrator script is kept as a thin experiment entry point.  This
module owns the noisy Isaac Sim UI callbacks, tensor-to-slider synchronization,
and YAML export side effects.

Calibration semantics preserved here:

- The exported YAML is a manual seed preset, not a validated grasp-cache shard
  and not evidence that the pose survives perturbation/settling.
- ``object_pose_cfg`` is expressed in the IsaacLab cfg/env frame and is the
  short-term value copied into reset/init config. ``object_pose_h`` is exported
  as an auxiliary hand-semantic-frame pose for future reset/cache work.
- Joint slider updates write both articulation state and PD target so the hand
  does not get pulled away from the manually chosen pre-grasp configuration.
- The object pose lock deliberately separates visual contact-basin editing from
  physical stability testing. Unlock/gizmo/read is an explicit one-way sync
  from the USD stage back into the UI/export state.
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import omni.ui as ui
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "single_asset_grasp_calibrator.py requires Isaac Sim GUI (`omni.ui`). "
        "Run without `--headless`, e.g. "
        "`source /home/hac/isaac/env_isaaclab/bin/activate && "
        "/home/hac/isaac/IsaacLab/isaaclab.sh -p "
        "source/anymani/anymani/tools/single_asset_grasp_calibrator.py`."
    ) from exc

import torch
import yaml
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim import utils as sim_utils_runtime
from isaaclab.utils import math as math_utils

from anymani.robots.hand_spawn import HandFrameCfg
from anymani.tools._single_asset_grasp_calibrator_helpers import (
    DEFAULT_LOCAL_CUBE_SIZE,
    DEFAULT_OBJECT_SCALE,
    _quat_from_rpy_xyz,
    _round_float,
    _rpy_xyz_from_quat,
)


class SingleAssetGraspCalibrationPanel:
    r"""Isaac Sim GUI panel for manual pre-grasp/contact-basin calibration.

    The panel maintains three runtime states: joint targets, object pose in the
    IsaacLab cfg/env frame, and a reset snapshot taken at script startup.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        robot: Articulation,
        obj: RigidObject,
        hand_frame_cfg: HandFrameCfg,
        hand_asset_ref: str,
        hand_source: str,
        object_source: str,
        initial_joint_pos: dict[str, float],
        initial_object_pos_cfg: tuple[float, float, float],
        initial_object_rpy_xyz: tuple[float, float, float],
        output_name: str | None,
        preset_dir: Path,
        latest_preset_path: Path,
        loaded_preset_path: Path | None,
    ) -> None:
        self.scene = scene
        self.robot = robot
        self.obj = obj
        self.hand_frame_cfg = hand_frame_cfg
        self.hand_asset_ref = hand_asset_ref
        self.hand_source = hand_source
        self.object_source = object_source
        self.output_name = output_name
        self.preset_dir = preset_dir
        self.latest_preset_path = latest_preset_path
        self.loaded_preset_path = loaded_preset_path

        self.device = robot.device
        self.joint_names = list(robot.joint_names)
        self.num_joints = len(self.joint_names)
        self.joint_limits = robot.data.joint_pos_limits[0].detach().cpu()

        self.joint_targets = robot.data.default_joint_pos.clone()
        self.joint_velocities = torch.zeros_like(robot.data.default_joint_vel)
        self.object_pos_cfg = torch.tensor(initial_object_pos_cfg, dtype=torch.float32, device=self.device)
        self.object_rpy_xyz = torch.tensor(initial_object_rpy_xyz, dtype=torch.float32, device=self.device)
        self.lock_object_pose = True

        self.joint_sliders: dict[str, dict[str, Any]] = {}
        self.object_sliders: dict[str, dict[str, Any]] = {}
        self.status_label: ui.Label | None = None
        self.lock_label: ui.Label | None = None

        for joint_name, joint_value in initial_joint_pos.items():
            if joint_name in self.joint_names:
                joint_idx = self.joint_names.index(joint_name)
                self.joint_targets[0, joint_idx] = float(joint_value)

        self.reset_joint_targets = self.joint_targets.clone()
        self.reset_object_pos_cfg = self.object_pos_cfg.clone()
        self.reset_object_rpy_xyz = self.object_rpy_xyz.clone()
        self.reset_lock_object_pose = True

        self._build_ui()
        self.apply_joint_state()
        self.apply_object_pose()
        self._set_status("Calibrator ready. Adjust joints/object, then Export Preset.")

    def _build_ui(self) -> None:
        self._window = ui.Window(
            "AnyMani Single-Asset Grasp Calibrator",
            width=760,
            height=900,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE,
            dock_preference=ui.DockPreference.LEFT_BOTTOM,
        )

        with self._window.frame:
            with ui.ScrollingFrame(
                height=ui.Fraction(1),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=8, height=0):
                    ui.Label(
                        "AnyMani Contact Basin / Pre-Grasp Calibrator",
                        height=28,
                        style={"font_size": 18, "color": 0xFFFFCC66},
                    )
                    ui.Label(f"Hand source: {self.hand_source}", height=20, style={"font_size": 11})
                    ui.Label(f"Hand asset: {self.hand_asset_ref}", height=20, style={"font_size": 11})
                    ui.Label(f"Object source: {self.object_source}", height=20, style={"font_size": 11})
                    ui.Label(
                        f"Loaded preset: {self.loaded_preset_path if self.loaded_preset_path else '[built-in seed]'}",
                        height=20,
                        style={"font_size": 11},
                    )
                    ui.Separator(height=2)
                    self._create_joint_controls()
                    ui.Separator(height=2)
                    self._create_object_controls()
                    ui.Separator(height=2)
                    self._create_action_buttons()

    def _create_joint_controls(self) -> None:
        finger_groups = {
            "Index": [name for name in self.joint_names if name.startswith("index_")],
            "Middle": [name for name in self.joint_names if name.startswith("middle_")],
            "Ring": [name for name in self.joint_names if name.startswith("ring_")],
            "Thumb": [name for name in self.joint_names if name.startswith("thumb_")],
        }
        grouped_names = {name for names in finger_groups.values() for name in names}
        finger_groups["Other"] = [name for name in self.joint_names if name not in grouped_names]

        ui.Label("Pre-Grasp Joint Configuration", height=24, style={"font_size": 16, "color": 0xFFAAFFAA})
        for group_name, joint_names in finger_groups.items():
            if not joint_names:
                continue
            with ui.CollapsableFrame(title=group_name, height=0, collapsed=False):
                with ui.VStack(spacing=4, height=0):
                    for joint_name in joint_names:
                        self._create_joint_slider(joint_name, self.joint_names.index(joint_name))

    def _create_joint_slider(self, joint_name: str, joint_idx: int) -> None:
        lower = float(self.joint_limits[joint_idx, 0].item())
        upper = float(self.joint_limits[joint_idx, 1].item())
        initial = float(self.joint_targets[0, joint_idx].item())

        with ui.HStack(spacing=6, height=24):
            ui.Label(joint_name, width=92, style={"color": 0xFFDDDDDD})
            slider = ui.FloatSlider(min=lower, max=upper, width=ui.Fraction(0.58), height=20)
            value_label = ui.Label("", width=140, alignment=ui.Alignment.LEFT, style={"font_size": 11})

            def on_value_changed(model, idx=joint_idx, label=value_label) -> None:
                value = float(model.as_float)
                self.joint_targets[0, idx] = value
                label.text = _format_angle(value)
                self.apply_joint_state()

            slider.model.add_value_changed_fn(on_value_changed)
            slider.model.set_value(initial)
            value_label.text = _format_angle(initial)
            self.joint_sliders[joint_name] = {"slider": slider, "label": value_label, "index": joint_idx}

    def _create_object_controls(self) -> None:
        ui.Label("Object Contact Basin Pose", height=24, style={"font_size": 16, "color": 0xFF88CCFF})
        self.lock_label = ui.Label("", height=20, style={"font_size": 12, "color": 0xFFFFFF99})
        self._refresh_lock_label()

        object_specs = [
            ("x", "pos", 0, -0.12, 0.12, "m"),
            ("y", "pos", 1, -0.02, 0.16, "m"),
            ("z", "pos", 2, 0.45, 0.68, "m"),
            ("roll", "rpy", 0, -math.pi, math.pi, "rad"),
            ("pitch", "rpy", 1, -math.pi, math.pi, "rad"),
            ("yaw", "rpy", 2, -math.pi, math.pi, "rad"),
        ]
        with ui.CollapsableFrame(title="Object Pose (cfg/env frame)", height=0, collapsed=False):
            with ui.VStack(spacing=4, height=0):
                for name, group, idx, lower, upper, unit in object_specs:
                    self._create_object_slider(name, group, idx, lower, upper, unit)

    def _create_object_slider(
        self,
        name: str,
        group: str,
        idx: int,
        lower: float,
        upper: float,
        unit: str,
    ) -> None:
        initial = float((self.object_pos_cfg if group == "pos" else self.object_rpy_xyz)[idx].item())
        with ui.HStack(spacing=6, height=24):
            ui.Label(name, width=60, style={"color": 0xFFDDDDDD})
            slider = ui.FloatSlider(min=lower, max=upper, width=ui.Fraction(0.62), height=20)
            value_label = ui.Label("", width=170, alignment=ui.Alignment.LEFT, style={"font_size": 11})

            def on_value_changed(model, pose_group=group, pose_idx=idx, label=value_label, field=name) -> None:
                value = float(model.as_float)
                if pose_group == "pos":
                    self.object_pos_cfg[pose_idx] = value
                    label.text = _format_distance(value)
                else:
                    self.object_rpy_xyz[pose_idx] = value
                    label.text = _format_angle(value)
                self.apply_object_pose()
                self._set_status(f"Updated object {field}; pose is locked to UI state.")

            slider.model.add_value_changed_fn(on_value_changed)
            slider.model.set_value(initial)
            value_label.text = _format_distance(initial) if group == "pos" else _format_angle(initial)
            self.object_sliders[name] = {"slider": slider, "label": value_label, "group": group, "index": idx}

    def _create_action_buttons(self) -> None:
        with ui.HStack(spacing=8, height=38):
            ui.Button("Apply Preset", clicked_fn=self.apply_all, height=32, style={"background_color": 0xFF336699})
            ui.Button("Apply Reset", clicked_fn=self.apply_reset, height=32, style={"background_color": 0xFF996633})
        with ui.HStack(spacing=8, height=38):
            ui.Button("Toggle Object Lock", clicked_fn=self.toggle_object_lock, height=32, style={"background_color": 0xFF665533})
            ui.Button(
                "Read Object From Stage",
                clicked_fn=self.read_object_from_stage,
                height=32,
                style={"background_color": 0xFF446644},
            )
        with ui.HStack(spacing=8, height=38):
            ui.Button("Export Preset", clicked_fn=self.export_preset, height=32, style={"background_color": 0xFF884488})

        self.status_label = ui.Label("", height=36, style={"font_size": 12, "color": 0xFFFFFFFF})

    def _set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.text = text
        print(f"[Calibrator] {text}")

    def _refresh_lock_label(self) -> None:
        if self.lock_label is None:
            return
        state = "LOCKED" if self.lock_object_pose else "UNLOCKED"
        hint = "unlock before using IsaacSim gizmo" if self.lock_object_pose else "move cube with gizmo, then Read Object From Stage"
        self.lock_label.text = f"Object pose lock: {state} ({hint})"

    def apply_joint_state(self) -> None:
        # Keep q_state and q_target identical, otherwise the PD controller drags the hand away from the manual pose.
        self.robot.write_joint_state_to_sim(self.joint_targets, self.joint_velocities)
        self.robot.set_joint_position_target(self.joint_targets)

    def _object_quat_wxyz(self) -> torch.Tensor:
        rpy = tuple(float(v) for v in self.object_rpy_xyz.detach().cpu().tolist())
        return _quat_from_rpy_xyz(rpy, self.device)

    def _object_pose_w(self) -> torch.Tensor:
        env_origin = self.scene.env_origins[0].to(self.device)
        pos_w = self.object_pos_cfg + env_origin
        quat_wxyz = self._object_quat_wxyz()[0]
        return torch.cat((pos_w, quat_wxyz), dim=0).reshape(1, 7)

    def apply_object_pose(self) -> None:
        root_pose = self._object_pose_w()
        zero_velocity = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        self.obj.write_root_pose_to_sim(root_pose)
        self.obj.write_root_velocity_to_sim(zero_velocity)

    def apply_all(self) -> None:
        self.apply_joint_state()
        self.apply_object_pose()
        self._set_status("Applied current joint/object preset to simulation.")

    def apply_reset(self) -> None:
        self.joint_targets[:] = self.reset_joint_targets
        self.object_pos_cfg[:] = self.reset_object_pos_cfg
        self.object_rpy_xyz[:] = self.reset_object_rpy_xyz
        self.lock_object_pose = bool(self.reset_lock_object_pose)

        self._sync_joint_sliders_from_state()
        self._sync_object_sliders_from_state()
        self._refresh_lock_label()
        self.apply_all()
        self._set_status("Reset to the pose loaded at script startup.")

    def toggle_object_lock(self) -> None:
        self.lock_object_pose = not self.lock_object_pose
        self._refresh_lock_label()
        self._set_status("Object pose lock toggled.")

    def read_object_from_stage(self) -> None:
        object_prim_path = self.obj.root_physx_view.prim_paths[0]
        object_prim = self.scene.stage.GetPrimAtPath(object_prim_path)
        pos_w_tuple, quat_w_tuple = sim_utils_runtime.resolve_prim_pose(object_prim)

        pos_w = torch.tensor(pos_w_tuple, dtype=torch.float32, device=self.device)
        quat_wxyz = torch.tensor(quat_w_tuple, dtype=torch.float32, device=self.device)
        env_origin = self.scene.env_origins[0].to(self.device)
        self.object_pos_cfg[:] = pos_w - env_origin
        self.object_rpy_xyz[:] = torch.tensor(_rpy_xyz_from_quat(quat_wxyz), dtype=torch.float32, device=self.device)

        self._sync_object_sliders_from_state()
        self.apply_object_pose()
        self.lock_object_pose = True
        self._refresh_lock_label()
        self._set_status(f"Read object pose from stage: {object_prim_path}; lock restored.")

    def _sync_object_sliders_from_state(self) -> None:
        for entry in self.object_sliders.values():
            group = entry["group"]
            idx = entry["index"]
            value = float((self.object_pos_cfg if group == "pos" else self.object_rpy_xyz)[idx].item())
            entry["slider"].model.set_value(value)
            entry["label"].text = _format_distance(value) if group == "pos" else _format_angle(value)

    def _sync_joint_sliders_from_state(self) -> None:
        for entry in self.joint_sliders.values():
            idx = entry["index"]
            value = float(self.joint_targets[0, idx].item())
            entry["slider"].model.set_value(value)
            entry["label"].text = _format_angle(value)

    def step_maintenance(self) -> None:
        if self.lock_object_pose:
            self.apply_object_pose()
            return
        zero_velocity = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        self.obj.write_root_velocity_to_sim(zero_velocity)

    def _hand_semantic_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w[:1]
        root_quat_w = self.robot.data.root_quat_w[:1]

        r_ha = torch.tensor(self.hand_frame_cfg.semantic_R_ha, dtype=torch.float32, device=self.device).reshape(3, 3)
        p_ha = torch.tensor(self.hand_frame_cfg.semantic_p_ha, dtype=torch.float32, device=self.device).reshape(1, 3)
        q_ha = math_utils.quat_from_matrix(r_ha.reshape(1, 3, 3))

        q_ah = math_utils.quat_inv(q_ha)
        p_ah = math_utils.quat_apply(q_ah, -p_ha)
        return math_utils.combine_frame_transforms(root_pos_w, root_quat_w, p_ah, q_ah)

    def _object_pose_h(self) -> tuple[list[float], list[float]]:
        pos_wh, quat_wh = self._hand_semantic_pose_w()
        object_pose_w = self._object_pose_w()
        pos_ho, quat_ho = math_utils.subtract_frame_transforms(
            pos_wh,
            quat_wh,
            object_pose_w[:, :3],
            object_pose_w[:, 3:7],
        )
        return (
            [_round_float(v) for v in pos_ho.detach().cpu()[0].tolist()],
            [_round_float(v) for v in quat_ho.detach().cpu()[0].tolist()],
        )

    def _joint_pos_rad_dict(self) -> dict[str, float]:
        return {joint_name: _round_float(self.joint_targets[0, idx].item()) for idx, joint_name in enumerate(self.joint_names)}

    def _joint_pos_deg_dict(self) -> dict[str, float]:
        return {
            joint_name: _round_float(math.degrees(self.joint_targets[0, idx].item()), ndigits=4)
            for idx, joint_name in enumerate(self.joint_names)
        }

    def _export_path(self) -> Path:
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        if self.output_name:
            name = self.output_name if self.output_name.endswith((".yaml", ".yml")) else f"{self.output_name}.yaml"
        else:
            name = f"single_asset_grasp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        return self.preset_dir / name

    def _preset_payload(self) -> dict[str, Any]:
        object_quat = self._object_quat_wxyz().detach().cpu()[0].tolist()
        object_pose_h_pos, object_pose_h_quat = self._object_pose_h()
        return {
            "schema_version": 1,
            "kind": "anymani_single_asset_grasp_preset",
            "asset": {
                "hand_source": self.hand_source,
                "hand_ref": self.hand_asset_ref,
                "hand_bundle": self.hand_asset_ref if self.hand_source == "generated_bundle" else None,
                "hand_usd": self.hand_asset_ref if self.hand_source == "official_leap_usd" else None,
                "hand_urdf": self.hand_asset_ref if self.hand_source == "official_leap_urdf" else None,
                "object_source": self.object_source,
                "object_id": "dex_cube" if self.object_source == "dex_cube_usd" else "local_cube",
                "object_scale": [_round_float(v) for v in DEFAULT_OBJECT_SCALE]
                if self.object_source == "dex_cube_usd"
                else None,
                "object_size": [_round_float(v) for v in DEFAULT_LOCAL_CUBE_SIZE]
                if self.object_source == "local_cube"
                else None,
            },
            "joint_pos_rad": self._joint_pos_rad_dict(),
            "joint_pos_deg": self._joint_pos_deg_dict(),
            "object_pose_cfg": {
                "pos": [_round_float(v) for v in self.object_pos_cfg.detach().cpu().tolist()],
                "rot_wxyz": [_round_float(v) for v in object_quat],
                "rpy_xyz_rad": [_round_float(v) for v in self.object_rpy_xyz.detach().cpu().tolist()],
            },
            "object_pose_h": {
                "pos": object_pose_h_pos,
                "rot_wxyz": object_pose_h_quat,
            },
            "notes": {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "source": "source/anymani/anymani/tools/single_asset_grasp_calibrator.py",
                "validated_cache": False,
                "comment": "Manual contact-basin/pre-grasp seed; not a settled grasp-cache shard.",
            },
        }

    def export_preset(self) -> None:
        payload = self._preset_payload()
        export_path = self._export_path()
        with export_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        self.latest_preset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.latest_preset_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)

        self._print_cfg_snippets(payload)
        self._set_status(f"Exported preset: {export_path}; asset latest refreshed: {self.latest_preset_path}.")

    def _print_cfg_snippets(self, payload: dict[str, Any]) -> None:
        print("\n" + "=" * 88)
        print("Joint preset (rad):")
        print("joint_pos = {")
        for joint_name, value in payload["joint_pos_rad"].items():
            print(f'    "{joint_name}": {value:.8f},')
        print("}")

        pos = payload["object_pose_cfg"]["pos"]
        quat = payload["object_pose_cfg"]["rot_wxyz"]
        print("\nObject init_state cfg snippet:")
        print("RigidObjectCfg.InitialStateCfg(")
        print(f"    pos=({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f}),")
        print(f"    rot=({quat[0]:.8f}, {quat[1]:.8f}, {quat[2]:.8f}, {quat[3]:.8f}),")
        print(")")
        print("=" * 88 + "\n")


def run_calibrator_simulator(
    simulation_app: Any,
    sim: SimulationContext,
    scene: InteractiveScene,
    panel: SingleAssetGraspCalibrationPanel,
    smoke_seconds: float | None = None,
) -> None:
    """Run the GUI calibration loop."""

    sim_dt = sim.get_physics_dt()
    smoke_start = time.monotonic() if smoke_seconds is not None else None
    while simulation_app.is_running():
        panel.step_maintenance()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if smoke_start is not None and time.monotonic() - smoke_start >= float(smoke_seconds):
            panel._set_status(f"Smoke completed after {smoke_seconds:.2f}s; hard-exiting smoke process.")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def _format_angle(value: float) -> str:
    return f"{value:+.4f} rad / {math.degrees(value):+6.1f} deg"


def _format_distance(value: float) -> str:
    return f"{value:+.5f} m"
