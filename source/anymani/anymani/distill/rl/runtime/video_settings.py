'Resolve viewer and recording settings without changing task or policy semantics.'

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@contextmanager
def native_video_writer(path: Path | None, settings: VideoSettings, *, policy_dt_s: float):
    'Handle native video writer.'

    if path is None:
        yield None
        return
    if path.suffix.lower() != ".mp4" or path.exists():
        raise ValueError("native video requires a new .mp4 output path")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), **cast(dict[str, Any], settings.writer_kwargs(policy_dt_s=policy_dt_s)))
    try:
        yield writer
    finally:
        writer.close()


def add_video_settings_arguments(parser: argparse.ArgumentParser) -> None:
    'Handle add video settings arguments.'


    parser.add_argument("--video_preset", choices=("standard", "paper_white"), default="standard")
    parser.add_argument("--video_resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--video_eye", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--video_lookat", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument(
        "--video_background",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Explicit scene-linear RGB background; does not change illumination.",
    )


@dataclass(frozen=True)
class VideoSettings:
    'Viewer and recording configuration; camera eye and look-at positions use metres.'

    preset: str = "standard"
    resolution: tuple[int, int] | None = None
    eye: tuple[float, float, float] | None = None  # units m
    lookat: tuple[float, float, float] | None = None  # units m
    background_linear_rgb: tuple[float, float, float] | None = None
    hide_ground_visual: bool = False

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        if self.preset not in {"standard", "paper_white"}:
            raise ValueError("unknown video preset")

        if self.resolution is not None:
            if len(self.resolution) != 2 or any(type(x) is not int or x <= 0 or x % 2 for x in self.resolution):
                raise ValueError("native yuv420p resolution requires positive even width and height")
        for name in ("eye", "lookat", "background_linear_rgb"):
            value = getattr(self, name)
            if value is not None and (len(value) != 3 or not all(math.isfinite(x) for x in value)):
                raise ValueError(f"video {name} requires three finite values")
        if self.background_linear_rgb is not None and min(self.background_linear_rgb) < 0:
            raise ValueError("video background must be nonnegative scene-linear RGB")

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> VideoSettings:
        'Handle from arguments.'

        paper = args.video_preset == "paper_white"
        resolution = tuple(args.video_resolution) if args.video_resolution is not None else None
        background = tuple(args.video_background) if args.video_background is not None else None
        return cls(
            preset=args.video_preset,
            resolution=resolution if resolution is not None else ((1920, 1080) if paper else None),
            eye=tuple(args.video_eye) if args.video_eye is not None else None,
            lookat=tuple(args.video_lookat) if args.video_lookat is not None else None,
            background_linear_rgb=background if background is not None else ((64.0, 64.0, 64.0) if paper else None),
            hide_ground_visual=paper,
        )

    def apply_viewer(self, env_cfg: Any) -> None:
        'Handle apply viewer.'

        for name in ("resolution", "eye", "lookat"):
            value = getattr(self, name)
            if value is not None:
                setattr(env_cfg.viewer, name, value)
        if self.eye is not None or self.lookat is not None:
            env_cfg.viewer.origin_type = "world"

    def apply_renderer(
        self,
        stage: Any,
        *,
        visible_environment_ids: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        'Handle apply renderer.'

        report: dict[str, object] = {
            "hidden_ground_visuals": [],
            "background_linear_rgb": None,
            "hidden_environment_visuals": [],
        }
        if self.background_linear_rgb is not None:
            import carb

            settings = carb.settings.get_settings()
            settings.set("/rtx/background/source/type", 2)
            settings.set("/rtx/background/source/color", list(self.background_linear_rgb))
            report["background_linear_rgb"] = list(settings.get("/rtx/background/source/color"))
        if self.hide_ground_visual:
            from pxr import UsdGeom

            hidden: list[str] = []
            for path in ("/World/ground", "/World/GroundPlane"):
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(path)
            report["hidden_ground_visuals"] = hidden
        if visible_environment_ids is not None:
            from pxr import UsdGeom

            if not visible_environment_ids or any(
                type(index) is not int or index < 0 for index in visible_environment_ids
            ):
                raise ValueError("visible environment IDs must be a nonempty tuple of nonnegative integers")
            keep = {f"env_{index}" for index in visible_environment_ids}
            scope = stage.GetPrimAtPath("/World/envs")
            if not scope.IsValid() or not keep.issubset({prim.GetName() for prim in scope.GetChildren()}):
                raise ValueError("requested visible environments are absent from the actual stage")
            hidden_envs: list[str] = []
            for prim in scope.GetChildren():
                if prim.GetName().startswith("env_") and prim.GetName() not in keep:
                    # units Hz
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden_envs.append(str(prim.GetPath()))
            report["hidden_environment_visuals"] = hidden_envs
            report["visible_environment_ids"] = list(visible_environment_ids)
        return report

    def writer_kwargs(self, *, policy_dt_s: float) -> dict[str, object]:
        'Handle writer kwargs.'

        if not math.isfinite(policy_dt_s) or policy_dt_s <= 0:
            raise ValueError("video policy_dt_s must be finite and positive")
        return {
            "fps": 1.0 / policy_dt_s,  # units Hz
            "codec": "libx264",
            "macro_block_size": 1,
            "pixelformat": "yuv420p",
        }

    def metadata(self, *, policy_dt_s: float, viewer: Any = None) -> dict[str, object]:
        'Handle metadata.'

        report = asdict(self)
        if viewer is not None:
            report["resolved_viewer"] = {
                **{name: list(getattr(viewer, name)) for name in ("resolution", "eye", "lookat")},
                "origin_type": getattr(viewer, "origin_type", None),
                "env_index": getattr(viewer, "env_index", None),
                "asset_name": getattr(viewer, "asset_name", None),
                "cam_prim_path": getattr(viewer, "cam_prim_path", None),
            }
        return {
            **report,
            "trajectory_source": "live_policy",
            "policy_dt_s": policy_dt_s,
            "encoding": self.writer_kwargs(policy_dt_s=policy_dt_s),
            "historical_trajectory_equivalence": "not_asserted",
        }
