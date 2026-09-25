(
    'Render switch for generated-hand colors; it does not change physics or '
    'learning config. Scene config is built after AppLauncher and before '
    'SimulationContext, so read resolved Kit settings only. Restore URDF colors '
    'for windows, streaming, XR, or offscreen cameras; skip plans and USD '
    'bindings for compute-only headless runs. Do not start Kit or import Isaac '
    'Lab; offline config checks default to disabled.'
)

from __future__ import annotations

import os


def generated_hand_visual_materials_enabled(*, override_env: str | None = None) -> bool:
    (
        'Decide from actual render intent whether to restore generated URDF colors. '
        'An explicit override environment variable keeps its 0/1 meaning; when unset, '
        'detect from the already-running app. Headless recording is identified from '
        'AppLauncher offscreen settings, without depending on a Camera or '
        'SimulationContext that does not exist yet.'
    )
    if override_env is not None and override_env in os.environ:
        return os.environ[override_env] == "1"  # Preserve explicit disable and the pregrasp-inspection tool enable convention.

    # Use the already-running app settings only; CPU tests/offline calls without Kit stay render-free.
    try:
        import carb
    except ModuleNotFoundError as exc:
        if exc.name != "carb":
            raise
        return False

    settings = carb.settings.get_settings()  # AppLauncher sets offscreen before task config import.
    return any(
        bool(settings.get(key))
        for key in (
            "/app/window/enabled",  # Local GUI, including ordinary play.
            "/app/livestream/enabled",  # Remote viewing without a local window.
            "/app/xr/enabled",  # XR viewing also needs visual materials.
            "/isaaclab/render/offscreen",  # --headless --video / --enable_cameras。
        )
    )
