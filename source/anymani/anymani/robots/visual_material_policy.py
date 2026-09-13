r"""Generated hand颜色恢复的渲染开关，不改变物理或学习配置。

场景配置在AppLauncher之后、SimulationContext创建之前构造，因此直接读取已经解析的Kit设置。
窗口、直播、XR或离屏摄像机需要URDF颜色；纯headless计算跳过材质计划与USD绑定。
这里不启动Kit，不导入Isaac Lab；普通离线配置检查默认关闭颜色恢复。
"""

from __future__ import annotations

import os


def generated_hand_visual_materials_enabled(*, override_env: str | None = None) -> bool:
    r"""按实际渲染用途决定是否恢复generated URDF颜色。

    Args:
        override_env: 可选的既有环境变量开关；显式0/1保留原语义，缺省时自动判断。

    Returns:
        是否需要可视材质。headless录像通过AppLauncher的offscreen设置识别，
        不依赖尚未创建的Camera传感器或SimulationContext实例。
    """
    if override_env is not None and override_env in os.environ:
        return os.environ[override_env] == "1"  # 保留显式关闭及预抓取查看工具的开启约定。

    # 仅使用已启动应用的设置接口；没有Kit的CPU合同测试和离线调用保持无渲染路径。
    try:
        import carb
    except ModuleNotFoundError as exc:
        if exc.name != "carb":
            raise
        return False

    settings = carb.settings.get_settings()  # AppLauncher已在任务配置导入前设置offscreen。
    return any(
        bool(settings.get(key))
        for key in (
            "/app/window/enabled",  # 本地GUI，包括普通play。
            "/app/livestream/enabled",  # 无本地窗口的远程查看。
            "/app/xr/enabled",  # XR查看同样需要视觉材质。
            "/isaaclab/render/offscreen",  # --headless --video / --enable_cameras。
        )
    )
