#!/usr/bin/env python3
r"""AnyMani 单资产 contact basin / pre-grasp 交互标定台。

本脚本服务当前单 asset 调试阶段的一个非常具体问题：在正式训练之前，先由人
在 Isaac Sim GUI 中把手调到一个合理的 pre-grasp joint configuration，并把
DexCube 放到一个看起来合理、可复现的 contact basin 中，然后导出可直接喂给
IsaacLab cfg / reset event 的数值。

科研语义边界：

- 本脚本不是训练环境；不注册 Gym task，不运行 PPO，不更新 reward / obs / action。
- 本脚本不是正式 `grasp_cache` 生成器；导出的 YAML 只是人工标定的 seed preset，
  还没有经过 settle / validation / perturbation robustness 检验。
- 本脚本的短期主产物是 `object_pose_cfg`，即 IsaacLab `RigidObjectCfg.InitialStateCfg`
  可直接使用的 env/world-frame pose；长期辅助产物是 `object_pose_h`，即未来
  grasp-cache 语义需要的 hand semantic frame `{h}` 下 object pose。

坐标系约定：

- `{e}`：IsaacLab cloned env frame。单 env 标定时 env origin 通常为 0，因此 `{e}` 与
  world `{w}` 只差一个平移；本脚本导出的 `object_pose_cfg.pos` 按 `{e}` 表达。
- `{a}`：hand raw asset/root frame，即 URDF/USD articulation root。
- `{h}`：hand semantic frame。当前 generated mother hand 默认 `{a}` 与 `{h}` 对齐，
  但导出时仍按 `HandFrameCfg.semantic_R_ha / semantic_p_ha` 显式计算，避免未来换资产
  时把坐标系假设悄悄写死。
- `{o}`：object body/root frame。`object_pose_h` 表示 $T^h_o$，即 object frame `{o}`
  相对 hand semantic frame `{h}` 的位姿。

使用方式：

```bash
cd /home/hac/isaac/AnyMani
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/single_asset_grasp_calibrator.py

# 从上次导出的 preset 恢复：
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/single_asset_grasp_calibrator.py \
  --preset source/anymani/anymani/tools/presets/latest.yaml

# 标定另一个单资产 bundle：
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/single_asset_grasp_calibrator.py \
  --hand-bundle source/anymani/anymani/assets/generated/.../<single_asset_bundle>
```
"""

from __future__ import annotations

import argparse  # CLI 参数：preset / hand bundle / AppLauncher flags
import math  # degree/radian 互转与 Euler slider 上下界
import os  # smoke 自终止时直接结束当前 Python 进程，避免 Isaac Sim shutdown 长时间挂住
import sys  # smoke 自终止前 flush stdout/stderr，保留 ready / completed 证据
import time  # smoke 验证用 wall-clock timeout，不改变训练/标定物理语义
from datetime import datetime  # 导出 YAML 时间戳，便于实验记录追踪
from itertools import combinations  # 枚举同一根 finger 内部 link pair，构造结构自碰撞过滤集合
from pathlib import Path  # preset / bundle 路径统一用 pathlib 表达
from typing import Any  # YAML payload 是异构 mapping，静态类型只能宽化为 Any

from isaaclab.app import AppLauncher  # 必须先启动 Isaac Sim app，后续才能导入 omni / pxr

# 先解析 AppLauncher 参数并启动 Isaac Sim。`omni.ui`、`pxr` 与若干 Kit API 必须在
# AppLauncher 创建 app 后再导入，否则普通 Python 解释器环境下没有 `omni` 模块。
parser = argparse.ArgumentParser(description="AnyMani 单资产 pre-grasp / contact basin GUI 标定台")  # 顶层 CLI
parser.add_argument(
    "--preset",  # YAML preset 路径；用于“上次导出 -> 本次继续微调”
    type=str,  # 路径字符串，后续统一转 `Path`
    default=None,  # None 表示尝试读取 `tools/presets/latest.yaml`
    help="可选：从既有 YAML preset 恢复 joint/object pose；不填时优先读取 tools/presets/latest.yaml。",
)
parser.add_argument(
    "--hand-bundle",  # 单资产 bundle；允许后续换另一个 generated asset 标定
    type=str,  # 既可以是本机路径，也可以是 asset-bank resolver 可解析 ID
    default=None,  # None 时使用当前 mother asset，保持本轮训练问题闭环
    help="可选：单资产 hand bundle 路径或 asset-bank 可解析 ID；不填时使用 right_t4_i4_m4_r4 mother asset。",
)
parser.add_argument(
    "--official-leap-urdf",  # 对照 probe：绕过 generated bundle，加载官方 LEAP 资产
    action="store_true",  # bool flag；默认仍走 AnyMani generated asset 主线
    help=(
        "可选：使用官方 LEAP 资产做抖动对照，而不是 AnyMani generated bundle。"
        "默认使用项目内已转换 USD；若传 --official-leap-urdf-path 则强制 raw URDF。"
    ),
)
parser.add_argument(
    "--official-leap-urdf-path",  # 官方 LEAP URDF 路径；默认指向项目内保留的 reference URDF
    type=str,  # 路径字符串，后续统一转 Path
    default=None,  # None 时不走 raw URDF，而是走项目内预转换 USD
    help="可选：官方 LEAP raw URDF 路径；仅在 --official-leap-urdf 时生效，且会强制走 URDF conversion。",
)
parser.add_argument(
    "--official-leap-usd-path",  # 官方 LEAP USD 路径；默认使用旧 LEAP env 已验证过的 edit USD
    type=str,  # 路径字符串，后续统一转 Path
    default=None,  # None 时使用 OFFICIAL_LEAP_USD_PATH
    help="可选：官方 LEAP 预转换 USD 路径；默认使用 source/anymani/assets/leap_hand_v1_right/leap_hand_right_edit.usd。",
)
parser.add_argument(
    "--output-name",  # 导出文件名；便于给某个肉眼 basin 起稳定名字
    type=str,  # `.yaml` 后缀可写可不写
    default=None,  # None 时自动使用时间戳，避免覆盖历史人工标定
    help="可选：导出 preset 文件名；不填时使用时间戳，同时刷新 latest.yaml。",
)
parser.add_argument(
    "--object-source",  # object 资产来源；默认本地几何体，避免远程 Nucleus/S3 卡住标定
    choices=("local_cube", "dex_cube_usd"),  # `dex_cube_usd` 保留原始带字母 DexCube 外观
    default="local_cube",  # 默认选择本地 cuboid，使官方 hand 消融不依赖网络资源
    help="object 来源：local_cube 使用本地 CuboidCfg；dex_cube_usd 使用远程 Isaac Nucleus DexCube USD。",
)
parser.add_argument(
    "--generated-collision-filter",  # 只作用于 generated hand 的 stage-level pair filter
    choices=("none", "finger_palm", "finger_palm_same_finger"),  # 保留 finger-finger，不做跨指过滤
    default="none",  # 默认不改物理，避免标定工具隐式改变训练候选资产语义
    help=(
        "generated hand 碰撞过滤消融：none 不过滤；finger_palm 过滤 finger 与 palm；"
        "finger_palm_same_finger 额外过滤同一根 finger 内部 link 之间的碰撞。"
    ),
)
parser.add_argument(
    "--smoke-seconds",  # 自动验证入口：ready 后跑若干秒就退出
    type=float,  # wall-clock 秒数；不作为仿真物理量导出
    default=None,  # None 表示正常 GUI 长跑
    help="可选：GUI/sim 初始化 smoke，ready 后运行指定秒数并自动退出；用于 agent 验证，不影响正常标定。",
)
AppLauncher.add_app_launcher_args(parser)  # 注入 IsaacLab 标准 `--device/--headless/...` 参数
args_cli = parser.parse_args()  # argparse 会在 `--help` 时提前退出，不启动 Kit
app_launcher = AppLauncher(args_cli)  # 启动 Isaac Sim / Kit runtime
simulation_app = app_launcher.app  # Kit app handle，主循环用它判断窗口是否仍在运行


"""Isaac Sim app 已启动，后续可以安全导入 GUI / USD / IsaacLab runtime 模块。"""

# ============================= 实现不变量 / Implementation invariants =============================
#
# 1. 文件级职责：本文件只生成“人工标定 preset”，不写训练环境、不写 reward、不写
#    `grasp_cache` store/sampler。这样做是为了让当前单 asset 调试先恢复研究动量，
#    不把“肉眼 contact basin 选择”过早伪装成经过统计验证的 reset 分布。
#
# 2. 启动顺序：`AppLauncher` 必须先于 `omni.ui` / `pxr` / 大部分 IsaacLab runtime import。
#    普通 uv python 中没有 Kit 注入的 `omni` / `pxr` 模块；因此本文件允许 `py_compile`
#    检查语法，但不能通过普通 `python script.py` 运行。
#
# 3. pose 主语义：短期训练接入使用 `object_pose_cfg`，即 IsaacLab init_state 所需的
#    env-frame pose；长期 reset/cache 语义同步导出 `object_pose_h`，但它在 V1 中只是
#    辅助字段，不代表已经有 validated grasp cache。
#
# 4. 编辑模式：object 默认 `disable_gravity=True` 且 pose lock 开启。这个选择服务
#    人工标定，而不是服务物理稳定性评估；zero-action settle 应作为后续单独 probe，
#    不应混进这个 GUI 工具的第一版职责。
#
# 5. joint 写入：每次 slider 更新都同时写 `write_joint_state_to_sim` 与
#    `set_joint_position_target`。这不是重复操作，而是避免 $q_{\text{state}}$ 与
#    $q_{\text{target}}$ 不一致导致 PD controller 把手拉离人工姿态。
#
# 6. object 写入：pose slider 维护的是 `{e}` 下的 cfg 值；写入 sim 时加
#    `scene.env_origins[0]` 得到 `{w}` 位置。单 env 下二者通常相等，但显式写出这个
#    转换可以防止未来复制到多 env debug 脚本时产生隐藏 frame bug。
#
# 7. UI 与 gizmo 同步：锁定时脚本每帧覆盖 object pose；解锁后用户可用 gizmo 拖动，
#    再通过 `Read Object From Stage` 把 stage pose 读回 UI 状态。不要试图让 slider
#    和 gizmo 双向实时同步，因为那会引入事件顺序和 Fabric/USD 缓存问题。
#
# 8. YAML 可读性：所有导出浮点数做轻量 round；这不是数值计算近似，只是为了让
#    人工 diff 和实验记录可读。IsaacLab reset 对 1e-8 级 pose 差异没有研究意义。
#
# 9. 资产切换：`--hand-bundle` 只承诺“另一个单资产 bundle”。V1 不承诺多资产 bank、
#    topology schema 混合、批量标定或自动选择；这些都应留给后续训练管线设计。
#
# 10. GUI 约束：本工具依赖 `omni.ui`，必须用非 headless Isaac Sim experience 运行。
#     `--headless` 可以用于很多训练 smoke，但不适合这里；如果强行 headless，脚本应
#     给出明确错误，而不是让用户面对 `ModuleNotFoundError: omni.ui` 这种低层异常。
#
# 11. 失败显式化：显式 `--preset` 读取失败应报错，不 silent fallback。标定工具最怕
#     用户以为自己在复现上一次 basin，实际却回到了内置 seed。

# GUI 标定必须依赖 `omni.ui`。headless experience 通常不加载该 extension，因此这里
# 把低层 import error 转成面向标定工作流的提示，避免用户误以为是 AnyMani 资产问题。
try:
    import omni.ui as ui  # Isaac Sim 内嵌 GUI toolkit
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "single_asset_grasp_calibrator.py requires Isaac Sim GUI (`omni.ui`). "
        "Run without `--headless`, e.g. "
        "`source /home/hac/isaac/env_isaaclab/bin/activate && "
        "/home/hac/isaac/IsaacLab/isaaclab.sh -p "
        "source/anymani/anymani/tools/single_asset_grasp_calibrator.py`."
    ) from exc
import isaaclab.sim as sim_utils  # spawn cfg namespace，例如 `UsdFileCfg` / `GroundPlaneCfg`
import torch  # IsaacLab runtime buffers 使用 torch tensor
import yaml  # preset 使用 YAML，便于人工读写与 diff
from anymani.assets.bank import HandBankCfg  # 复用 AnyMani asset-bank 显式单资产选择语义
from anymani.assets.bank.path_utils import resolve_bank_path  # 解析项目相对 bundle / asset ID
from anymani.tasks.gm.hand_spawn import (
    DEFAULT_HAND_ANCHOR_POS_E,  # 与 GM single asset 默认 hand anchor 对齐
    HandFrameCfg,  # `{a}` / `{h}` 语义 frame 配置
    HandSpawnAdapter,  # AnyMani hand bundle -> IsaacLab ArticulationCfg
    HandSpawnCfg,  # hand spawn 顶层配置
    HandUrdfSpawnCfg,  # URDF importer 参数
)
from isaaclab.actuators import ImplicitActuatorCfg  # raw official LEAP URDF probe 使用同类隐式 PD actuator
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg  # scene 资产类型
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # 单 env scene 构造
from isaaclab.sim import SimulationCfg, SimulationContext  # 仿真上下文与时序配置
from isaaclab.sim import utils as sim_utils_runtime  # `resolve_prim_pose` 用于读取 gizmo 后 stage pose
from isaaclab.sim.converters import UrdfConverterCfg  # raw official LEAP URDF importer 的 drive 配置
from isaaclab.utils import configclass  # IsaacLab cfg dataclass 装饰器
from isaaclab.utils import math as math_utils  # quaternion / frame transform 数学工具
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # DexCube USD 路径来源
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # stage-level collision filtering 需要 USD Physics schema

TOOLS_DIR = Path(__file__).resolve().parent
"""本工具脚本所在目录；preset 默认写到其下，便于与临时标定脚本一起管理。"""

PRESET_DIR = TOOLS_DIR / "presets"
"""人工标定 preset 输出目录；它不是正式 `gm/grasp_cache/artifacts`。"""

LATEST_PRESET_PATH = PRESET_DIR / "latest.yaml"
"""默认恢复入口；每次导出成功后刷新，方便下一轮继续微调。"""

DEFAULT_HAND_BUNDLE_ID = (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
    "single_palm_leap/right_t4_i4_m4_r4"
)
"""当前单 asset 调试默认 mother bundle，与 `gm/single_asset_env_cfg.py` 保持同一资产。"""

OFFICIAL_LEAP_URDF_PATH = Path(__file__).resolve().parents[2] / "assets/hands/leap_hand/leap_hand_right.urdf"
"""项目内保留的官方 LEAP raw URDF，用于隔离 generated collision 与脚本维护逻辑。"""

OFFICIAL_LEAP_USD_PATH = Path(__file__).resolve().parents[2] / "assets/leap_hand_v1_right/leap_hand_right_edit.usd"
"""项目内旧 LEAP 环境已使用的官方 LEAP 预转换 USD；比 raw URDF conversion 更适合 GUI 消融。"""

OFFICIAL_LEAP_ROOT_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
r"""官方 LEAP raw URDF 的标定台 root orientation，沿用旧 LEAP scene 的“手掌朝上”姿态。"""

DEFAULT_OBJECT_SCALE = (1.2, 1.2, 1.2)
"""DexCube isotropic scale；与当前 GM / LEAP reorientation probes 的物体尺度一致。"""

DEFAULT_LOCAL_CUBE_SIZE = (0.06, 0.06, 0.06)
r"""本地 cuboid fallback 的边长，单位 m。

该值服务 GUI / 消融稳定性：目标不是复刻 DexCube 字母外观，而是在无网络资源时
提供一个与当前 contact-basin 调试量级接近的刚体 cube。若需要完全复现训练中的
DexCube 外观，可用 `--object-source dex_cube_usd` 显式切回远程 USD。
"""

DEFAULT_OBJECT_POS_CFG = (0.0, 0.075, 0.56)
r"""默认 object 初始位置，单位 m，按 `{e}` / cfg frame 表达。

数值锚点说明：
早期 GM cfg 使用 `(0.0, 0.055, 0.56)`，但迁移 probe 中 `(0.0, 0.075, 0.56)`
更接近稳定托持 basin。这里选择更适合作为人工标定起点的 0.075，而不是宣称它
已经是最终训练默认值。
"""

DEFAULT_OBJECT_RPY_XYZ = (0.0, 0.0, 0.0)
"""默认 object 姿态的 XYZ Euler angle，单位 rad；导出时同时写 quaternion `wxyz`。"""

DEFAULT_PREGRASP_JOINT_POS_DEG = {
    "index_j0": 0.0,  # 食指 ab/ad 或根部侧摆初值，degree
    "index_j1": -22.0,  # 食指近端弯曲初值，degree
    "index_j2": 82.0,  # 食指中段弯曲初值，degree
    "index_j3": 74.0,  # 食指远端弯曲初值，degree
    "middle_j0": 0.0,  # 中指根部侧摆初值，degree
    "middle_j1": 0.0,  # 中指近端弯曲初值，degree
    "middle_j2": 72.0,  # 中指中段弯曲初值，degree
    "middle_j3": 62.0,  # 中指远端弯曲初值，degree
    "ring_j0": 6.0,  # 无名指根部侧摆初值，degree
    "ring_j1": 27.0,  # 无名指近端弯曲初值，degree
    "ring_j2": 81.0,  # 无名指中段弯曲初值，degree
    "ring_j3": 50.0,  # 无名指远端弯曲初值，degree
    "thumb_j0": 7.0,  # 拇指基座侧摆/旋转初值，degree；当前行为机制的关键自由度之一
    "thumb_j1": 90.0,  # 拇指掌内包络初值，degree；决定能否形成持续推转接触
    "thumb_j2": 33.0,  # 拇指中段弯曲初值，degree
    "thumb_j3": 98.0,  # 拇指远端弯曲初值，degree
}
"""用户人工测得的第一组 pre-grasp 角度，单位 degree；脚本启动时转成 rad。"""


def _deg_dict_to_rad(joint_pos_deg: dict[str, float]) -> dict[str, float]:
    r"""把人工读数中的 degree joint preset 转成 IsaacLab 使用的 rad。

    Args:
        joint_pos_deg (dict[str, float]): 关节名到角度的映射，单位 degree。

    Returns:
        dict[str, float]: 同名关节位置，单位 rad。
    """

    # IsaacLab articulation joint state 使用 rad；保留 degree 输入只为贴近人工 GUI / VSCode 插件读数。
    return {joint_name: math.radians(float(value_deg)) for joint_name, value_deg in joint_pos_deg.items()}


DEFAULT_PREGRASP_JOINT_POS_RAD = _deg_dict_to_rad(DEFAULT_PREGRASP_JOINT_POS_DEG)
"""默认 pre-grasp joint preset，单位 rad，按 generated hand joint names 显式索引。"""

GENERATED_FINGER_LINK_CHAINS = {
    "index": ("index_root_fixed_link", "index_mcp1", "index_mcp2", "index_pip", "index_dip", "index_tip"),
    "middle": ("middle_root_fixed_link", "middle_mcp1", "middle_mcp2", "middle_pip", "middle_dip", "middle_tip"),
    "ring": ("ring_root_fixed_link", "ring_mcp1", "ring_mcp2", "ring_pip", "ring_dip", "ring_tip"),
    "thumb": ("thumb_cmc1", "thumb_cmc2", "thumb_mcp", "thumb_dip", "thumb_tip"),
}
r"""Generated LEAP-like hand 的 finger link 链。

这些名字来自当前 `right_t4_i4_m4_r4/hand.urdf` 的结构语义，而不是 runtime joint
顺序。它们只用于本 GUI 标定台的碰撞过滤消融：

- 同一 tuple 内的 link 属于同一根 finger，可用于“单指内部自碰撞”过滤；
- 不同 tuple 之间属于不同 fingers，本消融刻意不互相过滤，保留 finger-finger 碰撞；
- `palm` 单独处理，避免把掌心与 finger 的结构装配接触误读为 hand-object contact。
"""

GENERATED_FINGER_LINK_CHAINS_BY_NAME = tuple(GENERATED_FINGER_LINK_CHAINS.values())
"""只保留 link 链 tuple，便于遍历全部 finger 而不关心 finger label。"""

GENERATED_COLLISION_GROUP_ROOT = "/World/anymani_calibrator_generated_collision_filters"
"""Generated-hand collision filter 的 USD scope；仅在标定脚本 stage 内临时 author。"""

def _round_float(value: float, ndigits: int = 8) -> float:
    r"""导出 YAML 前做轻量浮点截断，提升人工 diff 可读性。

    Args:
        value (float): 待导出的浮点数。
        ndigits (int): 小数位数，默认 8 位足够表达 meter/rad 级 reset preset。

    Returns:
        float: 截断后的普通 Python float。
    """

    return round(float(value), ndigits)


def _as_float_list(values: Any, *, expected_len: int, field_name: str) -> list[float]:
    r"""从 YAML 字段中解析定长 float list。

    Args:
        values (Any): YAML 读取出的字段值。
        expected_len (int): 期望长度。
        field_name (str): 错误消息中显示的字段名。

    Returns:
        list[float]: 定长普通 float 列表。

    Raises:
        ValueError: 字段不存在、不是序列或长度不匹配。
    """

    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"Preset field {field_name!r} must be a list of length {expected_len}, got {values!r}.")
    return [float(value) for value in values]


def _quat_from_rpy_xyz(rpy_xyz: tuple[float, float, float], device: str) -> torch.Tensor:
    r"""把 XYZ Euler angle 转成 IsaacLab root pose 使用的 quaternion `(w,x,y,z)`。

    Args:
        rpy_xyz (tuple[float, float, float]): `(roll,pitch,yaw)`，单位 rad。
        device (str): IsaacLab simulation device，例如 `"cuda:0"` 或 `"cpu"`。

    Returns:
        torch.Tensor: quaternion，形状 `[1,4]`，顺序 `(w,x,y,z)`。
    """

    roll = torch.tensor([float(rpy_xyz[0])], dtype=torch.float32, device=device)  # 绕 x 轴角，rad
    pitch = torch.tensor([float(rpy_xyz[1])], dtype=torch.float32, device=device)  # 绕 y 轴角，rad
    yaw = torch.tensor([float(rpy_xyz[2])], dtype=torch.float32, device=device)  # 绕 z 轴角，rad
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)  # `[1,4]`，IsaacLab `wxyz`


def _rpy_xyz_from_quat(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    r"""把 quaternion `(w,x,y,z)` 转成 UI slider 使用的 XYZ Euler angle。

    Args:
        quat_wxyz (torch.Tensor): quaternion，形状 `[4]` 或 `[1,4]`。

    Returns:
        tuple[float, float, float]: `(roll,pitch,yaw)`，单位 rad。
    """

    quat_batch = quat_wxyz.reshape(1, 4)  # 统一成 `[1,4]`，匹配 IsaacLab math API
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_batch)  # 每项形状 `[1]`
    return (float(roll[0].item()), float(pitch[0].item()), float(yaw[0].item()))


def _resolve_hand_bundle_input(hand_bundle: str | None) -> str:
    r"""解析 CLI hand bundle 输入，得到绝对 bundle 路径字符串。

    Args:
        hand_bundle (str | None): 用户传入的路径或 asset-bank 可解析 ID；为空则使用默认母体。

    Returns:
        str: 绝对 hand bundle 路径。
    """

    # 默认资产直接走 asset-bank resolver，避免把工作区绝对路径硬编码进脚本逻辑。
    if hand_bundle is None:
        return str(resolve_bank_path(DEFAULT_HAND_BUNDLE_ID))

    # 如果用户给的是本机路径，优先按路径展开；这让命令行补全和临时 bundle 更顺手。
    candidate = Path(hand_bundle).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    # 其余情况交给 AnyMani asset-bank resolver；它能处理项目相对路径或未来稳定 ID。
    return str(resolve_bank_path(hand_bundle))


def _build_hand_spawn_cfg(hand_bundle_path: str) -> HandSpawnCfg:
    r"""为单资产标定台构造 hand spawn cfg。

    Args:
        hand_bundle_path (str): 包含 `hand.urdf` / `hand.yaml` / `meshes` 的 bundle 路径。

    Returns:
        HandSpawnCfg: 可由 `HandSpawnAdapter` lower 成 IsaacLab `ArticulationCfg` 的配置。
    """

    return HandSpawnCfg(
        bank=HandBankCfg(
            source_mode="post_mutate",  # 当前 generated bundle 通过 post-mutate bank 入口解析
            selection_mode="explicit",  # 标定台只服务单资产，不从 bank 中随机采样
            containers=(hand_bundle_path,),  # 单元素 tuple，保持与 HandBankCfg schema 一致
            validate_mesh_relpaths=True,  # 启动时验证 URDF mesh 引用，避免标定了坏资产
            parse_visual_rgba=True,  # GUI 标定需要恢复 debug colors 便于肉眼识别 link
        ),
        frame=HandFrameCfg(
            semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # generated mother 默认 `{a}` 与 `{h}` 对齐
            semantic_p_ha=(0.0, 0.0, 0.0),  # raw asset origin 与 hand semantic origin 暂视为重合
            anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # hand semantic frame 初始朝向为 env identity
            anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E,  # hand semantic origin 初始高度，与 GM probe 对齐
        ),
        urdf=HandUrdfSpawnCfg(activate_contact_sensors=False),  # 标定台不需要 ContactSensor reporter
        asset_routing="round_robin",  # 单资产下等价于固定选择；保留 adapter 默认语义
        restore_visual_materials=True,  # GUI 标定优先可读性，允许关闭 instanceable 优化
        validate_same_schema=True,  # 即使单资产也验证 schema，暴露 bundle 结构错误
    )


def _resolve_official_leap_urdf_input(urdf_path_arg: str | None) -> Path:
    r"""解析官方 LEAP raw URDF probe 的输入路径。

    Args:
        urdf_path_arg (str | None): CLI `--official-leap-urdf-path`；为空时使用项目内 reference URDF。

    Returns:
        Path: 已展开的官方 LEAP URDF 绝对路径。

    Raises:
        FileNotFoundError: URDF 文件不存在。
    """

    urdf_path = Path(urdf_path_arg).expanduser() if urdf_path_arg is not None else OFFICIAL_LEAP_URDF_PATH
    urdf_path = urdf_path.resolve(strict=False)  # `strict=False` 便于错误消息保留用户原意路径
    if not urdf_path.is_file():
        raise FileNotFoundError(f"official LEAP URDF does not exist: {urdf_path}")
    return urdf_path


def _resolve_official_leap_usd_input(usd_path_arg: str | None) -> Path:
    r"""解析官方 LEAP 预转换 USD probe 的输入路径。

    Args:
        usd_path_arg (str | None): CLI `--official-leap-usd-path`；为空时使用旧 LEAP env 的 edit USD。

    Returns:
        Path: 已展开的官方 LEAP USD 绝对路径。

    Raises:
        FileNotFoundError: USD 文件不存在。
    """

    usd_path = Path(usd_path_arg).expanduser() if usd_path_arg is not None else OFFICIAL_LEAP_USD_PATH
    usd_path = usd_path.resolve(strict=False)  # `strict=False` 使错误消息保留预期绝对路径
    if not usd_path.is_file():
        raise FileNotFoundError(f"official LEAP USD does not exist: {usd_path}")
    return usd_path


def _build_generated_hand_articulation_cfg(hand_spawn_cfg: HandSpawnCfg) -> ArticulationCfg:
    r"""把 AnyMani generated hand bundle lower 成 IsaacLab articulation cfg。

    Args:
        hand_spawn_cfg (HandSpawnCfg): AnyMani generated bundle spawn 配置。

    Returns:
        ArticulationCfg: 可赋给 scene.robot 的 articulation 配置。
    """

    return HandSpawnAdapter(hand_spawn_cfg).build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")


def _build_official_leap_usd_articulation_cfg(usd_path: Path) -> ArticulationCfg:
    r"""构造官方 LEAP 预转换 USD 的对照 articulation cfg。

    这是官方资产消融的默认路径。相比 raw URDF，它复用了项目旧 LEAP 环境已经引用的
    USD 资产，避免把“URDF converter / collision mesh cooking 是否成功”混进
    “官方资产在同一 slider 维护逻辑下是否抖动”的研究问题。

    Args:
        usd_path (Path): 官方 LEAP 预转换 USD 路径。

    Returns:
        ArticulationCfg: 直接加载官方 LEAP USD 的 IsaacLab articulation 配置。
    """

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),  # 旧 LEAP env 已使用的官方 USD；不经过 raw URDF converter
            activate_contact_sensors=False,  # 本消融只看可视抖动，不需要 ContactSensor reporter
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 保持 articulation rigid bodies，而不是 kinematic preview
                disable_gravity=True,  # 标定台固定手，不测试手在重力下的漂移
                retain_accelerations=False,  # 与旧 LEAP cfg / generated probe 对齐
                enable_gyroscopic_forces=False,  # 与旧 LEAP cfg / generated probe 对齐
                angular_damping=0.01,  # 与旧 LEAP cfg / generated probe 对齐
                max_linear_velocity=1000.0,  # 与旧 LEAP cfg / generated probe 对齐
                max_angular_velocity=64.0 / math.pi * 180.0,  # 与旧 LEAP cfg / generated probe 对齐
                max_depenetration_velocity=1000.0,  # 保留接触 depenetration 上限，便于和 generated 现象对照
                max_contact_impulse=1e32,  # 旧 LEAP cfg 使用该兜底；避免官方 USD probe 因冲量上限不同失真
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,  # 官方资产也开启 self-collision，观察是否同样诱发 slider jitter
                solver_position_iteration_count=8,  # 与旧 LEAP cfg / generated probe 对齐
                solver_velocity_iteration_count=0,  # 与旧 LEAP cfg / generated probe 对齐
                sleep_threshold=0.005,  # 与旧 LEAP cfg / generated probe 对齐
                stabilization_threshold=0.0005,  # 与旧 LEAP cfg / generated probe 对齐
                fix_root_link=True,  # root 固定，保证 probe 不是 free-floating hand
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_ANCHOR_POS_E,  # 与 generated 标定台同一手根高度
            rot=OFFICIAL_LEAP_ROOT_ROT_WXYZ,  # 旧 LEAP env 使用的“手掌朝上”朝向
            joint_pos={".*": 0.0},  # 官方 USD 使用自己的 runtime joint names；不做 generated-name 映射
            joint_vel={".*": 0.0},  # 初始静止，避免 probe 叠加历史速度
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # 控制官方 USD 暴露出的全部 revolute joints
                effort_limit_sim=0.5,  # 旧 LEAP IsaacLab cfg 使用的 effort 上限
                velocity_limit_sim=100.0,  # 旧 LEAP IsaacLab cfg 使用的速度上限
                stiffness=3.0,  # 与旧 LEAP cfg / generated probe 对齐
                damping=0.1,  # 与旧 LEAP cfg / generated probe 对齐
                friction=0.01,  # 与旧 LEAP cfg / generated probe 对齐
                armature=0.001,  # 与旧 LEAP cfg / generated probe 对齐
            ),
        },
        soft_joint_pos_limit_factor=1.0,  # slider 上下界直接使用官方 USD joint limits
    )


def _build_official_leap_articulation_cfg(urdf_path: Path) -> ArticulationCfg:
    r"""构造官方 LEAP raw URDF 的对照 articulation cfg。

    这个 cfg 只服务“资产本身是否导致 slider 抖动”的控制变量实验。它不把官方
    joint names 映射到 AnyMani generated joint names，也不读取 generated preset 的
    canonical 关节语义；runtime 里 IsaacLab 实际暴露什么 joint name，UI 就显示什么。

    Args:
        urdf_path (Path): 官方 LEAP raw URDF 路径。

    Returns:
        ArticulationCfg: 直接加载官方 URDF 的 IsaacLab articulation 配置。
    """

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),  # 官方 raw URDF；不经过 AnyMani asset-bank/bundle adapter
            fix_base=True,  # 手掌固定，和 generated 标定台保持“手不掉落”的控制变量
            merge_fixed_joints=False,  # 保留官方 link/joint 层级，便于肉眼核对资产结构
            force_usd_conversion=False,  # 复用 IsaacLab converter cache；必要时用户可清 cache 重转
            make_instanceable=True,  # probe 不需要恢复 generated debug colors，可保留 instanceable 优化
            collision_from_visuals=False,  # 使用官方 URDF 自带 collision，不用 visual mesh 伪造碰撞
            self_collision=True,  # 与 generated hand probe 一样开启 self-collision
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",  # slider 写入的是关节位置目标 $q^\star$
                drive_type="force",  # 与 generated URDF importer 的 PD drive 语义一致
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=3.0,  # LEAP / generated probe 共用低刚度 PD 数值锚点
                    damping=0.1,  # LEAP / generated probe 共用阻尼锚点
                ),
            ),
            activate_contact_sensors=False,  # 本 probe 看可视抖动，不需要 ContactSensor reporter
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # 手固定且不受重力，避免把 probe 混成重力稳定性测试
                retain_accelerations=False,  # 与 generated hand spawn 参数一致
                enable_gyroscopic_forces=False,  # 与 generated hand spawn 参数一致
                angular_damping=0.01,  # 与 generated hand spawn 参数一致
                max_linear_velocity=1000.0,  # 与 generated hand spawn 参数一致
                max_angular_velocity=64.0 / math.pi * 180.0,  # 与 generated hand spawn 参数一致
                max_depenetration_velocity=1000.0,  # 接触穿透修正速度上限，与 generated hand probe 对齐
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,  # 保留官方资产自碰撞，便于观察是否同样诱发 slider jitter
                solver_position_iteration_count=8,  # 与 generated / GM probe 对齐
                solver_velocity_iteration_count=0,  # 与 generated / GM probe 对齐
                sleep_threshold=0.005,  # 与 generated / GM probe 对齐
                stabilization_threshold=0.0005,  # 与 generated / GM probe 对齐
                fix_root_link=True,  # root 固定，保持标定台不是 free-floating hand
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_ANCHOR_POS_E,  # root 高度与 generated hand anchor 对齐
            rot=OFFICIAL_LEAP_ROOT_ROT_WXYZ,  # 官方 LEAP raw URDF 的“手掌朝上”姿态
            joint_pos={".*": 0.0},  # 官方资产使用自己的 runtime joint names；不做 generated-name 映射
            joint_vel={".*": 0.0},  # 初始静止，避免 slider probe 叠加历史速度
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # 控制官方 URDF 暴露出的全部 revolute joints
                effort_limit_sim=0.95,  # 取官方 URDF effort 数值锚点；这里不绑定 generated preset
                velocity_limit_sim=8.48,  # 取官方 URDF velocity 数值锚点
                stiffness=3.0,  # 与 URDF converter drive gains 保持一致
                damping=0.1,  # 与 URDF converter drive gains 保持一致
                friction=0.01,  # 与 generated actuator probe 对齐
                armature=0.001,  # 与 generated actuator probe 对齐
            ),
        },
        soft_joint_pos_limit_factor=1.0,  # slider 上下界直接使用 asset joint limits
    )


def _build_object_spawn_cfg(object_source: str) -> Any:
    r"""构造标定 object 的 spawn cfg。

    Args:
        object_source (str): `local_cube` 或 `dex_cube_usd`。

    Returns:
        Any: IsaacLab spawn cfg；当前可能是 `CuboidCfg` 或 `UsdFileCfg`。

    Raises:
        ValueError: object source 未知。
    """

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,  # 保留 rigid object 动力学语义，导出 pose 可直接迁移到训练 reset
        disable_gravity=True,  # 标定台禁用重力；zero-action settle 应在独立 probe 中验证
        enable_gyroscopic_forces=True,  # 与 GM / LEAP object rigid props 保持一致
        solver_position_iteration_count=8,  # 与 GM / LEAP object 配置一致
        solver_velocity_iteration_count=0,  # 与 GM / LEAP object 配置一致
        sleep_threshold=0.005,  # 与训练 probe 对齐，避免隐藏物理参数差异
        stabilization_threshold=0.0025,  # 与训练 probe 对齐
        max_depenetration_velocity=1000.0,  # 与训练 probe 对齐
    )
    mass_props = sim_utils.MassPropertiesCfg(density=400.0)  # DexCube / local cube 共用密度锚点

    if object_source == "local_cube":
        return sim_utils.CuboidCfg(
            size=DEFAULT_LOCAL_CUBE_SIZE,  # 本地几何 cube，避免 Nucleus/S3 可用性影响 GUI 消融
            rigid_props=rigid_props,  # 与 DexCube USD object 的刚体参数保持一致
            mass_props=mass_props,  # 密度一致，使接触求解量级可比
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 显式添加 collision API，保证 RigidObject 可初始化
        )

    if object_source == "dex_cube_usd":
        return sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # 原始 DexCube 外观
            rigid_props=rigid_props,  # 与训练 probe 同一刚体参数
            mass_props=mass_props,  # DexCube 密度，与训练 probe 一致
            scale=DEFAULT_OBJECT_SCALE,  # object scale bucket，导出 metadata 也记录
        )

    raise ValueError(f"unknown object source: {object_source!r}")


def _load_yaml(path: Path) -> dict[str, Any]:
    r"""读取 YAML preset，返回普通 dict。

    Args:
        path (Path): YAML 文件路径。

    Returns:
        dict[str, Any]: YAML 顶层映射。

    Raises:
        ValueError: 文件不是 YAML mapping。
    """

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)  # 人工 preset 小文件，使用 safe loader 即可
    if not isinstance(payload, dict):
        raise ValueError(f"Preset {path} must contain a YAML mapping at top level.")
    return payload


def _select_start_preset(preset_arg: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    r"""选择启动时使用的 preset。

    Args:
        preset_arg (str | None): CLI `--preset`。

    Returns:
        tuple[Path | None, dict[str, Any] | None]: `(preset_path, payload)`；无 preset 时为 `(None, None)`。
    """

    # 用户显式指定 preset 时必须读取成功；这比 silently fallback 更安全，避免误以为恢复了旧标定。
    if preset_arg is not None:
        preset_path = Path(preset_arg).expanduser().resolve()
        return preset_path, _load_yaml(preset_path)

    # 未指定时尝试读取 latest.yaml；这支持“上次导出 -> 下次继续微调”的工作流。
    if LATEST_PRESET_PATH.exists():
        return LATEST_PRESET_PATH, _load_yaml(LATEST_PRESET_PATH)

    return None, None


def _joint_pos_from_preset(payload: dict[str, Any] | None) -> dict[str, float]:
    r"""从 preset payload 解析 joint pose，缺省时使用内置 pre-grasp。

    Args:
        payload (dict[str, Any] | None): YAML preset 顶层映射。

    Returns:
        dict[str, float]: 关节名到 rad 的映射。
    """

    if payload is None:
        return dict(DEFAULT_PREGRASP_JOINT_POS_RAD)
    joint_pos = payload.get("joint_pos_rad", None)
    if not isinstance(joint_pos, dict):
        return dict(DEFAULT_PREGRASP_JOINT_POS_RAD)
    return {str(joint_name): float(value) for joint_name, value in joint_pos.items()}


def _object_pose_from_preset(payload: dict[str, Any] | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    r"""从 preset payload 解析 object cfg pose，缺省时使用内置 contact-basin seed。

    Args:
        payload (dict[str, Any] | None): YAML preset 顶层映射。

    Returns:
        tuple[tuple[float, float, float], tuple[float, float, float]]: `(pos_cfg, rpy_xyz_rad)`。
    """

    if payload is None:
        return DEFAULT_OBJECT_POS_CFG, DEFAULT_OBJECT_RPY_XYZ

    object_pose_cfg = payload.get("object_pose_cfg", None)
    if not isinstance(object_pose_cfg, dict):
        return DEFAULT_OBJECT_POS_CFG, DEFAULT_OBJECT_RPY_XYZ

    pos_cfg = tuple(_as_float_list(object_pose_cfg.get("pos"), expected_len=3, field_name="object_pose_cfg.pos"))
    if "rpy_xyz_rad" in object_pose_cfg:
        rpy_xyz = tuple(
            _as_float_list(object_pose_cfg.get("rpy_xyz_rad"), expected_len=3, field_name="object_pose_cfg.rpy_xyz_rad")
        )
        return pos_cfg, rpy_xyz

    rot_wxyz = torch.tensor(
        _as_float_list(object_pose_cfg.get("rot_wxyz"), expected_len=4, field_name="object_pose_cfg.rot_wxyz"),
        dtype=torch.float32,
    )
    return pos_cfg, _rpy_xyz_from_quat(rot_wxyz)


def _hand_bundle_from_preset(payload: dict[str, Any] | None) -> str | None:
    r"""从 preset asset 字段读取 hand bundle。

    Args:
        payload (dict[str, Any] | None): YAML preset 顶层映射。

    Returns:
        str | None: preset 中记录的 bundle 路径；缺失时返回 None。
    """

    if payload is None:
        return None
    asset = payload.get("asset", None)
    if not isinstance(asset, dict):
        return None
    hand_bundle = asset.get("hand_bundle", None)
    return str(hand_bundle) if hand_bundle is not None else None


def _make_scene_cfg(
    hand_cfg: ArticulationCfg,
    object_source: str,
    object_pos_cfg: tuple[float, float, float],
    object_rpy_xyz: tuple[float, float, float],
    device: str,
) -> InteractiveSceneCfg:
    r"""构造单 env 标定 scene cfg。

    Args:
        hand_cfg (ArticulationCfg): 已完成 lower 的 hand articulation cfg；generated 与官方 URDF probe 共享此入口。
        object_source (str): object 来源，`local_cube` 避免远程依赖，`dex_cube_usd` 保留原始外观。
        object_pos_cfg (tuple[float, float, float]): object 初始位置，单位 m，cfg/env frame。
        object_rpy_xyz (tuple[float, float, float]): object 初始姿态，XYZ Euler rad。
        device (str): 仿真 device，用于把 rpy 转 quaternion。

    Returns:
        InteractiveSceneCfg: 可直接传给 `InteractiveScene` 的 cfg 实例。
    """

    object_quat = tuple(float(v) for v in _quat_from_rpy_xyz(object_rpy_xyz, device).cpu()[0].tolist())
    object_spawn_cfg = _build_object_spawn_cfg(object_source)  # 本地 cube 或远程 DexCube USD，按 CLI 控制

    @configclass
    class SingleAssetCalibrationSceneCfg(InteractiveSceneCfg):
        r"""标定台 scene：单手、单物体、地面和光源。

        本 scene 故意不安装 contact sensors / command marker / RL managers。标定时需要的是
        可直接操作的几何与 PhysX 状态，而不是完整训练 MDP。
        """

        robot: ArticulationCfg = hand_cfg
        """待标定的 hand articulation；可为 generated bundle 或官方 LEAP probe。"""

        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            spawn=object_spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos_cfg, rot=object_quat),  # cfg/env-frame 初始 pose
        )
        """标定 object；默认 local cube，必要时可切回远程 DexCube USD。"""

        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        """地面只作为视觉和物理兜底参照，不进入导出 preset。"""

        light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0),
        )
        """本地 dome light；不依赖远程 HDRI，保证标定台启动不被网络资源阻塞。"""

    return SingleAssetCalibrationSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)


def _apply_generated_collision_filter(scene: InteractiveScene, filter_mode: str) -> None:
    r"""给 generated hand 写入 stage-level collision filter 消融。

    该函数必须在 `sim.reset()` 前调用。PhysX 在 reset / handle 初始化阶段读取 USD
    physics schema；如果等到 reset 之后才 author filtering relationships，就可能出现
    GUI 里看见关系但 PhysX solver 没有采用的隐蔽错误。

    Args:
        scene (InteractiveScene): 已经 spawn 出 hand/object prim、但尚未 `sim.reset()` 的 scene。
        filter_mode (str): CLI `--generated-collision-filter` 的值。

    Raises:
        RuntimeError: 当前 USD build 缺少 `UsdPhysics.CollisionGroup`。
        ValueError: filter mode 未知。
    """

    if filter_mode == "none":
        return
    if filter_mode not in {"finger_palm", "finger_palm_same_finger"}:
        raise ValueError(f"unknown generated collision filter mode: {filter_mode!r}")

    # 这里用 external PhysicsCollisionGroup，而不是直接在 collider prim 上写
    # FilteredPairsAPI。URDF converter 可能把 collision mesh 放进 instance proxy 或更深层
    # 子树；collection include 指向 link prim 并使用 expandPrims，更能贴合“link-level
    # 结构接触过滤”的研究语义。
    if not hasattr(UsdPhysics, "CollisionGroup"):
        raise RuntimeError("Current USD build does not expose UsdPhysics.CollisionGroup")

    link_names = _generated_collision_filter_link_names(filter_mode)  # 本轮消融需要建 group 的 link 集合
    link_group_paths = _author_generated_link_collision_groups(scene, link_names)  # link name -> group prim path
    filtered_link_pairs = _generated_filtered_link_pairs(filter_mode)  # 无向 link pair 集合 $\mathcal{F}$

    authored_group_edges = 0  # 有向 filteredGroups edge 数；每个无向 pair 通常写双向 edge
    missing_link_names: set[str] = set()  # URDF/USD schema 不匹配时打印 warning，避免 silent wrong ablation

    # 把 link-level pair 规则写成 collision-group filteredGroups；双向写入避免 USD/PhysX 解释差异。
    for link_a, link_b in filtered_link_pairs:
        group_a_path = link_group_paths.get(link_a)
        group_b_path = link_group_paths.get(link_b)
        if group_a_path is None or group_b_path is None:
            missing_link_names.update(
                link for link, group_path in ((link_a, group_a_path), (link_b, group_b_path)) if group_path is None
            )
            continue
        authored_group_edges += _author_generated_filtered_group_edge(scene.stage, group_a_path, group_b_path)
        authored_group_edges += _author_generated_filtered_group_edge(scene.stage, group_b_path, group_a_path)

    if missing_link_names:
        print(
            "[WARN]: generated collision filter skipped missing hand links: "
            f"{sorted(missing_link_names)}"
        )

    print(
        "[INFO]: generated collision filter authored "
        f"mode={filter_mode!r}, groups={len(link_group_paths)}, "
        f"link_pairs={len(filtered_link_pairs)}, directed_edges={authored_group_edges}"
    )


def _generated_collision_filter_link_names(filter_mode: str) -> tuple[str, ...]:
    r"""返回当前 generated collision 消融需要建 group 的 link 名。

    Args:
        filter_mode (str): `none`、`finger_palm` 或 `finger_palm_same_finger`。

    Returns:
        tuple[str, ...]: 去重排序后的 link names；排序只服务日志和 stage diff 可复现。
    """

    if filter_mode == "none":
        return tuple()

    link_names = {"palm"}  # palm 是 finger-palm filter 的固定一端
    for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
        link_names.update(finger_link_chain)  # 所有 finger link 都需要能与 palm 建过滤 pair
    return tuple(sorted(link_names))


def _generated_filtered_link_pairs(filter_mode: str) -> tuple[tuple[str, str], ...]:
    r"""构造 generated-hand 的 link-level 过滤集合 $\mathcal{F}$。

    过滤集合按当前消融问题定义：

    $$
    \mathcal{F}_{palm}
      = \{(\text{palm}, l)\mid l \in \cup_f F_f\}
    $$

    若 `filter_mode == "finger_palm_same_finger"`，再加入：

    $$
    \mathcal{F}_{same}
      = \bigcup_f \{(a,b)\mid a,b\in F_f,\ a\ne b\}.
    $$

    注意这里没有加入 $F_i \times F_j,\ i\ne j$，所以不同 fingers 之间仍然保留碰撞。

    Args:
        filter_mode (str): `none`、`finger_palm` 或 `finger_palm_same_finger`。

    Returns:
        tuple[tuple[str, str], ...]: 去重排序后的无向 link pair。
    """

    filtered_pairs: set[tuple[str, str]] = set()
    if filter_mode == "none":
        return tuple()

    # palm-finger 全部过滤：这是用户本轮想看的“掌心是否把 generated finger 顶抖”的消融变量。
    for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
        filtered_pairs.update(tuple(sorted(("palm", link_name))) for link_name in finger_link_chain)

    # same-finger 内部 link 两两过滤：把单根 finger 当作由 joint 约束定义的机构链，
    # 不让相邻/同链碰撞 mesh 的轻微穿插反过来驱动整根手指抖动。
    if filter_mode == "finger_palm_same_finger":
        for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
            filtered_pairs.update(tuple(sorted(pair)) for pair in combinations(finger_link_chain, 2))

    return tuple(sorted(filtered_pairs))


def _author_generated_link_collision_groups(
    scene: InteractiveScene,
    link_names: tuple[str, ...],
) -> dict[str, str]:
    r"""为 generated hand 的每个 link 建一个外部 `PhysicsCollisionGroup`。

    Args:
        scene (InteractiveScene): 已 spawn 的单 env scene。
        link_names (tuple[str, ...]): 需要纳入 collision filter 规则的 generated link names。

    Returns:
        dict[str, str]: `link_name -> collision_group_path`；若 link prim 不存在则跳过。
    """

    stage = scene.stage  # 当前 USD stage；此时 env/Robot/link prim 已经存在
    root_layer = stage.GetRootLayer()  # 与 Isaac Sim cloner 一样把过滤 schema 固定写入 root layer
    link_group_paths: dict[str, str] = {}

    # 先定义 scope，后续 Sdf.PrimSpec 批量往该 scope 下写 PhysicsCollisionGroup。
    with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
        UsdGeom.Scope.Define(stage, GENERATED_COLLISION_GROUP_ROOT)

    collision_group_root_spec = root_layer.GetPrimAtPath(GENERATED_COLLISION_GROUP_ROOT)
    if collision_group_root_spec is None:
        raise RuntimeError(f"Failed to define collision group scope at {GENERATED_COLLISION_GROUP_ROOT}")

    # Sdf.ChangeBlock 减少逐条 author relationship 时的 stage notice 噪声；本脚本只有单 env，
    # 但沿用训练 env 里已验证的写法，避免未来改成多 env 标定时再重写。
    with Sdf.ChangeBlock():
        for link_name in link_names:
            first_env_link_path = _generated_link_prim_path(scene.env_prim_paths[0], link_name)
            if not stage.GetPrimAtPath(first_env_link_path).IsValid():
                continue

            collision_group = Sdf.PrimSpec(
                collision_group_root_spec,
                link_name,
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            for env_prim_path in scene.env_prim_paths:
                includes_rel.targetPathList.Append(_generated_link_prim_path(env_prim_path, link_name))

            link_group_paths[link_name] = f"{GENERATED_COLLISION_GROUP_ROOT}/{link_name}"

    return link_group_paths


def _generated_link_prim_path(env_prim_path: str, link_name: str) -> str:
    r"""构造 generated hand link prim 在一个 cloned env 内的 path。

    Args:
        env_prim_path (str): cloned env root path，例如 `/World/envs/env_0`。
        link_name (str): generated URDF link name，例如 `index_mcp1`。

    Returns:
        str: link prim path；collision group collection 会用 `expandPrims` 纳入其下 collider。
    """

    return f"{env_prim_path}/Robot/{link_name}"


def _author_generated_filtered_group_edge(stage, source_group_path: str, target_group_path: str) -> int:
    r"""写入一条有向 `physics:filteredGroups` edge。

    Args:
        stage: 当前 USD stage。
        source_group_path (str): 源 `PhysicsCollisionGroup` path。
        target_group_path (str): 需要与源 group 禁止碰撞的目标 group path。

    Returns:
        int: 若新增 target 返回 1；若 target 已存在返回 0。
    """

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        source_group = UsdPhysics.CollisionGroup.Get(stage, source_group_path)
        if not source_group:
            raise RuntimeError(f"Missing generated collision group at {source_group_path}")

        filtered_groups_rel = source_group.GetFilteredGroupsRel()
        if not filtered_groups_rel:
            filtered_groups_rel = source_group.CreateFilteredGroupsRel()

        target_path = Sdf.Path(target_group_path)  # relationship target 指向另一个 PhysicsCollisionGroup prim
        if target_path in set(filtered_groups_rel.GetTargets()):
            return 0

        filtered_groups_rel.AddTarget(target_path)
        return 1


class SingleAssetGraspCalibrationPanel:
    r"""Isaac Sim GUI 面板，负责 joint/object pose 标定与导出。

    面板维护三组状态：

    1. `joint_targets`: hand articulation 的当前标定关节角，形状 `[1,n_q]`，单位 rad；
    2. `object_pos_cfg`: object root 在 env frame `{e}` 下的位置，单位 m；
    3. `object_rpy_xyz`: object root 姿态的 XYZ Euler angle，单位 rad。
    4. `reset_*`: 本次脚本启动时的快照；`Apply Reset` 回到该快照，而不是重新读磁盘。

    每次点击导出时，脚本同时写：

    $$
    T^e_o \rightarrow \texttt{object\_pose\_cfg},\qquad
    T^h_o = (T^w_h)^{-1} T^w_o \rightarrow \texttt{object\_pose\_h}.
    $$

    其中 $T^w_h$ 由 hand root pose $T^w_a$ 与 `HandFrameCfg` 中的 $T^h_a$ 推导。
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
        loaded_preset_path: Path | None,
    ) -> None:
        r"""初始化标定面板与内部状态。

        Args:
            scene (InteractiveScene): 单 env 标定 scene。
            robot (Articulation): hand articulation。
            obj (RigidObject): DexCube rigid object。
            hand_frame_cfg (HandFrameCfg): `{a}` 到 `{h}` 的 frame 语义；官方 URDF probe 中仅作为 raw root 对照 frame。
            hand_asset_ref (str): 当前 hand asset 引用；generated 为 bundle path，official probe 为 URDF path。
            hand_source (str): hand 资产来源标签，例如 `generated_bundle`、`official_leap_usd` 或 `official_leap_urdf`。
            object_source (str): object 来源标签，`local_cube` 或 `dex_cube_usd`。
            initial_joint_pos (dict[str, float]): 初始 joint preset，单位 rad。
            initial_object_pos_cfg (tuple[float, float, float]): 初始 object 位置，单位 m。
            initial_object_rpy_xyz (tuple[float, float, float]): 初始 object Euler 姿态，单位 rad。
            output_name (str | None): CLI 指定的导出文件名。
            loaded_preset_path (Path | None): 启动时加载的 preset 路径；仅用于 UI 状态显示。
        """

        self.scene = scene  # IsaacLab scene；提供 env origin 与 stage handle
        self.robot = robot  # hand articulation；写 joint state / target
        self.obj = obj  # DexCube rigid object；写 root pose / velocity
        self.hand_frame_cfg = hand_frame_cfg  # frame 语义来源，导出 `{h}` / probe-root pose 时使用
        self.hand_asset_ref = hand_asset_ref  # 资产路径写入 preset，确保导出可复现
        self.hand_source = hand_source  # 资产来源标签；避免把官方 URDF probe 误读为 generated bundle
        self.object_source = object_source  # object 来源标签；local cube / DexCube USD 的 contact 结论要分开读
        self.output_name = output_name  # 可选固定文件名；None 时自动时间戳
        self.loaded_preset_path = loaded_preset_path  # 启动来源，仅用于人读状态

        self.device = robot.device  # torch device；与 IsaacLab buffers 对齐
        self.joint_names = list(robot.joint_names)  # runtime joint 顺序；导出时保持 name dict 避免顺序歧义
        self.num_joints = len(self.joint_names)  # DOF 数；当前 LEAP-like mother 为 16
        self.joint_limits = robot.data.joint_pos_limits[0].detach().cpu()  # `[n_q,2]`，rad，slider 上下界

        self.joint_targets = robot.data.default_joint_pos.clone()  # `[1,n_q]`，先从 URDF/default pose 出发
        self.joint_velocities = torch.zeros_like(robot.data.default_joint_vel)  # `[1,n_q]`，标定状态默认静止
        self.object_pos_cfg = torch.tensor(initial_object_pos_cfg, dtype=torch.float32, device=self.device)  # `{e}` pos
        self.object_rpy_xyz = torch.tensor(initial_object_rpy_xyz, dtype=torch.float32, device=self.device)  # XYZ Euler rad
        self.lock_object_pose = True  # 默认锁住 object，防止 gravity 在编辑时把 cube 拉走

        self.joint_sliders: dict[str, dict[str, Any]] = {}  # joint name -> slider/label/index
        self.object_sliders: dict[str, dict[str, Any]] = {}  # object pose field -> slider/label/index
        self.status_label: ui.Label | None = None  # 底部状态行；导出/读取后更新
        self.lock_label: ui.Label | None = None  # object lock 状态显示

        # 把 name-based preset 写入 runtime joint vector。未出现在 preset 中的关节保留 asset default，
        # 这样未来换一个 DOF 命名略有差异的单资产时不会直接崩溃。
        for joint_name, joint_value in initial_joint_pos.items():
            if joint_name in self.joint_names:
                joint_idx = self.joint_names.index(joint_name)  # runtime column id
                self.joint_targets[0, joint_idx] = float(joint_value)  # rad

        # 保存“本次脚本启动时”的 reset 快照。它不同于磁盘上的 latest.yaml：用户在本轮 GUI
        # 中调乱后点 `Apply Reset`，应回到刚启动时看到的姿态，而不是受后续导出覆盖影响。
        self.reset_joint_targets = self.joint_targets.clone()  # `[1,n_q]`，启动时 joint pose 快照，rad
        self.reset_object_pos_cfg = self.object_pos_cfg.clone()  # `[3]`，启动时 object cfg/env position，m
        self.reset_object_rpy_xyz = self.object_rpy_xyz.clone()  # `[3]`，启动时 object XYZ Euler，rad
        self.reset_lock_object_pose = True  # 启动默认 lock；reset 后也恢复到可稳定编辑的锁定状态

        self._build_ui()
        self.apply_joint_state()
        self.apply_object_pose()
        self._set_status("Calibrator ready. Adjust joints/object, then Export Preset.")

    def _build_ui(self) -> None:
        r"""创建 Isaac Sim `omni.ui` 标定窗口。

        UI 结构按科研工作流组织，而不是按代码模块组织：

        1. 顶部显示当前资产 / preset 来源；
        2. 中部左侧逻辑为 pre-grasp joint sliders；
        3. 中部右侧逻辑为 object contact-basin pose sliders；
        4. 底部是 Apply / Read / Export 等工作流按钮。
        """

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
        r"""创建按手指分组的 joint sliders。

        分组只影响 UI 可读性；导出仍是 `joint_name -> rad` 字典，避免把 runtime order
        当作隐式 contract 写进 preset。
        """

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
                        joint_idx = self.joint_names.index(joint_name)
                        self._create_joint_slider(joint_name, joint_idx)

    def _create_joint_slider(self, joint_name: str, joint_idx: int) -> None:
        r"""创建单个 joint slider。

        Args:
            joint_name (str): runtime joint name。
            joint_idx (int): runtime joint column index。
        """

        lower = float(self.joint_limits[joint_idx, 0].item())  # rad，下限
        upper = float(self.joint_limits[joint_idx, 1].item())  # rad，上限
        initial = float(self.joint_targets[0, joint_idx].item())  # rad，当前 preset 值

        with ui.HStack(spacing=6, height=24):
            ui.Label(f"{joint_name}", width=92, style={"color": 0xFFDDDDDD})
            slider = ui.FloatSlider(min=lower, max=upper, width=ui.Fraction(0.58), height=20)
            value_label = ui.Label("", width=140, alignment=ui.Alignment.LEFT, style={"font_size": 11})

            def on_value_changed(model, idx=joint_idx, label=value_label) -> None:
                value = float(model.as_float)  # rad，slider 模型值
                self.joint_targets[0, idx] = value  # `[1,n_q]`，写入 runtime target vector
                label.text = f"{value:+.4f} rad / {math.degrees(value):+6.1f} deg"
                self.apply_joint_state()

            slider.model.add_value_changed_fn(on_value_changed)
            slider.model.set_value(initial)
            value_label.text = f"{initial:+.4f} rad / {math.degrees(initial):+6.1f} deg"
            self.joint_sliders[joint_name] = {"slider": slider, "label": value_label, "index": joint_idx}

    def _create_object_controls(self) -> None:
        r"""创建 object contact-basin pose sliders。

        slider 直接维护 `object_pose_cfg`，也就是短期要填进 IsaacLab env cfg 的 pose；
        导出时再额外计算 `object_pose_h`，供未来 reset/cache 语义使用。
        """

        ui.Label("Object Contact Basin Pose", height=24, style={"font_size": 16, "color": 0xFF88CCFF})
        self.lock_label = ui.Label("", height=20, style={"font_size": 12, "color": 0xFFFFFF99})
        self._refresh_lock_label()

        object_specs = [
            ("x", "pos", 0, -0.12, 0.12, "m"),  # 左右方向搜索范围，单位 m
            ("y", "pos", 1, -0.02, 0.16, "m"),  # 掌心侧/指腹侧搜索范围，单位 m
            ("z", "pos", 2, 0.45, 0.68, "m"),  # 物体高度搜索范围，单位 m
            ("roll", "rpy", 0, -math.pi, math.pi, "rad"),  # object 绕 cfg x 轴姿态，rad
            ("pitch", "rpy", 1, -math.pi, math.pi, "rad"),  # object 绕 cfg y 轴姿态，rad
            ("yaw", "rpy", 2, -math.pi, math.pi, "rad"),  # object 绕 cfg z 轴姿态，rad
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
        r"""创建 object pose 单个 slider。

        Args:
            name (str): UI 字段名，例如 `x` 或 `yaw`。
            group (str): `pos` 或 `rpy`。
            idx (int): 在对应三维向量中的索引。
            lower (float): slider 下限。
            upper (float): slider 上限。
            unit (str): UI 显示单位。
        """

        initial = float((self.object_pos_cfg if group == "pos" else self.object_rpy_xyz)[idx].item())
        with ui.HStack(spacing=6, height=24):
            ui.Label(f"{name}", width=60, style={"color": 0xFFDDDDDD})
            slider = ui.FloatSlider(min=lower, max=upper, width=ui.Fraction(0.62), height=20)
            value_label = ui.Label("", width=170, alignment=ui.Alignment.LEFT, style={"font_size": 11})

            def on_value_changed(model, pose_group=group, pose_idx=idx, label=value_label, field=name) -> None:
                value = float(model.as_float)  # m 或 rad
                if pose_group == "pos":
                    self.object_pos_cfg[pose_idx] = value  # `{e}` position, meter
                    label.text = f"{value:+.5f} m"
                else:
                    self.object_rpy_xyz[pose_idx] = value  # XYZ Euler angle, rad
                    label.text = f"{value:+.4f} rad / {math.degrees(value):+6.1f} deg"
                self.apply_object_pose()
                self._set_status(f"Updated object {field}; pose is locked to UI state.")

            slider.model.add_value_changed_fn(on_value_changed)
            slider.model.set_value(initial)
            if group == "pos":
                value_label.text = f"{initial:+.5f} {unit}"
            else:
                value_label.text = f"{initial:+.4f} rad / {math.degrees(initial):+6.1f} deg"
            self.object_sliders[name] = {"slider": slider, "label": value_label, "group": group, "index": idx}

    def _create_action_buttons(self) -> None:
        r"""创建底部工作流按钮。

        按钮语义刻意保持少而清晰：

        - `Apply Preset`：把 UI 状态重新写入 sim；
        - `Apply Reset`：回到脚本启动时加载的 joint/object pose；
        - `Toggle Object Lock`：允许用户临时用 gizmo 拖动 object；
        - `Read Object From Stage`：把 gizmo 后的 stage pose 吸收到 UI 状态；
        - `Export Preset`：写 YAML 并打印 cfg 片段。
        """

        with ui.HStack(spacing=8, height=38):
            ui.Button("Apply Preset", clicked_fn=self.apply_all, height=32, style={"background_color": 0xFF336699})
            ui.Button("Apply Reset", clicked_fn=self.apply_reset, height=32, style={"background_color": 0xFF996633})
        with ui.HStack(spacing=8, height=38):
            ui.Button("Toggle Object Lock", clicked_fn=self.toggle_object_lock, height=32, style={"background_color": 0xFF665533})
            ui.Button("Read Object From Stage", clicked_fn=self.read_object_from_stage, height=32, style={"background_color": 0xFF446644})
        with ui.HStack(spacing=8, height=38):
            ui.Button("Export Preset", clicked_fn=self.export_preset, height=32, style={"background_color": 0xFF884488})

        self.status_label = ui.Label("", height=36, style={"font_size": 12, "color": 0xFFFFFFFF})

    def _set_status(self, text: str) -> None:
        r"""更新 UI 状态行并同步打印到终端。

        Args:
            text (str): 状态消息。
        """

        if self.status_label is not None:
            self.status_label.text = text
        print(f"[Calibrator] {text}")

    def _refresh_lock_label(self) -> None:
        r"""刷新 object lock 状态显示。"""

        if self.lock_label is not None:
            state = "LOCKED" if self.lock_object_pose else "UNLOCKED"
            hint = "unlock before using IsaacSim gizmo" if self.lock_object_pose else "move cube with gizmo, then Read Object From Stage"
            self.lock_label.text = f"Object pose lock: {state} ({hint})"

    def apply_joint_state(self) -> None:
        r"""把当前 joint preset 直接写入 hand state 与 PD target。

        这里同时写 state 与 target，避免 reset 后出现：

        $$
        q_{\text{state}} \ne q_{\text{target}}
        $$

        导致 PD controller 在下一步把手拉离人工标定姿态。
        """

        self.robot.write_joint_state_to_sim(self.joint_targets, self.joint_velocities)
        self.robot.set_joint_position_target(self.joint_targets)

    def _object_quat_wxyz(self) -> torch.Tensor:
        r"""返回当前 UI object rpy 对应的 quaternion。

        Returns:
            torch.Tensor: object quaternion，形状 `[1,4]`，顺序 `(w,x,y,z)`。
        """

        rpy = tuple(float(v) for v in self.object_rpy_xyz.detach().cpu().tolist())
        return _quat_from_rpy_xyz(rpy, self.device)

    def _object_pose_w(self) -> torch.Tensor:
        r"""把 UI 中的 cfg/env pose 转成 IsaacLab write API 需要的 world pose。

        Returns:
            torch.Tensor: root pose，形状 `[1,7]`，`[x,y,z,w,qx,qy,qz]`。
        """

        env_origin = self.scene.env_origins[0].to(self.device)  # `[3]`，当前单 env 的 world 平移
        pos_w = self.object_pos_cfg + env_origin  # `{w}` position，meter
        quat_wxyz = self._object_quat_wxyz()[0]  # `[4]`，object orientation
        return torch.cat((pos_w, quat_wxyz), dim=0).reshape(1, 7)

    def apply_object_pose(self) -> None:
        r"""把当前 object UI pose 写入 PhysX，并清零速度。"""

        root_pose = self._object_pose_w()  # `[1,7]`，world-frame root pose
        zero_velocity = torch.zeros(1, 6, dtype=torch.float32, device=self.device)  # `[1,6]`，线/角速度均为 0
        self.obj.write_root_pose_to_sim(root_pose)
        self.obj.write_root_velocity_to_sim(zero_velocity)

    def apply_all(self) -> None:
        r"""把 joint 与 object UI 状态一起写回仿真。"""

        self.apply_joint_state()
        self.apply_object_pose()
        self._set_status("Applied current joint/object preset to simulation.")

    def apply_reset(self) -> None:
        r"""回到脚本启动时加载的 joint/object 标定状态。

        `Apply Preset` 的含义是“应用当前 UI 状态”，而本函数的含义是“撤回到本轮
        会话初始状态”。二者分开可以支持一个高频人工工作流：

        1. 启动脚本加载内置 seed 或 `latest.yaml`；
        2. 大幅拖动关节 / 物体 pose 做探索；
        3. 如果探索方向不对，点 `Apply Reset` 直接回到本轮起点。
        """

        self.joint_targets[:] = self.reset_joint_targets  # `[1,n_q]`，恢复启动时 joint pose，rad
        self.object_pos_cfg[:] = self.reset_object_pos_cfg  # `[3]`，恢复启动时 object cfg/env position，m
        self.object_rpy_xyz[:] = self.reset_object_rpy_xyz  # `[3]`，恢复启动时 object XYZ Euler，rad
        self.lock_object_pose = bool(self.reset_lock_object_pose)  # reset 后回到锁定，防止 cube 自发漂移

        self._sync_joint_sliders_from_state()
        self._sync_object_sliders_from_state()
        self._refresh_lock_label()
        self.apply_all()
        self._set_status("Reset to the pose loaded at script startup.")

    def toggle_object_lock(self) -> None:
        r"""切换 object pose lock。

        解锁后用户可以用 IsaacSim gizmo 拖动 object；拖完后点击 `Read Object From Stage`
        将 stage 当前 pose 同步回 UI 状态和导出状态。
        """

        self.lock_object_pose = not self.lock_object_pose
        self._refresh_lock_label()
        self._set_status("Object pose lock toggled.")

    def read_object_from_stage(self) -> None:
        r"""从 USD stage / gizmo 当前状态读取 object pose，并同步回 UI sliders。

        该函数用于人工“拖物体到手上”的工作流：

        1. 点击 `Toggle Object Lock` 解锁；
        2. 用 IsaacSim gizmo 移动 / 旋转 cube；
        3. 点击本按钮，把当前 stage pose 吸收到 `object_pose_cfg`；
        4. 可继续锁住并导出。
        """

        object_prim_path = self.obj.root_physx_view.prim_paths[0]  # 真实 rigid body root prim path
        object_prim = self.scene.stage.GetPrimAtPath(object_prim_path)  # USD prim handle
        pos_w_tuple, quat_w_tuple = sim_utils_runtime.resolve_prim_pose(object_prim)  # world pose, wxyz quat

        pos_w = torch.tensor(pos_w_tuple, dtype=torch.float32, device=self.device)  # `[3]`，world position
        quat_wxyz = torch.tensor(quat_w_tuple, dtype=torch.float32, device=self.device)  # `[4]`，world orientation
        env_origin = self.scene.env_origins[0].to(self.device)  # `[3]`，env origin in world
        self.object_pos_cfg[:] = pos_w - env_origin  # cfg/env-frame position
        self.object_rpy_xyz[:] = torch.tensor(_rpy_xyz_from_quat(quat_wxyz), dtype=torch.float32, device=self.device)

        self._sync_object_sliders_from_state()
        self.apply_object_pose()
        self.lock_object_pose = True
        self._refresh_lock_label()
        self._set_status(f"Read object pose from stage: {object_prim_path}; lock restored.")

    def _sync_object_sliders_from_state(self) -> None:
        r"""把 object 内部状态写回 UI slider model。"""

        for name, entry in self.object_sliders.items():
            group = entry["group"]
            idx = entry["index"]
            value = float((self.object_pos_cfg if group == "pos" else self.object_rpy_xyz)[idx].item())
            entry["slider"].model.set_value(value)
            if group == "pos":
                entry["label"].text = f"{value:+.5f} m"
            else:
                entry["label"].text = f"{value:+.4f} rad / {math.degrees(value):+6.1f} deg"

    def _sync_joint_sliders_from_state(self) -> None:
        r"""把 joint 内部状态写回 UI slider model。

        `Apply Reset` 会直接覆盖 `joint_targets` tensor；如果不反向同步 slider，
        UI 显示值与真实 sim state 会分叉，下一次轻微拖动 slider 又会把旧 UI 值写回。
        """

        for joint_name, entry in self.joint_sliders.items():
            idx = entry["index"]  # runtime joint column id
            value = float(self.joint_targets[0, idx].item())  # rad，当前内部 joint target
            entry["slider"].model.set_value(value)
            entry["label"].text = f"{value:+.4f} rad / {math.degrees(value):+6.1f} deg"

    def step_maintenance(self) -> None:
        r"""每帧维护标定状态。

        当前只在 object lock 开启时持续写 object pose / zero velocity。这样可以让用户
        调 joint slider 时 cube 不因重力掉出手掌区域，从而把“视觉标定”与“物理验证”
        暂时分离。
        """

        if self.lock_object_pose:
            self.apply_object_pose()
        else:
            zero_velocity = torch.zeros(1, 6, dtype=torch.float32, device=self.device)  # `[1,6]`，gizmo 编辑时仍抑制漂移速度
            self.obj.write_root_velocity_to_sim(zero_velocity)  # 不锁 pose，只清零速度，尽量不妨碍 gizmo 改位姿

    def _hand_semantic_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""计算 hand semantic frame `{h}` 在 world `{w}` 下的 pose。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(p_wh, q_wh)`，形状分别为 `[1,3]` 与 `[1,4]`。
        """

        root_pos_w = self.robot.data.root_pos_w[:1]  # `[1,3]`，raw asset/root frame `{a}` origin in world
        root_quat_w = self.robot.data.root_quat_w[:1]  # `[1,4]`，$R_{wa}$，wxyz

        r_ha = torch.tensor(self.hand_frame_cfg.semantic_R_ha, dtype=torch.float32, device=self.device).reshape(3, 3)
        p_ha = torch.tensor(self.hand_frame_cfg.semantic_p_ha, dtype=torch.float32, device=self.device).reshape(1, 3)
        q_ha = math_utils.quat_from_matrix(r_ha.reshape(1, 3, 3))  # `$R_{ha}$` -> quaternion

        q_ah = math_utils.quat_inv(q_ha)  # `$R_{ah}=R_{ha}^{-1}$`
        p_ah = math_utils.quat_apply(q_ah, -p_ha)  # `$p_{ah}=-R_{ah}p_{ha}$`
        pos_wh, quat_wh = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, p_ah, q_ah)
        return pos_wh, quat_wh

    def _object_pose_h(self) -> tuple[list[float], list[float]]:
        r"""计算 object pose 在 hand semantic frame `{h}` 下的表达。

        Returns:
            tuple[list[float], list[float]]: `(pos_h, quat_h)`，分别为 3D 位置和 `wxyz` quaternion。
        """

        pos_wh, quat_wh = self._hand_semantic_pose_w()  # hand semantic pose in world
        object_pose_w = self._object_pose_w()  # object pose in world from UI state
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
        r"""返回当前 joint preset 的 name -> rad 字典。"""

        return {joint_name: _round_float(self.joint_targets[0, idx].item()) for idx, joint_name in enumerate(self.joint_names)}

    def _joint_pos_deg_dict(self) -> dict[str, float]:
        r"""返回当前 joint preset 的 name -> degree 字典，便于与人工 GUI 读数对照。"""

        return {
            joint_name: _round_float(math.degrees(self.joint_targets[0, idx].item()), ndigits=4)
            for idx, joint_name in enumerate(self.joint_names)
        }

    def _export_path(self) -> Path:
        r"""解析本次导出路径。

        Returns:
            Path: 目标 YAML 文件路径。
        """

        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        if self.output_name:
            name = self.output_name if self.output_name.endswith((".yaml", ".yml")) else f"{self.output_name}.yaml"
        else:
            name = f"single_asset_grasp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        return PRESET_DIR / name

    def _preset_payload(self) -> dict[str, Any]:
        r"""构造 YAML preset payload。"""

        object_quat = self._object_quat_wxyz().detach().cpu()[0].tolist()
        object_pose_h_pos, object_pose_h_quat = self._object_pose_h()
        return {
            "schema_version": 1,  # preset schema 版本；未来字段迁移时递增
            "kind": "anymani_single_asset_grasp_preset",  # 明确不是 `grasp_cache` shard
            "asset": {
                "hand_source": self.hand_source,  # 资产来源；official probe 不应被误读为 generated bundle
                "hand_ref": self.hand_asset_ref,  # 复现实验所需 hand asset 引用，可能是 bundle path 或 raw URDF path
                "hand_bundle": self.hand_asset_ref if self.hand_source == "generated_bundle" else None,  # generated 主线兼容字段
                "hand_usd": self.hand_asset_ref if self.hand_source == "official_leap_usd" else None,  # official USD 消融 probe 字段
                "hand_urdf": self.hand_asset_ref if self.hand_source == "official_leap_urdf" else None,  # raw URDF fallback 字段
                "object_source": self.object_source,  # local cube 消融和 DexCube USD 标定不能混读
                "object_id": "dex_cube" if self.object_source == "dex_cube_usd" else "local_cube",  # object 语义 ID
                "object_scale": [_round_float(v) for v in DEFAULT_OBJECT_SCALE]
                if self.object_source == "dex_cube_usd"
                else None,  # DexCube scale bucket，影响 contact basin
                "object_size": [_round_float(v) for v in DEFAULT_LOCAL_CUBE_SIZE]
                if self.object_source == "local_cube"
                else None,  # local cuboid 边长，单位 m
            },
            "joint_pos_rad": self._joint_pos_rad_dict(),  # 训练/reset 主用关节位置，单位 rad
            "joint_pos_deg": self._joint_pos_deg_dict(),  # 人工核对辅助读数，单位 degree
            "object_pose_cfg": {
                "pos": [_round_float(v) for v in self.object_pos_cfg.detach().cpu().tolist()],  # `{e}` position，可直接填 cfg
                "rot_wxyz": [_round_float(v) for v in object_quat],  # `{e}` orientation quaternion，IsaacLab `wxyz`
                "rpy_xyz_rad": [_round_float(v) for v in self.object_rpy_xyz.detach().cpu().tolist()],  # UI slider 姿态源
            },
            "object_pose_h": {
                "pos": object_pose_h_pos,  # `$p^h_o$`，未来 cache/reset 语义使用
                "rot_wxyz": object_pose_h_quat,  # `$q^h_o$`，未来 cache/reset 语义使用
            },
            "notes": {
                "exported_at": datetime.now().isoformat(timespec="seconds"),  # 人工导出时间，非随机种子
                "source": "source/anymani/anymani/tools/single_asset_grasp_calibrator.py",  # 生成脚本相对路径
                "validated_cache": False,  # 明确还未经过物理 settle/robustness 验证
                "comment": "Manual contact-basin/pre-grasp seed; not a settled grasp-cache shard.",  # 人读提醒
            },
        }

    def export_preset(self) -> None:
        r"""导出当前标定 preset 到 YAML，并在终端打印 cfg 片段。"""

        payload = self._preset_payload()
        export_path = self._export_path()
        with export_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        with LATEST_PRESET_PATH.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

        self._print_cfg_snippets(payload)
        self._set_status(f"Exported preset: {export_path}; latest.yaml refreshed.")

    def _print_cfg_snippets(self, payload: dict[str, Any]) -> None:
        r"""向终端打印可复制到 env cfg / reset event 的 Python 片段。

        Args:
            payload (dict[str, Any]): 刚导出的 YAML payload。
        """

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


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    panel: SingleAssetGraspCalibrationPanel,
    smoke_seconds: float | None = None,
) -> None:
    r"""运行 GUI 标定仿真循环。

    Args:
        sim (SimulationContext): IsaacLab simulation context。
        scene (InteractiveScene): 单 env 标定 scene。
        panel (SingleAssetGraspCalibrationPanel): GUI / 状态维护面板。
        smoke_seconds (float | None): 若不为 `None`，表示 ready 后按 wall-clock 跑若干秒并自终止。
    """

    sim_dt = sim.get_physics_dt()  # physics step，单位 s
    smoke_start = time.monotonic() if smoke_seconds is not None else None  # wall-clock 起点，只服务自动 smoke
    while simulation_app.is_running():
        panel.step_maintenance()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if smoke_start is not None and time.monotonic() - smoke_start >= float(smoke_seconds):
            panel._set_status(f"Smoke completed after {smoke_seconds:.2f}s; hard-exiting smoke process.")
            # Isaac Sim / Kit 的正常 shutdown 有时会在 extension teardown 阶段拖很久。
            # smoke 的研究语义只需要证明“scene 已创建、sim.reset 已通过、主循环已跑”；
            # 因此这里直接结束当前 Python 进程，避免外部 bash timeout 或用户手动 kill。
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def main() -> None:
    r"""脚本入口：解析 preset / asset，创建 scene 与 GUI，并进入仿真循环。"""

    preset_path, preset_payload = _select_start_preset(args_cli.preset)  # 启动时恢复 preset；无则使用内置 seed
    preset_hand_bundle = _hand_bundle_from_preset(preset_payload)  # YAML 中记录的资产路径，可被 CLI 覆盖

    # hand asset 分支是本次抖动消融的控制变量：generated 主线继续使用 AnyMani bundle
    # adapter；official probe 只替换 robot asset，不继承 generated 的关节命名/映射假设。
    # 默认 official 路径使用旧 LEAP env 已引用的预转换 USD；raw URDF conversion 在当前
    # PhysX 里可能因为 collision mesh cooking / articulation root 失败而污染消融结论。
    if args_cli.official_leap_urdf:
        if args_cli.generated_collision_filter != "none":
            raise ValueError("--generated-collision-filter only applies to AnyMani generated hands, not official LEAP probes.")
        if args_cli.hand_bundle is not None:
            raise ValueError("--official-leap-urdf is an asset-level ablation; do not combine it with --hand-bundle.")
        if args_cli.official_leap_urdf_path is not None and args_cli.official_leap_usd_path is not None:
            raise ValueError("Use either --official-leap-urdf-path or --official-leap-usd-path, not both.")
        if args_cli.official_leap_urdf_path is not None:
            official_urdf_path = _resolve_official_leap_urdf_input(args_cli.official_leap_urdf_path)  # raw official LEAP URDF
            hand_articulation_cfg = _build_official_leap_articulation_cfg(official_urdf_path)  # raw URDF fallback
            hand_asset_ref = str(official_urdf_path)  # UI / YAML 记录 raw URDF 路径
            hand_source = "official_leap_urdf"  # 明确本轮是 raw URDF fallback probe
        else:
            official_usd_path = _resolve_official_leap_usd_input(args_cli.official_leap_usd_path)  # preconverted official LEAP USD
            hand_articulation_cfg = _build_official_leap_usd_articulation_cfg(official_usd_path)  # 默认官方资产 probe
            hand_asset_ref = str(official_usd_path)  # UI / YAML 记录官方 USD 路径
            hand_source = "official_leap_usd"  # 明确本轮是官方 USD 消融 probe
        hand_frame_cfg = HandFrameCfg(anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E)  # official probe 导出时仅按 raw root frame 解释
    else:
        hand_bundle_path = _resolve_hand_bundle_input(args_cli.hand_bundle or preset_hand_bundle)  # 本次实际标定资产
        hand_spawn_cfg = _build_hand_spawn_cfg(hand_bundle_path)  # AnyMani bundle -> IsaacLab hand spawn cfg
        hand_articulation_cfg = _build_generated_hand_articulation_cfg(hand_spawn_cfg)  # generated 主线 robot cfg
        hand_frame_cfg = hand_spawn_cfg.frame  # generated `{a}->{h}` 语义 frame，用于导出 `object_pose_h`
        hand_asset_ref = hand_bundle_path  # UI / YAML 记录 generated bundle 路径
        hand_source = "generated_bundle"  # 默认主线资产来源

    initial_joint_pos = _joint_pos_from_preset(preset_payload)  # `joint_name -> rad`
    initial_object_pos_cfg, initial_object_rpy_xyz = _object_pose_from_preset(preset_payload)  # cfg/env pose seed

    sim_cfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=args_cli.device)  # 与 GM probe 物理 dt 对齐
    sim = SimulationContext(sim_cfg)  # IsaacLab 仿真上下文
    sim.set_camera_view([1.0, 1.0, 1.0], [0.0, 0.06, 0.55])  # 默认相机看向手掌与 cube 区域

    scene_cfg = _make_scene_cfg(
        hand_articulation_cfg,
        args_cli.object_source,
        initial_object_pos_cfg,
        initial_object_rpy_xyz,
        args_cli.device,
    )  # 单 env scene cfg；object_source 默认 local_cube，避免官方 hand 消融受远程 DexCube 影响
    scene = InteractiveScene(scene_cfg)  # 生成 hand/object/USD prims
    if hand_source == "generated_bundle":
        _apply_generated_collision_filter(scene, args_cli.generated_collision_filter)  # 必须在 `sim.reset()` 前写入 PhysX schema
    sim.reset()  # 激活 PhysX handles；此后可写 articulation / rigid object tensors
    scene.update(sim.get_physics_dt())  # 刷新 root/joint buffers，供 UI 初始化读取

    robot: Articulation = scene["robot"]  # hand articulation handle
    obj: RigidObject = scene["object"]  # DexCube rigid object handle
    panel = SingleAssetGraspCalibrationPanel(
        scene=scene,
        robot=robot,
        obj=obj,
        hand_frame_cfg=hand_frame_cfg,
        hand_asset_ref=hand_asset_ref,
        hand_source=hand_source,
        object_source=args_cli.object_source,
        initial_joint_pos=initial_joint_pos,
        initial_object_pos_cfg=initial_object_pos_cfg,
        initial_object_rpy_xyz=initial_object_rpy_xyz,
        output_name=args_cli.output_name,
        loaded_preset_path=preset_path,
    )

    print("\n" + "=" * 88)
    print("AnyMani single-asset grasp calibrator is running.")
    print("Workflow: adjust sliders -> optionally unlock/gizmo/read object -> Export Preset.")
    print(f"Preset directory: {PRESET_DIR}")
    print("=" * 88 + "\n")

    run_simulator(sim, scene, panel, smoke_seconds=args_cli.smoke_seconds)


if __name__ == "__main__":
    main()
    simulation_app.close()
