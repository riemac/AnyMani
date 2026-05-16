"""`HandCfg` 的树状可视化渲染工具。

这个文件专门承载“把 canonical hand schema 渲染成人能快速扫读的树结构”这一类
纯展示逻辑。它被从 `hand_generator.py` 中拆出来，主要是因为这些函数虽然很有用，
但它们并不参与：

- pre-made 选择
- connectivity lower
- mutate / validate / export 调度

把它们留在主 façade 文件里，会让科研侧在阅读 `HandGenerator` 的生成主线时，
频繁被“展示层细节”打断。

# NOTE:
这里的渲染结果并不是“花哨调试输出”，而是科研排障的重要界面：

- `render_hand_tree_txt()` 适合终端快速查看。

它始终基于同一个 `HandCfg` 真源，不引入任何新语义。
"""

from __future__ import annotations

import math
from typing import Any

from ...asset_base import HandCfg


def _axis_label(axis: tuple[float, float, float]) -> str:
    r"""把旋转轴向量压缩成 `+X` / `-Y` / `+Z` 这样的简短标签。

    这里刻意采用“取绝对值最大的主轴分量”的简化展示，而不是把完整三维向量原样打印，
    是因为当前 finger / thumb preset 的轴语义本来就主要沿坐标轴表达。
    对科研巡检来说，`+X / +Y / +Z` 比 `(1.0, 0.0, 0.0)` 更快读。
    """

    labels = ("X", "Y", "Z")  # 坐标轴名字母表
    idx = max(range(3), key=lambda i: abs(axis[i]))  # 取绝对值最大的主轴方向
    sign = "-" if axis[idx] < 0 else "+"  # 保留正负号，便于快速区分镜像方向
    return f"{sign}{labels[idx]}"


def _link_length(origin: Any) -> float:
    r"""从 `PoseCfg.pos` 计算子 link 相对父 link 的平移距离（米）。

    这里展示的是：
    $$
    \|\mathbf{t}\|_2 = \sqrt{x^2 + y^2 + z^2}
    $$

    它不是“mesh 长度”，而是 joint origin 相对父 link 的几何推进距离。
    在当前树状展示里，这个量最适合快速暴露：

    - link 间是否意外断开
    - 某条 finger 链是否突然被拉长/塌缩
    """

    if origin is None:
        return 0.0
    x, y, z = origin.pos  # joint frame 相对 parent link frame 的平移分量
    return math.sqrt(x * x + y * y + z * z)  # $\|\mathbf{t}\|_2$


def _fmt_vec(v: tuple[float, float, float]) -> str:
    r"""把三维向量格式化成固定宽度的 `(+x, +y, +z)` 字符串。"""

    x, y, z = v
    return f"({x:+.3f}, {y:+.3f}, {z:+.3f})"


def render_hand_tree_txt(hand_cfg: HandCfg) -> str:
    r"""把 `HandCfg` 渲染为富信息 ASCII 树字符串。

    每条 joint 行包含：

    - joint 名
    - child link 名
    - 关节类型
    - 旋转轴
    - parent→joint 的推进距离
    - 关节限位
    - 指尖标记

    这样科研侧在终端里就能快速对照：

    - 链是否接通
    - 哪个 joint 是 fixed / revolute
    - tip 是否落在预期位置
    """

    lines: list[str] = []

    # ── 顶层 palm 行 ──────────────────────────────────────────────────────
    dof = hand_cfg.dof_count  # 整手当前总自由度数
    lines.append(
        f"{hand_cfg.palm.name}"
        f"  [family={hand_cfg.family} · {hand_cfg.handedness} · dof={dof}]"
    )

    n_fingers = len(hand_cfg.fingers)  # finger 总数，决定树状分支符号
    for f_idx, finger in enumerate(hand_cfg.fingers):
        is_last_finger = f_idx == n_fingers - 1  # 最后一根 finger 用 `└──`
        f_branch = "└── " if is_last_finger else "├── "
        f_cont = "    " if is_last_finger else "│   "

        # ── finger 挂载行 ─────────────────────────────────────────────────
        mount_pos = _fmt_vec(finger.mount.pos) if finger.mount else "(+0.000, +0.000, +0.000)"
        mount_rpy = _fmt_vec(finger.mount.rpy) if finger.mount else "(+0.000, +0.000, +0.000)"
        lines.append(f"{f_branch}[{finger.name}]  mount={mount_pos} m  rpy={mount_rpy} rad")

        n_joints = len(finger.joints)  # 当前 finger 的 joint 数
        for j_idx, joint in enumerate(finger.joints):
            is_last = j_idx == n_joints - 1  # finger 内最后一个 joint 用 `└──`
            j_prefix = f"{f_cont}{'└── ' if is_last else '├── '}"

            # 旋转轴与 parent→joint 推进距离。
            axis_str = _axis_label(joint.axis) if joint.joint_type != "fixed" else "fixed"
            length = _link_length(joint.origin)

            # revolute 才展示限位；fixed joint 没有转角上下界。
            limit_str = ""
            if joint.limit is not None and joint.joint_type == "revolute":
                lo = joint.limit.lower
                hi = joint.limit.upper
                limit_str = f"  [{lo:+.2f}, {hi:+.2f}] rad"

            tip_str = "  ★ TIP" if joint.is_tip else ""  # tip 在科研巡检里应一眼可见

            lines.append(
                f"{j_prefix}{joint.name}  →  {joint.child}"
                f"  {joint.joint_type}  axis={axis_str}  len={length:.4f} m"
                f"{limit_str}{tip_str}"
            )

    return "\n".join(lines)
__all__ = [
    "render_hand_tree_txt",
]
