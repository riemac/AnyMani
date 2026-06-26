#!/usr/bin/env python3
r"""官方 LEAP `{h}` 语义 frame 的纯数学标定辅助工具。

本脚本不启动 Isaac Sim / Isaac Lab，也不读取或修改 URDF / USD。它只把用户从
VSCode URDF viewer、Isaac Sim viewer 或人工几何测量中得到的 `{h}` 标定量
转换成 `gm` MDP 后续需要的配置量。

核心区分：

- `{a}` 是 LEAP raw asset/root frame，来自资产文件自身；不要为了语义对齐去改 URDF/USD；
- `InitialStateCfg.pos/rot` 给出 $T_{ea}^{init}$，负责把 `{a}` 摆到训练所需姿态；
- `{h}` 是固定附着在 `{a}` 上的 hand semantic frame，只负责给 command / observation /
  contact 等 MDP 量提供一致手部语义。

用户手工标定时最自然读到的是

$$
T_{ah}=\begin{bmatrix}R_{ah}&p_{ah}\\0&1\end{bmatrix},
$$

其中 $R_{ah}$ 表示 `{h}` 的三个坐标轴在 `{a}` 中的方向，$p_{ah}$ 表示 `{h}`
原点在 `{a}` 下的位置。现有 `gm` 配置更常消费逆变换

$$
T_{ha}=T_{ah}^{-1},\qquad R_{ha}=R_{ah}^{\top},\qquad p_{ha}=-R_{ha}p_{ah}.
$$

典型用法：直接编辑本文件顶部 `USER_*` 配置区，然后用 VSCode / PyCharm / Python
运行按钮执行本文件。不要把本工具当成需要记长命令行参数的 CLI；它的核心用途是
让用户逐项手工核对标定数值。

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
python source/anymani/anymani/tools/leap_hand_frame_calibration.py
```

输出中的 `semantic_R_ha` / `semantic_p_ha` 可在后续 LEAP env cfg 中作为 frame
标定常量使用；输出中的 `T_eh_init` 只用于人眼核对“当前 root pose 摆好手以后，
hand semantic frame 在 env 中落在哪里”。配置区刻意让用户按“列向量直觉”填写：
分别写 $x_h^a$、$y_h^a$、$z_h^a$，脚本内部再拼成 row-major 矩阵 $R_{ah}$。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]


LEAP_DEFAULT_ROOT_POS_E: Vector3 = (0.0, 0.0, 0.5)
r"""官方 LEAP GM probe 当前使用的 raw asset root 位置 $p_{ea}^{init}$，单位 m。"""

LEAP_DEFAULT_ROOT_QUAT_WXYZ: QuaternionWxyz = (0.5, 0.5, -0.5, 0.5)
r"""官方 LEAP GM probe 当前使用的 raw asset root 姿态 $R_{ea}^{init}$，IsaacLab `(w,x,y,z)`。"""

IDENTITY_R: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
r"""默认 $R_{ah}=I$，仅作为脚本 dry-run 占位，不代表真实 LEAP `{h}` 已标定。"""


# ======================================================================================
# USER CONFIG: 手工标定区
# ======================================================================================
# 你从 VSCode URDF viewer / Isaac Sim viewer 读出的 `{h}` 标定量写在这里，然后直接运行本文件。
# 约定：$T_{ah}$ 表示 `{h}` frame 在 raw asset/root frame `{a}` 下的位姿。
USER_P_AH: Vector3 = (0.011, 0.0098, -0.0020)
r"""用户手工标定的 $p_{ah}$：`{h}` 原点在 `{a}` 下的位置，单位 m。"""

USER_X_H_IN_A: Vector3 = (0.0, 1.0, 0.0)
r"""用户手工标定的 $x_h^a$：`{h}` 的 x 轴方向在 `{a}` 下的列向量。"""

USER_Y_H_IN_A: Vector3 = (0.0, 0.0, 1.0)
r"""用户手工标定的 $y_h^a$：`{h}` 的 y 轴方向在 `{a}` 下的列向量。"""

USER_Z_H_IN_A: Vector3 = (1.0, 0.0, 0.0)
r"""用户手工标定的 $z_h^a$：`{h}` 的 z 轴方向在 `{a}` 下的列向量。"""

USER_ROOT_POS_E: Vector3 = LEAP_DEFAULT_ROOT_POS_E
r"""LEAP raw asset root 的任务摆放位置 $p_{ea}^{init}$；通常保持与 `GmLeapEnvCfg` 一致。"""

USER_ROOT_QUAT_WXYZ: QuaternionWxyz = LEAP_DEFAULT_ROOT_QUAT_WXYZ
r"""LEAP raw asset root 的任务摆放姿态 $R_{ea}^{init}$；通常保持与 `GmLeapEnvCfg` 一致。"""


@dataclass(frozen=True)
class FrameCalibration:
    r"""单个手部语义 frame 标定结果。

    该结构同时保存用户输入的 $T_{ah}$ 与代码反算得到的 $T_{ha}$，避免后续
    阅读者在配置文件里只看到一个矩阵却不知道它到底是哪个方向。

    Attributes:
        R_ah (Matrix3): `{h}` 坐标轴在 `{a}` 下的方向矩阵，满足 $v^a=R_{ah}v^h$。
        p_ah (Vector3): `{h}` 原点在 `{a}` 下的位置，单位 m。
        R_ha (Matrix3): 逆向旋转矩阵，满足 $v^h=R_{ha}v^a$。
        p_ha (Vector3): `{a}` 原点在 `{h}` 下的位置，单位 m。
        R_eh_init (Matrix3): 在当前 LEAP root init pose 下，`{h}` 轴在 `{e}` 下的方向。
        p_eh_init (Vector3): 在当前 LEAP root init pose 下，`{h}` 原点在 `{e}` 下的位置。
    """

    R_ah: Matrix3
    p_ah: Vector3
    R_ha: Matrix3
    p_ha: Vector3
    R_eh_init: Matrix3
    p_eh_init: Vector3


def calibrate_frame(
    R_ah: Matrix3,
    p_ah: Vector3,
    *,
    root_pos_e: Vector3 = LEAP_DEFAULT_ROOT_POS_E,
    root_quat_wxyz: QuaternionWxyz = LEAP_DEFAULT_ROOT_QUAT_WXYZ,
) -> FrameCalibration:
    r"""由人工标定的 $T_{ah}$ 反算 $T_{ha}$ 并给出当前 root pose 下的 $T_{eh}^{init}$。

    核心公式：
    $$
    R_{ha}=R_{ah}^{\top},\qquad p_{ha}=-R_{ha}p_{ah},
    $$
    以及仅用于核对的
    $$
    T_{eh}^{init}=T_{ea}^{init}T_{ah}.
    $$

    Args:
        R_ah (Matrix3): `{h}` 轴在 `{a}` 下的方向矩阵，列向量约定 $v^a=R_{ah}v^h$。
        p_ah (Vector3): `{h}` 原点在 `{a}` 下的位置，单位 m。
        root_pos_e (Vector3): LEAP raw asset root 在 `{e}` 下的位置，单位 m。
        root_quat_wxyz (QuaternionWxyz): LEAP raw asset root 在 `{e}` 下的姿态。

    Returns:
        FrameCalibration: 同时包含 $T_{ah}$、$T_{ha}$ 和 $T_{eh}^{init}$ 的标定结果。
    """

    _validate_rotation_matrix(R_ah, label="R_ah")  # 标定输入必须是 $SO(3)$，否则后续 axis 语义会漂移
    R_ea = quat_wxyz_to_matrix(root_quat_wxyz)  # $R_{ea}^{init}$，由 IsaacLab root quaternion 转矩阵
    R_ha = transpose3(R_ah)  # $R_{ha}=R_{ah}^{\top}$，逆向旋转
    p_ha = scale3(matvec3(R_ha, p_ah), -1.0)  # $p_{ha}=-R_{ha}p_{ah}$，raw origin in `{h}`
    R_eh_init = matmul3(R_ea, R_ah)  # $R_{eh}=R_{ea}R_{ah}$，仅用于人眼核对 `{h}` 朝向
    p_eh_init = add3(root_pos_e, matvec3(R_ea, p_ah))  # $p_{eh}=p_{ea}+R_{ea}p_{ah}$，单位 m
    return FrameCalibration(R_ah=R_ah, p_ah=p_ah, R_ha=R_ha, p_ha=p_ha, R_eh_init=R_eh_init, p_eh_init=p_eh_init)


def matrix_from_axis_columns(x_axis: Vector3, y_axis: Vector3, z_axis: Vector3) -> Matrix3:
    r"""由三根列向量拼出 row-major $R_{ah}$。

    用户手工标定时通常是“我看到 `{h}` 的 x/y/z 轴分别指向 `{a}` 的哪个方向”。
    这三个方向本质上是旋转矩阵的列：
    $$
    R_{ah}=\begin{bmatrix}x_h^a & y_h^a & z_h^a\end{bmatrix}.
    $$
    Python tuple 外层仍按行存储，因此这里显式把列向量转成 row-major 矩阵，避免用户
    把视觉上的列向量误写成行向量。

    Args:
        x_axis (Vector3): $x_h^a$，`{h}` x 轴在 `{a}` 下的单位方向。
        y_axis (Vector3): $y_h^a$，`{h}` y 轴在 `{a}` 下的单位方向。
        z_axis (Vector3): $z_h^a$，`{h}` z 轴在 `{a}` 下的单位方向。

    Returns:
        Matrix3: row-major $R_{ah}$，其三列分别为输入的三根轴。
    """

    x_axis = _as_vec3(x_axis)  # $x_h^a$，列 0
    y_axis = _as_vec3(y_axis)  # $y_h^a$，列 1
    z_axis = _as_vec3(z_axis)  # $z_h^a$，列 2
    return (
        (x_axis[0], y_axis[0], z_axis[0]),  # 第 0 行：三根轴的 a-x 分量
        (x_axis[1], y_axis[1], z_axis[1]),  # 第 1 行：三根轴的 a-y 分量
        (x_axis[2], y_axis[2], z_axis[2]),  # 第 2 行：三根轴的 a-z 分量
    )


def quat_wxyz_to_matrix(quat: QuaternionWxyz) -> Matrix3:
    r"""把 IsaacLab `(w,x,y,z)` 四元数转换成旋转矩阵。

    Args:
        quat (QuaternionWxyz): 四元数 $(w,x,y,z)$，表示 $R_{ea}$。

    Returns:
        Matrix3: 旋转矩阵 $R_{ea}$，列向量约定。
    """

    w, x, y, z = (float(value) for value in quat)  # IsaacLab 边界使用 `(w,x,y,z)` 顺序
    norm = math.sqrt(w * w + x * x + y * y + z * z)  # 四元数范数，防止用户输入未归一化
    if norm <= 1.0e-12:
        raise ValueError("root quaternion must be non-zero")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)  # 单位四元数，保证得到 $SO(3)$
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def matrix_to_tuple9(matrix: Matrix3) -> tuple[float, ...]:
    r"""把 $3\times3$ 矩阵转成配置层 row-major 9 元组。

    Args:
        matrix (Matrix3): 旋转矩阵。

    Returns:
        tuple[float, ...]: row-major 展平结果，适合写入 `semantic_R_ha`。
    """

    return tuple(value for row in matrix for value in row)  # row-major，和 `HandFrameCfg.semantic_R_ha` 一致


def matmul3(lhs: Matrix3, rhs: Matrix3) -> Matrix3:
    r"""计算 $3\times3$ 矩阵乘法 $C=AB$。"""

    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def matvec3(matrix: Matrix3, vector: Vector3) -> Vector3:
    r"""计算 $y=Rv$。"""

    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def transpose3(matrix: Matrix3) -> Matrix3:
    r"""计算 $R^{\top}$。"""

    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))  # type: ignore[return-value]


def add3(lhs: Vector3, rhs: Vector3) -> Vector3:
    r"""计算三维向量和 $a+b$。"""

    return tuple(lhs[index] + rhs[index] for index in range(3))  # type: ignore[return-value]


def scale3(vector: Vector3, scale: float) -> Vector3:
    r"""计算三维向量缩放 $s v$。"""

    return tuple(scale * value for value in vector)  # type: ignore[return-value]


def det3(matrix: Matrix3) -> float:
    r"""计算 $3\times3$ 矩阵行列式，用于检查 $R\in SO(3)$。"""

    a, b, c = matrix[0]  # 第一行
    d, e, f = matrix[1]  # 第二行
    g, h, i = matrix[2]  # 第三行
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)  # $\det(R)$


def frobenius_norm(matrix: Matrix3) -> float:
    r"""计算矩阵 Frobenius 范数 $\|A\|_F$。"""

    return math.sqrt(sum(value * value for row in matrix for value in row))  # $\sqrt{\sum_{ij} A_{ij}^2}$


def _validate_rotation_matrix(matrix: Matrix3, *, label: str) -> None:
    r"""检查输入矩阵是否足够接近 $SO(3)$。

    Args:
        matrix (Matrix3): 待检查矩阵。
        label (str): 错误信息中的矩阵名。

    Raises:
        ValueError: 当行列式或正交性偏差超过容差时抛出。
    """

    RtR = matmul3(transpose3(matrix), matrix)  # $R^\top R$，应等于单位阵
    orth_err = frobenius_norm(_matrix_sub3(RtR, IDENTITY_R))  # $\|R^\top R-I\|_F$
    determinant = det3(matrix)  # $\det(R)$，右手系旋转应接近 1
    if abs(determinant - 1.0) > 1.0e-4 or orth_err > 1.0e-4:
        raise ValueError(f"{label} must be in SO(3); det={determinant:.8f}, orth_err={orth_err:.8e}")


def _matrix_sub3(lhs: Matrix3, rhs: Matrix3) -> Matrix3:
    r"""计算 $A-B$，只用于旋转矩阵正交性检查。"""

    return tuple(
        tuple(lhs[row][col] - rhs[row][col] for col in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _as_matrix3(values: Iterable[float]) -> Matrix3:
    r"""把 9 个 row-major 数值解析成 $3\times3$ 矩阵。"""

    flat = tuple(float(value) for value in values)  # row-major `[r00,r01,...,r22]`
    if len(flat) != 9:
        raise ValueError(f"Expected 9 matrix values, got {len(flat)}")
    return (flat[0:3], flat[3:6], flat[6:9])  # type: ignore[return-value]


def _as_vec3(values: Iterable[float]) -> Vector3:
    r"""把 3 个数值解析成三维向量。"""

    vector = tuple(float(value) for value in values)  # `(x,y,z)`，单位由调用语义决定
    if len(vector) != 3:
        raise ValueError(f"Expected 3 vector values, got {len(vector)}")
    return vector  # type: ignore[return-value]


def _as_quat(values: Iterable[float]) -> QuaternionWxyz:
    r"""把 4 个数值解析成 IsaacLab `(w,x,y,z)` 四元数。"""

    quat = tuple(float(value) for value in values)  # `(w,x,y,z)`，IsaacLab root pose 边界约定
    if len(quat) != 4:
        raise ValueError(f"Expected 4 quaternion values, got {len(quat)}")
    return quat  # type: ignore[return-value]


def _format_tuple(values: Iterable[float]) -> str:
    r"""格式化浮点 tuple，便于直接复制到 Python cfg。"""

    return "(" + ", ".join(f"{value:.9g}" for value in values) + ")"  # 9 位有效数字足够表达标定输入


def print_report(calibration: FrameCalibration, *, root_pos_e: Vector3, root_quat_wxyz: QuaternionWxyz) -> None:
    r"""打印科研可读的标定报告。

    Args:
        calibration (FrameCalibration): `calibrate_frame` 返回的标定结果。
        root_pos_e (Vector3): 当前 LEAP raw root 位置数值锚点。
        root_quat_wxyz (QuaternionWxyz): 当前 LEAP raw root 姿态数值锚点。
    """

    print("\n=== LEAP hand semantic frame calibration ===")
    print("Input T_ah: `{h}` expressed in raw asset/root frame `{a}`")
    print(f"p_ah = {_format_tuple(calibration.p_ah)}  # meters")
    print(f"R_ah = {_format_tuple(matrix_to_tuple9(calibration.R_ah))}  # row-major")
    print("\nInverse T_ha for GM semantic config")
    print(f"semantic_R_ha = {_format_tuple(matrix_to_tuple9(calibration.R_ha))}")
    print(f"semantic_p_ha = {_format_tuple(calibration.p_ha)}  # meters")
    print("\nLEAP raw asset root init pose kept unchanged")
    print(f"root_pos_e = {_format_tuple(root_pos_e)}")
    print(f"root_quat_wxyz = {_format_tuple(root_quat_wxyz)}")
    print("\nDerived hand semantic frame in env under this root pose, for visual sanity check only")
    print(f"p_eh_init = {_format_tuple(calibration.p_eh_init)}  # meters")
    print(f"R_eh_init = {_format_tuple(matrix_to_tuple9(calibration.R_eh_init))}  # row-major")
    print("\nAxis sanity: columns of R_ah are x_h/y_h/z_h expressed in `{a}`; columns of R_eh_init are them in `{e}`.")


def main() -> None:
    r"""直接读取文件顶部 `USER_*` 配置并打印 $T_{ha}$ 报告。"""

    R_ah = matrix_from_axis_columns(USER_X_H_IN_A, USER_Y_H_IN_A, USER_Z_H_IN_A)  # 用户按列向量直觉填写的 $R_{ah}$
    p_ah = _as_vec3(USER_P_AH)  # 用户手工标定的 $p_{ah}$，单位 m
    root_pos_e = _as_vec3(USER_ROOT_POS_E)  # LEAP `InitialStateCfg.pos`，单位 m
    root_quat_wxyz = _as_quat(USER_ROOT_QUAT_WXYZ)  # LEAP `InitialStateCfg.rot`，wxyz
    calibration = calibrate_frame(R_ah, p_ah, root_pos_e=root_pos_e, root_quat_wxyz=root_quat_wxyz)
    print_report(calibration, root_pos_e=root_pos_e, root_quat_wxyz=root_quat_wxyz)


if __name__ == "__main__":
    main()


__all__ = [
    "FrameCalibration",
    "IDENTITY_R",
    "LEAP_DEFAULT_ROOT_POS_E",
    "LEAP_DEFAULT_ROOT_QUAT_WXYZ",
    "USER_P_AH",
    "USER_X_H_IN_A",
    "USER_Y_H_IN_A",
    "USER_Z_H_IN_A",
    "USER_ROOT_POS_E",
    "USER_ROOT_QUAT_WXYZ",
    "calibrate_frame",
    "det3",
    "matmul3",
    "matrix_from_axis_columns",
    "matrix_to_tuple9",
    "matvec3",
    "quat_wxyz_to_matrix",
    "transpose3",
]
