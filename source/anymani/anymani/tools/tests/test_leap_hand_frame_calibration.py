r"""Pure math tests for the LEAP hand semantic-frame calibration helper.

这些测试不启动 Isaac Sim，也不 import IsaacLab。它们只锁住一个科研语义：
用户手工读到的 $T_{ah}$ 与 GM 配置常用的 $T_{ha}$ 必须互为逆，且 LEAP 旧
`InitialStateCfg.pos/rot` 仍只作为 $T_{ea}^{init}$ 参与 $T_{eh}^{init}$ 的人眼核对。

测试放在 `tools/tests/` 而不是 `tasks/gm/tests/`，因为被测对象是纯数学标定工具，
不是 GM MDP term；后续若 GM env 真正消费 `semantic_R_ha/semantic_p_ha`，再在
`tasks/gm/tests/` 为 observation / command contract 单独补测试。
"""

from __future__ import annotations

import math

from anymani.tools.leap_hand_frame_calibration import (
    LEAP_DEFAULT_ROOT_QUAT_WXYZ,
    calibrate_frame,
    matmul3,
    matrix_from_axis_columns,
    matrix_to_tuple9,
    matvec3,
    quat_wxyz_to_matrix,
)


def _assert_tuple_close(actual: tuple[float, ...], expected: tuple[float, ...], *, atol: float = 1.0e-8) -> None:
    r"""逐元素比较两个浮点 tuple。

    Args:
        actual (tuple[float, ...]): 实际输出。
        expected (tuple[float, ...]): 期望输出。
        atol (float): 绝对误差阈值。
    """

    assert len(actual) == len(expected)  # shape contract：两个 tuple 长度必须一致
    for got, want in zip(actual, expected, strict=True):
        assert abs(got - want) <= atol  # 浮点误差范围内相等


def test_t_ah_inverse_produces_semantic_t_ha() -> None:
    r"""$T_{ha}=T_{ah}^{-1}$ 的旋转和平移方向不能写反。"""

    R_ah = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # $R_z(90^\circ)$
    p_ah = (1.0, 2.0, 3.0)  # `{h}` origin in `{a}`，单位 m

    calibration = calibrate_frame(R_ah, p_ah)  # 纯数学反算，不触碰 Isaac runtime

    _assert_tuple_close(matrix_to_tuple9(calibration.R_ha), (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    _assert_tuple_close(calibration.p_ha, (-2.0, 1.0, -3.0))  # $-R_{ha}p_{ah}$


def test_leap_default_root_pose_is_only_used_for_visual_t_eh_check() -> None:
    r"""旧 LEAP root quaternion 应只参与 $T_{eh}^{init}=T_{ea}^{init}T_{ah}$ 的核对。"""

    R_ah = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # 暂用 identity `$T_{ah}` dry-run
    p_ah = (0.0, 0.0, 0.0)  # `{h}` origin 与 `{a}` origin 重合，仅作公式 sanity

    calibration = calibrate_frame(R_ah, p_ah)  # 默认 root pose 为 LEAP 当前 cfg 数值锚点
    R_ea = quat_wxyz_to_matrix(LEAP_DEFAULT_ROOT_QUAT_WXYZ)  # $R_{ea}^{init}$

    _assert_tuple_close(matrix_to_tuple9(calibration.R_eh_init), matrix_to_tuple9(R_ea))
    _assert_tuple_close(calibration.p_eh_init, (0.0, 0.0, 0.5))  # $p_{eh}=p_{ea}$，因为 $p_{ah}=0$


def test_rotation_composition_matches_column_vector_convention() -> None:
    r"""列向量约定下 $R_{eh}=R_{ea}R_{ah}$，避免 row/column 语义被误改。"""

    angle = math.pi / 2.0  # 90 degree yaw，只用于构造易检验的 $R_{ah}$
    R_ah = ((math.cos(angle), -math.sin(angle), 0.0), (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0))
    p_ah = (0.1, 0.0, 0.0)  # `{h}` origin 在 `{a}` x 方向偏移 10cm

    calibration = calibrate_frame(R_ah, p_ah)  # 同时给出 $T_{eh}^{init}$
    R_ea = quat_wxyz_to_matrix(LEAP_DEFAULT_ROOT_QUAT_WXYZ)  # LEAP 当前 $R_{ea}^{init}$
    expected_R_eh = matmul3(R_ea, R_ah)  # $R_{eh}=R_{ea}R_{ah}$
    expected_p_eh = tuple(a + b for a, b in zip((0.0, 0.0, 0.5), matvec3(R_ea, p_ah), strict=True))

    _assert_tuple_close(matrix_to_tuple9(calibration.R_eh_init), matrix_to_tuple9(expected_R_eh))
    _assert_tuple_close(calibration.p_eh_init, expected_p_eh)


def test_matrix_from_axis_columns_matches_visual_axis_input() -> None:
    r"""用户按列向量填写 $x_h^a,y_h^a,z_h^a$ 时，脚本应拼成正确 row-major $R_{ah}$。"""

    R_ah = matrix_from_axis_columns(
        (0.0, 0.0, 1.0),  # $x_h^a=+z_a$
        (1.0, 0.0, 0.0),  # $y_h^a=+x_a$
        (0.0, 1.0, 0.0),  # $z_h^a=+y_a$
    )

    _assert_tuple_close(matrix_to_tuple9(R_ah), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
