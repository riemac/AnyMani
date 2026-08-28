r"""Canonical/native matched physical audit 的纯张量证据合同。

本模块不启动 Isaac Sim，也不拥有 scene、reset 或 scripted action。运行时 probe 只需按相同字段
记录 native、canonical 与 native-repeat 三条轨迹；本模块随后计算任务可解释的误差量：active joint
位置/速度、真实 fingertip 位姿、物体位姿、接触发生时刻、累计接触冲量与逐项 reward。

对指标族 $m$，canonical/native 误差记为 $E_m$，native 两次独立重复的数值波动记为 $S_m$。
第一轮审计只发布 $(E_m,S_m)$ 与比值，不在缺少任务容差 $T_m$ 时自行宣告物理等价：

$$
E_m \leq \max(T_m, kS_m).
$$

最终 $T_m$ 来自任务成功判据、接触阈值和可感知几何尺度；$k$ 是用户确认生产路线时使用的
保守倍数。把分析保持为纯张量函数，可使任意阈值在原始 artifact 上事后重算，而无需重跑 PhysX。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import torch

PHYSICAL_AUDIT_SCHEMA_VERSION: Final = "1.1.0"
"""Matched trace/summary schema；改变字段、单位或聚合公式时必须升级。"""

_REQUIRED_TRACE_FIELDS: Final = (
    "joint_pos_rad",
    "joint_vel_rad_s",
    "joint_target_rad",
    "tip_pos_m",
    "tip_quat_wxyz",
    "object_pos_m",
    "object_quat_wxyz",
    "object_lin_vel_m_s",
    "object_ang_vel_rad_s",
    "contact_force_N",
    "reward_terms",
)
"""每条 trace 的固定字段；首维均为同一 physics/policy 采样时间轴 $T$。"""


def candidate_indices_in_reference_semantic_order(
    reference_labels: tuple[str, ...],
    candidate_labels: tuple[str, ...],
) -> tuple[int, ...]:
    r"""返回 candidate axis 对齐到 reference semantic labels 的 gather indices。

    PhysX/importer body order、source sidecar finger order与 canonical schema order可以不同；matched audit
    比较的是同一语义实体，而不是偶然 row index。若 reference labels 为
    ``(index,middle,ring,thumb)``、candidate labels 为 ``(thumb,index,middle,ring)``，返回
    ``(1,2,3,0)``，使 ``candidate[...,indices,:]`` 与 reference 同轴。

    Args:
        reference_labels (tuple[str, ...]): 参考实体轴的唯一语义标签。
        candidate_labels (tuple[str, ...]): 待对齐实体轴的唯一语义标签。

    Returns:
        tuple[int, ...]: candidate 中按 reference label 顺序排列的整数 indices。

    Raises:
        ValueError: 任一轴有重复标签，或两侧语义标签集合不相等。
    """

    if len(set(reference_labels)) != len(reference_labels) or len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("matched semantic axes must contain unique labels")
    if set(reference_labels) != set(candidate_labels):
        raise ValueError(
            f"matched semantic axes differ: reference={reference_labels}, candidate={candidate_labels}"
        )
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}  # label→candidate row
    return tuple(candidate_index[label] for label in reference_labels)  # candidate gather order matching reference


def quaternion_geodesic_angle_wxyz(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    r"""计算单位四元数表示的 $SO(3)$ 测地角，输出单位为 rad。

    Isaac Lab runtime quaternion 顺序为 $(w,x,y,z)$。由于 $oldsymbol q$ 与
    $-\boldsymbol q$ 表示同一旋转，最短测地角使用绝对内积：

    $$
    d_R(q_1,q_2)=2\arccos\!\left(\left|\left\langle
    \frac{q_1}{\|q_1\|_2},\frac{q_2}{\|q_2\|_2}\right\rangle\right|\right).
    $$

    Args:
        reference (torch.Tensor): 参考 quaternion，形状 ``[...,4]``，顺序 $(w,x,y,z)$。
        candidate (torch.Tensor): 待比较 quaternion，与 ``reference`` 同形。

    Returns:
        torch.Tensor: principal angle，形状 ``[...]``，范围 $[0,\pi]$ rad。

    Raises:
        ValueError: 两个张量 shape 不同、末维不是 4 或存在零范数 quaternion。
    """

    if reference.shape != candidate.shape or reference.ndim < 1 or reference.shape[-1] != 4:
        raise ValueError("quaternion traces must have the same [...,4] shape")
    reference_norm = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)  # $\|q_1\|_2$，形状 ``[...,1]``
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)  # $\|q_2\|_2$，形状 ``[...,1]``
    if torch.any(reference_norm <= 0.0) or torch.any(candidate_norm <= 0.0):
        raise ValueError("quaternion traces must not contain zero-norm values")

    # $q\sim-q$，因此绝对内积给出双覆盖上的最短旋转；clamp 只吸收浮点归一化误差。
    reference_unit = reference / reference_norm  # 单位 quaternion，形状 ``[...,4]``
    candidate_unit = candidate / candidate_norm  # 单位 quaternion，形状 ``[...,4]``
    cosine_half_angle = torch.sum(reference_unit * candidate_unit, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(cosine_half_angle)  # $2\arccos(|q_1^Tq_2|)$，单位 rad


def first_contact_step(force_magnitude_N: torch.Tensor, threshold_N: float) -> int | None:
    r"""返回任一目标 contact channel 首次超过阈值的离散采样步。

    Args:
        force_magnitude_N (torch.Tensor): 非负接触力模长，形状 ``[T,C]``，单位 N。
        threshold_N (float): 与任务 contact bit 一致的严格阈值，单位 N。

    Returns:
        int | None: 首次满足 $\max_c f_{t,c}>\tau$ 的 $t$；全程无接触时为 ``None``。

    Raises:
        ValueError: 输入不是二维、含负值或阈值不是正数。
    """

    if force_magnitude_N.ndim != 2:
        raise ValueError("contact force trace must have shape [T,C]")
    if torch.any(force_magnitude_N < 0.0):
        raise ValueError("contact force magnitudes must be non-negative")
    if not math.isfinite(threshold_N) or threshold_N <= 0.0:
        raise ValueError("contact threshold must be finite and positive")
    active_steps = torch.nonzero(torch.any(force_magnitude_N > threshold_N, dim=-1), as_tuple=False).flatten()
    return None if active_steps.numel() == 0 else int(active_steps[0].item())


def integrated_contact_impulse_Ns(force_magnitude_N: torch.Tensor, sample_dt_s: float) -> torch.Tensor:
    r"""按 channel 积分离散接触力，得到短窗累计冲量近似。

    对采样间隔 $\Delta t$，采用左矩形和：

    $$
    J_c=\sum_{t=0}^{T-1} f_{t,c}\Delta t.
    $$

    Args:
        force_magnitude_N (torch.Tensor): 接触力模长，形状 ``[T,C]``，单位 N。
        sample_dt_s (float): 相邻记录间隔，单位 s；physics-step 记录时应等于 ``sim.dt``。

    Returns:
        torch.Tensor: 每个 contact channel 的累计冲量，形状 ``[C]``，单位 N·s。

    Raises:
        ValueError: 输入 shape/符号错误或采样周期无效。
    """

    if force_magnitude_N.ndim != 2:
        raise ValueError("contact force trace must have shape [T,C]")
    if torch.any(force_magnitude_N < 0.0):
        raise ValueError("contact force magnitudes must be non-negative")
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    return force_magnitude_N.sum(dim=0) * sample_dt_s  # $J_c=\sum_t f_{t,c}\Delta t$，``[C]`` N·s


def validate_physical_audit_trace(trace: Mapping[str, torch.Tensor]) -> int:
    r"""验证一条 matched trace 的字段、首维和物理 shape 合同。

    Args:
        trace (Mapping[str, torch.Tensor]): 由 runtime probe 形成的 GPU/CPU tensor 轨迹。

    Returns:
        int: 公共时间长度 $T$。

    Raises:
        KeyError: 缺少固定字段。
        ValueError: 时间轴、末维或数值有效性不满足合同。
    """

    missing = tuple(field for field in _REQUIRED_TRACE_FIELDS if field not in trace)  # schema 缺失字段
    if missing:
        raise KeyError(f"physical audit trace lacks fields: {missing}")
    time_steps = int(trace[_REQUIRED_TRACE_FIELDS[0]].shape[0])  # 所有字段共用采样时间轴 $T$
    if time_steps <= 0:
        raise ValueError("physical audit trace must contain at least one sample")
    for field in _REQUIRED_TRACE_FIELDS:
        value = trace[field]
        if value.ndim < 2 or value.shape[0] != time_steps:
            raise ValueError(f"trace field {field!r} must share time axis T={time_steps}")
        if not torch.isfinite(value).all():
            raise ValueError(f"trace field {field!r} contains non-finite values")

    # q、qd、target 必须共享 active source-joint 轴；native/canonical 在 probe 边界已按 routing gather。
    joint_shape = trace["joint_pos_rad"].shape
    if trace["joint_vel_rad_s"].shape != joint_shape or trace["joint_target_rad"].shape != joint_shape:
        raise ValueError("joint position, velocity and target traces must have the same [T,J] shape")
    if trace["tip_pos_m"].ndim != 3 or trace["tip_pos_m"].shape[-1] != 3:
        raise ValueError("tip_pos_m must have shape [T,K,3]")
    if trace["tip_quat_wxyz"].shape != (*trace["tip_pos_m"].shape[:-1], 4):
        raise ValueError("tip_quat_wxyz must have shape [T,K,4] aligned with tip_pos_m")
    if trace["object_pos_m"].shape != (time_steps, 3):
        raise ValueError("object_pos_m must have shape [T,3]")
    if trace["object_quat_wxyz"].shape != (time_steps, 4):
        raise ValueError("object_quat_wxyz must have shape [T,4]")
    if trace["object_lin_vel_m_s"].shape != (time_steps, 3):
        raise ValueError("object_lin_vel_m_s must have shape [T,3]")
    if trace["object_ang_vel_rad_s"].shape != (time_steps, 3):
        raise ValueError("object_ang_vel_rad_s must have shape [T,3]")
    if torch.any(trace["contact_force_N"] < 0.0):
        raise ValueError("contact_force_N must contain non-negative magnitudes")
    return time_steps


def summarize_physical_trace_error(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    sample_dt_s: float,
    contact_threshold_N: float,
) -> dict[str, float | int | None]:
    r"""汇总两条已对齐轨迹的任务尺度误差，不自行作 pass/fail 判定。

    ``reference`` 可为 native-first，``candidate`` 可为 canonical 或 native-repeat。两条轨迹必须已经
    使用同一 active source-joint、真实 fingertip、contact channel 与 reward-term 顺序。向量位置误差
    使用欧氏范数；关节标量与 reward term 同时报告 absolute max 与 RMS；姿态使用
    :func:`quaternion_geodesic_angle_wxyz`。

    Args:
        reference (Mapping[str, torch.Tensor]): 参考轨迹，通常为 native replicate 0。
        candidate (Mapping[str, torch.Tensor]): 待比较轨迹，canonical 或 native replicate 1。
        sample_dt_s (float): trace 采样周期，单位 s。
        contact_threshold_N (float): 首次接触判定阈值，单位 N。

    Returns:
        dict[str, float | int | None]: JSON-safe 原始误差指标；字段名携带单位或聚合语义。

    Raises:
        ValueError: trace 内部或两条 trace 之间 shape 不一致。
    """

    reference_steps = validate_physical_audit_trace(reference)  # 先验证参考时间轴与单位 shape
    candidate_steps = validate_physical_audit_trace(candidate)  # 再验证候选，错误不被广播掩盖
    if reference_steps != candidate_steps:
        raise ValueError("matched traces must have the same number of samples")
    for field in _REQUIRED_TRACE_FIELDS:
        if reference[field].shape != candidate[field].shape:
            raise ValueError(f"matched trace field {field!r} has different shapes")

    # 关节轨迹按全部时间与 active joints 聚合；target 误差应接近数值零，直接核对控制输入是否真匹配。
    joint_pos_error = candidate["joint_pos_rad"] - reference["joint_pos_rad"]  # ``[T,J]``，rad
    joint_vel_error = candidate["joint_vel_rad_s"] - reference["joint_vel_rad_s"]  # ``[T,J]``，rad/s
    joint_target_error = candidate["joint_target_rad"] - reference["joint_target_rad"]  # ``[T,J]``，rad

    # Fingertip 与 object translation 使用每个 pose 的 $L_2$ 距离，再跨时间/实体取 max 或 RMS。
    tip_position_error = torch.linalg.vector_norm(candidate["tip_pos_m"] - reference["tip_pos_m"], dim=-1)
    tip_orientation_error = quaternion_geodesic_angle_wxyz(
        reference["tip_quat_wxyz"], candidate["tip_quat_wxyz"]
    )
    object_position_error = torch.linalg.vector_norm(candidate["object_pos_m"] - reference["object_pos_m"], dim=-1)
    object_orientation_error = quaternion_geodesic_angle_wxyz(
        reference["object_quat_wxyz"], candidate["object_quat_wxyz"]
    )
    object_linear_velocity_error = torch.linalg.vector_norm(
        candidate["object_lin_vel_m_s"] - reference["object_lin_vel_m_s"], dim=-1
    )
    object_angular_velocity_error = torch.linalg.vector_norm(
        candidate["object_ang_vel_rad_s"] - reference["object_ang_vel_rad_s"], dim=-1
    )

    # Contact onset 与 impulse 分别刻画离散事件时间和短窗动量交换，不能由 peak force 互相替代。
    reference_onset = first_contact_step(reference["contact_force_N"], contact_threshold_N)
    candidate_onset = first_contact_step(candidate["contact_force_N"], contact_threshold_N)
    onset_delta = None if reference_onset is None or candidate_onset is None else abs(candidate_onset - reference_onset)
    reference_impulse = integrated_contact_impulse_Ns(reference["contact_force_N"], sample_dt_s)
    candidate_impulse = integrated_contact_impulse_Ns(candidate["contact_force_N"], sample_dt_s)
    impulse_error = candidate_impulse - reference_impulse  # ``[C]``，N·s
    reward_error = candidate["reward_terms"] - reference["reward_terms"]  # ``[T,R]``，各 term 自身单位

    return {
        "sample_count": reference_steps,
        "joint_pos_abs_max_rad": float(joint_pos_error.abs().max().item()),
        "joint_pos_rms_rad": float(joint_pos_error.square().mean().sqrt().item()),
        "joint_vel_abs_max_rad_s": float(joint_vel_error.abs().max().item()),
        "joint_vel_rms_rad_s": float(joint_vel_error.square().mean().sqrt().item()),
        "joint_target_abs_max_rad": float(joint_target_error.abs().max().item()),
        "tip_position_max_m": float(tip_position_error.max().item()),
        "tip_position_rms_m": float(tip_position_error.square().mean().sqrt().item()),
        "tip_orientation_max_rad": float(tip_orientation_error.max().item()),
        "tip_orientation_rms_rad": float(tip_orientation_error.square().mean().sqrt().item()),
        "object_position_max_m": float(object_position_error.max().item()),
        "object_position_rms_m": float(object_position_error.square().mean().sqrt().item()),
        "object_orientation_max_rad": float(object_orientation_error.max().item()),
        "object_orientation_rms_rad": float(object_orientation_error.square().mean().sqrt().item()),
        "object_linear_velocity_max_m_s": float(object_linear_velocity_error.max().item()),
        "object_linear_velocity_rms_m_s": float(object_linear_velocity_error.square().mean().sqrt().item()),
        "object_angular_velocity_max_rad_s": float(object_angular_velocity_error.max().item()),
        "object_angular_velocity_rms_rad_s": float(object_angular_velocity_error.square().mean().sqrt().item()),
        "reference_contact_onset_step": reference_onset,
        "candidate_contact_onset_step": candidate_onset,
        "contact_onset_abs_delta_steps": onset_delta,
        "contact_impulse_linf_error_Ns": float(impulse_error.abs().max().item()),
        "contact_impulse_l2_error_Ns": float(torch.linalg.vector_norm(impulse_error).item()),
        "reward_term_abs_max": float(reward_error.abs().max().item()),
        "reward_term_rms": float(reward_error.square().mean().sqrt().item()),
    }


def compare_canonical_against_native_repeat(
    native_reference: Mapping[str, torch.Tensor],
    native_repeat: Mapping[str, torch.Tensor],
    canonical: Mapping[str, torch.Tensor],
    *,
    sample_dt_s: float,
    contact_threshold_N: float,
) -> dict[str, object]:
    r"""同时发布 canonical 误差 $E_m$ 与 native-repeat 波动 $S_m$。

    数值比值只对三份 summary 中共同存在的有限浮点误差字段计算。若 $S_m=0$，比值为 ``None``，
    而不是用任意 epsilon 制造巨大或有限的伪结论；此时仍保留原始 $(E_m,S_m)$ 供任务容差判断。

    Args:
        native_reference (Mapping[str, torch.Tensor]): native replicate 0。
        native_repeat (Mapping[str, torch.Tensor]): 相同输入下独立 native replicate 1。
        canonical (Mapping[str, torch.Tensor]): 相同输入下 canonical trace。
        sample_dt_s (float): 公共采样周期，单位 s。
        contact_threshold_N (float): 公共首次接触阈值，单位 N。

    Returns:
        dict[str, object]: schema、$E_m$、$S_m$ 与可定义的 $E_m/S_m$。
    """

    common = {"sample_dt_s": sample_dt_s, "contact_threshold_N": contact_threshold_N}
    native_variation = summarize_physical_trace_error(
        native_reference,
        native_repeat,
        sample_dt_s=sample_dt_s,
        contact_threshold_N=contact_threshold_N,
    )  # $S_m$，相同 native 表示的重复波动
    canonical_error = summarize_physical_trace_error(
        native_reference,
        canonical,
        sample_dt_s=sample_dt_s,
        contact_threshold_N=contact_threshold_N,
    )  # $E_m$，canonical 相对同一 native reference 的误差

    # 只形成有物理意义的同名 scalar 比值；step index、sample count 与 missing onset 不参与除法。
    error_to_native_ratio: dict[str, float | None] = {}
    for name, error_value in canonical_error.items():
        variation_value = native_variation.get(name)
        if not isinstance(error_value, float) or not isinstance(variation_value, float):
            continue
        if not math.isfinite(error_value) or not math.isfinite(variation_value) or variation_value == 0.0:
            error_to_native_ratio[name] = None  # $S_m=0$ 时必须回到任务绝对容差 $T_m$
        else:
            error_to_native_ratio[name] = error_value / variation_value  # $E_m/S_m$，无量纲
    return {
        "schema_version": PHYSICAL_AUDIT_SCHEMA_VERSION,
        **common,
        "canonical_error_E": canonical_error,
        "native_repeat_variation_S": native_variation,
        "error_to_native_variation_ratio": error_to_native_ratio,
    }


__all__ = [
    "PHYSICAL_AUDIT_SCHEMA_VERSION",
    "candidate_indices_in_reference_semantic_order",
    "compare_canonical_against_native_repeat",
    "first_contact_step",
    "integrated_contact_impulse_Ns",
    "quaternion_geodesic_angle_wxyz",
    "summarize_physical_trace_error",
    "validate_physical_audit_trace",
]
