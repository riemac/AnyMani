r"""Static geometry evidence 的 proper-SE(3) 坐标重写。

对 hand-frame point/vector 和 spatial screw 使用：

$$
p'=Rp+t,\qquad n'=Rn,\qquad
\omega'=R\omega,\qquad v'=Rv-\omega'\times t.
$$

该操作只改变同一物理手的坐标描述。q、q_home、entity/JOINT routing、graph 和 masks 保持不变。
Reflection 的 det=-1，不属于 proper SE(3)，由入口直接拒绝。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .evidence import StaticGeometryEvidence


def _validate_proper_rotation(rotation: torch.Tensor) -> None:
    r"""验证 `[3,3]` 或 `[B,3,3]` rotation 属于 $SO(3)$。"""

    if rotation.ndim not in {2, 3} or rotation.shape[-2:] != (3, 3) or not rotation.is_floating_point():
        raise ValueError("rotation must have floating shape [3,3] or [B,3,3]")
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    gram = rotation.transpose(-2, -1) @ rotation
    if not torch.allclose(gram, identity.expand_as(gram), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("rotation must be orthogonal")
    determinant = torch.linalg.det(rotation)
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("rotation must be proper with determinant +1")


def _rotate(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    r"""把尾轴三维 vector 按 unbatched/batched rotation 共同旋转。"""

    if rotation.ndim == 2:
        return value @ rotation.T
    if value.shape[0] != rotation.shape[0]:
        raise ValueError("batched rotation must share evidence batch axis")
    return torch.einsum("b...j,bij->b...i", value, rotation)


def _translate(points: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    r"""沿尾轴广播 unbatched/batched translation。"""

    if translation.ndim == 1:
        return points + translation
    if points.shape[0] != translation.shape[0]:
        raise ValueError("batched translation must share evidence batch axis")
    shape = (translation.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    return points + translation.view(shape)


def rewrite_static_geometry_evidence_se3(
    evidence: StaticGeometryEvidence,
    *,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> StaticGeometryEvidence:
    r"""对 points、palm normal 与 spatial screws 执行同一个 proper-SE(3) coordinate rewrite。

    Args:
        evidence (StaticGeometryEvidence): `[K,3]` 或 `[B,K,3]` static geometry package。
        rotation (torch.Tensor): `[3,3]` 或 `[B,3,3]` proper rotation。
        translation (torch.Tensor): `[3]` 或 `[B,3]` coordinate translation，单位 m。

    Returns:
        StaticGeometryEvidence: 同一 physical hand 的等价坐标描述。
    """

    _validate_proper_rotation(rotation)
    if translation.ndim not in {1, 2} or translation.shape[-1] != 3:
        raise ValueError("translation must have shape [3] or [B,3]")
    if rotation.device != evidence.anchors.device or translation.device != evidence.anchors.device:
        raise ValueError("SE3 rewrite tensors must share evidence device")
    if rotation.dtype != evidence.anchors.dtype or translation.dtype != evidence.anchors.dtype:
        raise ValueError("SE3 rewrite tensors must share evidence dtype")
    if (rotation.ndim == 3 or translation.ndim == 2) and evidence.anchors.ndim != 3:
        raise ValueError("per-row SE3 rewrite requires batched StaticGeometryEvidence")
    if rotation.ndim != translation.ndim + 1:
        raise ValueError("rotation and translation must both be batched or both unbatched")

    anchors = _translate(_rotate(evidence.anchors, rotation), translation)  # $a'=Ra+t$
    home = _translate(_rotate(evidence.home_surface_points, rotation), translation)  # $p'=Rp+t$
    normal = _rotate(evidence.palm_normal, rotation)  # $n'=Rn$
    omega = _rotate(evidence.space_screws[..., :3], rotation)  # $\omega'=R\omega$
    linear_rotated = _rotate(evidence.space_screws[..., 3:], rotation)  # $Rv$
    translation_for_screw = translation
    while translation_for_screw.ndim < omega.ndim:
        translation_for_screw = translation_for_screw.unsqueeze(-2)
    linear = linear_rotated - torch.cross(
        omega,
        translation_for_screw.expand_as(omega),
        dim=-1,
    )  # $v'=Rv-\omega'\times t$
    screws = torch.cat((omega, linear), dim=-1)
    return replace(
        evidence,
        anchors=anchors,
        home_surface_points=home,
        palm_normal=normal,
        space_screws=screws,
    )


__all__ = ["rewrite_static_geometry_evidence_se3"]
