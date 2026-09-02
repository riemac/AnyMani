r"""Structured heterogeneous actor/critic的模型侧输入与输出合同。

Task交付named tensor tree；本模块不读取asset row、不解析flat offsets，也不拥有MDP。Ghost输入可以包含任意
finite poison，网络必须依靠bool masks保证输出、概率和gradient不受影响。Geometry tokens按统一
PALM1+JOINT16+TIP4 owner轴对齐。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

OWNER_COUNT = 21
JOINT_COUNT = 16
TIP_COUNT = 4
HISTORY_LENGTH = 30
GEOMETRY_WIDTH = 128


def _bool_mask(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    r"""把task transport的bool或0/1数值mask规约为bool。"""

    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype == torch.bool:
        return value
    torch._assert_async(
        torch.all(torch.isfinite(value) & ((value == 0) | (value == 1))),
        f"{name} numeric transport must contain only finite 0/1 values",
    )
    return value.to(dtype=torch.bool)


def _finite_shape(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    r"""验证rank/shape并原样返回tensor。

    Runtime finite检查由env/provider canary完成；在每个B=4096 forward对完整$Z$做reduction会引入额外GPU
    kernel并污染性能边界。
    """

    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class StructuredActorObservation:
    r"""Deployable$O^a=(O^a_{palm},O^a_{jnt},O^a_{tip})$ tensor tree。"""

    jnt_current: torch.Tensor  # $[B,16,3]$
    jnt_history: torch.Tensor  # $[B,30,16,4]$
    jnt_limits: torch.Tensor  # $[B,16,2]$
    tip_contact: torch.Tensor  # $[B,4,1]$
    jnt_valid: torch.Tensor  # bool$[B,16]$
    tip_valid: torch.Tensor  # bool$[B,4]$
    owner_valid: torch.Tensor  # bool$[B,21]$

    def __post_init__(self) -> None:
        r"""验证所有structured axes与device一致性。"""

        batch = self.jnt_current.shape[0]
        _finite_shape(self.jnt_current, name="jnt_current", shape=(batch, JOINT_COUNT, 3))
        _finite_shape(
            self.jnt_history,
            name="jnt_history",
            shape=(batch, HISTORY_LENGTH, JOINT_COUNT, 4),
        )
        _finite_shape(self.jnt_limits, name="jnt_limits", shape=(batch, JOINT_COUNT, 2))
        _finite_shape(self.tip_contact, name="tip_contact", shape=(batch, TIP_COUNT, 1))
        object.__setattr__(
            self, "jnt_valid", _bool_mask(self.jnt_valid, name="jnt_valid", shape=(batch, JOINT_COUNT))
        )
        object.__setattr__(
            self, "tip_valid", _bool_mask(self.tip_valid, name="tip_valid", shape=(batch, TIP_COUNT))
        )
        object.__setattr__(
            self, "owner_valid", _bool_mask(self.owner_valid, name="owner_valid", shape=(batch, OWNER_COUNT))
        )
        tensors = (
            self.jnt_current,
            self.jnt_history,
            self.jnt_limits,
            self.tip_contact,
            self.jnt_valid,
            self.tip_valid,
            self.owner_valid,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("structured actor tensors must share one device")
        expected_owner = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=self.owner_valid.device),
                self.jnt_valid,
                self.tip_valid,
            ),
            dim=-1,
        )
        torch._assert_async(
            torch.all(self.owner_valid == expected_owner),
            "owner_valid must equal [PALM,jnt_valid,tip_valid]",
        )

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> StructuredActorObservation:
        r"""从ManagerBased non-concatenated policy group恢复模型输入。"""

        required = {
            "jnt_current",
            "jnt_history",
            "jnt_limits",
            "tip_contact",
            "jnt_valid",
            "tip_valid",
            "owner_valid",
        }
        missing = required - set(observation)
        if missing:
            raise KeyError(f"structured actor observation misses terms {sorted(missing)}")
        return cls(**{name: observation[name] for name in required})


@dataclass(frozen=True)
class StructuredCriticObservation:
    r"""Privileged$O^c$ tensor tree；不继承actor object以避免职责混淆。"""

    jnt_state: torch.Tensor  # $[B,16,4]$
    owner_contact: torch.Tensor  # $[B,21,2]$，force N + bit
    obj: torch.Tensor  # $[B,1,15]$
    task: torch.Tensor  # $[B,1,8]$
    jnt_valid: torch.Tensor  # bool$[B,16]$
    tip_valid: torch.Tensor  # bool$[B,4]$
    owner_valid: torch.Tensor  # bool$[B,21]$

    def __post_init__(self) -> None:
        r"""验证privileged roles、masks与device。"""

        batch = self.jnt_state.shape[0]
        _finite_shape(self.jnt_state, name="jnt_state", shape=(batch, JOINT_COUNT, 4))
        _finite_shape(self.owner_contact, name="owner_contact", shape=(batch, OWNER_COUNT, 2))
        _finite_shape(self.obj, name="obj", shape=(batch, 1, 15))
        _finite_shape(self.task, name="task", shape=(batch, 1, 8))
        object.__setattr__(
            self, "jnt_valid", _bool_mask(self.jnt_valid, name="jnt_valid", shape=(batch, JOINT_COUNT))
        )
        object.__setattr__(
            self, "tip_valid", _bool_mask(self.tip_valid, name="tip_valid", shape=(batch, TIP_COUNT))
        )
        object.__setattr__(
            self, "owner_valid", _bool_mask(self.owner_valid, name="owner_valid", shape=(batch, OWNER_COUNT))
        )
        if len(
            {
                tensor.device
                for tensor in (
                    self.jnt_state,
                    self.owner_contact,
                    self.obj,
                    self.task,
                    self.jnt_valid,
                    self.tip_valid,
                    self.owner_valid,
                )
            }
        ) != 1:
            raise ValueError("structured critic tensors must share one device")
        expected_owner = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=self.owner_valid.device),
                self.jnt_valid,
                self.tip_valid,
            ),
            dim=-1,
        )
        torch._assert_async(
            torch.all(self.owner_valid == expected_owner),
            "critic owner_valid disagrees with joint/TIP masks",
        )

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> StructuredCriticObservation:
        r"""从ManagerBased non-concatenated critic group恢复模型输入。"""

        required = {"jnt_state", "owner_contact", "obj", "task", "jnt_valid", "tip_valid", "owner_valid"}
        missing = required - set(observation)
        if missing:
            raise KeyError(f"structured critic observation misses terms {sorted(missing)}")
        return cls(**{name: observation[name] for name in required})


@dataclass(frozen=True)
class GeometryTokenBatch:
    r"""冻结N040 unified owner tokens$Z^e$与mask。"""

    tokens: torch.Tensor  # $[B,21,128]$
    owner_valid: torch.Tensor  # bool$[B,21]$

    def __post_init__(self) -> None:
        r"""验证geometry shape/mask，不要求ghost token值为零。"""

        batch = self.tokens.shape[0]
        _finite_shape(self.tokens, name="geometry tokens", shape=(batch, OWNER_COUNT, GEOMETRY_WIDTH))
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="geometry owner_valid", shape=(batch, OWNER_COUNT)),
        )
        if self.tokens.device != self.owner_valid.device:
            raise ValueError("geometry tokens and mask must share device")


@dataclass(frozen=True)
class StructuredActorOutput:
    r"""Masked factorized Gaussian actor参数。"""

    mean: torch.Tensor  # $[B,16]$，ghost exact zero
    log_std: torch.Tensor  # scalar shared$\theta^{av}$


@dataclass(frozen=True)
class StructuredCriticOutput:
    r"""每environment一个hand-level scalar value。"""

    value: torch.Tensor  # $[B]$


__all__ = [
    "GEOMETRY_WIDTH",
    "HISTORY_LENGTH",
    "JOINT_COUNT",
    "OWNER_COUNT",
    "TIP_COUNT",
    "GeometryTokenBatch",
    "StructuredActorObservation",
    "StructuredActorOutput",
    "StructuredCriticObservation",
    "StructuredCriticOutput",
]
