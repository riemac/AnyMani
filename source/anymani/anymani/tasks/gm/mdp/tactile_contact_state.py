r"""GM tactile rotation 的共享 policy-rate 接触状态。

ContactSensor 在 physics rate 更新，但本任务的触觉判定是 20 Hz policy state。所有 actor、critic
与 reward consumer 必须读取同一个 EMA 快照：

$$
\bar f_t=(1-\alpha)\bar f_{t-1}+\alpha f_t,
\qquad c_t=\mathbf 1[\bar f_t>f_{th}],
$$

其中 $\alpha=0.5$、$f_{th}=0.25\,\mathrm N$。本模块以 `env.common_step_counter` 为时间戳，
保证 termination/reward/observation 在同一个 policy step 内无论调用多少次都只更新一次。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..contact_sensors import sensor_contact_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GmTactileContactState:
    r"""一个 vectorized env 唯一拥有的 tactile EMA/bits buffer。

    Args:
        env (ManagerBasedRLEnv): 提供 scene sensors、device、num_envs 与 policy-step counter 的环境。
        fingertip_sensor_names (Sequence[str]): 按 sidecar finger order 排列的 4 个 tip sensors。
        finger_non_tip_sensor_names (Sequence[str]): 只含 finger links、不含 palm 的 non-tip sensors。
        palm_sensor_name (str): 合法 palm support sensor。
        ema_alpha (float): 新 physics measurement 在 policy-rate EMA 中的权重。
        force_threshold (float): EMA 二值化阈值，单位 N。
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        fingertip_sensor_names: Sequence[str],
        finger_non_tip_sensor_names: Sequence[str],
        palm_sensor_name: str,
        ema_alpha: float = 0.5,
        force_threshold: float = 0.25,
    ) -> None:
        r"""校验 role schema，并分配共享 state buffers。"""

        self.fingertip_sensor_names = tuple(str(name) for name in fingertip_sensor_names)  # sidecar finger order
        self.finger_non_tip_sensor_names = tuple(str(name) for name in finger_non_tip_sensor_names)  # palm excluded
        self.palm_sensor_name = str(palm_sensor_name)  # 合法支撑面，不进入 bad-contact role
        if not self.fingertip_sensor_names:
            raise ValueError("fingertip_sensor_names must be non-empty.")
        sensor_names = (*self.fingertip_sensor_names, *self.finger_non_tip_sensor_names, self.palm_sensor_name)
        if any(not name for name in sensor_names) or len(set(sensor_names)) != len(sensor_names):
            raise ValueError(f"Tactile contact role sensor names must be non-empty and unique, got {sensor_names!r}.")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0,1], got {ema_alpha}.")
        if float(force_threshold) < 0.0:
            raise ValueError(f"force_threshold must be non-negative, got {force_threshold}.")

        self.ema_alpha = float(ema_alpha)  # $\alpha=0.5$ baseline
        self.force_threshold = float(force_threshold)  # $f_{th}=0.25$ N baseline
        self.sensor_names = sensor_names  # `[tip..., finger-non-tip..., palm]` canonical state order
        self.num_tips = len(self.fingertip_sensor_names)
        self.num_finger_non_tips = len(self.finger_non_tip_sensor_names)
        self.force_ema = torch.zeros(env.num_envs, len(sensor_names), dtype=torch.float32, device=env.device)  # N
        self.contact_bits = torch.zeros(env.num_envs, len(sensor_names), dtype=torch.bool, device=env.device)
        self.last_update_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)  # per-env stamp

    @property
    def tip_force_ema(self) -> torch.Tensor:
        r"""返回 `[B,K_tip]` 指尖 EMA 力幅值，单位 N。"""

        return self.force_ema[:, : self.num_tips]

    @property
    def tip_bits(self) -> torch.Tensor:
        r"""返回 `[B,K_tip]` 指尖 object-contact bits。"""

        return self.contact_bits[:, : self.num_tips]

    @property
    def finger_non_tip_force_ema(self) -> torch.Tensor:
        r"""返回 `[B,K_non-tip]` finger non-tip EMA 力，不包含 palm。"""

        start = self.num_tips
        return self.force_ema[:, start : start + self.num_finger_non_tips]

    @property
    def finger_non_tip_bits(self) -> torch.Tensor:
        r"""返回 `[B,K_non-tip]` finger non-tip bits，不包含 palm。"""

        start = self.num_tips
        return self.contact_bits[:, start : start + self.num_finger_non_tips]

    @property
    def palm_force_ema(self) -> torch.Tensor:
        r"""返回 `[B,1]` palm support EMA 力幅值，单位 N。"""

        return self.force_ema[:, -1:]

    @property
    def palm_bits(self) -> torch.Tensor:
        r"""返回 `[B,1]` palm support bits；该量只用于 critic/metric，不是 bad contact。"""

        return self.contact_bits[:, -1:]

    def ensure_updated(self, env: ManagerBasedRLEnv) -> None:
        r"""在当前 policy step 对尚未更新的 env 计算一次共享 EMA 快照。"""

        step = int(env.common_step_counter)  # ManagerBasedRLEnv 在 physics 后、termination 前递增
        update_mask = self.last_update_step != step  # partial reset 后可让 reset env 在本 stamp 保持零
        if not torch.any(update_mask):
            return

        raw_force = torch.stack(
            [sensor_contact_magnitude(env, sensor_name) for sensor_name in self.sensor_names], dim=-1
        )  # `[B,K]`，每个 sensor 内先取最大 body/filter-pair magnitude
        alpha = self.ema_alpha
        updated_ema = (1.0 - alpha) * self.force_ema[update_mask] + alpha * raw_force[update_mask]  # policy-rate EMA
        self.force_ema[update_mask] = updated_ema
        self.contact_bits[update_mask] = updated_ema > self.force_threshold  # baseline 使用严格 `>0.25 N`
        self.last_update_step[update_mask] = step

    def reset(self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        r"""清指定 env，并阻止 reset 后 observation 在当前 stamp 重新读取 sensor stale state。"""

        ids = _env_ids_tensor(env, env_ids)
        self.force_ema[ids] = 0.0
        self.contact_bits[ids] = False
        self.last_update_step[ids] = int(env.common_step_counter)  # 当前 stamp 已处理，下一 physics step 才重新采样


def get_tactile_contact_state(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: Sequence[str],
    finger_non_tip_sensor_names: Sequence[str],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
    *,
    update: bool = True,
) -> GmTactileContactState:
    r"""取得 env singleton，并按需幂等刷新当前 policy-step snapshot。"""

    expected_signature = (
        tuple(str(name) for name in fingertip_sensor_names),
        tuple(str(name) for name in finger_non_tip_sensor_names),
        str(palm_sensor_name),
        float(ema_alpha),
        float(force_threshold),
    )
    state = getattr(env, "_gm_tactile_contact_state", None)
    if state is None:
        state = GmTactileContactState(
            env=env,
            fingertip_sensor_names=fingertip_sensor_names,
            finger_non_tip_sensor_names=finger_non_tip_sensor_names,
            palm_sensor_name=palm_sensor_name,
            ema_alpha=ema_alpha,
            force_threshold=force_threshold,
        )
        setattr(env, "_gm_tactile_contact_state", state)  # actor/critic/reward 的唯一 contact predicate owner
    elif not isinstance(state, GmTactileContactState) or _state_signature(state) != expected_signature:
        raise RuntimeError(
            "This env already owns a tactile contact state with a different role/EMA contract; "
            f"existing={_state_signature(state) if isinstance(state, GmTactileContactState) else type(state)}, "
            f"requested={expected_signature}."
        )
    if update:
        state.ensure_updated(env)
    return state


def reset_tactile_contact_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    fingertip_sensor_names: Sequence[str],
    finger_non_tip_sensor_names: Sequence[str],
    palm_sensor_name: str,
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> None:
    r"""Reset event adapter：创建/取得 singleton，并只清本次 reset 的 env。"""

    state = get_tactile_contact_state(
        env=env,
        fingertip_sensor_names=fingertip_sensor_names,
        finger_non_tip_sensor_names=finger_non_tip_sensor_names,
        palm_sensor_name=palm_sensor_name,
        ema_alpha=ema_alpha,
        force_threshold=force_threshold,
        update=False,
    )
    state.reset(env, env_ids)


def _state_signature(state: GmTactileContactState) -> tuple[tuple[str, ...], tuple[str, ...], str, float, float]:
    r"""提取 singleton 配置签名，阻止同一 env 出现两套触觉定义。"""

    return (
        state.fingertip_sensor_names,
        state.finger_non_tip_sensor_names,
        state.palm_sensor_name,
        state.ema_alpha,
        state.force_threshold,
    )


def _env_ids_tensor(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    r"""把 Isaac Lab partial-reset id 表达统一成 env-device LongTensor。"""

    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        if env_ids == slice(None):
            return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


__all__ = ["GmTactileContactState", "get_tactile_contact_state", "reset_tactile_contact_state"]
