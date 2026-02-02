from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch

from isaaclab.managers import ObservationGroupCfg

from .obs_extractor import ObsGroupExtractor
from .teacher_policy import RLGamesTeacherPolicy


@dataclass
class Se3TermNames:
    names: list[str]


def _infer_teacher_obs_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """Infer expected teacher obs dimension from an rl_games checkpoint.

    Prefer `running_mean_std.running_mean` (policy obs normalization). Fall back to the
    RNN input projection if present.
    """

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support weights_only.
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(ckpt, dict):
        return None
    model = ckpt.get("model")
    if not isinstance(model, dict):
        return None

    mean_key = "running_mean_std.running_mean"
    if mean_key in model and hasattr(model[mean_key], "shape") and len(model[mean_key].shape) == 1:
        return int(model[mean_key].shape[0])

    for rnn_key in (
        "a2c_network.a_rnn.rnn.weight_ih_l0",
        "a2c_network.c_rnn.rnn.weight_ih_l0",
    ):
        if rnn_key in model and hasattr(model[rnn_key], "shape") and len(model[rnn_key].shape) == 2:
            return int(model[rnn_key].shape[1])
    return None


class DistillInfoWrapper(gym.Wrapper):
    """Inject teacher labels and alignment tensors into env extras.

    This wrapper computes, at *the current state* (pre-step):

    - `teacher_dq`: teacher relative joint delta (processed), shape (num_envs, 16)
    - `jacobian_pinv`: concatenated se3 Jacobian pseudoinverses for each finger,
      flattened to shape (num_envs, 96) for ExperienceBuffer aux storage.

    The wrapper is designed to be placed *below* IsaacLab's RlGamesVecEnvWrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        teacher_env_cfg: Any,
        teacher_agent_cfg: dict,
        teacher_checkpoint: str,
        teacher_obs_group: str = "policy",
        teacher_action_scale: float = 0.1,
        se3_term_names: list[str],
        align: str = "se3_to_relative_joint_delta",
        deterministic_teacher: bool = True,
    ) -> None:
        super().__init__(env)

        if align != "se3_to_relative_joint_delta":
            raise ValueError(f"Unsupported --align '{align}'.")

        self._teacher_env_cfg = teacher_env_cfg
        self._teacher_agent_cfg = teacher_agent_cfg
        self._teacher_checkpoint = teacher_checkpoint
        self._teacher_obs_group = teacher_obs_group
        self._teacher_action_scale = float(teacher_action_scale)
        self._deterministic_teacher = bool(deterministic_teacher)

        self._se3_term_names = list(se3_term_names)

        self._teacher_extractor: ObsGroupExtractor | None = None
        self._teacher_policy: RLGamesTeacherPolicy | None = None
        self._teacher_expected_obs_dim: int | None = None
        self._teacher_last_action: torch.Tensor | None = None

    def _ensure_teacher(self) -> None:
        if self._teacher_policy is not None and self._teacher_extractor is not None:
            return

        base_env = self.env.unwrapped
        device_name = str(base_env.device)

        obs_cfg = getattr(self._teacher_env_cfg, "observations")
        group_cfg_src: ObservationGroupCfg = getattr(obs_cfg, self._teacher_obs_group)

        if self._teacher_expected_obs_dim is None:
            self._teacher_expected_obs_dim = _infer_teacher_obs_dim_from_checkpoint(self._teacher_checkpoint)
        expected = self._teacher_expected_obs_dim

        # teacher action dim is inferred from teacher ActionsCfg (assumed continuous Box)
        # For our baseline joint teacher, it is 16.
        teacher_action_dim = 16

        if self._teacher_last_action is None:
            self._teacher_last_action = torch.zeros(
                base_env.num_envs,
                teacher_action_dim,
                device=base_env.device,
                dtype=torch.float32,
            )

        def _try_build_extractor(cfg: ObservationGroupCfg) -> tuple[ObsGroupExtractor, torch.Tensor]:
            extractor = ObsGroupExtractor(
                base_env,
                cfg,
                device=base_env.device,
                apply_corruption=False,
            )
            obs = extractor.compute(update_history=False)
            if isinstance(obs, dict):
                raise RuntimeError("Teacher obs group must be concatenated (Box) for rl_games teacher.")
            return extractor, obs

        def _configure_group(
            *,
            force_history_len: int | None,
            disable_terms: tuple[str, ...] = (),
        ) -> ObservationGroupCfg:
            cfg = copy.deepcopy(group_cfg_src)
            if force_history_len is not None and hasattr(cfg, "history_length"):
                setattr(cfg, "history_length", int(force_history_len))
                if hasattr(cfg, "flatten_history_dim"):
                    setattr(cfg, "flatten_history_dim", True)
            for name in disable_terms:
                if hasattr(cfg, name):
                    setattr(cfg, name, None)
            return cfg

        # Candidate configs in priority order:
        # - force history_length=1 (avoid checkpoint mismatch when cfg uses history stacking)
        # - additionally drop object pose terms (avoid privileged info & often matches older ckpts)
        # IMPORTANT: teacher obs cfg often includes `last_action`. If we evaluate it on the *student* env,
        # mdp.last_action will return the student's last action dim (e.g. 24 for 4xSE3), which breaks
        # checkpoint compatibility. Therefore, we drop the cfg's `last_action` term and append a
        # wrapper-maintained 16-dim teacher_last_action buffer.
        candidates: list[tuple[str, ObservationGroupCfg]] = [
            (
                "history=1,no_object_pose,no_last_action",
                _configure_group(force_history_len=1, disable_terms=("object_pos", "object_quat", "last_action")),
            ),
            ("history=1,no_last_action", _configure_group(force_history_len=1, disable_terms=("last_action",))),
            ("as_is,no_last_action", _configure_group(force_history_len=None, disable_terms=("last_action",))),
        ]

        chosen_tag: str | None = None
        chosen_obs: torch.Tensor | None = None
        last_obs_dim: int | None = None
        last_exc: Exception | None = None

        for tag, cfg in candidates:
            try:
                extractor, obs = _try_build_extractor(cfg)
                base_dim = int(obs.shape[-1])
                last_obs_dim = base_dim

                # Construct the final teacher obs that will be fed into the teacher policy.
                # Either:
                # - base_obs already matches expected
                # - or base_obs + teacher_action_dim matches expected (we append teacher_last_action)
                if expected is None:
                    self._teacher_extractor = extractor
                    chosen_tag = tag
                    chosen_obs = obs
                    break
                if base_dim == expected:
                    self._teacher_extractor = extractor
                    chosen_tag = tag
                    chosen_obs = obs
                    break
                if (base_dim + teacher_action_dim) == expected:
                    assert self._teacher_last_action is not None
                    self._teacher_extractor = extractor
                    chosen_tag = tag
                    chosen_obs = torch.cat([obs, self._teacher_last_action], dim=-1)
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        if self._teacher_extractor is None or chosen_obs is None:
            if last_exc is not None:
                raise RuntimeError(
                    f"Failed to build teacher obs extractor (expected={expected}, last_obs_dim={last_obs_dim})."
                ) from last_exc
            raise RuntimeError(f"Failed to build teacher obs extractor (expected={expected}, last_obs_dim={last_obs_dim}).")

        teacher_obs = chosen_obs
        if isinstance(teacher_obs, dict):
            raise RuntimeError("Teacher obs group must be concatenated (Box) for rl_games teacher.")

        if expected is not None and int(teacher_obs.shape[-1]) != expected:
            raise RuntimeError(
                f"Teacher obs dim mismatch: expected {expected}, got {int(teacher_obs.shape[-1])} (chosen='{chosen_tag}')."
            )

        if expected is not None:
            print(
                f"[distill] teacher obs group='{self._teacher_obs_group}' chosen='{chosen_tag}' obs_dim={int(teacher_obs.shape[-1])} expected={expected}"
            )

        self._teacher_policy = RLGamesTeacherPolicy.from_agent_cfg(
            self._teacher_agent_cfg["params"],
            checkpoint_path=self._teacher_checkpoint,
            obs_dim=int(teacher_obs.shape[-1]),
            action_dim=teacher_action_dim,
            device_name=device_name,
            action_scale=self._teacher_action_scale,
            num_envs=int(base_env.num_envs),
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ensure_teacher()
        assert self._teacher_extractor is not None
        assert self._teacher_policy is not None
        self._teacher_extractor.reset()
        self._teacher_policy.reset_all()
        if self._teacher_last_action is not None:
            self._teacher_last_action.zero_()
        return obs, info

    def step(self, action):
        self._ensure_teacher()
        assert self._teacher_extractor is not None
        assert self._teacher_policy is not None

        base_env = self.env.unwrapped

        # --- compute teacher labels at current state (pre-step) ---
        with torch.no_grad():
            teacher_obs_base = self._teacher_extractor.compute(update_history=True)
            if isinstance(teacher_obs_base, dict):
                raise RuntimeError("Teacher obs group must be concatenated (Tensor).")

            teacher_obs = teacher_obs_base
            if self._teacher_last_action is not None:
                expected = self._teacher_expected_obs_dim
                # Append teacher last_action if needed to match the checkpoint's expected dim.
                if expected is None or (int(teacher_obs_base.shape[-1]) + int(self._teacher_last_action.shape[-1]) == expected):
                    teacher_obs = torch.cat([teacher_obs_base, self._teacher_last_action], dim=-1)
            if isinstance(teacher_obs, dict):
                raise RuntimeError("Teacher obs group must be concatenated (Tensor).")

            teacher_raw = self._teacher_policy.act(teacher_obs, deterministic=self._deterministic_teacher)
            teacher_dq = teacher_raw * self._teacher_policy.action_scale

            # Update teacher last_action buffer with the *normalized* teacher action (as in mdp.last_action).
            if self._teacher_last_action is not None:
                self._teacher_last_action.copy_(teacher_raw)

            # compute jacobian pseudoinverses for each SE3 term
            j_pinv_list = []
            ad_list = []
            for name in self._se3_term_names:
                term = base_env.action_manager.get_term(name)
                jac = term._get_jacobian()  # noqa: SLF001
                jac_pinv = term._compute_jacobian_inverse(jac)  # noqa: SLF001
                j_pinv_list.append(jac_pinv.reshape(base_env.num_envs, -1))

                # If the action term applies an adjoint transform internally, export it.
                use_ad = bool(getattr(term.cfg, "is_xform", False) and getattr(term.cfg, "use_body_frame", False) and (not getattr(term.cfg, "use_xform_jacobian", False)))
                if use_ad and hasattr(term, "_Ad_b_bprime") and term._Ad_b_bprime is not None:  # noqa: SLF001
                    Ad = term._Ad_b_bprime  # noqa: SLF001
                    # Some terms store a single constant 6x6 adjoint (shared across envs).
                    if isinstance(Ad, torch.Tensor):
                        if Ad.ndim == 2:
                            Ad_flat = Ad.reshape(1, -1).expand(base_env.num_envs, -1)
                        elif Ad.ndim == 3:
                            Ad_flat = Ad.reshape(base_env.num_envs, -1)
                        elif Ad.ndim == 1:
                            Ad_flat = Ad.reshape(1, -1).expand(base_env.num_envs, -1)
                        else:
                            raise RuntimeError(f"Unexpected _Ad_b_bprime ndim={Ad.ndim} shape={tuple(Ad.shape)}")
                        ad_list.append(Ad_flat)
                    else:
                        ad_list.append(torch.zeros(base_env.num_envs, 36, device=base_env.device, dtype=torch.float32))
                else:
                    ad_list.append(torch.zeros(base_env.num_envs, 36, device=base_env.device, dtype=torch.float32))
            jacobian_pinv = torch.cat(j_pinv_list, dim=-1)
            ad_b_bprime = torch.cat(ad_list, dim=-1)

        obs, rew, terminated, truncated, extras = self.env.step(action)

        # inject
        extras["teacher_dq"] = teacher_dq
        extras["jacobian_pinv"] = jacobian_pinv
        extras["ad_b_bprime"] = ad_b_bprime

        # reset teacher history/rnn on done envs (so next label aligns with new episode)
        dones = (terminated | truncated)
        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._teacher_extractor.reset(done_ids)
            self._teacher_policy.reset_done(done_ids)
            if self._teacher_last_action is not None:
                self._teacher_last_action[done_ids] = 0

        return obs, rew, terminated, truncated, extras
