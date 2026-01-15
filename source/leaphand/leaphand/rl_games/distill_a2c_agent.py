from __future__ import annotations

from typing import Any

import time
import torch

from rl_games.common import a2c_common
from rl_games.common.experience import ExperienceBuffer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    # mask shape: (batch,) or (batch,1)
    if mask.ndim == 2:
        mask = mask.squeeze(-1)
    mask = mask.float()
    return (x * mask).sum() / (mask.sum().clamp_min(1.0))


class DistillA2CAgent(A2CAgent):
    """PPO(A2C continuous) + online imitation loss.

    This agent expects the environment to provide two extra tensors in infos:

    - `teacher_dq`: (num_envs, 16)
    - `jacobian_pinv`: (num_envs, 96)

    They are stored in ExperienceBuffer aux tensors and used to compute an imitation
    MSE loss between teacher joint delta-q and student's SE(3) action mapped to joint
    delta-q using stored Jacobian pseudoinverses.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        distill_cfg = self.config.get("distill", {})
        self.distill_coef = float(distill_cfg.get("imit_coef", 0.0))
        self.teacher_action_dim = int(distill_cfg.get("teacher_action_dim", 16))
        self.jacobian_pinv_dim = int(distill_cfg.get("jacobian_pinv_dim", 96))
        self.ad_b_bprime_dim = int(distill_cfg.get("ad_b_bprime_dim", 0))
        self.dt = float(distill_cfg.get("dt", 0.0))
        self.se3_terms = distill_cfg.get("se3_terms", [])
        self.teacher_joint_names = distill_cfg.get("teacher_joint_names", None)

        if self.distill_coef <= 0.0:
            raise ValueError("DistillA2CAgent requires distill.imit_coef > 0.")
        if not self.se3_terms:
            raise ValueError("Missing distill.se3_terms; cannot compute imitation mapping.")

        # Build joint-name based reorder indices (student dq order -> teacher dq order).
        self._dq_reindex: torch.Tensor | None = None
        if self.teacher_joint_names is not None:
            if not isinstance(self.teacher_joint_names, list) or len(self.teacher_joint_names) != self.teacher_action_dim:
                raise ValueError(
                    f"distill.teacher_joint_names must be a list of length {self.teacher_action_dim}."
                )
            student_joint_names: list[str] = []
            for t in self.se3_terms:
                student_joint_names.extend(list(t.get("joint_names", [])))
            if len(student_joint_names) != self.teacher_action_dim:
                raise ValueError(
                    f"Concatenated se3_terms[*].joint_names must have length {self.teacher_action_dim}. "
                    f"Got {len(student_joint_names)}."
                )
            try:
                idx = [student_joint_names.index(name) for name in self.teacher_joint_names]
            except ValueError as exc:
                raise ValueError(
                    "Failed to align joint names between teacher and student. "
                    "Ensure se3_terms[*].joint_names and teacher_joint_names use the same naming convention."
                ) from exc
            self._dq_reindex = torch.tensor(idx, device=self.ppo_device, dtype=torch.long)

        # cache per-term constants on device
        self._se3_constants: list[dict[str, Any]] = []
        for t in self.se3_terms:
            c = {
                "angular_scale": torch.tensor(t["angular_scale"], device=self.ppo_device, dtype=torch.float32),
                "angular_bias": torch.tensor(t["angular_bias"], device=self.ppo_device, dtype=torch.float32),
                "linear_scale": torch.tensor(t["linear_scale"], device=self.ppo_device, dtype=torch.float32),
                "linear_bias": torch.tensor(t["linear_bias"], device=self.ppo_device, dtype=torch.float32),
                "angular_vel_limits": None,
                "linear_vel_limits": None,
                "use_ad": bool(t.get("use_ad", False)),
                "nj": int(t.get("nj", 4)),
            }
            if t.get("angular_vel_limits") is not None:
                c["angular_vel_limits"] = torch.tensor(t["angular_vel_limits"], device=self.ppo_device, dtype=torch.float32)
            if t.get("linear_vel_limits") is not None:
                c["linear_vel_limits"] = torch.tensor(t["linear_vel_limits"], device=self.ppo_device, dtype=torch.float32)
            self._se3_constants.append(c)

    # ----------------- rollout collection -----------------

    def init_tensors(self):
        # Mirror rl_games.common.a2c_common.A2CBase.init_tensors, but allocate aux tensors too.
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": self.use_action_masks,
        }
        aux = {
            "teacher_dq": (self.teacher_action_dim,),
            "jacobian_pinv": (self.jacobian_pinv_dim,),
        }
        if self.ad_b_bprime_dim > 0:
            aux["ad_b_bprime"] = (self.ad_b_bprime_dim,)

        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device, aux_tensor_dict=aux)

        current_rewards_shape = (batch_size, self.value_size)
        self.init_current_rewards(batch_size, current_rewards_shape)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            if not ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0):
                raise ValueError(
                    f"Horizon length ({self.horizon_length}) times total agents ({total_agents}) divided by num minibatches ({self.num_minibatches}) must be divisible by sequence length ({self.seq_length})"
                )
            self.mb_rnn_states = [
                torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32, device=self.ppo_device)
                for s in self.rnn_states
            ]

        # Mirror rl_games.common.a2c_common.ContinuousA2CBase.init_tensors
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        if self.use_action_masks:
            self.update_list += ["action_masks"]

        # add distill tensors
        self.update_list += ["teacher_dq", "jacobian_pinv"]
        if self.ad_b_bprime_dim > 0:
            self.update_list += ["ad_b_bprime"]

        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def _update_aux_from_infos(self, infos: dict, n: int) -> None:
        if "teacher_dq" in infos:
            self.experience_buffer.update_data("teacher_dq", n, infos["teacher_dq"])
        if "jacobian_pinv" in infos:
            self.experience_buffer.update_data("jacobian_pinv", n, infos["jacobian_pinv"])
        if self.ad_b_bprime_dim > 0 and "ad_b_bprime" in infos:
            self.experience_buffer.update_data("ad_b_bprime", n, infos["ad_b_bprime"])

    def play_steps(self):
        update_list = self.update_list
        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                # teacher_dq/jacobian_pinv will be filled after env_step
                if k in ("teacher_dq", "jacobian_pinv", "ad_b_bprime"):
                    continue
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.perf_counter()  # type: ignore[name-defined]
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.perf_counter()  # type: ignore[name-defined]
            step_time += (step_time_end - step_time_start)

            self._update_aux_from_infos(infos, n)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += self.gamma * res_dict["values"] * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        return batch_dict

    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_length == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_length, :, :, :] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.rnn_states = res_dict["rnn_states"]
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones.byte())

            for k in update_list:
                if k in ("teacher_dq", "jacobian_pinv", "ad_b_bprime"):
                    continue
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.perf_counter()  # type: ignore[name-defined]
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.perf_counter()  # type: ignore[name-defined]
            step_time += (step_time_end - step_time_start)

            self._update_aux_from_infos(infos, n)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += self.gamma * res_dict["values"] * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        # match rl_games.common.a2c_common.A2CBase.play_steps_rnn(): pack RNN states for the dataset
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        batch_dict["rnn_states"] = states

        batch_dict["step_time"] = step_time

        return batch_dict

    # ----------------- dataset -----------------

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        # append aux tensors
        dataset_dict = {"teacher_dq": batch_dict["teacher_dq"], "jacobian_pinv": batch_dict["jacobian_pinv"]}
        if self.ad_b_bprime_dim > 0:
            dataset_dict["ad_b_bprime"] = batch_dict["ad_b_bprime"]
        self.dataset.update_values_dict({**self.dataset.values_dict, **dataset_dict})

    # ----------------- gradients / loss -----------------

    def _mu_to_dq_pred(self, mu: torch.Tensor, jacobian_pinv_flat: torch.Tensor, ad_b_bprime_flat: torch.Tensor | None) -> torch.Tensor:
        # mu shape: (B, 24) ; jacobian_pinv_flat: (B, 96)
        B = mu.shape[0]
        dq_parts = []
        pinv_parts = torch.split(jacobian_pinv_flat, [c["nj"] * 6 for c in self._se3_constants], dim=-1)
        ad_parts = None
        if ad_b_bprime_flat is not None:
            ad_parts = torch.split(ad_b_bprime_flat, [36 for _ in self._se3_constants], dim=-1)

        offset = 0
        for idx, c in enumerate(self._se3_constants):
            mu_i = mu[:, offset : offset + 6]
            offset += 6

            scaled_angular = mu_i[:, :3] * c["angular_scale"] + c["angular_bias"]
            scaled_linear = mu_i[:, 3:] * c["linear_scale"] + c["linear_bias"]
            twist_angular = scaled_angular
            twist_linear = scaled_linear
            if c["angular_vel_limits"] is not None:
                twist_angular = torch.clamp(
                    twist_angular,
                    min=c["angular_vel_limits"][0],
                    max=c["angular_vel_limits"][1],
                )
            if c["linear_vel_limits"] is not None:
                twist_linear = torch.clamp(
                    twist_linear,
                    min=c["linear_vel_limits"][0],
                    max=c["linear_vel_limits"][1],
                )
            twist = torch.cat([twist_angular, twist_linear], dim=-1)

            if c["use_ad"] and ad_parts is not None:
                Ad = ad_parts[idx].reshape(B, 6, 6)
                twist = (Ad @ twist.unsqueeze(-1)).squeeze(-1)

            pinv = pinv_parts[idx].reshape(B, c["nj"], 6)
            joint_vel = (pinv @ twist.unsqueeze(-1)).squeeze(-1)
            dq = joint_vel * self.dt
            dq_parts.append(dq)

        return torch.cat(dq_parts, dim=-1)

    def calc_gradients(self, input_dict):
        # extend base calc_gradients by adding imitation loss
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = self._preproc_obs(input_dict["obs"])
        rnn_masks = input_dict.get("rnn_masks", None)

        teacher_dq = input_dict["teacher_dq"].detach()
        jacobian_pinv = input_dict["jacobian_pinv"].detach()
        ad_b_bprime = input_dict.get("ad_b_bprime", None)
        if ad_b_bprime is not None:
            ad_b_bprime = ad_b_bprime.detach()

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }
        if self.is_rnn:
            batch_dict["rnn_states"] = input_dict.get("rnn_states", None)
            batch_dict["seq_length"] = self.seq_length

        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None

        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            loss, a_loss, c_loss, entropy, b_loss, sum_mask = self.calc_losses(
                self.actor_loss_func,
                old_action_log_probs_batch,
                action_log_probs,
                advantage,
                self.e_clip,
                value_preds_batch,
                values,
                return_batch,
                mu,
                entropy,
                rnn_masks,
            )

            dq_pred = self._mu_to_dq_pred(mu, jacobian_pinv, ad_b_bprime)
            if self._dq_reindex is not None:
                dq_pred = dq_pred.index_select(dim=-1, index=self._dq_reindex)
            imit_per_env = torch.mean((dq_pred - teacher_dq) ** 2, dim=-1)
            imit_loss = _masked_mean(imit_per_env, rnn_masks)
            loss = loss + self.distill_coef * imit_loss

            aux_loss = self.model.get_aux_loss()
            # rl_games stats writer expects lists of tensors.
            self.aux_loss_dict = {"imit_loss": [imit_loss.detach()]}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    if k in self.aux_loss_dict:
                        # accumulate values across minibatches
                        if isinstance(self.aux_loss_dict[k], list):
                            self.aux_loss_dict[k].append(v.detach())
                        else:
                            self.aux_loss_dict[k] = [self.aux_loss_dict[k], v.detach()]
                    else:
                        self.aux_loss_dict[k] = [v.detach()]

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )
