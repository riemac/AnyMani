'Runtime contracts for observers.'

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


def mean_policy_action_std(algo: Any) -> float | None:
    'Handle mean policy action std.'

    model = getattr(algo, "model", None)
    network = getattr(model, "a2c_network", None)
    parameter = getattr(network, "sigma", None)
    parameter_is_logstd = "LogStd" in type(model).__qualname__
    if parameter is None:
        parameter = getattr(network, "logstd", None)  # TCN/custom continuous-logstd network contract
        parameter_is_logstd = parameter is not None
    if parameter is None:
        policy = getattr(network, "policy", None)
        parameter = getattr(policy, "global_log_std", None)
        parameter_is_logstd = parameter is not None
    if not isinstance(parameter, torch.Tensor):
        return None
    with torch.no_grad():
        value = parameter.detach().float()
        if parameter_is_logstd:
            value = torch.exp(value)
        return value.mean().item()


class OneShotIsaacAlgoObserver(IsaacAlgoObserver):
    'Contract for one shot isaac algo observer.'

    def after_init(self, algo: Any) -> None:
        'Handle after init.'

        super().after_init(algo)
        self.last_central_value_loss: float | None = None
        central_value_net = getattr(algo, "central_value_net", None)
        train_net = getattr(central_value_net, "train_net", None)
        if not callable(train_net):
            return

        def train_net_with_cache():
            'Train net with cache.'

            loss = train_net()
            if isinstance(loss, torch.Tensor):
                self.last_central_value_loss = loss.detach().float().mean().item()
            elif isinstance(loss, Real):
                self.last_central_value_loss = float(loss)
            else:
                raise TypeError(f"central_value_net.train_net() returned unsupported loss type {type(loss)}.")
            return loss

        setattr(central_value_net, "train_net", train_net_with_cache)

    def process_infos(self, infos: dict, done_indices: torch.Tensor) -> None:
        'Handle process infos.'

        if not isinstance(infos, dict):
            raise ValueError(f"{type(self).__name__} expected infos as dict, got {type(infos)}.")
        filtered_infos = infos
        if "episode" in infos:
            done_count = int(done_indices.numel())
            filtered_infos = dict(infos)
            if done_count == 0:
                filtered_infos.pop("episode", None)
            elif isinstance(infos["episode"], dict):


                expanded_episode = {}
                for key, value in infos["episode"].items():
                    tensor = torch.as_tensor(value)
                    if tensor.numel() == 1:
                        tensor = tensor.reshape(1).expand(done_count).clone()
                    expanded_episode[key] = tensor
                filtered_infos["episode"] = expanded_episode
        super().process_infos(filtered_infos, done_indices)


__all__ = ["OneShotIsaacAlgoObserver", "mean_policy_action_std"]
