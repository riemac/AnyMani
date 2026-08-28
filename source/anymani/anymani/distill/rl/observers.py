r"""AnyMani rl_games observers 的 episode/TensorBoard 边界语义。

Isaac Lab 的 `extras["log"]` 会在 reset step 生成，并在后续非 reset steps 保留。上游
`IsaacAlgoObserver` 无条件追加 `infos["episode"]`，会把同一批已结束 episode 重复计入。
本模块只修正统计消费时机并缓存 central-value loss；不改变 rollout、advantage 或 PPO 更新。
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


def mean_policy_action_std(algo: Any) -> float | None:
    r"""读取 native rl_games `sigma` 或 AnyMani custom actor `logstd` 的实际标准差均值。"""

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
    r"""只在 `done_indices` 非空时消费一次 episode extras。

    `process_infos()` 仍把非 episode scalar 交给上游 observer；唯一变化是把 stale
    `episode` key 从无终止 step 的 infos 副本中移除。central value 的 `train_net()`
    返回实际 asymmetric critic loss，本 observer 缓存该标量供 console 使用。
    """

    def after_init(self, algo: Any) -> None:
        r"""初始化上游 meters，并为可选 central critic 安装无副作用 loss cache。"""

        super().after_init(algo)
        self.last_central_value_loss: float | None = None  # 最近一次 asymmetric critic epoch loss
        central_value_net = getattr(algo, "central_value_net", None)
        train_net = getattr(central_value_net, "train_net", None)
        if not callable(train_net):
            return

        def train_net_with_cache():
            r"""调用原 central update，并缓存其返回的平均 loss。"""

            loss = train_net()
            if isinstance(loss, torch.Tensor):
                self.last_central_value_loss = loss.detach().float().mean().item()
            elif isinstance(loss, Real):
                self.last_central_value_loss = float(loss)
            else:
                raise TypeError(f"central_value_net.train_net() returned unsupported loss type {type(loss)}.")
            return loss

        setattr(central_value_net, "train_net", train_net_with_cache)  # 只包统计缓存，不改变 optimizer/control flow

    def process_infos(self, infos: dict, done_indices: torch.Tensor) -> None:
        r"""过滤 stale payload，并按本 step 的结束 episode 数恢复 batch-mean 权重。"""

        if not isinstance(infos, dict):
            raise ValueError(f"{type(self).__name__} expected infos as dict, got {type(infos)}.")
        filtered_infos = infos
        if "episode" in infos:
            done_count = int(done_indices.numel())
            filtered_infos = dict(infos)  # 不原地修改 wrapper 返回给其他 observer/consumer 的 extras
            if done_count == 0:
                filtered_infos.pop("episode", None)
            elif isinstance(infos["episode"], dict):
                # Isaac Lab manager extras 已是 reset subset mean。重复 K 次后，上游 concat+mean 等价于
                # 按每个结束 episode 加权，而不是让 1-env 与 1000-env reset batches 拥有相同权重。
                expanded_episode = {}
                for key, value in infos["episode"].items():
                    tensor = torch.as_tensor(value)
                    if tensor.numel() == 1:
                        tensor = tensor.reshape(1).expand(done_count).clone()
                    expanded_episode[key] = tensor
                filtered_infos["episode"] = expanded_episode
        super().process_infos(filtered_infos, done_indices)


__all__ = ["OneShotIsaacAlgoObserver", "mean_policy_action_std"]
