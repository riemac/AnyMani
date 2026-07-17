r"""rl_games episode one-shot aggregation 与 central-value console cache contracts。"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from anymani.distill.rl.observers import OneShotIsaacAlgoObserver, mean_policy_action_std


class _Writer:
    r"""最小 SummaryWriter stub；本 contract 只检查 observer buffer lifecycle。"""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def add_scalar(self, *_args, **_kwargs) -> None:
        r"""接受 scalar 写入，不产生文件副作用。"""

        tag, value = _args[:2]
        self.scalars[str(tag)] = float(value)


def _algo() -> SimpleNamespace:
    r"""构造 IsaacAlgoObserver.after_init 所需的最小 algorithm surface。"""

    return SimpleNamespace(
        games_to_track=100,
        ppo_device="cpu",
        device="cpu",
        writer=_Writer(),
        write_stats=lambda *_args, **_kwargs: None,
    )


def test_episode_info_is_consumed_only_when_at_least_one_env_is_done() -> None:
    r"""无 done 的 stale `infos['episode']` 必须忽略；同一 reset batch 只追加一次。"""

    observer = OneShotIsaacAlgoObserver()
    observer.after_init(_algo())
    stale_info = {"episode": {"Task/episode_duration_s": torch.tensor([10.0])}}

    observer.process_infos(stale_info, torch.empty(0, dtype=torch.long))
    assert observer.ep_infos == []

    observer.process_infos(stale_info, torch.tensor([3, 7]))
    assert len(observer.ep_infos) == 1
    torch.testing.assert_close(observer.ep_infos[0]["Task/episode_duration_s"], torch.tensor([10.0, 10.0]))


def test_episode_batch_means_are_weighted_by_number_of_finished_episodes() -> None:
    r"""一个 3-env reset batch 的 scalar mean 权重应为单 env reset batch 的三倍。"""

    observer = OneShotIsaacAlgoObserver()
    algo = _algo()
    observer.after_init(algo)
    observer.process_infos({"episode": {"score": 1.0}}, torch.tensor([0]))
    observer.process_infos({"episode": {"score": 3.0}}, torch.tensor([1, 2, 3]))

    observer.after_print_stats(frame=10, epoch_num=2, total_time=1.0)

    assert algo.writer.scalars["Episode/score"] == 2.5  # $(1\times1+3\times3)/(1+3)$


def test_action_std_reads_custom_tcn_logstd() -> None:
    r"""TCN 的 trainable `logstd` 必须转换为实际 action standard deviation。"""

    network = SimpleNamespace(logstd=torch.tensor([-0.5, -0.5]))
    algo = SimpleNamespace(model=SimpleNamespace(a2c_network=network))

    torch.testing.assert_close(torch.tensor(mean_policy_action_std(algo)), torch.tensor(-0.5).exp())


def test_rsl_observer_caches_actual_central_value_loss() -> None:
    r"""Console observer 应缓存 central critic `train_net()` 返回值，而非 actor 占位 c_loss。"""

    observer = OneShotIsaacAlgoObserver()
    algo = _algo()
    central = SimpleNamespace(train_net=lambda: 0.375)
    algo.central_value_net = central
    observer.after_init(algo)

    loss = algo.central_value_net.train_net()

    assert loss == 0.375
    assert observer.last_central_value_loss == 0.375
