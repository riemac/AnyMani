r"""GM tactile rotation causal TCN 的纯 tensor contract tests。

测试不启动 Isaac Sim，也不依赖 rl_games rollout。它直接证伪时间编码器最容易被
实现错的三个命题：长度链是否为 $30\rightarrow12\rightarrow8\rightarrow4$；最终
latent 是否使用完整 temporal map；以及每个 temporal output 是否只读取其当前及过去
frame，而不越过自己的 receptive-field 右边界。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from anymani.distill.models.temporal_encoder import TactileTemporalConvEncoder
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

prefer_local_rl_games(strict=True)

from anymani.distill.rl.rl_games_networks import AnyManiTactileTcnNetwork  # noqa: E402


def _deterministic_positive_encoder() -> TactileTemporalConvEncoder:
    r"""返回所有线性权重为正的确定性 TCN。

    正权重与正输入保证 ReLU 不会把某个输入 frame 的扰动偶然截成零，因此“30 帧均
    可影响 latent”测试检验的是结构覆盖，而不是某次随机初始化的激活状态。

    Returns:
        TactileTemporalConvEncoder: 输入 52D、输出 64D 的 eval-mode encoder。
    """

    encoder = TactileTemporalConvEncoder(frame_dim=52, latent_dim=64)  # 生产配置的 shape contract
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.weight.fill_(0.01)  # 全正权重，使任意输入维度的正扰动沿所有可达路径传播
                if module.bias is not None:
                    module.bias.zero_()  # 去掉偏置，输出差异只来自输入 frame
    return encoder.eval()


def test_tcn_temporal_lengths_and_latent_shape() -> None:
    r"""三层卷积应保留 4 个 temporal positions，再投影为 64D latent。"""

    encoder = TactileTemporalConvEncoder(frame_dim=52, latent_dim=64)  # `[9,5,5] / [2,1,1]`
    history = torch.randn(7, 30, 52)  # `[B,T,D]`，非整齐 batch 用于覆盖 reshape 路径

    temporal_map, stage_lengths = encoder.temporal_features(history, return_stage_lengths=True)
    latent = encoder(history)

    assert stage_lengths == (12, 8, 4)  # oldest-frame 左填充 1 帧后的精确长度链
    assert temporal_map.shape == (7, 64, 4)  # 完整第三层 map，不截取 final position
    assert latent.shape == (7, 64)  # `[B,64]`，供 policy MLP 与当前 52D frame 拼接
    assert torch.isfinite(latent).all()


def test_every_history_frame_can_change_tcn_latent() -> None:
    r"""逐一扰动 30 个 frame 时，最终 latent 都必须改变。

    第三层单个 position 的 receptive field 只有 25 帧，但四个 position 的并集应覆盖
    全部 30 帧。该测试会直接捕获“只取 final position”或 stride alignment 漏掉 oldest/latest
    frame 的实现错误。
    """

    encoder = _deterministic_positive_encoder()  # 固定正权重，隔离网络初始化随机性
    baseline = torch.ones(1, 30, 52)  # 正输入使三层 ReLU 始终位于线性区
    baseline_latent = encoder(baseline)  # `[1,64]`，所有逐帧扰动的共同参照

    for frame_index in range(30):
        perturbed = baseline.clone()  # 每次只改变一个历史时间位置
        perturbed[:, frame_index, 0] += 1.0  # 扰动同一 feature，排除 feature routing 差异
        difference = torch.max(torch.abs(encoder(perturbed) - baseline_latent))
        assert difference > 0.0, f"history frame {frame_index} cannot influence the TCN latent"


def test_temporal_positions_do_not_read_beyond_their_receptive_field() -> None:
    r"""每个第三层 temporal position 只能读取其 25-frame causal receptive field。

    oldest-frame 左填充后，第三层 position $r$ 覆盖 padded indices
    $[2r,2r+24]$。因此第一个 position 不得响应原始 frame 24 之后的输入；最后一个
    position 必须响应最新 frame 29。
    """

    encoder = _deterministic_positive_encoder()  # temporal map 差异可直接解释为输入可达性
    baseline = torch.ones(1, 30, 52)  # `[1,30,52]`，正激活基线
    baseline_map = encoder.temporal_features(baseline)  # `[1,64,4]`

    future_for_first = baseline.clone()
    future_for_first[:, 29, 0] += 1.0  # 最新 frame 位于第一个 output 的 receptive field 之外
    changed_map = encoder.temporal_features(future_for_first)
    assert torch.equal(changed_map[:, :, 0], baseline_map[:, :, 0])  # position 0 不读取未来窗口末端

    assert not torch.equal(changed_map[:, :, -1], baseline_map[:, :, -1])  # position 3 覆盖最新 frame 29


def test_tcn_actor_matches_rl_games_continuous_logstd_contract() -> None:
    r"""TCN actor 应直接消费 `[B,30,52]` 并输出 16D continuous action。"""

    params = {
        "space": {"continuous": {"sigma_init": {"val": -0.5}}},
        "tactile_tcn": {
            "frame_dim": 52,
            "latent_dim": 64,
            "hidden_channels": [64, 64, 64],
            "mlp_units": [512, 256, 128],
            "activation": "elu",
        },
    }
    network = AnyManiTactileTcnNetwork(
        params,
        actions_num=16,
        input_shape=(30, 52),
        value_size=1,
        num_seqs=1,
    )
    history = torch.randn(11, 30, 52)  # `[B,T,D]`，wrapper 保留 rank-2 sample shape

    mu, logstd, value, states = network({"obs": history})

    assert mu.shape == (11, 16)  # 16 个 canonical revolute joints 的 action mean
    assert logstd.shape == (11, 16)  # fixed log sigma broadcast 到 batch/action 维
    assert value.shape == (11, 1)  # rl_games 主 model 仍要求占位 value tensor
    assert states is None  # TCN history 由 env observation 提供，不是 rl_games recurrent state
    assert torch.allclose(logstd, torch.full_like(logstd, -0.5))
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
