r"""Contract tests for the GM teacher rl_games Transformer adapter.

这些测试不启动 Isaac Sim，也不依赖 PhysX。它们只验证 `distill/rl` adapter 的
最小数学 / 接口契约：

1. flat obs 能按 GM policy obs 布局拆成 $N$ 个 JOINT token 和 1 个 COMMAND token；
2. rl_games `continuous_a2c_logstd` 期望的 `(mu, logstd, value, states)` 形状成立；
3. obs/action 维度不匹配时显式失败，而不是静默把错误维度喂进网络。
"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

prefer_local_rl_games(strict=True)

from anymani.distill.rl.rl_games_networks import AnyManiGmTransformerNetwork, GmFlatObsLayout  # noqa: E402


def _network_params() -> dict:
    r"""返回测试用的最小 Transformer 配置。

    Returns:
        dict: 与 `gm_teacher_transformer_ppo.yaml` 中 `network` 字段同形状的配置。
    """

    return {
        "space": {"continuous": {"sigma_init": {"val": -1.5}}},  # 初始 log std 数值锚点
        "transformer": {
            "embed_dim": 32,  # 测试用小宽度，保持 `embed_dim % num_heads == 0`
            "num_layers": 2,  # 至少两层，覆盖 encoder stack 而非单层特例
            "num_heads": 4,  # 多头注意力 contract
            "dropout": 0.0,  # 测试确定性，避免 dropout 随机扰动 shape 以外判断
            "token_feature_dim": 5,  # `[q, qdot, last_action, q_min, q_max]`
            "command_feature_dim": 6,  # `[axis_h, error_so3_h]`
        },
    }


def test_gm_transformer_forward_matches_rl_games_contract() -> None:
    r"""验证 Transformer teacher 满足 rl_games continuous logstd 网络契约。"""

    joint_count = 16  # debug asset `right_t4_i4_m4_r4` 的 DOF 数
    layout = GmFlatObsLayout(joint_count=joint_count)  # flat obs 布局 $5N+6$
    batch_size = 7  # 非 2 的幂，避免只在整齐 batch 上通过
    network = AnyManiGmTransformerNetwork(
        _network_params(),
        actions_num=joint_count,
        input_shape=(layout.expected_obs_dim,),
        value_size=1,
        num_seqs=1,
    )

    obs = torch.randn(batch_size, layout.expected_obs_dim)  # `[B,5N+6]`，模拟 rl_games norm 后 flat obs
    mu, logstd, value, states = network({"obs": obs})  # rl_games 会以 dict 形式调用网络

    assert mu.shape == (batch_size, joint_count)  # 每个 joint token 输出一个 action mean
    assert logstd.shape == (batch_size, joint_count)  # fixed sigma broadcast 到每个 batch/joint
    assert value.shape == (batch_size, 1)  # critic value 标量
    assert states is None  # 第一版不是 RNN，不返回 recurrent states
    assert torch.isfinite(mu).all()  # 防止初始化或 token 切片产生 NaN/Inf
    assert torch.isfinite(logstd).all()
    assert torch.isfinite(value).all()


def test_gm_transformer_rejects_obs_action_layout_mismatch() -> None:
    r"""obs/action schema 不一致时必须显式失败。"""

    joint_count = 16  # action 维度来自 env action space
    wrong_obs_dim = GmFlatObsLayout(joint_count=joint_count).expected_obs_dim - 1  # 模拟漏掉一个 obs 标量
    with pytest.raises(ValueError, match="layout mismatch"):
        AnyManiGmTransformerNetwork(
            _network_params(),
            actions_num=joint_count,
            input_shape=(wrong_obs_dim,),
            value_size=1,
            num_seqs=1,
        )
