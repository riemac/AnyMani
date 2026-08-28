r"""N000 current-frame + frozen $Z$ heterogeneous adapter 的纯张量合同。"""

from __future__ import annotations

import torch
from anymani.distill.rl.frozen_z import FrozenZProvider
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_MASKED_OBS_DIM,
    HeterogeneousN000MaskedPpoBuilder,
)
from anymani.distill.rl.masked_ppo import (
    ANYMANI_CHECKPOINT_IDENTITY_KEY,
    AnyManiMaskedContinuousModel,
    AnyManiMaskedPpoAgent,
    AnyManiMaskedPpoPlayer,
    validate_anymani_checkpoint_identity,
)
from rl_games.algos_torch import a2c_continuous


def _provider() -> FrozenZProvider:
    r"""构造 7/16 active-DOF 两资产的最小 manifest provider。"""

    joint = torch.tensor([[True] * 7 + [False] * 9, [True] * 16])
    owner = torch.zeros(2, 21, dtype=torch.bool)
    owner[:, 0] = True
    owner[:, 1:17] = joint
    owner[:, 17:] = True
    graph = torch.zeros(2, 21, 21, dtype=torch.long)
    graph[1, 0, 1:] = 2
    return FrozenZProvider(
        asset_ids=("seven", "sixteen"),
        physical_geometry_hashes=("p7", "p16"),
        owner_valid_mask=owner,
        joint_valid_mask=joint,
        shortest_path=graph,
        parent_direction=graph + 1,
        child_direction=graph + 2,
        dataset_digest="dataset",
        manifest_digest="manifest",
    )


def _model() -> AnyManiMaskedContinuousModel.Network:
    r"""构造不依赖 Isaac Sim 的 frozen-Z masked continuous model。"""

    builder = HeterogeneousN000MaskedPpoBuilder()
    builder.load(
        {
            "heterogeneous_policy": {"owner_feature_dim": 1, "joint_feature_dim": 3},
            "frozen_z_provider": _provider(),
        }
    )
    return AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": (HETEROGENEOUS_MASKED_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )


def _obs() -> torch.Tensor:
    r"""构造两项 N000 frame 与显式 routing metadata。"""

    obs = torch.zeros(2, HETEROGENEOUS_MASKED_OBS_DIM)
    obs[:, :16] = torch.arange(16) / torch.pi  # $q/\pi$
    obs[:, 16:32] = 0.25  # target/$\pi$
    obs[:, 32:48] = -0.5  # previous policy action
    obs[:, 48:52] = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    obs[:, 52] = torch.tensor([0.0, 1.0])  # asset row
    obs[0, 53:60] = 1.0  # 7 active joints
    obs[1, 53:] = 1.0  # 16 active joints
    return obs


def test_adapter_routes_n000_joint_tip_and_frozen_z_features() -> None:
    r"""52D frame 与 17D metadata 必须进入各自边界，不能互相泄漏。"""

    model = _model()
    policy_input = model.a2c_network._build_policy_input(_obs())

    torch.testing.assert_close(policy_input.joint_features[0, :, 0], _obs()[0, :16])
    torch.testing.assert_close(policy_input.joint_features[0, :, 1], _obs()[0, 16:32])
    torch.testing.assert_close(policy_input.joint_features[0, :, 2], _obs()[0, 32:48])
    torch.testing.assert_close(policy_input.owner_features[0, 17:21, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert policy_input.geometry_entities is not None
    assert policy_input.geometry_entities.shape == (2, 21, 128)
    assert policy_input.shortest_path[0, 0, 1] == 0
    assert policy_input.shortest_path[1, 0, 1] == 2


def test_masked_model_runs_sampling_backward_and_persists_frozen_provider() -> None:
    r"""策略/PPO计算链应有限，ghost action 为零，frozen bank 进入 checkpoint buffers。"""

    model = _model()
    sampled = model({"obs": _obs(), "is_train": False})
    active = model.a2c_network.last_active_joint_mask
    assert active is not None
    assert torch.all(sampled["actions"][~active] == 0.0)

    trained = model({"obs": _obs(), "prev_actions": torch.zeros(2, 16), "is_train": True})
    loss = trained["prev_neglogp"].mean() + trained["entropy"].mean() + trained["values"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(key.startswith("a2c_network.frozen_z_provider.z_table") for key in model.state_dict())
    assert sum(parameter.numel() for name, parameter in model.named_parameters() if "global_log_std" in name) == 1


def test_adapter_rejects_environment_mask_that_disagrees_with_manifest() -> None:
    r"""environment mask 被污染时必须在策略 forward 前失败。"""

    model = _model()
    obs = _obs()
    obs[0, 53] = 0.0
    try:
        model.a2c_network._build_policy_input(obs)
    except RuntimeError as exc:
        assert "active mask" in str(exc)
    else:
        raise AssertionError("routing mismatch must be rejected")


def test_checkpoint_identity_rejects_changed_dataset_before_buffer_restore() -> None:
    r"""dataset/manifest/row identity 改变时必须在 actor state dict 覆盖前失败。"""

    runtime = _provider().identity  # 当前环境按此 2-row manifest 构造
    validate_anymani_checkpoint_identity(runtime_identity=runtime, checkpoint_identity=dict(runtime))
    changed = dict(runtime)
    changed["dataset_digest"] = "different-dataset"
    try:
        validate_anymani_checkpoint_identity(runtime_identity=runtime, checkpoint_identity=changed)
    except RuntimeError as exc:
        assert "dataset_digest" in str(exc)
        assert "before model restore" in str(exc)
    else:
        raise AssertionError("changed dataset identity must fail before loading frozen buffers")


def test_checkpoint_identity_is_stable_after_provider_device_or_state_access() -> None:
    r"""checkpoint metadata 查询不得依赖 provider 当前 device，也不得重复改写 table digest。"""

    provider = _provider()
    before = provider.identity
    _ = provider.state_dict()  # 模拟 rl_games checkpoint 对 persistent buffers 的读取
    after = provider.identity

    assert before == after
    assert before["identity_digest"]
    assert before["z_table_sha256"]


def test_masked_agent_persists_identity_and_validates_before_base_restore(monkeypatch) -> None:
    r"""rl_games save/restore hook 必须写 identity，并在调用 upstream restore 前执行 gate。"""

    identity = _provider().identity
    network = type("Network", (), {"anymani_identity": identity})()
    model = type("Model", (), {"a2c_network": network})()
    agent = AnyManiMaskedPpoAgent.__new__(AnyManiMaskedPpoAgent)
    agent.model = model

    monkeypatch.setattr(a2c_continuous.A2CAgent, "get_full_state_weights", lambda _self: {"model": {}})
    state = agent.get_full_state_weights()
    assert state[ANYMANI_CHECKPOINT_IDENTITY_KEY] == identity

    restore_calls: list[bool] = []
    monkeypatch.setattr(
        a2c_continuous.A2CAgent,
        "set_full_state_weights",
        lambda _self, _weights, set_epoch=True: restore_calls.append(set_epoch),
    )
    agent.set_full_state_weights(state, set_epoch=False)
    assert restore_calls == [False]

    changed_state = dict(state)
    changed_state[ANYMANI_CHECKPOINT_IDENTITY_KEY] = {**identity, "manifest_digest": "different"}
    try:
        agent.set_full_state_weights(changed_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("identity mismatch must reject checkpoint before upstream restore")
    assert restore_calls == [False]  # mismatch 路径未进入 base state-dict/optimizer restore


def test_masked_player_validates_identity_before_model_restore(monkeypatch) -> None:
    r"""回放 player 与训练 agent 必须共享同一 pre-load identity gate。"""

    identity = _provider().identity

    class _Model:
        r"""只记录 ``load_state_dict`` 是否被调用的 player model 替身。"""

        a2c_network = type("Network", (), {"anymani_identity": identity})()

        def __init__(self) -> None:
            self.load_calls: list[dict] = []

        def load_state_dict(self, state: dict) -> None:
            self.load_calls.append(state)

    player = AnyManiMaskedPpoPlayer.__new__(AnyManiMaskedPpoPlayer)
    player.model = _Model()
    player.normalize_input = False
    player.env = None
    checkpoint = {
        "model": {"weights": torch.tensor([1.0])},
        ANYMANI_CHECKPOINT_IDENTITY_KEY: identity,
    }
    monkeypatch.setattr("anymani.distill.rl.masked_ppo.torch_ext.load_checkpoint", lambda _path: checkpoint)

    player.restore("valid.pth")
    assert len(player.model.load_calls) == 1

    checkpoint[ANYMANI_CHECKPOINT_IDENTITY_KEY] = {**identity, "dataset_digest": "different"}
    try:
        player.restore("invalid.pth")
    except RuntimeError:
        pass
    else:
        raise AssertionError("player must reject changed dataset before model buffer restore")
    assert len(player.model.load_calls) == 1  # mismatch 未调用第二次 load_state_dict
