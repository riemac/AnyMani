r"""验证释放旧展开批次不改变PPO样本对应、优势/价值目标或冻结策略参考。"""

import copy
import weakref
from types import SimpleNamespace

import torch
from anymani.distill.rl.palm_rotation_ppo import PalmRotationPpoAgent


def _batch():
    r"""两资产、各两副本、H3，按真实env-major轴排列；每样本具有唯一内容。"""
    batch, horizon = 12, 3
    labels = (torch.arange(batch) // horizon % 2).reshape(-1, 1)  # e0:a0、e1:a1、e2:a0、e3:a1。
    return {"obses": {"prototype_index": labels, "jnt_history": torch.arange(batch * 30 * 16 * 5).float().reshape(batch, 30, 16, 5),
                       "cached_geometry_tokens": torch.arange(batch * 21 * 128).float().reshape(batch, 21, 128)},
            "returns": torch.arange(batch).float().reshape(-1, 1), "values": torch.ones(batch, 1),
            "dones": torch.zeros(batch), "actions": torch.arange(batch * 16).float().reshape(batch, 16),
            "neglogpacs": torch.arange(batch).float(), "mus": torch.ones(batch, 16), "sigmas": torch.full((batch, 16), .6),
            "direct_means": torch.ones(batch, 16), "film_modulations": torch.zeros(batch, 16), "step_time": .125}


def _agent(release):
    r"""只跳过仿真器构造，实际调用生产prepare_dataset及父类完整数据装配。"""
    agent = PalmRotationPpoAgent.__new__(PalmRotationPpoAgent)
    agent.config = {"optimization_audit_frequency": 0, "release_rollout_batch": release}
    agent.asset_count, agent.horizon_length, agent.batch_size = 2, 3, 12
    agent.num_minibatches, agent.minibatch_size = 2, 6
    agent.actor_arm = "direct_token"
    agent.advantage_normalization_scope = "per_asset_rollout"
    agent.normalize_value = agent.is_rnn = agent.normalize_rms_advantage = agent.has_central_value = False
    agent.normalize_advantage = True
    agent.value_mean_std = None
    dataset = SimpleNamespace(values_dict={})
    dataset.update_values_dict = lambda values: setattr(dataset, "values_dict", values)
    agent.dataset = dataset
    return agent


def test_release_preserves_every_dataset_field_and_frees_observation_copy():
    r"""同一随机分层排列下新旧全字段逐值一致，旧大观测对象的引用确实消失。"""
    original = _batch()
    released = copy.deepcopy(original)  # 独立对象，确保弱引用检查不被对照批次持有。
    weak_history = weakref.ref(released["obses"]["jnt_history"])
    weak_geometry = weakref.ref(released["obses"]["cached_geometry_tokens"])
    before, after = _agent(False), _agent(True)
    torch.manual_seed(42)
    before.prepare_dataset(original)
    torch.manual_seed(42)
    after.prepare_dataset(released)
    assert released == {"step_time": .125} and "obses" in original
    assert weak_history() is None and weak_geometry() is None  # 输入旧副本不再被batch或agent持有。
    assert after._release_unused_rollout_cache is True
    for key, value in before.dataset.values_dict.items():
        other = after.dataset.values_dict[key]
        if isinstance(value, dict):
            for field in value:
                torch.testing.assert_close(value[field], other[field], rtol=0, atol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            assert value == other
    frozen = after.dataset.values_dict["rollout_mu"].clone()
    after.dataset.values_dict["mu"].add_(1)  # 模拟训练中可变KL参考更新。
    torch.testing.assert_close(after.dataset.values_dict["rollout_mu"], frozen, rtol=0, atol=0)


def test_env_major_storage_two_rollouts_gae_permutation_and_reference_isolation():
    r"""两轮真实buffer写入/GAE/展开/排列均一致；展开是视图，下一轮不改旧策略参考。"""
    import gymnasium as gym
    import numpy as np
    from anymani.distill.rl.runtime.palm_rotation_experience import EnvMajorExperienceBuffer
    from rl_games.common.a2c_common import A2CBase, swap_and_flatten01
    from rl_games.common.experience import ExperienceBuffer

    env_info = {"agents": 1, "value_size": 1, "action_space": gym.spaces.Box(-1, 1, (16,)),
                "observation_space": gym.spaces.Dict({
                    "prototype_index": gym.spaces.Box(0, 1, (1,), dtype=np.int64),
                    "jnt_history": gym.spaces.Box(-np.inf, np.inf, (30, 16, 5)),
                    "cached_geometry_tokens": gym.spaces.Box(-np.inf, np.inf, (21, 128))})}
    info = {"num_actors": 4, "horizon_length": 3, "has_central_value": False, "use_pinned_memory": False}
    classic = ExperienceBuffer(env_info, info, "cpu")
    efficient = EnvMajorExperienceBuffer(env_info, info, "cpu")
    for name in ("direct_means", "film_modulations"):
        classic.tensor_dict[name] = torch.zeros(3, 4, 16)
        efficient.tensor_dict[name] = efficient.side_channel(16)
    gae = SimpleNamespace(horizon_length=3, gamma=.99, tau=.95)
    previous = None
    for cycle in range(2):
        for t in range(3):
            obs = {"prototype_index": (torch.arange(4) % 2).reshape(4, 1),
                   "jnt_history": torch.arange(4 * 30 * 16 * 5).float().reshape(4, 30, 16, 5) + cycle * 1000 + t,
                   "cached_geometry_tokens": torch.arange(4 * 21 * 128).float().reshape(4, 21, 128) + cycle * 1000 + t}
            for buffer in (classic, efficient):
                buffer.update_data("obses", t, obs)  # 真实time-major写入接口，不要求连续stride。
                for name, tensor in buffer.tensor_dict.items():
                    if name == "obses":
                        continue
                    value = torch.full_like(tensor[t], .1 * (cycle + t + 1))
                    if name == "dones":
                        value = torch.tensor([0, t == 1, 0, 0], dtype=torch.uint8)
                    buffer.update_data(name, t, value)
        if previous is not None:
            torch.testing.assert_close(previous[0], previous[1], rtol=0, atol=0)  # 再写原始buffer后旧rollout_mu保持。
        batches = []
        for buffer in (classic, efficient):
            tensors = buffer.tensor_dict
            advantages = A2CBase.discount_values(gae, torch.tensor([0., 1., 0., 0.]), torch.ones(4, 1),
                                                  tensors["dones"].float(), tensors["values"], tensors["rewards"])
            values = buffer.get_transformed_list(swap_and_flatten01, list(tensors))
            values["returns"] = swap_and_flatten01(advantages + tensors["values"])
            values["step_time"] = .125
            batches.append(values)
        flat = batches[1]["obses"]["cached_geometry_tokens"]
        backing = efficient.tensor_dict["obses"]["cached_geometry_tokens"]
        assert flat.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        torch.testing.assert_close(batches[0]["returns"], batches[1]["returns"], rtol=0, atol=0)
        agents = (_agent(False), _agent(True))
        for agent, values in zip(agents, batches):
            torch.manual_seed(100 + cycle)
            agent.prepare_dataset(values)  # 真实优势归一化与分层排列路径。
        for key, value in agents[0].dataset.values_dict.items():
            other = agents[1].dataset.values_dict[key]
            if isinstance(value, dict):
                for field in value:
                    torch.testing.assert_close(value[field], other[field], rtol=0, atol=0)
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
        previous = (agents[1].dataset.values_dict["rollout_mu"], agents[1].dataset.values_dict["rollout_mu"].clone())

    agent = _agent(True)
    agent.config["env_major_rollout_storage"] = True
    agent.num_agents, agent.num_actors, agent.value_size = 1, 4, 1
    agent.env_info, agent.ppo_device, agent.use_action_masks = env_info, "cpu", False
    agent.init_tensors()  # 项目侧实际接管入口及两种诊断通道也采用相同布局。
    for name in ("direct_means", "film_modulations"):
        tensor = agent.experience_buffer.tensor_dict[name]
        flattened = swap_and_flatten01(tensor)
        assert flattened.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


def test_update_boundary_releases_dataset_before_resource_measurement(monkeypatch):
    r"""显存门应位于本轮数据释放之后；数值结果和紧凑统计继续传递。"""
    from anymani.distill.rl.masked_ppo import AnyManiMaskedPpoAgent

    agent = _agent(True)
    agent._reset_optimization_metrics = lambda: None
    expected_result = (1.0, 2.0, 3.0)
    observed = []

    def completed_update(self):
        self.dataset.values_dict = {"obs": torch.ones(64, 64)}  # 代表已完成使用的整批dataset。
        self.released_dataset_ref = weakref.ref(self.dataset.values_dict["obs"])
        return expected_result

    def record(result):
        assert agent.dataset.values_dict is None
        assert agent.released_dataset_ref() is None
        observed.append(result)

    monkeypatch.setattr(AnyManiMaskedPpoAgent, "train_epoch", completed_update)
    agent._record_update_metrics = record
    assert agent.train_epoch() == expected_result and observed == [expected_result]
