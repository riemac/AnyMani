r"""固定首轨迹审计：异步终止、reset后排除、分块边界与可重算文件身份。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from anymani.distill.diagnostics.analysis.rl.palm_rotation_trace import audit_palm_rotation_trace
from anymani.distill.diagnostics.recording.rl.palm_rotation import write_selected_trajectories_hdf5


def seal_relay_identity(case):
    r"""对接力夹具的新协议重新绑定两个HDF5身份，保留各自的轴说明。"""
    output, document = case
    identity = document['evaluation_identity']  # 已修改的协议。
    identity.pop('identity_digest', None)
    identity['identity_digest'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for path in (output.parent / 'evaluation.trace.h5', output.parent / 'evaluation.h5'):
        with h5py.File(path, 'r+') as stream:
            metadata = json.loads(str(stream.attrs['metadata_json']))  # 保留trace axes/reward names。
            metadata.update(identity)
            stream.attrs['metadata_json'] = json.dumps(metadata)


def relay_trace_case(case, *, replace=True):
    r"""一个旧动作后交接，第2个动作必须来自新Actor；边界包含两个仍存活副本。"""
    output, document = case
    identity = document['evaluation_identity']
    replacement = output.parent / 'replacement.pth'
    replacement.write_bytes(b'relay fixture; no model deserialization')  # reader只验证字节身份。
    before = np.full((2, 16), .1, dtype=np.float32)
    after = np.full((2, 16), .2 if replace else .1, dtype=np.float32)
    boundary_path = output.parent / 'boundary.h5'
    boundary_meta = {'method_identity_digest': identity['method_identity_digest'], 'boundary_step': 1,
                     'initial_checkpoint_sha256': identity['checkpoint_sha256'],
                     'replacement_checkpoint_sha256': hashlib.sha256(replacement.read_bytes()).hexdigest() if replace else None}
    write_selected_trajectories_hdf5(boundary_path, arrays={
        'actor_mean_before': before, 'actor_mean_after': after, 'active_first_trajectory': np.ones(2, dtype=bool),
        'obs__actor_jnt_current': np.zeros((2, 16, 5), dtype=np.float32),
        'obs__actor_jnt_history': np.zeros((2, 30, 16, 5), dtype=np.float32),
        'obs__geometry_tokens': np.zeros((2, 21, 128), dtype=np.float32),
        'controller__current_targets': np.full((2, 16), .25, dtype=np.float32),
        'physics__joint_position': np.zeros((2, 16), dtype=np.float32),
        'physics__object_root_state': np.zeros((2, 13), dtype=np.float32),
    }, metadata=boundary_meta)  # 固定边界输入；不假造仿真器内部快照。
    with h5py.File(output.parent / 'evaluation.trace.h5', 'r+') as trace:
        actions = np.stack([before, after, after])[:, None]  # [T,A,R,16]。
        trace.create_dataset('action', data=actions)
        trace.create_dataset('actor_checkpoint_phase', data=np.array([0, int(replace), int(replace)], dtype=np.int8)[:, None, None].repeat(2, axis=2))
    identity['actor_switch_source_sha256'] = 'c' * 64
    identity['protocol'].update({
        'evaluation_role': 'diagnostic', 'reliable_topology_coverage_protocol_matched': False,
        'scale_ready_protocol_matched': False,
        'actor_relay': {
            'boundary_step': 1, 'boundary_time_s': .05, 'replacement_requested': replace,
            'boundary_reached': True, 'performed': replace, 'final_actor_check': 'bitwise-equal',
            'initial_checkpoint_sha256': identity['checkpoint_sha256'],
            'replacement_checkpoint': str(replacement) if replace else None,
            'replacement_checkpoint_sha256': boundary_meta['replacement_checkpoint_sha256'],
            'first_replacement_action_step': 2 if replace else None,
            'state_continuity_check': 'bitwise-equal', 'torch_rng_check': 'bitwise-equal',
            'prefix_actor_check': 'bitwise-equal', 'replacement_actor_check': 'bitwise-equal',
            'boundary_snapshot': {'path': str(boundary_path), 'sha256': hashlib.sha256(boundary_path.read_bytes()).hexdigest()},
        },
    })
    document['artifact_type'] = 'anymani.palm_rotation_actor_relay_diagnostic'
    seal_relay_identity(case)  # 各自HDF5与JSON使用同一新评价身份。
    return case


@pytest.mark.parametrize('replace', [True, False])
def test_relay_and_unchanged_actor_control_have_explicit_valid_boundaries(trace_case, replace):
    r"""接力与不切换参照均保持相同首轨迹/终止统计，同时验证参数阶段。"""
    result = audit_palm_rotation_trace(publish(relay_trace_case(trace_case, replace=replace)))
    assert result['status'] == 'passed' and result['active_samples'] == 5


@pytest.mark.parametrize('defect', ['early_phase', 'boundary_action', 'boundary_survivor', 'formal_verdict', 'replacement_hash'])
def test_relay_rejects_timing_state_or_formal_role_corruption(trace_case, defect):
    r"""即使重新计算文件hash，也不能错配交接时刻、边界动作或诊断角色。"""
    case = relay_trace_case(trace_case)
    output, document = case
    relay = document['evaluation_identity']['protocol']['actor_relay']
    if defect == 'early_phase':
        with h5py.File(output.parent / 'evaluation.trace.h5', 'r+') as trace:
            trace['actor_checkpoint_phase'][0] = 1  # 第1个动作被错误标为新策略。
    elif defect in {'boundary_action', 'boundary_survivor'}:
        with h5py.File(relay['boundary_snapshot']['path'], 'r+') as boundary:
            if defect == 'boundary_action':
                boundary['actor_mean_after'][0, 0] = .9  # 实际执行动作仍为.2。
            else:
                boundary['active_first_trajectory'][0] = False  # 与交接时的首轨迹集合不符。
        relay['boundary_snapshot']['sha256'] = hashlib.sha256(Path(relay['boundary_snapshot']['path']).read_bytes()).hexdigest()
        seal_relay_identity(case)
    elif defect == 'formal_verdict':
        document['reliable_topology_coverage'] = {'passed_asset_count': 1}  # 混合策略不能冒充正式固定Actor。
    else:
        Path(relay['replacement_checkpoint']).write_bytes(b'wrong replacement checkpoint')
    with pytest.raises(ValueError):
        audit_palm_rotation_trace(publish(case))


@pytest.fixture
def trace_case(tmp_path):
    r"""一项资产的两个副本分别在step2掉落、step3超时；reset后的123 rad不得污染首轨迹。"""
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"integrity-only fixture; no model deserialization")
    identity = {
        "schema_version": "1.4.0",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "method_identity_digest": "a" * 64,
        "protocol": {
            "num_assets": 1,
            "replicas_per_asset": 2,
            "policy_steps": 3,
            "policy_dt_s": 0.05,
            "horizon_s": 0.15,
            "trace_stride": 1,
            "trace_rewards": True,
            "deterministic_actor_mean": True,
            "first_trajectory_only": True,
        },
    }
    identity["identity_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # 首轨迹的active包括终止步，post_state_valid排除已经自动reset的终止步。
    active = np.array([[[True, True]], [[True, True]], [[False, True]]])
    drop = np.zeros_like(active)
    drop[1, 0, 0] = True
    timeout = np.zeros_like(active)
    timeout[2, 0, 1] = True
    angles = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[123.0, 0.6]]], dtype=np.float32)
    pulses = np.zeros_like(active)
    pulses[1, 0, 0] = pulses[2, 0, 1] = True
    durations = np.array([[[0.05, 0.05]], [[0.1, 0.1]], [[0.05, 0.15]]], dtype=np.float32)
    trace_arrays = {
        "policy_step": np.arange(1, 4, dtype=np.int64),
        "active": active,
        "post_state_valid": active & ~(drop | timeout),
        "episode_duration_s": durations,
        "net_rotation_rad": angles,
        "absolute_path_rotation_rad": angles,
        "goal_success_pulse": pulses,
        "termination_object_out_of_anchor": drop,
        "termination_goal_axis_misaligned": np.zeros_like(active),
        "termination_time_out": timeout,
        "reward_terms_step": np.ones((3, 1, 2, 2), dtype=np.float32),
        "reward_step": np.full((3, 1, 2), 2.0, dtype=np.float32),
    }
    trace_arrays["reward_step"][2, 0, 0] = 99  # 无效新episode不参与首轨迹奖励重构误差。
    dense_arrays = {
        "signed_net_turns": np.array([[0.3, 0.6]]) / (2 * np.pi),
        "absolute_path_turns": np.array([[0.3, 0.6]]) / (2 * np.pi),
        "duration_s": np.array([[0.1, 0.15]]),
        "diagnostic_step_count": np.array([[2.0, 3.0]]),
        "goal_count": np.ones((1, 2)),
        "termination_drop": np.array([[True, False]]),
        "termination_axis": np.zeros((1, 2), dtype=bool),
        "termination_timeout": np.array([[False, True]]),
    }
    trace_path, dense_path = tmp_path / "evaluation.trace.h5", tmp_path / "evaluation.h5"
    write_selected_trajectories_hdf5(
        trace_path,
        arrays=trace_arrays,
        metadata={**identity, "axes": "time,asset,replica,feature", "reward_term_names": ["rotation", "failure"]},
    )
    write_selected_trajectories_hdf5(dense_path, arrays=dense_arrays, metadata=identity)
    document = {
        "evaluation_identity": identity,
        "checkpoint": str(checkpoint),
        "trajectory_hdf5": str(dense_path),
        "step_trace": {"path": str(trace_path), "samples": 3},
    }
    output = tmp_path / "evaluation.json"
    return output, document


def publish(case):
    r"""更新数组文件的真实SHA；使负例触发内容合同，而非全部只触发hash检查。"""
    output, document = case
    document["trajectory_hdf5_sha256"] = hashlib.sha256((output.parent / "evaluation.h5").read_bytes()).hexdigest()
    document["step_trace"]["sha256"] = hashlib.sha256((output.parent / "evaluation.trace.h5").read_bytes()).hexdigest()
    output.write_text(json.dumps(document))
    return output


@pytest.mark.parametrize("block_steps", [1, 2, 32])
def test_first_trajectory_survives_asynchronous_reset_and_chunking(trace_case, block_steps):
    r"""同一有效样本总体不随I/O分块改变，terminal只计一次，不接入第二回合。"""
    report = audit_palm_rotation_trace(publish(trace_case), block_steps=block_steps)
    assert report["status"] == "passed"
    assert report["active_samples"] == 5
    assert report["post_valid_samples"] == 3
    assert report["reward_reconstruction_max_abs_error"] == 0


@pytest.mark.parametrize("defect", ["reactivation", "post_mask", "reward", "nonfinite", "unfinished", "snapshot"])
def test_rehashed_but_semantically_invalid_trace_is_rejected(trace_case, defect):
    r"""哈希一致不等于物理/生命周期一致；逐一破坏核心不变量。"""
    output, _ = trace_case
    with h5py.File(output.parent / "evaluation.trace.h5", "r+") as trace:
        arrays = {name: value for name, value in trace.items() if isinstance(value, h5py.Dataset)}
        if defect == "reactivation":
            arrays["active"][2, 0, 0] = True
        elif defect == "post_mask":
            arrays["post_state_valid"][1, 0, 0] = True
        elif defect == "reward":
            arrays["reward_step"][0, 0, 0] = 2.01
        elif defect == "nonfinite":
            arrays["net_rotation_rad"][0, 0, 0] = np.nan
        elif defect == "unfinished":
            arrays["termination_time_out"][2, 0, 1] = False
            arrays["post_state_valid"][2, 0, 1] = True
        elif defect == "snapshot":
            arrays["net_rotation_rad"][1, 0, 0] = 0.1
    with pytest.raises(ValueError):
        audit_palm_rotation_trace(publish(trace_case), block_steps=1)


def test_file_replacement_without_hash_update_is_rejected(trace_case):
    r"""拒绝路径相同但字节已变的trace，防止不同checkpoint的文件被误配。"""
    output = publish(trace_case)
    with h5py.File(output.parent / "evaluation.trace.h5", "r+") as trace:
        reward = trace["reward_step"]
        assert isinstance(reward, h5py.Dataset)
        reward[0, 0, 0] = 0
    with pytest.raises(ValueError, match="SHA"):
        audit_palm_rotation_trace(output)


def sample_trace_case(case, *, seed=17):
    r"""沿用异步终止夹具，增加可重建的随机动作、latent、sigma与ghost掩码。"""
    output, document = case  # 同一3步、1资产、2副本的首轨迹合同
    identity = document["evaluation_identity"]  # 更改模式后重新发布合法身份
    identity["protocol"].update({
        "deterministic_actor_mean": False,  # 随机诊断不能声称均值执行
        "reliable_topology_coverage_protocol_matched": False,  # 正式一圈门不适用
        "scale_ready_protocol_matched": False,  # scale-ladder同样不适用
        "action_selection": {"mode": "sample", "seed": seed, "generator_device": "cpu",
                             "distribution": "masked-tanh-normal", "latent_action_epsilon": 1e-6,
                             "actor_parameter_check": "bitwise-equal"},  # 显式采样与冻结声明
    })
    identity["action_selection_source_sha256"] = "b" * 64  # 夹具只验证记录合同，真实采样在独立测试核对
    identity.pop("identity_digest")  # 对新的协议重新计算digest
    identity["identity_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()  # 不以旧均值identity掩盖sample语义
    document.update(artifact_type="anymani.palm_rotation_stochastic_diagnostic",
                    reliable_topology_coverage=None, scale_ladder=None, cohort=None)  # 均值正式结论为空
    for suffix in ("evaluation.h5", "evaluation.trace.h5"):
        with h5py.File(output.parent / suffix, "r+") as f:
            metadata = json.loads(f.attrs["metadata_json"])  # 保留trace axes和reward名称
            metadata.update(identity)  # 两份HDF5与JSON绑定同一动作模式
            f.attrs["metadata_json"] = json.dumps(metadata)
            if suffix.endswith("trace.h5"):
                shape = (3, 1, 2, 16)  # [T,A,R,J] canonical动作轴
                mask = np.ones(shape, dtype=bool)  # 每副本15个active关节
                mask[..., -1] = False  # 末槽ghost始终不执行动作
                latent = np.full(shape, .3, dtype=np.float32)  # 分布事实夹具，不检验小样本正态性
                for name, values in {
                    "action": np.tanh(latent) * mask,  # 记录执行动作，独立核对tanh/mask关系
                    "policy_action_mean": np.full(shape, .1, dtype=np.float32) * mask,
                    "policy_latent_sigma": np.full(shape, .2, dtype=np.float32),
                    "policy_latent_sample": latent,
                    "policy_joint_valid": mask,
                }.items():
                    f.create_dataset(name, data=values)  # 所有附加字段保留时间与首轨迹轴
    return publish(case)  # 字节已变，发布新的trace/terminal SHA


def test_sample_trace_requires_explicit_action_mode(trace_case):
    r"""默认审计仍要求均值；明确sample后才审计随机首轨迹和动作重构。"""
    output = sample_trace_case(trace_case)  # 带真实sample声明的文件
    with pytest.raises(ValueError, match="mode"):
        audit_palm_rotation_trace(output)  # 不自动扩大默认均值合同
    report = audit_palm_rotation_trace(output, action_mode="sample")
    assert report["status"] == "passed" and report["action_mode"] == "sample"  # 不是正式能力通过
    assert report["sampling_reconstruction_max_abs_error"] < 1e-6  # 能从latent逐步还原执行动作
    assert report["standard_normal_residual"]["count"] == 75  # 5个active样本×15关节，排除新回合和ghost


@pytest.mark.parametrize("defect", ["ghost", "latent", "formal_gate"])
def test_sample_trace_rejects_rehashed_action_or_formal_gate_defects(trace_case, defect):
    r"""字节一致仍须拒绝ghost动作、错误tanh关系或不适用的正式覆盖结论。"""
    output = sample_trace_case(trace_case)  # 合法随机首轨迹参照
    if defect == "formal_gate":
        trace_case[1]["reliable_topology_coverage"] = {"passed_asset_count": 1}  # 随机诊断冒用正式门
    else:
        with h5py.File(output.parent / "evaluation.trace.h5", "r+") as f:
            if defect == "ghost":
                f["action"][0, 0, 0, -1] = .1  # 无效关节产生非零执行动作
            else:
                f["policy_latent_sample"][0, 0, 0, 0] += .5  # 改变latent但未同步执行动作
    with pytest.raises(ValueError):
        audit_palm_rotation_trace(publish(trace_case), action_mode="sample")  # 使用新SHA排除纯文件损坏触发


def test_sample_trace_rejects_missing_seed(trace_case):
    r"""随机模式必须声明动作seed，不能用一个未知全局随机状态代表可复盘对照。"""
    output = sample_trace_case(trace_case, seed=None)  # 所有文件身份一致但采样声明缺失
    with pytest.raises(ValueError, match="seed"):
        audit_palm_rotation_trace(output, action_mode="sample")
