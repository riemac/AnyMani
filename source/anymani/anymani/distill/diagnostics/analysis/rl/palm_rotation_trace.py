r"""固定首轨迹的纯文件完整性审计，不构造模型或运行物理。

trace轴为[T,A,R,...]，terminal摘要轴为[A,R]。只允许trace_stride=1，
因为稀疏采样可能漏掉真正的终止步，不能重建完整首轨迹。
active包含终止步；post_state_valid排除该步已自动reset的关节/传感器状态。
每条轨迹恰好结束一次，后续新episode即使仍被模拟，也不能重新进入统计总体。

本模块检查文件链接、身份、声明协议、有限性、掩码和逐步到终止摘要的一致性。
它不定义停滞/反转分箱，不选择checkpoint，也不把完整性通过称为旋转能力通过。
默认合同要求确定性均值。显式action_mode=sample核对随机诊断的模式/seed、关闭的正式门及latent到动作的映射。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

# 该字典描述 runtime.phase_clock_contract(P) 的固定语义字段；P 属于被绑定身份，analysis 保持
# artifact-only，因而不 import runtime、model 或 teacher 来生成/修正证据。
PHASE_CLOCK_CONTRACT: dict[str, Any] = {
    "source": "physical-episode-length-buf",
    "encoding": ["sin", "cos"],
    "encoding_dtype": "float32",
    "reset": "physical-episode-counter-zero-before-returned-observation",
    "transport_key": "phase_clock",
    "actor_adapter": "zero-linear2to128-after-contextual-joints-before-existing-head-norm",
    "critic_adapter": "zero-linear2to896-after-readout-before-existing-value-norm",
}
PHASE_CLOCK_SEMANTICS = "sin/cos from physical episode counter before applied action"
PHASE_CLOCK_MAX_ABS_ERROR = 1.0e-6  # FP32 angle/sin/cos 与 artifact 保存值的最大允许误差。


def _sha256(path: Path) -> str:
    r"""流式核对真实文件字节，避免为哈希把大型trace全部读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_clock_contract(value: Any, *, where: str) -> dict[str, Any]:
    r"""验证 trace/checkpoint 可读的 phase contract，并规范化其数值精度边界。

    ``period_policy_steps`` 从被绑定的 metadata 读取，只要求整数 ``P>=2``；其 increment
    必须满足 ``2*pi/P``。source、encoding 顺序、reset 语义和两个零初始化 adapter 名称逐值吻合。
    """

    if not isinstance(value, dict):
        raise ValueError(f"{where} phase_clock contract must be a JSON object")
    contract = dict(value)
    expected_keys = set(PHASE_CLOCK_CONTRACT) | {"period_policy_steps", "increment_rad_per_policy_step"}
    missing = expected_keys - set(contract)
    if missing:
        raise ValueError(f"{where} phase_clock contract missing keys: {sorted(missing)}")
    unexpected = set(contract) - expected_keys
    if unexpected:
        raise ValueError(f"{where} phase_clock contract has unexpected keys: {sorted(unexpected)}")
    period = contract["period_policy_steps"]
    if type(period) is not int or period < 2:
        raise ValueError(f"{where} phase_clock period_policy_steps must be an integer >= 2")
    increment = contract["increment_rad_per_policy_step"]
    if type(increment) not in (int, float) or not math.isfinite(float(increment)):
        raise ValueError(f"{where} phase_clock increment must be finite")
    if not math.isclose(
        float(increment), 2.0 * math.pi / period, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{where} phase_clock increment disagrees with 2*pi/{period}")
    for key in PHASE_CLOCK_CONTRACT:
        if contract[key] != PHASE_CLOCK_CONTRACT[key]:
            raise ValueError(f"{where} phase_clock contract field {key!r} disagrees with runtime contract")
    return contract


def audit_palm_rotation_trace(evaluation: Path, *, block_steps: int = 32, action_mode: str = "mean") -> dict[str, Any]:
    r"""验证一个评价JSON及其checkpoint、terminal HDF5和dense trace的闭合关系。

    对环境(a,r)，令m_t为首轨迹active，d_t为三类终止的并集，则
    $m_1=1,\ m_{t+1}=m_t(1-d_t)$，且post_state_valid恰为$m_t(1-d_t)$。
    奖励仅在m_t=1上检查$|r_t-\sum_k r_{t,k}|\le10^{-4}$，与producer一致。
    净圈等终止量取最后一个有效step，不取整个模拟器窗口的最后一行。

    Args:
        evaluation: 已发布的固定评价JSON；其中路径与SHA是证据绑定，不猜测替代文件。
        block_steps: 一次读取的时间步数，默认32；只影响I/O内存，不改变统计总体。
        action_mode: 默认mean只接受均值首轨迹；sample必须与文件中的随机诊断声明逐值匹配。

    Returns:
        可JSON化的完整性证据；protocol原样保留，资产声明分母和能力门由调用方管理。

    Raises:
        ValueError: 文件身份、轴、数值、生命周期或终止摘要不一致。
    """

    def require(condition: bool, message: str) -> None:
        r"""失败必须带明确的证据边界，不能继续发布一个看似通过的审计。"""
        if not condition:
            raise ValueError(message)

    require(block_steps > 0, "block_steps must be positive")
    require(action_mode in {"mean", "sample"}, "action mode must be mean or sample")  # 调用方明确期望的证据类型
    evaluation = Path(evaluation).resolve()  # 报告绑定本次实际读取的JSON。
    document = json.loads(evaluation.read_text())
    identity = document["evaluation_identity"]
    payload = {key: value for key, value in identity.items() if key != "identity_digest"}
    expected_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(identity["identity_digest"] == expected_digest, "evaluation identity digest mismatch")
    protocol = identity["protocol"]  # 只核验声明与数组是否一致，不替调用方改变R16或资产分母。
    relay = protocol.get("actor_relay")  # 旧文件没有接力字段，仍按单一冻结策略解释。
    assets, replicas = int(protocol["num_assets"]), int(protocol["replicas_per_asset"])
    horizon, dt = int(protocol["policy_steps"]), float(protocol["policy_dt_s"])
    require(
        assets > 0 and replicas > 0 and horizon > 0 and np.isfinite(dt) and dt > 0, "invalid protocol dimensions/dt"
    )
    require(np.isclose(protocol["horizon_s"], horizon * dt), "horizon disagrees with policy_steps * dt")
    require(protocol["trace_stride"] == 1 and protocol["trace_rewards"], "audit requires dense stride-1 reward trace")
    require(protocol["first_trajectory_only"], "not fixed first trajectories")  # 两种动作模式共用首轨迹生命周期
    if protocol.get("evaluation_role") == "diagnostic":
        require(protocol.get("reliable_topology_coverage_protocol_matched") is False
                and protocol.get("scale_ready_protocol_matched") is False,
                "diagnostic role cannot match formal capability gates")
        require(all(document.get(key) is None for key in ("reliable_topology_coverage", "scale_ladder", "cohort")),
                "diagnostic role contains a formal capability verdict")
    if relay is not None:
        require(isinstance(relay, dict) and protocol.get("evaluation_role") == "diagnostic" and action_mode == "mean",
                "Actor relay requires an explicit deterministic diagnostic role")
        require(document.get("artifact_type") == "anymani.palm_rotation_actor_relay_diagnostic", "Actor relay artifact type mismatch")
        boundary = relay.get("boundary_step")  # k个旧动作完成后，从第k+1个动作起交接。
        require(type(boundary) is int and 0 < boundary < horizon, "invalid Actor relay boundary")
        boundary = int(relay["boundary_step"])  # 上述合同确认后使用明确的整数时刻。
        require(np.isclose(relay.get("boundary_time_s", -1), boundary * dt), "Actor relay time mismatch")
        require(relay.get("initial_checkpoint_sha256") == identity["checkpoint_sha256"], "relay initial checkpoint mismatch")
        requested = relay.get("replacement_requested")
        reached = relay.get("boundary_reached")
        require(type(requested) is bool and type(reached) is bool, "relay event flags must be boolean")
        require(relay.get("performed") is (requested and reached), "relay performed flag disagrees with event")
        require(relay.get("final_actor_check") == "bitwise-equal", "relay final Actor freeze check missing")
        require(bool(identity.get("actor_switch_source_sha256")), "relay implementation identity missing")
        require(bool(relay.get("replacement_checkpoint")) is requested, "relay replacement path mismatch")
        if requested:
            require(_sha256(Path(relay["replacement_checkpoint"])) == relay.get("replacement_checkpoint_sha256"),
                    "relay replacement checkpoint SHA mismatch")
            require(relay.get("first_replacement_action_step") == boundary + 1, "relay action boundary is off by one")
        else:
            require(relay.get("first_replacement_action_step") is None, "control condition claims replacement actions")
        if reached:
            for key in ("state_continuity_check", "torch_rng_check", "prefix_actor_check", "replacement_actor_check"):
                require(relay.get(key) == "bitwise-equal", f"relay {key} missing")
            snapshot = relay.get("boundary_snapshot")
            require(isinstance(snapshot, dict), "relay boundary snapshot missing")
            snapshot = cast(dict[str, Any], snapshot)  # 文件声明已确认是具名路径/hash映射。
            require(_sha256(Path(snapshot["path"])) == snapshot.get("sha256"), "relay snapshot SHA mismatch")
        else:
            require(relay.get("boundary_snapshot") is None, "unreached relay claims a boundary snapshot")
    selection = protocol.get("action_selection")  # 老均值文件缺少该字段，仍按显式mean布尔量解释
    require(protocol["deterministic_actor_mean"] is (action_mode == "mean"), "action mode does not match declared mean flag")
    if action_mode == "mean":
        require(selection is None or (isinstance(selection, dict) and selection.get("mode") == "mean" and selection.get("seed") is None),
                "mean action mode has inconsistent sampling metadata")  # 默认不消费动作seed
    else:
        require(isinstance(selection, dict) and selection.get("mode") == "sample", "sample action mode lacks metadata")
        seed = selection.get("seed")  # 独立动作流的来源
        require(type(seed) is int and 0 <= seed < 2**63, "sample action mode requires an explicit valid seed")
        require(selection.get("distribution") == "masked-tanh-normal", "unsupported sample distribution")
        require(selection.get("latent_action_epsilon") == 1e-6, "unsupported latent action boundary")  # 当前FP32生产合同
        require(selection.get("actor_parameter_check") == "bitwise-equal", "sample actor freeze check is missing")
        require(bool(selection.get("generator_device")) and bool(identity.get("action_selection_source_sha256")),
                "sample generator/source identity is missing")  # 随机流与实现来源都须声明
        require(document.get("artifact_type") == "anymani.palm_rotation_stochastic_diagnostic", "sample artifact type mismatch")
        require(protocol.get("reliable_topology_coverage_protocol_matched") is False
                and protocol.get("scale_ready_protocol_matched") is False,
                "sample diagnostic cannot match formal capability gates")  # 模式与正式协议不能混用
        require(all(document.get(key) is None for key in ("reliable_topology_coverage", "scale_ladder", "cohort")),
                "sample diagnostic contains a formal capability verdict")  # 拒绝伪造的正式覆盖结果

    # 先闭合真实字节与JSON声明，再检查内部语义；仅路径相同不能证明是同一checkpoint。
    trace_path = Path(document["step_trace"]["path"])
    trajectory_path = Path(document["trajectory_hdf5"])
    for path, expected in (
        (Path(document["checkpoint"]), identity["checkpoint_sha256"]),
        (trace_path, document["step_trace"]["sha256"]),
        (trajectory_path, document["trajectory_hdf5_sha256"]),
    ):
        require(_sha256(path) == expected, f"SHA mismatch: {path}")

    # 只缓存必要的标量序列，其余高维动作/传感器字段逐块检查后释放。
    terminal_fields = {
        "termination_object_out_of_anchor": "termination_drop",
        "termination_goal_axis_misaligned": "termination_axis",
        "termination_time_out": "termination_timeout",
    }
    final_fields = {
        "episode_duration_s": ("duration_s", 1.0),
        "net_rotation_rad": ("signed_net_turns", 1.0 / (2 * np.pi)),
        "absolute_path_rotation_rad": ("absolute_path_turns", 1.0 / (2 * np.pi)),
    }  # trace保存rad，终止摘要保存圈；duration始终为秒。
    scalar_names = {"active", "post_state_valid", "goal_success_pulse", "reward_step", *terminal_fields, *final_fields}
    required = scalar_names | {"reward_terms_step", "policy_step"}
    sampled_fields = {"action", "policy_action_mean", "policy_latent_sigma", "policy_latent_sample", "policy_joint_valid"}
    if action_mode == "sample":
        required |= sampled_fields  # 随机动作必须保存其真实分布参数与已执行latent样本
    if relay is not None:
        required.add("actor_checkpoint_phase")  # 实际每步所用的参数来源。
    shape = (assets, replicas)  # 每个cell是一条固定首轨迹，不跨资产或replica求均值。
    alive = np.ones(shape, dtype=bool)
    counts = np.zeros(shape, dtype=np.int64)
    goals = np.zeros(shape, dtype=np.int64)
    advances = np.zeros(shape, dtype=np.int64)  # 新任务的角度推进可以多于位置合格奖金次数。
    final = {name: np.zeros(shape) for name in final_fields}
    terminations = {name: np.zeros(shape, dtype=bool) for name in terminal_fields}
    post_count, reward_error = 0, 0.0
    sample_error, residual_count, residual_sum, residual_square_sum = 0.0, 0, 0.0, 0.0  # 动作重构与标准化噪声事实

    with h5py.File(trace_path, "r") as trace_file, h5py.File(trajectory_path, "r") as trajectory_file:
        # metadata中的扩展描述可不同，但共同的评价身份每个字段必须与JSON逐值相同。
        metadata_by_label: dict[str, dict[str, Any]] = {}
        for label, stream in (("trace", trace_file), ("trajectory", trajectory_file)):
            require(stream.attrs["schema_version"] == "1.0.0", f"unsupported {label} HDF5 schema")
            metadata = json.loads(str(stream.attrs["metadata_json"]))
            require(
                all(metadata.get(key) == value for key, value in identity.items()),
                f"{label} metadata identity mismatch",
            )
            require(isinstance(metadata, dict), f"{label} metadata must be a JSON object")
            metadata_by_label[label] = cast(dict[str, Any], metadata)
        trace_meta = json.loads(str(trace_file.attrs["metadata_json"]))
        # 当前格式只含根层Dataset；不把额外group或named datatype当成已审计数组。
        trace = {name: value for name, value in trace_file.items() if isinstance(value, h5py.Dataset)}
        trajectory = {name: value for name, value in trajectory_file.items() if isinstance(value, h5py.Dataset)}
        phase_metadata_keys = {"phase_clock", "pre_phase_clock_semantics"}
        phase_metadata_labels = {
            label for label, metadata in metadata_by_label.items() if phase_metadata_keys & set(metadata)
        }
        phase_dataset_present = "pre_phase_clock" in trace
        phase_declared = bool(phase_metadata_labels or phase_dataset_present)
        phase_contract: dict[str, Any] | None = None
        phase_semantics: str | None = None
        if phase_declared:
            require(phase_dataset_present, "phase metadata requires the pre_phase_clock dataset")
            require(
                phase_metadata_keys <= set(metadata_by_label["trace"]),
                "pre_phase_clock requires trace phase_clock and pre_phase_clock_semantics metadata",
            )
            phase_contract = _phase_clock_contract(
                metadata_by_label["trace"]["phase_clock"], where="trace metadata"
            )
            phase_semantics = metadata_by_label["trace"]["pre_phase_clock_semantics"]
            require(
                phase_semantics == PHASE_CLOCK_SEMANTICS,
                "trace pre_phase_clock_semantics disagrees with the phase contract",
            )
            # 若terminal文件也带扩展metadata，必须与trace逐字段相同；producer当前只在trace声明它。
            for label in phase_metadata_labels:
                metadata = metadata_by_label[label]
                require(
                    phase_metadata_keys <= set(metadata),
                    f"{label} phase metadata must include phase_clock and pre_phase_clock_semantics",
                )
                _phase_clock_contract(metadata["phase_clock"], where=f"{label} metadata")
                require(
                    metadata["phase_clock"] == phase_contract
                    and metadata["pre_phase_clock_semantics"] == phase_semantics,
                    f"{label} phase metadata disagrees with trace metadata",
                )
            required.add("pre_phase_clock")
        if "orientation_goal_count" in trajectory:
            required.add("goal_advance_pulse")
        require(
            len(trace) == len(trace_file) and len(trajectory) == len(trajectory_file),
            "HDF5 must contain root datasets only",
        )
        require(trace_meta["axes"] == "time,asset,replica,feature", "unsupported trace axes")
        require(required <= set(trace), f"missing trace fields: {required - set(trace)}")
        steps = np.asarray(trace["policy_step"][:])
        length = steps.size  # 所有副本提前失败时T可小于名义horizon，不能因此改变失败分母。
        require(steps.ndim == 1 and steps.dtype.kind in "iu", "policy_step must be an integer [T] axis")
        require(
            0 < length <= horizon and length == document["step_trace"]["samples"],
            "trace length disagrees with protocol",
        )
        require(np.array_equal(steps, np.arange(1, length + 1)), "policy_step is not contiguous from one")
        phase_max_abs_error: float | None = None
        if phase_contract is not None:
            # Producer先在整数policy-step域取模，再以FP32角度计算sin/cos，避免长轨迹取模漂移。
            period = int(phase_contract["period_policy_steps"])
            require(
                trace["pre_phase_clock"].shape == (length, *shape, 2),
                f"pre_phase_clock must have shape {(length, *shape, 2)}, got {trace['pre_phase_clock'].shape}",
            )
            require(
                trace["pre_phase_clock"].dtype == np.dtype(np.float32),
                f"pre_phase_clock must use float32, got {trace['pre_phase_clock'].dtype}",
            )
            require(
                trace["active"].shape == (length, *shape),
                f"active must have shape {(length, *shape)} before phase validation",
            )
            phase_index = np.remainder(steps.astype(np.int64) - 1, period).astype(np.float32)
            angle = phase_index * np.float32(2.0 * math.pi / period)
            expected_phase = np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)
            recorded_phase = np.asarray(trace["pre_phase_clock"][:])
            phase_error = np.abs(
                recorded_phase.astype(np.float64) - np.broadcast_to(expected_phase[:, None, None, :], recorded_phase.shape)
            )
            active_values = np.asarray(trace["active"][:], dtype=bool)
            valid_phase_error = phase_error[active_values]
            phase_max_abs_error = float(valid_phase_error.max()) if valid_phase_error.size else 0.0
            require(
                phase_max_abs_error <= PHASE_CLOCK_MAX_ABS_ERROR,
                f"pre_phase_clock disagrees with sin/cos contract: max_abs_error={phase_max_abs_error:.9g}",
            )
        if relay is not None:
            boundary = int(relay["boundary_step"])
            require(relay["boundary_reached"] is (length > boundary), "relay reach event disagrees with recorded steps")
            phase = np.asarray(trace["actor_checkpoint_phase"][:])
            expected_phase = ((steps > boundary) & relay["performed"]).astype(np.int8)
            require(phase.shape == (length, *shape) and phase.dtype.kind in "iu", "relay phase must use integer [T,A,R]")
            require(np.array_equal(phase, np.broadcast_to(expected_phase[:, None, None], phase.shape)),
                    "relay parameter phase disagrees with declared action boundary")
            if relay["boundary_reached"]:
                with h5py.File(relay["boundary_snapshot"]["path"], "r") as boundary_file:
                    boundary_meta = json.loads(str(boundary_file.attrs["metadata_json"]))
                    require(boundary_meta.get("method_identity_digest") == identity["method_identity_digest"]
                            and boundary_meta.get("boundary_step") == boundary,
                            "boundary snapshot method/time mismatch")
                    require(boundary_meta.get("initial_checkpoint_sha256") == identity["checkpoint_sha256"]
                            and boundary_meta.get("replacement_checkpoint_sha256") == relay.get("replacement_checkpoint_sha256"),
                            "boundary snapshot checkpoint identities mismatch")
                    boundary_required = {"actor_mean_before", "actor_mean_after", "active_first_trajectory",
                                         "obs__actor_jnt_current", "obs__actor_jnt_history", "obs__geometry_tokens",
                                         "controller__current_targets", "physics__joint_position", "physics__object_root_state"}
                    boundary_values = {name: np.asarray(value[:]) for name, value in boundary_file.items()
                                       if isinstance(value, h5py.Dataset)}  # 单一边界，读取全部具名输入。
                    require(len(boundary_values) == len(boundary_file), "boundary snapshot must contain root datasets only")
                    require(boundary_required <= set(boundary_values), "boundary snapshot lacks continuous state inputs")
                    for key, values in boundary_values.items():
                        require(values.ndim >= 1 and values.shape[0] == assets * replicas,
                                f"boundary snapshot {key} has invalid environment axis")
                        require(values.dtype.kind in "biuf", f"boundary snapshot {key} must be numeric or boolean")
                        require(bool(np.isfinite(values).all()), f"nonfinite boundary snapshot {key}")
                    boundary_alive = boundary_values["active_first_trajectory"].reshape(replicas, assets).T
                    require(np.array_equal(boundary_alive, trace["active"][boundary]), "boundary survivor set mismatch")
                    before_mean, after_mean = boundary_values["actor_mean_before"], boundary_values["actor_mean_after"]
                    require(before_mean.shape == after_mean.shape == (assets * replicas, 16), "boundary mean shape mismatch")
                    executed = after_mean.reshape(replicas, assets, 16).transpose(1, 0, 2)
                    require(np.array_equal(executed, trace["action"][boundary]), "first relay action differs from boundary mean")
                    if not relay["replacement_requested"]:
                        require(np.array_equal(before_mean, after_mean), "unchanged-Actor control changed boundary actions")
        for name, dataset in trace.items():
            if name != "policy_step":
                require(dataset.shape[:3] == (length, *shape), f"{name} must start with [T,A,R]")
                require(dataset.dtype.kind in "biuf", f"{name} has nonnumeric dtype")
        for name in scalar_names:
            require(trace[name].shape == (length, *shape), f"{name} must have scalar [T,A,R] shape")
        for name in ("active", "post_state_valid", "goal_success_pulse", *terminal_fields):
            require(trace[name].dtype.kind == "b", f"{name} must preserve boolean membership")
        if action_mode == "sample":
            for name in sampled_fields:
                require(trace[name].shape == (length, *shape, 16), f"{name} must use [T,A,R,16] action axes")
            require(trace["policy_joint_valid"].dtype.kind == "b", "sample joint mask must preserve boolean membership")
        term_count = len(trace_meta["reward_term_names"])
        require(
            term_count > 0 and trace["reward_terms_step"].shape == (length, *shape, term_count),
            "reward term axis mismatch",
        )

        # 时间分块不会截断轨迹：alive、累计量与最后有效快照跨块保留。
        for start in range(0, length, block_steps):
            stop = min(start + block_steps, length)
            block = {}
            for name, dataset in trace.items():
                if name == "policy_step":
                    continue
                values = dataset[start:stop]
                require(bool(np.isfinite(values).all()), f"nonfinite trace field {name}, steps {start + 1}:{stop}")
                if name in required:
                    block[name] = values
            active = block["active"]
            error = np.abs(block["reward_terms_step"].sum(axis=-1) - block["reward_step"])
            if active.any():
                reward_error = max(reward_error, float(error[active].max()))  # 只统计首轨迹，不纳入自动reset后的奖励。
            require(reward_error <= 1e-4, f"reward reconstruction error {reward_error} exceeds 1e-4")
            if action_mode == "sample":
                # 动作从记录的latent直接重建；白化噪声仅作描述，不用有限样本正态性宣告实现等价。
                joint_mask = block["policy_joint_valid"]  # [block,A,R,16]，与动作同轴
                action = block["action"]  # 实际交给控制器的有界动作
                center, sigma = block["policy_action_mean"], block["policy_latent_sigma"]  # FP32分布参数
                latent = block["policy_latent_sample"]  # 当前步已抽取的样本
                require(bool((sigma > 0).all()), "sample sigma must be positive")
                require(bool((np.abs(center) <= 1 + 1e-6).all()), "sample center escaped bounded action space")
                require(bool((np.abs(action) <= 1).all()) and bool((action[~joint_mask] == 0).all()),
                        "sample action violates bounds or ghost mask")
                observed = active[..., None] & joint_mask  # 只在首轨迹活动关节统计噪声
                if active.any():
                    reconstructed_action = np.tanh(latent) * joint_mask  # a=tanh(z)*M，允许跨设备FP32舍入
                    sample_error = max(sample_error, float(np.abs(reconstructed_action - action)[active].max()))
                require(sample_error <= 1e-6, "sample latent does not reconstruct the executed action")
                if observed.any():
                    # FP32夹紧边界与生产atanh一致，不能将±1附近的夹紧升级为FP64后改变latent location。
                    location = np.arctanh(np.clip(center.astype(np.float32), np.float32(-1 + 1e-6), np.float32(1 - 1e-6)))
                    residual = ((latent.astype(np.float64) - location) / sigma)[observed]  # epsilon=(z-m)/sigma
                    residual_count += int(residual.size)  # 累计有效joint-coordinate-steps
                    residual_sum += float(residual.sum())  # FP64累计标准正态残差
                    residual_square_sum += float(np.square(residual).sum())  # 二阶原点矩
            for offset in range(stop - start):
                membership = active[offset]  # m_t，包含本步结束的副本。
                require(
                    np.array_equal(membership, alive),
                    f"first-trajectory membership mismatch at step {start + offset + 1}",
                )
                done = np.zeros(shape, dtype=bool)
                for name in terminal_fields:
                    flags = block[name][offset]
                    done |= flags
                    terminations[name] |= membership & flags
                alive = membership & ~done  # 下一步membership，同时也是本步post-state的有效集合。
                require(
                    np.array_equal(block["post_state_valid"][offset], alive), "post_state_valid includes reset state"
                )
                post_count += int(alive.sum())
                counts += membership
                goals += membership & block["goal_success_pulse"][offset]
                if "goal_advance_pulse" in block:
                    advances += membership & block["goal_advance_pulse"][offset]
                for name in final:
                    final[name][membership] = block[name][offset][membership]
        require(not alive.any(), "unfinished first trajectories at trace end")

        # 所有terminal数组均须有限且按[A,R]对齐；从trace独立重建可观测的终止量。
        for name, dataset in trajectory.items():
            require(dataset.shape == shape and dataset.dtype.kind in "biuf", f"invalid trajectory field {name}")
            require(bool(np.isfinite(dataset[:]).all()), f"nonfinite trajectory field {name}")
        reconstructed = {"diagnostic_step_count": counts, "goal_count": goals}
        if "orientation_goal_count" in trajectory:
            reconstructed["orientation_goal_count"] = advances
        reconstructed.update({target: final[name] * factor for name, (target, factor) in final_fields.items()})
        reconstructed.update({target: terminations[name] for name, target in terminal_fields.items()})
        errors = {}
        for name, values in reconstructed.items():
            require(name in trajectory, f"missing trajectory field {name}")
            recorded = np.asarray(trajectory[name][:])
            errors[name] = float(np.max(np.abs(recorded.astype(np.float64) - values)))
            require(np.allclose(recorded, values, atol=1e-5, rtol=1e-6), f"terminal snapshot mismatch: {name}")
        require(
            np.allclose(final["episode_duration_s"], counts * dt, atol=1e-5, rtol=1e-6),
            "duration disagrees with active steps",
        )
        require(
            bool((final["absolute_path_rotation_rad"] + 1e-5 >= np.abs(final["net_rotation_rad"])).all()),
            "path shorter than net rotation",
        )

    return {
        "artifact_type": "anymani.palm_rotation_trace_integrity",
        "schema_version": "1.0.0",
        "status": "passed",
        "evaluation": str(evaluation),
        "evaluation_sha256": _sha256(evaluation),
        "auditor_source_sha256": _sha256(Path(__file__)),
        "evaluation_identity_digest": identity["identity_digest"],
        "method_identity_digest": identity["method_identity_digest"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "trace_sha256": document["step_trace"]["sha256"],
        "trajectory_sha256": document["trajectory_hdf5_sha256"],
        "protocol": protocol,
        "trace_steps": length,
        "active_samples": int(counts.sum()),
        "post_valid_samples": post_count,
        "reward_reconstruction_max_abs_error": reward_error,
        "terminal_snapshot_max_abs_errors": errors,
        "block_steps": block_steps,
        "action_mode": action_mode,  # 默认审计与随机诊断结果始终可区分
        "phase_clock": phase_contract,
        "pre_phase_clock_semantics": phase_semantics,
        "phase_clock_max_abs_error": phase_max_abs_error,
        "sampling_reconstruction_max_abs_error": sample_error if action_mode == "sample" else None,
        "standard_normal_residual": (
            {"count": residual_count, "mean": residual_sum / residual_count,
             "std": float(np.sqrt(max(0.0, residual_square_sum / residual_count - (residual_sum / residual_count) ** 2)))}
            if residual_count else None
        ),  # 描述性采样事实；与参数/统计冻结和物理能力分别判断
        "scope": "File/identity/finite/mask/reward/terminal integrity only; not capability or occupancy classification.",
    }


def main() -> None:
    r"""只读输入，审计成功后输出JSON；显式输出文件不得覆盖既有证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--block_steps", type=int, default=32)
    parser.add_argument("--action_mode", choices=("mean", "sample"), default="mean")  # 显式选择期望的动作合同
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_palm_rotation_trace(args.evaluation, block_steps=args.block_steps, action_mode=args.action_mode)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
