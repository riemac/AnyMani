r"""冻结族教师的旁路动作监督采集，供三组共享学生重复使用。

collector通过显式callback消费已有evaluator的动作前观察与pre-reset终止统计。
确定性模式原样返回教师中心；随机模式按该教师原来的masked tanh-Normal执行，
标签仍是动作前同一状态的教师中心。任何FK标签与物体质量指标都不反馈给教师动作。

动态Actor输入在20 Hz逐步保存，History30逐步核对重建；FP32 Z、动作中心与FK
每4步保存一次作为初始监督频率，不改变物理控制时间步。所有原始失败资产保留。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1
from anymani.distill.il.family_dataset import FamilyTrajectoryWriter
from anymani.distill.representations.sources.joint_frames import build_joint_kinematics_bank

ACTOR_ABI = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": 30,
    "joint_count": 16,
    "owner_count": 21,
    "geometry_width": 128,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": 15,
}  # 共享学生基础接口；静态FK信息对三个条件相同。


def _sha256(path: Path) -> str:
    """按真实文件字节散列，不用可复用路径冒充权重/数据身份。"""
    digest = hashlib.sha256()  # 大文件分块读取。
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: torch.Tensor) -> np.ndarray:
    """在下一次物理step前形成独立CPU副本，防止原地更新改变已采事实。"""
    return value.detach().cpu().numpy().copy()  # CPU输入也复制，保持相同生命周期语义。


class FrozenFamilyCollection:
    r"""只读教师状态并流式保存输入/监督；显式记录随机执行与确定性标签的区别。

    start在原evaluator完成全部模型/物理身份检查并reset后调用；act在每一步的
    transport.step之前调用；after_step只消费done更新首轨迹mask。
    finish使用原evaluator累计的净圈/路径/存活与终止事实，不另定义能力指标。
    """

    def __init__(
        self,
        output: Path,
        *,
        family: str,
        mode: str = "mean",
        seed: int | None = None,
        sample_stride: int = 4,
    ) -> None:
        """配置明确的采集路径和独立动作随机流；不创建环境或读取checkpoint。"""
        if family not in {"leap_right", "allegro_right"}:
            raise ValueError("family collection requires leap_right or allegro_right")
        if mode not in {"mean", "sample"} or (mode == "sample") != (seed is not None):
            raise ValueError("sample mode needs an action seed; mean mode must not specify one")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("action seed must be a nonnegative 63-bit integer")
        if output.exists() or sample_stride < 1:
            raise ValueError("collector needs a new output path and positive sample stride")
        self.output, self.family, self.mode, self.seed = output, family, mode, seed
        self.sample_stride = sample_stride  # 只影响监督保存，不改变环境时钟。
        self.writer: FamilyTrajectoryWriter | None = None  # start前没有可误读的完整数据。
        self.metadata: dict[str, Any] = {}  # 最终写入新数据的行为来源。
        self.active: torch.Tensor | None = None  # 每个env只属于其第一条轨迹。
        self.generator: torch.Generator | None = None  # 与全局Torch/环境随机流分离。
        self.prototype_index: torch.Tensor | None = None  # 路由数据，不是Actor特征。
        self.kinematics: Any = None  # 无学习参数的FK bank。
        self._step = 0  # 严格0-based控制动作计数。
        self.executed_steps = 0  # 只有transport.step正常返回才累计，供异常成本边界审计。

    def start(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        binding: Any,
        observation: Mapping[str, torch.Tensor],
        actor: Any,
        cohort_path: Path,
        cohort_members: Sequence[Mapping[str, Any]],
        steps: int,
        replicas: int,
    ) -> None:
        r"""绑定实际128资产/物理输入；H0是reset后已包含current0的真实历史。"""
        if self.writer is not None:
            raise RuntimeError("collector start may only occur once")
        self.actor = actor  # 仅用于结束时核对参数冻结，不赋予writer任何模型更新接口。
        self.initial_actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        policy, training = checkpoint_identity["policy"], checkpoint_identity["training"]
        if policy.get("arm") != "direct_token" or policy.get("actor_contact") != "tip-only-binary":
            raise ValueError("collector requires a frozen DirectToken TIP-only teacher")
        if training.get("history_encoder", "tcn") != "tcn" or training.get("phase_period_steps") is not None:
            raise ValueError("collector requires phase-free TCN History30")
        if training.get("sigma_mode", "global") != "global" or training.get("recovery_sigma_floor") is not None:
            raise ValueError("initial family collection requires the original global teacher distribution")
        count = len(binding.source_assets)  # 有序名义资产数A；不依策略表现删资产。
        if count != len(cohort_members) or observation["actor_jnt_current"].shape[0] != count * replicas:
            raise ValueError("collector cohort/environment axes disagree")
        expected_group = {"leap_right": "single_palm_leap", "allegro_right": "single_palm_allegro"}[self.family]
        for member, source in zip(cohort_members, binding.source_assets, strict=True):
            if member["provenance"]["group_name"] != expected_group or source.geometry_semantics.handedness != "right":
                raise ValueError("collector source is not the declared pure right-hand family")
        provider = runtime_identity["geometry_provider"]  # 真实N040来源域不同于canonical physical hash域。
        retained = provider["retained_artifact"]
        if retained["sha256"] != checkpoint_identity["geometry_provider"]["retained_artifact"]["sha256"]:
            raise ValueError("collector cannot replace the frozen N040 encoder")

        # 静态真值只lower一次；固定链折叠与canonical映射均来自已审计资产，不猜关节排列。
        slot_by_name = {name: i for i, name in enumerate(CANONICAL_HAND_SCHEMA_V1.joint_names)}
        mappings = [
            {source: slot_by_name[target] for source, target in a.routing.source_to_canonical}
            for a in binding.canonical_artifacts
        ]
        semantics = [source.geometry_semantics for source in binding.source_assets]
        cpu_bank = build_joint_kinematics_bank(semantics, mappings, dtype=torch.float64)
        device = observation["actor_jnt_current"].device  # 与teacher前向同一设备。
        self.kinematics = cpu_bank.to(device)
        self.prototype_index = torch.arange(count * replicas, device=device) % count
        self.active = torch.ones(count * replicas, dtype=torch.bool, device=device)
        if self.mode == "sample":
            assert self.seed is not None
            self.generator = torch.Generator(device=device).manual_seed(self.seed)  # 不消费环境全局RNG。
        env_assets = np.arange(count * replicas, dtype=np.int64) % count
        env_replicas = np.arange(count * replicas, dtype=np.int64) // count
        static = {"joint_kinematics": cpu_bank.features.float().numpy()}  # 输入统一FP32；解析FK独立使用FP64。
        for name in (
            "actor_jnt_limits",
            "jnt_valid",
            "tip_valid",
            "owner_valid",
            "shortest_path",
            "parent_direction",
            "child_direction",
        ):
            values = _numpy(observation[name])  # 静态只保存A行，避免每sample重复图矩阵。
            if not np.array_equal(values, values[:count][env_assets]):
                raise ValueError(f"static {name} differs across replicas of the same asset")
            static[name] = values[:count].copy()
        if not np.array_equal(static["jnt_valid"].astype(bool), cpu_bank.valid.numpy()):
            raise ValueError("FK static joint mask differs from the actual Actor mask")
        ordered_assets = [
            {
                "asset_index": i,
                "asset_id": source.asset_id,
                "source_urdf_path": str(source.urdf_path),
                "source_urdf_sha256": _sha256(source.urdf_path),
                "source_member_key": binding.source_member_keys[i],
                "canonical_physical_geometry_hash": artifact.physical_geometry_hash,
                "n040_input_fingerprint": provider["physical_geometry_hashes"][i],
                "configuration_domain_hash": artifact.source_content_hash,
                "source_geometry_semantics_hash": source.geometry_semantics.content_hash,
                "base_design_group": member["provenance"]["group_name"] + "/" + member["provenance"]["mother_name"],
                "source_provenance": dict(member["provenance"]),
            }
            for i, (source, artifact, member) in enumerate(
                zip(binding.source_assets, binding.canonical_artifacts, cohort_members, strict=True)
            )
        ]  # 两个hash域分别命名，留出隔离不能跨域作集合相交。
        self.metadata = {
            "family": self.family,
            "teacher_checkpoint": str(checkpoint_path.resolve()),
            "teacher_checkpoint_sha256": _sha256(checkpoint_path),
            "teacher_method_identity_digest": checkpoint_identity["identity_digest"],
            "runtime_identity_digest": runtime_identity["identity_digest"],
            "cohort_path": str(cohort_path.resolve()),
            "cohort_sha256": _sha256(cohort_path),
            "n040_sha256": retained["sha256"],
            "actor_abi": dict(ACTOR_ABI),
            "ordered_assets": ordered_assets,
            "geometry_cache_dtype": "float32",
            "target": "bounded_teacher_mean_before_action",
            "teacher_actor_freeze_contract": "all parameters and buffers must remain bitwise equal before finalize",
            "fk_target": "current_joint_frame_origin_in_hand_frame_metres",
            "kinematic_length_scale_m": 0.1,
            "fk_compute_dtype": "float64",  # 不受teacher TF32开关影响；保存仍为FP32米制。
            "protocol": {
                "action_mode": self.mode,
                "action_seed": self.seed,
                "steps": steps,
                "replicas": replicas,
                "sample_stride": self.sample_stride,
                "policy_dt_s": 0.05,
                "first_trajectory_only": True,
                "adr_enabled": False,
            },
            "collector_source_sha256": _sha256(Path(__file__)),
        }
        self.writer = FamilyTrajectoryWriter(
            self.output,
            self.metadata,
            steps=steps,
            env_asset_index=env_assets,
            env_replica_index=env_replicas,
            static=static,
            initial_history=_numpy(observation["actor_jnt_history"]),
            sample_stride=self.sample_stride,
        )  # start完成仍是incomplete；finalize成功才允许训练读取。

    def act(
        self,
        step: int,
        observation: Mapping[str, torch.Tensor],
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""保存同一动作前状态；mean原样执行或按教师原分布采样，不给有界动作加噪。"""
        if self.writer is None or self.active is None or step != self._step:
            raise RuntimeError("collector action requires a started, sequential rollout")
        action = mean  # 确定性采集保留原Tensor及原动作数值。
        if self.mode == "sample":
            from anymani.distill.rl.palm_rotation_ppo import PalmRotationMaskedContinuousModel

            location = PalmRotationMaskedContinuousModel.Network._action_to_latent(
                mean
            )  # 复用实际backend atanh/epsilon。
            valid = observation["jnt_valid"].bool()
            sigma = torch.exp(torch.where(valid, log_std.expand_as(mean), torch.zeros_like(mean)))
            sample = torch.normal(location, sigma, generator=self.generator)  # 独立且连续的原Normal随机流。
            action = torch.tanh(sample) * valid.to(dtype=mean.dtype)  # ghost动作精确零。
        fk = None  # 只在保存监督的时刻计算解析FK；不影响teacher行动。
        if step % self.sample_stride == 0:
            fk = _numpy(
                self.kinematics.joint_origins((observation["actor_jnt_current"][..., 0] * torch.pi).double(), self.prototype_index).float()
            )
        self.writer.append(
            step,
            jnt_current=_numpy(observation["actor_jnt_current"]),
            owner_contact=_numpy(observation["actor_owner_contact"]),
            teacher_mean=_numpy(mean),
            behavior_action=_numpy(action.clamp(-1, 1)),
            geometry_tokens=_numpy(observation["geometry_tokens"]),
            active=_numpy(self.active),
            history=_numpy(observation["actor_jnt_history"]),
            joint_origin_fk=fk,
        )  # 未active的自动reset行仍原样记录，只由mask排除，不能改写成假零动作。
        self._step += 1  # 与下一次transport.step严格一一对应。
        return action

    def after_step(self, done: torch.Tensor) -> None:
        """只消费当前step的done，保持每个环境首轨迹的不可逆结束状态。"""
        if self.active is None:
            raise RuntimeError("collector has not started")
        self.active &= ~done.bool()  # 不影响evaluator自己的active mask。
        self.executed_steps += 1  # 与真正完成的物理控制步一一对应。

    def finish(self, **summary: torch.Tensor) -> None:
        """原样保存已有pre-reset轨迹归约，随后关闭HDF5再计算文件身份。"""
        if self.writer is None or self.active is None:
            raise RuntimeError("collector has not started")
        current_state = self.actor.state_dict()  # 结束前验证真正执行模型，没有只检查磁盘checkpoint。
        if any(not torch.equal(value, current_state[name]) for name, value in self.initial_actor_state.items()):
            raise RuntimeError("teacher Actor parameters or buffers changed during collection")
        values = {name: _numpy(value) for name, value in summary.items()}
        values["terminated"] = _numpy(~self.active)  # 支持全部首轨迹提前结束的显式证明。
        self.writer.finalize(values)
        self.writer.close()  # HDF5 close可能更新header，散列必须在close之后。
        self.metadata["data_path"] = str(self.output.resolve())
        self.metadata["data_sha256"] = _sha256(self.output)
        self.metadata["teacher_actor_parameters_frozen_verified"] = True

    def close(self) -> None:
        """异常也关闭writer；未finalize的数据继续带incomplete标记。"""
        if self.writer is not None:
            self.writer.close()
