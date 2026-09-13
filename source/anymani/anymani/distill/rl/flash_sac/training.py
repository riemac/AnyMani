r"""FlashSAC的真实采样—回放—更新循环与逐回合证据。

一个vector step产生N条新transition，预热同样进入总预算。Q更新目标数量由已采样量决定，
U(C)=floor(max(C-C_warm,0)*UTD)，回放复用不增加C。本模块不创建第二个评价环境。
任务原始奖励、首30秒和完整回合由既有pre-reset记录器保存；Q的缩放奖励只用于学习。
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import read_checkpoint, write_checkpoint
from .config import FlashSACConfig
from .learner import FlashSACLearner
from .metrics import FirstThirtySecondsMetrics
from .observations import StaticHandBank, compact_observation
from .replay import CompactReplay


class FlashSACTrainingRun:
    r"""单进程、单环境集的异策略训练；学习状态与物理恢复段分别记账。

    Args:
        config: 已解析的同一方法配置，包括总新交互量与回放布局。
        environment: 已创建的FlashSACEnvironment，拥有唯一仿真器实例。
        provider: 按当前q重建N040 tokens的冻结provider。
        identity: 原生SAC方法身份，绑定任务、成员、预抓取、模型和实际源码。
        run_dir: 本次新运行目录，允许环境装配先写params，拒绝覆盖已有训练指标。
        resume: 完整resume checkpoint；重新创建物理环境形成显式新历史段。

    周期模型快照保留，包含回放的大状态只维护原子更新的resume.pt。
    保存操作同步进行，期间不append回放；因此无须复制整份CPU回放池。
    """

    def __init__(self, config: FlashSACConfig, environment: Any, provider: Any,
                 identity: Mapping[str, Any], run_dir: Path, *, resume: Path | None = None) -> None:
        self.config, self.env, self.provider = config, environment, provider  # 共享已创建的物理环境。
        self.identity, self.run_dir = dict(identity), Path(run_dir)  # 方法身份与产物位置分开。
        self.device = torch.device(environment.transport._rl_device)  # Actor/Q与transport使用相同设备。
        self.run_dir.mkdir(parents=True, exist_ok=True)  # 场景装配可能已经写入params。
        self.metrics_path = self.run_dir / "metrics.jsonl"  # SAC学习统计，不使用PPO的value-loss schema。
        self.asset_metrics_path = self.run_dir / "asset_metrics.jsonl"  # 每资产原始任务统计。
        if self.metrics_path.exists() or self.asset_metrics_path.exists():
            raise FileExistsError("use a new run directory; existing training metrics are immutable")
        self.learner = FlashSACLearner(config, self.device)  # 真正独立的Actor/双Q/温度/优化器。
        labels = environment.transport.prototype_index.detach().reshape(-1).long()  # [N]，只用于路由。
        self.replay = CompactReplay(config.replay_capacity, config.num_envs, labels,
                                    config.history_steps, config.n_step, config.gamma, config.replay_device)
        self.replay_rng = torch.Generator(device=self.replay.device).manual_seed(config.seed + 203)  # 独立抽样流。
        self.first30 = FirstThirtySecondsMetrics(labels)  # 训练窗口统计，不是冻结R16评价。
        self.resume_source = str(resume.resolve()) if resume is not None else None  # 恢复来源只属于实验lineage。

        # 模型和回放恢复后，物理状态从严格预抓取重新开始；历史与奖励trace在新段清零。
        restored = read_checkpoint(resume, expected_identity=self.identity, require_replay=True) if resume else None
        if restored is not None:
            self.learner.load_state_dict(restored["learner"])  # 参数、Adam、温度、统计与随机流。
            self.replay.load_state_dict(restored["replay"])  # 原transition保留，不将物理重启伪造成失败。
            experiment = restored["experiment"]  # 训练循环自身的状态。
            self.replay_rng.set_state(experiment["replay_rng"].cpu())  # 保持抽样随机流。
            random.setstate(experiment["python_rng"])  # 恢复任务可能消费的Python随机流。
            numpy_rng = experiment["numpy_rng"]  # checkpoint以纯标量和Tensor保存MT19937状态。
            np.random.set_state((numpy_rng[0], numpy_rng[1].cpu().numpy().astype(np.uint32),
                                 numpy_rng[2], numpy_rng[3], numpy_rng[4]))  # 恢复环境侧NumPy随机状态。
            self.first30.load_state_dict(experiment["first30"])  # 保留已有资产窗口与删失计数。
            environment.transport.set_env_state(experiment["environment"])  # 恢复原课程/ADR状态。
            self.replay.reset_streams()  # 下一条current开启新历史段。
            self.first30.reset_streams()  # 新物理回合允许重新结算首30秒。
            self.learner.reward_normalizer.returns.zero_()  # 统计trace不能跨物理重建拼接；全局moments保留。
        environment.transport.configure_training_evidence(self.run_dir, self.identity["identity_digest"])
        self.observation = environment.reset()  # live包含正常reset填充的History30。
        self.bank = StaticHandBank.from_live(self.observation, labels)  # 按资产保存一份静态限位/mask/图。
        if self.bank.asset_count != config.asset_count:
            raise ValueError("static bank and declared cohort asset counts disagree")
        if restored is not None:
            for name, value in self.bank.fields.items():
                if not torch.equal(value, restored["static_bank"][name]):
                    raise ValueError(f"restored static hand field differs: {name}")
        if self.replay.total_transitions != self.learner.collected_transitions:
            raise ValueError("replay and learner disagree on previously collected transitions")

        # 早期直接核对原ObservationManager与回放的History30；仅保留n条小型参考窗口。
        self._history_checks: deque[tuple[int, torch.Tensor]] = deque()  # (absolute sequence, producer history)。
        self._checked_histories = 0  # 完成逐字段匹配的vector-step数。
        self._terminal_count = self._timeout_count = 0  # 真实pre-reset事件的累计行数。
        self._start_time = time.perf_counter()  # 不含场景创建的训练墙钟起点。
        self._sampling_seconds = self._update_seconds = 0.0  # 独立计时，含各自必要同步。
        self._last_learning: dict[str, float] = {}  # 最近有定义的学习统计；Actor隔次更新。
        self._last_checkpoint = self.learner.collected_transitions  # 本段已发布的最近完整状态。
        self._last_logged = self.learner.collected_transitions  # 保证每个日志窗口含至少一次真实采样。
        self.attempted_transitions = self.learner.collected_transitions  # 失败时保守计入已请求的物理步预算。
        self._sigma_sum = torch.zeros((), device=self.device)  # 当前显示窗口的per-active sigma均值和。
        self._sigma_steps = 0  # 上项的vector-step分母。

    def _check_history(self) -> None:
        r"""比较成熟回放起点与真实采样时保存的History30，不读取模型生成的隐变量。"""
        if not self._history_checks:
            return  # 初始验证结束后不再保存完整live历史参考。
        sequence, expected = self._history_checks[0]  # 最早尚未成熟的起点。
        if self.replay.total_transitions // self.config.num_envs < sequence + self.config.n_step:
            return  # 等待真实未来transition，不以人工复制凑n-step目标。
        envs = torch.arange(self.config.num_envs, device=self.replay.device)  # 本次检查全部物理副本。
        batch = self.replay.gather(envs, torch.full_like(envs, sequence))  # 确定性审计入口，不消耗回放RNG。
        torch.testing.assert_close(batch["obs"]["actor_jnt_history"].cpu(), expected, rtol=0, atol=0)
        self._history_checks.popleft()  # 匹配成功后释放这份参考历史。
        self._checked_histories += 1  # 一个验证单元覆盖全部N个环境。

    def _save(self) -> Path:
        r"""只在完整采样/更新边界发布模型和resume；绝不从半次优化的异常出口保存。"""
        count = self.learner.collected_transitions  # C，真实新transition而非回放复用量。
        if count != self.replay.total_transitions:
            raise RuntimeError("checkpoint is not at a completed transition boundary")
        self.env.transport.training_evidence.drain(force=True)  # 完整回合先落盘，再保存模型。
        numpy_rng: Any = np.random.get_state()  # MT19937的624个uint32字与位置/高斯缓存；NumPy stub另允许非legacy字典。
        experiment = {
            "run_dir": str(self.run_dir.resolve()), "resume_source": self.resume_source,  # 实验lineage。
            "replay_rng": self.replay_rng.get_state(), "first30": self.first30.state_dict(),
            "python_rng": random.getstate(),  # Python状态由基础标量/元组组成。
            "numpy_rng": (str(numpy_rng[0]), torch.from_numpy(numpy_rng[1].astype(np.int64)),
                          int(numpy_rng[2]), int(numpy_rng[3]), float(numpy_rng[4])),  # weights_only可读的无损随机状态。
            "environment": self.env.transport.get_env_state(),  # 课程/ADR，与物理重启段明确分开。
            "collected_transitions": count, "physical_resume": "new-strict-pregrasp-history-segment",
        }
        learner_state = self.learner.state_dict()  # GPU模型引用由同步序列化消费。
        bank_state = self.bank.state_dict()  # 只克隆小型静态资产表。
        path = self.run_dir / "nn" / f"transitions_{count:09d}.pt"  # 不按训练reward重选或覆盖周期模型。
        write_checkpoint(path, learner_state=learner_state, identity=self.identity,
                         static_bank_state=bank_state, experiment_state=experiment)
        write_checkpoint(self.run_dir / "nn/resume.pt", learner_state=learner_state, identity=self.identity,
                         static_bank_state=bank_state, replay_state=self.replay.state_dict(), experiment_state=experiment)
        self._last_checkpoint = count  # 两份状态完整发布后才前移cursor。
        print(f"[FlashSAC CHECKPOINT] transitions={count} model={path.name}", flush=True)
        return path  # 用户分析与后续恢复都可定位明确的完整边界。

    def _log(self, metrics_stream: Any, asset_stream: Any) -> dict[str, Any]:
        r"""按窗口记录任务原始统计及SAC学习量；GPU余量测量发生在已消费batch释放后。"""
        from anymani.distill.diagnostics.recording.rl.runtime import read_linux_process_resources

        count = self.learner.collected_transitions  # 本窗口结束坐标。
        task = self.env.transport.drain_rollout_metrics()  # 每资产CPU向量，保留完整回合分母。
        gc.collect()  # 回收已经不再使用的Python引用环。
        resources: dict[str, Any] = dict(read_linux_process_resources(os.getpid()))  # RSS/swap/system RAM事实。
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # 释放未使用allocator缓存，不影响活跃模型或回放。
            free, total = torch.cuda.mem_get_info(self.device)  # 包含PhysX/driver的真实剩余量。
            resources.update(gpu_driver_free_bytes=free, gpu_driver_total_bytes=total,
                             gpu_allocated_bytes=torch.cuda.memory_allocated(self.device),
                             gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device))
            if free < self.config.gpu_headroom_bytes:
                raise RuntimeError(f"FlashSAC GPU headroom {free} B is below {self.config.gpu_headroom_bytes} B")
        sigma = float((self._sigma_sum / max(1, self._sigma_steps)).cpu())  # per-active latent sigma，不是物理弧度。
        row: dict[str, Any] = {
            "schema_version": "flash-sac-metrics-1", "identity_digest": self.identity["identity_digest"],
            "transitions": count, "critic_updates": self.learner.critic_updates,
            "actor_updates": self.learner.actor_updates, "window_transitions": count - self._last_logged,
            "training_wall_seconds": time.perf_counter() - self._start_time,
            "sampling_seconds": self._sampling_seconds, "update_seconds": self._update_seconds,
            "replay_occupancy": min(count, self.replay.actual_capacity), "replay_bytes": self.replay.storage_bytes,
            "policy_sigma_per_active": sigma, "learning_last": self._last_learning.copy(),
            "first30": self.first30.summary(), "resources": resources, "families": {},
        }
        # 固定训练cohort为LEAP128后接Allegro128；小型CPU fixture使用一个明确的all组。
        groups = {"LEAP": (0, 128), "Allegro": (128, 256)} if self.config.asset_count == 256 else {"all": (0, self.config.asset_count)}
        for name, (start, end) in groups.items():
            completed = task["completed_episode_count"][start:end]  # 异步完成回合数，不能平均各asset的均值。
            ended = int(completed.sum())  # 本窗口自然结束的真实回合分母。
            terminal = float((task["terminal_net_turns_mean"][start:end] * completed).sum() / ended) if ended else None
            group = {"live_net": float(task["net_turns_mean"][start:end].mean()),
                     "reward_per_step": float(task["reward_mean"][start:end].mean()),
                     "ended": ended, "terminal_net": terminal,
                     "first30": self.first30.summary(list(range(start, end)))}  # 训练窗口、跨策略版本。
            row["families"][name] = group  # 不把分族metadata拼到Actor输入。
        metrics_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")  # 每次一条完整JSONL。
        for asset in range(self.config.asset_count):
            record = {"transitions": count, "asset_index": asset,
                      **{key: float(value[asset]) for key, value in task.items()}}  # 原始任务项及每个分母。
            asset_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        metrics_stream.flush()  # 日志与checkpoint各自明确的durability边界。
        asset_stream.flush()  # 保留同一采样窗口的逐资产原始分母。
        free_gib = resources.get("gpu_driver_free_bytes", 0) / 2**30  # 仅显示层换算GiB。
        print(f"[FlashSAC] data={count/1e6:.3f}/{self.config.total_transitions/1e6:.3f}M "
              f"Q={self.learner.critic_updates} Actor={self.learner.actor_updates} "
              f"sigma={sigma:.3f} GPU-free={free_gib:.2f}GiB", flush=True)
        for name, group in row["families"].items():
            first = group["first30"]["first30_asset_mean_net"]  # 近期训练episode均值，不是R16中位数。
            first_text = "pending" if first is None else f"{first:.3f}"
            print(f"  {name}: train_live_net={group['live_net']:.3f} first30_train_mean={first_text} "
                  f"reward/step={group['reward_per_step']:+.4f} ended={group['ended']}", flush=True)
        self._last_logged = count  # 下一窗口从本次真实采样边界开始。
        self._sigma_sum.zero_()  # 只清当前显示窗口，不清学习统计。
        self._sigma_steps = 0  # 下个窗口的vector-step分母从零计数。
        return row  # 终局summary可引用最后一个有明确定义的日志窗口。

    def run(self, *, stop_after_transitions: int | None = None) -> dict[str, Any]:
        r"""推进到总预算或显式完整边界，前64步同时验证真实History30数据流。

        stop_after_transitions只截短本次进程，不改变方法总预算，适合完整checkpoint续接。
        达到总预算后只关闭训练、保存原始产物，不创建冻结策略回放。
        """
        limit = self.config.total_transitions if stop_after_transitions is None else stop_after_transitions
        if limit > self.config.total_transitions or limit <= self.learner.collected_transitions or limit % self.config.num_envs:
            raise ValueError("stop boundary must lie inside the remaining budget and on a vector-step boundary")
        last_row: dict[str, Any] = {}  # 至少一次log后才发布终局摘要。
        initial_sequence = self.replay.total_transitions // self.config.num_envs  # 本次物理段起点。
        with self.metrics_path.open("x", encoding="utf-8") as metrics_stream, self.asset_metrics_path.open("x", encoding="utf-8") as asset_stream:
            while self.learner.collected_transitions < limit:
                count = self.learner.collected_transitions  # 本次动作真正使用的策略版本坐标。
                self.env.transport.set_train_info(count)  # 逐回合记录保留起止版本，不伪装冻结策略。
                sequence = self.replay.total_transitions // self.config.num_envs  # 绝对vector-step索引。
                if sequence - initial_sequence < 64:
                    self._history_checks.append((sequence, self.observation["actor_jnt_history"].detach().cpu().clone()))
                begin = time.perf_counter()  # 采样包括Actor、六个物理子步、transfer与append。
                sample = self.learner.act(self.observation, explore=True)  # 行独立、时间相关的行为噪声。
                current = compact_observation(self.observation)  # 只保存动态单步字段。
                self.attempted_transitions += self.config.num_envs  # step若部分失败，也不把已请求交互隐去。
                step = self.env.step(sample.actions)  # 显式pre-reset next与reset后live状态分开。
                self.replay.append(current, sample.actions, step.reward, step.terminated, step.truncated, step.next_compact)
                self.learner.observe_transition(step.reward, step.terminated, step.truncated)  # 原始奖励统计与预算恰更新一次。
                snapshot = self.env.raw.command_manager.get_term("goal_pose").post_physics_evaluation_snapshot
                self.first30.observe(snapshot, count)  # 使用训练环境已形成的事实，无第二次step。
                self._terminal_count += int(step.terminated.sum().cpu())  # 小型事件计数，不保存全量GPU快照。
                self._timeout_count += int(step.truncated.sum().cpu())  # 有限时域终点同样保留。
                valid = self.observation["jnt_valid"].bool()  # sigma只按真实关节求均值。
                per_env_sigma = sample.log_std.detach().exp().masked_fill(~valid, 0).sum(-1) / sample.active_count
                self._sigma_sum += per_env_sigma.mean()  # 先每环境按active DoF平均，再按资产等副本平均。
                self._sigma_steps += 1  # 每个vector-step具有相同N与相同资产配额。
                self.observation = step.observation  # 后续控制必须使用reset后的实时观测。
                self._sampling_seconds += time.perf_counter() - begin  # 同步append已经完成全部跨设备复制。
                self._check_history()  # 与真实ObservationManager逐值核对，不使用模型输出作为参照。

                # 每次更新只消费成熟回放；UTD按新transition预算累计，与日志频率无关。
                desired = self.config.critic_update_budget(self.learner.collected_transitions)
                begin = time.perf_counter()  # 包括重建历史/几何、CPU传输和Actor/Q优化。
                while self.replay.ready and self.learner.critic_updates < desired:
                    batch = self.replay.sample(self.config.batch_size, generator=self.replay_rng)  # 每资产严格等额。
                    batch["obs"] = self.bank.assemble(batch["obs"], batch["asset_index"], self.provider, self.device)
                    batch["next_obs"] = self.bank.assemble(batch["next_obs"], batch["asset_index"], self.provider, self.device)
                    self._last_learning.update(self.learner.update(batch))  # 温度/Actor隔次更新，Q每次更新。
                    del batch  # 已消费的GPU批次不跨迭代保留。
                self._update_seconds += time.perf_counter() - begin  # learner返回标量时已同步关键GPU计算。
                count = self.learner.collected_transitions  # 完整step与其所需更新已经完成。
                if count // self.config.console_interval > self._last_logged // self.config.console_interval or count == limit:
                    last_row = self._log(metrics_stream, asset_stream)  # 小批终止记录也在此被drain。
                if count // self.config.checkpoint_interval > self._last_checkpoint // self.config.checkpoint_interval:
                    self._save()  # 完整边界原子发布，半次update异常不走这里。
            checkpoint = self._save() if self._last_checkpoint != limit else self.run_dir / "nn" / f"transitions_{limit:09d}.pt"
        report = {
            "status": "completed" if limit == self.config.total_transitions else "stopped-at-declared-boundary",
            "transitions": limit, "critic_updates": self.learner.critic_updates,
            "actor_updates": self.learner.actor_updates, "model_checkpoint": str(checkpoint),
            "history_vector_steps_verified": self._checked_histories, "physical_termination_rows": self._terminal_count,
            "timeout_rows": self._timeout_count, "last_metrics": last_row, "assessment": "saved-artifacts-only",
        }
        (self.run_dir / "training_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return report  # 环境由入口finally关闭，并补写仍在途回合的右删失记录。
