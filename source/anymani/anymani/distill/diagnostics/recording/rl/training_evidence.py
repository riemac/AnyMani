r"""训练期逐回合与逐资产奖励证据；消费既有pre-reset快照，不运行MDP。

每个policy step仅在设备端累加；rollout结束时转移小批终止记录，按容量批量写Parquet。
policy_version使用调用方给出的rollout起点transition计数，明确跨策略更新的episode。
回合奖励积分、逐资产每策略步奖励均值和首30秒窗口统计使用不同分母，分别保存可审计的原始量。
首窗可在回合尚未结束时发布；窗口边界来自实际物理时钟，当前步奖励与目标脉冲均先计入再结算。
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .episode_evidence import write_episode_evidence, write_first_window_evidence
from .first_window import FirstWindowStatistics


class TrainingEvidence:
    r"""保留每个episode的联合终止事实与奖励积分，独立于TensorBoard均值。

    $N$为并行环境数，$R$为实际奖励项数，$A=1+\max(asset\_index)$为资产索引范围。
    每个环境独立累计回合奖励；同资产多环境的逐步奖励另外汇总，供rollout边界按环境步样本数归约。
    可选首窗记录以回合首30秒或提前物理失败为单位，其近期统计由资产等权的FirstWindowStatistics归约。
    """

    def __init__(
        self,
        root: Path,
        identity_digest: str,
        asset_index: torch.Tensor,
        reward_names: Sequence[str],
        *,
        policy_dt_s: float = 0.05,
        flush_rows: int = 4096,
        first_window_root: Path | None = None,
    ) -> None:
        r"""分配episode积分与可选首30秒锁存器，不消耗训练Torch/NumPy随机流。

        first_window_root显式启用首30秒联合证据。它由实际经过时间锁存，与奖励课程的
        reference_seconds无关；跨策略版本的窗口保留真实起止版本，不能解释为冻结R16。

        Args:
            root: 完整回合终止及停训删失证据的分片目录。
            identity_digest: 调用方冻结的方法身份摘要，逐分片传递给写入器。
            asset_index: [N]非负整数张量，给出各物理环境的资产索引；缓冲区与它位于同一设备。
            reward_names: R个实际奖励项的有序名称，与capture输入矩阵的列一一对应。
            policy_dt_s: 秒/策略步，默认0.05；真实30秒对应600个已完成动作。
            flush_rows: CPU待发布事件达到此行数后批量写入，默认4096；不是奖励均值的分母。
            first_window_root: 显式启用首窗证据的独立目录；None表示仅记录回合证据。
        """
        # 资产映射固定训练证据的分组测度；不能把浮点或布尔张量隐式转换成手型索引。
        if asset_index.ndim != 1 or asset_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):  # 每环境一个整数索引
            raise ValueError("training evidence asset indices must be a one-dimensional integer tensor")  # 形状必须为[N]
        if asset_index.numel() == 0 or bool((asset_index < 0).any()):  # 至少一个物理环境且所有索引非负
            raise ValueError("training evidence requires nonempty nonnegative asset indices")  # A由最大合法索引确定
        self.root = Path(root)  # 完整episode结束/停训删失记录目录。
        self.identity_digest = identity_digest  # 原训练方法身份，由调用方冻结。
        self.segment_id = uuid.uuid4().hex  # 新物理进程段，与策略采样随机流独立。
        self.asset_index = asset_index.detach().long().clone()  # [N]索引只用于分组与来源记录。
        self.reward_names = tuple(reward_names)  # 保留RewardManager实际term顺序。
        self.policy_dt_s = policy_dt_s  # 秒/策略步，当前固定.05。
        self.flush_rows = flush_rows  # CPU积累到该数量再发布分片。
        self.policy_version = 0  # 当前rollout开始时的累计新transition坐标。
        self._next_shard = 0  # 同一物理段内单调增长的episode分片号。
        self._pending: list[dict[str, np.ndarray]] = []  # 已完成device transfer的待发布记录。
        self._gpu_rows: list[dict[str, torch.Tensor]] = []  # 当前rollout内已结束的少量episode行。
        self._last_snapshot: dict[str, torch.Tensor] | None = None  # close读取末帧的在途回合。
        n = asset_index.numel()  # N为并行环境数，各环境有独立episode时钟。
        self._episode_id = torch.zeros(n, dtype=torch.long, device=asset_index.device)  # 每环境递增计数。
        self._episode_start = torch.zeros_like(self._episode_id)  # 首个动作所用策略版本。
        self._steps = torch.zeros_like(self._episode_id)  # 本进程观察到的本回合策略步数。
        self._reward_sums = torch.zeros(n, len(reward_names), device=asset_index.device)  # [N,R]实际单步贡献积分。
        self._asset_reward_sums = torch.zeros(int(asset_index.max()) + 1, len(reward_names), device=asset_index.device)  # [A,R]本轮逐环境步贡献和
        self._asset_samples = torch.zeros(self._asset_reward_sums.shape[0], device=asset_index.device)  # [A]本轮环境步样本数，不是episode数

        # 首30秒事件独立于episode终点发布；只保存小型物理量，不复制观察、几何或训练图。
        self.first_window_root = Path(first_window_root) if first_window_root is not None else None  # 首窗与回合表使用不同统计单位
        if self.first_window_root is not None and self.first_window_root.resolve() == self.root.resolve():  # 不混写两种schema
            raise ValueError("first-window and episode evidence need separate directories")  # 路径解析后也必须独立
        self.first30_statistics = FirstWindowStatistics(self._asset_samples.numel()) if first_window_root is not None else None  # A个资产各留最近32窗
        self._next_window_shard = 0  # first30目录中的独立单调分片游标。
        self._gpu_window_rows: list[dict[str, torch.Tensor]] = []  # 本rollout首次完成/早失败窗口。
        self._pending_windows: list[dict[str, np.ndarray]] = []  # 批量CPU写入窗口表。
        self._first30_prefix: dict[str, torch.Tensor] = {}  # 第600步后各字段均保持锁存。
        if self.first30_statistics is not None:  # 仅显式启用时维护真实首窗的物理前缀
            for name in ("net_turns_first30", "absolute_path_turns_first30"):  # 净转与绝对路径共享同一起止时刻
                self._first30_prefix[name] = torch.zeros(n, device=asset_index.device)  # 圈数，未满30秒时为已观察前缀。
            self._first30_prefix["goal_count_first30"] = torch.zeros_like(self._episode_id)  # 已含当步成功pulse的目标数。
            self._first30_prefix["first30_complete"] = torch.zeros(n, dtype=torch.bool, device=asset_index.device)  # [N]是否已到30秒锁存边界
            self._first30_prefix["first30_safe"] = torch.zeros(n, dtype=torch.bool, device=asset_index.device)  # [N]完整首窗且边界无物理失败

    def capture(self, snapshot: Mapping[str, torch.Tensor], weighted_step_rewards: torch.Tensor) -> None:
        r"""输入奖励已经乘weight和dt；snapshot必须在自动reset之前冻结。

        先累计当前动作的奖励并结算首窗，再记录整回合终点并清零；同一步的目标脉冲由snapshot提供。
        Args:
            snapshot: 所有环境的post-physics、pre-reset终止/时间/旋转/目标事实，环境轴为N。
            weighted_step_rewards: [N,R]各奖励项本策略步的实际贡献，已含权重及dt，不再次缩放。
        """
        # 同一环境的R项奖励保持联合积分，终止动作自身的奖励也属于该回合。
        if weighted_step_rewards.shape != self._reward_sums.shape:  # 输入必须与[N,R]缓冲区逐项对应
            raise ValueError("reward evidence matrix must match [environments, reward terms]")  # 不广播缺少的环境或term
        first = self._steps == 0  # [N]本进程首次观察到该回合动作的环境
        self._episode_start[first] = self.policy_version  # 起始版本一旦设定，在本回合内保持不变
        self._steps.add_(1)  # 累计本进程观察到的策略步，用于新回合识别及停训删失选择
        self._reward_sums.add_(weighted_step_rewards.detach())  # $S_{e,r}\leftarrow S_{e,r}+\Delta R_{e,r}$，回合积分
        self._asset_reward_sums.index_add_(0, self.asset_index, weighted_step_rewards.detach())  # [A,R]同资产各环境本步贡献相加
        self._asset_samples.index_add_(0, self.asset_index, torch.ones_like(self.asset_index, dtype=torch.float32))  # [A]每环境每步贡献一个分母样本
        if self.first30_statistics is not None:
            self._capture_first_window(snapshot)  # 必须先结算旧回合窗口，再清理reset rows。
        terminal = (  # [N]整回合终点，包括物理失败及时间截断
            snapshot["termination_object_out_of_anchor"]  # 物体离开固定位置锚范围，对应drop记录
            | snapshot["termination_goal_axis_misaligned"]  # 旋转目标轴失配，对应axis记录
            | snapshot["termination_time_out"]  # timeout结束回合，但纯早timeout不产生首窗事件
        )
        ids = terminal.nonzero(as_tuple=False).reshape(-1)  # [E]本步结束的环境索引，E可为0
        self._gpu_rows.append(self._rows(snapshot, ids, censored=False))  # 冻结旧回合事实，随后reset不会改写
        self._steps[ids] = 0  # 下一动作重新建立新回合起始版本
        self._reward_sums[ids] = 0  # 仅清零已终止环境，其他回合继续积分
        for value in self._first30_prefix.values():  # 新回合从自己的零时刻开始观察首窗
            value[ids] = 0  # 新物理回合重新锁存；先前发布的窗口行已经clone。
        self._episode_id[ids] += 1  # 每环境独立递增；前一回合编号已写入克隆行
        self._last_snapshot = dict(snapshot)  # 仅复制字段映射；close在最后capture后、生产者再次改写张量前读取

    def _rows(
        self, snapshot: Mapping[str, torch.Tensor], ids: torch.Tensor, *, censored: bool
    ) -> dict[str, torch.Tensor]:
        r"""只选本次结束的环境；clone使下一step和reset不能覆盖已收集事实。"""
        duration = snapshot["episode_duration_s"][ids]  # [E]任务给出的实际回合时长，单位秒
        result = {  # 全部列共享同一批[E]环境索引与pre-reset终点
            "env_id": ids,  # 物理进程段中的环境索引
            "episode_id": self._episode_id[ids],  # 本环境刚结束或被删失的回合编号
            "asset_index": self.asset_index[ids],  # 回合所属资产，不按事件频率改变资产身份
            "policy_version_start": self._episode_start[ids],  # 首个动作的rollout版本坐标
            "policy_version_end": torch.full_like(ids, self.policy_version),  # 结算时版本，可晚于起始版本
            "policy_steps": torch.round(duration / self.policy_dt_s).long(),  # 实际时长/dt还原步数，round消除浮点时钟误差
            "duration_s": duration,  # 保留实际短回合或长回合终点，不替换为计划horizon
            "net_turns": snapshot["net_rotation_rad"][ids] / (2 * torch.pi),  # 有符号净转，弧度/(2π)转为圈
            "absolute_path_turns": snapshot["absolute_path_rotation_rad"][ids] / (2 * torch.pi),  # 绝对路径圈数，包含正反运动
            "max_positive_net_turns": snapshot["max_positive_net_rotation_rad"][ids] / (2 * torch.pi),  # 历史正向净转前沿，单位圈
            "goal_count": snapshot["completed_subgoals"][ids].long(),  # 严格目标累计计数，含当前成功脉冲
            "frontier_count": snapshot["rotation_frontier_count"][ids].long(),  # 独立的旋转前沿事件计数
            "termination_drop": snapshot["termination_object_out_of_anchor"][ids].bool(),  # 位置锚越界事实
            "termination_axis": snapshot["termination_goal_axis_misaligned"][ids].bool(),  # 目标轴失配事实
            "termination_timeout": snapshot["termination_time_out"][ids].bool(),  # 时间截断事实，不等于物理失败
            "censored": torch.full_like(ids, censored, dtype=torch.bool),  # 停训时尚未结束的回合单列为右删失
        }
        result.update({f"reward/{name}": self._reward_sums[ids, i] for i, name in enumerate(self.reward_names)})  # 每term的[E]回合积分
        if "completed_orientation_subgoals" in snapshot:  # 可选仅朝向目标事件，与严格位置+朝向目标区分
            result["orientation_goal_count"] = snapshot["completed_orientation_subgoals"][ids].long()  # [E]环境累计计数
        for name in ("adr_position_level", "adr_position_offset_x_h_m", "adr_position_offset_y_h_m", "net_turns_first30", "first30_complete"):
            if name in snapshot:  # 缺失扩展字段不补成零或默认档位
                result[name] = snapshot[name][ids]  # 只记录pre-reset的真实值，不能读reset后的新档位。
        result.update({name: value[ids] for name, value in self._first30_prefix.items()})  # 真实30秒窗口优先于课程参考时长。
        return {key: value.detach().clone() for key, value in result.items()}  # 独立设备端副本，隔离后续积分清零及物理reset

    def _capture_first_window(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        r"""读取首次30秒边界或此前物理失败的事实，目标计数含当前成功脉冲。

        未满30秒的prefix随真实物理步更新；第一次duration>=30时锁存，之后不再改变。
        complete之前的drop/axis提供一个实际观察到的失败窗口，纯早timeout只留episode记录。
        """
        previous_complete = self._first30_prefix["first30_complete"]  # [N]上一个策略步的锁存状态。
        fresh = ~previous_complete  # 只允许本回合尚未达到30秒的环境更新窗口。
        duration = snapshot["episode_duration_s"]  # 秒；不消费可能采用120秒参考的课程字段。
        complete = duration >= 30.0  # 原20Hz下精确对应第600个已完成动作。
        failure = snapshot["termination_object_out_of_anchor"] | snapshot["termination_goal_axis_misaligned"]  # [N]drop/axis物理终点
        values = {  # 当前动作后的同源前缀；fresh为假时不再写入
            "net_turns_first30": snapshot["net_rotation_rad"] / (2 * torch.pi),  # 有符号圈数。
            "absolute_path_turns_first30": snapshot["absolute_path_rotation_rad"] / (2 * torch.pi),  # 弧度转圈，与净转共享观察起止
            "goal_count_first30": snapshot["completed_subgoals"].long(),  # post-physics计数已经含当步pulse。
            "first30_complete": complete,  # [N]真实物理时钟是否达到30秒，与课程参考时长无关
            "first30_safe": complete & ~failure,  # 在30秒边界失败也必须不安全。
        }
        for name, current in values.items():  # 各物理量、目标计数与布尔标志使用相同锁存掩码
            self._first30_prefix[name] = torch.where(fresh, current, self._first30_prefix[name]).detach()  # 到达30秒后保留当时事实
        ids = (fresh & (complete | failure)).nonzero(as_tuple=False).reshape(-1)  # 每回合最多一个可观测窗口事件。
        result = {  # [E]已到观察边界的窗口；纯早timeout既不complete也不failure，故不进入ids
            "env_id": ids, "episode_id": self._episode_id[ids], "asset_index": self.asset_index[ids],  # 同回合键与资产来源
            "policy_version_start": self._episode_start[ids],  # 回合首个动作即首窗起点
            "policy_version_end": torch.full_like(ids, self.policy_version),  # 当前窗口结算版本，不等待回合最终版本
            "policy_steps": torch.round(duration[ids] / self.policy_dt_s).long(),  # 20Hz时完整窗600步，早失败保留实际短时步数
            "duration_s": duration[ids],  # 单位秒；不以30/duration放大运动或奖励
            "net_turns": self._first30_prefix["net_turns_first30"][ids],  # 有符号净转圈数
            "absolute_path_turns": self._first30_prefix["absolute_path_turns_first30"][ids],  # 相同前缀的绝对路径圈数
            "goal_count": self._first30_prefix["goal_count_first30"][ids],  # 含窗口末步成功脉冲的严格计数
            "complete": complete[ids], "safe": self._first30_prefix["first30_safe"][ids],  # 完成与安全分开，边界失败完整但不安全
            "termination_drop": snapshot["termination_object_out_of_anchor"][ids],  # 窗口终点的位置锚越界标志
            "termination_axis": snapshot["termination_goal_axis_misaligned"][ids],  # 窗口终点的轴失配标志
            "termination_timeout": snapshot["termination_time_out"][ids],  # 可与其他终止并发，纯早timeout不构成窗口
        }  # 所有行来自同一个pre-reset时刻，不混入新回合观察。
        self._gpu_window_rows.append({name: value.detach().clone() for name, value in result.items()})  # reset前克隆，后续新回合不覆盖

    def _drain_first_windows(self, *, force: bool) -> None:
        r"""rollout边界只搬运新窗口行；CPU近期统计与完整窗口表使用同一批事实。"""
        if self.first30_statistics is None:  # 无显式首窗目录时不维护首窗统计或事件表
            return  # 其他既有consumer可只启用episode证据。
        if self._gpu_window_rows:  # 按捕获顺序拼接本rollout事件，避免控制步内逐行CPU同步
            columns = {name: torch.cat([row[name] for row in self._gpu_window_rows]).cpu().numpy() for name in self._gpu_window_rows[0]}  # 各列[E]一次批量搬运
            self._gpu_window_rows.clear()  # device侧只保留本rollout的有限事件载荷。
            if columns["env_id"].size:  # 多个空事件批拼接后仍可能为E=0
                self.first30_statistics.add_batch(columns)  # 统计按资产保留近期32窗；不是冻结能力。
                self._pending_windows.append(columns)  # 原始窗口不因近期FIFO淘汰而丢失。
        if self._pending_windows and (force or sum(row["env_id"].size for row in self._pending_windows) >= self.flush_rows):  # 按事件行数而非环境步数触发发布
            assert self.first_window_root is not None  # 显式启用时构造器已绑定独立目录。
            columns = {name: np.concatenate([row[name] for row in self._pending_windows]) for name in self._pending_windows[0]}  # 多rollout窗口顺序保留
            destination = self.first_window_root / f"windows-{self.segment_id}-{self._next_window_shard:06d}.parquet"  # 段身份+独立首窗分片号
            write_first_window_evidence(
                destination, columns, identity_digest=self.identity_digest, segment_id=self.segment_id,  # CPU联合窗口及同源元数据
                policy_dt_s=self.policy_dt_s,  # 用同一秒/策略步尺度校验真实时长
            )  # 独立发布，不等待自然episode结束，也不覆盖旧分片。
            self._pending_windows.clear()  # 只有成功发布后才释放CPU事实。
            self._next_window_shard += 1  # 仅成功发布后推进窗口分片游标

    def drain(self, *, force: bool = False) -> dict[str, torch.Tensor]:
        r"""rollout边界批量搬运；返回[A,terms]奖励均值供已有分资产metrics使用。

        奖励均值分母是距上次drain以来该资产的环境步数，而非回合数、存活时长或首窗数。
        Args:
            force: 为True时将尚未达到flush_rows的非空回合/窗口批次也发布。
        Returns:
            dict: CPU张量reward_terms为[A,R]每环境步加权奖励贡献均值，sample_count为[A]实际分母。
                零样本资产返回零贡献并保留sample_count=0；调用方据此识别缺测，而非视为有效零奖励。
        """
        self._drain_first_windows(force=force)  # 控制步hot loop中不执行CPU分位数和Parquet压缩。
        if self._gpu_rows:  # 终止回合可以跨多个策略版本，但搬运按rollout边界批量执行
            columns = {key: torch.cat([row[key] for row in self._gpu_rows]).cpu().numpy() for key in self._gpu_rows[0]}  # 每列[E]CPU事件向量
            self._gpu_rows.clear()  # 本rollout设备行已转为CPU待发布批次
            if columns["env_id"].size:  # 无终止事件时仅更新奖励均值，不生成空回合分片
                self._pending.append(columns)  # 回合原始积分不因奖励均值归约而丢失
        result = {  # 奖励测度为逐环境步样本，不是首窗统计中的资产等权成功率
            "reward_terms": (self._asset_reward_sums / self._asset_samples.clamp_min(1)[:, None]).detach().cpu(),  # [A,R]和/[A,1]计数；仅零分母钳至1
            "sample_count": self._asset_samples.detach().cpu().clone(),  # [A]保留真实分母，包含未采样资产的0
        }
        self._asset_reward_sums.zero_()  # 只清零逐轮统计；未结束回合的_reward_sums积分继续保留
        self._asset_samples.zero_()  # 下次drain的环境步分母从零开始
        if self._pending and (force or sum(row["env_id"].size for row in self._pending) >= self.flush_rows):  # 回合事件数达到容量或显式强制发布
            columns = {key: np.concatenate([row[key] for row in self._pending]) for key in self._pending[0]}  # 各列按同一回合顺序拼接
            rewards = {name: columns.pop(f"reward/{name}") for name in self.reward_names}  # [E]回合积分单独传给writer，键配对不变
            destination = self.root / f"episodes-{self.segment_id}-{self._next_shard:06d}.parquet"  # 独立回合分片命名空间
            write_episode_evidence(
                destination,  # 本批回合的新发布路径
                columns,  # [E]联合终点、版本及可选首窗前缀
                reward_sums=rewards,  # 已含weight/dt的实际回合积分，不按时长再归一化
                identity_digest=self.identity_digest,  # 方法身份固定为调用方提供的摘要
                segment_id=self.segment_id,  # 物理进程段区分重建后的环境/回合编号
                policy_dt_s=self.policy_dt_s,  # 秒/策略步，与capture及_rows一致
            )
            self._pending.clear()  # 仅在整批成功发布后释放CPU回合事实
            self._next_shard += 1  # 成功发布后推进回合分片游标
        return result  # 即使尚未达到文件发布容量，也返回本轮逐资产奖励统计

    def close(self) -> None:
        r"""停训时单列尚未自然结束的回合，不能把这些右删失样本记为安全成功。"""
        # 最后一步已终止的环境在capture中清零，只有仍在途的环境进入停训删失记录。
        if self._last_snapshot is not None:  # 从未capture或已close时没有可新增的末帧证据
            ids = (self._steps > 0).nonzero(as_tuple=False).reshape(-1)  # [E]本进程已观察但未自然结束的回合
            self._gpu_rows.append(self._rows(self._last_snapshot, ids, censored=True))  # 实际末帧终点，不延伸到计划horizon
        self.drain(force=True)  # 发布所有非空待写批次；停训本身不会合成一个完整首窗
        self._last_snapshot = None  # 已消费末帧，避免再次close重复新增删失行
