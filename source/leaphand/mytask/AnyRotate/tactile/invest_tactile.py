"""Rollout a trained RL-Games policy and analyze LeapHand tactile/contact signals.

This script is designed for *post-hoc* tactile investigation:

- The policy was trained *without* tactile observations.
- During rollout, we read IsaacLab's ContactSensor buffers (and optionally contact points)
  for selected LeapHand fingertips / palm.
- We compute basic statistics online and export plots for downstream design decisions
  (observation integration, reward shaping).

Notes:
	- Contact sensors require `activate_contact_sensors=True` on the underlying asset spawner.
	  In this repo, `LEAP_HAND_CFG` already enables it.
	- Contact point tracking is optional and more expensive. Enable with `--track_contact_points`.
使用示例:
python source/leaphand/leaphand/mytask/AnyRotate/invest_tactile.py --task Template-Leaphand-Rot-Manager-v0 \
--checkpoint /home/hac/isaac/AnyRotate/logs/rl_games/leaphand_object_rot/2026-01-10_17-12-54/nn/leaphand_object_rot.pth \
--num_envs 4 --episodes_per_env 1 --headless --fingers index,thumb --force_threshold 1.0 \
--track_contact_points --max_contact_data_count_per_prim 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Rollout an RL-Games policy checkpoint and collect LeapHand tactile/contact statistics."
	)
	parser.add_argument(
		"--task",
		type=str,
		required=True,
		help="Task name registered in IsaacLab gym registry (e.g., Template-Leaphand-Rot-Manager-v0).",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		required=True,
		help="Path to RL-Games .pth checkpoint.",
	)
	parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs.")
	parser.add_argument(
		"--episodes_per_env",
		type=int,
		default=10,
		help="How many episodes to collect per sub-environment.",
	)
	parser.add_argument(
		"--max_steps_per_episode",
		type=int,
		default=0,
		help="Optional cap per episode (0 = no extra cap; rely on env termination).",
	)
	parser.add_argument(
		"--fingers",
		type=str,
		default="index,middle,ring,thumb",
		help=(
			"Comma-separated selection. Accepts aliases: index,middle,ring,thumb,palm or"
			" explicit sensor names: contact_index,contact_middle,contact_ring,contact_thumb,contact_palm"
		),
	)
	parser.add_argument(
		"--force_threshold",
		type=float,
		default=1.0,
		help="Contact threshold (N) applied to ||net_forces_w||.",
	)
	parser.add_argument(
		"--force_hist_max",
		type=float,
		default=50.0,
		help="Max force (N) for histogram range [0, max].",
	)
	parser.add_argument(
		"--force_hist_bins",
		type=int,
		default=100,
		help="Number of bins for force histogram.",
	)
	parser.add_argument(
		"--track_contact_points",
		action="store_true",
		default=False,
		help="Enable contact point tracking for selected sensors (more expensive).",
	)
	parser.add_argument(
		"--max_contact_data_count_per_prim",
		type=int,
		default=4,
		help="ContactSensorCfg.max_contact_data_count_per_prim (only used if tracking contact points).",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="",
		help="Output directory. Default: AnyRotate/tactile_analysis/run_<timestamp>.",
	)
	parser.add_argument(
		"--max_contact_points_per_step",
		type=int,
		default=20000,
		help=(
			"When --track_contact_points is enabled, cap the number of valid contact points exported per step "
			"(0 = no cap). This avoids large CPU transfers at high num_envs."
		),
	)
	parser.add_argument(
		"--deterministic",
		action="store_true",
		default=False,
		help="Use deterministic actions (if supported by the policy).",
	)
	parser.add_argument(
		"--real_time",
		action="store_true",
		default=False,
		help="Try to run in real-time (adds sleep to match env dt).",
	)

	parser.add_argument(
		"--disable_fabric",
		action="store_true",
		default=False,
		help="Disable fabric and use USD I/O operations.",
	)
	AppLauncher.add_app_launcher_args(parser)
	return parser


def _parse_finger_selection(selection: str) -> list[str]:
	alias_to_sensor = {
		"index": "contact_index",
		"middle": "contact_middle",
		"ring": "contact_ring",
		"thumb": "contact_thumb",
		"palm": "contact_palm",
	}
	parts = [p.strip() for p in selection.split(",") if p.strip()]
	if not parts:
		raise ValueError("--fingers is empty")
	sensor_names: list[str] = []
	for p in parts:
		sensor_names.append(alias_to_sensor.get(p, p))
	return sensor_names


@dataclass
class ForceHistogram:
	"""Streaming histogram + simple moments for scalar force magnitudes."""

	bins: torch.Tensor
	counts: torch.Tensor
	n: int = 0
	sum: float = 0.0
	sumsq: float = 0.0
	min: float = math.inf
	max: float = -math.inf

	@classmethod
	def create(cls, device: torch.device, max_force: float, num_bins: int) -> "ForceHistogram":
		edges = torch.linspace(0.0, float(max_force), int(num_bins) + 1, device=device)
		counts = torch.zeros(int(num_bins), device=device, dtype=torch.long)
		return cls(bins=edges, counts=counts)

	def update(self, x: torch.Tensor) -> None:
		# x: (N,) on same device
		if x.numel() == 0:
			return
		x = x.clamp(min=0.0, max=float(self.bins[-1].item()))
		# bucketize returns indices in [0, len(edges)] where i==0 means <edges[0]
		bin_ids = torch.bucketize(x, self.bins, right=False) - 1
		bin_ids = bin_ids.clamp(min=0, max=self.counts.numel() - 1)
		self.counts += torch.bincount(bin_ids, minlength=self.counts.numel())

		self.n += int(x.numel())
		self.sum += float(x.sum().item())
		self.sumsq += float((x * x).sum().item())
		self.min = min(self.min, float(x.min().item()))
		self.max = max(self.max, float(x.max().item()))

	def summary(self) -> dict:
		if self.n == 0:
			mean = float("nan")
			std = float("nan")
		else:
			mean = self.sum / self.n
			var = max(0.0, self.sumsq / self.n - mean * mean)
			std = math.sqrt(var)
		return {
			"n": int(self.n),
			"min": float(self.min) if self.n > 0 else float("nan"),
			"max": float(self.max) if self.n > 0 else float("nan"),
			"mean": float(mean),
			"std": float(std),
		}


@dataclass
class ContactStats:
	total_samples: int = 0
	contact_samples: int = 0

	def update(self, is_contact: torch.Tensor) -> None:
		# is_contact: bool tensor
		self.total_samples += int(is_contact.numel())
		self.contact_samples += int(is_contact.sum().item())

	@property
	def contact_rate(self) -> float:
		if self.total_samples == 0:
			return float("nan")
		return float(self.contact_samples) / float(self.total_samples)


def _ensure_output_dir(args: argparse.Namespace) -> str:
	if args.output_dir:
		out_dir = os.path.abspath(args.output_dir)
	else:
		stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		script_dir = os.path.dirname(os.path.abspath(__file__))
		out_dir = os.path.join(script_dir, "tactile_analysis", f"run_{stamp}")
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def _try_plot_force_hist(out_dir: str, sensor_name: str, hist: ForceHistogram) -> None:
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:
		print(f"[WARN] matplotlib not available, skip plotting. Reason: {exc}")
		return

	edges = hist.bins.detach().cpu().numpy()
	counts = hist.counts.detach().cpu().numpy()

	centers = 0.5 * (edges[:-1] + edges[1:])
	plt.figure(figsize=(7, 4))
	plt.plot(centers, counts, linewidth=1.5)
	plt.xlabel("||net contact force|| (N)")
	plt.ylabel("count")
	plt.title(f"Force magnitude histogram: {sensor_name}")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f"force_hist_{sensor_name}.png"), dpi=150)
	plt.close()


def _try_plot_contact_pos_heatmaps(
	out_dir: str,
	sensor_name: str,
	contact_pos_obj: list[np.ndarray],
	bins: int = 80,
) -> None:
	if len(contact_pos_obj) == 0:
		return
	pts = np.concatenate(contact_pos_obj, axis=0)
	if pts.size == 0:
		return
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:
		print(f"[WARN] matplotlib not available, skip plotting. Reason: {exc}")
		return

	def save_heat(xi: int, yi: int, name: str) -> None:
		plt.figure(figsize=(5, 4))
		plt.hist2d(pts[:, xi], pts[:, yi], bins=bins, cmap="viridis")
		plt.colorbar(label="count")
		plt.xlabel(["x", "y", "z"][xi] + " (object frame)")
		plt.ylabel(["x", "y", "z"][yi] + " (object frame)")
		plt.title(f"Contact position density ({name}): {sensor_name}")
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"contact_pos_{sensor_name}_{name}.png"), dpi=150)
		plt.close()

	save_heat(0, 1, "xy")
	save_heat(0, 2, "xz")
	save_heat(1, 2, "yz")


def _flatten_contact_positions(contact_pos_w: torch.Tensor) -> torch.Tensor:
	"""Convert ContactSensorData.contact_pos_w to a flat (K, 3) tensor, dropping NaNs.

	Expected shape: (num_envs, num_bodies, num_filtered, 3)
	"""

	pts = contact_pos_w.reshape(-1, 3)
	mask = torch.isfinite(pts).all(dim=-1)
	return pts[mask]


def _maybe_enable_contact_point_tracking(scene_cfg, sensor_names: Iterable[str], args: argparse.Namespace) -> None:
	if not args.track_contact_points:
		return
	for name in sensor_names:
		if not hasattr(scene_cfg, name):
			raise KeyError(f"SceneCfg has no sensor named '{name}'.")
		sensor_cfg = getattr(scene_cfg, name)
		# ContactSensorCfg
		sensor_cfg.track_contact_points = True
		sensor_cfg.track_pose = True
		sensor_cfg.max_contact_data_count_per_prim = int(args.max_contact_data_count_per_prim)


def main() -> None:
	"""主函数：加载训练好的RL-Games策略并进行触觉/接触信号分析"""
	
	# 构建并解析命令行参数
	parser = _build_parser()
	args = parser.parse_args()

	# 如果用户未指定headless模式，默认使用headless（分析脚本通常不需要GUI）
	if not hasattr(args, "headless") or args.headless is None:
		args.headless = True

	# 启动Isaac Sim应用
	app_launcher = AppLauncher(args)
	simulation_app = app_launcher.app

	"""以下是在Isaac Sim启动后的所有导入和逻辑"""

	# 导入gymnasium用于环境接口
	import gymnasium as gym

	# 导入RL-Games相关模块
	from rl_games.common import env_configurations, vecenv
	from rl_games.common.player import BasePlayer
	from rl_games.torch_runner import Runner

	# 导入IsaacLab环境和工具
	from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
	from isaaclab.utils.assets import retrieve_file_path
	from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

	# 导入IsaacLab的RL-Games包装器
	from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

	# 导入任务注册模块
	import isaaclab_tasks  # noqa: F401
	from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

	# 导入LeapHand任务定义
	import leaphand.tasks.manager_based.leaphand  # noqa: F401
	from isaaclab.utils import math as math_utils

	# 确保输出目录存在
	out_dir = _ensure_output_dir(args)

	# 解析用户指定的传感器名称（如index, middle, ring, thumb, palm）
	sensor_names = _parse_finger_selection(args.fingers)

	# 解析环境配置（从任务注册表中加载）
	env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
	# 加载RL-Games的智能体配置
	agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

	# （可选）在创建环境之前启用接触点跟踪
	# 这会修改场景配置中的ContactSensorCfg，启用track_contact_points和track_pose
	_maybe_enable_contact_point_tracking(env_cfg.scene, sensor_names, args)

	# 查找并设置checkpoint路径
	resume_path = retrieve_file_path(args.checkpoint)
	agent_cfg["params"]["load_checkpoint"] = True
	agent_cfg["params"]["load_path"] = resume_path
	print(f"[INFO] Loading model checkpoint: {resume_path}")

	# 为RL-Games包装环境，获取设备和裁剪参数
	rl_device = agent_cfg["params"]["config"]["device"]
	clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
	clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

	# 创建Isaac Lab环境
	env = gym.make(args.task, cfg=env_cfg)
	# 如果是多智能体环境，转换为单智能体
	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)
	# 用RL-Games包装器包装环境
	env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

	# 将环境注册到RL-Games框架中
	vecenv.register(
		"IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
	)
	env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

	# 加载训练好的智能体
	agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
	runner = Runner()
	runner.load(agent_cfg)
	agent: BasePlayer = runner.create_player()
	agent.restore(resume_path)  # 从checkpoint恢复权重
	agent.reset()
	# 如果指定了确定性模式，设置智能体为确定性推理
	if args.deterministic:
		agent.is_deterministic = True

	# 准备统计数据结构
	device = torch.device(rl_device)
	# 为每个传感器创建力直方图统计对象
	force_hist_by_sensor: dict[str, ForceHistogram] = {
		name: ForceHistogram.create(device=device, max_force=args.force_hist_max, num_bins=args.force_hist_bins)
		for name in sensor_names
	}
	# 为每个传感器创建接触统计对象（接触率等）
	contact_stats_by_sensor: dict[str, ContactStats] = {name: ContactStats() for name in sensor_names}

	# 存储接触点位置样本（在物体坐标系下，存储在CPU上作为numpy数组）
	contact_pos_obj_by_sensor: dict[str, list[np.ndarray]] = {name: [] for name in sensor_names}

	# Episode计数器：跟踪每个环境完成的episode数量
	num_envs = env.unwrapped.num_envs
	episodes_done = torch.zeros(num_envs, device=device, dtype=torch.int32)
	steps_in_episode = torch.zeros(num_envs, device=device, dtype=torch.int32)

	# 获取环境的时间步长
	dt = float(env.unwrapped.step_dt)

	# 重置环境，获取初始观测
	obs = env.reset()
	# 处理观测格式（可能是字典或直接的tensor）
	if isinstance(obs, dict):
		obs_for_agent = obs.get("obs", obs)
	else:
		obs_for_agent = obs

	# RL-Games批处理模式所需的初始化
	_ = agent.get_batch_size(obs_for_agent, 1)
	# 如果智能体使用RNN，初始化隐藏状态
	if agent.is_rnn:
		agent.init_rnn()

	# 初始化步数计数器和计时器
	step_counter = 0
	start_wall = time.time()
	print(
		f"[INFO] Start rollout: num_envs={num_envs}, episodes_per_env={args.episodes_per_env}, "
		f"sensors={sensor_names}, track_contact_points={args.track_contact_points}"
	)

	# 主循环：持续运行直到收集足够的episode数据或仿真停止
	while simulation_app.is_running():
		# 记录当前步的开始时间，用于实时模式的时间控制
		step_t0 = time.time()
		
		# 使用推理模式（禁用梯度计算）以提高性能
		with torch.inference_mode():
			# ========== 智能体推理 ==========
			# 将观测转换为智能体所需的torch格式
			obs_for_agent_t = agent.obs_to_torch(obs_for_agent)
			# 智能体根据观测生成动作（可选择确定性或随机策略）
			actions = agent.get_action(obs_for_agent_t, is_deterministic=agent.is_deterministic)
			# 环境执行动作，返回新观测、奖励、终止标志和额外信息
			obs, _, dones, _ = env.step(actions)

			# ========== Episode计数管理 ==========
			# 将终止标志转换为布尔掩码
			done_mask = dones.to(dtype=torch.bool)
			# 累加已完成的episode数量（每个环境独立计数）
			episodes_done += done_mask.to(dtype=torch.int32)
			# 所有环境的步数计数器加1
			steps_in_episode += 1
			# 对于已终止的环境，重置其步数计数器为0
			steps_in_episode = torch.where(done_mask, torch.zeros_like(steps_in_episode), steps_in_episode)

			# ========== 可选的每episode步数上限 ==========
			# 如果设置了max_steps_per_episode，检查是否有环境超过步数上限
			if args.max_steps_per_episode > 0:
				over_cap = steps_in_episode >= int(args.max_steps_per_episode)
				# 注意：无法通过rl-games包装器干净地强制重置单个环境
				# 因此在分析中，我们将超过上限视为episode终止，仅用于统计目的
				if bool(over_cap.any()):
					# 将超过上限的环境标记为已完成一个episode
					episodes_done += over_cap.to(dtype=torch.int32)
					# 重置这些环境的步数计数器
					steps_in_episode = torch.where(over_cap, torch.zeros_like(steps_in_episode), steps_in_episode)

			# ========== 收集触觉信号数据 ==========
			# 获取底层环境（去除包装器）
			base_env = env.unwrapped
			# 获取场景中的物体（被操作对象）
			obj = base_env.scene["object"]
			# 获取物体在世界坐标系下的位置 (num_envs, 3)
			obj_pos_w = obj.data.root_pos_w.to(device=device)
			# 获取物体在世界坐标系下的四元数姿态 (num_envs, 4)
			obj_quat_w = obj.data.root_quat_w.to(device=device)

			# 遍历所有选定的传感器（如index, middle, ring, thumb, palm）
			for name in sensor_names:
				# 获取当前传感器对象
				sensor = base_env.scene[name]
				
				# ===== 力信号处理 =====
				# 获取传感器在世界坐标系下的净接触力 (num_envs, num_bodies, 3)
				net_forces_w = sensor.data.net_forces_w.to(device=device)
				# 计算力的模长（向量范数） (num_envs, num_bodies)
				force_mag = torch.linalg.vector_norm(net_forces_w, dim=-1)
				# 对每个环境，取所有body上的最大力值作为该环境的代表力 (num_envs,)
				force_mag_max = torch.max(force_mag, dim=1)[0]

				# 更新该传感器的力直方图统计
				force_hist_by_sensor[name].update(force_mag_max)
				# 判断是否发生接触（力超过阈值）
				is_contact = force_mag_max > float(args.force_threshold)
				# 更新接触统计（接触率等）
				contact_stats_by_sensor[name].update(is_contact)

				# ===== 接触点位置跟踪（可选） =====
				# 如果启用了接触点跟踪且传感器数据中包含接触点位置
				if args.track_contact_points and sensor.data.contact_pos_w is not None:
					# 向量化坐标变换：将世界坐标系下的接触点转换到物体坐标系
					# 公式: p_obj = R_obj^T * (p_w - t_obj)
					# 其中 R_obj^T 是物体旋转的逆，t_obj 是物体位置
					
					# 获取所有接触点的世界坐标 (E=num_envs, B=num_bodies, M=max_contacts, 3)
					pts_w_full = sensor.data.contact_pos_w.to(device=device)
					# 创建有效性掩码：检查哪些接触点是有限值（非NaN/Inf） (E, B, M)
					valid_full = torch.isfinite(pts_w_full).all(dim=-1)
					
					# 如果存在至少一个有效的接触点
					if bool(valid_full.any()):
						# 将物体位姿广播到与接触点相同的维度 (E, 1, 1, *)
						obj_pos = obj_pos_w[:, None, None, :]  # (E, 1, 1, 3)
						obj_quat = obj_quat_w[:, None, None, :]  # (E, 1, 1, 4)
						
						# 计算接触点相对于物体位置的偏移
						pts_rel = pts_w_full - obj_pos
						# 应用四元数逆旋转，将相对位置转换到物体坐标系
						pts_obj_full = math_utils.quat_apply_inverse(obj_quat, pts_rel)
						# 只保留有效的接触点
						pts_obj = pts_obj_full[valid_full]
						
						# 如果设置了单步最大接触点数限制，进行截断
						if args.max_contact_points_per_step and pts_obj.shape[0] > int(args.max_contact_points_per_step):
							pts_obj = pts_obj[: int(args.max_contact_points_per_step)]
						
						# 将接触点数据转移到CPU并转换为numpy数组存储
						contact_pos_obj_by_sensor[name].append(pts_obj.detach().cpu().numpy())

			# ========== 准备下一步的观测 ==========
			# 处理观测格式：如果是字典，提取"obs"键；否则直接使用
			if isinstance(obs, dict):
				obs_for_agent = obs.get("obs", obs)
			else:
				obs_for_agent = obs

			# ========== 重置RNN状态（如果智能体使用循环神经网络） ==========
			# 对于已终止的episode，需要清零RNN的隐藏状态
			if agent.is_rnn and agent.states is not None:
				if bool(done_mask.any()):
					# 遍历所有RNN状态张量，将终止环境对应的状态清零
					for s in agent.states:
						s[:, done_mask, :] = 0.0

		# 全局步数计数器加1
		step_counter += 1

		# ========== 停止条件检查 ==========
		# 如果所有环境都已收集到足够数量的episode，退出主循环
		if bool((episodes_done >= int(args.episodes_per_env)).all()):
			break

		# ========== 实时模式时间控制（可选） ==========
		# 如果启用了实时模式，计算需要sleep的时间以匹配环境的dt
		if args.real_time:
			sleep_time = dt - (time.time() - step_t0)
			if sleep_time > 0:
				time.sleep(sleep_time)

		# ========== 周期性日志输出 ==========
		# 每200步打印一次进度信息
		if step_counter % 200 == 0:
			done_min = int(episodes_done.min().item())
			done_max = int(episodes_done.max().item())
			print(f"[INFO] step={step_counter} episodes_done range=[{done_min}, {done_max}]")

	# ========== Rollout结束，计算总耗时 ==========
	wall = time.time() - start_wall
	print(f"[INFO] Rollout finished. steps={step_counter}, wall_time={wall:.1f}s")

	# ========== 导出分析结果 ==========
	# 构建汇总字典，包含元数据和各传感器的统计信息
	summary: dict[str, dict] = {
		"meta": {
			"task": args.task,
			"checkpoint": os.path.abspath(args.checkpoint),
			"num_envs": int(num_envs),
			"episodes_per_env": int(args.episodes_per_env),
			"force_threshold": float(args.force_threshold),
			"track_contact_points": bool(args.track_contact_points),
			"rl_device": str(rl_device),
			"dt": float(dt),
			"steps": int(step_counter),
		},
		"sensors": {},
	}

	# 为每个传感器添加统计摘要
	for name in sensor_names:
		summary["sensors"][name] = {
			"force": force_hist_by_sensor[name].summary(),  # 力的统计信息（均值、标准差等）
			"contact_rate": float(contact_stats_by_sensor[name].contact_rate),  # 接触率
		}

	# 将汇总信息保存为JSON文件
	with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	# ========== 保存原始直方图数据 ==========
	for name in sensor_names:
		np.savez(
			os.path.join(out_dir, f"hist_{name}.npz"),
			bin_edges=force_hist_by_sensor[name].bins.detach().cpu().numpy(),
			counts=force_hist_by_sensor[name].counts.detach().cpu().numpy(),
		)

	# ========== 生成可视化图表 ==========
	for name in sensor_names:
		# 绘制力直方图
		_try_plot_force_hist(out_dir, name, force_hist_by_sensor[name])
		# 如果启用了接触点跟踪，绘制接触位置热力图
		if args.track_contact_points:
			_try_plot_contact_pos_heatmaps(out_dir, name, contact_pos_obj_by_sensor[name])

	print(f"[INFO] Wrote tactile analysis outputs to: {out_dir}")

	# 关闭环境和仿真应用
	env.close()
	simulation_app.close()


if __name__ == "__main__":
	main()

